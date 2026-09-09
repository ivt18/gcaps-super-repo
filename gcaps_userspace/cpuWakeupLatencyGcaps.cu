/**
 * ============================================================================
 * cpuWakeupLatencyGcaps.cu
 * ============================================================================
 *
 * Measures how long after the GPU segment finishes the waiting CPU thread is
 * unblocked.
 *
 *   W_i = (wakeup_cpu_ns + cpu_to_gpu_offset_ns) − completion_gpu_ns
 *
 *   completion_gpu_ns — %globaltimer written by the kernel's own last
 *                       instruction into a pinned slot.
 *   wakeup_cpu_ns     — CLOCK_MONOTONIC captured immediately after the GCAPS
 *                       segment-end wait returns, before the remove ioctl.
 *
 * The wakeup stamp is taken before the remove ioctl: the thread is running
 * again the moment the wait returns, and the ioctl that follows is GCAPS
 * bookkeeping rather than part of the wake path.  That ioctl is reported in its
 * own column instead.
 *
 * Per-job values, all in nanoseconds:
 *
 *   W_ns               W_i for that job, i.e. the formula above
 *   seg_begin_ioctl_ns the GCAPS add ioctl, paid before the kernel runs
 *   seg_end_ioctl_ns   the GCAPS remove ioctl, paid after the wait returns
 *
 * OUTPUT.  One CSV to stdout, or to the file given by --out.  Leading `#` lines
 * record the configuration (device, exec_us, n_jobs, warmup, ioctl_enabled,
 * realtime, pin_cpu, blocking_sync, release_period_us, release_late) and the
 * clock calibration.  Then a header row and one row per completed job:
 *
 *   job_id,completion_ns,wakeup_ns,W_ns,seg_begin_ioctl_ns,seg_end_ioctl_ns
 *
 * Usage: cpuWakeupLatencyGcaps [EXEC_US [N_JOBS]] [options]
 *   EXEC_US               Segment execution time in µs.        (default: 50)
 *   N_JOBS                Number of sequential job repetitions.(default: 500)
 *   -i 0|1                GCAPS ioctl elevation off/on.        (default: 1)
 *   --realtime            SCHED_FIFO 50 on the measuring thread; needed for -i 1.
 *   --warmup N            Throwaway releases before job 0.     (default: 10)
 *   --release-period-us N Absolute release grid period.        (default: 997)
 *   --cpu N               Pin the measuring thread to CPU N.   (default: 2)
 *   --no-pin              Do not pin.
 *   --spin                Do NOT use blocking sync (GCAPS's default -s 0 mode).
 *   --out FILE            Write the CSV here instead of stdout.  Needed when
 *                         running under run_gcaps_r35.sh, which is MANDATORY
 *                         for -i 1 (it makes SCHED_FIFO attainable despite
 *                         CONFIG_RT_GROUP_SCHED and disables railgating) and
 *                         whose own preflight output shares stdout.
 */

#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>

#include <cuda_runtime.h>
#include <linux/nvgpu.h>

#include "common/include/clock_calib.cuh"

/* Same band as the sibling binaries: the measuring thread runs at 50. */
static constexpr int RT_PRIORITY = 50;
/* cpuWakeupLatencyBenchSeq's RELEASE_CPU: the thread whose wake-up is timed. */
static constexpr int DEFAULT_PIN_CPU = 2;

/* Release cadence. */
static constexpr uint64_t DEFAULT_RELEASE_PERIOD_US = 997;
/* Minimum (approx) CPU time it takes to make a release, without the segment's own
 * execution.  Only used to warn when the requested release period is too short
 * to hold the grid.  The -i 1 value is larger because each release then also
 * pays the two GCAPS runlist-update ioctls. */
static constexpr uint64_t RELEASE_CYCLE_SLACK_US       = 40;
static constexpr uint64_t RELEASE_CYCLE_SLACK_IOCTL_US = 1250;
/* Benchmark start delay after initialisation, so job 0 sleeps to its grid
 * point instead of finding its deadline already past. */
static constexpr uint64_t STARTUP_MARGIN_NS         = 10000000ULL;   // 10 ms

static const char* GCAPS_CTRL_DEV = "/dev/nvgpu/igpu0/ctrl";

// ============================================================================
// Clocks
// ============================================================================

/**
 * host_ns — read CLOCK_MONOTONIC as a nanosecond timestamp on the host.
 *
 * Intra-CPU use only.  Anything converted into the GPU clock domain must use
 * clock_calib_host_ns() instead.
 *
 * @return current CLOCK_MONOTONIC value, in nanoseconds
 */
static inline uint64_t host_ns()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

/**
 * sleep_until_ns — block on a host thread until an absolute CLOCK_MONOTONIC deadline.
 *
 * A deadline already in the past returns immediately; the caller counts those
 * as release_late, which must read 0 for the sweep to be on its intended grid.
 *
 * @param target_ns absolute CLOCK_MONOTONIC deadline, in nanoseconds
 */
static void sleep_until_ns(uint64_t target_ns)
{
    struct timespec ts;
    ts.tv_sec  = (time_t)(target_ns / 1000000000ULL);
    ts.tv_nsec = (long)(target_ns % 1000000000ULL);
    while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, nullptr) != 0) {}
}

/**
 * wgc_gpu_ns — kernel to read the GPU's %globaltimer as a nanosecond timestamp.
 *
 * Device-side counterpart of host_ns().
 *
 * @return current %globaltimer value, in nanoseconds
 */
static __device__ __forceinline__ uint64_t wgc_gpu_ns()
{
    uint64_t t;
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
    return t;
}

/**
 * wgcBusyWaitStamp — the measured GPU segment: busy-wait, then stamp completion.
 *
 * The completion stamp is taken by the kernel's own last instruction.
 *
 * @param durationNs    device pointer to the busy-wait length, in ns
 *                      (0 is valid and yields a stamp-only segment)
 * @param completionOut device/pinned pointer receiving the %globaltimer value
 *                      at segment end
 */
__global__ void wgcBusyWaitStamp(const uint64_t* durationNs,
                                 uint64_t*       completionOut)
{
    if (threadIdx.x != 0) return;
    const uint64_t dur = *durationNs;
    const uint64_t t0  = wgc_gpu_ns();
    while (wgc_gpu_ns() - t0 < dur) { /* spin */ }
    *completionOut = wgc_gpu_ns();
}

/**
 * CUDA_CHECK — abort the enclosing function with EXIT_FAILURE on a CUDA error.
 *
 * Reports file, line and the runtime's own message.
 *
 * @param call a CUDA runtime call returning cudaError_t
 */
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(_e));               \
            return EXIT_FAILURE;                                               \
        }                                                                      \
    } while (0)

// ============================================================================
// GCAPS runlist-priority ioctl
// ============================================================================

/**
 * gcaps_runlist — issue one GCAPS runlist-priority ioctl.  add_req selects
 * which one: true ADDS this process to the GPU runlist at its real-time
 * priority (segment begin), false REMOVES it (segment end).
 *
 * The driver decides whether the caller is a real-time task by reading its
 * rt_priority, so this is only meaningful once apply_rt() has succeeded.
 *
 * @param fd        open file descriptor for the nvgpu control device
 * @param pid       process whose TSGs are admitted or evicted
 * @param add_req   true to add to the runlist (segment begin), false to remove
 *                  from it (segment end)
 * @param sync_mode driver-side path selector (true enforces GPU mutual exclusion)
 * @return the ioctl return value: >= 0 on success, < 0 with errno set
 */
static int gcaps_runlist(int fd, pid_t pid, bool add_req, bool sync_mode)
{
    struct nvgpu_gpu_runlist_update_rt_prio_args args;
    memset(&args, 0, sizeof(args));
    args.pid       = pid;
    args.add_req   = add_req;
    args.sync_mode = sync_mode;
    return ioctl(fd, NVGPU_GPU_IOCTL_RUNLIST_UPDATE_RT_PRIO, &args);
}

static int g_pinned_cpu = -1;   /* what actually took effect, for the banner */

/**
 * apply_rt — put the measuring thread on SCHED_FIFO and optionally pin it.
 *
 * Records the pin that actually took effect in g_pinned_cpu, so the banner and
 * the CSV report what happened rather than what was requested.
 *
 * @param pin_cpu CPU to pin to, or negative to leave affinity untouched
 */
static void apply_rt(int pin_cpu)
{
    struct sched_param sp;
    memset(&sp, 0, sizeof(sp));
    sp.sched_priority = RT_PRIORITY;
    if (sched_setscheduler(0, SCHED_FIFO, &sp) != 0)
        fprintf(stderr,
                "WARNING: SCHED_FIFO %d failed (errno=%d) — continuing with the "
                "default policy.  With -i 1 this INVALIDATES the run: the "
                "driver reads rt_priority to decide whether the caller is a "
                "real-time task at all.\n", RT_PRIORITY, errno);
    else
        fprintf(stderr, "  measuring thread set to SCHED_FIFO %d\n", RT_PRIORITY);

    if (pin_cpu >= 0) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(pin_cpu, &set);
        if (sched_setaffinity(0, sizeof(set), &set) != 0)
            fprintf(stderr, "WARNING: pin to CPU %d failed (errno=%d)\n",
                    pin_cpu, errno);
        else {
            g_pinned_cpu = pin_cpu;
            fprintf(stderr, "  measuring thread pinned to CPU %d\n", pin_cpu);
        }
    }
}

// ============================================================================
// main
// ============================================================================

/**
 * main — run the W_i sweep and write the per-release CSV.
 *
 * @param argc argument count
 * @param argv see the usage block at the top of this file
 * @return EXIT_SUCCESS, or EXIT_FAILURE on a bad option or a CUDA/ioctl failure
 */
int main(int argc, char** argv)
{
    uint64_t execUs          = 50;
    int      nJobs           = 500;
    int      warmup          = 10;
    int      ioctlEnabled    = 1;
    bool     realtime        = false;
    bool     blockingSync    = true;
    int      pinCpu          = DEFAULT_PIN_CPU;
    uint64_t releasePeriodUs = DEFAULT_RELEASE_PERIOD_US;
    const char* outPath      = nullptr;

    /* Positional args are collected separately so that, unlike
     * cpuWakeupLatencyStreamBaseline, option order does not matter. */
    std::vector<const char*> pos;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--realtime") == 0)            realtime = true;
        else if (strcmp(argv[i], "--spin") == 0)           blockingSync = false;
        else if (strcmp(argv[i], "--no-pin") == 0)         pinCpu = -1;
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc)
            ioctlEnabled = atoi(argv[++i]);
        else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc)
            warmup = atoi(argv[++i]);
        else if (strcmp(argv[i], "--cpu") == 0 && i + 1 < argc)
            pinCpu = atoi(argv[++i]);
        else if (strcmp(argv[i], "--release-period-us") == 0 && i + 1 < argc)
            releasePeriodUs = (uint64_t)atoll(argv[++i]);
        else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc)
            outPath = argv[++i];
        else pos.push_back(argv[i]);
    }
    if (pos.size() >= 1) execUs = (uint64_t)atoll(pos[0]);
    if (pos.size() >= 2) nJobs  = atoi(pos[1]);

    if (nJobs < 1) { fprintf(stderr, "N_JOBS must be >= 1\n"); return EXIT_FAILURE; }
    if (warmup < 0) warmup = 0;
    if (releasePeriodUs < 1) {
        fprintf(stderr, "--release-period-us must be >= 1\n");
        return EXIT_FAILURE;
    }
    /* The period must outlast a whole job, because if the next release is
     * already due when one finishes, every release runs back-to-back and each
     * W_i lands at the same point in the driver's periodic work instead of
     * sampling across all of it. */
    const uint64_t cycleSlackUs = ioctlEnabled ? RELEASE_CYCLE_SLACK_IOCTL_US
                                               : RELEASE_CYCLE_SLACK_US;
    if (releasePeriodUs <= execUs + cycleSlackUs)
        fprintf(stderr,
                "WARNING: release period %llu us is not comfortably above the "
                "job cycle (exec %llu us + ~%llu us of launch%s/wake) — the grid "
                "will degenerate to back-to-back releases and release_late will "
                "count them.  Try --release-period-us %llu.\n",
                (unsigned long long)releasePeriodUs,
                (unsigned long long)execUs,
                (unsigned long long)cycleSlackUs,
                ioctlEnabled ? " + two GCAPS ioctls" : "",
                (unsigned long long)(2 * (execUs + cycleSlackUs)));

    if (ioctlEnabled && !realtime)
        fprintf(stderr,
                "WARNING: -i 1 without --realtime.  GCAPS classifies the caller "
                "by rt_priority, so every ioctl will take the best-effort path "
                "and W_i will not describe GCAPS's real-time wake path.\n");

    // Must be set before the context is created: it makes the segment-end wait
    // sleep until the GPU interrupt arrives instead of spin-polling, which is
    // what makes W_i an OS wake-up rather than the cost of a spin loop.
    if (blockingSync)
        CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    fprintf(stderr, "=== CPU Wakeup Latency (GCAPS) ===\n");
    fprintf(stderr, "  device       : %s\n",   prop.name);
    fprintf(stderr, "  exec_us      : %llu\n", (unsigned long long)execUs);
    fprintf(stderr, "  n_jobs       : %d\n",   nJobs);
    fprintf(stderr, "  warmup       : %d\n",   warmup);
    fprintf(stderr, "  ioctl        : %s\n",   ioctlEnabled ? "on (GCAPS)"
                                                            : "off (TSG baseline)");
    fprintf(stderr, "  realtime     : %s\n",   realtime ? "yes" : "no");
    fprintf(stderr, "  wait         : cudaEventSynchronize (GCAPS's own)\n");
    fprintf(stderr, "  blocking sync: %s\n",   blockingSync ? "yes" : "no (spin)");
    fflush(stderr);

    // ---- GCAPS control device ---------------------------------------------
    int fd = -1;
    if (ioctlEnabled) {
        fd = open(GCAPS_CTRL_DEV, O_RDWR);
        if (fd < 0) {
            fprintf(stderr, "open %s failed (errno=%d) — is the patched nvgpu "
                            "loaded, and are you root?\n", GCAPS_CTRL_DEV, errno);
            return EXIT_FAILURE;
        }
    }

    // ---- Device-side duration and per-job completion slots -----------------
    const uint64_t durNs = execUs * 1000ULL;
    uint64_t* d_dur = nullptr;
    CUDA_CHECK(cudaMalloc(&d_dur, sizeof(uint64_t)));            // creates the context
    CUDA_CHECK(cudaMemcpy(d_dur, &durNs, sizeof(uint64_t), cudaMemcpyHostToDevice));

    uint64_t* h_completion = nullptr;
    CUDA_CHECK(cudaMallocHost((void**)&h_completion,
                              (size_t)nJobs * sizeof(uint64_t)));
    /* cudaMallocHost does NOT zero: the "did this job complete" test below is
     * `slot != 0`, so the whole buffer must start at 0 or an aborted run emits
     * rows built from uninitialised memory. */
    memset(h_completion, 0, (size_t)nJobs * sizeof(uint64_t));

    uint64_t* h_gpuTs = nullptr;
    CUDA_CHECK(cudaMallocHost((void**)&h_gpuTs, sizeof(uint64_t)));

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    /* The segment bracket's events, with GCAPS's own flags: blocking sync so
     * cudaEventSynchronize sleeps, timing disabled otherwise (the native GCAPS
     * apps' convention — W_i never uses cudaEventElapsedTime). */
    unsigned evFlags = blockingSync ? cudaEventBlockingSync
                                    : cudaEventDisableTiming;
    cudaEvent_t evStart = nullptr, evStop = nullptr;
    CUDA_CHECK(cudaEventCreateWithFlags(&evStart, evFlags));
    CUDA_CHECK(cudaEventCreateWithFlags(&evStop,  evFlags));

    /* RT goes on after the runtime is up, so the driver's worker threads were
     * created at default priority and do not inherit it. */
    if (realtime) apply_rt(pinCpu);

    const pid_t myPid = getpid();

    // ---- Start-of-run clock bracket (GPU idle) -----------------------------
    ClockBracket clk;
    if (clock_bracket_begin(&clk, stream, h_gpuTs) != 0) {
        fprintf(stderr, "start calibration failed\n");
        return EXIT_FAILURE;
    }
    fprintf(stderr, "  cpu_to_gpu_offset: %lld ns\n", (long long)clk.offset_start);
    fflush(stderr);

    // ---- Warm-up: absorb first-dispatch cold start; results discarded ------
    // The seq arm does this and the stream baseline does not; matching seq
    // keeps job 0 from being a cold-start outlier in this arm's tail stats.
    uint64_t* h_warm = nullptr;
    CUDA_CHECK(cudaMallocHost((void**)&h_warm, sizeof(uint64_t)));
    for (int i = 0; i < warmup; ++i) {
        *h_warm = 0;
        if (ioctlEnabled && gcaps_runlist(fd, myPid, true, false) < 0)
            fprintf(stderr, "WARNING: warm-up add ioctl failed (errno=%d)\n", errno);
        wgcBusyWaitStamp<<<1, 1, 0, stream>>>(d_dur, h_warm);
        cudaEventRecord(evStop, stream);
        cudaEventSynchronize(evStop);
        if (ioctlEnabled && gcaps_runlist(fd, myPid, false, false) < 0)
            fprintf(stderr, "WARNING: warm-up remove ioctl failed (errno=%d)\n", errno);
    }
    cudaFreeHost(h_warm);

    fprintf(stderr, "  Releasing %d jobs sequentially...\n", nJobs);
    fflush(stderr);

    // ---- Measurement loop --------------------------------------------------
    std::vector<uint64_t> wakeupTimes((size_t)nJobs, 0);
    std::vector<uint64_t> addIoctlNs((size_t)nJobs, 0);
    std::vector<uint64_t> remIoctlNs((size_t)nJobs, 0);
    bool anyFail = false;
    int  completed = 0;

    const uint64_t t0_ns      = host_ns() + STARTUP_MARGIN_NS;
    const uint64_t relPeriodNs = releasePeriodUs * 1000ULL;
    int late = 0;

    for (int i = 0; i < nJobs; ++i) {
        const uint64_t nominal = t0_ns + (uint64_t)i * relPeriodNs;
        if (host_ns() > nominal) ++late;
        sleep_until_ns(nominal);

        /* --- gcapsGpuSegBegin: event record, then the add ioctl ------------ */
        cudaEventRecord(evStart, stream);
        if (ioctlEnabled) {
            const uint64_t a0 = host_ns();
            if (gcaps_runlist(fd, myPid, true, false) < 0) {
                fprintf(stderr, "add ioctl failed (job %d, errno=%d)\n", i, errno);
                anyFail = true;
                break;
            }
            addIoctlNs[(size_t)i] = host_ns() - a0;
        }

        wgcBusyWaitStamp<<<1, 1, 0, stream>>>(d_dur, &h_completion[(size_t)i]);

        /* --- gcapsGpuSegEnd: event record, THE WAIT, then the remove ioctl -- */
        cudaEventRecord(evStop, stream);
        const cudaError_t syncErr = cudaEventSynchronize(evStop);
        wakeupTimes[(size_t)i] = clock_calib_host_ns()  /* CLOCK_MONOTONIC_RAW: this stamp is
                                       * converted with offset_at(); see the
                                       * two-clock rule in clock_calib.cuh */;

        if (syncErr != cudaSuccess) {
            fprintf(stderr, "segment wait failed (job %d): %s\n",
                    i, cudaGetErrorString(syncErr));
            anyFail = true;
            break;
        }

        if (ioctlEnabled) {
            const uint64_t r0 = host_ns();
            if (gcaps_runlist(fd, myPid, false, false) < 0) {
                fprintf(stderr, "remove ioctl failed (job %d, errno=%d)\n", i, errno);
                anyFail = true;
                break;
            }
            remIoctlNs[(size_t)i] = host_ns() - r0;
        }
        completed = i + 1;
    }

    fprintf(stderr, "  Done.\n");
    fflush(stderr);

    // ---- End-of-run clock bracket (GPU idle) -------------------------------
    // MUST be inside a GCAPS segment bracket.  clock_bracket_end() launches
    // CALIB_N_SAMPLES stamp kernels, and by this point the last segment's
    // REMOVE ioctl has run -- so this context's TSGs (compute AND copy engine)
    // are off the runlist, where unbracketed GPU work is never redispatched and
    // blocks forever.  That is the runlist-cache-desync mechanism documented on
    // SeqWorkload::verify(), and it hung this binary after "Done." until the
    // wrapper SIGKILLed it, losing the entire run's output.
    if (ioctlEnabled && gcaps_runlist(fd, myPid, true, false) < 0)
        fprintf(stderr, "WARNING: calibration add ioctl failed (errno=%d)\n", errno);

    if (clock_bracket_end(&clk, stream, h_gpuTs) != 0)
        fprintf(stderr, "WARNING: end-of-run calibration failed\n");

    if (ioctlEnabled && gcaps_runlist(fd, myPid, false, false) < 0)
        fprintf(stderr, "WARNING: calibration remove ioctl failed (errno=%d)\n", errno);
    const double calibStd = clk.std_start;

    // ---- W values ----------------------------------------------------------
    /* Only jobs that actually ran: `completed`, not nJobs.  (The sibling
     * binaries write `anyFail ? wakeupTimes.size() : nJobs`, which is the same
     * number either way because the vector is sized nJobs up front.) */
    const int rows = anyFail ? completed : nJobs;
    std::vector<int64_t> wVals;
    wVals.reserve((size_t)rows);
    for (int i = 0; i < rows; ++i) {
        if (h_completion[(size_t)i] == 0) continue;
        wVals.push_back(((int64_t)wakeupTimes[(size_t)i]
                         + clk.offset_at(wakeupTimes[(size_t)i]))
                        - (int64_t)h_completion[(size_t)i]);
    }

    double wMean = 0.0;
    for (int64_t v : wVals) wMean += (double)v;
    if (!wVals.empty()) wMean /= (double)wVals.size();
    double wVar = 0.0;
    for (int64_t v : wVals) { const double d = (double)v - wMean; wVar += d * d; }
    const double wStd = (wVals.size() > 1)
                      ? std::sqrt(wVar / (double)(wVals.size() - 1))
                      : 0.0;

    // ---- CSV ---------------------------------------------------------------
    // Reopened rather than passed around: every emit below is a printf, and the
    // wrapper this must run under writes its own preflight to the same stdout.
    if (outPath != nullptr) {
        if (freopen(outPath, "w", stdout) == nullptr) {
            fprintf(stderr, "cannot open --out %s (errno=%d)\n", outPath, errno);
            return EXIT_FAILURE;
        }
        fprintf(stderr, "  CSV -> %s\n", outPath);
    }

    printf("# CPU Wakeup Latency (GCAPS)\n");
    printf("# device: %s\n",              prop.name);
    printf("# exec_us: %llu\n",           (unsigned long long)execUs);
    printf("# n_jobs: %d\n",              nJobs);
    printf("# warmup: %d\n",              warmup);
    printf("# ioctl_enabled: %d\n",       ioctlEnabled);
    printf("# realtime: %s\n",            realtime ? "yes" : "no");
    printf("# pin_cpu: %d\n",             g_pinned_cpu);
    printf("# wait_primitive: cudaEventSynchronize\n");
    printf("# blocking_sync: %s\n",       blockingSync ? "yes" : "no");
    printf("# release_period_us: %llu\n", (unsigned long long)releasePeriodUs);
    printf("# release_late: %d  (grid points already past; want 0)\n", late);
    clock_bracket_report(&clk);
    printf("# W_i = (wakeup_ns + cpu_to_gpu_offset_ns) - completion_ns\n");
    printf("# completion_ns is the SEGMENT KERNEL's last-instruction stamp\n");
    printf("# wakeup_ns is taken when the segment-end wait returns, BEFORE the "
           "remove ioctl — see the file header\n");
    printf("# seg_begin_ioctl_ns / seg_end_ioctl_ns: the GCAPS add/remove "
           "runlist ioctls, wall, host-side (0 when -i 0)\n");

    if (wStd > 0.0 && calibStd >= wStd / 10.0)
        printf("# CALIBRATION_WARNING: calibration_std=%.1f ns, "
               "measurement_std=%.1f ns\n", calibStd, wStd);

    printf("job_id,completion_ns,wakeup_ns,W_ns,seg_begin_ioctl_ns,"
           "seg_end_ioctl_ns\n");

    int wIdx = 0;
    for (int i = 0; i < rows; ++i) {
        if (h_completion[(size_t)i] == 0) continue;
        printf("%d,%llu,%llu,%lld,%llu,%llu\n",
               i,
               (unsigned long long)h_completion[(size_t)i],
               (unsigned long long)wakeupTimes[(size_t)i],
               (long long)wVals[(size_t)wIdx++],
               (unsigned long long)addIoctlNs[(size_t)i],
               (unsigned long long)remIoctlNs[(size_t)i]);
    }
    fflush(stdout);

    cudaEventDestroy(evStop);
    cudaEventDestroy(evStart);
    cudaStreamDestroy(stream);
    cudaFreeHost(h_gpuTs);
    cudaFreeHost(h_completion);
    cudaFree(d_dur);
    if (fd >= 0) close(fd);

    return anyFail ? EXIT_FAILURE : EXIT_SUCCESS;
}
