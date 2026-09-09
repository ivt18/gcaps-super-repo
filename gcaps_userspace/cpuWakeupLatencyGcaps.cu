/**
 * ============================================================================
 * cpuWakeupLatencyGcaps.cu  — EQ 2.2, GCAPS arm
 * ============================================================================
 *
 * GCAPS counterpart to singleTaskSched's cpuWakeupLatencyBenchSeq (Sequence-
 * Scheduler) and cpuWakeupLatencyStreamBaseline (plain stream).  Measures how
 * long after the GPU segment finishes the waiting CPU thread is unblocked.
 *
 *   W_i = (wakeup_cpu_ns + cpu_to_gpu_offset_ns) − completion_gpu_ns
 *
 *   completion_gpu_ns — %globaltimer written by the SEGMENT KERNEL's own last
 *                       instruction into a pinned slot.  Identical kernel and
 *                       identical stamp point to both sibling binaries.
 *   wakeup_cpu_ns     — CLOCK_MONOTONIC captured immediately after the GCAPS
 *                       segment-end wait returns, BEFORE the remove ioctl.
 *
 * WHY BEFORE THE REMOVE IOCTL.  This is the counterpart of the seq binary's
 * "not ReleaseStatistics::completionTime" decision.  `gcapsGpuSegEnd` is
 *
 *     cudaEventRecord(stop, stream); cudaEventSynchronize(stop); ioctl(remove);
 *
 * The thread is *running again* the moment cudaEventSynchronize returns; the
 * remove ioctl that follows is GCAPS's runlist bookkeeping, not part of the
 * wake path.  Folding it into W_i would compare seq's wake path against
 * GCAPS's wake path *plus* a runlist reload, and the reload is already
 * measured properly elsewhere (the driver's own GCAPS_EV elapsed_us, and
 * scripts/measure_preempt_overhead.py).  It is reported here as a separate
 * column instead, mirroring how seq splits W into detect_ns + notify_ns:
 *
 *   W_ns                = wakeup − T_seg        (comparable across all arms)
 *   seg_end_ioctl_ns    = the remove ioctl      (GCAPS ε, wall, host-side)
 *   seg_begin_ioctl_ns  = the add ioctl         (paid before the kernel runs)
 *
 * THE WAIT PRIMITIVE is GCAPS's own and is NOT configurable here —
 * cudaEventSynchronize on the segment's `stop` event, created with
 * cudaEventBlockingSync, inside a context created with
 * cudaDeviceScheduleBlockingSync.  That is exactly what `gcapsGpuSegEnd` plus
 * SeqWorkload's suspend mode (-b 1) do, so W_i is GCAPS's real wake path and
 * not an approximation of it.  There is deliberately no switch for this: GCAPS
 * hard-codes cudaEventSynchronize in `gcapsGpuSegEnd`, so a stream-sync variant
 * would measure a configuration GCAPS cannot be run in.  The stream BASELINE
 * blocks in cudaStreamSynchronize because that is what a plain-streams program
 * does; that difference belongs to the arms being compared, not to a knob
 * inside this one.
 *
 * RELEASE CADENCE — jobs are released on an ABSOLUTE grid (t0 + i*period,
 * default 997 µs), matching both sibling binaries.  GCAPS has no monitor to
 * de-phase, so the co-primality argument in the seq header does not apply
 * here; the grid is kept anyway because EQ 2.2 is a head-to-head and an
 * unequal release rate or GPU duty cycle would leave a W_i difference
 * attributable to something other than the wake path.
 *
 * REAL-TIME.  `--realtime` is effectively MANDATORY for -i 1: the driver reads
 * the caller's rt_priority to decide whether it is a real-time task at all, so
 * without SCHED_FIFO the ioctl runs the best-effort path and measures nothing.
 *
 * RT is applied AFTER the CUDA runtime has been initialised, which is the rule
 * cpuWakeupLatencyBenchSeq states ("applied AFTER start() so CUDA's internal
 * threads do not inherit RT") — pthread_create defaults to
 * PTHREAD_INHERIT_SCHED, so a driver thread spawned by an RT thread would come
 * up SCHED_FIFO too.  MEASURED (CUDA 13.3, desktop driver): all four driver
 * worker threads appear at the FIRST cuda* runtime call and none afterwards,
 * including across hundreds of launch/event-sync cycles — so on that stack the
 * ordering is moot and cpuWakeupLatencyStreamBaseline, which switches to
 * SCHED_FIFO after its own first runtime call (cudaSetDeviceFlags), is equally
 * safe.  The ordering is kept because it costs nothing and makes the property
 * hold by construction rather than by driver version; re-check with
 * /proc/self/task if the CUDA version changes.  `--rt-early` moves the switch
 * ahead of the first runtime call, which is the one ordering that WOULD leak
 * RT into the driver's threads — for testing that, nothing else.
 *
 * Output columns are a superset of cpuWakeupLatencyStreamBaseline's shared set
 * (job_id,completion_ns,wakeup_ns,W_ns) so run_cpu_wakeup_latency.py can drive
 * this binary as a third arm.
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
 *   --rt-early            Apply SCHED_FIFO before the first CUDA call.
 *   --spin                Do NOT use blocking sync (GCAPS's default -b 0 mode).
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
/* cpuWakeupLatencyBenchSeq's RELEASE_CPU — the thread whose wake-up is timed. */
static constexpr int DEFAULT_PIN_CPU = 2;

/* Release cadence — MUST MATCH the sibling binaries. */
static constexpr uint64_t DEFAULT_RELEASE_PERIOD_US = 997;
/* Non-exec part of a job cycle — used only to warn when the requested release
 * period is too short for the grid to hold.
 *
 * With -i 0 that is a kernel launch plus the wake: tens of microseconds.  With
 * -i 1 it is dominated by the two GCAPS ioctls, each a full runlist reload with
 * wait_for_finish.  Sized from MEASUREMENT, not from adding up the parts: on the
 * R35.6.4 Orin, 500 solo jobs at exec=50 us gave an inter-release cycle of
 * p50 1263 us / max 1591 us, i.e. a non-exec part of ~1215 us.  (The two ioctls
 * are only ~590 us of that at p50 295 us each; the rest is launch, event record
 * and the wake.  Deriving the guard from the ioctl cost alone put it at 900 and
 * it STILL did not fire.)  Using the -i 0 figure for both is why a 997 us grid
 * silently degenerated to back-to-back releases -- 499 of 500 late. */
static constexpr uint64_t RELEASE_CYCLE_SLACK_US       = 40;
static constexpr uint64_t RELEASE_CYCLE_SLACK_IOCTL_US = 1250;
/* t0 is stamped this far in the FUTURE so job 0 actually sleeps to its grid
 * point instead of finding its deadline already past. */
static constexpr uint64_t STARTUP_MARGIN_NS         = 10000000ULL;   // 10 ms

static const char* GCAPS_CTRL_DEV = "/dev/nvgpu/igpu0/ctrl";

// ============================================================================
// Clocks
// ============================================================================

/**
 * host_ns — read CLOCK_MONOTONIC as a nanosecond timestamp.
 *
 * INTRA-CPU ONLY.  Drives the absolute release grid (sleep_until_ns), the
 * release_late check and the two ioctl durations — all of which are either
 * deadlines the kernel must accept or differences between two stamps on this
 * same clock.
 *
 * NOT for anything converted into the GPU %globaltimer domain: CLOCK_MONOTONIC
 * is frequency-slewed by NTP, and the slew RATE changes during a run, which
 * curves the CPU->GPU offset trajectory that clock_calib.cuh interpolates
 * linearly.  Cross-clock stamps use clock_calib_host_ns() (CLOCK_MONOTONIC_RAW)
 * instead.  The two clocks diverge by the accumulated slew since boot, so they
 * must never be subtracted from one another.
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
 * sleep_until_ns — block until an ABSOLUTE CLOCK_MONOTONIC deadline.
 *
 * Absolute rather than relative so the release grid cannot drift: a late wake
 * eats into the next period instead of pushing every subsequent release back.
 * EINTR retries the UNCHANGED deadline, so a signal resumes the cadence rather
 * than extending it.
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

// ============================================================================
// Segment kernel — byte-for-byte the sibling binaries' busy-wait + stamp, so
// the completion instant means the same thing in all three arms.
// ============================================================================

/**
 * wgc_gpu_ns — read the GPU's %globaltimer as a nanosecond timestamp.
 *
 * Device-side counterpart of host_ns().  %globaltimer is the clock W_i's
 * completion endpoint is expressed in, and it free-runs independently of the
 * CPU clocks, hence the calibration bracket.
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
 * Byte-for-byte the kernel cpuWakeupLatencyBenchSeq and
 * cpuWakeupLatencyStreamBaseline run, so the completion instant means the same
 * thing in all three EQ 2.2 arms and W_i is comparable across them.
 *
 * The completion stamp is taken by the kernel's OWN LAST INSTRUCTION, which is
 * what makes W_i independent of how long the segment ran: everything measured
 * afterwards is wake path, not execution.  Single-threaded (thread 0 only) so
 * the stamp is unambiguous.
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
 * Reports file, line and the runtime's own message.  Because it returns
 * EXIT_FAILURE it is only usable from a function whose return type is int
 * (in practice: main).
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
// GCAPS runlist-priority ioctl — the same call sequence as support.h's
// gcapsGpuSegBegin / gcapsGpuSegEnd, open-coded so the remove ioctl can be
// timed separately from the wait that precedes it.
// ============================================================================

/**
 * gcaps_runlist — issue one GCAPS runlist-priority ioctl.
 *
 * The same call support.h's gcapsGpuSegBegin/gcapsGpuSegEnd make, open-coded
 * here so the REMOVE ioctl can be timed separately from the wait that precedes
 * it — W_i deliberately ends before this call, and the ioctl is reported as its
 * own column (epsilon) rather than folded into the wake path.
 *
 * The driver decides whether the caller is a real-time task by reading its
 * rt_priority, so this is only meaningful once apply_rt() has succeeded.
 *
 * @param fd        open file descriptor for the nvgpu control device
 * @param pid       process whose TSGs are admitted or evicted
 * @param add_req   true to admit (segment begin), false to remove (segment end)
 * @param sync_mode driver-side path selector; UNRELATED to the CPU-side wait
 *                  primitive despite the name
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
 * Both steps are best-effort and warn rather than abort, but a failed
 * sched_setscheduler INVALIDATES a `-i 1` run: the driver reads rt_priority to
 * decide whether the caller is a real-time task at all, so every ioctl would
 * silently take the best-effort branch.  On this kernel
 * (CONFIG_RT_GROUP_SCHED=y) it fails even under sudo unless the caller has been
 * moved into a cgroup with a non-zero cpu.rt_runtime_us, which is why
 * run_gcaps_r35.sh is mandatory.
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
 * Sequence: parse options, apply RT (optionally before context creation via
 * --rt-early), open the GCAPS device, take the start calibration bracket with
 * the GPU idle, run `warmup` throwaway releases, then release n_jobs on an
 * absolute grid.  Each release is add ioctl -> segment kernel ->
 * cudaEventSynchronize -> RAW wakeup stamp -> remove ioctl, so the stamp lands
 * before any GCAPS bookkeeping.  Finally the end bracket closes and W_i is
 * computed in a post-run pass with the drift-corrected offset.
 *
 * The end-of-run calibration must itself be wrapped in a GCAPS add/remove
 * bracket: unbracketed GPU work after the last remove ioctl never completes,
 * because the process holds no runlist entry to run it on.
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
    bool     rtEarly         = false;
    bool     blockingSync    = true;
    int      pinCpu          = DEFAULT_PIN_CPU;
    uint64_t releasePeriodUs = DEFAULT_RELEASE_PERIOD_US;
    const char* outPath      = nullptr;

    /* Positional args are collected separately so that, unlike
     * cpuWakeupLatencyStreamBaseline, option order does not matter. */
    std::vector<const char*> pos;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--realtime") == 0)            realtime = true;
        else if (strcmp(argv[i], "--rt-early") == 0)       rtEarly = true;
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
    /* OVERLAP.  One segment in flight at a time: the next grid point must not
     * be due before the current job's remove ioctl has returned, or the grid
     * degenerates to back-to-back releases (every sleep expires in the past)
     * and W_i is measured at one fixed phase of whatever periodic activity the
     * driver has, instead of over a uniform sweep of it. */
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

    // ---- Blocking sync MUST be requested before the context exists ---------
    // Runtime-API equivalent of the driver-API CU_CTX_SCHED_BLOCKING_SYNC that
    // SeqWorkload::taskInit() uses in suspend mode (-b 1), and the same call
    // cpuWakeupLatencyStreamBaseline makes.  Without it the wait spin-polls and
    // W_i measures a spin loop rather than an OS wake-up.
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
    fprintf(stderr, "  realtime     : %s%s\n", realtime ? "yes" : "no",
            rtEarly ? " (applied before context creation)" : "");
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

    /* --rt-early: switch to SCHED_FIFO BEFORE any cuda* call, i.e. before the
     * driver spawns its worker threads.  That is the ordering under which they
     * would inherit RT (PTHREAD_INHERIT_SCHED).  Testing aid; see the header. */
    if (realtime && rtEarly) apply_rt(pinCpu);

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

    /* Default: RT after the runtime is up, so the driver's worker threads were
     * created at default priority.  Matches the seq arm's rule. */
    if (realtime && !rtEarly) apply_rt(pinCpu);

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
    printf("# rt_applied: %s\n",          realtime ? (rtEarly ? "before_context"
                                                              : "after_context")
                                                   : "none");
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
