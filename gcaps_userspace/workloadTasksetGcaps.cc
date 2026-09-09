/*
 * workloadTasksetGcaps.cc
 *
 * GCAPS-style mixed taskset on the GCAPS userspace harness — the GCAPS analog
 * of singleTaskSched's bench/workloadTasksetBench.cu.
 *
 * The task structure is ported verbatim from the source (GCAPS Table 4 for
 * C_i, T_i = D_i and CPU affinities); the GPU segments are the ported real
 * workloads (matmul / histogram / convolution / MLP) run as one GCAPS GPU
 * segment per period.  Workload SIZES match the source's enlarged tuning for
 * the fast AGX Orin GPU, and task 8 (mlp_1024x8) is the source's extension
 * BEYOND GCAPS Table 4: an 8-layer square MLP (width 1024) — its 16 kernels
 * (matmul + bias/ReLU per layer) all run inside the single GCAPS segment,
 * where the source runs them as 16 SequenceScheduler segments.
 *
 * Priorities are DEADLINE-MONOTONIC (shorter T = D => higher priority), NOT
 * the GCAPS Table 4 assignment, matching the source benchmark.  ALL tasks are
 * real-time (SCHED_FIFO); there are no best-effort tasks (which also avoids
 * the GCAPS driver deadlock with >1 best-effort GPU task — see
 * best-effort-tasks-bug.md).
 *
 *   1  hist_128M      histogram 128M   C=1ms  T=100ms CPU={1}   FIFO=8
 *   2  mm_2048        matmul 2048      C=2ms  T=150ms CPU={2}   FIFO=6
 *   3  cpu_only       (CPU only)       C=67ms T=200ms CPU={2}   FIFO=5
 *   4  conv_4096_k7   conv 4096^2 k7   C=12ms T=300ms CPU={1}   FIFO=2
 *   5  conv_4096_k15  conv 4096^2 k15  C=2ms  T=400ms CPU={1}   FIFO=1
 *   6  mm_2560        matmul 2560      C=4ms  T=200ms CPU={4}   FIFO=4
 *   7  hist_64M       histogram 64M    C=4ms  T=134ms CPU={4,5} FIFO=7
 *   8  mlp_1024x8     MLP 1024x8 (DNN) C=2ms  T=250ms CPU={3}   FIFO=3
 *
 * One forked process per task (separate CUcontext) — required because GCAPS's
 * runlist-priority ioctl acts on a pid, so tasks must be distinct processes
 * (unlike the source, which uses one in-process SequenceScheduler + threads).
 *
 * Per period we record cpu_phase / sched_preempt_overhead / gpu_exec / response:
 *   gpu_exec_ms               = cudaEvent(segment start -> stop) (on-GPU time)
 *   response_ms               = host wall, period release -> segment done
 *   cpu_phase_ms              = the C_i CPU busy-wait
 *   sched_preempt_overhead_ms = response - cpu_phase - gpu_exec
 *       — everything outside the CPU phase and the measured on-GPU window:
 *         kernel-launch and ioctl latency plus GPU-side scheduling/preemption
 *         delay.  GCAPS is preemptive (a segment can be preempted in favour of
 *         a higher-priority one), so this is scheduling + preemption overhead,
 *         not a FIFO queue wait.
 *
 * All tasks request SCHED_FIFO — run with sudo for that to take effect (warns
 * and continues otherwise).
 *
 * Start-up is STAGGERED: each task creates its CUDA context and runs its
 * warm-up executions (-w) in its own 1 s slot, so the contexts are brought up
 * one at a time. Creating and first-touching all contexts simultaneously spins
 * in the driver during concurrent context bring-up (and deadlocks under -i 1);
 * serialising it avoids that. The synchronized periodic run only begins after
 * every context is warm, so there is a ~(NUM_TASKS+1) s one-time warm-up
 * before the timed window.
 *
 * Verification runs AFTER the measurement window (post-run), not during init:
 * mlp_1024x8's host reference is a full 8-layer forward pass (~1e10 MACs,
 * tens of seconds on one A78 core), which used to overrun the init slot and
 * the whole experiment window, so the task silently recorded zero samples.
 * Post-run, a slow reference costs only shutdown time; a FAILED verdict still
 * invalidates the run.  Warm-up (formerly a side effect of verify) is now
 * explicit: -w N runs each task's segment N times during its init slot.
 *
 * Usage:  workloadTasksetGcaps [-i 0|1] [-b 0|1] [-d DURATION_S]
 *                              [-k N] [-S SCALE] [-w WARMUP]
 *         (defaults: -i 0 -b 0 -d 30 -S 1.0 -w 1, all GPU tasks)
 *         -k N : activate only the first N GPU tasks (CPU-only task always
 *                runs) — for bisecting how many concurrent GPU contexts the
 *                GCAPS elevation path tolerates before deadlocking.
 *         -S SCALE : multiply every GPU task's period (= deadline) by SCALE,
 *                leaving C_i / G_i and the CPU-only task's period unchanged, so
 *                SCALE < 1 raises GPU utilization by x1/SCALE (mirrors
 *                singleTaskSched's workloadTasksetBench -s SCALE). Higher
 *                utilization means more preemption churn and a higher chance of
 *                hitting the runlist-cache-desync deadlock (use a shorter -d).
 * Output (-i 1 -> gcaps, else tsg):
 *   results/workloadBench/taskset_{gcaps,tsg}_trace.csv
 *   results/workloadBench/taskset_{gcaps,tsg}_results.csv
 */

#include <fcntl.h>
#include <unistd.h>
#include <getopt.h>
#include <sched.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

#include <cuda_runtime.h>

#include "app/seqworkload/seqworkload.h"

// ============================================================================
// Task definitions (GCAPS Table 4 structure, ported real workloads)
// ============================================================================

static constexpr int NUM_TASKS = 8;

struct BenchTaskDef {
	const char*  name;
	bool         is_gpu;
	SeqWlType    wlType;
	unsigned int wlP1, wlP2;
	uint32_t     ci_ms;
	uint32_t     ti_ms;
	int          cpu_cores[2];   /* cpu_cores[1] = -1 for single-core */
	int          fifo_priority;  /* 0 = SCHED_OTHER */
};

/* Deadline-monotonic priorities (shorter T = D => higher priority), matching
 * the source benchmark: SCHED_FIFO ranks all 8 tasks 8..1 by deadline.  All
 * tasks are real-time — no best-effort tasks, which (a) avoids the GCAPS
 * driver best-effort deadlock (>1 concurrently-running best-effort GPU task —
 * see best-effort-tasks-bug.md) and (b) matches the SequenceScheduler
 * benchmark, which priority-schedules every GPU task. */
static const BenchTaskDef TASKS[NUM_TASKS] = {
	{"hist_128M",     true,  SeqWlType::HISTOGRAM,   128u << 20, 0,  1, 100, {1, -1}, 8},
	{"mm_2048",       true,  SeqWlType::MATMUL,      2048,       0,  2, 150, {2, -1}, 6},
	{"cpu_only",      false, SeqWlType::MATMUL,      0,          0, 67, 200, {2, -1}, 5},
	{"conv_4096_k7",  true,  SeqWlType::CONVOLUTION, 4096,       7, 12, 300, {1, -1}, 2},
	{"conv_4096_k15", true,  SeqWlType::CONVOLUTION, 4096,      15,  2, 400, {1, -1}, 1},
	{"mm_2560",       true,  SeqWlType::MATMUL,      2560,       0,  4, 200, {4, -1}, 4},
	{"hist_64M",      true,  SeqWlType::HISTOGRAM,   64u << 20,  0,  4, 134, {4,  5}, 7},
	{"mlp_1024x8",    true,  SeqWlType::MLP,         1024,       8,  2, 250, {3, -1}, 3},
};

// ============================================================================
// Time helpers
// ============================================================================

static uint64_t host_ns()
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static inline uint64_t thread_cpu_ns(bool* ok);

/* Consume duration_ns of CPU TIME, not of wall time.
 *
 * The wall-bounded version this replaces (`end = host_ns() + d; while (host_ns()
 * < end);`) silently shrank a task's execution time whenever it was preempted:
 * wall advanced during the preemption but the thread's CPU clock did not, so the
 * emulated C_i came out BELOW C_i and the task's utilisation read low.  That is
 * backwards -- a preempted task should still execute for C_i and simply finish
 * later.  Measured on a SCHED_OTHER run: cpu_only lost 18 ms over 5 s (33.13%
 * against its 33.5% duty), and the loss grows with contention, which is exactly
 * when the number matters most.
 *
 * CLOCK_THREAD_CPUTIME_ID is NOT vDSO-served -- each read is a real syscall,
 * ~0.5-1 us -- so spinning directly on it would spend most of C_i inside
 * clock_gettime.  Instead burn a bounded slice on the cheap vDSO CLOCK_MONOTONIC
 * and re-check the CPU clock once per slice: at 200 us the check costs well under
 * 1% of the slice, and any preemption inside a slice is picked up at its end and
 * repaid by another iteration.  The clock reads' own CPU is inside the accounted
 * total, so the loop self-corrects rather than overshooting.
 *
 * Falls back to the wall-bounded behaviour if the CPU clock is unavailable. */
static constexpr uint64_t CPU_SPIN_SLICE_NS = 200000;   /* 200 us */

static bool g_wall_bounded_ci = false;   /* -W: legacy wall-bounded C_i */

static void cpu_busy_wait_ns(uint64_t duration_ns)
{
	if (g_wall_bounded_ci) {
		const uint64_t end = host_ns() + duration_ns;
		while (host_ns() < end) {}
		return;
	}

	bool ok = true;
	const uint64_t cpu0 = thread_cpu_ns(&ok);
	if (!ok) {
		const uint64_t end = host_ns() + duration_ns;
		while (host_ns() < end) {}
		return;
	}

	uint64_t consumed = 0;
	while (consumed < duration_ns) {
		uint64_t slice = duration_ns - consumed;
		if (slice > CPU_SPIN_SLICE_NS) slice = CPU_SPIN_SLICE_NS;
		const uint64_t wend = host_ns() + slice;
		while (host_ns() < wend) {}
		consumed = thread_cpu_ns(&ok) - cpu0;
		if (!ok) break;
	}
}

static void sleep_until_abs_ns(uint64_t target_ns)
{
	struct timespec ts;
	ts.tv_sec  = (time_t)(target_ns / 1000000000ULL);
	ts.tv_nsec = (long)(target_ns % 1000000000ULL);
	while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, nullptr) != 0) {}
}

// Inherited by every forked child (set before fork()).
static uint64_t g_init_start_ns   = 0;   // epoch for the staggered per-task init
static uint64_t g_sync_start_ns   = 0;   // synchronized periodic-loop start
static uint64_t g_experiment_ns   = 0;
// Period (= deadline) multiplier for the GPU tasks only — leaves C_i / G_i (and
// the CPU-only task's period) unchanged, so SCALE < 1 raises GPU utilization by
// x1/SCALE. Mirrors singleTaskSched's workloadTasksetBench -s SCALE. Set by -S.
static double   g_period_scale    = 1.0;
// Warm-up executions per GPU task during its init slot (CLI -w).  Replaces the
// warm-up that verify() used to provide during init; verification itself now
// runs after the measurement window (post-run verify in run_task).
static int      g_warmup_runs     = 1;

// Per-task init is staggered by this much so the CUDA contexts are created
// and first touch the GPU one at a time. Creating/initialising all contexts
// simultaneously spins (and deadlocks under -i 1) in the driver during
// concurrent context bring-up; serialising it avoids the storm. See README.
static constexpr uint64_t INIT_STAGGER_NS = 1000000000ULL; /* 1 s per task */

// ============================================================================
// Child process: run one task for the experiment window, dump its trace rows.
// ============================================================================

/* CPU time of the CALLING THREAD.
 *
 * This is the GCAPS counterpart of the aggregate U that workloadTasksetBench
 * emits, and it is measurable here for a reason worth stating: this harness
 * creates no pthreads, so each GCAPS task is exactly ONE thread in one forked
 * process.  CLOCK_THREAD_CPUTIME_ID on that thread is therefore precisely that
 * task's CPU and EXCLUDES the CUDA driver's helper threads -- the same scope as
 * seq's per-thread U, which sums only threads it created.
 *
 * The GCAPS ioctls ARE included: syscall execution is charged as system time to
 * the calling thread.  Where seq splits its scheduling cost between task threads
 * and a monitor thread, GCAPS has no monitor -- it schedules inside the driver,
 * in each task's own syscall context -- so the whole of it lands here.  Same
 * total scope, different distribution.
 *
 * What this deliberately does NOT capture is the CUDA driver's per-context
 * helper threads.  GCAPS needs one PROCESS per task (the ioctl acts on a pid),
 * so it pays 8 contexts' worth of them against seq's 1, and that is a real cost
 * of the architecture.  scripts/bench/sample_cpu_util.py's process-subtree walk
 * is the lens that sees it; the two numbers together decompose the difference.
 */
static inline uint64_t thread_cpu_ns(bool* ok)
{
	struct timespec ts;
	if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts) != 0) {
		if (ok) *ok = false;
		return 0;
	}
	return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static void run_task(int task_idx, int fd, bool sync_mode, bool ioctl_enabled,
                     const char* mode_tag)
{
	const BenchTaskDef& td = TASKS[task_idx];

	cpu_set_t cpuset;
	CPU_ZERO(&cpuset);
	CPU_SET(td.cpu_cores[0], &cpuset);
	if (td.cpu_cores[1] >= 0) CPU_SET(td.cpu_cores[1], &cpuset);
	if (sched_setaffinity(gettid(), sizeof(cpuset), &cpuset) != 0)
		fprintf(stderr, "[task %s] WARNING: sched_setaffinity failed\n",
		        td.name);

	if (td.fifo_priority > 0) {
		struct sched_param sp;
		sp.sched_priority = td.fifo_priority;
		if (sched_setscheduler(0, SCHED_FIFO, &sp) != 0)
			fprintf(stderr, "[task %s] WARNING: SCHED_FIFO pri=%d failed "
			                "— continuing SCHED_OTHER\n",
			        td.name, td.fifo_priority);
	}

	/* Staggered start-up: each task initialises its CUDA context (and runs
	 * its -w warm-up executions) in its own slot so context bring-up is
	 * serialised across tasks rather than a simultaneous multi-context storm
	 * that hangs the driver. */
	sleep_until_abs_ns(g_init_start_ns + (uint64_t)task_idx * INIT_STAGGER_NS);

	SeqWorkload* wl = nullptr;
	if (td.is_gpu) {
		/* Waits always block rather than spin: this arm evaluates only the
		 * blocking variant, matching the stream baseline it is compared
		 * against.  SeqWorkload keeps the parameter for the other GCAPS
		 * binaries, which still expose it. */
		wl = new SeqWorkload(td.wlType, td.wlP1, td.wlP2, fd, sync_mode,
		                     ioctl_enabled, /*suspension=*/true);
		wl->taskInit();
		/* Warm-up only — verification is deferred to after the measurement
		 * window (see post-run verify below), so a slow host reference
		 * cannot overrun the init slot and eat the experiment.  warmup()
		 * launches WITHOUT the GCAPS bracket: no ioctl must run before the
		 * synchronized start, because GCAPS runlist rebuilds interleaved
		 * with the later tasks' context bring-up (a normal-path runlist
		 * writer) stale the driver's runlist cache and raise the odds of
		 * the mid-run cache-desync hang (see runlist-cache-desync-bug.md).
		 * Unbracketed launches are safe at init time precisely because no
		 * ioctl has run yet. */
		for (int w = 0; w < g_warmup_runs; ++w)
			wl->warmup();
		wl->recordPriority(td.fifo_priority);
	}

	/* Scale GPU-task periods (= deadlines) only; C_i and the CPU-only task are
	 * left unchanged so SCALE<1 tightens GPU contention. */
	const double   ti_ms_eff = td.is_gpu ? (double)td.ti_ms * g_period_scale
	                                     : (double)td.ti_ms;
	const uint64_t period_ns = (uint64_t)(ti_ms_eff * 1.0e6);
	const uint64_t ci_ns     = (uint64_t)td.ci_ms * 1000000ULL;
	const uint64_t end_ns    = g_sync_start_ns + g_experiment_ns;

	const int my_pid = getpid();

	struct Rec {
		uint32_t period_idx;
		double   period_start_ms, cpu_ms, ovh_ms, gpu_ms, resp_ms;
		bool     missed;
		/* Absolute CLOCK_MONOTONIC bounds of the GPU segment (0 for the
		 * CPU-only task). Same clock base as the driver's GCAPS_EV ts=, so
		 * measure_preempt_overhead.py can intersect driver suspend intervals
		 * with this window to recover active-execution time. */
		uint64_t seg_begin_ns, seg_done_ns;
	};
	std::vector<Rec> records;
	records.reserve(g_experiment_ns / period_ns + 4);

	bool     first_period      = true;
	uint32_t period_idx        = 0;
	uint64_t next_period_start = g_sync_start_ns;

	/* Window opens here: AFTER taskInit() and the warm-up runs, so context
	 * bring-up is excluded.  Between this read and g_sync_start_ns the thread
	 * only sleeps, and sleeping costs no CPU, so this is the CPU-at-window-open
	 * even though the sleep happens inside the loop below. */
	bool cpu_clk_ok = true;
	const uint64_t win_cpu0 = thread_cpu_ns(&cpu_clk_ok);

	while (host_ns() < end_ns) {
		const uint64_t period_start_ns = next_period_start;
		sleep_until_abs_ns(period_start_ns);
		if (host_ns() >= end_ns) break;

		cpu_busy_wait_ns(ci_ns);

		double resp_ms = 0.0, cpu_ms = 0.0, ovh_ms = 0.0, gpu_ms = 0.0;
		uint64_t seg_begin_ns = 0, seg_done_ns = 0;

		if (wl != nullptr) {
			seg_begin_ns = host_ns();
			wl->taskCallback(0, 0);            /* one GCAPS GPU segment */
			seg_done_ns  = host_ns();

			cpu_ms  = (double)(seg_begin_ns - period_start_ns) / 1.0e6;
			gpu_ms  = (double)wl->lastGpuMs();
			resp_ms = (double)(seg_done_ns - period_start_ns) / 1.0e6;
			ovh_ms  = resp_ms - cpu_ms - gpu_ms;
			if (ovh_ms < 0.0) ovh_ms = 0.0;
		} else {
			const uint64_t end_cpu_ns = host_ns();
			resp_ms = (double)(end_cpu_ns - period_start_ns) / 1.0e6;
			cpu_ms  = resp_ms;
		}

		if (!first_period) {
			Rec rec;
			rec.period_idx      = period_idx;
			rec.period_start_ms = (double)(period_start_ns - g_sync_start_ns)
			                      / 1.0e6;
			rec.cpu_ms   = cpu_ms;
			rec.ovh_ms   = ovh_ms;
			rec.gpu_ms   = gpu_ms;
			rec.resp_ms  = resp_ms;
			rec.missed   = (resp_ms > ti_ms_eff);
			rec.seg_begin_ns = seg_begin_ns;
			rec.seg_done_ns  = seg_done_ns;
			records.push_back(rec);
		}
		first_period = false;
		++period_idx;
		next_period_start += period_ns;
	}

	const uint64_t win_cpu1  = thread_cpu_ns(&cpu_clk_ok);
	const uint64_t win_wall1 = host_ns();

	/* Per-thread CPU for this task, in its own file rather than the trace
	 * fragment: merge_and_summarise() copies every fragment line verbatim into
	 * the trace CSV, so a scalar row there would corrupt it.  Written before
	 * the verify phase for the same reason the fragment is. */
	{
		char cpath[176];
		snprintf(cpath, sizeof(cpath),
		         "results/workloadBench/.cpu_%s_%d.csv", mode_tag, task_idx);
		uint64_t bd[6] = {0, 0, 0, 0, 0, 0};
		if (wl != nullptr) wl->cpuBreakdownNs(bd);
		FILE* cf = fopen(cpath, "w");
		if (cf) {
			fprintf(cf, "%d,%s,%llu,%llu,%d,%llu,%llu,%llu,%llu,%llu,%llu\n",
			        task_idx, td.name,
			        (unsigned long long)(win_cpu1 - win_cpu0),
			        (unsigned long long)(win_wall1 - g_sync_start_ns),
			        cpu_clk_ok ? 1 : 0,
			        (unsigned long long)bd[0], (unsigned long long)bd[1],
			        (unsigned long long)bd[2], (unsigned long long)bd[3],
			        (unsigned long long)bd[4], (unsigned long long)bd[5]);
			fclose(cf);
		}
	}

	/* Each child writes its own trace fragment; parent merges them.  Written
	 * BEFORE the post-run verify so the data is on disk even if the (possibly
	 * slow) verification is interrupted. */
	char path[160];
	snprintf(path, sizeof(path),
	         "results/workloadBench/.tsk_%s_%d.csv", mode_tag, task_idx);
	FILE* f = fopen(path, "w");
	if (!f) { fprintf(stderr, "[task %s] could not open %s\n", td.name, path);
	          return; }
	for (const Rec& r : records)
		fprintf(f, "%d,%s,%u,%.3f,%.3f,%.3f,%.3f,%.3f,%.0f,%d,%d,%llu,%llu,%d\n",
		        task_idx, td.name, r.period_idx, r.period_start_ms,
		        r.cpu_ms, r.ovh_ms, r.gpu_ms, r.resp_ms,
		        ti_ms_eff, (int)r.missed,
		        my_pid,
		        (unsigned long long)r.seg_begin_ns,
		        (unsigned long long)r.seg_done_ns,
		        td.fifo_priority);
	fclose(f);

	/* Post-run verification — off the timing path by design.  mlp_1024x8's
	 * host reference takes tens of seconds on one A78 core; announce it so a
	 * long silent tail is not mistaken for a hang.
	 *
	 * verify(false): check the residual outputs of the LAST executed segment
	 * (all kernels overwrite their outputs) instead of launching afresh.
	 * verify() runs its GPU work — including the D2H copies of the checks —
	 * inside a GCAPS segment bracket, because after the experiment every
	 * task's final gcapsGpuSegEnd left this context's TSG entries (compute
	 * AND copy engine) off the runlist, where unbracketed GPU work is never
	 * redispatched and blocks forever.  Under -i 1 the brackets serialise
	 * the tasks' verifications by priority; mlp's ~minute host pass holds
	 * its segment meanwhile — a shutdown cost only. */
	if (wl) {
		fprintf(stderr, "[task %s] post-run verify...\n", td.name);
		const bool ok = wl->verify(false);
		fprintf(stderr, "[task %s] post-run verify %s\n", td.name,
		        ok ? "PASS" : "FAILED");
		wl->taskFinish();
		delete wl;
	}
}

// ============================================================================
// Parent: merge per-task fragments into the trace + summary CSVs.
// ============================================================================

static double compute_p95(std::vector<double>& v)
{
	std::sort(v.begin(), v.end());
	if (v.empty()) return 0.0;
	size_t idx = (size_t)(0.95 * (double)v.size());
	if (idx >= v.size()) idx = v.size() - 1;
	return v[idx];
}

/* Aggregate the per-task CPU into U, in the units and file format
 * workloadTasksetBench uses (host_cpu_util_pct, taskset_<mode>_hostcpu.csv), so
 * a GCAPS number and a seq/stream number are directly comparable.
 *
 * Denominator: the LONGEST task window.  Every task shares g_sync_start_ns and
 * runs while host_ns() < end_ns, so the windows differ only by however far each
 * task's final period overran the end -- at most one period. */
static void report_host_cpu(const char* mode_tag)
{
	struct Row { char name[48]; unsigned long long cpu, wall; int ok;
	             unsigned long long segbeg, launch, segend, elapsed,
	                                probe, calls; };
	Row rows[NUM_TASKS];
	int  nrows = 0;
	unsigned long long total_cpu = 0, window = 0;

	for (int i = 0; i < NUM_TASKS; ++i) {
		char cpath[176];
		snprintf(cpath, sizeof(cpath),
		         "results/workloadBench/.cpu_%s_%d.csv", mode_tag, i);
		FILE* cf = fopen(cpath, "r");
		if (!cf) continue;
		int idx = 0, ok = 0;
		char nm[48] = {0};
		unsigned long long c = 0, w = 0, b0 = 0, b1 = 0, b2 = 0, b3 = 0,
		                   pr = 0, ca = 0;
		if (fscanf(cf, "%d,%47[^,],%llu,%llu,%d,%llu,%llu,%llu,%llu,%llu,%llu",
		           &idx, nm, &c, &w, &ok, &b0, &b1, &b2, &b3, &pr, &ca) == 11) {
			Row& r = rows[nrows++];
			snprintf(r.name, sizeof(r.name), "%s", nm);
			r.cpu = c; r.wall = w; r.ok = ok;
			r.segbeg = b0; r.launch = b1; r.segend = b2; r.elapsed = b3;
			r.probe = pr; r.calls = ca;
			if (ok) {
				total_cpu += c;
				if (w > window) window = w;
			}
		}
		fclose(cf);
		remove(cpath);
	}
	if (nrows == 0) return;

	const double util = (window > 0)
	                  ? 100.0 * (double)total_cpu / (double)window : 0.0;

	printf("\n# host_cpu_ns: %llu\n", total_cpu);
	printf("# host_wall_ns: %llu\n", window);
	printf("# host_cpu_util_pct: %.4f  (of ONE core; %d task threads, no "
	       "monitor -- GCAPS schedules in the driver, in each task's own "
	       "syscall context)\n", util, nrows);

	printf("\nWhole-window host CPU (CLOCK_THREAD_CPUTIME_ID, whole thread)\n");
	printf("includes each task's emulated C_i spin -- NOT scheduler overhead.\n");
	printf("EXCLUDES the CUDA driver's per-context helper threads: GCAPS runs one\n");
	printf("context PER TASK, and that cost is only visible to the process-subtree\n");
	printf("sampler (scripts/bench/sample_cpu_util.py).\n");
	printf("%-16s  %8s  %14s\n", "Thread", "cpu(s)", "% of one core");
	printf("%-16s  %8s  %14s\n", "----------------", "--------",
	       "--------------");

	char hcPath[176];
	snprintf(hcPath, sizeof(hcPath),
	         "results/workloadBench/taskset_%s_hostcpu.csv", mode_tag);
	FILE* hc = fopen(hcPath, "w");
	if (hc)
		fprintf(hc, "thread_name,thread_kind,cpu_ns,window_wall_ns,"
		            "pct_one_core,clocks_ok\n");

	for (int i = 0; i < nrows; ++i) {
		const Row& r = rows[i];
		const double pct = (r.wall > 0)
		                 ? 100.0 * (double)r.cpu / (double)r.wall : 0.0;
		printf("%-16s  %8.3f  %13.2f%%\n", r.name,
		       (double)r.cpu / 1.0e9, pct);
		if (hc)
			fprintf(hc, "%s,task,%llu,%llu,%.4f,%d\n",
			        r.name, r.cpu, r.wall, pct, r.ok);
	}

	/* Where a GPU task's CPU goes ABOVE its emulated C_i.  The four phases are
	 * the four statements of SeqWorkload::taskCallback(); everything else in the
	 * period is the C_i spin, which is exactly C_i by construction now that it
	 * is CPU-bounded.  probe is this instrument's own five clock reads per call,
	 * charged rather than hidden. */
	bool any_gpu = false;
	for (int i = 0; i < nrows; ++i) if (rows[i].calls) any_gpu = true;
	if (any_gpu) {
		printf("\nPer-period CPU inside taskCallback (us/call, "
		       "CLOCK_THREAD_CPUTIME_ID)\n");
		printf("%-16s %8s %10s %8s %9s %8s %7s\n", "Task", "calls",
		       "seg_begin", "launch", "seg_end", "elapsed", "probe");
		printf("%-16s %8s %10s %8s %9s %8s %7s\n", "----------------",
		       "--------", "----------", "--------", "---------",
		       "--------", "-------");
		for (int i = 0; i < nrows; ++i) {
			const Row& r = rows[i];
			if (!r.calls) continue;
			const double n = (double)r.calls;
			printf("%-16s %8llu %10.2f %8.2f %9.2f %8.2f %7.2f\n", r.name,
			       r.calls, r.segbeg / n / 1e3, r.launch / n / 1e3,
			       r.segend / n / 1e3, r.elapsed / n / 1e3, r.probe / n / 1e3);
		}
		printf("  seg_begin = cudaEventRecord + GCAPS add ioctl;  "
		       "launch = cudaEventRecord + kernels\n");
		printf("  seg_end   = cudaEventRecord + cudaEventSynchronize + remove "
		       "ioctl;  elapsed = cudaEventElapsedTime\n");
	}
	if (hc) {
		fprintf(hc, "TOTAL,aggregate,%llu,%llu,%.4f,1\n",
		        total_cpu, window, util);
		fclose(hc);
		printf("\nHost CPU written to %s\n", hcPath);
	}
}

static void merge_and_summarise(const char* mode_tag)
{
	char tracePath[128], resultsPath[128];
	snprintf(tracePath, sizeof(tracePath),
	         "results/workloadBench/taskset_%s_trace.csv", mode_tag);
	snprintf(resultsPath, sizeof(resultsPath),
	         "results/workloadBench/taskset_%s_results.csv", mode_tag);

	FILE* tr = fopen(tracePath, "w");
	if (!tr) { fprintf(stderr, "could not open %s\n", tracePath); return; }
	fprintf(tr, "task_id,task_name,period_idx,period_start_ms,cpu_phase_ms,"
	            "sched_preempt_overhead_ms,gpu_exec_ms,response_ms,"
	            "deadline_ms,missed,pid,seg_begin_ns,seg_done_ns,"
	            "fifo_priority\n");

	std::vector<double> resp[NUM_TASKS];

	for (int i = 0; i < NUM_TASKS; ++i) {
		char path[160];
		snprintf(path, sizeof(path),
		         "results/workloadBench/.tsk_%s_%d.csv", mode_tag, i);
		FILE* f = fopen(path, "r");
		if (!f) continue;
		char line[256];
		while (fgets(line, sizeof(line), f)) {
			fputs(line, tr);
			/* response_ms is field 8 (0-based 7). */
			char buf[256]; strncpy(buf, line, sizeof(buf)); buf[255] = 0;
			int field = 0; char* tok = strtok(buf, ",");
			double r = 0.0;
			while (tok) {
				if (field == 7) { r = atof(tok); break; }
				tok = strtok(nullptr, ","); ++field;
			}
			resp[i].push_back(r);
		}
		fclose(f);
		remove(path);
	}
	fclose(tr);
	printf("Trace written to %s\n", tracePath);

	FILE* csv = fopen(resultsPath, "w");
	if (csv)
		fprintf(csv, "task_id,name,mort_ms,mean_ms,min_ms,p95_ms,"
		             "avg_rel_range,sample_count\n");

	printf("\n%-16s  %9s  %9s  %9s  %9s  %10s  %8s\n",
	       "Task", "MORT(ms)", "Mean(ms)", "Min(ms)", "P95(ms)",
	       "Rel.Range", "Samples");
	for (int i = 0; i < NUM_TASKS; ++i) {
		std::vector<double>& v = resp[i];
		if (v.empty()) {
			printf("%-16s  (no samples)\n", TASKS[i].name);
			if (csv) fprintf(csv, "%d,%s,0,0,0,0,0,0\n", i, TASKS[i].name);
			continue;
		}
		const double mort = *std::max_element(v.begin(), v.end());
		const double vmin = *std::min_element(v.begin(), v.end());
		const double mean = std::accumulate(v.begin(), v.end(), 0.0)
		                    / (double)v.size();
		const double pp95 = compute_p95(v);   /* sorts v in place */
		const double rel  = (mort > 0.0) ? (mort - vmin) / mort : 0.0;
		printf("%-16s  %9.3f  %9.3f  %9.3f  %9.3f  %10.4f  %8zu\n",
		       TASKS[i].name, mort, mean, vmin, pp95, rel, v.size());
		if (csv)
			fprintf(csv, "%d,%s,%.6f,%.6f,%.6f,%.6f,%.6f,%zu\n",
			        i, TASKS[i].name, mort, mean, vmin, pp95, rel, v.size());
	}
	if (csv) { fclose(csv); printf("\nResults written to %s\n", resultsPath); }
}

// ============================================================================
// main
// ============================================================================

int main(int argc, char** argv)
{
	setvbuf(stdout, nullptr, _IONBF, 0);

	int ioctl_enabled = 0, sync_mode = 0;
	uint64_t duration_s = 30;
	int gpu_limit = -1;   /* -1 = all GPU tasks; else activate only first N */
	int opt;
	while ((opt = getopt(argc, argv, "i:b:d:k:S:w:W")) != EOF) {
		switch (opt) {
			case 'i': ioctl_enabled = atoi(optarg); break;
			case 'b': sync_mode     = atoi(optarg); break;
			case 'd': duration_s    = strtoull(optarg, nullptr, 10); break;
			case 'k': gpu_limit     = atoi(optarg); break;
			case 'S': g_period_scale = atof(optarg); break;
			case 'w': g_warmup_runs = atoi(optarg); break;
			case 'W': g_wall_bounded_ci = true; break;
			default:  fprintf(stderr, "bad option\n"); return 1;
		}
	}
	if (g_warmup_runs < 0) {
		fprintf(stderr, "warmup (-w) must be >= 0\n");
		return 1;
	}
	if (sync_mode && ioctl_enabled) {
		fprintf(stderr, "IOCTL and sync mode are mutually exclusive\n");
		return 1;
	}
	if (g_period_scale <= 0.0) {
		fprintf(stderr, "scale (-S) must be > 0\n");
		return 1;
	}
	const char* mode_tag = ioctl_enabled ? "gcaps" : "tsg";

	int fd = open("/dev/nvgpu/igpu0/ctrl", O_RDWR);
	if (ioctl_enabled && fd < 0) { perror("open /dev/nvgpu/igpu0/ctrl");
	                               return 1; }

	mkdir("results", 0755);
	mkdir("results/workloadBench", 0755);

	/* Lead time before the first task initialises (lets all forks settle). */
	static constexpr uint64_t PREINIT_NS = 500000000ULL;        /* 500 ms */
	/* The staggered inits occupy NUM_TASKS slots; the synchronized periodic
	 * run starts one extra margin later so every context is warmed and ready. */
	static constexpr uint64_t POSTINIT_MARGIN_NS = 1000000000ULL; /* 1 s */
	g_experiment_ns = duration_s * 1000000000ULL;
	g_init_start_ns = host_ns() + PREINIT_NS;
	g_sync_start_ns = g_init_start_ns
	                + (uint64_t)NUM_TASKS * INIT_STAGGER_NS
	                + POSTINIT_MARGIN_NS;

	printf("Taskset: mode=%s, duration=%llus, warmup=%d, staggered init "
	       "(%d x %.1f s) then experiment starts in %.1f s",
	       ioctl_enabled ? "GCAPS-ioctl" : "TSG-default",
	       (unsigned long long)duration_s, g_warmup_runs, NUM_TASKS,
	       (double)INIT_STAGGER_NS / 1.0e9,
	       (double)(g_sync_start_ns - host_ns()) / 1.0e9);
	if (gpu_limit >= 0)
		printf("  [GPU tasks capped at %d]", gpu_limit);
	if (g_period_scale != 1.0)
		printf("  [GPU periods x%.3g -> util x%.3g]",
		       g_period_scale, 1.0 / g_period_scale);
	printf("\n");

	/* Remove stale per-task trace fragments so a capped run (fewer tasks) does
	 * not merge leftovers from a previous, larger run. */
	for (int i = 0; i < NUM_TASKS; ++i) {
		char path[160];
		snprintf(path, sizeof(path),
		         "results/workloadBench/.tsk_%s_%d.csv", mode_tag, i);
		remove(path);
		snprintf(path, sizeof(path),
		         "results/workloadBench/.cpu_%s_%d.csv", mode_tag, i);
		remove(path);
	}

	std::vector<pid_t> children;
	int gpu_forked = 0;
	for (int i = 0; i < NUM_TASKS; ++i) {
		/* -k N: activate only the first N GPU tasks (skip the rest).
		 * CPU-only tasks always run — they create no GPU context. */
		if (TASKS[i].is_gpu && gpu_limit >= 0 && gpu_forked >= gpu_limit)
			continue;
		if (TASKS[i].is_gpu) ++gpu_forked;

		pid_t pid = fork();
		if (pid == 0) {
			run_task(i, fd, (bool)sync_mode, (bool)ioctl_enabled,
			         mode_tag);
			if (fd >= 0) close(fd);
			_exit(0);
		} else if (pid > 0) {
			children.push_back(pid);
		} else {
			perror("fork");
			for (pid_t c : children) { kill(c, SIGTERM); }
			return 1;
		}
	}

	for (pid_t c : children) { int st; waitpid(c, &st, 0); }
	if (fd >= 0) close(fd);

	merge_and_summarise(mode_tag);
	report_host_cpu(mode_tag);
	return 0;
}
