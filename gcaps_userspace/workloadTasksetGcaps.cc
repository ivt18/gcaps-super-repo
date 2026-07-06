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
 * Usage:  workloadTasksetGcaps [-i 0|1] [-s 0|1] [-b 0|1] [-d DURATION_S]
 *                              [-k N] [-S SCALE] [-w WARMUP]
 *         (defaults: -i 0 -s 0 -b 0 -d 30 -S 1.0 -w 1, all GPU tasks)
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

static void cpu_busy_wait_ns(uint64_t duration_ns)
{
	const uint64_t end = host_ns() + duration_ns;
	while (host_ns() < end) {}
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

static void run_task(int task_idx, int fd, bool sync_mode, bool ioctl_enabled,
                     bool suspension, const char* mode_tag)
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
		wl = new SeqWorkload(td.wlType, td.wlP1, td.wlP2, fd, sync_mode,
		                     ioctl_enabled, suspension);
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

	int ioctl_enabled = 0, suspension = 0, sync_mode = 0;
	uint64_t duration_s = 30;
	int gpu_limit = -1;   /* -1 = all GPU tasks; else activate only first N */
	int opt;
	while ((opt = getopt(argc, argv, "i:s:b:d:k:S:w:")) != EOF) {
		switch (opt) {
			case 'i': ioctl_enabled = atoi(optarg); break;
			case 's': suspension    = atoi(optarg); break;
			case 'b': sync_mode     = atoi(optarg); break;
			case 'd': duration_s    = strtoull(optarg, nullptr, 10); break;
			case 'k': gpu_limit     = atoi(optarg); break;
			case 'S': g_period_scale = atof(optarg); break;
			case 'w': g_warmup_runs = atoi(optarg); break;
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
			         (bool)suspension, mode_tag);
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
	return 0;
}
