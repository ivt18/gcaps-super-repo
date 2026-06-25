/*
 * workloadTasksetGcaps.cc
 *
 * GCAPS-style mixed taskset on the GCAPS userspace harness — the GCAPS analog
 * of singleTaskSched's bench/workloadTasksetBench.cu.
 *
 * The task structure is ported verbatim from the source (GCAPS Table 4): same
 * C_i, T_i = D_i, CPU affinities and SCHED_FIFO priorities; the GPU segments
 * are the ported real workloads (matmul / histogram / convolution) run as one
 * GCAPS GPU segment per period:
 *
 *   1  hist_16M       histogram 16M    C=1ms  T=100ms CPU={1}   FIFO=7
 *   2  mm_1024        matmul 1024      C=2ms  T=150ms CPU={2}   FIFO=6
 *   3  cpu_only       (CPU only)       C=67ms T=200ms CPU={2}   FIFO=5
 *   4  conv_1024_k7   conv 1024^2 k7   C=12ms T=300ms CPU={1}   FIFO=4
 *   5  conv_2048_k15  conv 2048^2 k15  C=2ms  T=400ms CPU={1}   FIFO=3
 *   6  mm_2048        matmul 2048      C=4ms  T=200ms CPU={4}   FIFO=2
 *   7  hist_4M        histogram 4M     C=4ms  T=67ms  CPU={4,5} FIFO=1
 *
 * NOTE: tasks 6 and 7 are GCAPS Table 4's best-effort tasks, but they are run
 * here at the two LOWEST real-time priorities rather than SCHED_OTHER — this
 * avoids a GCAPS driver deadlock with >1 best-effort GPU task (see
 * best-effort-tasks-bug.md) and matches the SequenceScheduler benchmark, which
 * has no best-effort class.
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
 * Start-up is STAGGERED: each task creates its CUDA context and runs verify()
 * in its own 1 s slot, so the 7 contexts are brought up one at a time. Creating
 * and first-touching all contexts simultaneously spins in the driver during
 * concurrent context bring-up (and deadlocks under -i 1); serialising it avoids
 * that. The synchronized periodic run only begins after every context is warm,
 * so there is a ~(NUM_TASKS+1) s one-time warm-up before the timed window.
 *
 * Usage:  workloadTasksetGcaps [-i 0|1] [-s 0|1] [-b 0|1] [-d DURATION_S] [-k N]
 *         (defaults: -i 0 -s 0 -b 0 -d 30, all GPU tasks)
 *         -k N : activate only the first N GPU tasks (CPU-only task always
 *                runs) — for bisecting how many concurrent GPU contexts the
 *                GCAPS elevation path tolerates before deadlocking.
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

static constexpr int NUM_TASKS = 7;

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

/* All GPU tasks use distinct SCHED_FIFO priorities (no best-effort tasks): the
 * two formerly best-effort tasks (mm_2048, hist_4M) are given the two LOWEST
 * real-time priorities instead of SCHED_OTHER. This (a) avoids the GCAPS driver
 * best-effort deadlock (>1 concurrently-running best-effort GPU task — see
 * best-effort-tasks-bug.md) and (b) matches the SequenceScheduler benchmark,
 * which priority-schedules every GPU task and has no best-effort class. */
static const BenchTaskDef TASKS[NUM_TASKS] = {
	{"hist_16M",      true,  SeqWlType::HISTOGRAM,   16u << 20, 0,  1, 100, {1, -1}, 7},
	{"mm_1024",       true,  SeqWlType::MATMUL,      1024,      0,  2, 150, {2, -1}, 6},
	{"cpu_only",      false, SeqWlType::MATMUL,      0,         0, 67, 200, {2, -1}, 5},
	{"conv_1024_k7",  true,  SeqWlType::CONVOLUTION, 1024,      7, 12, 300, {1, -1}, 4},
	{"conv_2048_k15", true,  SeqWlType::CONVOLUTION, 2048,     15,  2, 400, {1, -1}, 3},
	{"mm_2048",       true,  SeqWlType::MATMUL,      2048,      0,  4, 200, {4, -1}, 2},
	{"hist_4M",       true,  SeqWlType::HISTOGRAM,   4u << 20,  0,  4,  67, {4,  5}, 1},
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

// Per-task init is staggered by this much so the 7 CUDA contexts are created
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
	 * verify()) in its own slot so context bring-up is serialised across tasks
	 * rather than a simultaneous 7-context storm that hangs the driver. */
	sleep_until_abs_ns(g_init_start_ns + (uint64_t)task_idx * INIT_STAGGER_NS);

	SeqWorkload* wl = nullptr;
	if (td.is_gpu) {
		wl = new SeqWorkload(td.wlType, td.wlP1, td.wlP2, fd, sync_mode,
		                     ioctl_enabled, suspension);
		wl->taskInit();
		const bool ok = wl->verify();
		if (!ok) fprintf(stderr, "[task %s] WARNING: verify FAILED\n", td.name);
		wl->recordPriority(td.fifo_priority);
	}

	const uint64_t period_ns = (uint64_t)td.ti_ms * 1000000ULL;
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
			rec.missed   = (resp_ms > (double)td.ti_ms);
			rec.seg_begin_ns = seg_begin_ns;
			rec.seg_done_ns  = seg_done_ns;
			records.push_back(rec);
		}
		first_period = false;
		++period_idx;
		next_period_start += period_ns;
	}

	if (wl) { wl->taskFinish(); delete wl; }

	/* Each child writes its own trace fragment; parent merges them. */
	char path[160];
	snprintf(path, sizeof(path),
	         "results/workloadBench/.tsk_%s_%d.csv", mode_tag, task_idx);
	FILE* f = fopen(path, "w");
	if (!f) { fprintf(stderr, "[task %s] could not open %s\n", td.name, path);
	          return; }
	for (const Rec& r : records)
		fprintf(f, "%d,%s,%u,%.3f,%.3f,%.3f,%.3f,%.3f,%.0f,%d,%d,%llu,%llu\n",
		        task_idx, td.name, r.period_idx, r.period_start_ms,
		        r.cpu_ms, r.ovh_ms, r.gpu_ms, r.resp_ms,
		        (double)td.ti_ms, (int)r.missed,
		        my_pid,
		        (unsigned long long)r.seg_begin_ns,
		        (unsigned long long)r.seg_done_ns);
	fclose(f);
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
	            "deadline_ms,missed,pid,seg_begin_ns,seg_done_ns\n");

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
	while ((opt = getopt(argc, argv, "i:s:b:d:k:")) != EOF) {
		switch (opt) {
			case 'i': ioctl_enabled = atoi(optarg); break;
			case 's': suspension    = atoi(optarg); break;
			case 'b': sync_mode     = atoi(optarg); break;
			case 'd': duration_s    = strtoull(optarg, nullptr, 10); break;
			case 'k': gpu_limit     = atoi(optarg); break;
			default:  fprintf(stderr, "bad option\n"); return 1;
		}
	}
	if (sync_mode && ioctl_enabled) {
		fprintf(stderr, "IOCTL and sync mode are mutually exclusive\n");
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

	printf("Taskset: mode=%s, duration=%llus, staggered init "
	       "(%d x %.1f s) then experiment starts in %.1f s",
	       ioctl_enabled ? "GCAPS-ioctl" : "TSG-default",
	       (unsigned long long)duration_s, NUM_TASKS,
	       (double)INIT_STAGGER_NS / 1.0e9,
	       (double)(g_sync_start_ns - host_ns()) / 1.0e9);
	if (gpu_limit >= 0)
		printf("  [GPU tasks capped at %d]", gpu_limit);
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
