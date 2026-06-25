/*
 * preemptOverheadGcaps.cc
 *
 * Controlled victim + preemptor microbenchmark for measuring GCAPS preemption
 * cost. It deliberately creates the simplest scenario that still exercises the
 * GCAPS preemption path: a few tasks registered at *different* SCHED_FIFO
 * priorities, where a higher-priority job's GPU segment arrives while a
 * lower-priority job's GPU segment is already running on the device:
 *
 *   role       workload          prio  period    CPU    notes
 *   ---------  ----------------  ----  --------   -----  -----------------------
 *   victim     matmul Nv         2     Pv (long)  {1}    long GPU segment, the
 *                                                        thing that gets preempted
 *   preemptorA matmul Np         5     Pa (short) {2}    highest prio
 *   preemptorB histogram 8M      4     Pb (short) {3}    mid prio
 *
 * Three distinct priority levels (2 < 4 < 5): the victim is preempted by both
 * preemptors, and preemptorB is itself preempted by preemptorA — so the trace
 * contains preemptions at more than one priority level, like a real taskset,
 * but with a structure simple enough to attribute a clean per-preemption number.
 *
 * What this binary measures (two things, paired with the driver GCAPS_EV log
 * and measure_preempt_overhead.py):
 *
 *   1. Scheduling + context-switch / preemption overhead. Comes from the driver:
 *      each runlist-update ioctl logs `GCAPS_EV ... elapsed_us=<eps> preempted=<pid>`.
 *      The script keeps the events whose `preempted` field names a real victim
 *      (i.e. an actual preemption happened) and reports their elapsed_us — the
 *      cost of performing the preemption (runlist reload + GPU-side switch).
 *
 *   2. Execution-time extension of a preempted job. The cudaEvent window
 *      (ev_start..ev_stop) is GPU *wall* time = active execution + time the
 *      segment sat suspended while a higher-priority job ran. The driver's
 *      preempted/resumed events give each victim segment's suspended intervals;
 *      the script subtracts them to get *active* execution time, then compares a
 *      task's active execution on preempted vs non-preempted releases. The
 *      difference is how much the job's own (actively-executing) time grows when
 *      it is preempted — i.e. the save/restore + cold-cache cost charged to its
 *      execution, NOT the blocking time, which is excluded by construction.
 *
 * Per release we emit: role, name, pid, period_idx, absolute seg_begin_ns /
 * seg_done_ns (CLOCK_MONOTONIC, same base as the driver's GCAPS_EV ts=), the
 * cudaEvent GPU wall time, and the host response time.
 *
 * Like workloadTasksetGcaps, every task is a separate forked process (GCAPS's
 * ioctl acts on a pid) and CUDA contexts are brought up one at a time (staggered
 * init) to avoid the concurrent-context-init deadlock under -i 1. A baseline
 * window (PREEMPTOR_ACTIVATION_S) at the start runs the victim with the
 * preemptors idle, guaranteeing clean non-preempted samples for the comparison.
 *
 * Usage: preemptOverheadGcaps [-i 0|1] [-s 0|1] [-b 0|1] [-d DURATION_S]
 *                             [-N victim_matmul] [-P preemptor_matmul]
 *        (defaults: -i 0 -s 0 -b 0 -d 30 -N 4096 -P 1024)
 *        Run with sudo for SCHED_FIFO; needs sched_rt_runtime_us=-1 and -s 1
 *        (blocking sync) for the same reasons as workloadTasksetGcaps — see the
 *        userspace readme.
 * Output (-i 1 -> gcaps, else tsg):
 *   results/workloadBench/preempt_{gcaps,tsg}_trace.csv
 */

#include <fcntl.h>
#include <unistd.h>
#include <getopt.h>
#include <sched.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "app/seqworkload/seqworkload.h"

// ============================================================================
// Task definitions
// ============================================================================

static constexpr int NUM_TASKS = 3;

struct PreemptTaskDef {
	const char*  role;
	const char*  name;
	SeqWlType    wlType;
	unsigned int wlP1, wlP2;
	uint32_t     ti_ms;
	int          cpu_core;
	int          fifo_priority;
	bool         is_preemptor;   /* idle during the baseline window */
};

/* p1 of the two matmul tasks is overridden by -N / -P at runtime. */
static PreemptTaskDef TASKS[NUM_TASKS] = {
	{"victim",     "mm_victim",  SeqWlType::MATMUL,    4096,     0,  80, 1, 2, false},
	{"preemptorA", "mm_preA",    SeqWlType::MATMUL,    1024,     0,  17, 2, 5, true },
	{"preemptorB", "hist_preB",  SeqWlType::HISTOGRAM, 8u << 20, 0,  29, 3, 4, true },
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

static void sleep_until_abs_ns(uint64_t target_ns)
{
	struct timespec ts;
	ts.tv_sec  = (time_t)(target_ns / 1000000000ULL);
	ts.tv_nsec = (long)(target_ns % 1000000000ULL);
	while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, nullptr) != 0) {}
}

// Inherited by every forked child (set before fork()).
static uint64_t g_init_start_ns = 0;   // epoch for the staggered per-task init
static uint64_t g_sync_start_ns = 0;   // synchronized periodic-loop start
static uint64_t g_experiment_ns = 0;

static constexpr uint64_t INIT_STAGGER_NS = 1000000000ULL; /* 1 s per task */
/* Baseline window after the synchronized start during which the preemptors stay
 * idle, so the victim gets uncontended releases to anchor its active-execution
 * baseline against. */
static constexpr uint64_t PREEMPTOR_ACTIVATION_NS = 3000000000ULL; /* 3 s */

// ============================================================================
// Child process: run one task for the experiment window, dump its trace rows.
// ============================================================================

static void run_task(int task_idx, int fd, bool sync_mode, bool ioctl_enabled,
                     bool suspension, const char* mode_tag)
{
	const PreemptTaskDef& td = TASKS[task_idx];
	const int my_pid = getpid();

	cpu_set_t cpuset;
	CPU_ZERO(&cpuset);
	CPU_SET(td.cpu_core, &cpuset);
	if (sched_setaffinity(gettid(), sizeof(cpuset), &cpuset) != 0)
		fprintf(stderr, "[%s] WARNING: sched_setaffinity failed\n", td.name);

	if (td.fifo_priority > 0) {
		struct sched_param sp;
		sp.sched_priority = td.fifo_priority;
		if (sched_setscheduler(0, SCHED_FIFO, &sp) != 0)
			fprintf(stderr, "[%s] WARNING: SCHED_FIFO pri=%d failed — "
			                "continuing SCHED_OTHER\n", td.name, td.fifo_priority);
	}

	/* Staggered start-up: serialise CUDA context bring-up across tasks. */
	sleep_until_abs_ns(g_init_start_ns + (uint64_t)task_idx * INIT_STAGGER_NS);

	SeqWorkload wl(td.wlType, td.wlP1, td.wlP2, fd, sync_mode, ioctl_enabled,
	               suspension);
	wl.taskInit();
	if (!wl.verify())
		fprintf(stderr, "[%s] WARNING: verify FAILED\n", td.name);
	wl.recordPriority(td.fifo_priority);

	const uint64_t period_ns = (uint64_t)td.ti_ms * 1000000ULL;
	const uint64_t end_ns    = g_sync_start_ns + g_experiment_ns;
	/* Preemptors only start firing after the baseline window. */
	const uint64_t task_start_ns =
		g_sync_start_ns + (td.is_preemptor ? PREEMPTOR_ACTIVATION_NS : 0);

	struct Rec {
		uint32_t period_idx;
		uint64_t seg_begin_ns, seg_done_ns;
		double   gpu_ms, resp_ms;
	};
	std::vector<Rec> records;
	records.reserve(g_experiment_ns / period_ns + 4);

	bool     first_period      = true;
	uint32_t period_idx        = 0;
	uint64_t next_period_start  = task_start_ns;

	while (host_ns() < end_ns) {
		const uint64_t period_start_ns = next_period_start;
		sleep_until_abs_ns(period_start_ns);
		if (host_ns() >= end_ns) break;

		const uint64_t seg_begin_ns = host_ns();
		wl.taskCallback(0, 0);              /* one GCAPS GPU segment */
		const uint64_t seg_done_ns  = host_ns();

		if (!first_period) {               /* drop the first (cold) release */
			Rec rec;
			rec.period_idx   = period_idx;
			rec.seg_begin_ns = seg_begin_ns;
			rec.seg_done_ns  = seg_done_ns;
			rec.gpu_ms       = (double)wl.lastGpuMs();
			rec.resp_ms      = (double)(seg_done_ns - period_start_ns) / 1.0e6;
			records.push_back(rec);
		}
		first_period = false;
		++period_idx;
		next_period_start += period_ns;
	}

	wl.taskFinish();

	/* Each child writes its own trace fragment; parent merges them. */
	char path[160];
	snprintf(path, sizeof(path),
	         "results/workloadBench/.pmt_%s_%d.csv", mode_tag, task_idx);
	FILE* f = fopen(path, "w");
	if (!f) { fprintf(stderr, "[%s] could not open %s\n", td.name, path); return; }
	for (const Rec& r : records)
		fprintf(f, "%d,%s,%s,%d,%u,%llu,%llu,%.4f,%.4f\n",
		        task_idx, td.role, td.name, my_pid, r.period_idx,
		        (unsigned long long)r.seg_begin_ns,
		        (unsigned long long)r.seg_done_ns,
		        r.gpu_ms, r.resp_ms);
	fclose(f);
}

// ============================================================================
// main
// ============================================================================

int main(int argc, char** argv)
{
	setvbuf(stdout, nullptr, _IONBF, 0);

	int ioctl_enabled = 0, suspension = 0, sync_mode = 0;
	uint64_t duration_s = 30;
	unsigned int victim_n = 4096, preemptor_n = 1024;
	int opt;
	while ((opt = getopt(argc, argv, "i:s:b:d:N:P:")) != EOF) {
		switch (opt) {
			case 'i': ioctl_enabled = atoi(optarg); break;
			case 's': suspension    = atoi(optarg); break;
			case 'b': sync_mode     = atoi(optarg); break;
			case 'd': duration_s    = strtoull(optarg, nullptr, 10); break;
			case 'N': victim_n      = (unsigned int)atoi(optarg); break;
			case 'P': preemptor_n   = (unsigned int)atoi(optarg); break;
			default:  fprintf(stderr, "bad option\n"); return 1;
		}
	}
	if (sync_mode && ioctl_enabled) {
		fprintf(stderr, "IOCTL and sync mode are mutually exclusive\n");
		return 1;
	}
	TASKS[0].wlP1 = victim_n;      /* victim matmul size */
	TASKS[1].wlP1 = preemptor_n;   /* preemptorA matmul size */
	const char* mode_tag = ioctl_enabled ? "gcaps" : "tsg";

	int fd = open("/dev/nvgpu/igpu0/ctrl", O_RDWR);
	if (ioctl_enabled && fd < 0) { perror("open /dev/nvgpu/igpu0/ctrl");
	                               return 1; }

	mkdir("results", 0755);
	mkdir("results/workloadBench", 0755);

	static constexpr uint64_t PREINIT_NS = 500000000ULL;          /* 500 ms */
	static constexpr uint64_t POSTINIT_MARGIN_NS = 1000000000ULL; /* 1 s */
	g_experiment_ns = duration_s * 1000000000ULL;
	g_init_start_ns = host_ns() + PREINIT_NS;
	g_sync_start_ns = g_init_start_ns
	                + (uint64_t)NUM_TASKS * INIT_STAGGER_NS
	                + POSTINIT_MARGIN_NS;

	printf("Preempt microbench: mode=%s, duration=%llus, victim_mm=%u, "
	       "preemptor_mm=%u, baseline window=%.1fs, experiment starts in %.1fs\n",
	       ioctl_enabled ? "GCAPS-ioctl" : "TSG-default",
	       (unsigned long long)duration_s, victim_n, preemptor_n,
	       (double)PREEMPTOR_ACTIVATION_NS / 1.0e9,
	       (double)(g_sync_start_ns - host_ns()) / 1.0e9);

	/* Remove stale per-task trace fragments. */
	for (int i = 0; i < NUM_TASKS; ++i) {
		char path[160];
		snprintf(path, sizeof(path),
		         "results/workloadBench/.pmt_%s_%d.csv", mode_tag, i);
		remove(path);
	}

	std::vector<pid_t> children;
	for (int i = 0; i < NUM_TASKS; ++i) {
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
			for (pid_t c : children) kill(c, SIGTERM);
			return 1;
		}
	}
	for (pid_t c : children) { int st; waitpid(c, &st, 0); }
	if (fd >= 0) close(fd);

	/* Merge per-task fragments into one trace CSV. */
	char tracePath[128];
	snprintf(tracePath, sizeof(tracePath),
	         "results/workloadBench/preempt_%s_trace.csv", mode_tag);
	FILE* tr = fopen(tracePath, "w");
	if (!tr) { fprintf(stderr, "could not open %s\n", tracePath); return 1; }
	fprintf(tr, "task_id,role,task_name,pid,period_idx,seg_begin_ns,"
	            "seg_done_ns,gpu_exec_ms,response_ms\n");
	for (int i = 0; i < NUM_TASKS; ++i) {
		char path[160];
		snprintf(path, sizeof(path),
		         "results/workloadBench/.pmt_%s_%d.csv", mode_tag, i);
		FILE* f = fopen(path, "r");
		if (!f) continue;
		char line[256];
		while (fgets(line, sizeof(line), f)) fputs(line, tr);
		fclose(f);
		remove(path);
	}
	fclose(tr);
	printf("Trace written to %s\n", tracePath);
	return 0;
}
