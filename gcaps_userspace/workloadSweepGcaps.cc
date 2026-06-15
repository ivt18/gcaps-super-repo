/*
 * workloadSweepGcaps.cc
 *
 * Per-workload size sweep on the GCAPS userspace harness — the GCAPS analog of
 * singleTaskSched's bench/workloadSweepBench.cu.
 *
 * For each workload configuration (matmul 256..2048, histogram 1M..64M,
 * convolution 512^2..2048^2 x kernel width 3/7/15 — identical to the source
 * sweep) the workload runs as one GCAPS GPU segment, released N times
 * back-to-back in isolation.  Per release we record:
 *
 *   gpu_exec_ms               = cudaEvent(segment start -> stop) (on-GPU time)
 *   response_ms               = host wall time release -> segment done
 *   sched_preempt_overhead_ms = response_ms - gpu_exec_ms
 *       — everything outside the measured on-GPU window: kernel-launch and
 *         ioctl latency plus GPU-side scheduling/preemption delay (GCAPS is
 *         preemptive, so this is scheduling + preemption overhead, not a FIFO
 *         queue wait).
 *
 * The GCAPS segment is bracketed by gcapsGpuSegBegin/End; pass -i 1 for the
 * GCAPS ioctl mode, -i 0 for the default TSG round-robin baseline.  Output
 * file is chosen by mode so the two runs feed the plotter as two series:
 *   -i 1  ->  results/workloadBench/sweep_gcaps.csv
 *   -i 0  ->  results/workloadBench/sweep_tsg.csv
 *
 * Usage:  workloadSweepGcaps [-i 0|1] [-s 0|1] [-b 0|1] [-n N] [-w WARMUP]
 *         (defaults: -i 0 -s 0 -b 0 -n 100 -w 10)
 */

#include <fcntl.h>
#include <unistd.h>
#include <getopt.h>
#include <sys/stat.h>
#include <time.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>

#include "app/seqworkload/seqworkload.h"

struct SweepConfig {
	SeqWlType    type;
	unsigned int p1, p2;
};

struct ReleaseSample {
	double ovh_ms;        /* scheduling + preemption overhead */
	double exec_ms;
	double response_ms;
};

static uint64_t host_ns()
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static double p95(std::vector<double> v)
{
	std::sort(v.begin(), v.end());
	if (v.empty()) return 0.0;
	size_t idx = (size_t)(0.95 * (double)v.size());
	if (idx >= v.size()) idx = v.size() - 1;
	return v[idx];
}

/* Identical configuration set to buildSweepWorkloads() in workloads.cuh. */
static std::vector<SweepConfig> buildSweepConfigs()
{
	std::vector<SweepConfig> c;
	for (int sz : {256, 512, 1024, 2048})
		c.push_back({SeqWlType::MATMUL, (unsigned)sz, 0});
	for (unsigned sz : {1u << 20, 4u << 20, 16u << 20, 64u << 20})
		c.push_back({SeqWlType::HISTOGRAM, sz, 0});
	for (int img : {512, 1024, 2048})
		for (int kw : {3, 7, 15})
			c.push_back({SeqWlType::CONVOLUTION, (unsigned)img, (unsigned)kw});
	return c;
}

int main(int argc, char** argv)
{
	setvbuf(stdout, nullptr, _IONBF, 0);

	int  ioctl_enabled = 0, suspension = 0, sync_mode = 0;
	int  nReleases = 100, nWarmup = 10;
	int  opt;
	while ((opt = getopt(argc, argv, "i:s:b:n:w:")) != EOF) {
		switch (opt) {
			case 'i': ioctl_enabled = atoi(optarg); break;
			case 's': suspension    = atoi(optarg); break;
			case 'b': sync_mode     = atoi(optarg); break;
			case 'n': nReleases     = atoi(optarg); break;
			case 'w': nWarmup       = atoi(optarg); break;
			default:  fprintf(stderr, "bad option\n"); return 1;
		}
	}
	if (sync_mode && ioctl_enabled) {
		fprintf(stderr, "IOCTL and sync mode are mutually exclusive\n");
		return 1;
	}

	int fd = -1;
	if (ioctl_enabled) {
		fd = open("/dev/nvgpu/igpu0/ctrl", O_RDWR);
		if (fd < 0) { perror("open /dev/nvgpu/igpu0/ctrl"); return 1; }
	}

	const std::vector<SweepConfig> cfgs = buildSweepConfigs();
	printf("Sweep: %zu configurations, mode=%s, releases=%d warmup=%d\n",
	       cfgs.size(), ioctl_enabled ? "GCAPS-ioctl" : "TSG-default",
	       nReleases, nWarmup);

	std::vector<std::string>               names(cfgs.size());
	std::vector<const char*>               types(cfgs.size());
	std::vector<std::vector<ReleaseSample>> samples(cfgs.size());

	for (size_t i = 0; i < cfgs.size(); ++i) {
		SeqWorkload wl(cfgs[i].type, cfgs[i].p1, cfgs[i].p2,
		               fd, (bool)sync_mode, (bool)ioctl_enabled,
		               (bool)suspension);
		wl.taskInit();
		names[i] = wl.name();
		types[i] = seqWlTypeName(cfgs[i].type);

		const bool ok = wl.verify();
		printf("  verify %-16s %s\n", wl.name(), ok ? "PASS" : "FAIL");
		if (!ok) { fprintf(stderr, "verification failed — aborting\n");
		           wl.taskFinish(); return 1; }

		samples[i].reserve(nReleases);
		for (int r = 0; r < nWarmup + nReleases; ++r) {
			const uint64_t t0 = host_ns();
			wl.taskCallback(0, 0);           /* one GCAPS GPU segment */
			const uint64_t t1 = host_ns();
			if (r < nWarmup) continue;

			ReleaseSample s{};
			s.response_ms = (double)(t1 - t0) / 1.0e6;
			s.exec_ms     = (double)wl.lastGpuMs();
			s.ovh_ms      = s.response_ms - s.exec_ms;
			if (s.ovh_ms < 0.0) s.ovh_ms = 0.0;
			samples[i].push_back(s);
		}
		printf("  swept %-16s (%zu samples)\n", wl.name(), samples[i].size());
		wl.taskFinish();
	}

	if (fd >= 0) close(fd);

	/* ---- CSV ---- */
	mkdir("results", 0755);
	mkdir("results/workloadBench", 0755);
	const char* csvPath = ioctl_enabled
		? "results/workloadBench/sweep_gcaps.csv"
		: "results/workloadBench/sweep_tsg.csv";
	FILE* csv = fopen(csvPath, "w");
	if (!csv) { fprintf(stderr, "could not open %s\n", csvPath); return 1; }
	fprintf(csv, "workload,type,release_idx,sched_preempt_overhead_ms,"
	             "gpu_exec_ms,response_ms\n");
	for (size_t i = 0; i < cfgs.size(); ++i)
		for (size_t r = 0; r < samples[i].size(); ++r)
			fprintf(csv, "%s,%s,%zu,%.6f,%.6f,%.6f\n",
			        names[i].c_str(), types[i], r,
			        samples[i][r].ovh_ms, samples[i][r].exec_ms,
			        samples[i][r].response_ms);
	fclose(csv);
	printf("\nCSV written to %s\n", csvPath);

	/* ---- Summary table ---- */
	printf("\n%-16s  %10s  %10s  %10s  %10s  %10s\n",
	       "Workload", "Mean(ms)", "Min(ms)", "Max(ms)", "P95(ms)",
	       "Ovh(ms)");
	for (size_t i = 0; i < cfgs.size(); ++i) {
		if (samples[i].empty()) continue;
		std::vector<double> resp, ovh;
		for (const ReleaseSample& s : samples[i]) {
			resp.push_back(s.response_ms);
			ovh.push_back(s.ovh_ms);
		}
		double sum = 0, osum = 0;
		for (double v : resp) sum  += v;
		for (double v : ovh)  osum += v;
		printf("%-16s  %10.3f  %10.3f  %10.3f  %10.3f  %10.4f\n",
		       names[i].c_str(),
		       sum / (double)resp.size(),
		       *std::min_element(resp.begin(), resp.end()),
		       *std::max_element(resp.begin(), resp.end()),
		       p95(resp),
		       osum / (double)ovh.size());
	}
	return 0;
}
