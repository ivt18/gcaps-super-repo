/*
 * ===========================================================================
 * VENDORED COPY — do not edit here.
 * ===========================================================================
 * Source: singleTaskSched/include/clock_calib.cuh (the thesis repo).
 *
 * cpuWakeupLatencyGcaps.cu reports W_i on the same CPU→GPU converted axis as
 * cpuWakeupLatencyBenchSeq / cpuWakeupLatencyStreamBaseline, and the three are
 * compared head to head.  The conversion has to be bit-for-bit the same
 * procedure — same sample count, same RTT-midpoint estimator, same median,
 * same two-point drift interpolation — or a W_i difference between the arms
 * could be the calibration rather than the wake path.  Hence a copy rather
 * than a reimplementation.
 *
 * Keep in sync with the original if it changes.
 * ===========================================================================
 */

/**
 * clock_calib.cuh — shared CPU↔GPU clock calibration with drift correction.
 *
 * The eval/bench binaries that report a metric spanning the CPU and GPU clocks
 * (response, scheduling overhead O_i, wakeup latency W_i, …) convert a CPU
 * CLOCK_MONOTONIC timestamp into the GPU %globaltimer domain via an offset:
 *
 *     gpu_ns ≈ cpu_ns + offset
 *
 * A single offset measured once at startup goes stale as the two oscillators
 * drift apart, biasing later samples by drift_rate × time-since-calibration.
 * This header provides a two-point calibration bracket (start + end of run) and
 * a linearly interpolated offset_at(t) so each sample is converted with the
 * offset that was valid at its own wall-clock time — removing the drift.
 *
 * Usage (bench-owned bracket; the scheduler's internal offset is not used):
 *     uint64_t* h_gpuTs; cudaMallocHost(&h_gpuTs, sizeof(uint64_t));
 *     cudaStream_t s; cudaStreamCreate(&s);
 *     ClockBracket br;
 *     clock_bracket_begin(&br, s, h_gpuTs);   // GPU idle, before the run
 *     ... run ...                              // record raw CPU/GPU timestamps
 *     clock_bracket_end(&br, s, h_gpuTs);      // GPU idle, after the run
 *     clock_bracket_report(&br);               // drift + 10 µs verdict
 *     // convert each sample: gpu = cpu_ns + br.offset_at(cpu_ns)
 *
 * A variant seeds the start bracket from an offset a scheduler already measured
 * at create() (getCpuToGpuOffset()) via clock_bracket_seed(), so only the end
 * bracket runs the stamp kernel; drift is still recovered from the two points.
 *
 * Both brackets must be measured with the GPU idle (before start() / after
 * stop()+sync for the persistent schedulers) so clock_bracket_*() can use
 * cudaStreamSynchronize without deadlocking on the persistent kernel.
 *
 * One translation unit per executable includes this header, so the plain
 * __global__ stamp kernel raises no ODR concern (mirrors workloads.cuh).
 */

#ifndef CLOCK_CALIB_CUH
#define CLOCK_CALIB_CUH

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

#ifndef CALIB_N_SAMPLES
#define CALIB_N_SAMPLES 64
#endif

/* Internal host monotonic clock — named to avoid colliding with callers'
 * own host_ns()/host_nanoseconds() helpers. */
static inline uint64_t clock_calib_host_ns()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static __device__ __forceinline__ uint64_t clock_calib_gpu_ns()
{
    uint64_t t;
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
    return t;
}

/* Writes %globaltimer to a pinned slot (calibration / completion stamp). */
__global__ void clockCalibStamp(uint64_t* out)
{
    if (threadIdx.x == 0) *out = clock_calib_gpu_ns();
}

/* One offset measurement: CALIB_N_SAMPLES RTT-midpoint samples, median offset,
 * sample std-dev.  gpu_ns ≈ cpu_ns + *offset_out.  Returns 0, or -1 on a sync
 * failure.  h_gpuTs is a caller-provided pinned uint64 scratch slot. */
static inline int clock_calibrate(cudaStream_t stream, uint64_t* h_gpuTs,
                                  int64_t* offset_out, double* std_out)
{
    int64_t offsets[CALIB_N_SAMPLES];
    for (int i = 0; i < CALIB_N_SAMPLES; ++i) {
        *h_gpuTs = 0;
        const uint64_t cpuBefore = clock_calib_host_ns();
        clockCalibStamp<<<1, 1, 0, stream>>>(h_gpuTs);
        if (cudaStreamSynchronize(stream) != cudaSuccess)
            return -1;
        const uint64_t cpuAfter = clock_calib_host_ns();
        const uint64_t rtt = cpuAfter - cpuBefore;
        offsets[i] = (int64_t)(*h_gpuTs) - (int64_t)(cpuBefore + rtt / 2);
    }

    int64_t sorted[CALIB_N_SAMPLES];
    for (int i = 0; i < CALIB_N_SAMPLES; ++i) sorted[i] = offsets[i];
    std::sort(sorted, sorted + CALIB_N_SAMPLES);
    *offset_out = sorted[CALIB_N_SAMPLES / 2];

    double mean = 0.0;
    for (int i = 0; i < CALIB_N_SAMPLES; ++i) mean += (double)offsets[i];
    mean /= CALIB_N_SAMPLES;
    double var = 0.0;
    for (int i = 0; i < CALIB_N_SAMPLES; ++i) {
        const double d = (double)offsets[i] - mean;
        var += d * d;
    }
    *std_out = std::sqrt(var / (CALIB_N_SAMPLES - 1));
    return 0;
}

/* A start/end calibration bracket for linear drift correction over a run. */
struct ClockBracket {
    int64_t  offset_start = 0, offset_end = 0;
    uint64_t t_start = 0, t_end = 0;     /* CLOCK_MONOTONIC ns at each bracket */
    double   std_start = 0.0, std_end = 0.0;
    bool     have_start = false, have_end = false;

    /* CPU→GPU offset at CPU time t (ns).  Linear interpolation between the two
     * brackets; falls back to the start offset (Tier 1) when the end bracket is
     * absent (e.g. it failed, or only one was taken). */
    int64_t offset_at(uint64_t t_cpu_ns) const
    {
        if (!have_end || t_end == t_start)
            return offset_start;
        const double frac =
            (double)((int64_t)t_cpu_ns - (int64_t)t_start) /
            (double)((int64_t)t_end - (int64_t)t_start);
        return offset_start +
               (int64_t)llround((double)(offset_end - offset_start) * frac);
    }
};

/* Measure the start bracket now (GPU must be idle).  Returns 0 or -1. */
static inline int clock_bracket_begin(ClockBracket* b, cudaStream_t stream,
                                      uint64_t* h_gpuTs)
{
    if (clock_calibrate(stream, h_gpuTs, &b->offset_start, &b->std_start) != 0)
        return -1;
    b->t_start = clock_calib_host_ns();
    b->have_start = true;
    return 0;
}

/* Seed the start bracket from an externally-measured offset (e.g. a scheduler's
 * getCpuToGpuOffset(), calibrated at create()) instead of running the stamp
 * kernel.  Anchors it at the current CPU time, so call as soon as possible after
 * that offset was measured to keep the anchor error well below the run's drift.
 * Pair with clock_bracket_end() to recover drift over the run. */
static inline void clock_bracket_seed(ClockBracket* b, int64_t offset,
                                      double std_ns)
{
    b->offset_start = offset;
    b->std_start    = std_ns;
    b->t_start      = clock_calib_host_ns();
    b->have_start   = true;
}

/* Measure the end bracket now (GPU must be idle).  Returns 0 or -1. */
static inline int clock_bracket_end(ClockBracket* b, cudaStream_t stream,
                                    uint64_t* h_gpuTs)
{
    if (clock_calibrate(stream, h_gpuTs, &b->offset_end, &b->std_end) != 0)
        return -1;
    b->t_end = clock_calib_host_ns();
    b->have_end = true;
    return 0;
}

/* Print the offset, measured drift, and worst-case offset error vs a budget
 * (default 10 µs).  Goes to stdout as '#'-prefixed lines so CSV consumers that
 * use comment='#' ignore them. */
static inline void clock_bracket_report(const ClockBracket* b,
                                        double budget_us = 10.0)
{
    if (!b->have_start) {
        printf("# clock bracket: no calibration\n");
        return;
    }
    printf("# cpu_to_gpu_offset_ns: %lld\n", (long long)b->offset_start);
    printf("# calibration_std_ns: %.1f\n", b->std_start);
    if (!b->have_end) {
        printf("# clock drift: no end bracket (using start offset)\n");
        return;
    }
    const int64_t  drift = b->offset_end - b->offset_start;
    const uint64_t span  = b->t_end - b->t_start;
    const double   ppm   = span ? (double)drift / (double)span * 1.0e6 : 0.0;
    printf("# clock_drift_ns: %lld  over %.1f s  (%.3f ppm)\n",
           (long long)drift, (double)span / 1.0e9, ppm);
    printf("# clock_drift_worstcase_ns: %lld  [%s %.0f us]\n",
           (long long)llabs(drift),
           (llabs(drift) <= (int64_t)(budget_us * 1000.0)) ? "within" : "EXCEEDS",
           budget_us);
}

#endif /* CLOCK_CALIB_CUH */
