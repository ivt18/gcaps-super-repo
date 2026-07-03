#ifndef __SEQWORKLOAD_H__
#define __SEQWORKLOAD_H__

#include <cstdint>
#include <cuda.h>
#include "../base_task.h"

/*
 * SeqWorkload — the matmul / histogram / convolution / MLP workloads from the
 * singleTaskSched SequenceScheduler benchmark suite (bench/workloads.cuh),
 * ported into the GCAPS userspace harness.
 *
 * The kernels are identical in shape to the source; the SequenceScheduler /
 * inter-segment-buffer plumbing is dropped (every kernel takes plain device
 * pointers, exactly the source's stream-baseline path).  The whole workload
 * runs as a single GCAPS GPU segment, bracketed by gcapsGpuSegBegin/End so the
 * ioctl runlist-priority elevation applies to it just like the native GCAPS
 * apps (MatrixMulGPU, Hist, ...).
 *
 *   MATMUL       compute-intensive: tiled square matmul        (1 kernel)
 *   HISTOGRAM    memory-intensive : 256-bin histogram          (2 kernels)
 *   CONVOLUTION  hybrid           : separable 2D box convolution(2 kernels)
 *   MLP          DNN-style        : square fully-connected net (2*L kernels:
 *                matmul + bias/ReLU per layer, all in one GCAPS segment)
 */

enum class SeqWlType { MATMUL, HISTOGRAM, CONVOLUTION, MLP };

const char* seqWlTypeName(SeqWlType t);

class SeqWorkload : public BaseTask {
public:
	/* p1 = matmul n / histogram n_elements / convolution image side (w=h) /
	 *      mlp layer width;
	 * p2 = convolution kernel width (odd) / mlp number of layers,
	 *      ignored otherwise. */
	SeqWorkload(SeqWlType type, unsigned int p1, unsigned int p2,
	            int fd, bool sync_mode, bool ioctl_enabled, bool suspension);
	~SeqWorkload();

	void taskInit() override;
	void taskCallback(int insId, int nIter) override;
	void taskFinish() override;
	void recordPriority(int priority) override;

	/* Run the workload once on the stream (no ioctl) and check correctness. */
	bool verify();

	const char* name() const { return name_; }
	/* GPU-segment time (ms) measured by cudaEvents on the last taskCallback. */
	float lastGpuMs() const { return last_gpu_ms; }

private:
	void launchKernels();   /* enqueue this workload's kernels on `stream` */

	SeqWlType    type;
	unsigned int p1, p2;
	char         name_[48];

	/* macro-referenced members (names fixed by gcapsGpuSegBegin/End) */
	cudaStream_t stream;
	cudaEvent_t  start, stop;
	int          fd;
	bool         sync_mode;
	bool         ioctl_enabled;
	bool         suspend_ = false;   /* blocking-sync (vs busy spin) for waits */
	int          event_flags = 0;

	/* timing events (always timing-enabled) bracketing the kernels */
	cudaEvent_t  ev_start, ev_stop;
	float        last_gpu_ms = 0.0f;

	int          prio = 0;
	CUcontext    ctx;
	CUdevice     device;

	/* matmul */
	int    mmN = 0;
	float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;

	/* histogram */
	uint32_t  histN = 0;
	uint32_t *d_histIn = nullptr, *d_bins = nullptr, *d_partials = nullptr;

	/* convolution */
	int    convW = 0, convH = 0, convKr = 0;
	float *d_convIn = nullptr, *d_convOut = nullptr, *d_coef = nullptr,
	      *d_tmp = nullptr;

	/* mlp (square layers: batch = in = out = mlpW) */
	int    mlpW = 0, mlpL = 0;          /* layer width; number of layers */
	float *d_mlpInput = nullptr;        /* W*W input activations */
	float *d_mlpWeights = nullptr;      /* L*W*W weight matrices (layer l at +l*W*W) */
	float *d_mlpBias = nullptr;         /* L*W biases (layer l at +l*W) */
	float *d_mlpAct0 = nullptr,         /* W*W ping-pong activation slabs */
	      *d_mlpAct1 = nullptr;

	/* Input activations consumed by layer l / output slab of the last layer. */
	float* mlpLayerInput(int l) const;
	float* mlpOutputSlab() const;
};

#endif // __SEQWORKLOAD_H__
