#include <unistd.h>
#include <stdio.h>
#include <assert.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sched.h>
#include <cmath>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>

#include <linux/nvgpu.h>
#include "seqworkload.h"
#include <common/include/support.h>
#include <helper_cuda.h>
#include <helper_functions.h>

/* GCAPS_CUDA13_COMPAT: on CUDA 13 cuCtxCreate resolves to cuCtxCreate_v4, which takes a
 * CUctxCreateParams* as its second argument. */
#if defined(CUDA_VERSION) && CUDA_VERSION >= 13000
#define cuCtxCreateCompat(pctx, flags, dev) cuCtxCreate((pctx), NULL, (flags), (dev))
#else
#define cuCtxCreateCompat(pctx, flags, dev) cuCtxCreate((pctx), (flags), (dev))
#endif

// ============================================================================
// Ported kernels (singleTaskSched bench/workloads.cuh, SequenceScheduler
// plumbing removed — plain device pointers, identical math/launch geometry).
// ============================================================================

__device__ inline uint32_t seq_wang_hash(uint32_t s)
{
	s = (s ^ 61u) ^ (s >> 16);
	s *= 9u;
	s = s ^ (s >> 4);
	s *= 0x27d4eb2du;
	s = s ^ (s >> 15);
	return s;
}

__global__ void seq_fill_uint_kernel(uint32_t* p, uint32_t n, uint32_t seed)
{
	for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
	     i += gridDim.x * blockDim.x)
		p[i] = seq_wang_hash(i ^ seed);
}

__global__ void seq_fill_float_kernel(float* p, uint32_t n, uint32_t seed)
{
	for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
	     i += gridDim.x * blockDim.x)
		p[i] = (float)(seq_wang_hash(i ^ seed) & 0xFFFFu) / 65536.0f;
}

/* Signed fill in [-scale, scale).  Used for MLP weights (scale = sqrt(3/W),
 * giving unit-variance pre-activations) and biases (small scale). */
__global__ void seq_fill_signed_kernel(float* p, uint32_t n, uint32_t seed,
                                       float scale)
{
	for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
	     i += gridDim.x * blockDim.x) {
		const float f = (float)(seq_wang_hash(i ^ seed) & 0xFFFFu) / 65536.0f;
		p[i] = scale * (2.0f * f - 1.0f);
	}
}

// ---- MATMUL — compute-intensive, 1 kernel ----------------------------------
#define MM_TILE 16

/** C = A * B, square n x n, n must be a multiple of MM_TILE. */
__global__ void seq_matmul_kernel(const float* A, const float* B, float* C,
                                  int n)
{
	__shared__ float As[MM_TILE][MM_TILE];
	__shared__ float Bs[MM_TILE][MM_TILE];

	const int tx  = threadIdx.x, ty = threadIdx.y;
	const int row = blockIdx.y * MM_TILE + ty;
	const int col = blockIdx.x * MM_TILE + tx;

	float acc = 0.0f;
	for (int t = 0; t < n; t += MM_TILE) {
		As[ty][tx] = A[row * n + t + tx];
		Bs[ty][tx] = B[(t + ty) * n + col];
		__syncthreads();
		#pragma unroll
		for (int k = 0; k < MM_TILE; ++k)
			acc += As[ty][k] * Bs[k][tx];
		__syncthreads();
	}
	C[row * n + col] = acc;
}

// ---- HISTOGRAM — memory-intensive, 2 kernels -------------------------------
#define HIST_BINS    256
#define HIST_BLOCKS  256
#define HIST_THREADS 256   /* must equal HIST_BINS */

__global__ void seq_hist_partial_kernel(const uint32_t* in, uint32_t n,
                                        uint32_t* partials)
{
	__shared__ uint32_t s_bins[HIST_BINS];
	s_bins[threadIdx.x] = 0;
	__syncthreads();

	for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
	     i += gridDim.x * blockDim.x)
		atomicAdd(&s_bins[in[i] & (HIST_BINS - 1)], 1u);
	__syncthreads();

	partials[blockIdx.x * HIST_BINS + threadIdx.x] = s_bins[threadIdx.x];
}

__global__ void seq_hist_reduce_kernel(const uint32_t* partials, uint32_t* out)
{
	uint32_t sum = 0;
	for (int b = 0; b < HIST_BLOCKS; ++b)
		sum += partials[b * HIST_BINS + threadIdx.x];
	out[threadIdx.x] = sum;
}

// ---- CONVOLUTION — hybrid, 2 kernels (separable box, clamped borders) ------
#define CONV_BLOCK 16

__global__ void seq_conv_row_kernel(const float* in, const float* coef,
                                    float* tmp, int w, int h, int kr)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= w || y >= h) return;

	float acc = 0.0f;
	for (int k = -kr; k <= kr; ++k) {
		const int xx = min(max(x + k, 0), w - 1);
		acc += coef[k + kr] * in[y * w + xx];
	}
	tmp[y * w + x] = acc;
}

__global__ void seq_conv_col_kernel(const float* tmp, const float* coef,
                                    float* out, int w, int h, int kr)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= w || y >= h) return;

	float acc = 0.0f;
	for (int k = -kr; k <= kr; ++k) {
		const int yy = min(max(y + k, 0), h - 1);
		acc += coef[k + kr] * tmp[yy * w + x];
	}
	out[y * w + x] = acc;
}

// ---- MLP — DNN-style, 2 kernels per layer (matmul -> bias+ReLU) ------------
// Square fully-connected network: batch = in_features = out_features = W, so
// each layer's GEMM is W x W and reuses seq_matmul_kernel verbatim.  L layers
// of [Z = A * Wt] -> [A' = ReLU(Z + bias)] give 2*L kernels, all launched
// inside ONE GCAPS GPU segment (the source runs them as 2*L SequenceScheduler
// segments).  Activations ping-pong between two scratch slabs.

/* Fused bias add + ReLU, in place over an n x n activation matrix.
 * bias is indexed by output feature (column) = i % n. */
__global__ void seq_mlp_relu_bias_kernel(float* x, const float* bias, int n)
{
	const uint32_t total = (uint32_t)n * (uint32_t)n;
	for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total;
	     i += gridDim.x * blockDim.x) {
		const float v = x[i] + bias[i % (uint32_t)n];
		x[i] = v > 0.0f ? v : 0.0f;
	}
}

// ============================================================================
// Host helpers
// ============================================================================

const char* seqWlTypeName(SeqWlType t)
{
	switch (t) {
		case SeqWlType::MATMUL:      return "matmul";
		case SeqWlType::HISTOGRAM:   return "histogram";
		case SeqWlType::CONVOLUTION: return "convolution";
		case SeqWlType::MLP:         return "mlp";
	}
	return "?";
}

static dim3 matmulGrid(int n)
{
	return dim3((unsigned)(n / MM_TILE), (unsigned)(n / MM_TILE), 1);
}

static dim3 convGrid(int w, int h)
{
	return dim3((unsigned)((w + CONV_BLOCK - 1) / CONV_BLOCK),
	            (unsigned)((h + CONV_BLOCK - 1) / CONV_BLOCK), 1);
}

// ============================================================================
// SeqWorkload
// ============================================================================

SeqWorkload::SeqWorkload(SeqWlType type_, unsigned int p1_, unsigned int p2_,
                         int fd_, bool sync_mode_, bool ioctl_enabled_,
                         bool suspension_)
{
	type = type_;
	p1   = p1_;
	p2   = p2_;
	fd   = fd_;
	sync_mode     = sync_mode_;
	ioctl_enabled = ioctl_enabled_;
	suspend_      = suspension_;

	if (suspension_)
		event_flags |= cudaEventBlockingSync;
	else
		event_flags = cudaEventDisableTiming;

	switch (type) {
	case SeqWlType::MATMUL:
		mmN = (int)p1;
		snprintf(name_, sizeof(name_), "matmul_%d", mmN);
		break;
	case SeqWlType::HISTOGRAM:
		histN = p1;
		if (p1 >= (1u << 20) && (p1 % (1u << 20)) == 0)
			snprintf(name_, sizeof(name_), "hist_%uM", p1 >> 20);
		else
			snprintf(name_, sizeof(name_), "hist_%u", p1);
		break;
	case SeqWlType::CONVOLUTION:
		convW = convH = (int)p1;
		convKr = (int)p2 / 2;
		snprintf(name_, sizeof(name_), "conv_%d_k%d", (int)p1, (int)p2);
		break;
	case SeqWlType::MLP:
		mlpW = (int)p1;
		mlpL = (int)p2;
		snprintf(name_, sizeof(name_), "mlp_%dx%d", mlpW, mlpL);
		break;
	}
}

/* Output of layer l lives in slab (l & 1); the final output is layer L-1. */
float* SeqWorkload::mlpOutputSlab() const
{
	return ((mlpL - 1) & 1) ? d_mlpAct1 : d_mlpAct0;
}

/* Input activations consumed by layer l: the network input for l==0, else the
 * previous layer's output slab. */
float* SeqWorkload::mlpLayerInput(int l) const
{
	if (l == 0) return d_mlpInput;
	return ((l - 1) & 1) ? d_mlpAct1 : d_mlpAct0;
}

SeqWorkload::~SeqWorkload() {}

void SeqWorkload::taskInit()
{
	cuInit(0);
	cuDeviceGet(&device, 0);
	/* Blocking-sync context (suspend mode) so CPU waits sleep instead of
	 * busy-polling — incl. taskInit's cudaDeviceSynchronize and verify()'s
	 * stream sync, which the per-event blocking flag does not cover. Critical
	 * under SCHED_FIFO + sched_rt_runtime_us=-1: a spinning RT wait can starve
	 * the nvgpu driver thread and wedge the GPU. */
	cuCtxCreateCompat(&ctx, suspend_ ? CU_CTX_SCHED_BLOCKING_SYNC : 0, device);

	/* macro-bracket events follow the native apps' timing-disable convention */
	if (event_flags != 0) {
		checkCudaErrors(cudaEventCreateWithFlags(&start, event_flags));
		checkCudaErrors(cudaEventCreateWithFlags(&stop, event_flags));
	} else {
		checkCudaErrors(cudaEventCreate(&start));
		checkCudaErrors(cudaEventCreate(&stop));
	}
	/* dedicated timing-enabled events for the gpu_exec measurement */
	checkCudaErrors(cudaEventCreate(&ev_start));
	checkCudaErrors(cudaEventCreate(&ev_stop));

	checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

	switch (type) {
	case SeqWlType::MATMUL: {
		const size_t bytes = (size_t)mmN * mmN * sizeof(float);
		checkCudaErrors(cudaMalloc(&d_A, bytes));
		checkCudaErrors(cudaMalloc(&d_B, bytes));
		checkCudaErrors(cudaMalloc(&d_C, bytes));
		seq_fill_float_kernel<<<256, 256>>>(d_A,
			(uint32_t)((size_t)mmN * mmN), 0xA0A0u);
		seq_fill_float_kernel<<<256, 256>>>(d_B,
			(uint32_t)((size_t)mmN * mmN), 0xB0B0u);
		break;
	}
	case SeqWlType::HISTOGRAM: {
		checkCudaErrors(cudaMalloc(&d_histIn, (size_t)histN * sizeof(uint32_t)));
		checkCudaErrors(cudaMalloc(&d_bins, HIST_BINS * sizeof(uint32_t)));
		checkCudaErrors(cudaMalloc(&d_partials,
			(size_t)HIST_BLOCKS * HIST_BINS * sizeof(uint32_t)));
		seq_fill_uint_kernel<<<256, 256>>>(d_histIn, histN, 0xC0C0u);
		break;
	}
	case SeqWlType::CONVOLUTION: {
		const size_t bytes = (size_t)convW * convH * sizeof(float);
		const int kw = (int)p2;
		checkCudaErrors(cudaMalloc(&d_convIn, bytes));
		checkCudaErrors(cudaMalloc(&d_convOut, bytes));
		checkCudaErrors(cudaMalloc(&d_tmp, bytes));
		checkCudaErrors(cudaMalloc(&d_coef, (size_t)kw * sizeof(float)));
		float coef[64];
		for (int i = 0; i < kw; ++i) coef[i] = 1.0f / (float)kw;
		checkCudaErrors(cudaMemcpy(d_coef, coef, (size_t)kw * sizeof(float),
		                           cudaMemcpyHostToDevice));
		seq_fill_float_kernel<<<256, 256>>>(d_convIn,
			(uint32_t)((size_t)convW * convH), 0xD0D0u);
		break;
	}
	case SeqWlType::MLP: {
		const int W = mlpW, L = mlpL;
		const size_t mat = (size_t)W * W * sizeof(float);
		checkCudaErrors(cudaMalloc(&d_mlpInput,   mat));
		checkCudaErrors(cudaMalloc(&d_mlpAct0,    mat));
		checkCudaErrors(cudaMalloc(&d_mlpAct1,    mat));
		checkCudaErrors(cudaMalloc(&d_mlpWeights, (size_t)L * mat));
		checkCudaErrors(cudaMalloc(&d_mlpBias,
			(size_t)L * W * sizeof(float)));

		seq_fill_float_kernel<<<256, 256>>>(d_mlpInput,
			(uint32_t)((size_t)W * W), 0xE0E0u);
		const float wscale = sqrtf(3.0f / (float)W);   /* unit-variance Z */
		for (int l = 0; l < L; ++l) {
			seq_fill_signed_kernel<<<256, 256>>>(
				d_mlpWeights + (size_t)l * W * W,
				(uint32_t)((size_t)W * W),
				0x1234u + (uint32_t)l * 7919u, wscale);
			seq_fill_signed_kernel<<<256, 256>>>(
				d_mlpBias + (size_t)l * W,
				(uint32_t)W, 0x5678u + (uint32_t)l * 104729u, 0.01f);
		}
		break;
	}
	}
	checkCudaErrors(cudaDeviceSynchronize());
}

void SeqWorkload::launchKernels()
{
	switch (type) {
	case SeqWlType::MATMUL:
		seq_matmul_kernel<<<matmulGrid(mmN), dim3(MM_TILE, MM_TILE, 1), 0,
		                    stream>>>(d_A, d_B, d_C, mmN);
		break;
	case SeqWlType::HISTOGRAM:
		seq_hist_partial_kernel<<<HIST_BLOCKS, HIST_THREADS, 0, stream>>>(
			d_histIn, histN, d_partials);
		seq_hist_reduce_kernel<<<1, HIST_THREADS, 0, stream>>>(
			d_partials, d_bins);
		break;
	case SeqWlType::CONVOLUTION:
		seq_conv_row_kernel<<<convGrid(convW, convH),
		                      dim3(CONV_BLOCK, CONV_BLOCK, 1), 0, stream>>>(
			d_convIn, d_coef, d_tmp, convW, convH, convKr);
		seq_conv_col_kernel<<<convGrid(convW, convH),
		                      dim3(CONV_BLOCK, CONV_BLOCK, 1), 0, stream>>>(
			d_tmp, d_coef, d_convOut, convW, convH, convKr);
		break;
	case SeqWlType::MLP: {
		const int W = mlpW, L = mlpL;
		for (int l = 0; l < L; ++l) {
			float* inPtr  = mlpLayerInput(l);
			float* outPtr = (l & 1) ? d_mlpAct1 : d_mlpAct0;
			seq_matmul_kernel<<<matmulGrid(W), dim3(MM_TILE, MM_TILE, 1),
			                    0, stream>>>(
				inPtr, d_mlpWeights + (size_t)l * W * W, outPtr, W);
			seq_mlp_relu_bias_kernel<<<256, 256, 0, stream>>>(
				outPtr, d_mlpBias + (size_t)l * W, W);
		}
		break;
	}
	}
}

void SeqWorkload::taskCallback(int insId, int nIter)
{
	int pid = getpid();

	gcapsGpuSegBegin(fd, pid, sync_mode, ioctl_enabled);
	cudaEventRecord(ev_start, stream);
	launchKernels();
	cudaEventRecord(ev_stop, stream);
	gcapsGpuSegEnd(fd, pid, sync_mode, stream, ioctl_enabled);

	/* gcapsGpuSegEnd already synchronised the stream, so ev_stop is ready. */
	last_gpu_ms = 0.0f;
	cudaEventElapsedTime(&last_gpu_ms, ev_start, ev_stop);
}

void SeqWorkload::warmup()
{
	launchKernels();
	cudaStreamSynchronize(stream);
}

bool SeqWorkload::verify(bool relaunch)
{
	/* The whole GPU-touching part — the optional relaunch AND the D2H
	 * copies inside verifyChecks() — runs within one GCAPS segment
	 * bracket.  After any ioctl activity this context's TSG entries
	 * (compute and copy engine) are off the runlist, and unbracketed GPU
	 * work — a kernel launch or a plain cudaMemcpy — is never redispatched
	 * by the driver (runlist-cache-desync mechanism), blocking forever.
	 * The bracket re-adds the TSG for the duration of the verification.
	 * With ioctl mode off the macros reduce to event records. */
	int pid = getpid();
	gcapsGpuSegBegin(fd, pid, sync_mode, ioctl_enabled);
	if (relaunch) {
		launchKernels();
		checkCudaErrors(cudaStreamSynchronize(stream));
	}
	const bool ok = verifyChecks();
	gcapsGpuSegEnd(fd, pid, sync_mode, stream, ioctl_enabled);
	return ok;
}

bool SeqWorkload::verifyChecks()
{
	switch (type) {
	case SeqWlType::MATMUL: {
		const int n = mmN;
		float* rowA = (float*)malloc((size_t)n * sizeof(float));
		float* colB = (float*)malloc((size_t)n * sizeof(float));
		bool ok = true;
		for (int s = 0; s < 4 && ok; ++s) {
			const int i = (s * 977)  % n;
			const int j = (s * 1409) % n;
			cudaMemcpy(rowA, d_A + (size_t)i * n, (size_t)n * sizeof(float),
			           cudaMemcpyDeviceToHost);
			cudaMemcpy2D(colB, sizeof(float), d_B + j,
			             (size_t)n * sizeof(float), sizeof(float), n,
			             cudaMemcpyDeviceToHost);
			double ref = 0.0;
			for (int k = 0; k < n; ++k) ref += (double)rowA[k] * colB[k];
			float got;
			cudaMemcpy(&got, d_C + (size_t)i * n + j, sizeof(float),
			           cudaMemcpyDeviceToHost);
			if (fabs(ref - got) > 1e-2 * fmax(1.0, fabs(ref))) ok = false;
		}
		free(rowA); free(colB);
		return ok;
	}
	case SeqWlType::HISTOGRAM: {
		uint32_t bins[HIST_BINS];
		cudaMemcpy(bins, d_bins, sizeof(bins), cudaMemcpyDeviceToHost);
		uint64_t total = 0;
		for (int b = 0; b < HIST_BINS; ++b) total += bins[b];
		return total == histN;
	}
	case SeqWlType::CONVOLUTION: {
		const int w = convW, h = convH, kr = convKr;
		const int win = 2 * kr + 1;
		float* inWin = (float*)malloc((size_t)win * win * sizeof(float));
		bool ok = true;
		for (int s = 0; s < 4 && ok; ++s) {
			const int x = kr + (s * 1031) % (w - 2 * kr);
			const int y = kr + (s * 1523) % (h - 2 * kr);
			cudaMemcpy2D(inWin, (size_t)win * sizeof(float),
			             d_convIn + (size_t)(y - kr) * w + (x - kr),
			             (size_t)w * sizeof(float),
			             (size_t)win * sizeof(float), win,
			             cudaMemcpyDeviceToHost);
			const float c = 1.0f / (float)win;
			double ref = 0.0;
			for (int ky = 0; ky < win; ++ky) {
				double rowAcc = 0.0;
				for (int kx = 0; kx < win; ++kx)
					rowAcc += (double)c * inWin[ky * win + kx];
				ref += (double)c * rowAcc;
			}
			float got;
			cudaMemcpy(&got, d_convOut + (size_t)y * w + x, sizeof(float),
			           cudaMemcpyDeviceToHost);
			if (fabs(ref - got) > 1e-4) ok = false;
		}
		free(inWin);
		return ok;
	}
	case SeqWlType::MLP: {
		/* Full forward pass in double on the host, spot-checked vs the GPU. */
		const int W = mlpW, L = mlpL;
		const size_t N = (size_t)W * W;
		float*  hIn  = (float*)malloc(N * sizeof(float));
		float*  hW   = (float*)malloc((size_t)L * N * sizeof(float));
		float*  hB   = (float*)malloc((size_t)L * W * sizeof(float));
		float*  hOut = (float*)malloc(N * sizeof(float));
		double* cur  = (double*)malloc(N * sizeof(double));
		double* nxt  = (double*)malloc(N * sizeof(double));
		cudaMemcpy(hIn, d_mlpInput, N * sizeof(float),
		           cudaMemcpyDeviceToHost);
		cudaMemcpy(hW, d_mlpWeights, (size_t)L * N * sizeof(float),
		           cudaMemcpyDeviceToHost);
		cudaMemcpy(hB, d_mlpBias, (size_t)L * W * sizeof(float),
		           cudaMemcpyDeviceToHost);
		cudaMemcpy(hOut, mlpOutputSlab(), N * sizeof(float),
		           cudaMemcpyDeviceToHost);

		for (size_t i = 0; i < N; ++i) cur[i] = (double)hIn[i];
		for (int l = 0; l < L; ++l) {
			const float* Wl = hW + (size_t)l * N;
			const float* Bl = hB + (size_t)l * W;
			for (int r = 0; r < W; ++r)
				for (int c = 0; c < W; ++c) {
					double acc = 0.0;
					for (int k = 0; k < W; ++k)
						acc += cur[(size_t)r * W + k] *
						       (double)Wl[(size_t)k * W + c];
					acc += (double)Bl[c];
					nxt[(size_t)r * W + c] = acc > 0.0 ? acc : 0.0;
				}
			double* t = cur; cur = nxt; nxt = t;
		}

		bool ok = true;
		for (int s = 0; s < 256 && ok; ++s) {
			const size_t idx = ((size_t)s * 2654435761u) % N;
			const double ref = cur[idx];
			const double got = (double)hOut[idx];
			if (fabs(ref - got) > 1e-2 * fmax(1.0, fabs(ref))) ok = false;
		}
		free(hIn); free(hW); free(hB); free(hOut); free(cur); free(nxt);
		return ok;
	}
	}
	return false;
}

void SeqWorkload::taskFinish()
{
	cudaFree(d_A);      cudaFree(d_B);       cudaFree(d_C);
	cudaFree(d_histIn); cudaFree(d_bins);    cudaFree(d_partials);
	cudaFree(d_convIn); cudaFree(d_convOut); cudaFree(d_coef); cudaFree(d_tmp);
	cudaFree(d_mlpInput); cudaFree(d_mlpWeights); cudaFree(d_mlpBias);
	cudaFree(d_mlpAct0);  cudaFree(d_mlpAct1);
	cudaEventDestroy(start);    cudaEventDestroy(stop);
	cudaEventDestroy(ev_start); cudaEventDestroy(ev_stop);
	cudaStreamDestroy(stream);
	cuCtxDestroy(ctx);
}

void SeqWorkload::recordPriority(int priority)
{
	this->prio = priority;
}
