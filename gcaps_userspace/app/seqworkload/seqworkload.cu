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

// ============================================================================
// Host helpers
// ============================================================================

const char* seqWlTypeName(SeqWlType t)
{
	switch (t) {
		case SeqWlType::MATMUL:      return "matmul";
		case SeqWlType::HISTOGRAM:   return "histogram";
		case SeqWlType::CONVOLUTION: return "convolution";
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
	}
}

SeqWorkload::~SeqWorkload() {}

void SeqWorkload::taskInit()
{
	cuInit(0);
	cuDeviceGet(&device, 0);
	cuCtxCreate(&ctx, 0, device);

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

bool SeqWorkload::verify()
{
	launchKernels();
	checkCudaErrors(cudaStreamSynchronize(stream));

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
	}
	return false;
}

void SeqWorkload::taskFinish()
{
	cudaFree(d_A);      cudaFree(d_B);       cudaFree(d_C);
	cudaFree(d_histIn); cudaFree(d_bins);    cudaFree(d_partials);
	cudaFree(d_convIn); cudaFree(d_convOut); cudaFree(d_coef); cudaFree(d_tmp);
	cudaEventDestroy(start);    cudaEventDestroy(stop);
	cudaEventDestroy(ev_start); cudaEventDestroy(ev_stop);
	cudaStreamDestroy(stream);
	cuCtxDestroy(ctx);
}

void SeqWorkload::recordPriority(int priority)
{
	this->prio = priority;
}
