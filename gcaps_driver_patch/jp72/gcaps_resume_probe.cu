/*
 * gcaps_resume_probe.cu — does bracketing real GPU work with the GCAPS ioctl
 * break that work, in a SINGLE process with no scheduling logic involved?
 *
 * Everything about GCAPS rests on this claim (ECRTS 2024 §5.2):
 *
 *   "even if the active TSGs of a task are removed from the runlist, they are
 *    kept in the scheduler data structure of the GPU driver and won't be lost;
 *    hence, we can add those TSGs back to the runlist ... to resume their
 *    execution"
 *
 * That is an assertion about L4T R35.  On JP 7.2 / r39.2 the runlist gained
 * domains (nvgpu_runlist_update_locked takes a nvgpu_runlist_domain*, plus a
 * shadow domain), so it needs re-testing rather than assuming.
 *
 * One process.  One stream.  No priorities, no other tasks, no Algorithm 1.
 * Each iteration does exactly what gcapsGpuSegBegin/End brackets:
 *
 *     ioctl(add=true) -> H2D copy -> kernel -> D2H copy -> wait -> ioctl(add=false)
 *
 * The wait is a polled cudaEventQuery with a deadline, so a stall is REPORTED
 * rather than hanging forever.  Progress is written to a log file with fsync
 * after every step, so the breadcrumb survives even if the board resets.
 *
 *   nvcc -O2 -arch=sm_87 -o gcaps_resume_probe gcaps_resume_probe.cu
 *   ./gcaps_resume_probe --no-ioctl      # control: must pass
 *   ./gcaps_resume_probe                 # GCAPS bracket, best-effort caller
 *   sudo chrt -f 50 ./gcaps_resume_probe # GCAPS bracket, RT caller
 *
 * Interpretation:
 *   control passes, ioctl arm stalls  -> the ioctl breaks its OWN work; the
 *                                        §5.2 resume assumption is false on
 *                                        r39.2 and GCAPS cannot work as-is.
 *   both pass                         -> a single process is fine; the failure
 *                                        needs >1 contender, so re-run two
 *                                        copies concurrently.
 */

#include <cuda_runtime.h>

#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <sched.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <time.h>
#include <unistd.h>

#define NVGPU_GPU_IOCTL_MAGIC 'G'

struct nvgpu_gpu_runlist_update_rt_prio_args {
	bool sync_mode;
	pid_t pid;
	bool add_req;
};

#define NVGPU_GPU_IOCTL_RUNLIST_UPDATE_RT_PRIO \
	_IOWR(NVGPU_GPU_IOCTL_MAGIC, 49, \
		struct nvgpu_gpu_runlist_update_rt_prio_args)

static const char *DEV = "/dev/nvgpu/igpu0/ctrl";
static const char *LOG = "gcaps_resume_probe.log";

static FILE *g_logfp;

static void say(const char *fmt, ...)
{
	char buf[512];
	va_list ap;
	va_start(ap, fmt);
	vsnprintf(buf, sizeof(buf), fmt, ap);
	va_end(ap);

	printf("%s\n", buf);
	fflush(stdout);
	if (g_logfp) {
		fprintf(g_logfp, "%s\n", buf);
		fflush(g_logfp);
		fsync(fileno(g_logfp));      /* survive a hard reset */
	}
}

__global__ void busy_kernel(float *d, int n, long long spin_ns)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)
		d[i] = d[i] * 1.000001f + 1.0f;

	if (i == 0) {
		long long t0, t1;
		asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
		do {
			asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1));
		} while (t1 - t0 < spin_ns);
	}
}

static double now_s(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec + ts.tv_nsec / 1e9;
}

static int do_ioctl(int fd, pid_t pid, bool add)
{
	struct nvgpu_gpu_runlist_update_rt_prio_args a;
	memset(&a, 0, sizeof(a));
	a.pid = pid;
	a.sync_mode = false;
	a.add_req = add;
	return ioctl(fd, NVGPU_GPU_IOCTL_RUNLIST_UPDATE_RT_PRIO, &a);
}

int main(int argc, char **argv)
{
	bool use_ioctl = true;
	int iters = 20;
	double timeout_s = 5.0;

	for (int i = 1; i < argc; i++) {
		if (!strcmp(argv[i], "--no-ioctl"))      use_ioctl = false;
		else if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
	}

	g_logfp = fopen(LOG, "w");
	struct sched_param sp;
	int pol = sched_getscheduler(0);
	sched_getparam(0, &sp);
	say("gcaps_resume_probe: ioctl=%s iters=%d pid=%d sched=%s prio=%d",
	    use_ioctl ? "yes" : "no", iters, (int)getpid(),
	    pol == SCHED_FIFO ? "FIFO" : "OTHER", sp.sched_priority);

	int fd = open(DEV, O_RDWR);
	if (fd < 0) { say("FAIL open %s: %s", DEV, strerror(errno)); return 1; }

	const int N = 1 << 20;
	float *h = (float *)malloc(N * sizeof(float));
	float *d = NULL;
	cudaStream_t stream;
	cudaEvent_t ev;

	if (cudaMalloc(&d, N * sizeof(float)) != cudaSuccess) { say("FAIL cudaMalloc"); return 1; }
	cudaStreamCreate(&stream);
	cudaEventCreate(&ev);

	/* warm the context so lazy channel creation happens BEFORE any ioctl */
	cudaMemcpyAsync(d, h, N * sizeof(float), cudaMemcpyHostToDevice, stream);
	busy_kernel<<<(N + 255) / 256, 256, 0, stream>>>(d, N, 1000000LL);
	cudaStreamSynchronize(stream);
	say("context warmed");

	for (int it = 0; it < iters; it++) {
		if (use_ioctl) {
			int rc = do_ioctl(fd, getpid(), true);
			say("iter %d: ioctl add rc=%d errno=%d", it, rc, rc < 0 ? errno : 0);
			if (rc < 0) { say("FAIL add ioctl"); return 1; }
		}

		cudaMemcpyAsync(d, h, N * sizeof(float), cudaMemcpyHostToDevice, stream);
		busy_kernel<<<(N + 255) / 256, 256, 0, stream>>>(d, N, 2000000LL);
		cudaMemcpyAsync(h, d, N * sizeof(float), cudaMemcpyDeviceToHost, stream);
		cudaEventRecord(ev, stream);
		say("iter %d: work submitted", it);

		double deadline = now_s() + timeout_s;
		cudaError_t q;
		while ((q = cudaEventQuery(ev)) == cudaErrorNotReady) {
			if (now_s() > deadline) break;
			usleep(1000);
		}

		if (q == cudaErrorNotReady) {
			say("iter %d: *** STALLED *** work did not complete in %.1fs", it, timeout_s);
			say("VERDICT: GPU work bracketed by the GCAPS ioctl does not complete.");
			say("         The §5.2 resume assumption does not hold on this driver.");
			return 2;
		} else if (q != cudaSuccess) {
			say("iter %d: cuda error %d (%s)", it, q, cudaGetErrorString(q));
			return 3;
		}
		say("iter %d: completed", it);

		if (use_ioctl) {
			int rc = do_ioctl(fd, getpid(), false);
			say("iter %d: ioctl remove rc=%d errno=%d", it, rc, rc < 0 ? errno : 0);
			if (rc < 0) { say("FAIL remove ioctl"); return 1; }
		}
	}

	say("PASS: all %d iterations completed", iters);
	close(fd);
	return 0;
}
