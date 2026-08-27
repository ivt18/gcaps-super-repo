/*
 * gcaps_smoke.c — minimal smoke test for the GCAPS runlist ioctl on
 * JetPack 7.2 / Jetson Linux r39.2.
 *
 * Deliberately self-contained: the ioctl number and args struct are declared
 * inline rather than pulled from the patched UAPI header, so a pass proves the
 * *driver* really exposes ioctl 49 with the expected argument size — it cannot
 * be faked by a stale or mismatched header.
 *
 *   cc -O2 -o gcaps_smoke gcaps_smoke.c
 *   ./gcaps_smoke                 # best-effort caller (rt_priority == 0)
 *   chrt -f 50 ./gcaps_smoke      # exercises the RT branch (needs rtprio)
 *
 * Then confirm the driver logged its event record:
 *   sudo dmesg | tail -20 | grep GCAPS_EV
 *
 * Safe to run: add_req=true and add_req=false are always paired, so runlist
 * state is restored before exit.  Run it with no CUDA workload active.
 */

#include <errno.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <unistd.h>

#define NVGPU_GPU_IOCTL_MAGIC 'G'

/* Must match the kernel struct exactly — the ioctl number encodes its size. */
struct nvgpu_gpu_runlist_update_rt_prio_args {
	bool sync_mode;
	pid_t pid;
	bool add_req;
};

#define NVGPU_GPU_IOCTL_RUNLIST_UPDATE_RT_PRIO \
	_IOWR(NVGPU_GPU_IOCTL_MAGIC, 49, \
		struct nvgpu_gpu_runlist_update_rt_prio_args)

/*
 * The negative control: nr 50 is one past NVGPU_GPU_IOCTL_LAST after the GCAPS
 * patch.  The dispatcher guards with
 *     _IOC_NR(cmd) > NVGPU_GPU_IOCTL_LAST  ->  -EINVAL
 * so 49 succeeding while 50 is rejected pins LAST at exactly 49 — which is what
 * proves the patch bumped it, not merely that a case was added.
 */
#define NVGPU_GPU_IOCTL_ONE_PAST_LAST \
	_IOWR(NVGPU_GPU_IOCTL_MAGIC, 50, \
		struct nvgpu_gpu_runlist_update_rt_prio_args)

static const char *DEV = "/dev/nvhost-ctrl-gpu";

static int fails;

static void check(const char *what, int ok, const char *detail)
{
	printf("  [%s] %-38s %s\n", ok ? "PASS" : "FAIL", what, detail);
	if (!ok)
		fails++;
}

int main(void)
{
	struct nvgpu_gpu_runlist_update_rt_prio_args a;
	char buf[128];
	int fd, rc;

	printf("GCAPS smoke test\n");
	printf("  struct size = %zu bytes, ioctl nr = %lu, _IOC_SIZE = %lu\n\n",
	       sizeof(a),
	       (unsigned long)_IOC_NR(NVGPU_GPU_IOCTL_RUNLIST_UPDATE_RT_PRIO),
	       (unsigned long)_IOC_SIZE(NVGPU_GPU_IOCTL_RUNLIST_UPDATE_RT_PRIO));

	fd = open(DEV, O_RDWR);
	if (fd < 0) {
		printf("  [FAIL] open %s: %s\n", DEV, strerror(errno));
		return 1;
	}
	check("open " , 1, DEV);

	/* 1. Negative control: an unimplemented ioctl must be rejected.  If this
	 *    "succeeds" the dispatcher is not discriminating and test 2 proves
	 *    nothing. */
	memset(&a, 0, sizeof(a));
	a.pid = getpid();
	rc = ioctl(fd, NVGPU_GPU_IOCTL_ONE_PAST_LAST, &a);
	snprintf(buf, sizeof(buf), "rc=%d errno=%s (want EINVAL)",
		 rc, rc < 0 ? strerror(errno) : "-");
	check("control: ioctl 50 > LAST rejected", rc < 0 && errno == EINVAL, buf);

	/* 2. Enter a GPU segment. */
	memset(&a, 0, sizeof(a));
	a.pid = getpid();
	a.sync_mode = false;
	a.add_req = true;
	rc = ioctl(fd, NVGPU_GPU_IOCTL_RUNLIST_UPDATE_RT_PRIO, &a);
	snprintf(buf, sizeof(buf), "rc=%d errno=%s",
		 rc, rc < 0 ? strerror(errno) : "-");
	check("GCAPS ioctl 49, add_req=true", rc == 0, buf);
	if (rc < 0 && errno == EINVAL)
		printf("         -> EINVAL here means LAST < 49: STOCK driver loaded.\n");

	/* 3. Leave the GPU segment — always paired with step 2. */
	memset(&a, 0, sizeof(a));
	a.pid = getpid();
	a.sync_mode = false;
	a.add_req = false;
	rc = ioctl(fd, NVGPU_GPU_IOCTL_RUNLIST_UPDATE_RT_PRIO, &a);
	snprintf(buf, sizeof(buf), "rc=%d errno=%s",
		 rc, rc < 0 ? strerror(errno) : "-");
	check("GCAPS ioctl 49, add_req=false", rc == 0, buf);

	close(fd);

	printf("\n%s (%d failure%s)\n", fails ? "SMOKE TEST FAILED" : "SMOKE TEST PASSED",
	       fails, fails == 1 ? "" : "s");
	printf("Now check the driver side:  sudo dmesg | grep GCAPS_EV | tail -5\n");
	return fails ? 1 : 0;
}
