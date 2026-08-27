#!/usr/bin/env python3
"""
Apply the GCAPS driver patches to the JetPack 7.2 / Jetson Linux r39.2 nvgpu tree.

The original gcaps_driver_patch/*.patch files are context diffs against L4T
R35.6.0 (kernel 5.10) and will NOT apply here: the nvgpu tree moved out-of-tree,
the UAPI header was restructured (~1.2k -> ~5.9k lines), and ioctl 43 is now
taken by NVIDIA.  This script performs the same four edits by anchoring on
stable surrounding text instead of line numbers.

Usage:
    cd <nvgpu source root>        # the dir containing drivers/ and include/
    python3 apply_gcaps_jp72.py [--check] [--revert-hint]

Idempotent: every edit is marked with "GCAPS" and re-running is a no-op.

Verified against r39.2 on Jetson AGX Orin (ga10b), kernel 6.8.12-1021-tegra.
"""

import argparse
import os
import re
import sys

SCHED_H = "drivers/gpu/nvgpu/include/nvgpu/sched.h"
SCHED_C = "drivers/gpu/nvgpu/os/linux/sched.c"
IOCTL_C = "drivers/gpu/nvgpu/os/linux/ioctl_ctrl.c"
UAPI_H = "include/uapi/linux/nvgpu-ctrl.h"

MARK = "GCAPS"

# The ioctl slot.  R35 used 43; on r39.2 NVIDIA occupies 43..48
# (GET_GPC_PHYSICAL_MAP .. GET_FBP_LOCAL_TO_LOGICAL_MAP), so GCAPS moves to 49.
GCAPS_IOCTL_NR = 49


# --------------------------------------------------------------------------
# 1. include/uapi/linux/nvgpu-ctrl.h
# --------------------------------------------------------------------------

UAPI_BLOCK = """
/* GCAPS: arguments for the runlist RT-priority control ioctl.
 * Field names and types are kept identical to the original R35 patch so the
 * existing gcaps_userspace sources compile unchanged. */
struct nvgpu_gpu_runlist_update_rt_prio_args {
	bool sync_mode;	/* true if gpu sync-based mode is used */
	/* [in] caller's pid */
	pid_t pid;
	/*
	 * [in] the caller wants to be added/removed to the runlist
	 * !add should be called when at the end of the caller's function
	 */
	bool add_req;
};

/* GCAPS: slot 43 (used by the original R35 patch) is taken on r39.2 by
 * NVGPU_GPU_IOCTL_GET_GPC_PHYSICAL_MAP; 43..48 are all occupied. */
#define NVGPU_GPU_IOCTL_RUNLIST_UPDATE_RT_PRIO \\
	_IOWR(NVGPU_GPU_IOCTL_MAGIC, {nr}, \\
		struct nvgpu_gpu_runlist_update_rt_prio_args)
#define NVGPU_GPU_IOCTL_LAST		\\
	_IOC_NR(NVGPU_GPU_IOCTL_RUNLIST_UPDATE_RT_PRIO)
""".replace("{nr}", str(GCAPS_IOCTL_NR))


def patch_uapi(text):
    # Replace the existing NVGPU_GPU_IOCTL_LAST definition (a #define whose
    # body is a line-continued _IOC_NR(...)).
    pat = re.compile(
        r"#define\s+NVGPU_GPU_IOCTL_LAST\s*\\\s*\n\s*_IOC_NR\([^)]*\)\s*\n")
    m = pat.search(text)
    if not m:
        raise LookupError(
            "could not find the NVGPU_GPU_IOCTL_LAST definition in %s" % UAPI_H)
    return text[:m.start()] + UAPI_BLOCK.lstrip("\n") + text[m.end():]


# --------------------------------------------------------------------------
# 2. drivers/gpu/nvgpu/include/nvgpu/sched.h
# --------------------------------------------------------------------------

SCHED_H_TYPES = """
/* GCAPS: preemptive GPU scheduling state. */
#include <linux/rtmutex.h>

#define RL_CTRL_NO_RT_PID 0
#define RL_CTRL_NO_RT_PRIO 0

/* valid only if #pid is a rt task; records the rt task in the runlist */
struct nvgpu_rt_rl {
	pid_t pid;
	int rt_prio;
};

struct nvgpu_rl_ctrl {
	/* the bitmap of TSGs which are scheduled in runlists */
	unsigned long *tsg_running;
	/* pid of the rt task whose TSGs are in the runlist; 0 if none */
	struct nvgpu_rt_rl rt_task_in_rl;
	/* the bitmap of TSGs waiting to be added to runlists */
	unsigned long *tsg_pending;
	unsigned long *curr_tsgs_in_rl;
};

"""

SCHED_H_FIELDS = """
	/* GCAPS: runlist control state.  cs_lock is an rt_mutex deliberately -
	 * it gives priority inheritance over the ioctl critical section.  Do not
	 * downgrade it to struct mutex on a non-PREEMPT_RT kernel. */
	struct nvgpu_rl_ctrl rl_ctrl;
	struct nvgpu_mutex sync_fence_lock;
	struct rt_mutex cs_lock;
"""


def patch_sched_h(text):
    anchor = "struct nvgpu_sched_ctrl {"
    i = text.find(anchor)
    if i < 0:
        raise LookupError("could not find 'struct nvgpu_sched_ctrl {' in %s"
                          % SCHED_H)
    text = text[:i] + SCHED_H_TYPES.lstrip("\n") + text[i:]

    # Append the new members just before the struct's closing brace.
    i = text.find(anchor)
    end = text.find("\n};", i)
    if end < 0:
        raise LookupError("could not find the end of struct nvgpu_sched_ctrl")
    return text[:end] + "\n" + SCHED_H_FIELDS + text[end + 1:]


# --------------------------------------------------------------------------
# 3. drivers/gpu/nvgpu/os/linux/sched.c
# --------------------------------------------------------------------------

SCHED_C_INIT_FN = """
/* GCAPS: allocate the runlist-control bitmaps. */
static int gk20a_sched_rl_ctrl_init(struct gk20a *g)
{
	struct nvgpu_sched_ctrl *sched = &g->sched_ctrl;
	struct nvgpu_rl_ctrl *rl_ctrl = &sched->rl_ctrl;

	rl_ctrl->tsg_running = nvgpu_kzalloc(g, sched->bitmap_size);
	if (rl_ctrl->tsg_running == NULL)
		return -ENOMEM;

	rl_ctrl->tsg_pending = nvgpu_kzalloc(g, sched->bitmap_size);
	if (rl_ctrl->tsg_pending == NULL)
		goto free_running;

	rl_ctrl->curr_tsgs_in_rl = nvgpu_kzalloc(g, sched->bitmap_size);
	if (rl_ctrl->curr_tsgs_in_rl == NULL)
		goto free_pending;

	*rl_ctrl->tsg_running = 0;
	*rl_ctrl->tsg_pending = 0;
	*rl_ctrl->curr_tsgs_in_rl = 0;

	rl_ctrl->rt_task_in_rl.pid = RL_CTRL_NO_RT_PID;
	rl_ctrl->rt_task_in_rl.rt_prio = RL_CTRL_NO_RT_PRIO;

	return 0;

free_pending:
	nvgpu_kfree(g, rl_ctrl->tsg_pending);
	rl_ctrl->tsg_pending = NULL;
free_running:
	nvgpu_kfree(g, rl_ctrl->tsg_running);
	rl_ctrl->tsg_running = NULL;
	return -ENOMEM;
}

/* GCAPS: release the runlist-control bitmaps. */
static void gk20a_sched_rl_ctrl_cleanup(struct gk20a *g)
{
	struct nvgpu_rl_ctrl *rl_ctrl = &g->sched_ctrl.rl_ctrl;

	nvgpu_kfree(g, rl_ctrl->curr_tsgs_in_rl);
	nvgpu_kfree(g, rl_ctrl->tsg_pending);
	nvgpu_kfree(g, rl_ctrl->tsg_running);
	rl_ctrl->curr_tsgs_in_rl = NULL;
	rl_ctrl->tsg_pending = NULL;
	rl_ctrl->tsg_running = NULL;
}

"""

SCHED_C_INIT_CALL = """	/* GCAPS */
	err = gk20a_sched_rl_ctrl_init(g);
	if (err != 0) {
		nvgpu_err(g, "gk20a_sched_rl_ctrl_init failed");
		return err;
	}
	nvgpu_mutex_init(&sched->sync_fence_lock);
	rt_mutex_init(&sched->cs_lock);

"""

SCHED_C_CLEANUP_CALL = """	/* GCAPS.  rt_mutex_destroy() no longer exists on kernel 6.x, so the
	 * rt_mutex needs no teardown. */
	nvgpu_mutex_destroy(&sched->sync_fence_lock);
	gk20a_sched_rl_ctrl_cleanup(g);

"""


def _insert_before_line(text, needle, block, what):
    i = text.find(needle)
    if i < 0:
        raise LookupError("could not find %r (%s)" % (needle, what))
    bol = text.rfind("\n", 0, i) + 1
    return text[:bol] + block + text[bol:]


def patch_sched_c(text):
    anchor_fn = "int gk20a_sched_ctrl_init(struct gk20a *g)"
    i = text.find(anchor_fn)
    if i < 0:
        raise LookupError("could not find gk20a_sched_ctrl_init() in %s"
                          % SCHED_C)
    bol = text.rfind("\n", 0, i) + 1
    text = text[:bol] + SCHED_C_INIT_FN.lstrip("\n") + text[bol:]

    text = _insert_before_line(text, "sched->sw_ready = true;",
                               SCHED_C_INIT_CALL, "sched_ctrl_init tail")
    text = _insert_before_line(text, "sched->sw_ready = false;",
                               SCHED_C_CLEANUP_CALL, "sched_ctrl_cleanup tail")
    return text


# --------------------------------------------------------------------------
# 4. drivers/gpu/nvgpu/os/linux/ioctl_ctrl.c
# --------------------------------------------------------------------------

IOCTL_FN = r"""
/* ---------------------------- GCAPS ---------------------------------- */

#define GCAPS_NO_PRIO (-1)

/*
 * RCU-safe pid -> rt_priority.  The original R35 patch called
 * pid_task(find_vpid(...)) bare; that is a use-after-free hazard and splats
 * under CONFIG_PROVE_RCU.  The lookups cannot simply be wrapped in one big
 * rcu_read_lock() because this ioctl sleeps (runlist reload), so each lookup
 * copies out what it needs under a short read-side section instead.
 */
static int gcaps_rt_prio_of(pid_t pid)
{
	struct task_struct *t;
	int prio = GCAPS_NO_PRIO;

	rcu_read_lock();
	t = pid_task(find_vpid(pid), PIDTYPE_PID);
	if (t != NULL)
		prio = (int)t->rt_priority;
	rcu_read_unlock();

	return prio;
}

/*
 * TSGs belonging to the desktop compositor are never evicted.  Ubuntu 24.04
 * (JetPack 7.x) defaults to Wayland, so Xorg may be absent; Xwayland is added
 * for the XWayland-backed case.  Extend this list rather than the old
 * hard-coded "j < 2" bound.
 */
static bool gcaps_is_except_proc(pid_t tgid)
{
	/* Verified on the JP 7.2 box (GNOME/X11, Ubuntu 24.04): the processes
	 * actually holding nvgpu fds are Xorg, gnome-shell and mutter-x11-frames.
	 * mutter-x11-frames is a SEPARATE process only since GNOME 43 - on the
	 * R35 board's Ubuntu 20.04 it lived inside gnome-shell, which is why the
	 * original two-entry list was complete there and is NOT here.  Evicting
	 * it mid-render is what makes runlist.update(wait_for_finish=true) time
	 * out and escalate to gv11b_fifo_recover.
	 *
	 * task->comm is truncated to TASK_COMM_LEN-1 (15) chars, so
	 * "mutter-x11-frames" shows up as "mutter-x11-fram".  strncmp over that
	 * length lets full names be listed here and still match. */
	static const char * const except[] = {
		"Xorg", "gnome-shell", "Xwayland", "mutter-x11-frames",
	};
	struct task_struct *t;
	char comm[TASK_COMM_LEN];
	unsigned int k;

	comm[0] = '\0';

	rcu_read_lock();
	t = pid_task(find_vpid(tgid), PIDTYPE_PID);
	if (t != NULL)
		strscpy(comm, t->comm, sizeof(comm));
	rcu_read_unlock();

	if (comm[0] == '\0')
		return false;

	for (k = 0; k < ARRAY_SIZE(except); k++) {
		if (strncmp(except[k], comm, TASK_COMM_LEN - 1) == 0)
			return true;
	}

	return false;
}

/*
 * Pick the pending TSG set belonging to the highest-priority RT task.
 * Caller holds sched->cs_lock.
 */
static void nvgpu_get_tsgs_with_highest_prio_locked(struct gk20a *g,
						pid_t *pid_next,
						int *rt_prio_next,
						unsigned long *tsgs_next)
{
	struct nvgpu_fifo *f = &g->fifo;
	struct nvgpu_sched_ctrl *sched = &g->sched_ctrl;
	struct nvgpu_tsg *tsg;
	int prio_highest = 0;
	unsigned int i;

	*pid_next = -1;
	*tsgs_next = 0;
	*rt_prio_next = 0;

	for (i = 0; i < f->num_channels; i++) {
		int prio;

		tsg = &f->tsg[i];
		if (!NVGPU_SCHED_ISSET(tsg->tsgid, sched->active_tsg_bitmap))
			continue;

		if (!nvgpu_test_bit(i, sched->rl_ctrl.tsg_pending))
			continue;

		prio = gcaps_rt_prio_of(tsg->tgid);
		if (prio >= 1 && prio <= 99 && prio > prio_highest) {
			prio_highest = prio;
			*pid_next = tsg->tgid;
		}
	}

	if (*pid_next == -1)
		return;

	*rt_prio_next = prio_highest;

	for (i = 0; i < f->num_channels; i++) {
		tsg = &f->tsg[i];
		if (tsg->tgid == *pid_next)
			nvgpu_set_bit(i, tsgs_next);
	}
}

/*
 * GCAPS runlist control.  See gcaps_driver_patch/readme.md for the algorithm;
 * the logic below is a faithful port of the R35 implementation.
 */
static int nvgpu_ioctl_runlist_update_rt_prio(struct gk20a *g,
		struct nvgpu_gpu_runlist_update_rt_prio_args *args)
{
	struct nvgpu_fifo *f = &g->fifo;
	struct nvgpu_sched_ctrl *sched = &g->sched_ctrl;
	struct nvgpu_rl_ctrl *rl_ctrl = &sched->rl_ctrl;
	struct nvgpu_rt_rl *rt_task_in_rl = &rl_ctrl->rt_task_in_rl;
	unsigned long *new_tsg_in_rl;
	unsigned long *tsgs_cpid;
	unsigned long *tsgs_next;
	pid_t cpid = args->pid;
	bool add_req = args->add_req;
	bool sync_mode = args->sync_mode;
	int caller_prio;
	bool exist = false;
	unsigned int i;
	int err = 0;
	ktime_t start_time;
	ktime_t stop_time;
	s64 elapsed_time;
	pid_t pid_next = -1;
	int rt_prio_next = 0;

	/* measurement instrumentation, consumed by measure_preempt_overhead.py */
	bool rl_updated = false;
	pid_t preempted_pid = -1;
	pid_t resumed_pid = -1;

	/* sync-based approach: no runlist work.  Checked BEFORE allocating -
	 * the R35 version returned here without freeing, leaking three bitmaps
	 * on every sync-mode ioctl. */
	if (sync_mode) {
		if (add_req)
			nvgpu_mutex_acquire(&sched->sync_fence_lock);
		else
			nvgpu_mutex_release(&sched->sync_fence_lock);
		return 0;
	}

	new_tsg_in_rl = nvgpu_kzalloc(g, sched->bitmap_size);
	tsgs_cpid = nvgpu_kzalloc(g, sched->bitmap_size);
	tsgs_next = nvgpu_kzalloc(g, sched->bitmap_size);
	if (new_tsg_in_rl == NULL || tsgs_cpid == NULL || tsgs_next == NULL) {
		err = -ENOMEM;
		goto out_free;
	}

	*new_tsg_in_rl = 0;
	*tsgs_cpid = 0;
	*tsgs_next = 0;

	caller_prio = gcaps_rt_prio_of(cpid);

	/* GCAPS: take a GPU power reference before touching runlist registers.
	 * Every other register-touching handler in this file does this; the R35
	 * patch did not.  With railgating enabled the GPU powers down between
	 * segments and the runlist write lands in an unmapped BAR ("Attempted
	 * access to GPU regs after unmapping"), corrupting driver state: later
	 * ioctls return -ENODEV and the Tegra watchdog resets the board.
	 * Taken OUTSIDE cs_lock so a GPU power-up cannot extend the critical
	 * section (that would inflate the blocking term in the analysis). */
	err = gk20a_busy(g);
	if (err != 0) {
		nvgpu_err(g, "failed to power on gpu for runlist update");
		goto out_free;
	}

	rt_mutex_lock(&sched->cs_lock);
	start_time = ktime_get();

	/* never evict the compositor's TSGs */
	for (i = 0; i < f->num_channels; i++) {
		if (gcaps_is_except_proc(f->tsg[i].tgid))
			nvgpu_set_bit(i, new_tsg_in_rl);
	}

	/* TSGs of the caller */
	for (i = 0; i < f->num_channels; i++) {
		struct nvgpu_tsg *tsg = &f->tsg[i];

		if (!NVGPU_SCHED_ISSET(tsg->tsgid, sched->active_tsg_bitmap))
			continue;
		if (tsg->tgid == cpid)
			nvgpu_set_bit(i, tsgs_cpid);
	}

	if (add_req) {
		if (caller_prio <= 0) {
			/* best-effort caller */
			if (rt_task_in_rl->pid == RL_CTRL_NO_RT_PID) {
				*rl_ctrl->tsg_running |= *tsgs_cpid;
				*rl_ctrl->tsg_pending &= ~(*tsgs_cpid);
			} else {
				*rl_ctrl->tsg_pending |= *tsgs_cpid;
				*rl_ctrl->tsg_running &= ~(*tsgs_cpid);
			}
		} else {
			/* RT caller.  Equal priorities keep FIFO order (no
			 * interleaving), so the test is strictly greater. */
			if (caller_prio > rt_task_in_rl->rt_prio) {
				preempted_pid = rt_task_in_rl->pid;
				*rl_ctrl->tsg_pending |= *rl_ctrl->tsg_running;
				*rl_ctrl->tsg_running = *tsgs_cpid;
				*rl_ctrl->tsg_pending &= ~(*tsgs_cpid);
				rt_task_in_rl->pid = cpid;
				rt_task_in_rl->rt_prio = caller_prio;
			} else {
				*rl_ctrl->tsg_pending |= *tsgs_cpid;
				*rl_ctrl->tsg_running &= ~(*tsgs_cpid);
			}
		}
	} else {
		nvgpu_get_tsgs_with_highest_prio_locked(g, &pid_next,
						&rt_prio_next, tsgs_next);
		if (pid_next > 0) {
			if (pid_next == cpid) {
				*rl_ctrl->tsg_pending &= ~(*tsgs_next);
				*rl_ctrl->tsg_running &= ~(*tsgs_next);
				if (*rl_ctrl->tsg_running == 0) {
					rt_task_in_rl->pid = RL_CTRL_NO_RT_PID;
					rt_task_in_rl->rt_prio =
						RL_CTRL_NO_RT_PRIO;
				}
			} else {
				resumed_pid = pid_next;
				*rl_ctrl->tsg_running = *tsgs_next;
				*rl_ctrl->tsg_pending &= ~(*tsgs_next);
				rt_task_in_rl->pid = pid_next;
				rt_task_in_rl->rt_prio = rt_prio_next;
			}
		} else {
			*rl_ctrl->tsg_pending &= ~(*tsgs_cpid);
			*rl_ctrl->tsg_running &= ~(*tsgs_cpid);

			for (i = 0; i < f->num_channels; i++) {
				struct nvgpu_tsg *tsg = &f->tsg[i];
				int prio;

				if (!NVGPU_SCHED_ISSET(tsg->tsgid,
						sched->active_tsg_bitmap))
					continue;
				if (!nvgpu_test_bit(i, rl_ctrl->tsg_running))
					continue;

				prio = gcaps_rt_prio_of(tsg->tgid);
				if (prio >= 1 && prio <= 99) {
					rt_task_in_rl->pid = tsg->tgid;
					rt_task_in_rl->rt_prio = prio;
					exist = true;
					break;
				}
			}

			if (!exist) {
				*rl_ctrl->tsg_running = *rl_ctrl->tsg_pending;
				*rl_ctrl->tsg_pending = 0;
				rt_task_in_rl->pid = RL_CTRL_NO_RT_PID;
				rt_task_in_rl->rt_prio = RL_CTRL_NO_RT_PRIO;
			}
		}
	}

	if ((*rl_ctrl->tsg_running & *rl_ctrl->tsg_pending) != 0)
		nvgpu_warn(g, "running and pending tsgs are not exclusive");

	/* OR, not assign: the exception TSGs are already set */
	*new_tsg_in_rl |= *rl_ctrl->tsg_running;

	if (*new_tsg_in_rl != *rl_ctrl->curr_tsgs_in_rl) {
		rl_updated = true;
		for (i = 0; i < f->num_channels; i++) {
			struct nvgpu_tsg *tsg = &f->tsg[i];
			struct nvgpu_channel *ch;
			bool add = nvgpu_test_bit(i, new_tsg_in_rl);

			/* GCAPS: the R35 original walked EVERY tsg slot here,
			 * including torn-down ones, dereferencing tsg->runlist
			 * and iterating tsg->ch_list on them.  Both other loops
			 * in this function gate on active_tsg_bitmap; this one
			 * did not.  A stale slot is a use-after-free walk under
			 * the runlist lock.  Inactive TSGs are not in our
			 * bitmaps anyway, so skipping them cannot change which
			 * TSGs GCAPS intends to schedule. */
			if (!NVGPU_SCHED_ISSET(tsg->tsgid,
					sched->active_tsg_bitmap))
				continue;
			if (tsg->runlist == NULL)
				continue;

			nvgpu_list_for_each_entry(ch, &tsg->ch_list,
					nvgpu_channel, ch_entry) {
				/* runlist lock is taken inside */
				err = g->ops.runlist.update(g, tsg->runlist,
						ch, add, true);
				if (err < 0) {
					nvgpu_err(g, "runlist update failed");
					goto done;
				}
			}
		}
		*rl_ctrl->curr_tsgs_in_rl = *new_tsg_in_rl;
	}

done:
	stop_time = ktime_get();
	elapsed_time = ktime_to_ns(ktime_sub(stop_time, start_time)) / 1000;
	pr_info("process %d elapsed time: %lld", cpid, elapsed_time);
	/* GCAPS_EV: one structured record per ioctl.
	 *   ts          - ktime (ns), same base as userspace CLOCK_MONOTONIC
	 *   prio        - caller SCHED_FIFO rt_priority (1..99), -1 if unknown
	 *   add         - 1 = entering a GPU segment, 0 = leaving it
	 *   rlupd       - 1 = an actual runlist reload happened
	 *   elapsed_us  - duration of the critical section (GCAPS epsilon)
	 *   preempted   - pid evicted to pending by this add, or -1
	 *   resumed     - pid re-admitted to the runlist by this remove, or -1 */
	pr_info("GCAPS_EV ts=%lld cpid=%d prio=%d add=%d rlupd=%d elapsed_us=%lld preempted=%d resumed=%d",
		ktime_to_ns(stop_time), cpid, caller_prio,
		add_req ? 1 : 0, rl_updated ? 1 : 0, elapsed_time,
		preempted_pid, resumed_pid);
	rt_mutex_unlock(&sched->cs_lock);
	gk20a_idle(g);

out_free:
	/* the R35 version never freed these - a leak on every ioctl */
	nvgpu_kfree(g, tsgs_next);
	nvgpu_kfree(g, tsgs_cpid);
	nvgpu_kfree(g, new_tsg_in_rl);
	return err;
}

/* -------------------------- end GCAPS -------------------------------- */

"""

IOCTL_CASE = """	/* GCAPS */
	case NVGPU_GPU_IOCTL_RUNLIST_UPDATE_RT_PRIO:
		err = nvgpu_ioctl_runlist_update_rt_prio(g,
			(struct nvgpu_gpu_runlist_update_rt_prio_args *)buf);
		break;

"""


def patch_ioctl_c(text):
    if "#include <nvgpu/runlist.h>" not in text:
        anchor = "#include <nvgpu/channel.h>"
        if anchor not in text:
            raise LookupError("could not find %r in %s" % (anchor, IOCTL_C))
        text = text.replace(
            anchor, anchor + "\n#include <nvgpu/runlist.h>	/* GCAPS */", 1)

    anchor_fn = "long gk20a_ctrl_dev_ioctl(struct file *filp"
    i = text.find(anchor_fn)
    if i < 0:
        raise LookupError("could not find gk20a_ctrl_dev_ioctl() in %s"
                          % IOCTL_C)
    bol = text.rfind("\n", 0, i) + 1
    text = text[:bol] + IOCTL_FN.lstrip("\n") + text[bol:]

    m = re.search(r"\n(\t*)default:\n\t*nvgpu_log_info\(g,\s*\n?\s*"
                  r"\"unrecognized gpu ioctl cmd", text)
    if not m:
        raise LookupError(
            "could not find the gpu ioctl switch default: case in %s" % IOCTL_C)
    ins = m.start() + 1
    return text[:ins] + IOCTL_CASE + text[ins:]


# --------------------------------------------------------------------------

EDITS = [
    (UAPI_H, patch_uapi),
    (SCHED_H, patch_sched_h),
    (SCHED_C, patch_sched_c),
    (IOCTL_C, patch_ioctl_c),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report status without writing anything")
    ap.add_argument("--root", default=".",
                    help="nvgpu source root (default: cwd)")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    missing = [p for p, _ in EDITS if not os.path.isfile(os.path.join(root, p))]
    if missing:
        sys.exit("not an r39.2 nvgpu source root (%s): missing %s"
                 % (root, ", ".join(missing)))

    rc = 0
    for rel, fn in EDITS:
        path = os.path.join(root, rel)
        with open(path, "r", encoding="utf-8", errors="surrogateescape") as fh:
            text = fh.read()

        if MARK in text:
            print("  skip    %s (already patched)" % rel)
            continue
        if args.check:
            print("  todo    %s" % rel)
            continue

        try:
            new = fn(text)
        except LookupError as exc:
            print("  FAIL    %s: %s" % (rel, exc))
            rc = 1
            continue

        with open(path + ".gcaps-orig", "w", encoding="utf-8",
                  errors="surrogateescape") as fh:
            fh.write(text)
        with open(path, "w", encoding="utf-8", errors="surrogateescape") as fh:
            fh.write(new)
        print("  patched %s  (backup: %s.gcaps-orig)" % (rel, rel))

    if rc == 0 and not args.check:
        print("\nAll four edits applied.  GCAPS ioctl nr = %d" % GCAPS_IOCTL_NR)
        print("Next: build the module, then rebuild gcaps_userspace against")
        print("the patched include/uapi/linux/nvgpu-ctrl.h.")
    return rc


if __name__ == "__main__":
    sys.exit(main())
