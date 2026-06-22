# GCAPS best-effort task deadlock

## Summary

GCAPS deadlocks when **two or more macro-instrumented best-effort (non-real-time)
GPU tasks run concurrently**. A lost-update in the runlist *remove* logic evicts
a still-running best-effort task from the GPU runlist and never puts it back. That
task's GPU segment can no longer make progress, so its `gcapsGpuSegEnd()` (which
ends in a `cudaEventSynchronize`) blocks forever and the whole taskset stalls.

The defect exists in the **published Algorithm 1** of the GCAPS paper
(ECRTS 2024, line 24) and is faithfully reproduced in the driver implementation.
It is undocumented; the authors' evaluations never exercised it because every
configuration they actually *ran* had at most one macro-instrumented best-effort
GPU task.

## Root cause

`gcaps_driver_patch/ioctl_ctrl.c.patch`, function
`nvgpu_ioctl_runlist_update_rt_prio`, the "caller requests to be removed" branch,
sub-case "no pending real-time task":

```c
} else { /* no pending rt task */
    *rl_ctrl->tsg_pending &= ~(*tsgs_cpid);
    *rl_ctrl->tsg_running &= ~(*tsgs_cpid);
    /* ... scan tsg_running for any real-time task ... */
    if (exist == false) {            /* only best-effort tasks remain */
        *rl_ctrl->tsg_running = *rl_ctrl->tsg_pending;   /* <-- BUG: overwrite */
        *rl_ctrl->tsg_pending = 0;
        rt_task_in_rl->pid     = RL_CTRL_NO_RT_PID;
        rt_task_in_rl->rt_prio = RL_CTRL_NO_RT_PRIO;
    }
}
```

This is line 24 of Algorithm 1 in the paper:

```
18: else                      ▷ τi requests to be removed
19:   τk ← highest-priority RT task in task_pending
20:   if τk exists then
21:     Move τk to task_running
22:     Remove τi from task_running
23:   else                    ▷ no pending real-time task
24:     task_running ← task_pending          ◀ overwrite
25:     task_pending ← ∅
26: Add all TSGs of tasks in task_running to the runlist
```

`task_running ← task_pending` (`tsg_running = tsg_pending`) **replaces** the
running set with the pending set. It therefore drops any task that was *running*
but is neither the caller nor in the pending set. The paper's prose states the
intent — *"if there are only best-effort tasks, the scheduler adds all of them to
the runlist to resume their progress in a time-shared manner"* — but the
assignment only re-adds the *pending* best-effort tasks, silently evicting the
ones that were already running.

## Minimal trigger

Two best-effort GPU tasks `A` and `B`, both calling `gcapsGpuSegBegin/End`:

1. `A` begins a segment → no real-time task running → `task_running = {A}`.
2. `B` begins a segment → no real-time task running → `task_running = {A, B}`
   (best-effort tasks are admitted to the runlist concurrently — by design).
3. `A` finishes → `gcapsGpuSegEnd` → no pending real-time task →
   `task_running ← task_pending` (= `∅`). **`B` is dropped from the runlist**,
   although it is mid-segment and never requested removal.
4. `B`'s TSG is off the runlist → its kernel never completes → its
   `gcapsGpuSegEnd` (`cudaEventSynchronize`) blocks forever → **deadlock**.

A *single* best-effort GPU task never triggers it (there is no second running
task to lose). Real-time tasks are immune: at most one real-time task is in
`task_running` at any time, so there is never a second running task to drop.

## Why it was never observed

- The runnable case study (`gcaps_userspace/taskset.csv`) has **exactly one**
  macro-instrumented best-effort GPU task (task 6, `mmul_gpu_2`).
- The paper's Table 4 lists a *second* best-effort task (task 7, a graphics
  app), but it is **unmodified graphics code that does not call the GCAPS
  macros** — it never drives the runlist state machine, so it cannot be the
  second concurrently-running macro task.
- The schedulability experiment that sweeps the best-effort-task ratio up to 90%
  (paper Fig. 8f) is purely **analytical** (response-time test); it never runs
  the driver.

## How it was found

Porting the GCAPS Table 4 taskset into the `workloadTasksetGcaps` benchmark with
**two** best-effort GPU tasks (`mm_2048`, `hist_4M`, both `SCHED_OTHER`).
Bisecting with the benchmark's `-k N` knob (activate only the first N GPU tasks):

- `-k 5` → one best-effort GPU task → **completes**.
- `-k 6` → two best-effort GPU tasks → **deadlocks**; a stuck child's backtrace
  sits in `cudaEventSynchronize` inside `SeqWorkload::taskCallback`
  (the `gcapsGpuSegEnd` wait), GPU not wedged (interruptible sleep).

Observed on Jetson AGX Orin (L4T R35.6) with `SCHED_FIFO` active.

## Fixes

**Driver (root cause).** Preserve the already-running tasks instead of
overwriting — union rather than assign:

```c
*rl_ctrl->tsg_running |= *rl_ctrl->tsg_pending;   /* keep running + resume pending */
*rl_ctrl->tsg_pending = 0;
```

(Equivalently in the pseudocode: `task_running ← task_running ∪ task_pending`
after removing the caller.)

**Userspace (workaround, used by this benchmark).** Give every GPU task a
real-time priority (`SCHED_FIFO`, ≥ 1) so the buggy best-effort path is never
taken. `gcaps_userspace/workloadTasksetGcaps.cc` now assigns the two formerly
best-effort tasks (`mm_2048`, `hist_4M`) the two **lowest** real-time priorities
instead of `SCHED_OTHER`. This also makes the benchmark a more direct comparison
with the SequenceScheduler benchmark in `msc-thesis-on-gpu-sched`, which
priority-schedules every GPU task and has no best-effort class.
