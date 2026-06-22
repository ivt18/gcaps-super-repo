# GCAPS runlist-cache desync deadlock

## Summary

GCAPS keeps a **private cache of the runlist contents** (`curr_tsgs_in_rl`) and
uses it to *skip* the hardware runlist update when it thinks nothing changed. But
the real runlist membership is owned by the nvgpu driver
(`domain->active_channels` / `active_tsgs`) and is mutated **independently** by
the normal channel submit/teardown/recovery paths — which GCAPS neither tracks
nor hooks. When those paths change the runlist underneath GCAPS, its cache goes
stale; the next GCAPS request can then compute "no change needed" and **skip a
hardware update that was actually required**. A TSG that should be on the runlist
is left off it, its kernel is never dispatched, and the task's
`gcapsGpuSegEnd` / `cudaEventSynchronize` blocks on a GPU fence forever — a
deadlock.

This is a **design-level** flaw (a private shadow copy of a concurrently-mutated
resource, used to suppress updates, with no invalidation), distinct from and
deeper than the best-effort overwrite bug in
[`best-effort-tasks-bug.md`](best-effort-tasks-bug.md).

## Symptom

On Jetson AGX Orin (L4T R35.6) running the `workloadTasksetGcaps` mixed taskset
under `-i 1` with all tasks real-time:

- short runs complete (`-d 10`), longer runs hang (`-d 20`) — i.e. **duration /
  iteration-count dependent**, the signature of a timing race rather than a
  deterministic bug;
- the parent is blocked in `wait4` on **one** child; the other children have
  exited cleanly (zombies);
- the stuck child is the **highest-priority** task (`hist_16M`), parked in
  `ioctl → libnvrm_host1x → libcuda → cudaEventSynchronize → taskCallback`
  (a GPU fence wait for a kernel that never runs);
- the GPU is **not** wedged (every other task finished, so the hardware was
  servicing kernels up to the end);
- `dmesg` is clean — none of the driver's own asserts fire.

## Root cause

Two independent writers to the hardware runlist membership, only one of which
GCAPS tracks.

**Authoritative state (nvgpu).** What is actually on the runlist is
`domain->active_channels` / `domain->active_tsgs`, maintained by
`nvgpu_runlist_modify_active_locked` (`common/fifo/runlist.c`). Adding/removing a
channel sets/clears that bit and rebuilds+submits the runlist — **unless the bit
is already in the requested state, in which case it returns "no change" and does
nothing** (the `nvgpu_test_and_set_bit` / `nvgpu_test_and_clear_bit` early-outs,
and the `if (!update) return 0;` in `nvgpu_runlist_update_locked`).

**GCAPS's private view.** A separate bitmap `curr_tsgs_in_rl` inside
`struct nvgpu_rl_ctrl`. GCAPS never reads `active_channels`; it tracks only its
own `tsg_running` / `tsg_pending` / `curr_tsgs_in_rl`.

**The normal nvgpu path mutates `active_channels` without GCAPS's knowledge.**
`nvgpu_channel_enable/disable` calls
`g->ops.runlist.update(g, runlist, ch, add, true)` (`common/fifo/channel.c`),
which is exactly the submit-time runlist construction the GCAPS paper itself
describes ("as commands are submitted, TSG entries are added to the runlist",
ECRTS 2024 §2). Channel recovery/abort does the same via
`nvgpu_runlist_reload_ids` (`common/fifo/channel.c`). The GCAPS patch only
modifies `os/linux/ioctl_ctrl.c`, `os/linux/sched.{c,h}` and the UAPI header — it
never hooks `channel.c`, so it cannot observe these changes.

**The fatal skip.** The GCAPS handler gates the entire hardware update on its
private cache (`nvgpu_ioctl_runlist_update_rt_prio`, `os/linux/ioctl_ctrl.c`):

```c
*new_tsg_in_rl |= (*rl_ctrl->tsg_running);

/* update runlist */
if (*new_tsg_in_rl != *rl_ctrl->curr_tsgs_in_rl) {   /* <-- skip if cache "matches" */
    for (i = 0; i < f->num_channels; i++) {
        ... g->ops.runlist.update(g, tsg->runlist, ch, add, true); ...
    }
    *rl_ctrl->curr_tsgs_in_rl = *new_tsg_in_rl;
}
```

When the normal path has changed `active_channels` since GCAPS last ran,
`curr_tsgs_in_rl` is stale. If GCAPS's next request then computes
`new_tsg_in_rl == curr_tsgs_in_rl`, it **skips the loop entirely** — even though
the real runlist no longer matches `new_tsg_in_rl`. The intended TSG is never
(re)added, its kernel is never dispatched, and the task hangs on its fence.

This explains the highest-priority task hanging specifically: GCAPS believes the
runlist is already "just the top task" (`curr` = `{hist_16M} | exceptions`), but
the normal path has quietly dropped that TSG; the skip then means nothing ever
puts it back.

## Why it is duration-dependent

The divergence requires a particular interleaving — a normal-path runlist write
landing between two GCAPS ioctls such that GCAPS's next `add` yields
`new == stale-curr`. That is rare per GPU segment, so the probability of hitting
it accumulates with the number of segments executed. Short runs usually survive;
longer runs reliably hit it.

It is also far more likely on this hardware than on the authors': sub-millisecond
kernels on a 1.3 GHz AGX Orin churn the add/remove cycle orders of magnitude
faster than the 10–44 ms case-study segments on Jetson Xavier NX / Orin Nano, so
the racy window is hit far more often.

## Why `dmesg` is clean

The driver's consistency asserts guard a *different* invariant. The
"`running and pending tsgs are not exclusive`" warning checks
`tsg_running ∩ tsg_pending`, and "`this must be reported`" guards the
`pid_next == caller` removal case. Both bitmaps stay internally consistent here —
the inconsistency is between GCAPS's `curr_tsgs_in_rl` and the nvgpu-owned
`active_channels`, which nothing checks.

## Relation to the best-effort bug

Independent. The best-effort bug
([`best-effort-tasks-bug.md`](best-effort-tasks-bug.md)) is a deterministic
lost-update (`task_running = task_pending`) triggered by >1 concurrently-running
best-effort GPU task, and is avoided by running all GPU tasks as real-time. This
cache-desync bug remains even with all tasks real-time, because it is about
GCAPS's private runlist cache diverging from the driver's authoritative state.

## How it was found

Running `workloadTasksetGcaps -i 1 -s 1` with all tasks real-time: `-d 10`
completes; `-d 20` hangs with the highest-priority task stuck in a GPU fence
ioctl, others exited, `dmesg` clean. Reading the deployed R35.6 nvgpu source
(`common/fifo/runlist.c`, `common/fifo/channel.c`) confirmed `active_channels` is
the authoritative runlist membership, is written by the normal channel path
independently of GCAPS, and is never reconciled with `curr_tsgs_in_rl`.

## Fix direction (driver)

Not applied here. The cache must not be allowed to suppress a needed update:

- **Simplest:** drop the `if (*new_tsg_in_rl != *rl_ctrl->curr_tsgs_in_rl)`
  skip-optimization and reconcile the runlist on every ioctl (the per-channel
  `runlist.update` is already a no-op when a bit is unchanged, so correctness is
  restored at the cost of a full per-channel scan each call).
- **Better:** derive the "intended" set from the authoritative
  `domain->active_channels` / `active_tsgs` each time instead of keeping a
  private shadow copy, so the two can never diverge.
- **Or:** invalidate / refresh `curr_tsgs_in_rl` whenever the normal path changes
  the runlist (would require hooking the channel enable/disable path).

Workaround for data collection without a driver change: use the longest duration
that reliably completes (bisect from `-d 10`) and/or aggregate several short runs.
