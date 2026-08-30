# GCAPS preempted-caller runlist overwrite deadlock

## Summary

GCAPS deadlocks when a task that has **already been preempted** issues its
segment-end request while a **third task is pending**. The remove logic
overwrites the whole running set with the next pending task, discarding the
higher-priority task that had just preempted the caller. That task's TSG is
dropped from the hardware runlist **while its kernel is in flight**; the fence
never signals, and its `gcapsGpuSegEnd()` (ending in `cudaEventSynchronize`)
blocks forever.

This is the same lost-update class as
[`best-effort-tasks-bug.md`](best-effort-tasks-bug.md) — a wholesale assignment
to `tsg_running` where a set difference was meant — but in the **sibling
branch** of the same `if`. The consequence is that the documented workaround for
the best-effort bug (*run every GPU task as real-time*) **does not close this
one**. It is also distinct from
[`runlist-cache-desync-bug.md`](runlist-cache-desync-bug.md): no update is
skipped here, every event in the failing window carries `rlupd=1`, and the
failure is fully deterministic rather than duration-dependent.

Present in the published R35 patch, and carried forward into the JetPack 7.2
port.

## Symptom

On Jetson AGX Orin (L4T R35.6.4) running `workloadTasksetGcaps -i 1 -s 1` with
all eight tasks real-time, railgating disabled:

- the parent blocks in `wait4` on **one** child; every other child completes its
  full measurement window and exits cleanly;
- the stuck child is the **highest-priority** task (`hist_128M`, GCAPS priority
  8, period 100 ms), parked in `cudaEventSynchronize` on a kernel that never
  runs;
- it dies on its **26th segment, at t+2.49 s**, identically at `-d 10` and
  `-d 20` and across reboots — 26 `add=1` against 25 `add=0`, every other task
  perfectly balanced;
- the GPU is **not** wedged: the other seven tasks run their full windows at
  their exact expected job counts, and their trace fragments are written
  normally;
- `dmesg` is clean — the driver's own asserts guard a different invariant.

Before this was understood, three *additional* tasks appeared stuck. They were
not: `workloadTasksetGcaps` serialises its post-run verification by GCAPS
priority, and `mlp_1024x8` holds its segment for ~a minute doing a host-side
reference pass, so priorities 3, 2 and 1 queue behind it. Any timeout landing
inside that tail reports them as hangs. Allow 180 s+ for a 20 s run.

## Root cause

`gcaps_driver_patch/ioctl_ctrl.c.patch`, function
`nvgpu_ioctl_runlist_update_rt_prio`, the "caller requests to be removed"
branch, sub-case "a pending real-time task exists":

```c
} else { /* the caller task requests to be removed */
    nvgpu_get_tsgs_with_highest_prio_locked(g, &pid_next, &rt_prio_next, tsgs_next);
    if (pid_next > 0) {
        if (pid_next == cpid) {           /* case 1: caller is the next task */
            ...
        } else {
            task_next = pid_task(find_vpid(pid_next), PIDTYPE_PID);
            resumed_pid = pid_next;
            /* found the next rt task of #op_type, add ONLY this task to runlists */
            *rl_ctrl->tsg_running = *tsgs_next;          /* <-- BUG: overwrite */
            *rl_ctrl->tsg_pending &= ~(*tsgs_next);
            ...
        }
    } else { /* no pending rt task */
        ...
    }
}
```

The assignment encodes the assumption **"the caller is the task currently
running, so replacing the running set removes the caller and admits the next"**.
That is true only when the caller was never preempted.

When a higher-priority task has already evicted the caller, `tsg_running` holds
the **preemptor**, not the caller — and line 222 throws it away. The preemptor is
still executing; it is simply erased from GCAPS's model of the runlist, and the
subsequent rebuild removes its channels from the hardware runlist.

A second lost update sits in the same branch: the caller clears `tsgs_next` from
`tsg_pending`, never `tsgs_cpid`, so a preempted caller's own pending bit
survives its own removal.

### The guarded sibling

The `else` branch immediately below (`no pending rt task`) performs the same kind
of wholesale assignment, but **only after checking that no real-time task is
still running**:

```c
for (i = 0; i < f->num_channels; i++) {
    if (nvgpu_test_bit(i, sched->rl_ctrl.tsg_running)) {
        if (task && (task->rt_priority <= 99 && task->rt_priority >= 1)) {
            exist = true;                     /* a RT task IS still running */
            break;
        }
    }
}
if (exist == false)                           /* only then is the overwrite safe */
    *rl_ctrl->tsg_running = *rl_ctrl->tsg_pending;
```

So the invariant *was* recognised — it is enforced in one path and omitted in
its sibling. (That guard is also exactly what the best-effort bug defeats, by
making `exist` false while best-effort tasks are still running.)

## Observed trace

Kernel events around the fatal segment (`GCAPS_EV`, times relative to the
overwrite; pid 4934 = `hist_128M` prio 8, 4937 = `conv_4096_k7` prio 2,
4941 = `mlp_1024x8` prio 3, 4939 = `mm_2560` prio 4):

```
 -4.57 ms  cpid=4939 prio=4 add=0 rlupd=1  resumed=4937     mm_2560 ends, conv_k7 resumed
 +0.00 ms  cpid=4934 prio=8 add=1 rlupd=1  preempted=4937   hist_128M arrives, EVICTS conv_k7
                                                            => tsg_running={4934}
                                                               tsg_pending={4937}
 +0.14 ms  cpid=4941 prio=3 add=1 rlupd=0                   mlp arrives, pended
                                                            => tsg_pending={4937,4941}
 +1.30 ms  cpid=4937 prio=2 add=0 rlupd=1  resumed=4941     conv_k7 segment-end:
                                                               pid_next=4941 != cpid=4937
                                                               line 222: tsg_running={4941}
                                                               *** 4934 SILENTLY DROPPED ***
```

`new_tsg_in_rl` is then built from `tsg_running` = `{4941}`, differs from
`curr_tsgs_in_rl`, and the runlist is rebuilt without `hist_128M`. No further
event for pid 4934 is ever logged.

Note `conv_4096_k7` reaches its segment-end *while preempted* because its GPU
work had already completed; the eviction landed in the window between fence
completion and the remove ioctl — about 1.3 ms here.

## Why it always hits the highest-priority task

Only a task that **preempts** someone can be the victim, because the preemptor is
what occupies `tsg_running` when the preempted task's remove arrives. The
highest-priority task is by definition the one that preempts most often, and
`hist_128M` also has the shortest period (100 ms), so it is preempting almost
continuously. The victim is therefore essentially always the top-priority task.

`runlist-cache-desync-bug.md` records the same observation — highest-priority
task stuck on a fence, board healthy, `dmesg` clean — and attributes it to the
`curr_tsgs_in_rl` cache skip. The observation was right; the attribution was not.
Every event in the failing window has `rlupd=1`, so no update was ever skipped.

## Why it is deterministic, not duration-dependent

The three preconditions below are all functions of the periodic task phasing,
which is fixed. The failure therefore recurs at the same job index every run —
segment 26, t+2.49 s — and `-d 10` and `-d 20` fail *identically*. There is no
duration ceiling and no bisectable `-d` threshold.

Preconditions:

1. task A is running and is preempted by a higher-priority B;
2. a third task C is pending at that instant, so `pid_next > 0` and
   `pid_next != A` — otherwise control reaches the **guarded** sibling branch;
3. A's GPU work has already completed, so its host thread reaches
   `gcapsGpuSegEnd` while still preempted (a window of roughly the fence-wake
   latency).

## Why `./main` does not hit it

`main -f taskset.csv -d 30 -i 1` completes cleanly (1650 events, 825 `add=1` /
825 `add=0`). That is exposure, not immunity — condition 1 is simply far rarer:

| binary | events | real evictions | rate |
|---|---|---|---|
| `main`, 6 tasks, `-d 10` | 548 | 10 | 1.8 % |
| `workloadTasksetGcaps`, `-d 20` | 1221 | 192 | 15.7 % |

`main`'s segments rarely overlap — 158 of its 168 add events report
`preempted=0`, i.e. they arrived to an idle runlist. `workloadTasksetGcaps` is
deliberately contended (GCAPS Table 4 structure with real workloads), so
preemptions are ~9x denser. Combined with the narrow race in condition 3, that is
the whole difference.

**A long enough `main` run, or a denser taskset, should eventually hit the same
wall.** Clean `main` results mean the bug is rare at that contention level, not
that the code is correct.

## Fix

A preempted caller's segment-end says nothing about who should run next: the
preemptor still holds the GPU and outranks every pending task. Remove only the
caller's own bits and leave the running set alone. Otherwise, remove the caller
and admit the next task as a set *difference and union*, never an assignment:

```c
} else if (!bitmap_intersects(rl_ctrl->tsg_running, tsgs_cpid, nbits)) {
        /* Caller was already preempted.  The preemptor is still running and
         * outranks everything pending -- drop our bits, touch nothing else. */
        bitmap_andnot(rl_ctrl->tsg_pending, rl_ctrl->tsg_pending, tsgs_cpid, nbits);
} else {
        /* Caller really was the running task: remove IT, admit the next. */
        task_next   = pid_task(find_vpid(pid_next), PIDTYPE_PID);
        resumed_pid = pid_next;
        bitmap_andnot(rl_ctrl->tsg_running, rl_ctrl->tsg_running, tsgs_cpid, nbits);
        bitmap_andnot(rl_ctrl->tsg_pending, rl_ctrl->tsg_pending, tsgs_cpid, nbits);
        bitmap_or    (rl_ctrl->tsg_running, rl_ctrl->tsg_running, tsgs_next, nbits);
        bitmap_andnot(rl_ctrl->tsg_pending, rl_ctrl->tsg_pending, tsgs_next, nbits);
        rt_task_in_rl->pid     = pid_next;
        rt_task_in_rl->rt_prio = rt_prio_next;
}
```

Two notes on the form:

- `nbits = f->num_channels`. The bitmaps are allocated multi-word
  (`sched->bitmap_size`) and bits are set and tested per channel across **all**
  words (`nvgpu_test_bit(i, ...)`, `i < f->num_channels`), but every set
  operation in the published patch is `*ptr` — word 0 only. Any TSG at index
  >= 64 is therefore invisible to the set algebra. That truncation is **latent**
  on this board (the failure reproduces bit-for-bit on a freshly booted, quiet
  GPU, so TSG-id allocation is not a factor), but the fix above needs
  `bitmap_intersects` anyway, so converting the whole function at once is the
  natural move. The same ~23-statement conversion is already applied in
  `gcaps_driver_patch/jp72/apply_gcaps_jp72.py`.
- The wholesale assignment at the `exist == false` site (the best-effort bug)
  wants the same treatment; both are the same defect in two places.

## Reproduction

```bash
cd ~/GCAPS/gcaps-super-repo/gcaps_userspace
sudo GCAPS_MAIN=$PWD/workloadTasksetGcaps \
     ~/GCAPS/run_gcaps_r35.sh --timeout 180 -i 1 -s 1 -d 20
```

Expect: seven tasks reach `post-run verify PASS`; `hist_128M` never announces
verification and the run is killed by the cap. Confirm with the per-pid balance
over the whole boot log — exactly one pid, the priority-8 one, has an unmatched
`add=1`:

```bash
E=$(journalctl -k -b 0 --no-pager | grep GCAPS_EV)
for p in $(echo "$E" | grep -oE 'cpid=[0-9]+' | cut -d= -f2 | sort -un); do
  a1=$(echo "$E" | grep -c "cpid=$p .*add=1")
  a0=$(echo "$E" | grep -c "cpid=$p .*add=0")
  pr=$(echo "$E" | grep -m1 -oE "cpid=$p prio=[0-9]+" | cut -d= -f3)
  printf "pid %-7s prio %-3s add1=%-5s add0=%-5s unmatched=%s\n" $p $pr $a1 $a0 $((a1-a0))
done
```

Do **not** count events with a `tail -N` window: clipping the opening `add=1`
makes the stuck task appear balanced. Count over the whole boot.

Railgating must be off for any of this — see the missing `gk20a_busy()` power
reference documented in
`msc-thesis-on-gpu-sched/singleTaskSched/docs/gcaps_build_procedure_jp72.md`;
with railgating on, a six-task run wedges the board and the watchdog resets it.

## Relation to the other two bugs

| | trigger | mechanism | workaround |
|---|---|---|---|
| `best-effort-tasks-bug.md` | >1 concurrent macro-instrumented best-effort GPU task | `tsg_running = tsg_pending` at the `exist == false` site | run all GPU tasks real-time |
| **this bug** | preempted caller's segment-end with a third task pending | `tsg_running = tsgs_next` in the sibling branch | none — all-real-time does not help |
| `runlist-cache-desync-bug.md` | normal-path runlist write between two GCAPS ioctls | `curr_tsgs_in_rl` skip suppresses a needed update | none proposed |

The first two are the same defect in two adjacent branches. The third remains a
genuine design-level concern (a private shadow copy of a concurrently-mutated
resource, used to suppress updates, with no invalidation), but it is **not** what
produces the `workloadTasksetGcaps` hang: that failure is fully explained by the
overwrite above, and the doc's duration-dependence claim does not reproduce.
