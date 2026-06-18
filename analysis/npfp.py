"""
Non-Preemptive Fixed-Priority (NP-FP) GPU scheduler — response-time analysis.

Models the on-GPU CUDA-graph tail-launch real-time scheduler (`singleTaskSched`)
running on a Jetson Orin (JetPack 6.2).  Each task's GPU work is submitted as an
ordered chain of non-preemptible segments; once a segment begins executing on the
GPU it runs to completion, and a higher-priority task may only be dispatched at a
segment boundary.  This is exactly the uniprocessor Fixed-Preemption-Point (FPP)
model of Buttazzo, Bertogna & Yao [1, Sec. VII], with the GPU as the single serial
resource.  See msc-thesis-on-gpu-sched/singleTaskSched/docs/response_time_analysis.md.

This RTA is a HYBRID of that single-resource FPP model and the CPU-side terms of
the GCAPS analysis (gcaps.py), per design decisions taken for this codebase:

  GPU side (from the FPP doc, treating the GPU as one serial resource)
  --------------------------------------------------------------------
  * GPU occupancy is the kernel-execution portion Ge of each segment only.  The
    dispatched segment graphs are kernel-only and the memory-copy/staging Gm is
    host-resident work done off the serialized GPU path (Model 1, confirmed in
    singleTaskSched: the host allocates and stages a sequence's memory; segment
    graphs contain only kernel nodes).  Gm is therefore charged on the CPU side,
    NOT in any GPU term.
  * Dispatch-inclusive segment WCET: each segment's effective time folds in one
    flat per-dispatch overhead epsilon+rho (SCHEDULING_OVERHEAD), so the overhead
    scales with SEGMENT count, not job count [doc "Definitions"].
        q_eff(i,k) = ges[k].Ge_hi + SCHEDULING_OVERHEAD
        C_i^G      = Ge_hi + n_seg * SCHEDULING_OVERHEAD
  * Segment-level blocking B_i = longest lower-priority GPU SEGMENT (q^max, Ge-
    based), not a whole lower-priority job [doc Eq. 23].  The integer-model "-1"
    of Eq. 23 is dropped because times here are continuous (epsilon-unit) floats.
  * Self-pushing active period [doc Eq. 24-25, Sec. VII-A]: all jobs k in the
    level-i active period are checked and the response is the maximum over them:
        s_{i,k} = B_i + k*C_i_own - q_i^last + (interference, carry-in form)
        f_{i,k} = s_{i,k} + q_i^last
        R_i     = max_{k in [1,K_i]} { f_{i,k} - (k-1)*T_i }
  * Higher-priority GPU interference is charged with the doc's carry-in count
    (floor(s/T_h)+1) over ALL higher-GPU-priority GPU tasks regardless of core
    (one shared GPU).

  Decoupled CPU / GPU priority
  ----------------------------
  * A task carries two fixed priorities that need not match: the CPU priority
    (Task.prio, used for the same-core CPU term and the hpp_set membership) and
    the GPU priority (Task.prio_gpu, used for GPU blocking, GPU interference and
    the self-pushing horizon).  Larger == higher for both.  prio_gpu defaults to
    prio (RM) at generation; assigning a different prio_gpu reshapes only the GPU
    terms.  A CPU-higher-but-GPU-lower task therefore contributes CPU preemption
    yet acts as a GPU blocker, and vice versa -- captured directly here.

  CPU side (retained from the GCAPS analysis)
  -------------------------------------------
  * Same-core higher-priority tasks (hpp_set) preempt tau_i's CPU thread.  This
    term is kept with the busy-wait / self-suspend split:
        busy    : C_h + G_h     (CPU spins through the whole GPU phase)
        suspend : C_h + Gm_h    (CPU sleeps during pure GPU execution Ge)
  * A same-core higher-priority GPU task is charged on BOTH resources, but each
    component once: its CPU/staging part (C_h + Gm_h, or C_h + G_h busy) in the
    CPU term, and its kernel execution (Ge_h) in the GPU interference sum.  The
    CPU core and the GPU are distinct resources, so both delays are real; because
    Gm is GPU-excluded under Model 1, there is no Gm double-count (this matches
    the GCAPS treatment).

  Modelling assumption (flagged, revisitable)
  -------------------------------------------
  * The level-i active period that fixes K_i is computed from GPU resource demand
    only.  Self-pushing is the GPU non-preemption phenomenon the doc models; CPU
    interference is layered into each job's response but does not grow K_i.  A
    purely CPU-bound task therefore has K_i = 1 (CPU is preemptive: no self-push).

[1] Buttazzo, Bertogna, Yao, "Limited Preemptive Scheduling for Real-Time
    Systems: A Survey," IEEE Trans. Industrial Informatics, 9(1):3-15, 2013.
"""

from typing import List, Optional, Tuple
import math

from sched_common import *

# ---------------------------------------------------------------------------
# Normalised per-DISPATCH (per-segment) overhead epsilon+rho of the CUDA-graph
# tail-launch scheduler.
#
# Measured worst-case on Jetson Orin (JetPack 6.2): ~155 µs.
# The epsilon unit used throughout this codebase is normalised to the GCAPS
# overhead on an Orin Nano: ~1149 µs.  Ratio: 155 / 1149 ≈ 0.135.
#
# In the FPP model this is folded into EACH segment's effective WCET, so it
# scales with segment count (see module docstring).
#
# To update once you have better empirical measurements from your board:
#   1. Measure the worst-case dispatch latency D_measured (in µs).
#   2. Measure (or keep) the epsilon baseline E_baseline (in µs).
#   3. Set SCHEDULING_OVERHEAD = D_measured / E_baseline.
# A single change here propagates correctly to all tasksets and experiments.
# ---------------------------------------------------------------------------
SCHEDULING_OVERHEAD = 0.135

# Generous upper bound on a feasible level-i active period (epsilon units).
# Task periods in this codebase are <= ~500, so a busy period that grows past
# this means the GPU is over-utilised over the window -> unschedulable.
_ACTIVE_PERIOD_CAP = 1e9


def _gpu_params(task: Task) -> Tuple[float, float, float, int]:
    """
    Dispatch-inclusive FPP parameters for one task's GPU segments.

    GPU occupancy is the kernel-execution portion Ge of each segment only: the
    dispatched segment graphs are kernel-only, and the memory-copy/staging (Gm)
    is host-resident work done off the serialized GPU path (Model 1; see module
    docstring).  Gm is charged on the CPU side, not here.

    @return (C_G, q_last, q_max, n_seg)
        C_G    = sum of effective segment WCETs = Ge_hi + n_seg*SCHEDULING_OVERHEAD
        q_last = effective WCET of the final (run-to-completion) segment
        q_max  = effective WCET of the longest segment
        n_seg  = number of GPU segments (0 for a CPU-only task)
    """
    n = len(task.ges)
    if n == 0:
        return 0.0, 0.0, 0.0, 0
    seg_eff = [g.Ge_hi + SCHEDULING_OVERHEAD for g in task.ges]
    return sum(seg_eff), seg_eff[-1], max(seg_eff), n


def _num_jobs(tasks: List[Task], gp, i: int, B_i: float, uses_gpu_i: bool,
              hi_gpu: List[int]) -> Optional[int]:
    """
    Number of jobs K_i of tau_i in the longest level-i active period
    (the self-pushing horizon).  Computed from GPU resource demand only.

    @param hi_gpu  indices of GPU-using tasks with higher GPU priority than tau_i
    @return K_i (>= 1), or None if the GPU active period diverges (over-utilised).
    """
    if not uses_gpu_i:
        return 1  # CPU is preemptive: no self-pushing for a CPU-only task

    C_G_i = gp[i][0]
    T_i = tasks[i].T

    L = B_i + C_G_i
    while True:
        nxt = B_i
        nxt += math.ceil(L / T_i) * C_G_i                 # tau_i itself
        for h in hi_gpu:                                  # higher-GPU-priority tasks
            nxt += math.ceil(L / tasks[h].T) * gp[h][0]
        if nxt == L:
            break
        if nxt > _ACTIVE_PERIOD_CAP:
            return None  # GPU over-utilised over the window
        L = nxt

    return max(1, math.ceil(L / T_i))


def _finish_time(tasks: List[Task], gp, i: int, k: int,
                 B_i: float, C_own_i: float, q_last_i: float,
                 uses_gpu_i: bool, hi_gpu: List[int], no_suspension: bool) -> Optional[float]:
    """
    Finishing time f_{i,k} of the k-th job of tau_i in the level-i active period
    (doc Eq. 24-25), extended with the retained CPU same-core interference term.

    @param hi_gpu  indices of GPU-using tasks with higher GPU priority than tau_i
    Least-fixed-point iteration of the start time s_{i,k}; returns
    f_{i,k} = s_{i,k} + q_last_i, or None if the recurrence cannot converge
    below tau_i's deadline (deadline miss / divergence).
    """
    T_i = tasks[i].T
    D_i = T_i  # constrained deadline

    base = B_i + k * C_own_i - q_last_i
    s = base
    while True:
        nxt = base

        # GPU interference: all higher-GPU-PRIORITY tasks (shared GPU, any core),
        # counted with the doc's carry-in form (floor(s/T_h)+1).  Only tau_i tasks
        # that actually use the GPU are subject to GPU contention.  hi_gpu uses the
        # task's GPU priority (prio_gpu), which may differ from its CPU priority.
        if uses_gpu_i:
            for h in hi_gpu:
                nxt += (math.floor(s / tasks[h].T) + 1) * gp[h][0]

        # CPU interference: same-core higher-CPU-PRIORITY tasks (hpp_set) preempt
        # the CPU thread.  A same-core task is charged here AS WELL AS in the GPU
        # sum above when it also has higher GPU priority (distinct resources, and
        # CPU vs GPU priority are now independent -> both delays counted).
        for h in tasks[i].hpp_set:
            if no_suspension:
                cpu_dem = tasks[h].C_hi + tasks[h].G_hi    # busy-wait: spins through G
            else:
                cpu_dem = tasks[h].C_hi + tasks[h].Gm_hi   # suspend: only C + memcpy
            nxt += (math.floor(s / tasks[h].T) + 1) * cpu_dem

        if nxt == s:
            return s + q_last_i

        s = nxt
        # Monotone-increasing; bail as soon as this job's response exceeds D_i.
        if s + q_last_i - (k - 1) * T_i > D_i:
            return None


def rt_test(tasks: List[Task], no_suspension: bool) -> bool:
    """
    Hybrid FPP (GPU) + CPU-interference response-time test for the NP-FP
    on-GPU sequence scheduler.  See module docstring for the full model.

    CPU priority is taken from the list order: hpp_set holds the absolute indices
    of higher-CPU-priority same-core tasks (taskset_generation() RM-sorts the list
    so index 0 = highest CPU priority).  GPU contention is keyed independently on
    Task.prio_gpu (computed inline, not from list order), so GPU priority may
    differ from CPU priority.

    @param tasks         RM-sorted list from taskset_generation()
    @param no_suspension True  -> busy-wait: CPU thread spins during GPU exec
                         False -> self-suspend: CPU releases during pure GPU phase
    @return True if every RT task (prio >= 0) meets its deadline; False on miss
    """
    gp = [_gpu_params(t) for t in tasks]

    for i in range(len(tasks)):
        if tasks[i].prio == -1:
            continue  # best-effort tasks are not subject to RT analysis

        C_G_i, q_last_i, q_max_i, n_i = gp[i]
        uses_gpu_i = n_i > 0

        # Own per-job demand that must complete before the final GPU segment
        # starts: CPU work, the host-side memcpy/staging (Gm), and the
        # dispatch-inclusive GPU kernel (Ge) segments.  Numerically this is
        # C_hi + G_hi + n_seg*overhead -- the job still performs its copy; only
        # the GPU-resource occupancy C_G_i is Ge-based (Model 1).
        C_own_i = tasks[i].C_hi + tasks[i].Gm_hi + C_G_i

        # -------------------------------------------------------------------
        # GPU contention is keyed on the GPU priority (prio_gpu), which may
        # differ from the CPU priority.  Larger prio_gpu == higher priority.
        # GPU priorities of GPU-contending RT tasks are assumed distinct (RM by
        # default gives distinct values); ties use strict comparison.  Off-core
        # tasks still count: the GPU is one shared resource.
        # -------------------------------------------------------------------
        hi_gpu, lo_gpu = [], []
        if uses_gpu_i:
            for h in range(len(tasks)):
                if h == i or gp[h][3] == 0:
                    continue  # self, or a CPU-only task that never touches the GPU
                if tasks[h].prio_gpu > tasks[i].prio_gpu:
                    hi_gpu.append(h)
                elif tasks[h].prio_gpu < tasks[i].prio_gpu:
                    lo_gpu.append(h)

        # -------------------------------------------------------------------
        # Non-preemptive GPU blocking B_i (doc Eq. 23): the longest single
        # lower-GPU-priority GPU SEGMENT.  Only meaningful if tau_i uses the GPU;
        # a CPU-only task never waits for the GPU.  (best-effort GPU tasks have
        # prio_gpu = -1, so they fall into lo_gpu and can still block.)
        # -------------------------------------------------------------------
        B_i = 0.0
        if uses_gpu_i and lo_gpu:
            B_i = max(gp[l][2] for l in lo_gpu)

        # -------------------------------------------------------------------
        # Self-pushing: number of jobs K_i in the level-i active period.
        # -------------------------------------------------------------------
        K_i = _num_jobs(tasks, gp, i, B_i, uses_gpu_i, hi_gpu)
        if K_i is None:
            tasks[i].WR = -1
            return False  # GPU over-utilised -> unschedulable

        # -------------------------------------------------------------------
        # Check every job in the active period; R_i is the maximum response.
        # -------------------------------------------------------------------
        WR_max = 0.0
        for k in range(1, K_i + 1):
            f = _finish_time(tasks, gp, i, k, B_i, C_own_i, q_last_i,
                             uses_gpu_i, hi_gpu, no_suspension)
            if f is None:
                tasks[i].WR = -1
                return False  # deadline missed for some job in the active period

            R = f - (k - 1) * tasks[i].T
            if R > tasks[i].T:
                tasks[i].WR = -1
                return False

            if R > WR_max:
                WR_max = R

        tasks[i].WR = WR_max  # store the worst-case response time

    return True
