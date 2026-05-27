"""
Non-Preemptive Fixed-Priority (NP-FP) GPU scheduler — response-time analysis.

Models the on-GPU CUDA graph tail-launch real-time scheduler running on a
Jetson Orin (JetPack 6.2).  Each task submits its entire GPU work as one
non-preemptive CUDA graph; once a graph begins executing on the GPU it runs
to completion before any other task's graph can start.

Compared to GCAPS (gcaps.py):

  1. scheduling_overhead replaces GCAPS's per-segment (epsilon + 3*epsilon*|ges|)
     cost.  The CUDA graph tail-launch dispatcher has a single flat per-job
     overhead regardless of how many GPU segments the task contains.

  2. Non-preemptive blocking B_i: τi may arrive while a lower-priority GPU
     job is already executing.  That job cannot be interrupted, so τi waits
     for it to finish.  GCAPS (preemptive GPU) has no such term.

  3. CPU interference (hpp_set) does NOT add 2*epsilon per segment because
     there is no kernel-context switch overhead mid-job.

  4. GPU interference (hp_set ∖ hpp_set) charges full G_hi per occurrence —
     the entire GPU graph runs without interruption — whereas GCAPS charges
     only the last segment Ge_hi in its carry-in model.

  5. The jitter for off-core GPU interference is J_h = WR_h − G_hi (time from
     release until the GPU section starts), not WR_h − Ge_hi as in GCAPS.
"""

from typing import List
import math

from sched_common import *

# ---------------------------------------------------------------------------
# Normalised per-job dispatch overhead of the CUDA graph tail-launch scheduler.
#
# Measured worst-case on Jetson Orin (JetPack 6.2): ~155 µs.
# The epsilon unit used throughout this codebase is normalised to the GCAPS
# overhead on an Orin Nano: ~1149 µs.  Ratio: 155 / 1149 ≈ 0.135.
#
# To update once you have better empirical measurements from your board:
#   1. Measure the worst-case dispatch latency D_measured (in µs).
#   2. Measure (or keep) the epsilon baseline E_baseline (in µs).
#   3. Set SCHEDULING_OVERHEAD = D_measured / E_baseline.
# A single change here propagates correctly to all tasksets and experiments.
# ---------------------------------------------------------------------------
SCHEDULING_OVERHEAD = 0.135


def rt_test(tasks: List[Task], no_suspension: bool) -> bool:
    """
    Fixed-point RTA for NP-FP GPU scheduling.

    Tasks MUST be sorted in RM order (index 0 = shortest period = highest
    priority), which is the invariant maintained by taskset_generation().
    Higher-priority tasks' WR values are read as carry-in jitter in the
    GPU-interference term, so the outer loop processes tasks in index order;
    by the time task i is analysed, all tasks in hp_set (index < i) already
    have valid WR values.

    The WCRT bound for task τi is:

        WR = C_hi + G_hi
           + SCHEDULING_OVERHEAD          (flat per-job dispatch cost)
           + B_i                          (NP blocking by one lower-prio GPU job)
           + CPU_interference             (hpp_set: same-core higher-prio tasks)
           + GPU_interference             (hp_set ∖ hpp_set: off-core GPU tasks)

    @param tasks         RM-sorted list from taskset_generation()
    @param no_suspension True  → busy-wait: CPU thread spins during GPU exec
                         False → self-suspend: CPU releases during pure GPU phase
    @return True if every RT task (prio ≥ 0) meets its deadline; False on miss
    """
    for i in range(len(tasks)):
        if tasks[i].prio == -1:
            continue  # best-effort tasks are not subject to RT analysis

        # -------------------------------------------------------------------
        # Non-preemptive blocking  B_i
        # -------------------------------------------------------------------
        # τi may become the highest-priority ready task while a lower-priority
        # GPU job is already in flight.  NP scheduling means that job cannot
        # be evicted; τi must wait for it to finish.
        #
        # Worst case = the longest single lower-priority GPU job.  Only tasks
        # with G_hi > 0 can occupy the GPU.  Lower-priority tasks have indices
        # > i in RM-sorted order (shorter period → higher priority → lower index).
        lp_gpu_times = [tasks[l].G_hi
                        for l in range(i + 1, len(tasks))
                        if tasks[l].G_hi > 0]
        B_i = max(lp_gpu_times) if lp_gpu_times else 0

        WR_prev = 0
        WR_curr = 0

        while True:
            # ---------------------------------------------------------------
            # Base execution cost: one full job of τi (CPU + GPU work).
            # ---------------------------------------------------------------
            WR_curr = tasks[i].C_hi + tasks[i].G_hi

            # ---------------------------------------------------------------
            # Per-job dispatch overhead (CUDA graph tail-launch).
            # Flat cost per job, independent of GPU-segment count.
            # GCAPS charges epsilon + 3*epsilon*|ges|; NP-FP charges a single
            # constant because the graph is pre-compiled and submitted once.
            # ---------------------------------------------------------------
            WR_curr += SCHEDULING_OVERHEAD

            # ---------------------------------------------------------------
            # Non-preemptive blocking: at most one lower-priority GPU job.
            # ---------------------------------------------------------------
            WR_curr += B_i

            # ---------------------------------------------------------------
            # CPU interference from same-core higher-priority tasks (hpp_set)
            # ---------------------------------------------------------------
            # hpp_set holds absolute task indices of higher-priority tasks that
            # share the same CPU core as τi.  These tasks can preempt τi's CPU
            # thread.  The number of preemptions within τi's busy window of
            # length WR_prev is bounded by ceil(WR_prev / T_h).
            for h in tasks[i].hpp_set:
                if no_suspension:
                    # Busy-wait mode: the CPU thread spins during the entire
                    # execution of task h (both CPU and GPU portions), so the
                    # full C_h + G_h occupies the CPU during each preemption.
                    # CPU-only tasks (G_hi = 0) reduce to just C_h naturally.
                    WR_curr += math.ceil(WR_prev / tasks[h].T) * \
                               (tasks[h].C_hi + tasks[h].G_hi)
                else:
                    # Self-suspend mode: the CPU thread suspends during the
                    # pure-GPU phase of task h.  Only the CPU-bound portion
                    # C_h + Gm_h (memory copy) charges the CPU; the remaining
                    # Ge_h = G_h − Gm_h executes off the CPU critical path.
                    # CPU-only tasks (G_hi = Gm_hi = 0) reduce to just C_h.
                    # Note: GCAPS adds 2*epsilon*|ges| on top; NP-FP does not
                    # because there are no per-segment context-switch costs.
                    WR_curr += math.ceil(WR_prev / tasks[h].T) * \
                               (tasks[h].C_hi + tasks[h].Gm_hi)

            # ---------------------------------------------------------------
            # GPU interference from off-core higher-priority GPU tasks
            # (hp_set ∖ hpp_set)
            # ---------------------------------------------------------------
            # hp_set holds all higher-priority tasks regardless of core.
            # Tasks in hpp_set are already counted in CPU interference above.
            # Off-core tasks do not preempt τi's CPU thread, but their GPU
            # jobs must complete before τi can acquire the GPU (NP property).
            #
            # The full G_hi is charged per occurrence — NP scheduling means
            # the entire GPU graph runs to completion without interruption.
            # GCAPS preemptive analysis charges only the last segment Ge_hi
            # since earlier segments may complete before τi's window opens;
            # that does not apply here.
            for h in tasks[i].hp_set:
                if h in tasks[i].hpp_set:
                    continue  # same-core task already counted in CPU term
                if tasks[h].G_hi == 0:
                    continue  # CPU-only task: never touches the GPU

                # Carry-in jitter J_h: time from task h's release until its
                # GPU section begins (the non-GPU portion of its response).
                # tasks[h].WR was computed during h's earlier iteration using
                # the same no_suspension flag, so it is internally consistent.
                # J_h = C_h + overhead_h + B_h + own_interference_h.
                J_h = tasks[h].WR - tasks[h].G_hi

                # ceil((WR_prev + J_h) / T_h): worst-case number of times
                # task h's GPU job falls within τi's busy period of length
                # WR_prev, accounting for the carry-in from a job that started
                # before τi's window opened.
                WR_curr += math.ceil((WR_prev + J_h) / tasks[h].T) * \
                           tasks[h].G_hi

            # ---------------------------------------------------------------
            # Fixed-point convergence check
            # ---------------------------------------------------------------
            # WR_curr is a non-decreasing function of WR_prev (ceil is
            # monotone), so the sequence is non-decreasing from the second
            # iteration onward.  Bail early if the deadline is already exceeded
            # since the bound can only grow.
            if WR_prev == WR_curr:
                if WR_curr <= tasks[i].T:
                    tasks[i].WR = WR_curr  # store for use as J in lower-prio tasks
                    break
                else:
                    tasks[i].WR = -1
                    return False  # deadline missed at fixed point
            elif WR_curr > tasks[i].T:
                tasks[i].WR = -1
                return False  # bound exceeded deadline; early exit

            WR_prev = WR_curr

    return True
