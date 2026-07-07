#!/usr/bin/env python3
"""
plot_workload_bench.py

Analysis and plotting for the GCAPS varied-workload benchmark suite — the
GCAPS analog of singleTaskSched's scripts/bench/plot_workload_bench.py.

  Sweep mode   — workloadSweepGcaps -i 1 / -i 0
                 (sweep_gcaps.csv, sweep_tsg.csv)
  Taskset mode — workloadTasksetGcaps -i 1 / -i 0
                 (taskset_gcaps_trace.csv, taskset_tsg_trace.csv)

The two compared series are GCAPS (ioctl, -i 1) and the default TSG
round-robin baseline (-i 0).  Inputs are read from --results-dir; whichever
CSVs are present are analysed, missing ones are skipped with a notice.

The per-release/per-period response decomposition is
  cpu_phase + sched_preempt_overhead + gpu_exec.
"sched_preempt_overhead" is response minus the CPU phase minus the measured
on-GPU execution window: kernel-launch and ioctl latency plus GPU-side
scheduling/preemption delay.  GCAPS is preemptive, so this is scheduling +
preemption overhead, not a FIFO queue wait.

Console output:
  - Per-configuration sweep summary (mean / p95 response, overhead),
    GCAPS vs TSG baseline
  - Per-task taskset summary (MORT / mean / misses), GCAPS vs TSG
  - Per-task worst-case GPU exec time C_i = max(gpu_exec) and the resulting
    taskset GPU utilization U = Σ C_i/T_i, one table per series

The driver-measured GCAPS ε (runlist-update overhead, α+θ — Def. 2 of the paper)
is read from the captured GCAPS_EV kernel log (taskset_gcaps_events.log, written
by measure_preempt_overhead.py --run taskset) when present. It drives a dedicated
epsilon.pdf and, attributed per release (the add+remove ioctls inside each GPU
segment), splits the red "overhead" band into ε (red) + other latency (grey) in
the breakdown and Gantt. Absent the log, those fall back to the single overhead
band and epsilon.pdf is skipped.

Figures written to --results-dir:
  sweep_response.pdf      — mean response per configuration, GCAPS vs TSG
                            (one panel per workload type, log y)
  sweep_overhead.pdf      — per-config sched+preempt overhead: response delta
                            (GCAPS - TSG) and GCAPS overhead
  epsilon.pdf             — driver-measured ε: elapsed_us histogram (no-op vs
                            runlist-reload modes, à la the paper's Fig. 12) and
                            per-task ε box plot  [needs taskset_gcaps_events.log]
  taskset_mort.pdf        — MORT and mean response per task, GCAPS vs TSG
  taskset_breakdown.pdf   — stacked mean cpu / overhead / gpu per task; overhead
                            split into ε + other when the event log is present
  taskset_response_overhead.pdf
                          — per-task box plots of the response overhead
                            (response − cpu − gpu = ε + launch + queuing/
                            interference), GPU tasks, GCAPS vs TSG
  taskset_gantt.pdf       — stacked GCAPS-vs-TSG execution Gantt over the
                            window [--gantt-start, +--gantt-duration]; each
                            period tiled CPU (green) | overhead (red, split into
                            ε + other when available) | GPU (purple); with the
                            event log the driver-reported suspended intervals are
                            carved out of the GPU block (olive), so preemptions
                            are visible and purple reads as GPU actively
                            executing; rows ordered by the trace's fifo_priority
                            (highest on top, CPU-only task last)

Usage:
    python3 scripts/plot_workload_bench.py \\
        [--results-dir results/workloadBench]
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

GCAPS_COLOR = '#1f77b4'
TSG_COLOR = '#ff7f0e'

# Per-phase colours — kept identical to the stacked breakdown bar chart
# (plot_taskset_breakdown) so the Gantt and the bars read the same.
CPU_COLOR = '#2ca02c'       # green  — CPU phase
OVERHEAD_COLOR = '#d62728'  # red    — scheduling + preemption overhead
GPU_COLOR = '#9467bd'       # purple — GPU execution

# Overhead is split into the driver-measured GCAPS epsilon (the runlist-update
# ioctl cost, alpha+theta) and the remaining latency (kernel launch + GPU-side
# scheduling/preemption delay) when the GCAPS_EV event log is available.
EPS_COLOR = '#d62728'       # red    — measured epsilon (runlist update)
OTHER_OVH_COLOR = '#9e9e9e' # grey   — other overhead (launch + GPU-side delay)

# The GPU (cudaEvent) wall window INCLUDES time the segment sat suspended while a
# higher-priority job ran (verified on AGX Orin: gpu_wall ≈ active + suspended).
# When the GCAPS_EV log is present the Gantt carves those driver-reported
# suspended intervals out of the purple GPU block so preemptions are visible and
# the purple then reads as GPU *actively executing* (matching the seq Gantt).
SUSPENDED_COLOR = '#bcbd22' # olive  — suspended (preempted, GPU idle for this job)

# One structured driver record per runlist-update ioctl (see the driver patch):
#   GCAPS_EV ts=<ns> cpid=<pid> prio=<n> add=<0|1> rlupd=<0|1> \
#            elapsed_us=<eps> preempted=<pid|-1> resumed=<pid|-1>
GCAPS_EV_RE = re.compile(
    r'GCAPS_EV\s+ts=(\d+)\s+cpid=(-?\d+)\s+prio=(-?\d+)\s+add=(\d+)\s+'
    r'rlupd=(\d+)\s+elapsed_us=(-?\d+)\s+preempted=(-?\d+)\s+resumed=(-?\d+)')

# Figure-width scaling for the (very wide) taskset Gantt, in inches per second
# of trace time (override with --gantt-inches-per-sec).
GANTT_INCHES_PER_S = 10.0
GANTT_MAX_WIDTH_IN = 600.0
GANTT_ROW_HEIGHT_IN = 1.05   # vertical inches allotted to each task row
GANTT_DPI = 200              # raster fallback / embedded-image quality


# ── Loading ─────────────────────────────────────────────────────────────────

def load_sweep(path: str) -> dict[str, dict[str, np.ndarray]] | None:
    """workload -> {type, overhead, exec, response}; preserves file order."""
    if not os.path.isfile(path):
        print(f'  [skip] {path} not found')
        return None
    rows = defaultdict(lambda: {'type': '', 'overhead': [], 'exec': [],
                                'response': []})
    order: list[str] = []
    with open(path) as f:
        for r in csv.DictReader(f):
            wl = r['workload']
            if wl not in rows:
                order.append(wl)
            rows[wl]['type'] = r['type']
            rows[wl]['overhead'].append(float(r['sched_preempt_overhead_ms']))
            rows[wl]['exec'].append(float(r['gpu_exec_ms']))
            rows[wl]['response'].append(float(r['response_ms']))
    return {wl: {'type': rows[wl]['type'],
                 'overhead': np.array(rows[wl]['overhead']),
                 'exec': np.array(rows[wl]['exec']),
                 'response': np.array(rows[wl]['response'])}
            for wl in order}


def load_taskset(path: str) -> dict[str, dict[str, np.ndarray]] | None:
    """task_name -> {start, cpu, overhead, gpu, response, deadline, missed,
    and (if the trace carries them) pid, seg_begin, seg_done, plus
    fifo_priority (int, the task's assigned SCHED_FIFO priority; None for
    older traces without the column)."""
    if not os.path.isfile(path):
        print(f'  [skip] {path} not found')
        return None
    rows = defaultdict(lambda: defaultdict(list))
    prio: dict[str, int | None] = {}
    order: list[str] = []
    have_pid = False
    with open(path) as f:
        for r in csv.DictReader(f):
            t = r['task_name']
            if t not in rows:
                order.append(t)
                pv = r.get('fifo_priority')
                prio[t] = int(pv) if pv not in (None, '') else None
            rows[t]['start'].append(float(r['period_start_ms']))
            rows[t]['cpu'].append(float(r['cpu_phase_ms']))
            rows[t]['overhead'].append(float(r['sched_preempt_overhead_ms']))
            rows[t]['gpu'].append(float(r['gpu_exec_ms']))
            rows[t]['response'].append(float(r['response_ms']))
            rows[t]['deadline'].append(float(r['deadline_ms']))
            rows[t]['missed'].append(int(r['missed']))
            # pid / absolute segment bounds (added for epsilon attribution);
            # older traces without these columns degrade gracefully.
            if 'pid' in r and r['pid'] != '':
                have_pid = True
                rows[t]['pid'].append(int(r['pid']))
                rows[t]['seg_begin'].append(float(r['seg_begin_ns']))
                rows[t]['seg_done'].append(float(r['seg_done_ns']))
    if not have_pid:
        print(f'  [note] {os.path.basename(path)} has no pid/seg columns — '
              'epsilon attribution unavailable (rebuild workloadTasksetGcaps)')
    return {t: {**{k: np.array(v) for k, v in rows[t].items()},
                'fifo_priority': prio.get(t)}
            for t in order}


def load_events(path: str) -> list[dict] | None:
    """Parse GCAPS_EV lines from a captured kernel-log file ->
    list of {ts, cpid, add, rlupd, eps_us, preempted, resumed}, sorted by ts.
    None if absent/empty."""
    if not os.path.isfile(path):
        print(f'  [skip] {path} not found')
        return None
    evs: list[dict] = []
    with open(path) as f:
        for line in f:
            m = GCAPS_EV_RE.search(line)
            if m:
                evs.append({'ts': int(m.group(1)), 'cpid': int(m.group(2)),
                            'add': int(m.group(4)), 'rlupd': int(m.group(5)),
                            'eps_us': int(m.group(6)),
                            'preempted': int(m.group(7)),
                            'resumed': int(m.group(8))})
    if not evs:
        print(f'  [skip] no GCAPS_EV lines in {path}')
        return None
    evs.sort(key=lambda e: e['ts'])
    return evs


def pid_to_name(taskset: dict[str, dict[str, np.ndarray]]) -> dict[int, str]:
    """Map each task's pid -> task_name (GPU tasks that issued ioctls)."""
    m: dict[int, str] = {}
    for t, d in taskset.items():
        if 'pid' in d and len(d['pid']):
            m[int(d['pid'][0])] = t
    return m


def attribute_epsilon(taskset: dict[str, dict[str, np.ndarray]],
                      events: list[dict]) -> dict[str, np.ndarray] | None:
    """Per-release measured epsilon (ms): for each release of each task, sum the
    elapsed_us of the GCAPS_EV events with that task's pid whose ts falls inside
    the release's GPU segment [seg_begin, seg_done] (normally the add + remove
    ioctls = 2*epsilon). Returns {task -> array aligned with the release arrays}
    or None if the trace lacks pid/segment bounds."""
    if not any('pid' in d for d in taskset.values()):
        return None
    by_pid: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for ev in events:
        by_pid[ev['cpid']].append((ev['ts'], ev['eps_us']))
    for pid in by_pid:
        by_pid[pid].sort()

    out: dict[str, np.ndarray] = {}
    for t, d in taskset.items():
        n = len(d['start'])
        eps = np.zeros(n)
        if 'pid' in d and len(d['pid']):
            evs = by_pid.get(int(d['pid'][0]), [])
            for i in range(n):
                b, e = d['seg_begin'][i], d['seg_done'][i]
                eps[i] = sum(us for ts, us in evs if b <= ts <= e) / 1.0e3
        out[t] = eps
    return out


def reconstruct_suspend_intervals(events: list[dict]) -> dict[int, list[tuple[int, int]]]:
    """Pair each `preempted=V` event with the next `resumed=V` -> per victim pid
    a sorted list of (start_ns, end_ns) suspended intervals.  Mirrors
    measure_preempt_overhead.py's reconstruction so the Gantt and the numeric
    analysis agree.  Only victims with pid > 0 count (preempted=-1 = the add
    admitted into an idle runlist, i.e. no real preemption)."""
    pending: dict[int, int] = {}          # victim pid -> ts it was preempted
    intervals: list[tuple[int, int, int]] = []
    for e in events:                      # events already sorted by ts
        if e['preempted'] > 0:
            pending.setdefault(e['preempted'], e['ts'])
        if e['resumed'] > 0 and e['resumed'] in pending:
            intervals.append((e['resumed'], pending.pop(e['resumed']), e['ts']))
    by_pid: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for pid, s, t in intervals:
        if t > s:
            by_pid[pid].append((s, t))
    for pid in by_pid:
        by_pid[pid].sort()
    return by_pid


def attribute_suspensions(taskset: dict[str, dict[str, np.ndarray]],
                          events: list[dict]) -> dict[str, list] | None:
    """Per-release list of driver-reported suspended intervals (absolute ns) that
    fall within each release's GPU segment [seg_begin, seg_done].  Returns
    {task -> [ [(s_ns, e_ns), ...] per release ]} aligned with the release
    arrays, or None if the trace lacks pid/segment bounds."""
    if not any('pid' in d for d in taskset.values()):
        return None
    by_pid = reconstruct_suspend_intervals(events)
    out: dict[str, list] = {}
    for t, d in taskset.items():
        n = len(d['start'])
        per_release: list[list[tuple[int, int]]] = [[] for _ in range(n)]
        if 'pid' in d and len(d['pid']):
            ivs = by_pid.get(int(d['pid'][0]), [])
            for i in range(n):
                b, e = int(d['seg_begin'][i]), int(d['seg_done'][i])
                if e <= b:
                    continue
                for s, t2 in ivs:
                    lo, hi = max(s, b), min(t2, e)
                    if hi > lo:
                        per_release[i].append((lo, hi))
        out[t] = per_release
    return out


# ── Sweep analysis ──────────────────────────────────────────────────────────

def sweep_summary(gcaps, tsg) -> None:
    print('\n── Sweep summary (response time, ms) ──────────────────────────')
    print(f'{"Workload":<16} {"gcaps mean":>10} {"gcaps p95":>10}'
          f' {"gcaps ovh":>10} {"tsg mean":>9} {"tsg p95":>9} {"Δmean":>8}')
    names = list(gcaps.keys()) if gcaps else list(tsg.keys())
    for wl in names:
        s = gcaps.get(wl) if gcaps else None
        b = tsg.get(wl) if tsg else None
        sm = np.mean(s['response']) if s is not None else float('nan')
        sp = np.percentile(s['response'], 95) if s is not None else float('nan')
        so = np.mean(s['overhead']) if s is not None else float('nan')
        bm = np.mean(b['response']) if b is not None else float('nan')
        bp = np.percentile(b['response'], 95) if b is not None else float('nan')
        print(f'{wl:<16} {sm:>10.3f} {sp:>10.3f} {so:>10.4f}'
              f' {bm:>9.3f} {bp:>9.3f} {sm - bm:>8.4f}')


def plot_sweep_response(gcaps, tsg, out_dir: str) -> None:
    types = ['matmul', 'histogram', 'convolution', 'mlp']
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.2))
    src = gcaps or tsg
    for ax, wtype in zip(axes, types):
        names = [wl for wl, d in src.items() if d['type'] == wtype]
        if not names:
            continue
        x = np.arange(len(names))
        width = 0.38
        if gcaps:
            m = [np.mean(gcaps[wl]['response']) for wl in names]
            lo = [m[i] - np.min(gcaps[wl]['response'])
                  for i, wl in enumerate(names)]
            hi = [np.max(gcaps[wl]['response']) - m[i]
                  for i, wl in enumerate(names)]
            ax.bar(x - width / 2, m, width, yerr=[lo, hi], capsize=2,
                   label='GCAPS', color=GCAPS_COLOR)
        if tsg:
            m = [np.mean(tsg[wl]['response']) for wl in names]
            lo = [m[i] - np.min(tsg[wl]['response'])
                  for i, wl in enumerate(names)]
            hi = [np.max(tsg[wl]['response']) - m[i]
                  for i, wl in enumerate(names)]
            ax.bar(x + width / 2, m, width, yerr=[lo, hi], capsize=2,
                   label='TSG baseline', color=TSG_COLOR)
        ax.set_title(wtype)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        ax.set_yscale('log')
        ax.set_ylabel('response time (ms)')
        ax.grid(axis='y', alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle('Workload size sweep — response time (mean, min–max)')
    fig.tight_layout()
    path = os.path.join(out_dir, 'sweep_response.pdf')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {path}')


def plot_sweep_overhead(gcaps, tsg, out_dir: str) -> None:
    names = list(gcaps.keys())
    x = np.arange(len(names))
    width = 0.38
    fig, ax = plt.subplots(figsize=(11, 4.2))
    overhead = [np.mean(gcaps[wl]['overhead']) * 1e3 for wl in names]
    ax.bar(x - width / 2, overhead, width,
           label='gcaps sched+preempt overhead', color=GCAPS_COLOR)
    if tsg:
        delta = [(np.mean(gcaps[wl]['response']) -
                  np.mean(tsg[wl]['response'])) * 1e3 for wl in names]
        ax.bar(x + width / 2, delta, width,
               label='response Δ (gcaps − tsg)', color=TSG_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('µs')
    ax.set_title('Per-release scheduling + preemption overhead '
                 '(uncontended sweep)')
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(out_dir, 'sweep_overhead.pdf')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {path}')


# ── Taskset analysis ────────────────────────────────────────────────────────

def taskset_summary(gcaps, tsg) -> None:
    print('\n── Taskset summary (response time, ms) ────────────────────────')
    print(f'{"Task":<16} {"gcaps MORT":>10} {"gcaps mean":>10}'
          f' {"gcaps miss":>10} {"tsg MORT":>9} {"tsg mean":>9}'
          f' {"tsg miss":>9}')
    names = list(gcaps.keys()) if gcaps else list(tsg.keys())
    for t in names:
        s = gcaps.get(t) if gcaps else None
        b = tsg.get(t) if tsg else None

        def stats(d):
            if d is None:
                return float('nan'), float('nan'), float('nan')
            return (np.max(d['response']), np.mean(d['response']),
                    100.0 * np.mean(d['missed']))

        smo, sme, smi = stats(s)
        bmo, bme, bmi = stats(b)
        print(f'{t:<16} {smo:>10.3f} {sme:>10.3f} {smi:>9.1f}%'
              f' {bmo:>9.3f} {bme:>9.3f} {bmi:>8.1f}%')


def taskset_utilization(gcaps, tsg) -> None:
    """Per-task worst-case GPU execution time and resulting GPU utilization."""
    print('\n── Taskset GPU utilization (worst-case GPU exec time) ─────────')
    for data, label in ((gcaps, 'GCAPS'), (tsg, 'TSG baseline')):
        if not data:
            continue
        print(f'\n{label}:')
        print(f'{"Task":<16} {"T (ms)":>9} {"C_gpu WCET (ms)":>16}'
              f' {"U_i = C/T":>11}')
        total = 0.0
        for t in data:
            T = float(data[t]['deadline'][0])
            C = float(np.max(data[t]['gpu']))
            u = C / T if T > 0 else float('nan')
            total += u
            print(f'{t:<16} {T:>9.3f} {C:>16.3f} {u:>11.4f}')
        print(f'{"":<16} {"":>9} {"total U_gpu":>16} {total:>11.4f}')


def plot_taskset_mort(gcaps, tsg, out_dir: str) -> None:
    names = list(gcaps.keys()) if gcaps else list(tsg.keys())
    x = np.arange(len(names))
    width = 0.38
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.2))
    for ax, fn, title in ((ax1, np.max, 'MORT'), (ax2, np.mean, 'Mean')):
        if gcaps:
            ax.bar(x - width / 2, [fn(gcaps[t]['response']) for t in names],
                   width, label='GCAPS', color=GCAPS_COLOR)
        if tsg:
            ax.bar(x + width / 2, [fn(tsg[t]['response']) for t in names],
                   width, label='TSG baseline', color=TSG_COLOR)
        dl_src = gcaps or tsg
        if dl_src:
            # Per-task deadline (T_i = D_i): a black tick spanning the bar pair.
            for j, t in enumerate(names):
                d = float(dl_src[t]['deadline'][0])
                ax.hlines(d, x[j] - width, x[j] + width, color='k',
                          linewidth=1.6, zorder=5)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('response time (ms)')
        ax.set_title(f'{title} response time per task')
        ax.grid(axis='y', alpha=0.3)
        handles, _ = ax.get_legend_handles_labels()
        handles.append(plt.Line2D([0], [0], color='k', linewidth=1.6,
                                  label='deadline (T$_i$ = D$_i$)'))
        ax.legend(handles=handles, fontsize=8)
    fig.tight_layout()
    path = os.path.join(out_dir, 'taskset_mort.pdf')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {path}')


def plot_taskset_breakdown(gcaps, tsg, out_dir: str,
                           eps: dict[str, np.ndarray] | None = None) -> None:
    names = list(gcaps.keys()) if gcaps else list(tsg.keys())
    x = np.arange(len(names))
    width = 0.38
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for off, data, tag, alpha in ((-width / 2, gcaps, 'gcaps', 1.0),
                                  (width / 2, tsg, 'tsg', 0.6)):
        if data is None:
            continue
        cpu = np.array([np.mean(data[t]['cpu']) for t in names])
        overhead = np.array([np.mean(data[t]['overhead']) for t in names])
        gpu = np.array([np.mean(data[t]['gpu']) for t in names])
        ax.bar(x + off, cpu, width, color=CPU_COLOR, alpha=alpha,
               label=f'{tag}: cpu phase')
        if tag == 'gcaps' and eps is not None:
            # Split overhead into measured ε (runlist update) and the remainder;
            # ε clipped to the slack so the stack still sums to the response.
            e = np.array([np.mean(eps[t]) if t in eps else 0.0 for t in names])
            e = np.minimum(e, overhead)
            other = np.clip(overhead - e, 0.0, None)
            ax.bar(x + off, e, width, bottom=cpu, color=EPS_COLOR, alpha=alpha,
                   label=f'{tag}: ε (runlist update)')
            ax.bar(x + off, other, width, bottom=cpu + e, color=OTHER_OVH_COLOR,
                   alpha=alpha, label=f'{tag}: other overhead')
            ax.bar(x + off, gpu, width, bottom=cpu + e + other, color=GPU_COLOR,
                   alpha=alpha, label=f'{tag}: gpu exec')
        else:
            ax.bar(x + off, overhead, width, bottom=cpu, color=OVERHEAD_COLOR,
                   alpha=alpha, label=f'{tag}: sched+preempt overhead')
            ax.bar(x + off, gpu, width, bottom=cpu + overhead, color=GPU_COLOR,
                   alpha=alpha, label=f'{tag}: gpu exec')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('mean time (ms)')
    ax.set_title('Mean response-time decomposition per task '
                 '(left: GCAPS, right: TSG baseline)')
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    path = os.path.join(out_dir, 'taskset_breakdown.pdf')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {path}')


def plot_taskset_response_overhead(gcaps, tsg, out_dir: str) -> None:
    """Per-task distribution of the response-time overhead = response - cpu_phase
    - gpu_exec, i.e. everything in the period outside the task's own CPU busy-wait
    and its measured on-GPU window. It bundles three things: the GCAPS runlist
    -update ioctls (≈2·ε — small at this scale, see epsilon.pdf), kernel-launch /
    sync latency, and GPU-side queuing / priority interference (the part that
    blows up for low-priority tasks waiting behind higher-priority ones)."""
    src = gcaps or tsg
    names = [t for t in src if np.any(src[t]['gpu'] > 0)]  # GPU tasks only
    fig, ax = plt.subplots(figsize=(11, 4.6))
    data, labels, colors = [], [], []
    for t in names:
        if gcaps:
            data.append(gcaps[t]['overhead'])
            labels.append(f'{t}\ngcaps')
            colors.append(GCAPS_COLOR)
        if tsg:
            data.append(tsg[t]['overhead'])
            labels.append(f'{t}\ntsg')
            colors.append(TSG_COLOR)
    bp = ax.boxplot(data, showfliers=True, patch_artist=True,
                    flierprops={'markersize': 2})
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax.set_ylabel('response overhead (ms)')
    ax.set_title('Per-task response overhead   (response − CPU phase − GPU exec)',
                 fontsize=12, pad=26)
    ax.text(0.5, 1.005,
            '= GCAPS ε (≈2× runlist-update ioctl, small here — see epsilon.pdf)  '
            '+  kernel launch/sync  +  GPU-side queuing / priority interference '
            '(dominant for low-priority tasks)',
            transform=ax.transAxes, ha='center', va='bottom', fontsize=8,
            color='#555555')
    ax.legend(handles=[mpatches.Patch(facecolor=GCAPS_COLOR, alpha=0.6,
                                      label='GCAPS (ioctl)'),
                       mpatches.Patch(facecolor=TSG_COLOR, alpha=0.6,
                                      label='TSG baseline')],
              fontsize=8, loc='upper left')
    ax.tick_params(axis='x', labelsize=7)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, 'taskset_response_overhead.pdf')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {path}')


# ── Epsilon (driver-measured runlist-update overhead) ───────────────────────

def epsilon_summary(events: list[dict]) -> None:
    print('\n── GCAPS epsilon — runlist-update overhead (Def. 2, µs) ───────')

    def line(tag: str, a: list[int]) -> None:
        if a:
            arr = np.array(a, float)
            print(f'  {tag:<24} n={len(arr):<5d} min={arr.min():7.0f}'
                  f' med={np.median(arr):7.0f} mean={arr.mean():7.0f}'
                  f' p95={np.percentile(arr, 95):7.0f} max={arr.max():7.0f}')
        else:
            print(f'  {tag:<24} (none)')

    line('all ioctls', [e['eps_us'] for e in events])
    line('runlist reload (rlupd=1)', [e['eps_us'] for e in events if e['rlupd']])
    line('no-op (rlupd=0)', [e['eps_us'] for e in events if not e['rlupd']])
    line('preemptions', [e['eps_us'] for e in events if e['preempted'] > 0])


def plot_epsilon(events: list[dict], pid2name: dict[int, str], out_dir: str,
                 names_order: list[str] | None = None) -> None:
    """epsilon.pdf — left: histogram of elapsed_us split into the no-op and
    runlist-reload modes (the bimodal structure the paper's Fig. 12 shows);
    right: per-task epsilon box plot over the runlist-reload events only
    (rlupd=1) — the no-op IOCTL-only calls are excluded so the box reflects the
    real reload cost (α+θ) rather than the bimodal mix."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.4))

    allv = np.array([e['eps_us'] for e in events], float)
    eps_reload = [e['eps_us'] for e in events if e['rlupd']]
    eps_noop = [e['eps_us'] for e in events if not e['rlupd']]
    bins = np.linspace(0.0, max(float(allv.max()), 1.0), 40)
    ax1.hist([eps_reload, eps_noop], bins=bins, stacked=True,
             color=[EPS_COLOR, OTHER_OVH_COLOR],
             label=['runlist reload (rlupd=1)', 'no-op (rlupd=0)'])
    ax1.set_xlabel('ε = runlist-update overhead (µs)')
    ax1.set_ylabel('frequency')
    ax1.set_title('GCAPS ε distribution (Def. 2)')
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(axis='y', alpha=0.3)
    # Stats box in the upper-left (the bimodal gap leaves it empty) so it does
    # not collide with the legend in the upper-right.
    stats = (f'n={len(allv)}\nmin={allv.min():.0f}\nmed={np.median(allv):.0f}'
             f'\nmean={allv.mean():.0f}\nmax={allv.max():.0f}  µs')
    ax1.text(0.03, 0.97, stats, transform=ax1.transAxes, ha='left', va='top',
             fontsize=9, family='monospace',
             bbox=dict(boxstyle='round', fc='white', ec='#cccccc', alpha=0.9))

    # Per-task box over runlist-reload events only (rlupd=1): the no-op IOCTL
    # calls (~tens of µs) are excluded so the box shows the real reload cost,
    # not the bimodal mix that otherwise inflates the spread of mid-priority
    # tasks whose requests are sometimes granted (reload) and sometimes not.
    per: dict[str, list[int]] = defaultdict(list)
    for e in events:
        if not e['rlupd']:
            continue
        nm = pid2name.get(e['cpid'])
        if nm is not None:
            per[nm].append(e['eps_us'])
    names = [n for n in (names_order or list(per.keys())) if n in per]
    if names:
        bp = ax2.boxplot([per[n] for n in names], showfliers=True,
                         patch_artist=True, flierprops={'markersize': 2})
        for patch in bp['boxes']:
            patch.set_facecolor(EPS_COLOR)
            patch.set_alpha(0.6)
        ax2.set_xticks(np.arange(1, len(names) + 1))
        ax2.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    else:
        ax2.text(0.5, 0.5, 'no per-task pid mapping\n(rebuild '
                 'workloadTasksetGcaps for the pid column)', ha='center',
                 va='center', transform=ax2.transAxes, fontsize=9)
    ax2.set_ylabel('ε (µs)')
    ax2.set_title('Per-task ε — runlist reload only (rlupd=1)')
    ax2.grid(axis='y', alpha=0.3)

    fig.suptitle('Driver-measured GCAPS runlist-update overhead  (ε = α + θ)')
    # Reserve headroom so the suptitle clears both subplot titles.
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = os.path.join(out_dir, 'epsilon.pdf')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {path}')


def _subtract_intervals(a: float, b: float,
                        holes: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Complement of `holes` within [a, b] as a list of (start, width) spans."""
    hs = sorted((max(s, a), min(e, b)) for s, e in holes if min(e, b) > max(s, a))
    out: list[tuple[float, float]] = []
    cur = a
    for s, e in hs:
        if s > cur:
            out.append((cur, s - cur))
        cur = max(cur, e)
    if b > cur:
        out.append((cur, b - cur))
    return out


def _draw_gantt_panel(ax, data, names: list[str], title: str,
                      labels: list[str] | None = None,
                      start_ms: float | None = None,
                      end_ms: float | None = None,
                      eps: dict[str, np.ndarray] | None = None,
                      susp: dict[str, list] | None = None) -> None:
    """Draw one Gantt panel: one row per task, each period tiled as
    CPU (green) | overhead (red) | GPU (purple) broken_barh segments.

    When ``susp`` is given (GCAPS panel only), the driver-reported suspended
    intervals are carved out of the purple GPU block in SUSPENDED_COLOR, so a
    preempted release renders as GPU–suspended–GPU and the purple then reads as
    GPU *actively executing* (the seq Gantt's convention).

    When [start_ms, end_ms] is given, periods that fall entirely outside the
    window are skipped — visually identical to the caller's xlim clip, but it
    avoids drawing (and rasterising) the whole trace, which dominates render
    time for long runs / many tasks."""
    bar_h = 0.72
    for row, t in enumerate(names):
        d = data.get(t)
        if d is None or len(d['start']) == 0:
            continue
        start = d['start']
        cpu = np.clip(d['cpu'], 0.0, None)
        overhead = np.clip(d['overhead'], 0.0, None)
        gpu = np.clip(d['gpu'], 0.0, None)

        keep = np.ones(len(start), dtype=bool)
        if start_ms is not None:
            span = np.maximum(d['response'], d['deadline'])
            keep = (start <= end_ms) & (start + span >= start_ms)
        s = start[keep]
        c = cpu[keep]
        o = overhead[keep]
        g = gpu[keep]

        y0 = row - bar_h / 2
        ax.broken_barh(list(zip(s, c)), (y0, bar_h),
                       facecolors=CPU_COLOR, edgecolors='none')

        eps_arr = eps.get(t) if eps is not None else None
        if eps_arr is not None and len(eps_arr) == len(overhead):
            # Split the overhead band into the measured ε (runlist update) and
            # the remaining latency (launch + GPU-side delay), ε clipped to the
            # measured slack so the tiles still sum to the response.
            e = np.minimum(np.clip(eps_arr, 0.0, None), overhead)[keep]
            other = np.clip(o - e, 0.0, None)
            ax.broken_barh(list(zip(s + c, e)), (y0, bar_h),
                           facecolors=EPS_COLOR, edgecolors='none')
            ax.broken_barh(list(zip(s + c + e, other)), (y0, bar_h),
                           facecolors=OTHER_OVH_COLOR, edgecolors='none')
        else:
            ax.broken_barh(list(zip(s + c, o)), (y0, bar_h),
                           facecolors=OVERHEAD_COLOR, edgecolors='none')

        sus_t = susp.get(t) if susp is not None else None
        if (sus_t is not None and len(sus_t) == len(start)
                and 'seg_begin' in d):
            # Carve the driver-reported suspended intervals out of each purple
            # GPU block.  The GPU block [g0, g1] spans the last gpu_ms of the
            # segment window [seg_begin, seg_done]; map an absolute suspended
            # interval to the axis via  ms(x) = (start+cpu) + (x - seg_begin)/1e6
            # (seg_begin -> start+cpu, seg_done -> start+resp = g1) and clip to
            # [g0, g1].  The ~ε front-loading of the overhead band shifts the
            # block by <1 ms, negligible at Gantt scale.
            idxs = np.nonzero(keep)[0]
            active_spans: list[tuple[float, float]] = []
            susp_spans: list[tuple[float, float]] = []
            for k in idxs:
                g0 = float(start[k] + cpu[k] + overhead[k])
                g1 = g0 + float(gpu[k])
                if g1 <= g0:
                    continue
                sb = float(d['seg_begin'][k])
                base_ms = float(start[k] + cpu[k])
                holes = [(base_ms + (hs - sb) / 1.0e6, base_ms + (he - sb) / 1.0e6)
                         for hs, he in sus_t[k]]
                holes = [(max(a, g0), min(b, g1)) for a, b in holes
                         if min(b, g1) > max(a, g0)]
                active_spans += _subtract_intervals(g0, g1, holes)
                susp_spans += [(a, b - a) for a, b in holes]
            if active_spans:
                ax.broken_barh(active_spans, (y0, bar_h),
                               facecolors=GPU_COLOR, edgecolors='none')
            if susp_spans:
                ax.broken_barh(susp_spans, (y0, bar_h),
                               facecolors=SUSPENDED_COLOR, edgecolors='none')
        else:
            ax.broken_barh(list(zip(s + c + o, g)), (y0, bar_h),
                           facecolors=GPU_COLOR, edgecolors='none')

        # Deadlines: downward triangle at release + relative deadline,
        # red+larger when the release missed.  Clipped to the window when given.
        dl = start + d['deadline']
        missed = d['missed'].astype(bool)
        vis = np.ones(len(dl), dtype=bool)
        if start_ms is not None:
            vis = (dl >= start_ms) & (dl <= end_ms)
        ok = (~missed) & vis
        ms = missed & vis
        if np.any(ok):
            ax.plot(dl[ok], np.full(ok.sum(), row), marker='v',
                    linestyle='none', color='#555555', markersize=6,
                    alpha=0.6, zorder=3)
        if np.any(ms):
            ax.plot(dl[ms], np.full(ms.sum(), row), marker='v',
                    linestyle='none', color=OVERHEAD_COLOR, markersize=10,
                    alpha=1.0, zorder=4)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(labels if labels is not None else names, fontsize=13)
    ax.set_ylim(-0.6, len(names) - 0.4)
    ax.invert_yaxis()  # first task (highest priority) on top
    ax.tick_params(axis='x', labelsize=11)
    ax.set_title(title, fontsize=15, fontweight='bold', loc='left')
    ax.grid(axis='x', linestyle=':', linewidth=0.6, alpha=0.4)


def _gantt_priority_order(panels) -> tuple[list[str], dict[str, str]]:
    """Order tasks by their ASSIGNED SCHED_FIFO priority (higher = higher
    priority, drawn on top) and build a per-task row label that carries it.

    Priority is read straight from the trace's fifo_priority column, so the row
    order follows whatever priorities the taskset assigns (no deadline-monotonic
    assumption).  The CPU-only task (no GPU segment in any period) sorts below
    the GPU tasks.  When the column is absent (older traces) the original trace
    order and plain names are kept."""
    meta: dict[str, int | None] = {}
    cpu_only: dict[str, bool] = {}
    names: list[str] = []
    for d, _ in panels:
        for t in d:
            if t not in meta:
                meta[t] = d[t].get('fifo_priority')
                cpu_only[t] = not np.any(d[t]['gpu'] > 0)
                names.append(t)

    if all(v is None for v in meta.values()):
        return names, {t: t for t in names}      # no priority info in trace

    # GPU tasks first, descending FIFO priority (higher number = higher
    # priority); the CPU-only task last.
    names = sorted(names, key=lambda t: (1, 0) if cpu_only[t]
                   else (0, -(meta[t] if meta[t] is not None else 0)))
    labels = {t: (f'{t}  (CPU)' if cpu_only[t] else f'{t}  (prio {meta[t]})')
              for t in names}
    return names, labels


def plot_taskset_gantt(gcaps, tsg, out_dir: str,
                       inches_per_sec: float = GANTT_INCHES_PER_S,
                       start_s: float = 0.0,
                       duration_s: float = 10.0,
                       eps: dict[str, np.ndarray] | None = None,
                       susp: dict[str, list] | None = None) -> None:
    """Stacked GCAPS-vs-TSG Gantt over the window [start_s, start_s +
    duration_s], shared time axis.

    Wide on purpose (scaled to the window length) so individual periods stay
    legible when panning/zooming the PDF.  ``inches_per_sec`` sets how many
    figure inches each second of trace time occupies.
    """
    panels = [(d, lbl) for d, lbl in
              ((gcaps, 'GCAPS'), (tsg, 'TSG baseline'))
              if d]

    # Task rows ordered by assigned FIFO priority (highest on top), each row
    # labelled with its priority.
    names, prio_labels = _gantt_priority_order(panels)

    # Window [start_ms, end_ms]; periods entirely outside are skipped when
    # drawing, and segments straddling an edge render as partial bars.
    start_ms = max(0.0, start_s) * 1000.0
    end_ms = start_ms + max(0.0, duration_s) * 1000.0

    fig_w = max(24.0, min(GANTT_MAX_WIDTH_IN,
                          max(0.0, duration_s) * inches_per_sec))
    fig_h = len(panels) * (2.0 + GANTT_ROW_HEIGHT_IN * len(names))
    fig, axes = plt.subplots(len(panels), 1, figsize=(fig_w, fig_h),
                             sharex=True, squeeze=False)
    axes = axes[:, 0]

    row_labels = [prio_labels[t] for t in names]
    for ax, (d, lbl) in zip(axes, panels):
        # ε and suspended intervals are only defined for the GCAPS (ioctl) panel;
        # the TSG baseline issues no runlist-update ioctls, so its overhead band
        # stays unsplit and its GPU block uncarved.
        _draw_gantt_panel(ax, d, names, lbl, row_labels, start_ms, end_ms,
                          eps=eps if lbl == 'GCAPS' else None,
                          susp=susp if lbl == 'GCAPS' else None)
    axes[-1].set_xlabel('time since experiment start (ms)', fontsize=13)
    axes[-1].set_xlim(left=start_ms, right=end_ms)

    ovh_handles = (
        [mpatches.Patch(facecolor=EPS_COLOR, label='ε: runlist update (GCAPS)'),
         mpatches.Patch(facecolor=OTHER_OVH_COLOR,
                        label='other overhead (launch + GPU-side)')]
        if eps is not None else
        [mpatches.Patch(facecolor=OVERHEAD_COLOR, label='sched+preempt overhead')]
    )
    gpu_handles = (
        [mpatches.Patch(facecolor=GPU_COLOR, label='GPU exec (active)'),
         mpatches.Patch(facecolor=SUSPENDED_COLOR,
                        label='suspended (preempted, GCAPS)')]
        if susp is not None else
        [mpatches.Patch(facecolor=GPU_COLOR, label='GPU exec')]
    )
    legend_handles = [
        mpatches.Patch(facecolor=CPU_COLOR, label='CPU phase'),
        *ovh_handles,
        *gpu_handles,
        plt.Line2D([0], [0], marker='v', color='#555555', linestyle='none',
                   markersize=6, label='deadline'),
        plt.Line2D([0], [0], marker='v', color=OVERHEAD_COLOR,
                   linestyle='none', markersize=10, label='missed deadline'),
    ]
    axes[0].legend(handles=legend_handles, fontsize=12, loc='upper right',
                   framealpha=0.9, edgecolor='#cccccc', ncol=len(legend_handles))
    fig.suptitle('Taskset execution Gantt — CPU | overhead | GPU per period  '
                 f'(window {start_s:g}–{start_s + duration_s:g} s)',
                 fontsize=16, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    path = os.path.join(out_dir, 'taskset_gantt.pdf')
    fig.savefig(path, bbox_inches='tight', dpi=GANTT_DPI)
    plt.close(fig)
    if fig_w > 200.0:
        print(f'  note: figure is {fig_w:.0f} in wide; some PDF viewers cap at '
              '~200 in (14400 pt). Lower --gantt-inches-per-sec if it will '
              'not open.')
    print(f'  wrote {path}  ({fig_w:.0f}×{fig_h:.0f} in, dpi={GANTT_DPI})')


# ── main ────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results-dir', default='results/workloadBench')
    ap.add_argument('--gantt-inches-per-sec', type=float,
                    default=GANTT_INCHES_PER_S,
                    help='figure width allotted to each second of trace time, '
                         f'in inches (default {GANTT_INCHES_PER_S}); larger = '
                         'wider/more legible')
    ap.add_argument('--gantt-start', type=float, default=0.0,
                    help='second at which the Gantt window begins '
                         '(default 0)')
    ap.add_argument('--gantt-duration', type=float, default=10.0,
                    help='length of the Gantt window in seconds, from '
                         '--gantt-start (default 10)')
    args = ap.parse_args()
    out = args.results_dir

    print('Loading sweep CSVs…')
    sweep_gcaps = load_sweep(os.path.join(out, 'sweep_gcaps.csv'))
    sweep_tsg = load_sweep(os.path.join(out, 'sweep_tsg.csv'))

    print('Loading taskset traces…')
    ts_gcaps = load_taskset(os.path.join(out, 'taskset_gcaps_trace.csv'))
    ts_tsg = load_taskset(os.path.join(out, 'taskset_tsg_trace.csv'))

    if sweep_gcaps or sweep_tsg:
        sweep_summary(sweep_gcaps, sweep_tsg)
        plot_sweep_response(sweep_gcaps, sweep_tsg, out)
        if sweep_gcaps:
            plot_sweep_overhead(sweep_gcaps, sweep_tsg, out)

    if ts_gcaps or ts_tsg:
        taskset_summary(ts_gcaps, ts_tsg)
        taskset_utilization(ts_gcaps, ts_tsg)
        # Driver-measured epsilon from the captured GCAPS_EV kernel log (written
        # by measure_preempt_overhead.py --run taskset). Optional: absent log ->
        # the ε figure/overlay are skipped and the other plots are unchanged.
        print('Loading GCAPS_EV event log…')
        ts_events = load_events(os.path.join(out, 'taskset_gcaps_events.log'))
        ts_eps = None
        ts_susp = None
        if ts_events and ts_gcaps:
            epsilon_summary(ts_events)
            plot_epsilon(ts_events, pid_to_name(ts_gcaps), out,
                         list(ts_gcaps.keys()))
            ts_eps = attribute_epsilon(ts_gcaps, ts_events)
            ts_susp = attribute_suspensions(ts_gcaps, ts_events)
            if ts_eps is None:
                print('  [note] trace lacks pid/seg columns — ε overlay and '
                      'suspended blocks on breakdown/Gantt skipped '
                      '(ε figure still drawn)')
        plot_taskset_mort(ts_gcaps, ts_tsg, out)
        plot_taskset_breakdown(ts_gcaps, ts_tsg, out, ts_eps)
        plot_taskset_response_overhead(ts_gcaps, ts_tsg, out)
        plot_taskset_gantt(ts_gcaps, ts_tsg, out, args.gantt_inches_per_sec,
                           args.gantt_start, args.gantt_duration, ts_eps,
                           ts_susp)

    if not any((sweep_gcaps, sweep_tsg, ts_gcaps, ts_tsg)):
        print('No input CSVs found — run the bench binaries first.')
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
