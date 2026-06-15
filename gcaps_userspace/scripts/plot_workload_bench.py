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

Figures written to --results-dir:
  sweep_response.pdf      — mean response per configuration, GCAPS vs TSG
                            (one panel per workload type, log y)
  sweep_overhead.pdf      — per-config sched+preempt overhead: response delta
                            (GCAPS - TSG) and GCAPS overhead
  taskset_mort.pdf        — MORT and mean response per task, GCAPS vs TSG
  taskset_breakdown.pdf   — stacked mean cpu / overhead / gpu components per task
  taskset_overhead.pdf    — sched+preempt-overhead distribution box plots,
                            GPU tasks
  taskset_gantt.pdf       — stacked GCAPS-vs-TSG execution Gantt over the
                            window [--gantt-start, +--gantt-duration]; each
                            period tiled CPU (green) | overhead (red) |
                            GPU (purple); width scaled to the window length

Usage:
    python3 scripts/plot_workload_bench.py \\
        [--results-dir results/workloadBench]
"""

from __future__ import annotations

import argparse
import csv
import os
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
    """task_name -> {start, cpu, overhead, gpu, response, deadline, missed}."""
    if not os.path.isfile(path):
        print(f'  [skip] {path} not found')
        return None
    rows = defaultdict(lambda: defaultdict(list))
    order: list[str] = []
    with open(path) as f:
        for r in csv.DictReader(f):
            t = r['task_name']
            if t not in rows:
                order.append(t)
            rows[t]['start'].append(float(r['period_start_ms']))
            rows[t]['cpu'].append(float(r['cpu_phase_ms']))
            rows[t]['overhead'].append(float(r['sched_preempt_overhead_ms']))
            rows[t]['gpu'].append(float(r['gpu_exec_ms']))
            rows[t]['response'].append(float(r['response_ms']))
            rows[t]['deadline'].append(float(r['deadline_ms']))
            rows[t]['missed'].append(int(r['missed']))
    return {t: {k: np.array(v) for k, v in rows[t].items()} for t in order}


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
    types = ['matmul', 'histogram', 'convolution']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
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
        if gcaps:
            ax.plot(x, [gcaps[t]['deadline'][0] for t in names], 'k_',
                    markersize=18, label='deadline')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('response time (ms)')
        ax.set_title(f'{title} response time per task')
        ax.grid(axis='y', alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(out_dir, 'taskset_mort.pdf')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {path}')


def plot_taskset_breakdown(gcaps, tsg, out_dir: str) -> None:
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


def plot_taskset_overhead(gcaps, tsg, out_dir: str) -> None:
    src = gcaps or tsg
    names = [t for t in src if np.any(src[t]['gpu'] > 0)]  # GPU tasks only
    fig, ax = plt.subplots(figsize=(11, 4.2))
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
    ax.set_ylabel('sched + preempt overhead (ms)')
    ax.set_title('Scheduling + preemption overhead distribution per GPU task')
    ax.tick_params(axis='x', labelsize=7)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, 'taskset_overhead.pdf')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {path}')


def _draw_gantt_panel(ax, data, names: list[str], title: str) -> None:
    """Draw one Gantt panel: one row per task, each period tiled as
    CPU (green) | overhead (red) | GPU (purple) broken_barh segments."""
    bar_h = 0.72
    for row, t in enumerate(names):
        d = data.get(t)
        if d is None or len(d['start']) == 0:
            continue
        start = d['start']
        cpu = np.clip(d['cpu'], 0.0, None)
        overhead = np.clip(d['overhead'], 0.0, None)
        gpu = np.clip(d['gpu'], 0.0, None)

        cpu_seg = list(zip(start, cpu))
        o_seg = list(zip(start + cpu, overhead))
        g_seg = list(zip(start + cpu + overhead, gpu))

        y0 = row - bar_h / 2
        ax.broken_barh(cpu_seg, (y0, bar_h), facecolors=CPU_COLOR,
                       edgecolors='none')
        ax.broken_barh(o_seg, (y0, bar_h), facecolors=OVERHEAD_COLOR,
                       edgecolors='none')
        ax.broken_barh(g_seg, (y0, bar_h), facecolors=GPU_COLOR,
                       edgecolors='none')

        # Deadlines: downward triangle at release + relative deadline,
        # red+larger when the release missed.
        dl = start + d['deadline']
        missed = d['missed'].astype(bool)
        if np.any(~missed):
            ax.plot(dl[~missed], np.full((~missed).sum(), row), marker='v',
                    linestyle='none', color='#555555', markersize=6,
                    alpha=0.6, zorder=3)
        if np.any(missed):
            ax.plot(dl[missed], np.full(missed.sum(), row), marker='v',
                    linestyle='none', color=OVERHEAD_COLOR, markersize=10,
                    alpha=1.0, zorder=4)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=13)
    ax.set_ylim(-0.6, len(names) - 0.4)
    ax.invert_yaxis()  # first task on top
    ax.tick_params(axis='x', labelsize=11)
    ax.set_title(title, fontsize=15, fontweight='bold', loc='left')
    ax.grid(axis='x', linestyle=':', linewidth=0.6, alpha=0.4)


def plot_taskset_gantt(gcaps, tsg, out_dir: str,
                       inches_per_sec: float = GANTT_INCHES_PER_S,
                       start_s: float = 0.0,
                       duration_s: float = 10.0) -> None:
    """Stacked GCAPS-vs-TSG Gantt over the window [start_s, start_s +
    duration_s], shared time axis.

    Wide on purpose (scaled to the window length) so individual periods stay
    legible when panning/zooming the PDF.  ``inches_per_sec`` sets how many
    figure inches each second of trace time occupies.
    """
    panels = [(d, lbl) for d, lbl in
              ((gcaps, 'GCAPS'), (tsg, 'TSG baseline'))
              if d]

    # Consistent task ordering across panels: union, gcaps order first.
    names: list[str] = []
    for d, _ in panels:
        for t in d:
            if t not in names:
                names.append(t)

    # Window [start_ms, end_ms]; bars outside are clipped by set_xlim, and
    # segments straddling an edge render as partial bars.
    start_ms = max(0.0, start_s) * 1000.0
    end_ms = start_ms + max(0.0, duration_s) * 1000.0

    fig_w = max(24.0, min(GANTT_MAX_WIDTH_IN,
                          max(0.0, duration_s) * inches_per_sec))
    fig_h = len(panels) * (2.0 + GANTT_ROW_HEIGHT_IN * len(names))
    fig, axes = plt.subplots(len(panels), 1, figsize=(fig_w, fig_h),
                             sharex=True, squeeze=False)
    axes = axes[:, 0]

    for ax, (d, lbl) in zip(axes, panels):
        _draw_gantt_panel(ax, d, names, lbl)
    axes[-1].set_xlabel('time since experiment start (ms)', fontsize=13)
    axes[-1].set_xlim(left=start_ms, right=end_ms)

    legend_handles = [
        mpatches.Patch(facecolor=CPU_COLOR, label='CPU phase'),
        mpatches.Patch(facecolor=OVERHEAD_COLOR, label='sched+preempt overhead'),
        mpatches.Patch(facecolor=GPU_COLOR, label='GPU exec'),
        plt.Line2D([0], [0], marker='v', color='#555555', linestyle='none',
                   markersize=6, label='deadline'),
        plt.Line2D([0], [0], marker='v', color=OVERHEAD_COLOR,
                   linestyle='none', markersize=10, label='missed deadline'),
    ]
    axes[0].legend(handles=legend_handles, fontsize=12, loc='upper right',
                   framealpha=0.9, edgecolor='#cccccc', ncol=5)
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
        plot_taskset_mort(ts_gcaps, ts_tsg, out)
        plot_taskset_breakdown(ts_gcaps, ts_tsg, out)
        plot_taskset_overhead(ts_gcaps, ts_tsg, out)
        plot_taskset_gantt(ts_gcaps, ts_tsg, out, args.gantt_inches_per_sec,
                           args.gantt_start, args.gantt_duration)

    if not any((sweep_gcaps, sweep_tsg, ts_gcaps, ts_tsg)):
        print('No input CSVs found — run the bench binaries first.')
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
