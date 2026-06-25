#!/usr/bin/env python3
"""
Measure GCAPS preemption overhead from the patched driver's GCAPS_EV log plus a
benchmark trace.  Two quantities are reported:

  (1) Scheduling + context-switch / preemption overhead.
      Each runlist-update ioctl logs a line like

          GCAPS_EV ts=<ns> cpid=<pid> prio=<n> add=<0|1> rlupd=<0|1> \
                   elapsed_us=<eps> preempted=<pid|-1> resumed=<pid|-1>

      `elapsed_us` is the duration of the ioctl critical section, which performs
      the runlist reload and (with wait_for_finish) the GPU-side switch.  The
      events whose `preempted` names a real victim are the ones where a running
      lower-priority job was actually evicted by a higher-priority arrival; their
      `elapsed_us` is the per-preemption scheduling+context-switch cost.  (The
      paired `resumed` event is the cost of putting the victim back.)

  (2) How much a job's *execution* time is extended when it is preempted.
      The benchmark's cudaEvent window (gpu_exec_ms) is GPU *wall* time =
      active execution + time the segment sat suspended while a higher-priority
      job ran.  From the preempted/resumed events we reconstruct each pid's
      suspended intervals and subtract the part overlapping a release's GPU
      window, giving *active* execution time (suspension excluded, by
      construction).  For each task we then compare active execution on
      non-preempted vs preempted releases:

          extension = active_exec(preempted) - median active_exec(not preempted)

      i.e. the save/restore + cold-cache cost charged to the job's own running
      time, NOT the blocking time.

Usage
-----
Run a benchmark and analyse it (needs the patched driver, sudo, and the
kernel-log readable; clears dmesg first):

    sudo python3 scripts/measure_preempt_overhead.py --run microbench -d 30
    sudo python3 scripts/measure_preempt_overhead.py --run taskset   -d 30

Analyse already-captured data (no device needed):

    python3 scripts/measure_preempt_overhead.py \
        --events results/workloadBench/preempt_gcaps_events.log \
        --trace  results/workloadBench/preempt_gcaps_trace.csv

`--events` accepts any text containing GCAPS_EV lines (e.g. `dmesg` output or a
saved `dmesg | grep GCAPS_EV`).  The trace kind (microbench vs taskset) is
detected from its CSV header.
"""

import argparse
import math
import os
import re
import subprocess
import sys

GCAPS_EV_RE = re.compile(
    r"GCAPS_EV\s+ts=(\d+)\s+cpid=(-?\d+)\s+prio=(-?\d+)\s+add=(\d+)\s+"
    r"rlupd=(\d+)\s+elapsed_us=(-?\d+)\s+preempted=(-?\d+)\s+resumed=(-?\d+)"
)


# --------------------------------------------------------------------------- #
# small stats helpers (stdlib only)
# --------------------------------------------------------------------------- #
def pct(sorted_vals, p):
    """Nearest-rank percentile of an already-sorted list."""
    if not sorted_vals:
        return float("nan")
    k = max(1, math.ceil(p / 100.0 * len(sorted_vals))) - 1
    return sorted_vals[k]


def median(vals):
    if not vals:
        return float("nan")
    s = sorted(vals)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def describe(vals):
    if not vals:
        return dict(n=0, min=float("nan"), mean=float("nan"),
                    median=float("nan"), p95=float("nan"), max=float("nan"))
    s = sorted(vals)
    return dict(n=len(s), min=s[0], mean=sum(s) / len(s), median=median(s),
                p95=pct(s, 95), max=s[-1])


def fmt(d, unit):
    if d["n"] == 0:
        return "        (no samples)"
    return (f"n={d['n']:<5d} min={d['min']:8.3f} mean={d['mean']:8.3f} "
            f"med={d['median']:8.3f} p95={d['p95']:8.3f} max={d['max']:8.3f} {unit}")


def linfit(xs, ys):
    """Least-squares slope and Pearson r of ys on xs."""
    n = len(xs)
    if n < 2:
        return float("nan"), float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    syy = sum((y - my) ** 2 for y in ys)
    slope = sxy / sxx if sxx > 0 else float("nan")
    r = sxy / (sxx * syy) ** 0.5 if sxx > 0 and syy > 0 else float("nan")
    return slope, r


# --------------------------------------------------------------------------- #
# event parsing + suspend-interval reconstruction
# --------------------------------------------------------------------------- #
class Event:
    __slots__ = ("ts", "cpid", "prio", "add", "rlupd", "eps", "preempted",
                 "resumed")

    def __init__(self, m):
        (self.ts, self.cpid, self.prio, self.add, self.rlupd, self.eps,
         self.preempted, self.resumed) = (
            int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)),
            int(m.group(5)), int(m.group(6)), int(m.group(7)), int(m.group(8)))


def parse_events(text):
    evs = [Event(m) for m in (GCAPS_EV_RE.search(l) for l in text.splitlines())
           if m]
    evs.sort(key=lambda e: e.ts)
    return evs


def reconstruct_suspend_intervals(events):
    """Pair each `preempted=V` with the next `resumed=V` -> (V, start_ns, end_ns).

    Only victims with pid > 0 count (preempted=-1 means the add admitted into an
    idle runlist, i.e. no real preemption)."""
    pending = {}            # victim pid -> ts it was preempted
    intervals = []          # (pid, start_ns, end_ns)
    for e in events:
        if e.preempted > 0:
            # if already pending (shouldn't happen), keep the earlier start
            pending.setdefault(e.preempted, e.ts)
        if e.resumed > 0 and e.resumed in pending:
            intervals.append((e.resumed, pending.pop(e.resumed), e.ts))
    by_pid = {}
    for pid, s, t in intervals:
        if t > s:
            by_pid.setdefault(pid, []).append((s, t))
    for pid in by_pid:
        by_pid[pid].sort()
    return by_pid


def overlap_ns(a0, a1, b0, b1):
    return max(0, min(a1, b1) - max(a0, b0))


# --------------------------------------------------------------------------- #
# trace loading (microbench or taskset) -> uniform release records
# --------------------------------------------------------------------------- #
class Release:
    __slots__ = ("pid", "name", "role", "begin", "end", "gpu_ms")


def load_trace(path):
    with open(path) as f:
        header = f.readline().strip().split(",")
        cols = {name: i for i, name in enumerate(header)}
        need = ("pid", "seg_begin_ns", "seg_done_ns", "gpu_exec_ms")
        for c in need:
            if c not in cols:
                sys.exit(f"trace {path} missing column '{c}' (header: {header})")
        has_role = "role" in cols
        name_col = cols.get("task_name", cols.get("name"))
        rels = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = line.split(",")
            begin = int(p[cols["seg_begin_ns"]])
            end = int(p[cols["seg_done_ns"]])
            if begin == 0 or end <= begin:
                continue                    # CPU-only task / no GPU segment
            r = Release()
            r.pid = int(p[cols["pid"]])
            r.name = p[name_col] if name_col is not None else f"pid{r.pid}"
            r.role = p[cols["role"]] if has_role else ""
            r.begin = begin
            r.end = end
            r.gpu_ms = float(p[cols["gpu_exec_ms"]])
            rels.append(r)
    return rels


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def report(events, rels, label, trace_path=None):
    print(f"\n===================== {label} =====================")

    if not events:
        print("No GCAPS_EV events found — was the run done with the patched "
              "driver and -i 1 (GCAPS ioctl) enabled?")
    # ---- (1) scheduling + context-switch / preemption overhead ----------- #
    preempt_eps = [e.eps for e in events if e.preempted > 0]
    resume_eps = [e.eps for e in events if e.resumed > 0]
    rlupd_eps = [e.eps for e in events if e.rlupd == 1]
    noop_eps = [e.eps for e in events if e.rlupd == 0]
    add_eps = [e.eps for e in events if e.add == 1]
    rem_eps = [e.eps for e in events if e.add == 0]

    print("\n(1) Scheduling + context-switch / preemption overhead "
          "(ioctl critical section, GCAPS epsilon)")
    print(f"  preempting add (a real preemption) : {fmt(describe(preempt_eps), 'us')}")
    print(f"  resuming remove (victim put back)  : {fmt(describe(resume_eps), 'us')}")
    print(f"  any real runlist reload (rlupd=1)  : {fmt(describe(rlupd_eps), 'us')}")
    print(f"  bookkeeping only      (rlupd=0)    : {fmt(describe(noop_eps), 'us')}")
    print(f"  all add ioctls                     : {fmt(describe(add_eps), 'us')}")
    print(f"  all remove ioctls                  : {fmt(describe(rem_eps), 'us')}")
    print(f"  -> headline per-preemption overhead (worst case) = "
          f"{describe(preempt_eps)['max']:.3f} us")

    # ---- (2) execution-time extension when preempted --------------------- #
    susp = reconstruct_suspend_intervals(events)

    # Earliest preemption anywhere in the run: releases finishing before this
    # ran with the preemptors idle, so they form an uncontaminated baseline
    # (the microbench's PREEMPTOR_ACTIVATION window). Falls back to all
    # non-preempted releases when there is no such idle phase (e.g. taskset).
    first_pre_ts = min((s for ivs in susp.values() for (s, _t) in ivs),
                       default=None)

    # annotate each release with suspended time + #preemptions during its window
    by_name = {}
    for r in rels:
        ivs = susp.get(r.pid, [])
        s_ns = 0
        npre = 0
        for (s, t) in ivs:
            o = overlap_ns(r.begin, r.end, s, t)
            if o > 0:
                s_ns += o
                npre += 1
        susp_ms = s_ns / 1.0e6
        active_ms = r.gpu_ms - susp_ms
        if active_ms < 0:
            active_ms = 0.0
        by_name.setdefault(r.name, []).append(
            dict(pid=r.pid, begin=r.begin, end=r.end, gpu=r.gpu_ms,
                 susp=susp_ms, active=active_ms, npre=npre))

    print("\n(2) Execution-time extension when preempted")
    n_preemptions_total = sum(len(v) for v in susp.values())
    print(f"  reconstructed {n_preemptions_total} preemption interval(s) "
          f"across {len(susp)} pid(s)")
    print("  extension reported two ways (they differ by the suspended time):")
    print("    raw    = gpu_wall(preempted) - baseline gpu_wall   "
          "[correct if cudaEvent already EXCLUDES suspension]")
    print("    active = (gpu_wall - suspended) - baseline         "
          "[correct if cudaEvent INCLUDES suspension]")
    print("  the slope d(gpu_wall)/d(suspended) over preempted releases says "
          "which holds.")

    for name in sorted(by_name):
        recs = by_name[name]
        all_clean = [x for x in recs if x["npre"] == 0]
        idle = [x for x in all_clean
                if first_pre_ts is not None and x["end"] <= first_pre_ts]
        if idle:
            base_recs, base_src = idle, "preemptor-idle window"
        else:
            base_recs, base_src = all_clean, "all non-preempted (no idle window)"
        base_gpu = median([x["gpu"] for x in base_recs]) if base_recs \
            else float("nan")
        base_active = median([x["active"] for x in base_recs]) if base_recs \
            else float("nan")

        pre = [x for x in recs if x["npre"] > 0]
        print(f"\n  [{name}]  releases={len(recs)}  "
              f"non-preempted={len(all_clean)}  preempted={len(pre)}")
        print(f"    baseline ({base_src}, n={len(base_recs)}): "
              f"gpu_wall median = {base_gpu:.3f} ms")
        if not pre:
            continue
        print(f"    gpu_wall when preempted       : "
              f"{fmt(describe([x['gpu'] for x in pre]), 'ms')}")
        print(f"    suspended (blocking)          : "
              f"{fmt(describe([x['susp'] for x in pre]), 'ms')}")
        slope, corr = linfit([x["susp"] for x in pre], [x["gpu"] for x in pre])
        if slope == slope and abs(slope) < 0.3:
            verdict = "~0 -> gpu_wall EXCLUDES suspension; trust 'raw'"
        elif slope == slope and slope > 0.7:
            verdict = "~1 -> gpu_wall INCLUDES suspension; trust 'active'"
        else:
            verdict = "ambiguous (mixed regimes or too few samples)"
        print(f"    d(gpu_wall)/d(suspended)      : slope={slope:.3f} "
              f"r={corr:.3f}   [{verdict}]")
        if base_recs:
            ext_raw = [x["gpu"] - base_gpu for x in pre]
            ext_act = [x["active"] - base_active for x in pre]
            pp_raw = [(x["gpu"] - base_gpu) / x["npre"] for x in pre]
            pp_act = [(x["active"] - base_active) / x["npre"] for x in pre]
            print(f"    EXTENSION raw                 : {fmt(describe(ext_raw), 'ms')}")
            print(f"    EXTENSION raw / preemption    : {fmt(describe(pp_raw), 'ms')}")
            print(f"    EXTENSION active              : {fmt(describe(ext_act), 'ms')}")
            print(f"    EXTENSION active / preemption : {fmt(describe(pp_act), 'ms')}")

    # per-release dump for offline correlation checking (gpu_wall vs suspended)
    if trace_path:
        dump = os.path.splitext(trace_path)[0] + "_perrelease.csv"
        try:
            with open(dump, "w") as f:
                f.write("task_name,pid,seg_begin_ns,seg_done_ns,gpu_wall_ms,"
                        "suspended_ms,active_ms,n_preemptions\n")
                for name in sorted(by_name):
                    for x in by_name[name]:
                        f.write(f"{name},{x['pid']},{x['begin']},{x['end']},"
                                f"{x['gpu']:.4f},{x['susp']:.4f},"
                                f"{x['active']:.4f},{x['npre']}\n")
            print(f"\nPer-release table written to {dump}")
        except OSError as e:
            print(f"\n(could not write per-release dump: {e})")


# --------------------------------------------------------------------------- #
# run a benchmark and capture the kernel log
# --------------------------------------------------------------------------- #
def run_and_capture(kind, duration, extra, events_path, no_sudo):
    sudo = [] if no_sudo else ["sudo"]
    if kind == "microbench":
        binary, trace = ("./preemptOverheadGcaps",
                         "results/workloadBench/preempt_gcaps_trace.csv")
    else:
        binary, trace = ("./workloadTasksetGcaps",
                         "results/workloadBench/taskset_gcaps_trace.csv")
    if not os.path.exists(binary):
        sys.exit(f"{binary} not found — build it first (make {binary[2:]})")

    cmd = sudo + [binary, "-i", "1", "-s", "1", "-d", str(duration)] + extra
    print("clearing kernel log (dmesg -C) ...")
    subprocess.run(sudo + ["dmesg", "-C"], check=False)
    print("running:", " ".join(cmd))
    subprocess.run(cmd, check=False)
    print("capturing GCAPS_EV lines from dmesg ...")
    res = subprocess.run(sudo + ["dmesg"], capture_output=True, text=True)
    ev_lines = [l for l in res.stdout.splitlines() if "GCAPS_EV" in l]
    with open(events_path, "w") as f:
        f.write("\n".join(ev_lines) + ("\n" if ev_lines else ""))
    print(f"  {len(ev_lines)} GCAPS_EV line(s) -> {events_path}")
    if len(ev_lines) == 0:
        print("  WARNING: 0 events. Check: patched driver loaded? -i 1? "
              "kernel.dmesg_restrict (read dmesg as root)? log buffer overflow "
              "(increase CONFIG_LOG_BUF_SHIFT / use a shorter -d)?")
    return events_path, trace


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", choices=("microbench", "taskset"),
                    help="run this benchmark (GCAPS -i 1 -s 1) and capture dmesg")
    ap.add_argument("-d", "--duration", type=int, default=30,
                    help="experiment duration in seconds for --run (default 30)")
    ap.add_argument("--extra", default="",
                    help="extra args passed through to the benchmark binary")
    ap.add_argument("--no-sudo", action="store_true",
                    help="do not prefix dmesg/benchmark with sudo")
    ap.add_argument("--events", help="file containing GCAPS_EV lines to analyse")
    ap.add_argument("--trace", help="benchmark trace CSV to analyse")
    args = ap.parse_args()

    if args.run:
        os.makedirs("results/workloadBench", exist_ok=True)
        tag = "preempt" if args.run == "microbench" else "taskset"
        events_path = f"results/workloadBench/{tag}_gcaps_events.log"
        events_path, trace = run_and_capture(
            args.run, args.duration, args.extra.split() if args.extra else [],
            events_path, args.no_sudo)
        args.events, args.trace = events_path, trace

    if not args.events or not args.trace:
        ap.error("provide --run, or both --events and --trace")

    with open(args.events) as f:
        events = parse_events(f.read())
    rels = load_trace(args.trace)
    report(events, rels, os.path.basename(args.trace), args.trace)


if __name__ == "__main__":
    main()
