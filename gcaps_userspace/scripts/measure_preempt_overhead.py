#!/usr/bin/env python3
"""
Measure GCAPS preemption overhead from the patched driver's GCAPS_EV event ring
plus a benchmark trace.  Two quantities are reported:

  (1) Scheduling + context-switch / preemption overhead.
      Each runlist-update ioctl records a line like

          GCAPS_EV ts=<ns> cpid=<pid> prio=<n> add=<0|1> rlupd=<0|1> \
                   elapsed_us=<eps> preempted=<pid|-1> resumed=<pid|-1> \
                   elapsed_ns=<eps_ns>

      into the driver's event ring, which this script drains from
      /proc/gcaps_events after the run.  The record is no longer printk'd from
      inside the critical section, so it no longer inflates the ε it reports
      (nor the blocking every other task sees).  See scripts/gcaps_events.py.

      `elapsed_ns` is the duration of the ioctl critical section, which performs
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
Run a benchmark and analyse it (needs the patched driver and sudo; resets the
driver's event ring first, then drains it):

    sudo python3 scripts/measure_preempt_overhead.py --run microbench -d 30
    sudo python3 scripts/measure_preempt_overhead.py --run taskset   -d 30

Analyse already-captured data (no device needed):

    python3 scripts/measure_preempt_overhead.py \
        --events results/workloadBench/preempt_gcaps_events.log \
        --trace  results/workloadBench/preempt_gcaps_trace.csv

`--events` accepts any text containing GCAPS_EV lines — a /proc/gcaps_events
drain, or an old `dmesg`/`dmesg | grep GCAPS_EV` capture.  The trace kind
(microbench vs taskset) is detected from its CSV header.

`--from-dmesg` restores the old capture path (stream `dmesg --follow` during
the run) for a driver loaded with gcaps_ev_printk=1.  It is strictly worse:
printk sits back on the ioctl path, and the kernel log buffer holds ~860
records against the ring's 8192.
"""

import argparse
import math
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gcaps_events

# Percentile of non-preempted active time used as the warm-execution baseline.
# Low enough to sit on the warm floor (immune to start-up warmup / contention,
# which only inflate), not the noisy single minimum.
BASELINE_PCTL = 20


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

    def __init__(self, d):
        self.ts = d["ts"]
        self.cpid = d["cpid"]
        self.prio = d["prio"]
        self.add = d["add"]
        self.rlupd = d["rlupd"]
        # microseconds, float: derived from elapsed_ns when the driver gave it,
        # so the small (rlupd=0, ~28 us) mode keeps its sub-us digits.
        self.eps = d["eps_us"]
        self.preempted = d["preempted"]
        self.resumed = d["resumed"]


def parse_events(text):
    return [Event(d) for d in gcaps_events.parse_text(text)]


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
def report(events, rels, label, trace_path=None, baselines=None):
    baselines = baselines or {}
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
    print(f"  baseline = warm floor: p{BASELINE_PCTL} of non-preempted active "
          "time (robust to start-up clock-ramp/cache warmup, which inflates "
          "the first releases).")

    for name in sorted(by_name):
        recs = by_name[name]
        all_clean = [x for x in recs if x["npre"] == 0]
        clean_active = sorted(x["active"] for x in all_clean)
        clean_gpu = sorted(x["gpu"] for x in all_clean)

        pre = [x for x in recs if x["npre"] > 0]
        print(f"\n  [{name}]  releases={len(recs)}  "
              f"non-preempted={len(all_clean)}  preempted={len(pre)}")

        if name in baselines:
            # Isolated measurement supplied by the user (clean uncontended time).
            base_active = base_gpu = baselines[name]
            ctx = (f"  [in-run non-preempted: min {clean_active[0]:.3f}, "
                   f"median {median(clean_active):.3f}]" if clean_active else "")
            print(f"    baseline (override) : {base_active:.3f} ms{ctx}")
        elif clean_active:
            # Steady-state WARM floor of non-preempted active execution. A low
            # percentile isolates the uncontended warm time: start-up warmup and
            # any residual contention only inflate active time above it. NOTE:
            # unreliable when non-preempted releases are bimodal (a contended
            # bulk above a rare clean floor) — pass --baseline-ms NAME=MS then.
            base_active = pct(clean_active, BASELINE_PCTL)
            base_gpu = pct(clean_gpu, BASELINE_PCTL)
            print(f"    baseline warm floor (p{BASELINE_PCTL} of {len(all_clean)} "
                  f"non-preempted) : {base_active:.3f} ms  "
                  f"(min {clean_active[0]:.3f}, median {median(clean_active):.3f}, "
                  f"max {clean_active[-1]:.3f})")
        else:
            base_active = base_gpu = float("nan")
            print("    baseline: (no non-preempted releases — pass "
                  "--baseline-ms NAME=MS)")
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
        if base_active == base_active:  # baseline available (not NaN)
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
# run a benchmark and capture its events
# --------------------------------------------------------------------------- #
def run_and_capture(kind, duration, extra, events_path, no_sudo, from_dmesg):
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

    if not from_dmesg:
        # Drain the driver's ring instead of the kernel log. The records are
        # already in the ring when each ioctl returns, so there is nothing to
        # stream and nothing to flush -- and the ring survives a busy run that
        # would have wrapped the 128 KiB log buffer many times over.
        gcaps_events.require("Or re-run with --from-dmesg.")
        print(f"resetting {gcaps_events.PROC_PATH} ...")
        gcaps_events.reset(sudo=not no_sudo)
        print("running:", " ".join(cmd))
        subprocess.run(cmd, check=False)
        n_ev, n_drop = gcaps_events.drain(events_path)
        print(f"  {n_ev} GCAPS_EV record(s) -> {events_path}")
        if n_drop:
            print(f"  WARNING: the ring overwrote {n_drop} older record(s). "
                  f"Raise GCAPS_EV_RING_SIZE or shorten the run (-d); the "
                  f"analysis below sees only the last {n_ev}.")
        if n_ev == 0:
            print("  WARNING: 0 events. Check: patched driver loaded? -i 1?")
        return events_path, trace

    print("clearing kernel log (dmesg -C) ...")
    subprocess.run(sudo + ["dmesg", "-C"], check=False)
    # Stream the kernel log DURING the run (dmesg --follow) instead of dumping
    # the ring buffer afterwards: the default ~128 KiB buffer holds only ~860
    # GCAPS_EV lines, so a post-run dump silently keeps just the LAST few
    # seconds of a busy run. Events lost that way make their releases look
    # "non-preempted" (poisoning the in-run warm floor with wall times that
    # still include unattributed suspension) and undercount preemptions.
    print("running:", " ".join(cmd), " [streaming dmesg --follow]")
    raw_path = events_path + ".raw"
    with open(raw_path, "w") as rawf:
        follower = subprocess.Popen(sudo + ["dmesg", "--follow"],
                                    stdout=rawf, stderr=subprocess.DEVNULL)
        try:
            subprocess.run(cmd, check=False)
            time.sleep(1.0)                    # let trailing events flush
        finally:
            follower.terminate()
            try:
                follower.wait(timeout=5)
            except subprocess.TimeoutExpired:
                follower.kill()
    with open(raw_path) as f:
        ev_lines = [l.rstrip("\n") for l in f if "GCAPS_EV" in l]
    os.remove(raw_path)
    with open(events_path, "w") as f:
        f.write("\n".join(ev_lines) + ("\n" if ev_lines else ""))
    print(f"  {len(ev_lines)} GCAPS_EV line(s) -> {events_path}")
    if len(ev_lines) == 0:
        print("  WARNING: 0 events. Check: patched driver loaded? -i 1? "
              "nvgpu loaded with gcaps_ev_printk=1? "
              "kernel.dmesg_restrict (read dmesg as root)?")
    return events_path, trace


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", choices=("microbench", "taskset"),
                    help="run this benchmark (GCAPS -i 1 -s 1) and drain "
                         "/proc/gcaps_events")
    ap.add_argument("-d", "--duration", type=int, default=30,
                    help="experiment duration in seconds for --run (default 30)")
    ap.add_argument("--extra", default="",
                    help="extra args passed through to the benchmark binary")
    ap.add_argument("--no-sudo", action="store_true",
                    help="do not prefix the benchmark / ring reset with sudo")
    ap.add_argument("--from-dmesg", action="store_true",
                    help="capture from the kernel log instead of "
                         "/proc/gcaps_events (needs nvgpu gcaps_ev_printk=1; "
                         "puts printk back on the measured ioctl path)")
    ap.add_argument("--events", help="file containing GCAPS_EV lines to analyse")
    ap.add_argument("--trace", help="benchmark trace CSV to analyse")
    ap.add_argument("--baseline-ms", action="append", default=[],
                    metavar="NAME=MS",
                    help="override a task's baseline active-execution time (ms) "
                         "with a separate isolated measurement instead of the "
                         "in-run warm floor, e.g. --baseline-ms mm_victim=27.0 "
                         "(repeatable, or comma-separated). Use this when a "
                         "task's non-preempted releases are contended/bimodal "
                         "and the in-run floor is unreliable.")
    args = ap.parse_args()

    if args.run:
        os.makedirs("results/workloadBench", exist_ok=True)
        tag = "preempt" if args.run == "microbench" else "taskset"
        events_path = f"results/workloadBench/{tag}_gcaps_events.log"
        events_path, trace = run_and_capture(
            args.run, args.duration, args.extra.split() if args.extra else [],
            events_path, args.no_sudo, args.from_dmesg)
        args.events, args.trace = events_path, trace

    if not args.events or not args.trace:
        ap.error("provide --run, or both --events and --trace")

    baselines = {}
    for item in args.baseline_ms:
        for part in item.split(","):
            part = part.strip()
            if not part:
                continue
            if "=" not in part:
                ap.error(f"--baseline-ms expects NAME=MS, got '{part}'")
            name, val = part.split("=", 1)
            try:
                baselines[name.strip()] = float(val)
            except ValueError:
                ap.error(f"--baseline-ms value not a number: '{part}'")

    with open(args.events) as f:
        events = parse_events(f.read())
    rels = load_trace(args.trace)
    report(events, rels, os.path.basename(args.trace), args.trace, baselines)


if __name__ == "__main__":
    main()
