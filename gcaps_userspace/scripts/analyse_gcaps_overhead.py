#!/usr/bin/env python3
"""
Extract the worst-case GCAPS runlist-update overhead (epsilon) from the records
the patched nvgpu driver keeps for each runlist-update IOCTL.

Each value is one IOCTL critical-section duration (GCAPS epsilon = alpha +
theta, Def. 2 of the ECRTS'24 paper).  The distribution is typically bimodal: a
small mode (IOCTL calls that did not require an actual runlist update) and the
real runlist-update mode.  The maximum can be fed into the schedulability
analysis as the worst-case epsilon:

    python3 ../../analysis/experiments.py -g 1 -e 4 -n 200 --gcaps-overhead-us <MAX>

This reports epsilon over *all* ioctl calls (admissions, rejections, removals,
preemptions).  To isolate the overhead of an actual *preemption* (a higher-prio
job evicting a running lower-prio one) and the resulting execution-time
extension, use measure_preempt_overhead.py, which reads the same records but
also pairs them against a benchmark trace.

Where the records come from
---------------------------
The driver stores them in a lock-free ring instead of printk-ing them from
inside the critical section — which used to inflate the very epsilon reported
here, and the blocking every other task saw.  Drain the ring after a run:

    sudo ./workloadTasksetGcaps -i 1 -s 1 -d 30
    python3 analyse_gcaps_overhead.py /proc/gcaps_events

With no argument and no piped input the ring is read for you.  A file argument
or stdin is used instead when given, so saved drains and legacy kernel-log
captures still work:

    python3 analyse_gcaps_overhead.py                        # /proc/gcaps_events
    python3 analyse_gcaps_overhead.py events.log             # a saved drain
    dmesg | grep -E 'GCAPS_EV|elapsed time' | python3 analyse_gcaps_overhead.py

The driver's `elapsed_ns` field is preferred over `elapsed_us` when present.
That matters at the small mode: the no-op IOCTL path is sub-microsecond, so
`elapsed_us` truncated it to a flat 0 and made that mode unreadable.  Captures
without `elapsed_ns` (older driver, or `gcaps_ev_printk=1` dmesg lines) fall
back to microseconds, and the legacy `process <pid> elapsed time: <us>` line is
used if no structured record is present at all.
"""
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gcaps_events


def main():
    if len(sys.argv) > 1:
        src = open(sys.argv[1])
    elif sys.stdin.isatty():
        gcaps_events.require()
        src = open(gcaps_events.PROC_PATH)
    else:
        src = sys.stdin
    with src:
        text = src.read()

    # Prefer the structured record; fall back to the legacy printk line. Using
    # only one avoids double-counting when both are present in the same buffer.
    evs = gcaps_events.parse_text(text)
    if evs:
        vals = [e["eps_us"] for e in evs]
        exact = all(e["exact_ns"] for e in evs)
    else:
        vals = [float(m) for m in re.findall(r'elapsed time:\s*(\d+)', text)]
        exact = False

    if not vals:
        print("No epsilon samples found.", file=sys.stderr)
        sys.exit(1)

    dropped = gcaps_events.dropped_in(text)
    if dropped:
        print(f"WARNING: the driver's ring overwrote {dropped} older record(s) "
              f"before this drain; everything below is over the survivors only.",
              file=sys.stderr)

    vals.sort()
    n = len(vals)

    def pct(p):  # nearest-rank percentile
        return vals[max(1, math.ceil(p / 100.0 * n)) - 1]

    print(f"samples    : {n}")
    print(f"resolution : {'ns' if exact else 'us (truncated by the driver)'}")
    print(f"min        : {vals[0]:.3f} us")
    print(f"mean       : {sum(vals) / n:.3f} us")
    print(f"median     : {pct(50):.3f} us")
    print(f"p95        : {pct(95):.3f} us")
    print(f"p99        : {pct(99):.3f} us")
    print(f"max        : {vals[-1]:.3f} us")
    print()
    # The analysis takes an integer microsecond bound, so round UP: it has to
    # stay an upper bound on every sample observed.
    print(f"Worst-case GCAPS overhead (epsilon) = {vals[-1]:.3f} us")
    print(f"  -> pass to the analysis with: "
          f"--gcaps-overhead-us {math.ceil(vals[-1])}")


if __name__ == "__main__":
    main()
