#!/usr/bin/env python3
"""
Extract the worst-case GCAPS runlist-update overhead (epsilon) from the kernel
log lines emitted by the patched nvgpu driver, e.g.:

    [   70.609387] process 3488 elapsed time: 1265

Each value is one runlist-update IOCTL duration in microseconds (GCAPS epsilon =
alpha + theta, Def. 2 of the ECRTS'24 paper).  The distribution is typically
bimodal: a small mode (IOCTL calls that did not require an actual runlist update)
and the real runlist-update mode.  The maximum can be fed into the schedulability
analysis as the worst-case epsilon:

    python3 ../../analysis/experiments.py -g 1 -e 4 -n 200 --gcaps-overhead-us <MAX>

This reports epsilon over *all* ioctl calls (admissions, rejections, removals,
preemptions).  To isolate the overhead of an actual *preemption* (a higher-prio
job evicting a running lower-prio one) and the resulting execution-time
extension, use measure_preempt_overhead.py, which reads the richer GCAPS_EV log.

The patched driver emits both an old-style line and a structured one:

    [   70.609387] process 3488 elapsed time: 1265
    [   70.609390] GCAPS_EV ts=70609390123 cpid=3488 prio=4 add=1 rlupd=1 \
                   elapsed_us=1265 preempted=3490 resumed=-1

This script prefers the GCAPS_EV `elapsed_us` field when present (so it also
works on a `dmesg | grep GCAPS_EV` capture) and falls back to the legacy line.

Usage:
    python3 analyse_gcaps_overhead.py [elapsed_times.txt]   # file arg, or stdin
    dmesg | grep -E 'GCAPS_EV|elapsed time' | python3 analyse_gcaps_overhead.py
"""
import sys
import re
import math


def main():
    src = open(sys.argv[1]) if len(sys.argv) > 1 else sys.stdin
    with src:
        text = src.read()
    # Prefer the structured GCAPS_EV line; fall back to the legacy line. Using
    # only one avoids double-counting when both are present in the same buffer.
    ev = re.findall(r'GCAPS_EV\b[^\n]*?elapsed_us=(\d+)', text)
    vals = [int(x) for x in ev] if ev else \
        [int(m) for m in re.findall(r'elapsed time:\s*(\d+)', text)]

    if not vals:
        print("No 'elapsed time' samples found.", file=sys.stderr)
        sys.exit(1)

    vals.sort()
    n = len(vals)

    def pct(p):  # nearest-rank percentile
        return vals[max(1, math.ceil(p / 100.0 * n)) - 1]

    print(f"samples : {n}")
    print(f"min     : {vals[0]} us")
    print(f"mean    : {sum(vals) / n:.1f} us")
    print(f"median  : {pct(50)} us")
    print(f"p95     : {pct(95)} us")
    print(f"p99     : {pct(99)} us")
    print(f"max     : {vals[-1]} us")
    print()
    print(f"Worst-case GCAPS overhead (epsilon) = {vals[-1]} us")
    print(f"  -> pass to the analysis with: --gcaps-overhead-us {vals[-1]}")


if __name__ == "__main__":
    main()
