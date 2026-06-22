#!/usr/bin/env python3
"""
Extract the worst-case GCAPS runlist-update overhead (epsilon) from the kernel
log lines emitted by the patched nvgpu driver, e.g.:

    [   70.609387] process 3488 elapsed time: 1265

Each "elapsed time" value is one runlist-update IOCTL duration in microseconds
(GCAPS epsilon = alpha + theta, Def. 2 of the ECRTS'24 paper).  The distribution
is typically bimodal: a small mode (IOCTL calls that did not require an actual
runlist update) and the real runlist-update mode.  The maximum can be fed into
the schedulability analysis as the worst-case epsilon:

    python3 ../../analysis/experiments.py -g 1 -e 4 -n 200 --gcaps-overhead-us <MAX>

Usage:
    python3 analyse_gcaps_overhead.py [elapsed_times.txt]   # file arg, or stdin
    dmesg | grep 'elapsed time' | python3 analyse_gcaps_overhead.py
"""
import sys
import re
import math


def main():
    src = open(sys.argv[1]) if len(sys.argv) > 1 else sys.stdin
    with src:
        vals = [int(m) for m in re.findall(r'elapsed time:\s*(\d+)', src.read())]

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
