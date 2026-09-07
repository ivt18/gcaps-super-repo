#!/usr/bin/env python3
"""
Shared access to the patched nvgpu driver's GCAPS_EV measurement records.

WHERE THE RECORDS COME FROM
---------------------------
The driver used to emit each record with pr_info() from *inside* the cs_lock
critical section — inside the very interval the record reports, and inside the
interval every other GCAPS task blocks on.  It no longer does: the ioctl stores
the record into a lock-free ring (after the unlock, after the end-of-section
ktime_get()), and the ring is drained out of band through

    /proc/gcaps_events            # read  -> "#" summary + one GCAPS_EV per line
    echo > /proc/gcaps_events     # write -> reset the ring (root)

Nothing is formatted or printed on the measured path any more, so neither the
reported ε nor the benchmark's response window carries printk cost.

The line format is unchanged apart from one appended field:

    GCAPS_EV ts=<ns> cpid=<pid> prio=<n> add=<0|1> rlupd=<0|1> \
             elapsed_us=<eps> preempted=<pid|-1> resumed=<pid|-1> \
             elapsed_ns=<eps_ns>

`elapsed_ns` is the same interval untruncated.  It matters: the no-op path
(rlupd=0) is sub-microsecond, so `elapsed_us` reported it as a flat 0.  Every
reader here prefers `elapsed_ns` when present and falls back to `elapsed_us`,
so old captures still parse.

The ring holds 8192 records — over an order of magnitude more than the ~860
GCAPS_EV lines that fit in the default 128 KiB kernel log buffer, which is what
forced the old capture path to stream `dmesg --follow` during the run.

LEGACY CAPTURES still work: pass any text containing GCAPS_EV lines (a dmesg
dump, a saved grep) to load_file().  Setting the driver's `gcaps_ev_printk=1`
module parameter restores the kernel-log lines for debugging — but printk is
back on the ioctl path when it is on, so do not measure with it set.
"""

import os
import re
import subprocess
import sys

PROC_PATH = "/proc/gcaps_events"

# `elapsed_ns` is optional so that dmesg captures from the older driver, which
# never emitted it, still parse.
GCAPS_EV_RE = re.compile(
    r"GCAPS_EV\s+ts=(?P<ts>\d+)\s+cpid=(?P<cpid>-?\d+)\s+prio=(?P<prio>-?\d+)"
    r"\s+add=(?P<add>\d+)\s+rlupd=(?P<rlupd>\d+)"
    r"\s+elapsed_us=(?P<elapsed_us>-?\d+)"
    r"\s+preempted=(?P<preempted>-?\d+)\s+resumed=(?P<resumed>-?\d+)"
    r"(?:\s+elapsed_ns=(?P<elapsed_ns>-?\d+))?"
)

# "# GCAPS_EV records=<n> dropped=<n> ring=<n>", the first line /proc emits.
SUMMARY_RE = re.compile(
    r"#\s*GCAPS_EV\s+records=(\d+)\s+dropped=(\d+)\s+ring=(\d+)"
)


def parse_line(line):
    """One GCAPS_EV line -> dict, or None if the line is not one.

    `eps_us` is a float derived from elapsed_ns when the driver supplied it, so
    sub-microsecond no-op ioctls are no longer all reported as exactly 0.
    """
    m = GCAPS_EV_RE.search(line)
    if not m:
        return None
    ns = m.group("elapsed_ns")
    return {
        "ts": int(m.group("ts")),
        "cpid": int(m.group("cpid")),
        "prio": int(m.group("prio")),
        "add": int(m.group("add")),
        "rlupd": int(m.group("rlupd")),
        "eps_us": (int(ns) / 1000.0) if ns is not None
                  else float(m.group("elapsed_us")),
        "eps_ns": int(ns) if ns is not None
                  else int(m.group("elapsed_us")) * 1000,
        "exact_ns": ns is not None,
        "preempted": int(m.group("preempted")),
        "resumed": int(m.group("resumed")),
    }


def parse_text(text):
    """All GCAPS_EV records in a blob of text, sorted by ts."""
    evs = [e for e in (parse_line(l) for l in text.splitlines()) if e]
    evs.sort(key=lambda e: e["ts"])
    return evs


def load_file(path):
    """Records from a captured file (procfs drain, dmesg dump, saved grep)."""
    with open(path) as f:
        return parse_text(f.read())


def dropped_in(text):
    """Records the ring overwrote before it was drained, or None if unknown."""
    m = SUMMARY_RE.search(text)
    return int(m.group(2)) if m else None


# --------------------------------------------------------------------------- #
# live capture
# --------------------------------------------------------------------------- #
def available():
    return os.path.exists(PROC_PATH)


def require(hint=""):
    """Exit with actionable guidance when the driver has no event ring."""
    if available():
        return
    sys.exit(
        f"{PROC_PATH} not found.\n"
        "  The loaded nvgpu.ko predates the GCAPS event ring (or is the stock\n"
        "  driver). Rebuild and install the patched module -- or, to capture\n"
        "  from the kernel log instead, load nvgpu with gcaps_ev_printk=1.\n"
        f"  {hint}".rstrip()
    )


def reset(sudo=True):
    """Empty the ring so a capture starts from a known state."""
    cmd = (["sudo"] if sudo else []) + ["sh", "-c", f": > {PROC_PATH}"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"could not reset {PROC_PATH}: {r.stderr.strip()}\n"
                 "  (resetting the ring needs root)")


def drain(dest_path):
    """Copy the ring into dest_path.  Returns (n_records, n_dropped).

    Reading needs no privilege (the entry is 0644); only reset() does.
    """
    with open(PROC_PATH) as f:
        text = f.read()
    with open(dest_path, "w") as f:
        f.write(text)
    n_rec = len(parse_text(text))
    return n_rec, dropped_in(text)
