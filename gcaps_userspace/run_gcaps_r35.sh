#!/bin/bash
# Preflight + run wrapper for the GCAPS userspace on L4T R35 (JetPack 5.x).
#
# WHY THIS EXISTS -- two failures on 2026-08-28, both silent:
#
#   1. FAKE PASS.  This kernel has CONFIG_RT_GROUP_SCHED=y, and systemd gives
#      user session cgroups cpu.rt_runtime_us = 0.  sched_setscheduler() then
#      returns EPERM *even as root*, five of six tasks drop out, and the one
#      best-effort task runs alone to completion.  The run LOOKS like a pass.
#      The driver logs prio=0 and takes the best-effort branch throughout.
#
#   2. BOARD FREEZE.  RT busy-wait tasks (suspension=0) that deadlock will
#      saturate their CPUs at FIFO 66-70; systemd stops petting the Tegra
#      watchdog (120 s) and the board hard-resets, destroying the evidence.
#
# So: this script proves RT actually works BEFORE running, bounds the run well
# inside the watchdog, and -- crucially -- verifies AFTERWARDS that the driver
# saw non-zero priorities.  A run that fails that check is void, not a pass.
#
#   sudo ./run_gcaps_r35.sh -f taskset.csv -d 10 -i 1
#   sudo ./run_gcaps_r35.sh --check              # preflight only, run nothing
#   sudo ./run_gcaps_r35.sh --timeout 60 -f ... -d 20 -i 1
#
# To drive a binary other than ./main, the assignment must come AFTER sudo --
# sudo's env_reset drops variables set before it, and the wrapper then silently
# falls back to ./main (check the "ok binary ..." preflight line):
#
#   sudo GCAPS_MAIN=./workloadTasksetGcaps ./run_gcaps_r35.sh -i 1 -s 1 -d 10
#
# --timeout must stay UNDER the watchdog (120 s), so a binary whose legitimate
# runtime exceeds that cannot be supervised here.  workloadTasksetGcaps is one:
# its post-run verify serialises by GCAPS priority and mlp_1024x8 spends ~a
# minute on a host-side reference pass.  Keep -d small.  A SIGKILL during that
# phase does NOT lose data -- the per-task CSVs are written BEFORE verify runs
# -- it only costs the correctness verdict.
#
# Unlike the JP7.2 wrapper this moves only ITSELF into the root cpu cgroup, so
# it leaves no lasting change to your shell or session.

set -uo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Resolve ./main.  NOTE: under sudo, $HOME is /root -- never use it here.
# Prefer an explicit override, then the script's own dir, then the directory the
# user invoked from, then the invoking user's real home.
USER_HOME="$(getent passwd "${SUDO_USER:-$(id -un)}" 2>/dev/null | cut -d: -f6)"
MAIN=""
for cand in "${GCAPS_MAIN:-}" \
            "$SELF_DIR/main" \
            "$PWD/main" \
            "${USER_HOME:-/home/nvidia}/GCAPS/gcaps-super-repo/gcaps_userspace/main"; do
    [[ -n "$cand" && -x "$cand" ]] && { MAIN="$cand"; break; }
done
[[ -n "$MAIN" ]] || MAIN="$PWD/main"          # for the error message below
RUN_DIR="$(dirname "$MAIN")"

CPU_CG=/sys/fs/cgroup/cpu,cpuacct
RAILGATE=$(ls /sys/devices/platform/*.ga10b/railgate_enable 2>/dev/null | head -1)
GCAPS_SYM=nvgpu_ioctl_runlist_update_rt_prio
WDT_SEC=120
TIMEOUT=40

fail() { echo "PREFLIGHT FAIL: $*" >&2; exit 1; }
ok()   { echo "  ok    $*"; }
warn() { echo "  WARN  $*" >&2; }

CHECK_ONLY=0
ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --check)   CHECK_ONLY=1; shift ;;
        --timeout) TIMEOUT="$2"; shift 2 ;;
        *)         ARGS+=("$1"); shift ;;
    esac
done

echo "== GCAPS R35 preflight =="

[[ $EUID -eq 0 ]] || fail "must run as root (SCHED_FIFO, cgroup and railgate control need it)"
[[ -x "$MAIN" ]]  || fail "no ./main found. Looked at: \$GCAPS_MAIN, $SELF_DIR/main,
       $PWD/main, ${USER_HOME:-/home/nvidia}/GCAPS/gcaps-super-repo/gcaps_userspace/main
       Build it (make main -j8), cd to its directory, or set GCAPS_MAIN=<path>."
ok "binary $MAIN"

# ---- 1. ioctl number: R35 uses 43.  49 is the r39.2 number and would call
#         NVGPU_GPU_IOCTL_GET_GPC_* instead -- returning 0 and doing nothing.
gcaps_mode=0; prev=""
for a in "${ARGS[@]:-}"; do
    [[ "$prev" == "-i" && "$a" == "1" ]] && gcaps_mode=1
    prev="$a"
done

if command -v objdump >/dev/null 2>&1; then
    n43=$(objdump -d "$MAIN" 2>/dev/null | grep -ci '#0x472b')
    n49=$(objdump -d "$MAIN" 2>/dev/null | grep -ci '#0x4731')
    if [[ "$n49" -gt 0 && "$n43" -eq 0 ]]; then
        fail "binary encodes ioctl 49 (the r39.2 number). On R35 that is a
       different ioctl entirely -- GCAPS would be silently INERT.
       Rebuild against the vendored uapi (ioctl 43)."
    elif [[ "$n43" -gt 0 ]]; then
        ok "binary encodes ioctl 43"
    else
        warn "found neither ioctl constant in $MAIN -- check manually"
    fi
fi

# ---- 2. GCAPS driver actually loaded (only matters for -i 1)
if [[ $gcaps_mode -eq 1 ]]; then
    grep -qw "$GCAPS_SYM" /proc/kallsyms 2>/dev/null \
        || fail "'-i 1' requested but the STOCK nvgpu is loaded.
       Switch to the GCAPS driver and reboot."
    ok "GCAPS driver loaded"
else
    ok "TSG baseline mode (-i 0): driver variant irrelevant"
fi

# ---- 3. THE IMPORTANT ONE: make SCHED_FIFO actually attainable.
#         With RT_GROUP_SCHED, a task needs RT bandwidth in ITS cgroup.
if [[ -f "$CPU_CG/cgroup.procs" ]]; then
    root_rt=$(cat "$CPU_CG/cpu.rt_runtime_us" 2>/dev/null || echo 0)
    if [[ "$root_rt" -le 0 ]]; then
        fail "root cpu cgroup has no RT bandwidth (cpu.rt_runtime_us=$root_rt)"
    fi
    if echo $$ > "$CPU_CG/cgroup.procs" 2>/dev/null; then
        ok "moved into the root cpu cgroup (rt_runtime_us=$root_rt)"
    else
        warn "could not move into the root cpu cgroup -- RT may fail"
    fi
else
    ok "no cgroup-v1 cpu controller -- RT bandwidth not cgroup-gated"
fi

chrt -f 50 true 2>/dev/null \
    || fail "SCHED_FIFO still unavailable after the cgroup move.
       Every RT task would drop to best-effort and the run would be a FAKE PASS."
ok "SCHED_FIFO verified attainable"

# ---- 4. railgating off: the ioctl writes runlist registers without taking a
#         gk20a_busy() power reference (defect 1), so a railgated GPU means
#         writes into an unmapped BAR.
railgate_changed=0
if [[ -n "$RAILGATE" && -w "$RAILGATE" ]]; then
    if [[ "$(cat "$RAILGATE")" == "0" ]]; then
        ok "railgating already disabled"
    else
        echo 0 > "$RAILGATE" && railgate_changed=1 && ok "railgating DISABLED (restored on exit)"
    fi
else
    warn "railgate control not found -- cannot rule out defect 1"
fi
restore_railgate() {
    [[ $railgate_changed -eq 1 ]] && echo 1 > "$RAILGATE" 2>/dev/null && echo "  restored railgate_enable=1"
}
trap restore_railgate EXIT

# ---- 5. keep the run inside the watchdog window
if [[ -r /sys/class/watchdog/watchdog0/timeout ]]; then
    WDT_SEC=$(cat /sys/class/watchdog/watchdog0/timeout)
fi
if [[ "$TIMEOUT" -ge "$WDT_SEC" ]]; then
    fail "--timeout $TIMEOUT >= watchdog $WDT_SEC s. A hang would reset the board
       and destroy the log. Pick something well under $WDT_SEC."
fi
ok "run capped at ${TIMEOUT}s (watchdog ${WDT_SEC}s)"

# ---- 6. persistent journal, so a hang leaves evidence
[[ -d /var/log/journal ]] && ok "journal is persistent" \
    || warn "journal is VOLATILE -- a reboot will destroy the crash log"

# ---- 7. the driver's GCAPS_EV ring. The driver no longer printk's the records
#         from inside the ioctl critical section (that inflated the epsilon it
#         reported and the blocking every other task saw), so the post-run
#         verification below reads the ring instead of the journal.
GCAPS_EV_PROC=/proc/gcaps_events
have_ev_ring=0
if [[ $gcaps_mode -eq 1 ]]; then
    if [[ -e "$GCAPS_EV_PROC" ]]; then
        have_ev_ring=1
        ok "event ring at $GCAPS_EV_PROC"
    else
        warn "no $GCAPS_EV_PROC -- driver predates the event ring.
      Falling back to the journal, which needs nvgpu gcaps_ev_printk=1
      and only holds the last few seconds of a busy run."
    fi
fi

echo "== preflight passed =="
[[ $CHECK_ONLY -eq 1 ]] && { echo "(--check: not running)"; exit 0; }
[[ ${#ARGS[@]} -gt 0 ]] || fail "no arguments for ./main (e.g. -f taskset.csv -d 10 -i 1)"

cd "$RUN_DIR" || exit 1
mkdir -p timelog
START=$(date +%s)

# Empty the ring so the counts below describe THIS run and nothing else.
# MUST be a real write: the driver resets on write(), and ': > file' truncates
# via open(O_TRUNC) without ever issuing one, so it silently does nothing and
# the next run's counts include the previous run's events.
if [[ $have_ev_ring -eq 1 ]]; then
    echo > "$GCAPS_EV_PROC"
    # grep -c PRINTS 0 and EXITS 1 when there are no matches, so '|| echo 0'
    # appends a second line and $left becomes "0\n0" -- a [[ ]] syntax error.
    left=$(grep -c '^GCAPS_EV' "$GCAPS_EV_PROC" 2>/dev/null || true)
    left=${left:-0}
    [[ "$left" -eq 0 ]] || warn "ring still holds $left record(s) after reset --
      the post-run counts below will include earlier runs"
fi

echo "== running: $MAIN ${ARGS[*]} =="
timeout -s KILL "$TIMEOUT" "$MAIN" "${ARGS[@]}"
rc=$?
echo "== main exited rc=$rc ==" 
[[ $rc -eq 137 ]] && echo "   (137 = killed by the ${TIMEOUT}s cap -- it HUNG)"

# ---- 8. POST-RUN: did the driver actually see real-time priorities?
#         This is what catches the fake pass after the fact.
if [[ $gcaps_mode -eq 1 ]]; then
    echo "== post-run verification =="
    if [[ $have_ev_ring -eq 1 ]]; then
        EV=$(grep GCAPS_EV "$GCAPS_EV_PROC" 2>/dev/null)
        n_drop=$(sed -n 's/^#.*dropped=\([0-9]*\).*/\1/p' "$GCAPS_EV_PROC")
    else
        EV=$(journalctl -k --since "@$START" --no-pager 2>/dev/null | grep GCAPS_EV)
        n_drop=0
    fi
    n_ev=$(grep -c GCAPS_EV <<< "$EV")
    n_add=$(grep -c 'add=1' <<< "$EV")
    n_rem=$(grep -c 'add=0' <<< "$EV")
    # filter to GCAPS_EV first: an empty $EV is one empty line to <<<, which
    # 'grep -v' would happily count as a non-zero-priority event
    n_rt=$(grep GCAPS_EV <<< "$EV" | grep -vc 'prio=0 ')
    echo "  GCAPS_EV events : $n_ev   (add=1: $n_add, add=0: $n_rem)"
    echo "  non-zero prio   : $n_rt"
    [[ "${n_drop:-0}" -gt 0 ]] && \
        warn "the ring overwrote $n_drop record(s) -- shorten -d or raise
      GCAPS_EV_RING_SIZE before trusting an epsilon from this run"

    if [[ "$n_ev" -eq 0 ]]; then
        echo "  *** VOID: no GCAPS_EV events -- the ioctl never reached the driver." >&2
        exit 1
    elif [[ "$n_rt" -eq 0 ]]; then
        echo "  *** VOID: every event has prio=0. RT was not in effect;" >&2
        echo "      this is the FAKE PASS, not a result." >&2
        exit 1
    elif [[ "$n_rem" -eq 0 ]]; then
        echo "  *** HANG SIGNATURE: add=1 with no add=0 -- no task ever" >&2
        echo "      completed a GPU segment." >&2
    else
        echo "  OK: real-time priorities in effect, segments opening and closing"
    fi
fi
exit $rc
