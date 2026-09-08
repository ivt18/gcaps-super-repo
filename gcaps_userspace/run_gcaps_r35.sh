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
#   sudo ./run_gcaps_r35.sh --no-platform ...    # leave clocks/cpuidle alone
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
        --no-platform) PLATFORM=0; shift ;;
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

# ---- 4b. platform state: locked clocks, no deep idle
#
# WHY: measured on this board 2026-09-08 -- with the governor at schedutil and
# c7 (5000 us exit latency) enabled, cpuWakeupLatencyGcaps reported a CPU<->GPU
# clock calibration std of 17.9 us against a W_i of 27.6 us.  The calibration
# was noisier than the quantity being measured, and the drift check EXCEEDED its
# 10 us budget.  Same root cause as the rho_head overruns in the RTA work: they
# vanished with clocks locked and c7 off.
#
# The board is SHARED and this is global state, so everything is saved and put
# back on exit -- including on SIGINT/SIGTERM -- and --no-platform skips it.
nvpm_prior=""
clock_store="/var/tmp/gcaps_l4t_dfs.conf"
clock_stored=0
idle_disabled=()

if [[ ${PLATFORM:-1} -eq 1 ]]; then
    if command -v nvpmodel >/dev/null 2>&1; then
        # 'nvpmodel -q' prints the mode NAME then the mode NUMBER; 0 is MAXN.
        nvpm_prior=$(nvpmodel -q 2>/dev/null | sed -n '2p' | tr -dc '0-9')
        if [[ -n "$nvpm_prior" && "$nvpm_prior" != "0" ]]; then
            if nvpmodel -m 0 >/dev/null 2>&1; then
                ok "nvpmodel mode $nvpm_prior -> 0 (MAXN, restored on exit)"
            else
                warn "nvpmodel -m 0 failed"; nvpm_prior=""
            fi
        else
            ok "nvpmodel already mode ${nvpm_prior:-?}"
            nvpm_prior=""          # nothing to put back
        fi
    fi

    # jetson_clocks gets a timeout and a closed stdin.  Sending its output to
    # /dev/null is what turns a stall or a prompt into an unexplained freeze with
    # no indication of which preflight step is stuck, so: announce first, bound
    # the wait, and report the exit status.
    if command -v jetson_clocks >/dev/null 2>&1; then
        # jetson_clocks --store PROMPTS ("File ... already exists. Can I
        # overwrite it? Y/N:") when the target exists -- /usr/bin/jetson_clocks
        # line ~700.  With its output sent to /dev/null that prompt is invisible
        # and 'read answer' blocks on the terminal, which is exactly how a run
        # hung here with no indication of the step.  Closing stdin alone would
        # only convert the hang into a permanent "cannot lock clocks", so remove
        # the file first: it is ours (script-owned path), written fresh each run
        # and consumed by the restore below.
        rm -f "$clock_store"
        echo "  ..    storing current clocks (jetson_clocks --store)"
        if timeout 60 jetson_clocks --store "$clock_store" </dev/null >/dev/null 2>&1; then
            clock_stored=1
            echo "  ..    locking clocks (jetson_clocks)"
            rc_jc=0
            timeout 60 jetson_clocks </dev/null >/dev/null 2>&1 || rc_jc=$?
            if [[ $rc_jc -eq 0 ]]; then
                ok "clocks LOCKED to max (restored on exit)"
            elif [[ $rc_jc -eq 124 ]]; then
                warn "jetson_clocks TIMED OUT after 60 s -- DVFS still active and
      the clocks may be half-applied; timings will be noisy"
            else
                warn "jetson_clocks failed (rc=$rc_jc) -- DVFS still active"
            fi
        else
            warn "jetson_clocks --store failed or timed out -- NOT locking clocks,
      since they could not be put back afterwards"
        fi
    fi

    # c7 is matched by NAME, not by state index -- the index is not guaranteed.
    # Count the outcomes separately: "found none", "already off" and "could not
    # write" are three different things, and collapsing them reports all-clear
    # for a loop that examined nothing (which is what an unprivileged run did).
    c7_total=0; c7_already=0
    for st in /sys/devices/system/cpu/cpu*/cpuidle/state*/; do
        [[ -r "$st/name" ]] || continue
        [[ "$(cat "$st/name")" == "c7" ]] || continue
        c7_total=$((c7_total + 1))
        if [[ "$(cat "$st/disable" 2>/dev/null)" == "1" ]]; then
            c7_already=$((c7_already + 1))
            continue
        fi
        { echo 1 > "$st/disable"; } 2>/dev/null && idle_disabled+=("$st/disable")
    done
    n_dis=${#idle_disabled[@]}
    n_fail=$((c7_total - c7_already - n_dis))
    if (( c7_total == 0 )); then
        warn "no c7 idle state found -- cannot rule out a deep-idle exit latency
      in the measurements"
    elif (( n_fail > 0 )); then
        warn "could NOT disable c7 on $n_fail of $c7_total cpu(s) -- its 5 ms exit
      latency will land in the timings"
    elif (( n_dis > 0 )); then
        ok "c7 deep idle DISABLED on $n_dis of $c7_total cpu(s) (restored on exit)"
    else
        ok "c7 deep idle already off on all $c7_total cpu(s)"
    fi
else
    warn "--no-platform: clocks and cpuidle left alone -- expect a noisy clock
      calibration and W_i you cannot trust"
fi

restore_platform() {
    local n=0 f
    for f in "${idle_disabled[@]:-}"; do
        [[ -n "$f" ]] && { echo 0 > "$f"; } 2>/dev/null && n=$((n + 1))
    done
    (( n )) && echo "  restored c7 deep idle on $n cpu(s)"
    # nvpmodel first: it re-clamps the frequency table, so restoring it after
    # jetson_clocks would undo the restored DVFS state.
    [[ -n "$nvpm_prior" ]] && nvpmodel -m "$nvpm_prior" >/dev/null 2>&1 \
        && echo "  restored nvpmodel mode $nvpm_prior"
    [[ $clock_stored -eq 1 ]] \
        && timeout 60 jetson_clocks --restore "$clock_store" </dev/null >/dev/null 2>&1 \
        && echo "  restored clocks from $clock_store"
    return 0
}

# ONE trap: a second 'trap ... EXIT' would REPLACE the first, silently leaving
# the railgate (or the platform state) unrestored.
cleanup() { restore_railgate; restore_platform; }
trap cleanup EXIT INT TERM

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
