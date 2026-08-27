#!/bin/bash
# Preflight + run wrapper for the GCAPS userspace on JetPack 7.2 / L4T r39.2.
#
# WHY THIS EXISTS
#
# The GCAPS driver patch calls g->ops.runlist.update() from its ioctl handler
# WITHOUT taking a gk20a_busy() power reference -- unlike every other
# register-touching handler in ioctl_ctrl.c.  With GPU railgating enabled the
# GPU powers down between segments, and the next ioctl writes runlist registers
# into an unmapped BAR:
#
#   nvgpu: Attempted access to GPU regs after unmapping! r=0x00c00080
#     nvgpu_warn_on_no_regs <- nvgpu_writel <- ga10b_runlist_hw_submit
#     <- runlist_submit_powered <- nvgpu_runlist_do_update
#     <- nvgpu_ioctl_runlist_update_rt_prio
#
# Driver state is corrupted, later ioctls return ENODEV, every GPU task blocks,
# and the Tegra watchdog resets the board ~2 min later.  Diagnosed 2026-08-27
# after it took the board down four times.  Full write-up:
#   msc-thesis-on-gpu-sched/singleTaskSched/docs/gcaps_build_procedure_jp72.md
#
# COUNTER-INTUITIVE: quiescing the GPU makes this MORE likely, not less -- an
# idle GPU railgates more often.
#
# Usage:
#   sudo ./run_gcaps.sh -f taskset.csv -d 10 -i 1
#   sudo ./run_gcaps.sh --check              # preflight only, run nothing

set -uo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN="$SELF_DIR/main"
RAILGATE=/sys/devices/platform/bus@0/17000000.gpu/railgate_enable
GCAPS_SYM=nvgpu_ioctl_runlist_update_rt_prio
GCAPS_IOCTL_NR=49

CHECK_ONLY=0
[[ "${1:-}" == "--check" ]] && { CHECK_ONLY=1; shift; }

fail() { echo "PREFLIGHT FAIL: $*" >&2; exit 1; }
ok()   { echo "  ok    $*"; }
warn() { echo "  WARN  $*" >&2; }

[[ $EUID -eq 0 ]] || fail "must run as root (SCHED_FIFO and railgate control need it)"

echo "== GCAPS preflight =="

# 1. Binary present.
[[ -x "$MAIN" ]] || fail "no executable at $MAIN -- build it first (see Step 7)"
ok "binary $MAIN"

# 2. Is the caller asking for GCAPS mode?  Only then does the driver matter.
gcaps_mode=0
prev=""
for a in "$@"; do
    [[ "$prev" == "-i" && "$a" == "1" ]] && gcaps_mode=1
    prev="$a"
done

# 3. GCAPS driver actually loaded.  /proc/kallsyms shows static symbols too,
#    and kptr_restrict hides addresses, not names.
if [[ $gcaps_mode -eq 1 ]]; then
    if grep -qw "$GCAPS_SYM" /proc/kallsyms 2>/dev/null; then
        ok "GCAPS driver loaded ($GCAPS_SYM present)"
    else
        fail "'-i 1' requested but the STOCK nvgpu is loaded.
       Stage the GCAPS build and reboot:  sudo stage-gcaps.sh && sudo reboot"
    fi
else
    ok "TSG baseline mode (-i 0): driver variant irrelevant"
fi

# 4. The ioctl-number trap: a binary built against the stale R35 uapi header
#    calls ioctl 43, which on r39.2 is GET_GPC_LOCAL_TO_PHYSICAL_MAP -- it
#    returns 0 and does NOTHING, so GCAPS is silently inert.
if command -v objdump >/dev/null 2>&1; then
    # ARM64 splits the constant: 0x4731 = nr 49, 0x472b = nr 43.
    n49=$(objdump -d "$MAIN" 2>/dev/null | grep -ci '#0x4731')
    n43=$(objdump -d "$MAIN" 2>/dev/null | grep -ci '#0x472b')
    if [[ "$n43" -gt 0 ]]; then
        fail "binary encodes ioctl 43 (stale R35 uapi header).
       On r39.2 that is GET_GPC_LOCAL_TO_PHYSICAL_MAP: it returns 0 and does
       nothing, so GCAPS would be silently INERT.  Rebuild with
       NVGPU_UAPI=<sources>/nvgpu/include/uapi"
    elif [[ "$n49" -gt 0 ]]; then
        ok "binary encodes ioctl $GCAPS_IOCTL_NR"
    else
        warn "could not find either ioctl constant in $MAIN -- check manually"
    fi
else
    warn "objdump not available -- skipping the ioctl-number check"
fi

# 5. THE IMPORTANT ONE.  Railgating must be off.
railgate_changed=0
if [[ -w "$RAILGATE" ]]; then
    if [[ "$(cat "$RAILGATE")" == "0" ]]; then
        ok "railgating already disabled"
    else
        echo 0 > "$RAILGATE" || fail "could not disable railgating at $RAILGATE"
        railgate_changed=1
        ok "railgating DISABLED (was 1) -- will restore on exit"
    fi
elif [[ -e "$RAILGATE" ]]; then
    fail "$RAILGATE exists but is not writable"
else
    fail "railgate control not found at $RAILGATE.
       Locate it (find /sys/devices -name 'railgate_enable') and update this
       script -- do NOT run GCAPS without it."
fi

restore_railgate() {
    if [[ $railgate_changed -eq 1 ]]; then
        echo 1 > "$RAILGATE" 2>/dev/null && echo "  restored railgate_enable=1"
    fi
}
trap restore_railgate EXIT

echo "== preflight passed =="

if [[ $CHECK_ONLY -eq 1 ]]; then
    echo "(--check: not running)"
    exit 0
fi

[[ $# -gt 0 ]] || fail "no arguments for ./main (e.g. -f taskset.csv -d 10 -i 1)"

echo "== running: $MAIN $* =="
cd "$SELF_DIR" || exit 1
mkdir -p timelog          # absent => "Error opening file for writing!"
"$MAIN" "$@"
rc=$?
echo "== main exited rc=$rc =="

# A hang here is the runlist-cache-desync deadlock, NOT the railgate crash:
# board stays up, Ctrl-C is enough.  See gcaps-super-repo/runlist-cache-desync-bug.md
exit $rc
