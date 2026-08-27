#!/bin/bash
# Stage a freshly built GCAPS nvgpu.ko for the next boot, and prove it landed.
#
# Exists because "switch-nvgpu gcaps" only copies nvgpu_gcaps.ko -> nvgpu.ko;
# it does NOT refresh nvgpu_gcaps.ko from your build tree.  Staging a stale
# nvgpu_gcaps.ko silently boots a months-old driver while `switch-nvgpu status`
# cheerfully reports "GCAPS".  That already cost one debug cycle.
#
#   sudo ./stage-gcaps.sh [path/to/nvgpu.ko]
#
# Does NOT reboot, and does NOT rmmod: on a shared board the display stack
# holds the driver, so the swap must happen at the next boot.

set -eu

SRC=${1:-/mnt/nvme/ONGPUSCHEDULER/gcaps/nvgpu/drivers/gpu/nvgpu/nvgpu.ko}
KVER=$(uname -r)
M=/lib/modules/$KVER/updates

[ "$(id -u)" -eq 0 ] || { echo "ERROR: run with sudo" >&2; exit 1; }
[ -f "$SRC" ] || { echo "ERROR: no module at $SRC" >&2; exit 1; }

# Refuse to proceed without a stock backup - otherwise there is no way back.
if [ ! -f "$M/nvgpu_original.ko" ]; then
	echo "ERROR: $M/nvgpu_original.ko is missing." >&2
	echo "       Save the stock driver first, or you cannot revert:" >&2
	echo "         sudo cp $M/nvgpu.ko $M/nvgpu_original.ko" >&2
	exit 1
fi

# vermagic must match the running kernel or the module will not load.
want=$(modinfo -F vermagic nvgpu)
got=$(modinfo -F vermagic "$SRC")
if [ "$want" != "$got" ]; then
	echo "ERROR: vermagic mismatch - refusing to stage." >&2
	echo "  running: $want" >&2
	echo "  built  : $got" >&2
	exit 1
fi

tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT
cp "$SRC" "$tmp"
strip --strip-debug "$tmp"

install -m644 "$tmp" "$M/nvgpu_gcaps.ko"
cp "$M/nvgpu_gcaps.ko" "$M/nvgpu.ko"
depmod -a

sum() { md5sum "$1" | cut -c1-12; }
printf '\n'
printf '  source        %s  %s bytes\n' "$(sum "$SRC")"            "$(stat -c%s "$SRC")"
printf '  staged gcaps  %s  %s bytes\n' "$(sum "$M/nvgpu_gcaps.ko")" "$(stat -c%s "$M/nvgpu_gcaps.ko")"
printf '  active .ko    %s\n'           "$(sum "$M/nvgpu.ko")"
printf '  original      %s\n'           "$(sum "$M/nvgpu_original.ko")"
printf '\n'

if cmp -s "$M/nvgpu.ko" "$M/nvgpu_gcaps.ko"; then
	echo "OK: nvgpu.ko == nvgpu_gcaps.ko (GCAPS will load at next boot)"
else
	echo "MISMATCH: nvgpu.ko != nvgpu_gcaps.ko" >&2
	exit 1
fi

echo
echo "Reboot to load it.  Then confirm the RUNNING driver is this build:"
echo "  md5sum \$(modinfo -F filename nvgpu)   # expect $(sum "$M/nvgpu_gcaps.ko")"
