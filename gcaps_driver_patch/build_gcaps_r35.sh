#!/usr/bin/env bash
#
# build_gcaps_r35.sh — apply the GCAPS patches, build nvgpu.ko natively on the
#                      R35 Orin, verify it, and stage/install it.
#
# Run ON THE ORIN (JetPack 5.1.4 / L4T R35.6.4, kernel 5.10, aarch64) after a
# `git pull`.  Replaces the by-hand sequence in
# singleTaskSched/docs/gcaps_build_procedure.md, which describes the older
# cross-compile-from-a-laptop route: building natively is far faster and is
# vermagic-correct by construction.
#
#   ./build_gcaps_r35.sh --check          # preflight only, changes nothing
#   ./build_gcaps_r35.sh                  # patch + build + verify + stage
#   sudo ./build_gcaps_r35.sh             # ... and install over nvgpu.ko
#   ./build_gcaps_r35.sh --no-install     # patch + build + verify + stage only
#
# THREE BUILD TRAPS this script guards, each of which has cost a cycle before:
#
#   1. srctree.nvgpu defaults to the headers package's include-ONLY nvgpu/
#      copy, so it MUST be overridden to the real source tree.
#   2. srctree.nvidia is required or linux/nvmap_exports.h is not found.
#   3. `execvp: /bin/sh: Argument list too long` at link — 723 objects times
#      long absolute paths is ~82 KB, and Kbuild embeds the command TWICE (the
#      run and the .cmd file) against a 128 KB MAX_ARG_STRLEN.  Build from a
#      SHORT path.  The guard below refuses a path long enough to hit it.
#
# nvgpu is built -Werror, so any function added by a patch must be static or
# declared in a header.
#
# NOTE ON VERIFICATION: `srcversion` is absent from these modules (the stock one
# included), so it cannot be used to prove a source match.  What is checked
# instead is vermagic against the running kernel plus a modpost log with no
# undefined symbols — the same pair used when the current module was qualified.
#
# NEVER live-swaps the module: the board is SHARED, and rmmod'ing nvgpu out from
# under another user wedges the GPU.  Install writes the file and tells you to
# reboot.

set -uo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- knobs (env-overridable) ------------------------------------------------
KG="${KG:-$HOME/kg}"                    # nvgpu source tree  (kernel/nvgpu)
KN="${KN:-$HOME/kn}"                    # nvidia source tree (kernel/nvidia)
KREL="$(uname -r)"
KDIR="${KDIR:-/lib/modules/$KREL/build}"
MODULE_PATH="${MODULE_PATH:-/lib/modules/$KREL/kernel/drivers/gpu/nvgpu/nvgpu.ko}"
VARIANT_DIR="${VARIANT_DIR:-$(dirname "$MODULE_PATH")}"
VARIANT="${VARIANT:-gcaps}"
JOBS="${JOBS:-$(nproc)}"
LOG="${LOG:-/var/tmp/gcaps_build_$(date +%Y%m%d_%H%M%S).log}"

# M= path longer than this risks trap 3.  723 objects x (P+40) chars, embedded
# twice, must stay under 128 KB => P < ~50.
MAX_M_PATH_LEN=48

CHECK_ONLY=0
DO_INSTALL=1
for a in "$@"; do
    case "$a" in
        --check)      CHECK_ONLY=1 ;;
        --no-install) DO_INSTALL=0 ;;
        --variant=*)  VARIANT="${a#*=}" ;;
        -j*)          JOBS="${a#-j}" ;;
        -h|--help)    sed -n '2,40p' "$0"; exit 0 ;;
        *)            echo "unknown argument: $a" >&2; exit 1 ;;
    esac
done

fail() { echo "FAIL: $*" >&2; exit 1; }
ok()   { echo "  ok    $*"; }
warn() { echo "  WARN  $*" >&2; }
step() { echo; echo "== $* =="; }

# The four files the GCAPS patch set touches, paired with their patches.  All
# paths are relative to the nvgpu tree root ($KG).  The UAPI header is not
# optional: ioctl_ctrl.c needs the args struct and the ioctl number from it.
PATCH_TARGETS=(
    "drivers/gpu/nvgpu/os/linux/ioctl_ctrl.c:ioctl_ctrl.c.patch"
    "drivers/gpu/nvgpu/os/linux/sched.c:sched.c.patch"
    "drivers/gpu/nvgpu/include/nvgpu/sched.h:sched.h.patch"
    "include/uapi/linux/nvgpu-ctrl.h:nvgpu-ctrl.h.patch"
)

# =============================================================================
step "preflight"
# =============================================================================

[[ "$(uname -m)" == "aarch64" ]] \
    || fail "not aarch64 — this script builds natively ON the Orin.
      For the cross-compile route see docs/gcaps_build_procedure.md."

ok "kernel $KREL"

[[ -d "$KDIR" ]] || fail "kernel headers not found: $KDIR
      Install nvidia-l4t-kernel-headers matching $KREL."
ok "headers $KDIR"

[[ -d "$KG/drivers/gpu/nvgpu" ]] \
    || fail "nvgpu sources not at \$KG=$KG (expected $KG/drivers/gpu/nvgpu).
      Extract kernel_src.tbz2 from the L4T r35_release_v6.4 public sources,
      or set KG=<path>."
ok "nvgpu sources $KG"

[[ -d "$KN" ]] \
    || fail "nvidia sources not at \$KN=$KN.
      Required for linux/nvmap_exports.h (trap 2); set KN=<path>."
ok "nvidia sources $KN"

M_PATH="$KG/drivers/gpu"
if (( ${#M_PATH} > MAX_M_PATH_LEN )); then
    fail "build path is ${#M_PATH} chars, over the ${MAX_M_PATH_LEN}-char limit (trap 3):
        $M_PATH
      Kbuild embeds the full object list twice and will die with
      'execvp: /bin/sh: Argument list too long' at link.  Move the tree
      somewhere short (~/kg) — a symlink does not help, Kbuild resolves it."
fi
ok "build path ${#M_PATH} chars (limit $MAX_M_PATH_LEN)"

command -v patch >/dev/null || fail "patch(1) not installed"

for entry in "${PATCH_TARGETS[@]}"; do
    tgt="${entry%%:*}"; pf="${entry##*:}"
    [[ -f "$KG/$tgt" ]]      || fail "source file missing: $KG/$tgt"
    [[ -f "$SELF_DIR/$pf" ]] || fail "patch missing: $SELF_DIR/$pf"
done
ok "all 4 patch targets and patch files present"

[[ -f "$MODULE_PATH" ]] && ok "install target $MODULE_PATH" \
    || warn "install target not found: $MODULE_PATH"
[[ -f "$MODULE_PATH.prebuilt-bak" ]] \
    && ok "pristine backup present (.prebuilt-bak)" \
    || warn "no .prebuilt-bak beside the module — one will be made on first install"

echo "  staged variants in $VARIANT_DIR:"
ls -1 "$VARIANT_DIR"/nvgpu_*.ko 2>/dev/null | sed 's/^/      /' || echo "      (none)"

# =============================================================================
step "patches"
# =============================================================================

# Idempotent by construction: a reverse dry-run that succeeds means the patch is
# already in, and a forward dry-run that fails on a tree that is ALSO not
# already patched means the tree is in an unknown state — refuse rather than
# half-apply.  Default fuzz is kept deliberately: these patches carry
# "\ No newline at end of file" markers mid-file and legitimately apply at
# fuzz 1-2.
apply_patch() {
    local tgt="$KG/$1" pf="$SELF_DIR/$2"

    if patch -R -p0 -f --dry-run "$tgt" < "$pf" >/dev/null 2>&1; then
        ok "already applied: $1"
        return
    fi
    if ! patch -p0 -f --dry-run "$tgt" < "$pf" >/dev/null 2>&1; then
        fail "cannot apply $2 to $1 — the tree is neither pristine nor patched.
      Restore $tgt from ${tgt}.gcaps-orig, or re-extract the sources."
    fi
    [[ -f "$tgt.gcaps-orig" ]] || cp -p "$tgt" "$tgt.gcaps-orig"
    patch -p0 -f "$tgt" < "$pf" >/dev/null \
        || fail "patch reported success in dry-run but failed for real: $1"
    ok "patched: $1  (original kept as $(basename "$tgt").gcaps-orig)"
}

if (( CHECK_ONLY )); then
    for entry in "${PATCH_TARGETS[@]}"; do
        tgt="${entry%%:*}"; pf="${entry##*:}"
        if patch -R -p0 -f --dry-run "$KG/$tgt" < "$SELF_DIR/$pf" >/dev/null 2>&1; then
            ok "would skip (already applied): $tgt"
        elif patch -p0 -f --dry-run "$KG/$tgt" < "$SELF_DIR/$pf" >/dev/null 2>&1; then
            ok "would apply: $tgt"
        else
            warn "would FAIL: $tgt (tree neither pristine nor patched)"
        fi
    done
    echo; echo "== --check: nothing was modified =="
    exit 0
fi

for entry in "${PATCH_TARGETS[@]}"; do
    apply_patch "${entry%%:*}" "${entry##*:}"
done

# =============================================================================
step "build  (log: $LOG)"
# =============================================================================

echo "  make -C $KDIR M=$M_PATH srctree.nvgpu=$KG srctree.nvidia=$KN modules -j$JOBS"
make -C "$KDIR" M="$M_PATH" \
     srctree.nvgpu="$KG" srctree.nvidia="$KN" \
     modules -j"$JOBS" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
(( rc == 0 )) || fail "build failed (rc=$rc) — see $LOG
      nvgpu is -Werror: a function added by a patch must be static or declared."

KO="$KG/drivers/gpu/nvgpu/nvgpu.ko"
[[ -f "$KO" ]] || fail "build reported success but $KO does not exist"
ok "built $KO"

# =============================================================================
step "verify"
# =============================================================================

# modpost emits undefined symbols as a WARNING, not an error, for out-of-tree
# modules — so a clean rc is not enough.  A module with an unresolved symbol
# loads only to fail at insmod, long after this script has said "fine".
if grep -nE "undefined!|Symbol .* undefined|no symbol version for" "$LOG"; then
    fail "modpost reported unresolved symbols (above) — do not install this module"
fi
ok "0 unresolved symbols"

vm="$(modinfo -F vermagic "$KO" 2>/dev/null)"
[[ -n "$vm" ]] || fail "modinfo could not read a vermagic from $KO"
if [[ "$vm" != "$KREL"* ]]; then
    fail "vermagic mismatch — module '$vm' vs running kernel '$KREL'.
      insmod would refuse it.  Wrong \$KDIR, or the headers do not match
      the running kernel."
fi
ok "vermagic $vm"

sha="$(sha256sum "$KO" | cut -c1-8)"
ok "sha256 $sha…"

# =============================================================================
step "stage"
# =============================================================================

STAGED="$VARIANT_DIR/nvgpu_$VARIANT.ko"
if [[ -w "$VARIANT_DIR" ]]; then
    cp -p "$KO" "$STAGED" && ok "staged $STAGED"
else
    warn "cannot write $VARIANT_DIR (need root) — leaving the build at $KO"
    STAGED="$KO"
fi

# =============================================================================
step "install"
# =============================================================================

if (( ! DO_INSTALL )); then
    echo "  --no-install: stopping here."
    echo "  To install:  sudo cp $STAGED $MODULE_PATH && sudo depmod -a && sudo reboot"
    exit 0
fi

if [[ $EUID -ne 0 ]]; then
    echo "  not root — skipping install."
    echo "  To install:  sudo cp $STAGED $MODULE_PATH && sudo depmod -a && sudo reboot"
    exit 0
fi

# Only ever create .prebuilt-bak from a module we have NOT installed, or the
# pristine NVIDIA build is lost the second time this runs.
if [[ -f "$MODULE_PATH" && ! -f "$MODULE_PATH.prebuilt-bak" ]]; then
    cp -p "$MODULE_PATH" "$MODULE_PATH.prebuilt-bak"
    ok "saved pristine module as $(basename "$MODULE_PATH").prebuilt-bak"
fi

cp -p "$STAGED" "$MODULE_PATH" || fail "install failed"
ok "installed $MODULE_PATH  (sha256 $sha…)"
depmod -a && ok "depmod -a"

cat <<EOF

== REBOOT REQUIRED ==
  The board is SHARED — do NOT rmmod/insmod nvgpu live; that wedges the GPU for
  anyone else using it.  Reboot to pick the new module up:

      sudo reboot

  After the reboot, and before any GCAPS run:

      # railgating: this patch set has NO gk20a_busy() power reference, so a
      # railgated GPU means runlist writes into an unmapped BAR -> ENODEV ->
      # watchdog board reset.  run_gcaps_r35.sh does this for ./main, but not
      # for binaries launched directly.
      echo 0 | sudo tee /sys/devices/platform/bus@0/17000000.gpu/railgate_enable

      # the event ring should now exist:
      ls -l /proc/gcaps_events
EOF
