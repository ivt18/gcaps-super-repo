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
#   ./build_gcaps_r35.sh                  # patch, build, verify, stage, install
#   ./build_gcaps_r35.sh --no-install     # stop after staging
#   ./build_gcaps_r35.sh --repatch-from 50cbdb4   # tree is at an OLDER revision
#
# DO NOT run this under sudo.  It refuses to start as root, for two reasons:
# sudo rewrites HOME, so $KG/$KN would default to /root/kg and /root/kn; and
# patching and building as root leaves root-owned files scattered through your
# source tree, which breaks the next ordinary `make` there.  The script calls
# sudo itself for the two steps that need it (staging into /lib/modules and
# depmod), and validates the credential up front so a password prompt cannot
# interrupt the build.
#
# --repatch-from REV is the normal case on a board that has built GCAPS before:
# $KG is already patched, but at the PREVIOUS revision of these patch files, so
# the current ones neither apply forward nor reverse.  Given the git revision the
# tree was last patched at, each file is reverse-patched with THAT revision's
# patch and forward-patched with the current one, on a temp copy, moved into
# place only if both succeed.  Verified to produce a file byte-identical to
# patching a pristine tree with the current patch set.
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
# twice, must stay under 128 KB => P < ~50.  Overridable because the object
# count is not a constant of nature -- but raise it only with that arithmetic
# redone, not to silence the guard.
MAX_M_PATH_LEN="${MAX_M_PATH_LEN:-48}"

CHECK_ONLY=0
DO_INSTALL=1
REPATCH_FROM=""
while (( $# )); do
    case "$1" in
        --check)           CHECK_ONLY=1 ;;
        --no-install)      DO_INSTALL=0 ;;
        --variant)         [[ $# -ge 2 ]] || { echo "--variant needs a value" >&2; exit 1; }
                           VARIANT="$2"; shift ;;
        --variant=*)       VARIANT="${1#*=}" ;;
        --repatch-from)    [[ $# -ge 2 ]] || { echo "--repatch-from needs a revision" >&2; exit 1; }
                           REPATCH_FROM="$2"; shift ;;
        --repatch-from=*)  REPATCH_FROM="${1#*=}" ;;
        -j*)               JOBS="${1#-j}" ;;
        -h|--help)         sed -n '2,50p' "$0"; exit 0 ;;
        *)                 echo "unknown argument: $1" >&2; exit 1 ;;
    esac
    shift
done

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

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

if [[ $EUID -eq 0 ]]; then
    fail "do not run this as root.
      sudo rewrites HOME, so \$KG would default to /root/kg — and patching and
      building as root leaves root-owned files in your source tree that break
      your next ordinary build there.
      Run it as yourself: '$0'.  It calls sudo only to install."
fi
ok "running as $(id -un) (not root)"

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

# Ask for the sudo credential NOW rather than letting a password prompt appear
# minutes later, after the build, with the terminal scrolled past the question.
SUDO=""
if (( DO_INSTALL && ! CHECK_ONLY )); then
    command -v sudo >/dev/null || fail "sudo not installed (use --no-install)"
    sudo -v || fail "sudo credential refused — re-run with --no-install to build only"
    SUDO="sudo"
    ok "sudo available for the install step"
fi

for entry in "${PATCH_TARGETS[@]}"; do
    tgt="${entry%%:*}"; pf="${entry##*:}"
    [[ -f "$KG/$tgt" ]]      || fail "source file missing: $KG/$tgt"
    [[ -f "$SELF_DIR/$pf" ]] || fail "patch missing: $SELF_DIR/$pf"
done
ok "all 4 patch targets and patch files present"

[[ -f "$MODULE_PATH" ]] && ok "install target $MODULE_PATH" \
    || warn "install target not found: $MODULE_PATH"

# A pristine (stock NVIDIA) module to fall back to.  Do NOT manufacture one from
# whatever is currently installed -- on a board that has run GCAPS before that is
# a GCAPS build, and calling it "prebuilt" would destroy the only real reference.
PRISTINE_REF=""
for cand in "$MODULE_PATH.prebuilt-bak" "$VARIANT_DIR/nvgpu_original.ko" \
            "$VARIANT_DIR/nvgpu_prebuilt.ko"; do
    [[ -f "$cand" ]] && { PRISTINE_REF="$cand"; break; }
done
[[ -n "$PRISTINE_REF" ]] \
    && ok "pristine module available: $(basename "$PRISTINE_REF")" \
    || warn "no pristine module found beside $VARIANT_DIR — keep one, it is the
      only way back to the stock driver without re-flashing"

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
classify() {   # -> APPLIED | PRISTINE | UNKNOWN
    local tgt="$KG/$1" pf="$SELF_DIR/$2"
    if patch -R -p0 -f --dry-run "$tgt" < "$pf" >/dev/null 2>&1; then echo APPLIED
    elif patch -p0 -f --dry-run "$tgt" < "$pf" >/dev/null 2>&1;   then echo PRISTINE
    else echo UNKNOWN; fi
}

# UNKNOWN on a board that has built GCAPS before almost always means "patched at
# an older revision of these files", not "corrupted".  --repatch-from names the
# revision it was last patched at: reverse THAT patch out, apply the current one.
# All of it on a temp copy, moved into place only if both halves succeed, so a
# failure leaves the tree exactly as it was.
migrate_patch() {
    local rel="$1" pf="$2" tgt="$KG/$1"
    local oldpf="$WORK/old_$pf" tmp="$WORK/mig_$(basename "$rel")"

    git -C "$SELF_DIR" show "$REPATCH_FROM:gcaps_driver_patch/$pf" > "$oldpf" 2>/dev/null \
        || fail "cannot read $pf at revision '$REPATCH_FROM'.
      Is $SELF_DIR inside the git repo, and is that revision valid?"

    cp -p "$tgt" "$tmp"
    patch -s -R -p0 -f "$tmp" < "$oldpf" >/dev/null 2>&1 \
        || fail "$rel is not at revision '$REPATCH_FROM' either — cannot reverse it.
      Find the revision the tree WAS patched at, or restore $rel from a
      pristine source tree and re-run without --repatch-from."

    # $tmp is pristine at exactly this point -- the only moment a genuine
    # unpatched copy exists, so keep one for future runs to classify against.
    [[ -f "$tgt.gcaps-orig" ]] || cp -p "$tmp" "$tgt.gcaps-orig"

    patch -s -p0 -f "$tmp" < "$SELF_DIR/$pf" >/dev/null 2>&1 \
        || fail "reversed $rel back to pristine, but the CURRENT patch will not
      apply to it.  Tree left untouched."

    cp -p "$tgt" "$tgt.gcaps-prev"
    mv "$tmp" "$tgt"
    ok "repatched from $REPATCH_FROM: $rel  (previous kept as $(basename "$tgt").gcaps-prev)"
}

apply_patch() {
    local rel="$1" pf="$2" tgt="$KG/$1"

    case "$(classify "$rel" "$pf")" in
        APPLIED)  ok "already applied: $rel"; return ;;
        PRISTINE) ;;
        UNKNOWN)
            [[ -n "$REPATCH_FROM" ]] || fail "cannot apply $pf to $rel — the tree is neither pristine nor patched.
      If this board has built GCAPS before, the tree is patched at an OLDER
      revision of these files: re-run with --repatch-from <git-rev>, naming the
      revision it was last patched at.  Otherwise restore $rel from a pristine
      source tree."
            migrate_patch "$rel" "$pf"
            return ;;
    esac

    [[ -f "$tgt.gcaps-orig" ]] || cp -p "$tgt" "$tgt.gcaps-orig"
    patch -p0 -f "$tgt" < "$SELF_DIR/$pf" >/dev/null \
        || fail "patch passed its dry-run but failed for real: $rel"
    ok "patched: $rel  (original kept as $(basename "$tgt").gcaps-orig)"
}

if (( CHECK_ONLY )); then
    n_unknown=0
    for entry in "${PATCH_TARGETS[@]}"; do
        tgt="${entry%%:*}"; pf="${entry##*:}"
        case "$(classify "$tgt" "$pf")" in
            APPLIED)  ok   "would skip (already applied): $tgt" ;;
            PRISTINE) ok   "would apply: $tgt" ;;
            UNKNOWN)  warn "would FAIL: $tgt (neither pristine nor patched)"
                      (( n_unknown++ )) ;;
        esac
    done
    if (( n_unknown )); then
        echo
        echo "  $n_unknown file(s) are patched at some OTHER revision of these patches."
        echo "  That is the normal state on a board that has built GCAPS before."
        echo "  Re-run naming the revision the tree was last patched at, e.g."
        echo "      $0 --repatch-from <git-rev>"
        echo "  (git log --oneline -- gcaps_driver_patch/ lists the candidates)"
    fi
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
if [[ -n "$SUDO" ]] || [[ -w "$VARIANT_DIR" ]]; then
    $SUDO cp -p "$KO" "$STAGED" && ok "staged $STAGED"
else
    warn "cannot write $VARIANT_DIR — leaving the build at $KO"
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

# Snapshot whatever is installed right now, under a name that says exactly that.
# It is NOT called .prebuilt-bak: on a board that has run GCAPS before the
# installed module is a GCAPS build, and naming it "prebuilt" would overwrite the
# one real route back to the stock driver.  The pristine reference found during
# preflight (e.g. nvgpu_original.ko) is left strictly alone.
if [[ -f "$MODULE_PATH" ]]; then
    BAK="$MODULE_PATH.bak-$(date +%Y%m%d_%H%M%S)"
    $SUDO cp -p "$MODULE_PATH" "$BAK"
    ok "previous module saved as $(basename "$BAK")"
fi

$SUDO cp -p "$STAGED" "$MODULE_PATH" || fail "install failed"
ok "installed $MODULE_PATH  (sha256 $sha…)"
$SUDO depmod -a && ok "depmod -a"

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
