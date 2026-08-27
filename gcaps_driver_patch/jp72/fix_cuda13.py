#!/usr/bin/env python3
"""CUDA 13 compatibility fixes for the GCAPS userspace.

Two CUDA 12 -> 13 breakages, neither related to the GCAPS driver port:

  1. cudaDeviceProp lost ::computeMode and ::clockRate.  The cudaComputeMode
     enum and the cudaDevAttrClockRate attribute both still exist, so the
     removed members are replaced by macros that query/degrade correctly.
  2. cuCtxCreate now resolves to cuCtxCreate_v4, which takes an extra
     CUctxCreateParams* second argument.  Every call site goes through a
     cuCtxCreateCompat() macro instead.

Idempotent: restores from .cuda12-orig first, so re-running is safe.
"""
import glob
import os
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
MARK = "GCAPS_CUDA13_COMPAT"

HELPER_COMPAT = """
/* %s: CUDA 13 removed cudaDeviceProp::computeMode and ::clockRate. */
#if defined(CUDART_VERSION) && CUDART_VERSION >= 13000
static inline int helper_clock_rate_khz(int dev)
{
  int v = 0;
  cudaDeviceGetAttribute(&v, cudaDevAttrClockRate, dev);
  return v;
}
#define HELPER_COMPUTE_MODE(p)      (cudaComputeModeDefault)
#define HELPER_CLOCK_RATE(p, dev)   helper_clock_rate_khz(dev)
#else
#define HELPER_COMPUTE_MODE(p)      ((p).computeMode)
#define HELPER_CLOCK_RATE(p, dev)   ((p).clockRate)
#endif
""" % MARK

CTX_COMPAT = """
/* %s: on CUDA 13 cuCtxCreate resolves to cuCtxCreate_v4, which takes a
 * CUctxCreateParams* as its second argument. */
#if defined(CUDA_VERSION) && CUDA_VERSION >= 13000
#define cuCtxCreateCompat(pctx, flags, dev) cuCtxCreate((pctx), NULL, (flags), (dev))
#else
#define cuCtxCreateCompat(pctx, flags, dev) cuCtxCreate((pctx), (flags), (dev))
#endif
""" % MARK


def restore(path):
    if os.path.isfile(path + ".cuda12-orig"):
        open(path, "w").write(open(path + ".cuda12-orig").read())


def insert_after_includes(text, block, limit=120):
    last, count = 0, 0
    for line in text.splitlines(keepends=True):
        count += len(line)
        if line.startswith("#include"):
            last = count
        if line.count("\n") and text[:count].count("\n") > limit:
            break
    return (text[:last] + block + text[last:]) if last else None


def patch_helper(path):
    restore(path)
    t = open(path).read()
    n_cm, n_cr = t.count("deviceProp.computeMode"), t.count("deviceProp.clockRate")
    if not (n_cm or n_cr):
        return "nothing to do"
    orig = t
    t = t.replace("deviceProp.computeMode", "HELPER_COMPUTE_MODE(deviceProp)")
    t = t.replace("deviceProp.clockRate", "HELPER_CLOCK_RATE(deviceProp, current_device)")
    out = insert_after_includes(t, HELPER_COMPAT)
    if out is None:
        return "FAIL: no #include anchor"
    open(path + ".cuda12-orig", "w").write(orig)
    open(path, "w").write(out)
    return "patched (%d computeMode, %d clockRate)" % (n_cm, n_cr)


def patch_ctx(path):
    restore(path)
    t = open(path).read()
    n = t.count("cuCtxCreate(")
    if not n:
        return "nothing to do"
    orig = t
    # Replace call sites BEFORE injecting the macro, so the macro body
    # (which legitimately names cuCtxCreate) is not itself rewritten.
    t = t.replace("cuCtxCreate(", "cuCtxCreateCompat(")
    out = insert_after_includes(t, CTX_COMPAT)
    if out is None:
        return "FAIL: no #include anchor"
    open(path + ".cuda12-orig", "w").write(orig)
    open(path, "w").write(out)
    return "patched (%d call site%s)" % (n, "" if n == 1 else "s")


rc = 0
h = os.path.join(ROOT, "common/cuda_inc/helper_cuda.h")
print("  %-38s %s" % ("common/cuda_inc/helper_cuda.h", patch_helper(h)))

for p in sorted(glob.glob(os.path.join(ROOT, "app/*/*.cu"))):
    if "cuCtxCreate" not in open(p).read() and not os.path.isfile(p + ".cuda12-orig"):
        continue
    r = patch_ctx(p)
    print("  %-38s %s" % (os.path.relpath(p, ROOT), r))
    if r.startswith("FAIL"):
        rc = 1
sys.exit(rc)
