# GCAPS preemption overhead — measurement results

How much does a GCAPS preemption cost? Two quantities, in a scenario where tasks
are registered at different SCHED_FIFO priorities and a higher-priority job's GPU
segment arrives while a lower-priority one is running:

1. **Scheduling + context-switch / preemption overhead** — the cost of actually
   performing the preemption (runlist reload + GPU-side switch).
2. **Execution-time extension of the preempted job** — by how much the victim's
   *active* GPU execution grows. The time it sits **suspended** (blocked while a
   higher-priority job runs) is **excluded** — only actively-executing time is
   counted, compared against the job's clean uncontended execution time.

## Headline result

| Quantity | Result |
|---|---|
| Per-preemption scheduling + context-switch overhead | **≈ 0.6 ms** (619 µs median, 631 µs p95, **662 µs worst**; n=555) |
| Active-execution extension per preemption (victim) | **≈ 0** (median −0.05 ms, p95 +0.19 ms; n=88) |

**GCAPS preemption does not measurably extend a job's active execution time.**
The cost of a preemption is the ~0.6 ms scheduling/context-switch overhead plus
the suspension (blocking) time — and suspension is response-time, not
execution-time, so it is excluded by construction.

Both numbers are robust across configurations: compute-intensive victims of
~27 ms and ~53 ms (matmul 2048 / 2560) and a **memory-intensive 2 GiB
streaming victim** ([`-M` run](#memory-intensive-run--m)) give ε medians of
601–623 µs and victim extensions of −0.05…−0.03 ms/preemption. In particular,
**memory footprint alone does not create a preemption penalty** — a preempting
task does not measurably slow a streaming victim's own execution.

## ε scales with the number of resident tasks (2026-07-06)

ε is **not a platform constant** — it roughly doubles from the 3-task
microbench to the 8-task taskset, same board, same protocol, `-d 10`:

| run | preempting-add ε median | p95 | max |
|---|---|---|---|
| microbench (3 tasks) | 620 µs | 638 µs | 664 µs |
| taskset (8 tasks) | 1218 µs | 1292 µs | 1300 µs |

This is consistent with the runlist rebuild's per-channel work scaling with
the number of channels bound on the device (the rebuild loops over channels —
see [runlist-cache-desync-bug.md](runlist-cache-desync-bug.md)). Two
consequences: (1) a worst-case ε for schedulability analysis must be measured
at the target taskset scale (here 1300 µs, not the microbench's 664 µs);
(2) every additional GPU-using process on the system — including unmanaged
ones, whose channels still reside on the device — is expected to inflate ε
further.

## Setup

- **Hardware / OS:** Jetson AGX Orin **32 GB** (`ga10b`, 1792 CUDA cores; 29 GiB
  visible to Linux), L4T R35.6.0 / JetPack 5.1.4, GCAPS-patched `nvgpu`.
  NOTE: this is a different board from the SequenceScheduler experiments
  (AGX Orin 64 GB, JetPack 6 / CUDA 12.6) — absolute times are not comparable
  across the two; compare each scheduler against its own same-board baseline.
  MAXN power, locked clocks, RT throttling off:
  ```bash
  sudo nvpmodel -m 0 && sudo jetson_clocks
  sudo sysctl -w kernel.sched_rt_runtime_us=-1
  ```
- **Driver instrumentation** ([`gcaps_driver_patch/ioctl_ctrl.c.patch`](gcaps_driver_patch/ioctl_ctrl.c.patch)):
  `nvgpu_ioctl_runlist_update_rt_prio` emits one structured record per
  runlist-update ioctl:
  ```
  GCAPS_EV ts=<ns> cpid=<pid> prio=<n> add=<0|1> rlupd=<0|1> elapsed_us=<eps> preempted=<pid|-1> resumed=<pid|-1>
  ```
  `elapsed_us` is the ioctl critical-section time (GCAPS ε). `preempted`/`resumed`
  (matched by pid) reconstruct each job's **suspended intervals**.
  `measure_preempt_overhead.py` captures these by **streaming `dmesg --follow`
  during the run** — a post-run dump truncates to the ring buffer, see
  *Methodology notes*.
- **Controlled microbenchmark** ([`gcaps_userspace/preemptOverheadGcaps.cc`](gcaps_userspace/preemptOverheadGcaps.cc)):
  three tasks at three distinct SCHED_FIFO priorities.  Primary
  (compute-intensive) configuration, `-N 2560 -P 1024`:

  | role | workload | prio | period |
  |---|---|---|---|
  | victim | matmul 2560 (~53.4 ms) | 2 | 80 ms |
  | preemptorB | histogram 8M (~0.28 ms) | 4 | 29 ms |
  | preemptorA | matmul 1024 (~3.3 ms) | 5 | 17 ms |

  The victim (lowest priority, long segment) is preempted by both; preemptorB is
  itself preempted by preemptorA, so preemptions occur at more than one level.
  GPU utilization ≈ 0.87 (every segment < its period — see *Tuning matters*).
  Run: `sudo ./preemptOverheadGcaps -i 1 -s 1 -d 10 -N 2560 -P 1024`.

  The `-M` flag swaps all three workloads for **large-footprint histograms**
  (victim 512Mi elements = 2 GiB, preA 64Mi, preB 128Mi; ≈ 2.75 GiB resident,
  U ≈ 0.82) with the same roles/periods/priorities — the
  [memory-intensive variant](#memory-intensive-run--m).

The realistic mixed taskset
([`workloadTasksetGcaps.cc`](gcaps_userspace/workloadTasksetGcaps.cc), `--run taskset`)
is also instrumented and can be measured the same way.

## How to reproduce

```bash
# 1. isolated baselines — covers every taskset/microbench workload
sudo ./workloadSweepGcaps -i 1 -s 1 -n 100 -w 20
awk -F, '$1=="matmul_2560"{print $5}' results/workloadBench/sweep_gcaps.csv \
  | sort -n | awk '{a[NR]=$1} END{print "median="a[int((NR+1)/2)]}'

# 2. compute-intensive microbench + anchored analysis
sudo python3 scripts/measure_preempt_overhead.py --run microbench -d 10 \
     --extra "-N 2560 -P 1024"
python3 scripts/measure_preempt_overhead.py \
    --events results/workloadBench/preempt_gcaps_events.log \
    --trace  results/workloadBench/preempt_gcaps_trace.csv \
    --baseline-ms mm_victim=53.375

# 3. memory-intensive variant (all-histogram, ~2.75 GiB resident)
sudo python3 scripts/measure_preempt_overhead.py --run microbench -d 10 \
     --extra "-M -N 512 -P 64"
```

With the streaming event capture the in-run non-preempted floor is clean and
matches the isolated baseline (53.370 vs 53.375 ms), so the `--baseline-ms`
anchor is a cross-check rather than a necessity; unanchored analysis of a
complete-capture run gives the same result.

## Detailed results (compute-intensive, 10 s, complete capture)

### (1) Scheduling + context-switch / preemption overhead

`elapsed_us` of each ioctl, split by what it did (555 real preemptions over a
10 s run; 1,558 events = every add/remove ioctl of the run):

| event | n | min | median | p95 | max |
|---|---:|---:|---:|---:|---:|
| preempting add (a real preemption) | 555 | 588 | 619 | 631 | **662 µs** |
| resuming remove (victim put back) | 621 | 603 | 645 | 664 | 679 µs |
| real runlist reload (`rlupd=1`) | 1484 | 261 | 624 | 658 | 718 µs |
| bookkeeping only (`rlupd=0`) | 74 | 13 | 27 | 33 | 36 µs |

The sharp bimodal split — real reloads ~0.6 ms vs no-op calls ~20 µs — confirms
the instrumentation isolates the runlist-update work.  Across all runs
(compute and memory variants, 10–30 s) the preempting-add median stays within
601–623 µs; the ~3 % upward drift at the highest preemption rates is real but
small.

### (2) Execution-time extension when preempted

Victim (`mm_victim`, matmul 2560), baseline = isolated **53.375 ms**
(= in-run clean floor 53.370, band 53.355–53.401 over 36 non-preempted
releases). 88 of 124 releases preempted, ~6 preemptions each:

| metric | median | mean | p95 |
|---|---:|---:|---:|
| gpu_wall when preempted | 72.31 ms | 73.04 | 76.74 |
| suspended (blocking, excluded) | 19.16 ms | 20.02 | 23.27 |
| **extension (active)** | **−0.23 ms** | −0.36 | +1.10 |
| **extension per preemption (active)** | **−0.05 ms** | −0.06 | +0.19 |

`active = gpu_wall − suspended ≈ 53.1 ms ≈ baseline 53.375` → extension ≈ 0.
The regime slope is poorly conditioned here (0.60 — suspensions cluster in a
narrow band), but the accounting identity resolves it: raw extension median
18.93 ms ≈ suspended median 19.16 ms, i.e. the wall-time inflation is entirely
the suspension itself.

## Memory-intensive run (`-M`)

Same process with every task a large-footprint streaming histogram
(victim 512Mi elements = **2 GiB**, preA 64Mi, preB 128Mi; U ≈ 0.82), probing
whether preemption costs grow under memory pressure (cache/TLB thrash from the
preempting task). 10 s, complete capture, 472 real preemptions:

| quantity | compute run (matmul 2560) | memory run (hist 512M) |
|---|---:|---:|
| preempting add ε (median / p95 / worst) | 619 / 631 / 662 µs (n=555) | 622 / 666 / 679 µs (n=472) |
| victim clean floor (in-run, non-preempted) | 53.355–53.401 ms | 29.302–29.390 ms |
| victim regime slope d(wall)/d(susp) | 0.60 (identity: raw ≈ suspended) | 0.88, r=0.93 |
| victim extension, active, per release (median / p95) | −0.23 / +1.10 ms | −0.18 / +0.88 ms |
| victim extension, active, per preemption (median / p95) | **−0.05 / +0.19 ms** | **−0.04 / +0.20 ms** |

**ε and the victim extension are independent of memory footprint.** A task
touching 2 GiB per segment is preempted as cheaply as a compute-bound matmul —
for a *streaming* victim there is no measurable cache-thrash penalty on its own
execution. (A streaming workload re-reads nothing, so it has almost no cached
working set to lose; an L2-resident *reuse-heavy* victim would be the sharper
probe of the thrash hypothesis and is future work.)

Note the `-M` scenario is far more preemption-dense (preB overlaps a preA
arrival almost every period: 129 of 241 preB releases preempted vs 13 in the
compute run), which raises the desync-deadlock hit rate — see *Caveats*.

## Per-preemption suspended-time under-attribution (≈ ε)

The singly-preempted middle task (`hist_preB`) shows a small, tightly
clustered, **positive** active extension that reproduces across both variants:

| preB variant | segment | preemptions | active extension / preemption (median / p95) |
|---|---:|---:|---:|
| compute run, hist 8M (32 MB) | 0.284 ms | 13 | +0.53 / +0.58 ms |
| memory run, hist 128M (512 MB) | 7.34 ms | 129 | +0.575 / +0.65 ms |

The same ~0.55 ms constant on segments differing 26× in length and 16× in
footprint rules out a cache effect (and +187 % of a 0.28 ms streaming kernel is
not physically a cache penalty). Its magnitude ≈ ε (~0.6 ms) points at the
attribution boundary: the driver's `preempted→resumed` interval appears to
miss ≈ one ioctl-length of the real kernel pause per preemption, so ~ε of
suspension is misattributed to active time. The victim does not show it
because its ~6 chained/overlapping suspensions per release merge and the edges
are absorbed. Conservative reading: **per-preemption suspended time is
under-reported by ≈ ε for singly-preempted tasks; the victim's active
extension remains ≈ 0 in all configurations** (if anything, the same bias
means the true extension is slightly more negative, i.e. still ≈ 0).
Filtering preB releases to exactly-one attributed suspend interval would pin
this down further.

## Earlier truncated-capture runs (historical)

The original 30 s runs (victim matmul 2048 and 2560) predate the streaming
event capture: the post-run `dmesg` dump kept only the kernel ring buffer
(~128 KiB ≈ 861 GCAPS_EV lines ≈ the **last ~4.5 s** of a busy run). Only
releases inside that window got suspension attribution; every earlier
preempted release was misclassified as "non-preempted" with its wall time
still including the unattributed suspension. That artifact produced the
"bimodal floor" those runs showed (e.g. 53.4 ms clean + ~19 ms suspension
≈ the 72 ms "contended bulk") — it was **mislabelled preemption, not co-run
contention**. Their headline conclusions (ε ≈ 0.6 ms: 601/604 µs medians;
anchored victim extension ≈ 0: −0.03/−0.035 ms per preemption) were
nevertheless correct, because the isolated `--baseline-ms` anchor bypassed the
poisoned floor and ε needs no attribution. The complete-capture 10 s runs
above supersede them.

## Methodology notes

- **`active = gpu_wall − suspended`.** The cudaEvent window (`gpu_wall`) on AGX Orin
  *includes* the suspended time (verified: `gpu_wall ≈ active + suspended`, e.g.
  72.3 − 19.2 ≈ 53.1 ≈ the 53.375 baseline), so subtracting the driver-reported
  suspension recovers the active execution. The analysis prints a
  `d(gpu_wall)/d(suspended)` slope per task to confirm this regime (≈ 1 →
  includes → use `active`). When the slope is poorly conditioned (narrow
  suspension band, or a singly-preempted task whose wall clusters at one
  value), fall back to the identity check: `raw ≈ suspended` ⇒ wall includes
  suspension.
- **Capture the event log completely.** The kernel ring buffer holds only
  ~861 GCAPS_EV lines; a post-run `dmesg` dump silently truncates a busy run to
  its last few seconds, misclassifying earlier releases as non-preempted and
  poisoning the in-run floor (see *Earlier truncated-capture runs*).
  `measure_preempt_overhead.py --run` therefore streams `dmesg --follow` for
  the duration of the run. Sanity-check: captured lines ≈ 2 × total segments.
- **Baseline.** With complete capture, the victim's in-run non-preempted floor
  is tight (±25 µs band) and matches the isolated sweep median
  (`workloadSweepGcaps` matmul 2560 = 53.375 ms; hist 512M ≈ 29.34 ms), so the
  in-run floor is self-anchoring. Passing the isolated value via
  `--baseline-ms` remains a good cross-check, and is **mandatory** when
  analysing a truncated-capture trace.
- **Report the median.** A few preempted releases over-subtract (a suspend interval
  slightly overlaps the add/remove ioctl outside the real kernel pause), giving a
  noisy negative tail; the median / per-preemption median (≈ 0) is the robust
  statistic.
- **Tuning matters.** Every task's GPU segment must be shorter than its period or the
  scenario is overloaded and the measurement is meaningless (e.g. an oversized victim
  matmul, or a preemptor matmul exceeding its 17 ms period). Verify with
  `baseline < period` per task, and keep total GPU utilization below ~0.9 —
  e.g. `-N 2560 -P 1024` ≈ 0.87 is fine, `-N 2560 -P 1536` ≈ 1.34 is overloaded;
  the `-M` defaults (512/64) ≈ 0.82.

## Caveats

- **Runlist-cache-desync deadlock** ([`runlist-cache-desync-bug.md`](runlist-cache-desync-bug.md)):
  a driver-level race that can hang longer runs; the hit rate scales with
  preemption/ioctl churn × duration. The preemption-dense `-M` variant
  deadlocked at `-d 30` and runs reliably at `-d 10` (which, with complete
  capture, already yields more attributed preemptions than a truncated 30 s
  run). The real fix is a one-line driver change (drop the `curr_tsgs_in_rl`
  skip).
- **Best-effort-task deadlock** ([`best-effort-tasks-bug.md`](best-effort-tasks-bug.md)):
  avoided by running all GPU tasks as real-time.
- The metric cannot perfectly separate active from suspended per release (the cudaEvent
  window can't be sub-divided), which is the source of the noisy tail — and of the
  ≈ ε per-preemption under-attribution visible on singly-preempted tasks; the
  central tendency is robust.

## Artifacts

- Driver patch: [`gcaps_driver_patch/ioctl_ctrl.c.patch`](gcaps_driver_patch/ioctl_ctrl.c.patch)
- Microbench: [`gcaps_userspace/preemptOverheadGcaps.cc`](gcaps_userspace/preemptOverheadGcaps.cc)
- Taskset: [`gcaps_userspace/workloadTasksetGcaps.cc`](gcaps_userspace/workloadTasksetGcaps.cc)
- Analysis: [`gcaps_userspace/scripts/measure_preempt_overhead.py`](gcaps_userspace/scripts/measure_preempt_overhead.py)
- Per-run details: [`gcaps_userspace/readme.md` § Measuring Preemption Overhead](gcaps_userspace/readme.md)
