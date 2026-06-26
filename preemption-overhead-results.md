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
| Per-preemption scheduling + context-switch overhead | **≈ 0.6 ms** (601 µs median, 631 µs p95, **659 µs worst**) |
| Active-execution extension per preemption | **≈ 0** (median −0.03 ms, p95 +0.16 ms) |

**GCAPS preemption does not measurably extend a job's active execution time.**
The cost of a preemption is the ~0.6 ms scheduling/context-switch overhead plus
the suspension (blocking) time — and suspension is response-time, not
execution-time, so it is excluded by construction.

## Setup

- **Hardware / OS:** Jetson AGX Orin (`ga10b`), L4T R35.6.0 / JetPack 5.1.4, GCAPS-patched
  `nvgpu`. MAXN power, locked clocks, RT throttling off:
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
- **Controlled microbenchmark** ([`gcaps_userspace/preemptOverheadGcaps.cc`](gcaps_userspace/preemptOverheadGcaps.cc)):
  three tasks at three distinct SCHED_FIFO priorities —

  | role | workload | prio | period |
  |---|---|---|---|
  | victim | matmul 2048 (~27 ms) | 2 | 80 ms |
  | preemptorB | histogram 8M (~0.28 ms) | 4 | 29 ms |
  | preemptorA | matmul 1024 (~3.3 ms) | 5 | 17 ms |

  The victim (lowest priority, long segment) is preempted by both; preemptorB is
  itself preempted by preemptorA, so preemptions occur at more than one level.
  Run: `sudo ./preemptOverheadGcaps -i 1 -s 1 -d 30 -N 2048 -P 1024`.

The realistic 7-task GCAPS Table-4 taskset
([`workloadTasksetGcaps.cc`](gcaps_userspace/workloadTasksetGcaps.cc), `--run taskset`)
is also instrumented and can be measured the same way.

## How to reproduce

```bash
# 1. clean uncontended victim baseline (matmul 2048, in isolation)
sudo ./workloadSweepGcaps -i 1 -s 1 -n 100 -w 20
awk -F, '$1=="matmul_2048"{print $5}' results/workloadBench/sweep_gcaps.csv \
  | sort -n | awk '{a[NR]=$1} END{print "median="a[int((NR+1)/2)]}'
# -> ~27.075 ms (n=100, range 27.064-27.086)

# 2. run the microbench under GCAPS and analyse, anchoring the victim to the
#    isolated baseline
sudo python3 scripts/measure_preempt_overhead.py --run microbench -d 30 \
     --extra "-N 2048 -P 1024"
python3 scripts/measure_preempt_overhead.py \
    --events results/workloadBench/preempt_gcaps_events.log \
    --trace  results/workloadBench/preempt_gcaps_trace.csv \
    --baseline-ms mm_victim=27.075
```

## Detailed results

### (1) Scheduling + context-switch / preemption overhead

`elapsed_us` of each ioctl, split by what it did (171 real preemptions over a 30 s run):

| event | n | min | median | p95 | max |
|---|---:|---:|---:|---:|---:|
| preempting add (a real preemption) | 171 | 586 | 601 | 631 | **659 µs** |
| resuming remove (victim put back) | 209 | 604 | 631 | 660 | 671 µs |
| real runlist reload (`rlupd=1`) | 841 | 532 | 572 | 647 | 671 µs |
| bookkeeping only (`rlupd=0`) | 46 | 12 | 13 | 29 | 31 µs |

The sharp bimodal split — real reloads ~0.6 ms vs no-op calls ~13 µs — confirms the
instrumentation isolates the runlist-update work.

### (2) Execution-time extension when preempted

Victim (`mm_victim`), baseline = isolated **27.075 ms**, 52 preempted releases:

| metric | median | mean | p95 |
|---|---:|---:|---:|
| gpu_wall when preempted | 36.20 ms | 36.73 | 40.14 |
| suspended (blocking, excluded) | 9.08 ms | 10.05 | 14.08 |
| **extension (active)** | **−0.11 ms** | −0.39 | +0.38 |
| **extension per preemption (active)** | **−0.03 ms** | −0.09 | +0.16 |

`active = gpu_wall − suspended ≈ 27.1 ms ≈ baseline 27.075` → extension ≈ 0.
`hist_preB` likewise ≈ 0 (per-preemption median −0.28 ms, p95 +0.57 ms).

## Methodology notes

- **`active = gpu_wall − suspended`.** The cudaEvent window (`gpu_wall`) on AGX Orin
  *includes* the suspended time (verified: `gpu_wall ≈ active + suspended`, e.g.
  40.0 − 13.0 = 27.0), so subtracting the driver-reported suspension recovers the
  active execution. The analysis prints a `d(gpu_wall)/d(suspended)` slope per task
  to confirm this regime (≈ 1 → includes → use `active`).
- **Baseline must be clean.** The victim's *in-run* non-preempted releases are not a
  valid baseline: they are bimodal — a contended ~36 ms bulk over a rare ~27 ms
  clean floor (the victim, lowest priority, is slowed by co-running preemptors even
  when not GCAPS-evicted). So the baseline is taken from an **isolated** measurement
  (`workloadSweepGcaps` matmul 2048 = 27.075 ms) via `--baseline-ms`. This isolated
  value matches the in-run clean floor (`min 27.09 ms`), so the two agree.
- **Report the median.** A few preempted releases over-subtract (a suspend interval
  slightly overlaps the add/remove ioctl outside the real kernel pause), giving a
  noisy negative tail; the median / per-preemption median (≈ 0) is the robust
  statistic.
- **Tuning matters.** Every task's GPU segment must be shorter than its period or the
  scenario is overloaded and the measurement is meaningless (e.g. an oversized victim
  matmul, or a preemptor matmul exceeding its 17 ms period). Verify with
  `baseline < period` per task.

## Caveats

- **Runlist-cache-desync deadlock** ([`runlist-cache-desync-bug.md`](runlist-cache-desync-bug.md)):
  a driver-level race that can hang longer runs. Larger GPU segments (lower ioctl
  churn) and shorter `-d` reduce the hit rate; the real fix is a one-line driver
  change (drop the `curr_tsgs_in_rl` skip). The 30 s microbench runs here completed.
- **Best-effort-task deadlock** ([`best-effort-tasks-bug.md`](best-effort-tasks-bug.md)):
  avoided by running all GPU tasks as real-time.
- The metric cannot perfectly separate active from suspended per release (the cudaEvent
  window can't be sub-divided), which is the source of the noisy tail; the central
  tendency is robust.

## Artifacts

- Driver patch: [`gcaps_driver_patch/ioctl_ctrl.c.patch`](gcaps_driver_patch/ioctl_ctrl.c.patch)
- Microbench: [`gcaps_userspace/preemptOverheadGcaps.cc`](gcaps_userspace/preemptOverheadGcaps.cc)
- Taskset: [`gcaps_userspace/workloadTasksetGcaps.cc`](gcaps_userspace/workloadTasksetGcaps.cc)
- Analysis: [`gcaps_userspace/scripts/measure_preempt_overhead.py`](gcaps_userspace/scripts/measure_preempt_overhead.py)
- Per-run details: [`gcaps_userspace/readme.md` § Measuring Preemption Overhead](gcaps_userspace/readme.md)
