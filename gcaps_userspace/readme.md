# GCAPS Userspace Implementation
This folder includes the userspace implementation for GCAPS method [1].
The two macros to mark the GPU segment boundaries are defined [here](common/include/support.h#L99-L131).
:exclamation::exclamation: **Before proceeding, please finish deploying the driver code first.**

## Preparations
On target Jetson device:
```bash
sudo nvpmodel -m 2 # set power mode
sudo /usr/bin/jetson_clocks # set clock speed to the max
sudo sysctl -w kernel.sched_rt_runtime_us=-1 # enable real-time priorities
```

## How to Run
First compile the executable:
```bash
cd 
git clone https://github.com/rtenlab/gcaps-super-repo.git
cd gcaps-super-repo/gcaps_userspace
make main
```

Then the case study can be run with the following command:
```bash
./main -f taskset.csv -d <duration> -i <ioctl enabled> -s <suspension enabled> -b <synchronization-based>
```

The details regarding each input argument are as follows:
- [-d]: running duration of the case study in seconds. We used 30 for the evaluation in the paper.
- [-i]: whether the GCAPS IOCTL-based approach is enabled. 1 - enabled, 0 - not enabled.
- [-s]: whether self-suspension mode is enabled. 1 - enabled, 0 - not enabled.
- [-b]: whether synchronization-based mode is enabled (the approach in previous literature [2]). The program will be aborted if [-b] and [-i] are set at the same time.

### Example Usage
```bash
# Run gcaps ioctl-based approach with self-suspension for 10 seconds
./main -f taskset.csv -d 10 -i 1 -s 1 -b 0

# Run default tsg round-robin approach with busy-waiting for 10 seconds
./main -f taskset.csv -d 10 -i 0 -s 0 -b 0
```

### Interpreting the Results
```bash
$ ./main -f taskset.csv -d 10 -i 1 -s 1 -b 0
Program configurations:
taskset: taskset.csv
duration: 10
ioctl enabled: 1
suspension: 1
sync mode: 0
---------------------------------------
...
[3525:4], 38.755001, 62.129757, 65.802002, 88.791000
[3526:5], 65.133003, 77.673798, 67.499397, 90.161003
[3522:2], 12.165000, 15.928424, 17.643499, 20.466000
[3523:3], 63.625999, 64.924858, 64.586853, 67.499001
[3524:6], 51.171001, 59.056057, 57.769749, 64.033997
[3521:1], 7.837000, 8.040460, 7.978450, 9.189000
```
For each row, it shows: [pid, task id], min, mean, perc95, max. The unit is in milliseconds.


## Varied-Workload Benchmarks
Two additional benchmarks run real workloads (tiled matmul, 256-bin histogram,
separable box convolution — ported from the SequenceScheduler suite) through
the GCAPS harness, comparing GCAPS ioctl (`-i 1`) against the default TSG
round-robin baseline (`-i 0`). Each release runs as one GCAPS GPU segment, and
the response time is decomposed into `cpu_phase + sched_preempt_overhead +
gpu_exec` (`sched_preempt_overhead` = launch/ioctl latency plus GPU-side
scheduling/preemption delay).

**Before running** (in addition to the Preparations section above):
```bash
sudo nvpmodel -m 0 && sudo jetson_clocks          # all cores + max, fixed clocks
sudo sysctl -w kernel.sched_rt_runtime_us=-1      # REQUIRED for SCHED_FIFO here
```
On these L4T kernels (`CONFIG_RT_GROUP_SCHED` + systemd cgroups) `SCHED_FIFO`
fails — even under `sudo` — unless `sched_rt_runtime_us` is `-1`; without it the
taskset's real-time tasks silently fall back to `SCHED_OTHER`.

Build them with:
```bash
make workloadSweepGcaps workloadTasksetGcaps
```

**Size sweep** (`workloadSweepGcaps`) — one workload at a time in isolation,
swept over matmul 256–2048, histogram 1M–64M, and convolution 512²–2048² ×
kernel width 3/7/15; each config released `-n` times. Writes
`results/workloadBench/sweep_{gcaps,tsg}.csv`.
```bash
./workloadSweepGcaps [-i 0|1] [-s 0|1] [-b 0|1] [-n N] [-w WARMUP]
# defaults: -i 0 -s 0 -b 0 -n 100 -w 10
sudo ./workloadSweepGcaps -i 1   # GCAPS
./workloadSweepGcaps -i 0        # TSG baseline
```

**Mixed taskset** (`workloadTasksetGcaps`) — the 7-task GCAPS Table 4 structure
(same C_i, T_i, CPU affinities and SCHED_FIFO priorities) with the real
workloads as the per-period GPU segments, one forked process per task. Writes
`results/workloadBench/taskset_{gcaps,tsg}_{trace,results}.csv`. Real-time
tasks need `sudo` for SCHED_FIFO.
```bash
sudo ./workloadTasksetGcaps [-i 0|1] [-s 0|1] [-b 0|1] [-d DURATION_S]
# defaults: -i 0 -s 0 -b 0 -d 30
sudo ./workloadTasksetGcaps -i 1 -s 1 -d 30   # GCAPS
sudo ./workloadTasksetGcaps -i 0 -s 1 -d 30   # TSG baseline
```
**Always pass `-s 1` (suspend / blocking sync) for the taskset.** With `-s 0`
(busy spin) the real-time tasks spin-wait on the GPU and, with RT throttling
disabled, starve the nvgpu driver thread and wedge the GPU. `-s 1` makes those
waits sleep instead.

**Staggered start-up.** The 7 CUDA contexts are created one at a time (one 1 s
slot per task), because bringing them all up simultaneously spins in the driver
during concurrent context init and deadlocks under `-i 1`. The timed window
therefore begins only after a one-time warm-up of ~(`NUM_TASKS`+1) s — the
binary prints when the experiment will start. The GPU workloads themselves are
very light (≈17% total GPU utilisation on AGX Orin), so the taskset is easily
schedulable once init is serialised.

If it appears to hang, see [Troubleshooting](#troubleshooting-the-taskset) below.

**Plotting** — run both modes of a benchmark, then:
```bash
python3 scripts/plot_workload_bench.py [--results-dir results/workloadBench]
```
This loads whichever of the four CSVs exist and writes `sweep_response.pdf`,
`sweep_overhead.pdf`, `taskset_mort.pdf`, `taskset_breakdown.pdf`,
`taskset_overhead.pdf`, and `taskset_gantt.pdf` into the results directory.

### Troubleshooting the taskset
- **It hangs (no progress, prompt never returns).** Almost always one of:
  forgot `-s 1` (busy spin starves the driver — see above), forgot
  `sched_rt_runtime_us=-1` (RT tasks fall back to `SCHED_OTHER`), or a previous
  hung run left the GPU wedged. Recover with `sudo pkill -9 -f workloadTasksetGcaps`;
  if `ps -eLo pid,stat,comm | grep -i workload` shows any process in `D`
  (uninterruptible) state, the GPU is wedged and only `sudo reboot` clears it.
  Always reboot after a hang before retrying.
- **`could not open results/workloadBench/...csv`.** A previous `sudo` run made
  `results/` root-owned; run with `sudo` consistently, or
  `sudo chown -R $USER:$USER results`.
- **`cudaErrorLaunchTimeout` / spinning in `taskInit`.** The simultaneous
  context-init storm — fixed by the staggered start-up; make sure you rebuilt
  (`make clean && make workloadTasksetGcaps`) and are on `-s 1`.
- **Power mode.** Mode numbers are board-specific; use `-m 0` (MAXN) on Orin for
  all cores + max clocks. Copying another board's mode number (e.g. `-m 2`) can
  silently select a reduced-core/low-clock profile.


## References
[1] Yidi Wang, Cong Liu, Daniel Wong, and Hyoseung Kim. GCAPS: GPU Context-Aware Preemptive Priority-based Scheduling for Real-Time Tasks. In Euromicro Conference on Real-Time Systems (ECRTS), 2024.
[2] Björn B Brandenburg. The FMLP+: An asymptotically optimal real-time locking protocol for suspension-aware analysis. In 2014 26th Euromicro Conference on Real-Time Systems, pages 61–71. IEEE, 2014.
