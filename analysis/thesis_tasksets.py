'''
Run the GCAPS response-time analysis (gcaps.rt_test, ioctl approach,
self-suspension mode) on the synthetic task sets of the GPU-resident scheduler
evaluation (companion repo msc-thesis-on-gpu-sched, branch single-task,
singleTaskSched/taskSets/<family>/ts_*.json), and optionally compare its verdicts
with the verdicts of the GPU-resident scheduler's own analysis.

Mapping of one task of those sets to a GCAPS Task (all times in microseconds):
  T       = period_us             (the sets have implicit deadlines; checked)
  C_hi    = cpu_phase_us          (the application's own CPU work only)
  one GPU segment if wcet_us > 0:
            Ge = wcet_us          (pure GPU work)
            Gm = --gm             (CPU work that submits the segment)
            G  = Gm + Ge
  cpu     = cpu                   (the core the task's thread is pinned to)
  prio    = the sets' deadline-monotonic `priority` (0 = highest), converted to
            GCAPS's "larger is higher"; the GPU priority equals the CPU
            priority, as in the released GCAPS driver.
  hp_set / hpp_set are built exactly as taskset_gen.taskset_generation builds
  them: every higher-priority task, and those on the same core.

epsilon (the runlist update delay) is set on the gcaps module before each run.

Usage (from this directory, like experiments.py):
  python3 thesis_tasksets.py --tasksets <.../singleTaskSched/taskSets> \
      [--results <dir with <family>/schedulability_results.csv>] \
      [--epsilon 1200 2300] [--gm 42.4] [--families ...]

--results is the directory of a schedulability_sweep.py run; its CSVs give the
GPU-resident scheduler's verdicts (rta_schedulable, rta_bounds_valid), which are
then reported next to the GCAPS counts.
'''

import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import sched_common
import gcaps

DEFAULT_FAMILIES = [
    "2-tasks-schedulability-50", "4-tasks-schedulability-50",
    "6-tasks-schedulability-50", "8-tasks-schedulability-50",
    "4-tasks-sharedmon-50", "8-tasks-sharedmon-50",
]


def build(ts, gm_us):
    raw = sorted(ts["tasks"], key=lambda t: t["priority"])   # highest first
    n = len(raw)
    tasks = []
    for i, t in enumerate(raw):
        if t["deadline_us"] != t["period_us"]:
            raise ValueError("task set has a constrained deadline; rt_test checks against T")
        k = sched_common.Task()
        k.id = i
        k.prio = n - i
        k.prio_gpu = k.prio
        k.T = float(t["period_us"])
        k.C_hi = float(t["cpu_phase_us"])
        k.cpu = t["cpu"]
        if t["wcet_us"] > 0:
            ge = float(t["wcet_us"])
            k.ges = [sched_common.GPUseg(G=gm_us + ge, Gm=gm_us, Ge=ge)]
            k.G_hi, k.Gm_hi, k.Ge_hi = gm_us + ge, gm_us, ge
        tasks.append(k)
    for i in range(n):
        for h in range(i):
            tasks[i].hp_set.append(h)
            if tasks[i].cpu == tasks[h].cpu:
                tasks[i].hpp_set.append(h)
    return tasks


def our_verdicts(path):
    v = {}
    with open(path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r["rta_schedulable"] == "True":
                v[r["task_set_id"]] = "sched"
            elif r["rta_bounds_valid"] == "False":
                v[r["task_set_id"]] = "undef"
            else:
                v[r["task_set_id"]] = "unsched"
    return v


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--tasksets', required=True, help='taskSets root of the companion repo')
    ap.add_argument('--results', default=None, help='schedulability_sweep.py results root')
    ap.add_argument('--epsilon', type=float, nargs='+', default=[1200.0, 2300.0],
                    help='runlist update delay(s), us')
    ap.add_argument('--gm', type=float, default=42.4,
                    help='CPU work that submits one GPU segment, us')
    ap.add_argument('--families', nargs='+', default=DEFAULT_FAMILIES)
    args = ap.parse_args()

    cols = ['family', 'n']
    if args.results:
        cols += ['ours sched', 'ours undef', 'ours unsched']
    cols += [f'GCAPS eps={e:g}' for e in args.epsilon]
    if args.results:
        cols += [f'both eps={e:g}' for e in args.epsilon]
    print('\t'.join(cols))

    total = defaultdict(int)
    for fam in args.families:
        files = sorted(glob.glob(os.path.join(args.tasksets, fam, '*.json')))
        ours = (our_verdicts(os.path.join(args.results, fam, 'schedulability_results.csv'))
                if args.results else None)
        c = defaultdict(int)
        for f in files:
            sid = os.path.splitext(os.path.basename(f))[0]
            with open(f, encoding='utf-8') as fh:
                ts = json.load(fh)
            c['n'] += 1
            if ours is not None:
                c['ours ' + ours[sid]] += 1
            for e in args.epsilon:
                gcaps.epsilon = e
                if gcaps.rt_test(build(ts, args.gm), False, 0):
                    c[f'GCAPS eps={e:g}'] += 1
                    if ours is not None and ours[sid] == 'sched':
                        c[f'both eps={e:g}'] += 1
        for k in cols[1:]:
            total[k] += c[k]
        print('\t'.join([fam] + [str(c[k]) for k in cols[1:]]))
    print('\t'.join(['TOTAL'] + [str(total[k]) for k in cols[1:]]))


if __name__ == '__main__':
    main()
