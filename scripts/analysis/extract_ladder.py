#!/usr/bin/env python
"""Locked extraction for the 64-run confirmatory dimension ladder.

Read-only. Emits one CSV row per run. Governed by docs/prereg_ladder_analysis.md.

Columns
  task, arm, seed          decoded from the array index via slurm/ladder_matrix.sh
  final_return             eval/episode_return at the last logged checkpoint
  sigma_mean               train/pi_sigma_mean at the last checkpoint.
                           NOT the per-coordinate median: per-coordinate sigmas
                           are never logged, so that quantity is unrecoverable.
  sigma_min, sigma_max     train/pi_sigma_{min,max}, for width context
  ess                      train/ess (arm A logs 0.0: pathwise uses no weights)
  alpha_kl                 dual on the forward KL E[log pi_old - log pi_theta]
                           against kl_bound. Verified identical in both arms:
                           config sets mstep_decoupled=false, so `decoupled` is
                           False for A and B alike and update_kl_lagrangian
                           applies the same lagrangian_loss in both. The E-step
                           dual against eps_E is a SEPARATE field, `eta`.
  eta                      E-step dual vs eps_E; weighted_mle only (A logs 0.0)
  entropy, kl              train/entropy, train/kl at the last checkpoint
  collapse                 POST-HOC flag: final_return < 0.05 * best on that task
  nan_flag                 any non-finite among the extracted numeric fields
  wall_clock_s             from the run's ledger entry
  n_evals, final_step      completeness of the log
"""
import argparse, csv, glob, json, math, os, re, sys

TASKS = ["g1", "leap", "hopper", "walker"]
ARMS = ["A", "B"]
SEEDS = list(range(101, 109))
LEDGER_TASK = {"g1": "G1JoystickFlatTerrain", "leap": "LeapCubeRotateZAxis",
               "hopper": "HopperHop", "walker": "WalkerRun"}
KV = re.compile(r"([A-Za-z_]\w*)=(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")
FIELDS = ["ret", "ent", "sigma", "temp", "kl", "ess", "w_max", "eta", "step"]


def decode(idx):
    """Mirror of ladder_decode in slurm/ladder_matrix.sh."""
    return TASKS[idx // 16], ARMS[idx % 2], SEEDS[(idx // 2) % 8]


def last_metrics(path):
    last, n = None, 0
    if not os.path.exists(path):
        return None, 0
    with open(path, errors="replace") as fh:
        for line in fh:
            if " step=" not in line:
                continue
            d = dict(KV.findall(line))
            if "step" not in d:
                continue
            # sigma_min/max are positional in "sigma=%.3f [%.3f,%.3f]"
            m = re.search(r"sigma=[\d.eE+-]+ \[([\d.eE+-]+),([\d.eE+-]+)\]", line)
            if m:
                d["sigma_min"], d["sigma_max"] = m.group(1), m.group(2)
            last, n = d, n + 1
    return last, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", default="/hpcwork/qzi10910/reppo_runs/logs")
    ap.add_argument("--jobid", default="3397984")
    ap.add_argument("--ledger", default="ledger/runs.d")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows = []
    for idx in range(64):
        task, arm, seed = decode(idx)
        met, n_evals = last_metrics(f"{a.logdir}/reppo-ladder_{a.jobid}_{idx}.out")
        if met is None:
            print(f"MISSING LOG idx={idx}", file=sys.stderr)
            continue
        lp = f"{a.ledger}/conf-{task}_{arm}_s{seed}.json"
        led = json.load(open(lp)) if os.path.exists(lp) else {}

        def f(k):
            try:
                return float(met.get(k, "nan"))
            except ValueError:
                return float("nan")

        vals = {k: f(k) for k in ("ret", "sigma", "sigma_min", "sigma_max",
                                  "ess", "kl", "ent", "eta", "temp")}
        rows.append(dict(
            idx=idx, task=task, arm=arm, seed=seed,
            final_return=vals["ret"], sigma_mean=vals["sigma"],
            sigma_min=vals["sigma_min"], sigma_max=vals["sigma_max"],
            ess=vals["ess"], alpha_kl=float("nan"), eta=vals["eta"],
            entropy=vals["ent"], kl=vals["kl"], alpha_ent=vals["temp"],
            nan_flag=int(any(math.isnan(v) for k, v in vals.items()
                             if k not in ("sigma_min", "sigma_max"))),
            wall_clock_s=led.get("wall_clock_s", ""),
            gpu_hours=led.get("gpu_hours", ""),
            status=led.get("status", ""), git_sha=led.get("git_sha", ""),
            n_evals=n_evals, final_step=int(float(met.get("step", "nan"))),
        ))

    # alpha_kl / final return come from the export line (authoritative final values)
    for r in rows:
        p = f"{a.logdir}/reppo-ladder_{a.jobid}_{r['idx']}.out"
        for line in open(p, errors="replace"):
            if "_final |" in line:
                m = re.search(r"alpha_kl ([\d.eE+-]+)", line)
                if m:
                    r["alpha_kl"] = float(m.group(1))
                m = re.search(r"return ([\d.eE+-]+)", line)
                if m:
                    r["final_return"] = float(m.group(1))
                m = re.search(r"ess ([\d.eE+-]+)", line)
                if m:
                    r["ess"] = float(m.group(1))

    # POST-HOC collapse flag: < 5% of the best return seen on that task
    best = {t: max(r["final_return"] for r in rows if r["task"] == t) for t in TASKS}
    for r in rows:
        r["collapse"] = int(r["final_return"] < 0.05 * best[r["task"]])
        r["task_best"] = best[r["task"]]

    cols = ["idx", "task", "arm", "seed", "final_return", "sigma_mean", "sigma_min",
            "sigma_max", "ess", "alpha_kl", "eta", "entropy", "kl", "alpha_ent",
            "collapse", "nan_flag", "wall_clock_s", "gpu_hours", "status",
            "n_evals", "final_step", "task_best", "git_sha"]
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in sorted(rows, key=lambda r: (TASKS.index(r["task"]), r["arm"], r["seed"])):
            w.writerow({c: r.get(c, "") for c in cols})
    print(f"wrote {len(rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()
