"""Primary paired analysis, exactly as preregistered in
docs/prereg_corrected_operator_replication.md sections 4 and 5.

Final score  = mean of the final three logged evaluations (indices 18,19,20 of 21).
Secondary    = last logged evaluation alone (the 64-run study's locked definition).
Contrast     = PW-1 minus WML-32, paired within seed. Positive = pathwise higher.
Unit         = the seed pair. Uncertainty = paired percentile bootstrap, 10000
               resamples, np.random.default_rng(20260902). Exact two-sided sign test.
"""
import json, glob, os, csv
import numpy as np
from math import comb

RNG = np.random.default_rng(20260902)
NB = 10000
led = [json.loads(l) for l in open("ledger/runs_faithful_repair.jsonl")]
ROOT = "/hpcwork/qzi10910/reppo_runs"

rows = []
for r in led:
    npz = "%s/%s/metrics.npz" % (ROOT, r["output_path"])
    z = np.load(npz)
    ret = np.asarray(z["eval/episode_return"]).reshape(-1)
    g = lambda k: (np.asarray(z[k]).reshape(np.asarray(z[k]).shape[0], -1)[:, 0]
                   if k in z.files else np.full(ret.shape, np.nan))
    kl_s, kl_a = g("train/fr_kl_q50"), g("train/fr_kl_analytic_med")
    rows.append(dict(
        run_id=r["run_id"], task=r["task_key"], env=r["task"], d=r["d"], arm=r["arm"],
        seed=r["seed"], gpu=r["gpu_architecture"], slurm=r["slurm_job"],
        n_evals=len(ret),
        score_window3=float(np.mean(ret[-3:])),     # PRIMARY
        score_last=float(ret[-1]),                   # SECONDARY (64-run definition)
        ret_curve=";".join("%.4f" % v for v in ret),
        gate_operator_frac=float(np.nanmean(g("train/fr_gate_operator"))),
        gate_klonly_frac=float(np.nanmean(g("train/fr_gate_kl_only"))),
        kl_sampled_med=float(np.nanmean(kl_s)),
        kl_analytic_med=float(np.nanmean(kl_a)),
        kl_bias=float(np.nanmean(g("train/fr_kl_sampled_minus_analytic_mean"))),
        lag_eff=float(np.nanmean(g("train/fr_lag_eff"))),
        lag_raw=float(np.nanmean(g("train/fr_lag_raw"))),
        sigma_mean=float(np.nanmean(g("train/fr_sigma_mean"))),
        action_sat=float(np.nanmean(g("train/fr_action_sat"))),
        ess=float(np.nanmean(g("train/ess"))), eta=float(np.nanmean(g("train/eta"))),
        critic_loss=float(np.nanmean(g("train/value_loss"))),
    ))
os.makedirs("reports/artifacts", exist_ok=True)
with open("reports/artifacts/corrected_runs.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
    for r in rows: w.writerow(r)

def iqm(x):
    x = np.sort(np.asarray(x, float)); n = len(x); k = n // 4
    return float(np.mean(x[k:n - k])) if n >= 4 else float(np.mean(x))

def paired(pw, wm):
    d = np.asarray(pw) - np.asarray(wm)
    b = np.array([np.median(RNG.choice(d, len(d), replace=True)) for _ in range(NB)])
    lo, hi = np.percentile(b, [2.5, 97.5])
    npos = int((d > 0).sum()); n = len(d)
    k = max(npos, n - npos)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n)
    return d, float(np.median(d)), float(d.mean()), lo, hi, npos, n, p

print("=" * 78)
print("PRIMARY PAIRED ANALYSIS  --  PW-1 minus WML-32  (positive = pathwise higher)")
print("score = mean of final three evaluations; unit = seed pair; n = 8")
print("=" * 78)
out = []
for score_key, label in (("score_window3", "PRIMARY final-window(3)"),
                         ("score_last", "SECONDARY last-eval (64-run definition)")):
    print("\n### %s" % label)
    for task in ("walker", "g1"):
        sub = [r for r in rows if r["task"] == task]
        pw = [r for r in sub if r["arm"].startswith("PW")]
        wm = [r for r in sub if r["arm"].startswith("WML")]
        pw.sort(key=lambda r: r["seed"]); wm.sort(key=lambda r: r["seed"])
        assert [r["seed"] for r in pw] == [r["seed"] for r in wm]
        a = [r[score_key] for r in pw]; b = [r[score_key] for r in wm]
        d, med, mean, lo, hi, npos, n, p = paired(a, b)
        det = (lo > 0) or (hi < 0)
        print("  %-7s d=%-3d  PW-1 IQM=%8.2f   WML-32 IQM=%8.2f" %
              (task, sub[0]["d"], iqm(a), iqm(b)))
        print("          seeds      : %s" % [r["seed"] for r in pw])
        print("          PW-1       : %s" % np.round(a, 2).tolist())
        print("          WML-32     : %s" % np.round(b, 2).tolist())
        print("          paired diff: %s" % np.round(d, 2).tolist())
        print("          median %+8.2f   mean %+8.2f   95%% CI [%+8.2f, %+8.2f]"
              % (med, mean, lo, hi))
        print("          sign %d/%d positive   exact two-sided p = %.4f   -> %s"
              % (npos, n, p, "DETECTED (CI excludes zero)" if det else "not detected"))
        if score_key == "score_window3":
            out.append(dict(task=task, d=sub[0]["d"], score=label,
                            seeds=";".join(str(r["seed"]) for r in pw),
                            pw=";".join("%.4f" % v for v in a),
                            wml=";".join("%.4f" % v for v in b),
                            diffs=";".join("%.4f" % v for v in d),
                            pw_iqm=iqm(a), wml_iqm=iqm(b), median_diff=med,
                            mean_diff=mean, ci_lo=lo, ci_hi=hi, n_pos=npos, n=n,
                            p_exact=p, detected=int(det)))
with open("reports/artifacts/corrected_paired_results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader()
    for r in out: w.writerow(r)
print("\nwrote corrected_runs.csv (%d rows), corrected_paired_results.csv" % len(rows))
