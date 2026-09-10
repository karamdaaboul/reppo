#!/usr/bin/env python3
"""KL mean/width split, PW vs WML baselines, from the 48 instrumentation reruns.

docs/prereg_estep_concentration_amendment1.md. Read-only.
Also tests whether the reruns are valid samples of the same arm, which matters because
G1_SAME_CODE_REPRODUCIBLE = NO (reports/g1_nondeterminism.md).
"""
import csv, os, sys
import numpy as np

os.chdir("/hpcwork/qzi10910/estep_wt"); sys.path.insert(0, os.getcwd())
RR = "/hpcwork/qzi10910/logsplit_wt/outputs"
SEEDS, NB, BOOT = list(range(301, 309)), 10000, 20260912
TASKS = {"walker": "WalkerRun", "g1": "G1JoystickFlatTerrain", "leap": "LeapCubeRotateZAxis"}
ARMS = {"WML": "weighted_mle", "PW": "pathwise_fa_noent"}

man = {}
for r in csv.DictReader(open("reports/artifacts/exports_manifest.csv")):
    man[r["dir"]] = r


def series(d, k):
    p = os.path.join(d, "metrics.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p)
    return np.asarray(z[k]).reshape(-1) if k in z.files else None


def sw3(d):
    v = series(d, "eval/episode_return")
    return float(np.mean(v[18:21])) if v is not None and v.size == 21 else None


def final(d, k):
    v = series(d, k)
    return float(v[-1]) if v is not None and v.size else None


rows = []
for task, env in TASKS.items():
    for arm, tag in ARMS.items():
        for sd in SEEDS:
            rd = os.path.join(RR, "%s_%s_s%d" % (task, arm, sd))
            cd = man.get("exports/%s_%s_s%d_final" % (env, tag, sd), {}).get("hydra_run_dir", "")
            rows.append(dict(
                task=task, arm=arm, seed=sd,
                rerun_sw3=sw3(rd), canon_sw3=sw3(cd) if cd and os.path.exists(cd) else None,
                kl_mean=final(rd, "train/fr_kl_mean_part_med"),
                kl_width=final(rd, "train/fr_kl_width_part_med"),
                kl_resid=final(rd, "train/fr_kl_split_resid_max"),
                kl_analytic=final(rd, "train/fr_kl_analytic_med")))

print("=" * 88)
print("VALIDITY  --  split residual (exactness of the decomposition)")
print("=" * 88)
res = [r["kl_resid"] for r in rows if r["kl_resid"] is not None]
if not res:
    print("  NO RUN logged fr_kl_split_resid_max -- validity gate NOT EVALUATED")
else:
    print("  max over %d/%d runs: %.3e     median: %.3e"
          % (len(res), len(rows), max(res), np.median(res)))
print("  the split is exact algebra; a large residual would mean a logging bug\n")

print("=" * 88)
print("ARE THE RERUNS VALID SAMPLES?  score_window3, rerun vs canonical")
print("=" * 88)
print("  %-7s %-4s | %11s %11s | %9s | %s" % ("task", "arm", "rerun med", "canon med", "delta", "per-seed |delta|"))
print("  " + "-" * 84)
for task in TASKS:
    for arm in ARMS:
        sub = [r for r in rows if r["task"] == task and r["arm"] == arm
               and r["rerun_sw3"] is not None and r["canon_sw3"] is not None]
        if not sub:
            print("  %-7s %-4s | canonical metrics not found" % (task, arm)); continue
        rr = np.array([r["rerun_sw3"] for r in sub]); cc = np.array([r["canon_sw3"] for r in sub])
        d = rr - cc
        print("  %-7s %-4s | %11.3f %11.3f | %+9.3f | n=%d  max %.3f  mean %.3f"
              % (task, arm, np.median(rr), np.median(cc), np.median(rr) - np.median(cc),
                 len(sub), np.abs(d).max(), np.abs(d).mean()))
    print("  " + "-" * 84)

rng = np.random.default_rng(BOOT)


def boot_ratio(per):
    a = np.asarray(per, float)
    i = rng.integers(0, a.size, size=(NB, a.size))
    s = np.exp(np.median(a[i], axis=1))
    return float(np.exp(np.median(a))), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


print("\n" + "=" * 88)
print("THE PAYLOAD  --  how each operator spends its KL budget")
print("  forward KL(pi_old || pi_theta), pre-tanh, median over states; final logged value")
print("=" * 88)
print("  %-7s %-4s | %11s %11s | %11s | %s" % ("task", "arm", "mean part", "width part", "total", "width share"))
print("  " + "-" * 84)
summ = []
for task in TASKS:
    for arm in ARMS:
        # filter on BOTH keys: a run logging the mean but not the width would put a
        # None into the width median (crash) or yield a silent NaN row.
        sub = [r for r in rows if r["task"] == task and r["arm"] == arm
               and r["kl_mean"] is not None and r["kl_width"] is not None]
        if not sub:
            print("  %-7s %-4s | no run logged the split" % (task, arm)); continue
        m = float(np.median([r["kl_mean"] for r in sub]))
        w = float(np.median([r["kl_width"] for r in sub]))
        sh = w / (m + w) if (m + w) > 0 else float("nan")
        summ.append(dict(task=task, arm=arm, mean_part=m, width_part=w, width_share=sh))
        print("  %-7s %-4s | %11.5f %11.5f | %11.5f | %9.1f%% | n=%d"
              % (task, arm, m, w, m + w, 100 * sh, len(sub)))
    print("  " + "-" * 84)

print("\n" + "=" * 88)
print("WML / PW ratio, paired within seed  (median of per-seed log ratios, exponentiated)")
print("=" * 88)
print("  %-7s | %-28s | %-28s" % ("task", "mean part", "width part"))
print("  " + "-" * 84)
out = []
for task in TASKS:
    cells = []
    for k in ("kl_mean", "kl_width"):
        per = []
        for sd in SEEDS:
            w = [r for r in rows if r["task"] == task and r["arm"] == "WML" and r["seed"] == sd][0][k]
            p = [r for r in rows if r["task"] == task and r["arm"] == "PW" and r["seed"] == sd][0][k]
            if w and p and w > 0 and p > 0:
                per.append(np.log(w / p))
        cells.append(((boot_ratio(per) if per else (float("nan"),) * 3), len(per)))
        out.append(dict(task=task, part=k, ratio=cells[-1][0][0],
                        lo=cells[-1][0][1], hi=cells[-1][0][2], n=cells[-1][1]))
    f = lambda u: "%8.3fx [%.3f, %.3f]%s n=%d" % (u[0][0], u[0][1], u[0][2],
        " *" if (u[0][1] - 1) * (u[0][2] - 1) > 0 else "  ", u[1])
    print("  %-7s | %-31s | %-31s" % (task, f(cells[0]), f(cells[1])))
print("  " + "-" * 84)
print("  * interval excludes 1")

with open("reports/artifacts/klsplit_rows.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
with open("reports/artifacts/klsplit_summary.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(summ[0].keys())); w.writeheader(); w.writerows(summ)
print("\n  wrote reports/artifacts/klsplit_rows.csv and klsplit_summary.csv")
