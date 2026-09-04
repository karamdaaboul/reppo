"""Preregistered primary analysis for corrected LEAP.

docs/prereg_leap_corrected.md sec 6-7, inherited verbatim from
scripts/analysis/fr_analyse.py: score = mean of the final three logged
evaluations; contrast PW-1 minus WML-32 paired within seed; paired percentile
bootstrap, 10000 resamples, np.random.default_rng(20260902); exact two-sided
sign test; IQM per arm.
"""
from __future__ import annotations
import glob, json, os
import numpy as np
from math import comb

os.chdir("/rwthfs/rz/cluster/home/qzi10910/repos/reppo")
RNG = np.random.default_rng(20260902)
NB = 10000
ROOT = os.environ.get("HPCWORK", "/hpcwork/qzi10910") + "/reppo_runs"
SEEDS = list(range(301, 309))

def score(arm, seed):
    p = "%s/outputs/leap_corrected/leap_%s_s%d/metrics.npz" % (ROOT, arm, seed)
    z = np.load(p)
    r = np.asarray(z["eval/episode_return"]).reshape(-1)
    return float(np.mean(r[-3:])), float(r[-1]), len(r)

def iqm(x):
    x = np.sort(np.asarray(x, float)); n = len(x); k = n // 4
    return float(x[k:n-k].mean()) if n >= 4 else float(x.mean())

pw, wml, rows = [], [], []
for s in SEEDS:
    a, a_last, na = score("PW", s)
    b, b_last, nb = score("WML", s)
    pw.append(a); wml.append(b)
    rows.append((s, a, b, a-b, a_last, b_last, na, nb))

print("=== per-seed, score_window3 (mean of final three of %d logged evals) ===" % rows[0][6])
print("  %-5s %12s %12s %12s | %12s %12s" % ("seed","PW-1","WML-32","Delta","PW last","WML last"))
for s,a,b,d,al,bl,na,nb in rows:
    print("  %-5d %12.4f %12.4f %+12.4f | %12.4f %12.4f" % (s,a,b,d,al,bl))

d = np.array([r[3] for r in rows])
med, mean = float(np.median(d)), float(d.mean())
boot = np.array([np.median(RNG.choice(d, len(d), replace=True)) for _ in range(NB)])
lo, hi = np.percentile(boot, [2.5, 97.5])
npos = int((d > 0).sum()); n = len(d)
p = 2 * sum(comb(n, k) for k in range(max(npos, n-npos), n+1)) / 2**n
p = min(1.0, p)

print()
print("=== PRIMARY: Delta_LEAP = R_PW - R_WML, paired within seed ===")
print("  IQM        PW-1 %.4f   WML-32 %.4f" % (iqm(pw), iqm(wml)))
print("  median     %+.4f" % med)
print("  mean       %+.4f" % mean)
print("  95%% CI     [%+.4f, %+.4f]   (paired percentile bootstrap, %d resamples,"
      " default_rng(20260902))" % (lo, hi, NB))
print("  n_pos      %d/%d      exact two-sided sign test p = %.6f" % (npos, n, p))
print()
print("=== preregistered decision rule (sec 7), applied mechanically ===")
print("  point estimate positive?          %s" % (med > 0))
print("  CI entirely ABOVE zero?           %s" % (lo > 0))
print("  CI contains zero?                 %s" % (lo <= 0 <= hi))
print("  CI entirely BELOW zero (CI_upper<0)? %s" % (hi < 0))
if med > 0 and lo > 0:
    verdict = "PW-supported"
elif hi < 0:
    verdict = "Strong falsifier"
elif lo <= 0 <= hi:
    verdict = "Inconclusive"
else:
    verdict = "Inconclusive"
print()
print("  CLASSIFICATION: %s" % verdict)
json.dump(dict(per_seed=[dict(seed=r[0], pw=r[1], wml=r[2], delta=r[3],
                              pw_last=r[4], wml_last=r[5]) for r in rows],
               median=med, mean=mean, ci=[float(lo), float(hi)], n_pos=npos, n=n,
               p_sign=p, iqm_pw=iqm(pw), iqm_wml=iqm(wml),
               n_boot=NB, rng=20260902, classification=verdict),
          open("reports/artifacts/leap_corrected_results.json","w"), indent=1)
print("\n  wrote reports/artifacts/leap_corrected_results.json")
