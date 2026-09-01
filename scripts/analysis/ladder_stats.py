#!/usr/bin/env python
"""Per-task statistics and diagnostics for the confirmatory ladder.

Read-only. Governed by docs/prereg_ladder_analysis.md. No pooling across tasks.
RNG seed 20260901 for every resample. 10^4 resamples.
"""
import argparse, csv, itertools, numpy as np
from collections import defaultdict

SEED = 20260901
NB = 10_000
TASKS = ["hopper", "walker", "leap", "g1"]
D = {"hopper": 4, "walker": 6, "leap": 16, "g1": 29}


def iqm(x):
    x = np.sort(np.asarray(x, float)); n = len(x); k = n // 4
    return float(x[k:n - k].mean())


def boot_ci(x, stat, rng, nb=NB):
    x = np.asarray(x, float)
    b = np.array([stat(rng.choice(x, len(x), replace=True)) for _ in range(nb)])
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--csv", required=True); a = ap.parse_args()
    rows = list(csv.DictReader(open(a.csv)))
    F = defaultdict(dict); G = defaultdict(dict)
    for r in rows:
        k = (r["task"], r["arm"]); s = int(r["seed"])
        F[k][s] = float(r["final_return"])
        G[k][s] = {c: float(r[c]) for c in ("sigma_mean", "ess", "alpha_kl", "eta", "entropy")}
        G[k][s]["collapse"] = int(r["collapse"]); G[k][s]["nan"] = int(r["nan_flag"])
    SEEDS = sorted(F[("g1", "A")])

    print("=" * 100)
    print("STEP 3 - PER-TASK STATISTICS (no pooling). RNG seed = %d, %d resamples" % (SEED, NB))
    print("=" * 100)
    fired = []
    for t in TASKS:
        rng = np.random.default_rng(SEED)
        print(f"\n### {t}  (d={D[t]}, n=8 per arm)")
        print(f"  {'arm':4} {'IQM*':>9} {'mean':>9} {'95% CI mean':>24} {'median':>9} {'95% CI median':>24}")
        for arm in "AB":
            v = [F[(t, arm)][s] for s in SEEDS]
            lo1, hi1 = boot_ci(v, np.mean, rng); lo2, hi2 = boot_ci(v, np.median, rng)
            print(f"  {arm:4} {iqm(v):9.3f} {np.mean(v):9.3f} {f'[{lo1:9.3f},{hi1:9.3f}]':>24} "
                  f"{np.median(v):9.3f} {f'[{lo2:9.3f},{hi2:9.3f}]':>24}")
        d = np.array([F[(t, 'A')][s] - F[(t, 'B')][s] for s in SEEDS])
        rng = np.random.default_rng(SEED)
        lo, hi = boot_ci(d, np.mean, rng)
        obs = abs(d.mean())
        cnt = sum(1 for sg in itertools.product([1, -1], repeat=8)
                  if abs(np.mean(np.array(sg) * d)) >= obs - 1e-12)
        det = (lo > 0) or (hi < 0)
        fired.append((t, det, lo, hi))
        print(f"  paired A-B: mean={d.mean():+.3f}  median={np.median(d):+.3f}  "
              f"95% paired CI=[{lo:+.3f},{hi:+.3f}]  exact p={cnt/256:.4f}  "
              f"W/L={int((d>0).sum())}/{int((d<0).sum())}")
        print(f"  per-seed differences: " + " ".join(f"{x:+.2f}" for x in d))
        print(f"  R1 -> gap {'DETECTED' if det else 'not detected'} (paired CI "
              f"{'excludes' if det else 'includes'} zero)")

    print("\n" + "=" * 100)
    print("STEP 4 - DIAGNOSTICS (medians over 8 seeds)")
    print("=" * 100)
    print(f"{'task':7} {'arm':4} {'sigma_mean':>11} {'ESS':>7} {'alpha_kl':>9} {'eta':>8} "
          f"{'entropy':>9} {'collapse':>9} {'NaN':>4}")
    ratios = {}
    for t in TASKS:
        med = {}
        for arm in "AB":
            g = [G[(t, arm)][s] for s in SEEDS]
            m = {c: float(np.median([x[c] for x in g])) for c in
                 ("sigma_mean", "ess", "alpha_kl", "eta", "entropy")}
            med[arm] = m
            print(f"{t:7} {arm:4} {m['sigma_mean']:11.4f} {m['ess']:7.2f} {m['alpha_kl']:9.4f} "
                  f"{m['eta']:8.4f} {m['entropy']:9.3f} "
                  f"{sum(x['collapse'] for x in g):9d} {sum(x['nan'] for x in g):4d}")
        r = max(med['A']['sigma_mean'], med['B']['sigma_mean']) / min(med['A']['sigma_mean'], med['B']['sigma_mean'])
        ratios[t] = r
    print("\nR2 - median sigma_mean ratio between arms (threshold 1.5x):")
    for t in TASKS:
        print(f"  {t:7} ratio = {ratios[t]:.3f}x  -> {'FIRES: width is a candidate cause, no operator claim' if ratios[t] > 1.5 else 'ok, operator claim permitted'}")
    print("\nRULES FIRED")
    for t, det, lo, hi in fired:
        print(f"  R1 {t:7}: {'DETECTED' if det else 'not detected'}  CI=[{lo:+.3f},{hi:+.3f}]")


if __name__ == "__main__":
    main()
