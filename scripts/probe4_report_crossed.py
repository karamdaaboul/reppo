"""Probe 4 aggregation and the committed statistics.

Committed procedure (docs/prospective_padding_error_field_analysis.md Sec. 1,
"States, seeds, and uncertainty"), applied verbatim:

  * Arm summary: seed MEDIAN primary, mean secondary.
  * Paired contrasts resample matched IDs.
  * 10,000 hierarchical bootstrap replicates with np.random.default_rng(20260831):
    seeds, then whole lanes within seed, then estimator batches / z-samples within
    lane.  Percentile 95% intervals.
  * Report seed values, zero-step frequency, and paired EXACT sign tests.
  * "Do not portray five seeds as precise population evidence."

Committed falsification rule for Probe 4:
  Prediction: WML is affected more.  FALSIFIED if paired median D_WML - D_PW <= 0.
"""
import glob, json, os, sys
import numpy as np
from itertools import product

OUT = sys.argv[1]
LAW = sys.argv[2] if len(sys.argv) > 2 else "ckpt"
RNG = np.random.default_rng(20260831)
NBOOT = 10000
CELLS = [f"{c}_{o}" for c in ("A", "B") for o in ("PW", "WML")]

files = sorted(glob.glob(f"{OUT}/probe4_s*_{LAW}.npz"))
if not files:
    sys.exit(f"no Probe 4 npz for law={LAW} in {OUT}")
data = {}
for f in files:
    z = np.load(f)
    data[int(z["seed"])] = z
seeds = sorted(data)
print(f"law={LAW}  seeds={seeds}  ({len(files)} files)\n")

# ---------------- per-seed values ----------------------------------------
print("=== per-seed medians (statistical unit = the training seed/checkpoint) ===")
hdr = f"{'seed':>4} " + " ".join(f"{c:>12}" for c in CELLS) + \
      f" {'B:WML-PW':>10} {'A:WML-PW':>10}"
print(hdr)
seedvals = {c: [] for c in CELLS}
for s in seeds:
    z = data[s]
    row = []
    for c in CELLS:
        v = float(np.median(z[f"{c}_D"]))
        seedvals[c].append(v); row.append(v)
    dB = row[CELLS.index("B_WML")] - row[CELLS.index("B_PW")]
    dA = row[CELLS.index("A_WML")] - row[CELLS.index("A_PW")]
    print(f"{s:>4} " + " ".join(f"{v:12.6g}" for v in row) + f" {dB:10.5g} {dA:10.5g}")
for c in CELLS:
    seedvals[c] = np.array(seedvals[c])

# ---------------- primary outcome ----------------------------------------
print("\n=== PRIMARY: paired D_WML - D_PW, within each critic (seed median) ===")
res = {}
for critic in ("A", "B"):
    d = seedvals[f"{critic}_WML"] - seedvals[f"{critic}_PW"]
    med = float(np.median(d))
    npos = int((d > 0).sum()); n = len(d)
    # exact sign test, two-sided, n=5
    from math import comb
    k = max(npos, n - npos)
    p_exact = 2 * sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n
    p_exact = min(1.0, p_exact)

    # hierarchical bootstrap: seeds -> lanes within seed -> samples within lane
    boots = np.empty(NBOOT)
    for b in range(NBOOT):
        ss = RNG.choice(seeds, size=len(seeds), replace=True)
        per = []
        for s in ss:
            z = data[s]
            lanes = z["env_id"]; ul = np.unique(lanes)
            lsel = RNG.choice(ul, size=len(ul), replace=True)
            idx = np.concatenate([np.flatnonzero(lanes == l) for l in lsel])
            idx = idx[RNG.integers(0, len(idx), len(idx))]     # within-lane resample
            per.append(np.median(z[f"{critic}_WML_D"][idx]) -
                       np.median(z[f"{critic}_PW_D"][idx]))
        boots[b] = np.median(per)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    res[critic] = dict(seed_diffs=d.tolist(), median=med, mean=float(d.mean()),
                       n_pos=npos, n=n, p_sign_exact=p_exact,
                       ci95=[float(lo), float(hi)],
                       verdict=("SUPPORTS committed prediction" if med > 0
                                else "FALSIFIES committed prediction (median <= 0)"))
    print(f"\ncritic {critic}-trained:")
    print(f"  seed diffs      : {np.array2string(d, precision=6)}")
    print(f"  paired MEDIAN   : {med:+.6g}   (committed primary summary)")
    print(f"  paired mean     : {d.mean():+.6g}   (secondary)")
    print(f"  sign pattern    : {npos}/{n} positive   exact two-sided p = {p_exact:.4f}")
    print(f"  hier. bootstrap : 95% CI [{lo:+.6g}, {hi:+.6g}]  ({NBOOT} reps, rng 20260831)")
    print(f"  COMMITTED RULE  : falsified if median <= 0  ->  {res[critic]['verdict']}")

# ---------------- secondaries -------------------------------------------
print("\n=== secondary (committed): L, cosine, weight KL, ESS -- seed medians ===")
print(f"{'cell':>8} {'D':>12} {'L':>10} {'cos':>10} {'wKL':>10} {'ESS':>8} "
      f"{'Ve<=0':>7} {'QbarMC':>9}")
sec = {}
for c in CELLS:
    r = {}
    for f in ("D", "L", "cos", "wkl", "ess"):
        v = [float(np.nanmedian(data[s][f"{c}_{f}"])) for s in seeds]
        r[f] = float(np.median(v))
    r["Ve_nonpos"] = int(sum(int(data[s][f"{c}_Ve_nonpos"]) for s in seeds))
    r["Qbar_mc"] = float(np.mean([float(data[s][f"{c}_Qbar_mc"]) for s in seeds]))
    r["Qspread"] = float(np.mean([float(data[s][f"{c}_Qspread"]) for s in seeds]))
    sec[c] = r
    print(f"{c:>8} {r['D']:12.6g} {r['L']:10.5g} {r['cos']:10.6g} "
          f"{r['wkl']:10.4g} {r['ess']:8.3g} {r['Ve_nonpos']:7d} {r['Qbar_mc']:9.3g}")

print("\n=== integrity ===")
tot_ve = sum(sec[c]["Ve_nonpos"] for c in CELLS)
print(f"  V_e <= 0 tripwire total (must be 0)        : {tot_ve}")
print(f"  Qbar MC s.e. vs padded Q spread            : "
      + ", ".join(f"{c} {sec[c]['Qbar_mc']/max(sec[c]['Qspread'],1e-30):.4f}" for c in CELLS))
print(f"  eta read verbatim from ckpt (A, B) per seed: "
      + ", ".join(f"s{s}:({float(data[s]['eta_A']):.4g},{float(data[s]['eta_B']):.4g})"
                  for s in seeds))
zero = {c: float(np.mean([np.mean(data[s][f"{c}_D"] == 0) for s in seeds])) for c in CELLS}
print(f"  zero-step frequency                        : "
      + ", ".join(f"{c} {zero[c]:.4f}" for c in CELLS))

json.dump(dict(law=LAW, seeds=seeds, primary=res, secondary=sec,
               zero_step_freq=zero, nboot=NBOOT, rng_seed=20260831),
          open(f"{OUT}/probe4_result_{LAW}.json", "w"), indent=2)
print(f"\nwrote {OUT}/probe4_result_{LAW}.json")
