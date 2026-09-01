"""Aggregate Step 2, evaluate the preregistered gates P1-P5, and draw the figure.

docs/prereg_ubar_ratio.md Sec. 6-7. The independent unit is the checkpoint seed;
states and action clouds are repeated measurements within a checkpoint.
"""
import glob, os, sys, json
import numpy as np, pandas as pd
from scipy.special import gammaln, chdtri
from scipy.stats import spearmanr

OUT = "reports/artifacts"
EPS = 1e-12
RNG = np.random.default_rng(20260902)
TASK = {"HopperHop": "hopper", "WalkerRun": "walker",
        "LeapCubeRotateZAxis": "leap", "G1JoystickFlatTerrain": "g1"}

# ---------------- per-checkpoint aggregation --------------------------------
rows, per_state = [], []
for f in sorted(glob.glob(f"{OUT}/ubar/*.npz")):
    z = np.load(f, allow_pickle=True)
    g = lambda k: np.asarray(z[k])
    d, Mv = int(z["d"]), int(z["M"])
    task = TASK[str(z["env"])]
    arm = "A" if str(z["mode"]) == "pathwise" else "B"
    pad = int(z["action_pad"])
    cond = f"{task}{'-pad16' if pad else ''}"
    nub, nc, nlin = g("fit_n_ubar"), g("fit_n_c"), g("fit_n_lin")
    raw_nub = g("raw_n_ubar")
    exp_rms = np.sqrt(d / Mv)
    exp_mean = np.sqrt(2.0 / Mv) * np.exp(gammaln((d + 1) / 2) - gammaln(d / 2))
    exp_med = np.sqrt(chdtri(d, 0.5)) / np.sqrt(Mv)
    ub_fit_vec, ub_raw_vec = g("ubar_fit_vec"), g("ubar_raw_vec")
    R = g("fit_R_exact")
    cvc = g("fit_cos_v_c")
    cl_R = g("cloud_R_exact")            # (16, 256)
    rows.append(dict(
        condition=cond, task=task, arm=arm, d=d, pad=pad, seed=int(z["seed"]),
        checkpoint=str(z["tag"]), n_states=int(z["N"]),
        eta=float(z["eta"]), eta_src=str(z["eta_src"]),
        eta_measured=float(z["eta_measured"]),
        clip_rate=float(z["clip_rate"]),
        # P3 primary
        R2_exact=float(np.sqrt(np.sum(nub ** 2) / max(np.sum(nc ** 2), EPS))),
        R2_linear=float(np.sqrt(np.sum(nub ** 2) / max(np.sum(nlin ** 2), EPS))),
        R_exact_med=float(np.median(R)),
        R_exact_q25=float(np.quantile(R, .25)), R_exact_q75=float(np.quantile(R, .75)),
        R_linear_med=float(np.median(g("fit_R_linear"))),
        frac_R_gt_05=float(np.mean(R > 0.5)), frac_R_gt_1=float(np.mean(R > 1)),
        frac_R_gt_3=float(np.mean(R > 3)),
        # P1
        norm_ratio_med=float(np.median(g("fit_norm_ratio"))),
        cosine_linear_med=float(np.median(g("fit_cosine_linear"))),
        residual_linear_med=float(np.median(g("fit_residual_linear"))),
        # direction
        cos_v_c_med=float(np.median(cvc)),
        cos_ubar_c_med=float(np.median(g("fit_cos_ubar_c"))),
        cross_fraction_med=float(np.median(g("fit_cross_fraction"))),
        direction_change_med=float(np.median(g("fit_direction_change"))),
        frac_cosvc_lt_09=float(np.mean(cvc < 0.9)), frac_cosvc_lt_05=float(np.mean(cvc < 0.5)),
        # P2
        ubar_raw_rms=float(np.sqrt(np.mean(raw_nub ** 2))), ubar_raw_rms_exp=float(exp_rms),
        ubar_raw_mean=float(raw_nub.mean()), ubar_raw_mean_exp=float(exp_mean),
        ubar_raw_med=float(np.median(raw_nub)), ubar_raw_med_exp=float(exp_med),
        ubar_fit_rms=float(np.sqrt(np.mean(nub ** 2))),
        ubar_fit_meanvec_norm=float(np.linalg.norm(ub_fit_vec.mean(0))),
        ubar_raw_meanvec_norm=float(np.linalg.norm(ub_raw_vec.mean(0))),
        ubar_fit_minus_raw_rms=float(np.sqrt(np.mean(
            np.sum((ub_fit_vec - ub_raw_vec) ** 2, -1)))),
        # context
        logit_spread_med=float(np.median(g("logit_spread"))),
        q_sd_med=float(np.median(g("q_sd"))), ess_med=float(np.median(g("ess"))),
        w_max_med=float(np.median(g("w_max"))), sigma_med=float(np.median(g("sigma_mean"))),
        # cloud vs state variability
        cloud_between_state_sd=float(np.std(cl_R.mean(0))),
        cloud_within_state_sd=float(np.mean(np.std(cl_R, axis=0))),
    ))
    sub = np.linspace(0, len(R) - 1, 128).astype(int)
    for i in sub:
        per_state.append(dict(
            condition=cond, task=task, arm=arm, d=d, seed=int(z["seed"]), state=int(i),
            eta=float(z["eta"]), eta_src=str(z["eta_src"]),
            n_ubar=float(nub[i]), n_c=float(nc[i]), n_lin=float(nlin[i]),
            n_v=float(g("fit_n_v")[i]), R_exact=float(R[i]),
            R_linear=float(g("fit_R_linear")[i]),
            norm_ratio=float(g("fit_norm_ratio")[i]),
            cosine_linear=float(g("fit_cosine_linear")[i]),
            residual_linear=float(g("fit_residual_linear")[i]),
            cos_ubar_c=float(g("fit_cos_ubar_c")[i]), cos_v_c=float(cvc[i]),
            cross_fraction=float(g("fit_cross_fraction")[i]),
            direction_change=float(g("fit_direction_change")[i]),
            logit_spread=float(g("logit_spread")[i]), q_sd=float(g("q_sd")[i]),
            ess=float(g("ess")[i]), w_max=float(g("w_max")[i]),
            n_ubar_raw=float(raw_nub[i])))

df = pd.DataFrame(rows).sort_values(["d", "task", "pad", "arm", "seed"])
df.to_csv(f"{OUT}/ubar_per_checkpoint.csv", index=False)
pd.DataFrame(per_state).to_csv(f"{OUT}/ubar_per_state.csv", index=False)
print("checkpoints aggregated: %d" % len(df))
print("conditions            : %d" % df.groupby(["condition", "arm"]).ngroups)

def boot(vals, n=10000):
    v = np.asarray(vals, float)
    b = [np.median(RNG.choice(v, len(v), replace=True)) for _ in range(n)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))

# ---------------- P1 gate ----------------------------------------------------
print("\n=== P1 first-order identity: c ~= m_hat/eta  (checkpoint-level medians) ===")
print("gate: norm_ratio in [0.8,1.25] AND cosine_linear>=0.95 AND residual_linear<=0.25")
p1 = []
for (cond, arm), s in df.groupby(["condition", "arm"]):
    nr, cl, rl = s.norm_ratio_med.median(), s.cosine_linear_med.median(), s.residual_linear_med.median()
    ok = (0.8 <= nr <= 1.25) and cl >= 0.95 and rl <= 0.25
    p1.append(dict(condition=cond, arm=arm, d=int(s.d.iloc[0]), n=len(s),
                   norm_ratio=nr, cosine_linear=cl, residual_linear=rl,
                   logit_spread=s.logit_spread_med.median(), P1=("PASS" if ok else "FAIL")))
p1 = pd.DataFrame(p1).sort_values(["d", "condition", "arm"])
print(p1.to_string(index=False, float_format=lambda x: "%.4g" % x))
print("\nP1 verdict: %d/%d conditions pass" % ((p1.P1 == "PASS").sum(), len(p1)))
P1_ANY = (p1.P1 == "PASS").any()

# ---------------- P2 raw-Gaussian sanity ------------------------------------
print("\n=== P2 raw-Gaussian sanity (ubar_raw only; NOT applied to ubar_fit) ===")
p2 = df.groupby(["condition", "arm", "d"]).agg(
    rms=("ubar_raw_rms", "median"), rms_exp=("ubar_raw_rms_exp", "first"),
    mean=("ubar_raw_mean", "median"), mean_exp=("ubar_raw_mean_exp", "first"),
    med=("ubar_raw_med", "median"), med_exp=("ubar_raw_med_exp", "first")).reset_index()
p2["rms_ratio"] = p2["rms"] / p2["rms_exp"]
p2["mean_ratio"] = p2["mean"] / p2["mean_exp"]
p2["med_ratio"] = p2["med"] / p2["med_exp"]
print(p2.to_string(index=False, float_format=lambda x: "%.4f" % x))
print("worst |ratio-1| across all three benchmarks: %.4f"
      % max(np.abs(p2[["rms_ratio", "mean_ratio", "med_ratio"]].values - 1).max(), 0))

print("\n=== P2b clipping / transformation-induced bias in ubar_fit ===")
p2b = df.groupby(["condition", "arm", "d"]).agg(
    clip_rate=("clip_rate", "median"), fit_rms=("ubar_fit_rms", "median"),
    raw_rms=("ubar_raw_rms", "median"),
    fit_meanvec=("ubar_fit_meanvec_norm", "median"),
    raw_meanvec=("ubar_raw_meanvec_norm", "median"),
    fit_minus_raw=("ubar_fit_minus_raw_rms", "median")).reset_index()
print(p2b.to_string(index=False, float_format=lambda x: "%.5g" % x))

# ---------------- P3 ---------------------------------------------------------
print("\n=== P3 noise-to-signal, seed as the unit (R2_exact is PRIMARY) ===")
p3 = []
for (cond, arm), s in df.groupby(["condition", "arm"]):
    lo, hi = boot(s.R2_exact.values)
    p3.append(dict(condition=cond, arm=arm, d=int(s.d.iloc[0]), n=len(s),
                   R2_exact_med=s.R2_exact.median(), ci_lo=lo, ci_hi=hi,
                   R2_exact_iqr=s.R2_exact.quantile(.75) - s.R2_exact.quantile(.25),
                   R_exact_med=s.R_exact_med.median(),
                   R2_linear_med=s.R2_linear.median(),
                   cos_v_c=s.cos_v_c_med.median(), cos_ubar_c=s.cos_ubar_c_med.median(),
                   dir_change=s.direction_change_med.median(),
                   fR05=s.frac_R_gt_05.median(), fR1=s.frac_R_gt_1.median(),
                   fR3=s.frac_R_gt_3.median(),
                   fcos09=s.frac_cosvc_lt_09.median(), fcos05=s.frac_cosvc_lt_05.median(),
                   ess=s.ess_med.median(), spread=s.logit_spread_med.median()))
p3 = pd.DataFrame(p3).sort_values(["d", "condition", "arm"])
print(p3.to_string(index=False, float_format=lambda x: "%.4g" % x))

# ---------------- P4 matched dimension contrast ------------------------------
print("\n=== P4 matched within-task dimension contrast: WalkerRun d=6 vs d=22 ===")
p4lines = []
for arm in ("A", "B"):
    lo = df[(df.task == "walker") & (df.pad == 0) & (df.arm == arm)].R2_exact.values
    hi = df[(df.task == "walker") & (df.pad == 16) & (df.arm == arm)].R2_exact.values
    if len(lo) == 0 or len(hi) == 0:
        continue
    ratios = np.array([h / l for h in hi for l in lo])       # all-pairs, seeds do not match
    med = float(np.median(ratios))
    verdict = ("DIMENSION AMPLIFICATION REFUTED IN THE MATCHED ESTIMATOR PROBE" if med < 1.3
               else "DIMENSION AMPLIFICATION OBSERVED; UBAR IS A LIVE ESTIMATOR-LEVEL MECHANISM"
               if med > 2.0 else "DIMENSION AMPLIFICATION UNDECIDED")
    print("  arm %s : d=6 R2=%s" % (arm, np.round(lo, 4)))
    print("           d=22 R2=%s" % np.round(hi, 4))
    print("           all-pairs Rho median=%.4f  IQR=[%.4f, %.4f]  n_pairs=%d"
          % (med, np.percentile(ratios, 25), np.percentile(ratios, 75), len(ratios)))
    print("           VERDICT: %s" % verdict)
    p4lines.append(dict(arm=arm, rho_median=med, verdict=verdict,
                        lo=list(map(float, lo)), hi=list(map(float, hi))))
print("  NOTE: unpaired in seed (101-108 vs 0-4) and alpha differs "
      "(0.014509915 vs 0.01528); registered as a limitation.")

print("\n=== cross-task descriptive contrast (formal verdict fixed in advance) ===")
print("  CAUSAL DIMENSION SCALING NOT IDENTIFIABLE FROM THE AVAILABLE CHECKPOINTS.")
for arm in ("A", "B"):
    s = df[df.arm == arm].groupby("d").R2_exact.median()
    print("  arm %s  R2_exact by d: %s" % (arm, ", ".join("d=%d:%.4g" % (k, v) for k, v in s.items())))

# ---------------- P5 weak-coupling calibration -------------------------------
print("\n=== P5 weak-coupling calibration: R_linear vs 1/logit_spread (per state) ===")
ps = pd.read_csv(f"{OUT}/ubar_per_state.csv")
for (cond, arm), s in ps.groupby(["condition", "arm"]):
    pred = 1.0 / np.maximum(s.logit_spread.values, EPS)
    rho, _ = spearmanr(s.R_linear.values, pred)
    print("  %-14s %s  median R_linear=%.4g  median 1/spread=%.4g  ratio=%.4g  spearman=%+.3f"
          % (cond, arm, np.median(s.R_linear), np.median(pred),
             np.median(s.R_linear) / max(np.median(pred), EPS), rho))

json.dump(dict(p1=p1.to_dict("records"), p3=p3.to_dict("records"), p4=p4lines,
               P1_any_pass=bool(P1_ANY)), open(f"{OUT}/ubar_gates.json", "w"), indent=2)
print("\nwrote ubar_per_checkpoint.csv, ubar_per_state.csv, ubar_gates.json")
