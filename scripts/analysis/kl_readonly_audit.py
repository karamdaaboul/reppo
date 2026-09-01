#!/usr/bin/env python
"""Read-only KL / recoverability audit over the 64 confirmatory ladder runs.

SCOPE, fixed before the numbers are read
----------------------------------------
This script RETRIEVES retained quantities and computes descriptive diagnostics.
It performs no training, writes nothing into the run tree, and licenses no
causal claim.  In particular:

* The logged scalar `train/kl` is proven in the code trace to be
  `kl.mean()` (src/jaxrl/reppo.py:1002) of the SAME per-state tensor `kl`
  (shape (B,), src/jaxrl/reppo.py:722) that the defective branch
  `jnp.where(kl < cfg.kl_bound, objective, kl * sg(lagrangian) * reduce_kl)`
  (src/jaxrl/reppo.py:850-855) tests elementwise.  Because a finite mean never
  exceeds the maximum, `logged_kl_t >= kl_bound` PROVES at least one element
  took the gated branch at that logging point.  This is a LOWER BOUND on gate
  activity and identifies neither the fraction of elements firing nor the
  behaviour at logging points below the bound.
* No quantity here estimates omega = ||grad e|| / ||e||.  Critic gradient norms,
  Bellman residuals and seed disagreement are NOT omega and are not computed as
  substitutes.
* Associations are reported at the seed level with n=8 per arm.  They are
  descriptive.  No mediation, no causal attribution.

Outputs: per-run CSV, per-(task,arm) summary CSV, figures, and the numbers the
report quotes.
"""
import json, os, glob, sys
import numpy as np
import pandas as pd

OUT = sys.argv[1] if len(sys.argv) > 1 else "reports/artifacts"
CONF = "/hpcwork/qzi10910/reppo_runs/outputs/conf"
EXPORTS = "exports"
KL_BOUND = 0.1          # config/reppo.yaml:60 and every experiment_override
os.makedirs(OUT, exist_ok=True)

TASKMAP = {"g1": ("G1JoystickFlatTerrain", 29), "leap": ("LeapCubeRotateZAxis", 16),
           "hopper": ("HopperHop", 4), "walker": ("WalkerRun", 6)}


def longest_true_run(mask):
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


rows = []
for d in sorted(glob.glob(f"{CONF}/*")):
    run = os.path.basename(d)
    npz = f"{d}/metrics.npz"
    if not os.path.exists(npz):
        continue
    task_key, arm, seedtok = run.split("_")
    env_name, dim = TASKMAP[task_key]
    seed = int(seedtok[1:])
    z = np.load(npz)

    def g(k, idx=None):
        if k not in z.files:
            return None
        a = np.asarray(z[k], dtype=np.float64)
        return a.reshape(a.shape[0], -1)[:, 0] if idx is None else a

    kl = g("train/kl")
    lag = g("train/lagrangian")
    n = kl.size
    above = kl >= KL_BOUND
    over = np.maximum(kl - KL_BOUND, 0.0)
    late = slice(n - max(1, n // 3), n)

    # meta.json (exported checkpoint) for the arm/return provenance
    tag = f"{env_name}_{'pathwise' if arm=='A' else 'weighted_mle'}_fa_s{seed}_final"
    alt = f"{env_name}_{'pathwise' if arm=='A' else 'weighted_mle'}_s{seed}_final"
    meta = {}
    for cand in (tag, alt):
        p = f"{EXPORTS}/{cand}/meta.json"
        if os.path.exists(p):
            meta = json.load(open(p)); break

    rows.append(dict(
        run=run, task=task_key, env=env_name, d=dim, arm=arm, seed=seed,
        n_points=n,
        # --- committed KL diagnostics -------------------------------------
        F_run=float(above.mean()),
        overshoot_run=float(over.mean()),
        kl_q25=float(np.quantile(kl, .25)), kl_med=float(np.median(kl)),
        kl_q75=float(np.quantile(kl, .75)),
        kl_iqr=float(np.quantile(kl, .75) - np.quantile(kl, .25)),
        kl_q90=float(np.quantile(kl, .90)), kl_q95=float(np.quantile(kl, .95)),
        kl_max=float(kl.max()), kl_min=float(kl.min()),
        area_above=float(over.sum()),
        longest_above=int(longest_true_run(above)),
        first_cross=int(np.argmax(above)) if above.any() else -1,
        late_frac_above=float(above[late].mean()),
        # --- proven lower bound on gate activity --------------------------
        frac_points_gate_proven=float(above.mean()),
        # --- KL multiplier (recoverable trajectory) ------------------------
        lag_first=float(lag[0]), lag_med=float(np.median(lag)),
        lag_last=float(lag[-1]), lag_max=float(lag.max()),
        # --- associates ----------------------------------------------------
        final_return=float(g("eval/episode_return")[-1]),
        sigma_mean_last=float(g("train/pi_sigma_mean")[-1]),
        sigma_max_last=float(g("train/pi_sigma_max")[-1]),
        entropy_last=float(np.asarray(z["train/entropy"]).reshape(n, -1).mean(1)[-1]),
        abs_pred_action_last=float(g("train/abs_pred_action")[-1]),
        value_loss_last=float(g("train/value_loss")[-1]),
        q_last=float(g("train/q")[-1]),
        ess_last=float(g("train/ess")[-1]),
        eta_last=float(g("train/eta")[-1]),
        temp_last=float(np.asarray(z["train/temp"]).ravel()[-1]),
        # --- gated-computation tripwires ----------------------------------
        est_all_zero=bool(np.all(np.abs(g("train/est_h_norm")) == 0)),
        gradnorm_all_zero=bool(np.all(np.abs(g("train/grad_norm_actor")) == 0)),
        iqm_all_zero=bool(np.all(np.abs(g("eval/episode_return_iqm")) == 0)),
        meta_alpha_kl=meta.get("alpha_kl"), meta_action_pad=meta.get("action_pad"),
    ))
    np.save(f"{OUT}/kl_curve_{run}.npy", kl)
    np.save(f"{OUT}/lag_curve_{run}.npy", lag)

df = pd.DataFrame(rows).sort_values(["d", "task", "arm", "seed"])
df.to_csv(f"{OUT}/kl_per_run.csv", index=False)

agg = df.groupby(["task", "d", "arm"]).agg(
    n=("seed", "size"),
    F_run_med=("F_run", "median"), F_run_min=("F_run", "min"), F_run_max=("F_run", "max"),
    overshoot_med=("overshoot_run", "median"),
    kl_med_med=("kl_med", "median"), kl_iqr_med=("kl_iqr", "median"),
    kl_q75_med=("kl_q75", "median"), kl_max_max=("kl_max", "max"),
    longest_above_med=("longest_above", "median"),
    late_frac_med=("late_frac_above", "median"),
    lag_med_med=("lag_med", "median"), lag_max_max=("lag_max", "max"),
    ret_med=("final_return", "median"), sigma_med=("sigma_mean_last", "median"),
).reset_index().sort_values(["d", "arm"])
agg.to_csv(f"{OUT}/kl_by_task_arm.csv", index=False)

print("=== per (task,arm) ===")
print(agg.to_string(index=False))
print()
print("=== gated-computation tripwires (must all be True: nothing was logged) ===")
print(df[["est_all_zero", "gradnorm_all_zero", "iqm_all_zero"]].all().to_string())
print()
print("=== g1 per-seed KL dispersion (the reported anomaly, recomputed from raw) ===")
print(df[df.task == "g1"][["arm", "seed", "kl_med", "kl_iqr", "kl_q75", "kl_max",
                           "F_run", "lag_med", "final_return"]].to_string(index=False))
print()
print("=== paired seed-level associations (n=8 per arm, DESCRIPTIVE) ===")
for t in ["g1", "leap", "hopper", "walker"]:
    s = df[df.task == t]
    for col in ["kl_iqr", "F_run", "lag_med", "sigma_mean_last"]:
        a = s[s.arm == "A"].sort_values("seed")[col].values
        b = s[s.arm == "B"].sort_values("seed")[col].values
        print(f"{t:7s} {col:16s} A_med={np.median(a):.5g}  B_med={np.median(b):.5g}  "
              f"B/A={np.median(b)/np.median(a) if np.median(a) else float('nan'):.4g}")
    r = s[["kl_iqr", "F_run", "lag_med", "final_return", "sigma_mean_last"]].corr(method="spearman")
    print(f"        spearman(kl_iqr, return) = {r.loc['kl_iqr','final_return']:+.3f}   "
          f"spearman(F_run, return) = {r.loc['F_run','final_return']:+.3f}   (n=16)")
print()
print("=== whole-dataset: is the trust region binding everywhere? ===")
print(f"runs with F_run > 0      : {(df.F_run > 0).sum()} / {len(df)}")
print(f"runs with F_run >= 0.5   : {(df.F_run >= 0.5).sum()} / {len(df)}")
print(f"median F_run over 64 runs: {df.F_run.median():.4f}")
print(f"median kl_med over 64    : {df.kl_med.median():.5f}   (bound {KL_BOUND})")
