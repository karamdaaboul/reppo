"""Aggregate Step 3 (P6): trainer-faithful batch action-space and gradient results."""
import glob, os
import numpy as np, pandas as pd

OUT = "reports/artifacts"
EPS = 1e-12
TASK = {"HopperHop": "hopper", "WalkerRun": "walker",
        "LeapCubeRotateZAxis": "leap", "G1JoystickFlatTerrain": "g1"}

act, grad = [], []
for f in sorted(glob.glob(f"{OUT}/ubar_batch/*.npz")):
    z = np.load(f, allow_pickle=True)
    act += list(z["action"]); grad += list(z["grad"])
A = pd.DataFrame(act); G = pd.DataFrame(grad)
for df in (A, G):
    df["task"] = df.env.map(TASK)
    df["arm"] = np.where(df["mode"] == "pathwise", "A", "B")
    df["pad"] = np.where(df.tag.str.contains("pad16"), 16, 0)
    df["condition"] = df.task + np.where(df.pad == 16, "-pad16", "")
A.to_csv(f"{OUT}/ubar_batch_action.csv", index=False)
G.to_csv(f"{OUT}/ubar_batch_gradient.csv", index=False)
print("minibatches: %d   gradient triples: %d   checkpoints: %d"
      % (len(A), len(G), A.tag.nunique()))
print("minibatches per task-arm condition (prereg target >= 512):")
print(A.groupby(["condition", "arm"]).size().to_string())

print("\n=== P6a action-space batch decomposition ===")
a = A.groupby(["condition", "arm", "d"]).agg(
    n_mb=("R_batch_action", "size"), B=("B", "first"),
    R_batch_action=("R_batch_action", "median"),
    n_mean_ubar=("n_mean_ubar", "median"), n_mean_c=("n_mean_c", "median"),
    n_mean_ubar_raw=("n_mean_ubar_raw", "median"),
    clip=("clip_rate", "median")).reset_index()
a["sqrt_d_over_MB"] = np.sqrt(a.d / (32 * a.B))
a["raw_vs_indep"] = a.n_mean_ubar_raw / a.sqrt_d_over_MB
print(a.to_string(index=False, float_format=lambda x: "%.5g" % x))
print("\n  'raw_vs_indep' = measured ||mean_batch(ubar_raw)|| / sqrt(d/(M*B)).")
print("  ~1 confirms that WITHIN a minibatch the draws are independent across states")
print("  (the sharing established in Step 0.4 is ACROSS minibatches, not within one).")

print("\n=== P6b actor-gradient decomposition, seed as the unit ===")
print("  resid_* must be ~0: it is the numerical check of g_full = g_uniform + g_centered")
gg = G.groupby(["condition", "arm", "d"]).agg(
    n=("R_theta_full", "size"),
    resid_full=("resid_full", "max"), resid_meanout=("resid_meanout", "max"),
    R_full=("R_theta_full", "median"), R_meanhead=("R_theta_meanhead", "median"),
    R_scalehead=("R_theta_scalehead", "median"), R_trunk=("R_theta_trunk", "median"),
    R_meanout=("R_meanout", "median"), R_kl=("R_kl", "median"),
    cos_fc=("cos_full_centered_full", "median"),
    cos_uc=("cos_uniform_centered_full", "median"),
    cos_mo=("cos_meanout_full_centered", "median")).reset_index()
print(gg.to_string(index=False, float_format=lambda x: "%.5g" % x))

def verdict(r):
    if r > 3: return "DOMINATES"
    if r < 1: return "centered component larger"
    return "MATERIAL BUT NON-DOMINANT"

print("\n=== P6 preregistered materiality verdicts ===")
print("  metric of record: mean-output and empirical-KL (invariant); Euclidean is")
print("  parameterization dependent and reported alongside.")
for _, r in gg.sort_values("d").iterrows():
    print("  %-14s arm %s d=%-3d  R_meanout=%.4g -> %-26s | R_KL=%.4g | R_euclid=%.4g"
          % (r.condition, r.arm, r.d, r.R_meanout, verdict(r.R_meanout), r.R_kl, r.R_full))
hi = gg[(gg.d == gg.d.max())]
print("\n  highest-d condition (d=%d):" % gg.d.max())
for _, r in hi.iterrows():
    print("    arm %s  R_meanout=%.4g  R_KL=%.4g  -> %s"
          % (r.arm, r.R_meanout, r.R_kl, verdict(r.R_meanout)))
dom = bool((hi.R_meanout > 3).any() or (hi.R_kl > 3).any())
print("\n  DOMINANCE HEADLINE REQUIRED: %s" % ("YES" if dom else "NO"))
G[["kl_full", "kl_full_fd_check"]].head(0)
if "kl_full_fd_check" in G:
    rel = np.abs(G.kl_full_fd_check / np.maximum(G.kl_full, EPS) - 1)
    print("  KL-metric cross-check vs repo decoupled_kls: median |ratio-1| = %.4g" % rel.median())
