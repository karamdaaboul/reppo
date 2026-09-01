import pandas as pd, numpy as np
df = pd.read_csv("reports/artifacts/planted_sweep.csv")
print("cells:", len(df))
print("\n=== omega calibration: measured vs nominal ===")
print("max |meas/nom - 1| = %.3e" % np.abs(df.om_meas / df.omega - 1).max())
rr = df.ratio_e * df.r ** 2
print("\n=== ratio_e * r^2 (analytic prediction: constant) ===")
print("median=%.4f  q05=%.4f  q95=%.4f" % (rr.median(), rr.quantile(.05), rr.quantile(.95)))
print("\n=== NONCOLLAPSED by d ===")
for d, g in df.groupby("d"):
    sl, ic = np.polyfit(np.log(g.r), np.log(g.ratio_e), 1)
    mis = int((g[g.r < 1].ratio_e < 1).sum() + (g[g.r > 1].ratio_e > 1).sum())
    print("  d=%3d n=%3d slope=%+.3f  crossover r*=%.4f  misclassified=%d"
          % (d, len(g), sl, np.exp(-ic / sl), mis))
print("\n=== NONCOLLAPSED by sigma / omega ===")
for key in ("sigma", "omega"):
    for v, g in df.groupby(key):
        sl, ic = np.polyfit(np.log(g.r), np.log(g.ratio_e), 1)
        mis = int((g[g.r < 1].ratio_e < 1).sum() + (g[g.r > 1].ratio_e > 1).sum())
        print("  %-6s=%-6s n=%3d crossover r*=%.4f  mis=%d" % (key, v, len(g), np.exp(-ic / sl), mis))
df["rbin"] = pd.cut(np.log10(df.r), bins=np.arange(-2.5, 2.6, 0.5))
print("\n=== OPERATIONAL: trust-region update MSE vs exact oracle ===")
print(df.groupby("rbin", observed=True).agg(
    n=("r", "size"), r_med=("r", "median"),
    mse_zo_pw=("mse_ratio_zo_pw", "median"), mse_wml_pw=("mse_ratio_wml_pw", "median"),
    win_zo=("win_zo_vs_pw", "median"), win_wml=("win_wml_vs_pw", "median"),
    cos_pw=("cos_pw", "median"), cos_zo=("cos_zo", "median"),
    cos_wml=("cos_wml", "median")).to_string())
print("\n=== actual WML vs manuscript g_ZO (first-order equivalence) ===")
print("  spearman(MSE_WML/PW, MSE_ZO/PW) = %+.4f"
      % df.mse_ratio_wml_pw.corr(df.mse_ratio_zo_pw, method="spearman"))
print("  median |log(mse_wml/mse_zo)|    = %.4f"
      % np.abs(np.log(df.mse_ratio_wml_pw / df.mse_ratio_zo_pw)).median())
lo, hi = df[df.r < 1], df[df.r > 1]
print("\n  r<1: median MSE ratio ZO/PW=%.4g  WML/PW=%.4g" % (lo.mse_ratio_zo_pw.median(), lo.mse_ratio_wml_pw.median()))
print("  r>1: median MSE ratio ZO/PW=%.4g  WML/PW=%.4g" % (hi.mse_ratio_zo_pw.median(), hi.mse_ratio_wml_pw.median()))
print("\n  ZO  beats PW on MSE: %d/%d cells" % ((df.mse_ratio_zo_pw < 1).sum(), len(df)))
print("  WML beats PW on MSE: %d/%d cells" % ((df.mse_ratio_wml_pw < 1).sum(), len(df)))
print("  r>1 subset, WML beats PW: %d/%d" % ((hi.mse_ratio_wml_pw < 1).sum(), len(hi)))
print("  r<1 subset, WML beats PW: %d/%d" % ((lo.mse_ratio_wml_pw < 1).sum(), len(lo)))
print("\n=== where DOES the operational MSE cross over? ===")
for nm, col in (("ZO", "mse_ratio_zo_pw"), ("WML", "mse_ratio_wml_pw")):
    sub = df[(df[col] > 0)]
    sl, ic = np.polyfit(np.log(sub.r), np.log(sub[col]), 1)
    print("  %-3s slope=%+.3f  implied crossover r*=%.4f  (vs r*=1 for the error-induced variance)"
          % (nm, sl, np.exp(-ic / sl)))
