#!/usr/bin/env python
"""Does the panel-B collapse follow from the RMS derivation?

The derivation behind reports/planted_multimode.md Sec. 8:

    Var_e[PW] ~ (1/M) E||grad e||^2        Var_e[ZO] ~ (d/M) E[e^2] / sigma^2

so with  omega_RMS^2 := E||grad e||^2 / E[e^2]  and  r_RMS := sigma omega_RMS / sqrt(d),

    (Var_e[ZO] / Var_e[PW]) * r_RMS^2  ~  1        for every J, r, d.

This script computes E||grad e||^2 and E[e^2] from EACH SWEEP'S OWN SAMPLES by
replaying its RNG stream (same seed, same draw order), joins them to the stored
per-block variances, and reports the product.  No estimator is re-run.  The replay
is verified exactly: the recomputed Var_e[PW] must match the stored column
bit-for-bit, or nothing else here is trusted.

    ./.venv/bin/python scripts/planted/check_rms_collapse.py
"""
import os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import multimode_sweep as MM
import planted_error_sweep as P
from decompose_sweep import tr_cov

ART = "reports/artifacts"
RUNS = {"old": ("planted_multimode",      20260904, [1, 4], [4, 16, 64]),
        "rep": ("planted_j4_replication", 20260905, [1, 4], [4, 16, 64]),
        "j8":  ("planted_j8",             20260906, [1, 8], [16, 64])}


def replay(seed, js, ds):
    """Re-draw exactly what multimode_sweep.main drew, and return per-block
    realised field moments over the block's own R x M action samples."""
    rng = np.random.default_rng(seed)
    jmax = max(js)
    rows = []
    for d in ds:
        for r_eff in MM.R_GRID:
            om_of = {J: r_eff * np.sqrt(J * d) / MM.SIGMA for J in js}
            for k in range(MM.N_DIR):
                v_sig = rng.normal(size=d); v_sig /= np.linalg.norm(v_sig)
                V = np.linalg.qr(rng.normal(size=(d, jmax)))[0].T
                ph = rng.uniform(0, 2 * np.pi, size=jmax)
                u = rng.normal(size=(MM.N_BATCH, MM.M, d))
                a = MM.SIGMA * u
                for J in js:
                    c = np.full(J, MM.AMP / J)
                    om = om_of[J]
                    e = MM.e_mm(a, V[:J], ph[:J], c, om)          # (R, M)
                    ge = MM.grad_e_mm(a, V[:J], ph[:J], c, om)    # (R, M, d)
                    rows.append(dict(
                        d=d, r_eff=r_eff, J=J, direction=k,
                        e2=float(np.mean(e ** 2)),
                        g2=float(np.mean(np.sum(ge ** 2, -1))),
                        mean_e2=float(np.sum(e.mean()) ** 2),
                        mean_g2=float(np.sum(ge.mean((0, 1)) ** 2)),
                        var_pw_e_replay=tr_cov(ge.mean(1)),       # exact check
                    ))
            print(f"    replayed d={d} r_eff={r_eff}", flush=True)
    return pd.DataFrame(rows)


def main():
    out = []
    for tag, (pre, seed, js, ds) in RUNS.items():
        print(f"\n=== {tag}: {pre}  seed {seed}  J {js}  d {ds} ===")
        rep = pd.read_csv(f"{ART}/{pre}_replicates.csv")
        R = replay(seed, js, ds)
        m = rep.merge(R, on=["d", "r_eff", "J", "direction"])
        assert len(m) == len(rep) == len(R), (len(m), len(rep), len(R))
        dev = float(np.abs(m.var_pw_e_replay / m.var_pw_e - 1).max())
        print(f"  replay check: max |Var_e[PW] replayed / stored - 1| = {dev:.3e}  "
              f"{'EXACT' if dev < 1e-12 else 'MISMATCH -- stop'}")
        assert dev < 1e-12
        m["run"] = tag
        out.append(m)
    D = pd.concat(out, ignore_index=True)

    sig, M = MM.SIGMA, MM.M
    # per-block quantities
    D["omega_rms2"] = D.g2 / D.e2
    D["r_rms2"] = sig ** 2 * D.omega_rms2 / D.d
    D["ratio"] = D.var_zo_e / D.var_pw_e
    D["product_rms"] = D.ratio * D.r_rms2
    D["product_nom"] = D.ratio * (D.r_eff * np.sqrt(D.J)) ** 2     # analytic RMS = nominal
    D["product_inf"] = D.ratio * D.r_eff ** 2                       # sup-norm convention
    # the two halves of the derivation, separately
    D["pw_over_pred"] = D.var_pw_e * M / D.g2
    D["zo_over_pred"] = D.var_zo_e * M * sig ** 2 / (D.d * D.e2)

    # ---- cell level: ratio of block-mean variances, moments pooled over blocks
    C = (D.groupby(["run", "J", "d", "r_eff"])
           .agg(var_zo=("var_zo_e", "mean"), var_pw=("var_pw_e", "mean"),
                e2=("e2", "mean"), g2=("g2", "mean"),
                mean_g2=("mean_g2", "mean"), mean_e2=("mean_e2", "mean"))
           .reset_index())
    C["r_rms2"] = sig ** 2 * (C.g2 / C.e2) / C.d
    C["ratio"] = C.var_zo / C.var_pw
    C["product_rms"] = C.ratio * C.r_rms2
    C["product_nom"] = C.ratio * (C.r_eff * np.sqrt(C.J)) ** 2
    C["product_inf"] = C.ratio * C.r_eff ** 2
    C["pw_over_pred"] = C.var_pw * M / C.g2
    C["zo_over_pred"] = C.var_zo * M * sig ** 2 / (C.d * C.e2)
    C["r_rms_over_r_nom"] = np.sqrt(C.r_rms2) / (C.r_eff * np.sqrt(C.J))
    C.to_csv(f"{ART}/planted_rms_collapse.csv", index=False)

    pd.set_option("display.width", 160)
    print("\n" + "=" * 96)
    print("(Var_e[ZO]/Var_e[PW]) x r_RMS^2, r_RMS from the sweep's OWN E||grad e||^2, E[e^2]")
    print("=" * 96)
    print(f"  {'run':>4} {'J':>2} | {'min':>7} {'median':>7} {'max':>7} | "
          f"{'x r_nom^2':>9} {'x r_inf^2':>9} | {'r_RMS/r_nom':>11}")
    for (run, J), g in C.groupby(["run", "J"]):
        print(f"  {run:>4} {J:>2} | {g.product_rms.min():>7.4f} "
              f"{g.product_rms.median():>7.4f} {g.product_rms.max():>7.4f} | "
              f"{g.product_nom.median():>9.4f} {g.product_inf.median():>9.4f} | "
              f"{g.r_rms_over_r_nom.median():>11.4f}")
    print(f"\n  ALL cells: product_rms in [{C.product_rms.min():.4f}, "
          f"{C.product_rms.max():.4f}], median {C.product_rms.median():.4f}, "
          f"n = {len(C)}")

    print("\n  by d (median over cells):")
    print(C.pivot_table(index="d", columns="J", values="product_rms",
                        aggfunc="median").round(4).to_string())
    print("\n  by r_eff (median over cells):")
    print(C.pivot_table(index="r_eff", columns="J", values="product_rms",
                        aggfunc="median").round(4).to_string())

    print("\n" + "=" * 96)
    print("The two halves of the derivation, each against its own prediction")
    print("=" * 96)
    print("  Var_e[PW] * M / E||grad e||^2      (1 if PW ~ (1/M) E||grad e||^2)")
    print(C.pivot_table(index="d", columns="J", values="pw_over_pred",
                        aggfunc="median").round(4).to_string())
    print("\n  Var_e[ZO] * M sigma^2 / (d E[e^2]) (1 if ZO ~ (d/M) E[e^2]/sigma^2)")
    print(C.pivot_table(index="d", columns="J", values="zo_over_pred",
                        aggfunc="median").round(4).to_string())
    print(f"\n  (M-1)/M = {(M-1)/M:.4f}   (M+1)/M = {(M+1)/M:.4f}   for M = {M}")
    print("\n  blurred-mean share of each moment (why r_RMS != r_nom at small omega*sigma):")
    C["mean_g_share"] = C.mean_g2 / C.g2
    C["mean_e_share"] = C.mean_e2 / C.e2
    print("    ||E grad e||^2 / E||grad e||^2, max over cells: "
          f"{C.mean_g_share.max():.4f}     (E e)^2 / E e^2, max: {C.mean_e_share.max():.4f}")
    print(f"\nwrote {ART}/planted_rms_collapse.csv  ({len(C)} cells)")


if __name__ == "__main__":
    main()
