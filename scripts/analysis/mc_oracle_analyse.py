"""Analysis of the MC Q^pi oracle WalkerRun feasibility pilot.

Implements docs/prereg_mc_oracle_walker_pilot.md sections 8, 9, 11 and 12 exactly:
noise-debiased cross-product estimators, the K/(K-1) finite-K centering correction,
the paired stratified hierarchical bootstrap, and the A-E feasibility thresholds.

Nothing here is chosen after the fact. The step sizes, horizons, subset, bootstrap
scheme and thresholds are all read from the preregistration constants.

Usage:
  mc_oracle_analyse.py <pw_pilot.npz> <wml_pilot.npz> <pw_horizon.npz> <wml_horizon.npz> <outdir>
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

C_STEPS = (0.10, 0.05)
N_BOOT = 10000
BOOT_SEED = 20260902
CLIP = 0.999
THRESH_STAB = (0.8, 1.25)     # prereg 11 criterion C and D
THRESH_TAIL = 0.25            # prereg 11 criterion D
THRESH_WIDTH = 0.60           # prereg 11 criterion E


# ------------------------------------------------------------------ load / shape
def load_run(path):
    z = np.load(path, allow_pickle=True)
    d = {k: z[k] for k in z.files}
    d["source"] = np.array([str(s) for s in d["source"]])
    d["branch_kind"] = np.array([str(s) for s in d["branch_kind"]])
    return d


def group_means(q):
    """(2, R, S, K, Br) rollouts -> per-group means and standard errors."""
    m = q.mean(1)                                   # (2, S, K, Br)
    se = q.std(1, ddof=1) / np.sqrt(q.shape[1])
    return m, se


def fd_index(r, c):
    """Branch indices (plus, minus) per coordinate for step size c."""
    kind, bc, bj, bs = r["branch_kind"], r["branch_c"], r["branch_j"], r["branch_sign"]
    d = int(r["mu"].shape[1])
    plus, minus = [], []
    for j in range(d):
        p = np.where((kind == "fd") & np.isclose(bc, c) & (bj == j) & (bs == 1))[0]
        m = np.where((kind == "fd") & np.isclose(bc, c) & (bj == j) & (bs == -1))[0]
        assert len(p) == 1 and len(m) == 1, (c, j, len(p), len(m))
        plus.append(int(p[0])); minus.append(int(m[0]))
    return np.array(plus), np.array(minus)


def errors(r, q_key="q_oracle"):
    """e_A, e_B of shape (S, K, Br), from the two independent rollout groups."""
    m, se = group_means(r[q_key])
    e = r["q_phi"][None] - m                        # (2, S, K, Br)
    return e[0], e[1], se


# ------------------------------------------------------------------- estimators
def stat_D(eA, eB):
    """Noise-debiased centered error power at the base branch (prereg 8)."""
    a, b = eA[..., 0], eB[..., 0]                   # (S, K)
    K = a.shape[1]
    ca = a - a.mean(1, keepdims=True)
    cb = b - b.mean(1, keepdims=True)
    return K / (K - 1) * float((ca * cb).mean())


def stat_N(eA, eB, plus, minus, c, sigma=None, whiten=True):
    """Noise-debiased gradient energy (prereg 7-8).

    ``whiten=True`` gives sum_j z_j^2 with z_j = sigma_j d/dy_j e (the r_RMS
    numerator); ``whiten=False`` divides by sigma^2 to give ||grad_y e||^2, used only
    for the secondary omega_RMS diagnostic.
    """
    zA = (eA[..., plus] - eA[..., minus]) / (2 * c)   # (S, K, d)
    zB = (eB[..., plus] - eB[..., minus]) / (2 * c)
    if not whiten:
        s2 = sigma[:, None, :] ** 2
        return float((zA * zB / s2).sum(-1).mean())
    return float((zA * zB).sum(-1).mean())


def r_rms(N, D, d):
    if not (N > 0 and D > 0):
        return np.nan
    return float(np.sqrt(N / (d * D)))


def all_stats(r, sel_states=None, sel_k=None):
    """D, N_c and r_RMS_c for one run, optionally on a subset of states/perturbations."""
    eA, eB, _ = errors(r)
    if sel_states is not None:
        eA, eB = eA[sel_states], eB[sel_states]
    if sel_k is not None:
        eA, eB = eA[:, sel_k], eB[:, sel_k]
    d = int(r["mu"].shape[1])
    out = {"D": stat_D(eA, eB)}
    for c in C_STEPS:
        p, m = fd_index(r, c)
        out["N_%.2f" % c] = stat_N(eA, eB, p, m, c)
        out["r_%.2f" % c] = r_rms(out["N_%.2f" % c], out["D"], d)
    return out


# -------------------------------------------------------------------- bootstrap
def boot_indices(source, K, n_boot, seed):
    """Stratified hierarchical resample: states within stratum, then perturbations.

    Returned index arrays are shared by both checkpoints, which is what makes the
    PW - WML contrast paired (prereg 9).
    """
    rng = np.random.default_rng(seed)
    pw = np.where(source == "PW")[0]
    wml = np.where(source == "WML")[0]
    S = len(source)
    sidx = np.empty((n_boot, S), dtype=np.int64)
    sidx[:, :len(pw)] = rng.choice(pw, size=(n_boot, len(pw)), replace=True)
    sidx[:, len(pw):] = rng.choice(wml, size=(n_boot, len(wml)), replace=True)
    kidx = rng.integers(0, K, size=(n_boot, S, K))
    return sidx, kidx


def boot_stats(r, sidx, kidx, states=None):
    """Bootstrap distribution of D, N_c, r_c. ``states`` restricts to one stratum."""
    eA, eB, _ = errors(r)
    d = int(r["mu"].shape[1])
    S, K = eA.shape[0], eA.shape[1]
    keep = np.ones(S, bool) if states is None else np.isin(np.arange(S), states)
    fdp = {c: fd_index(r, c) for c in C_STEPS}
    out = {k: np.empty(len(sidx)) for k in
           ["D"] + ["N_%.2f" % c for c in C_STEPS] + ["r_%.2f" % c for c in C_STEPS]}
    for b in range(len(sidx)):
        si = sidx[b][keep[sidx[b]]]
        if len(si) == 0:
            for k in out:
                out[k][b] = np.nan
            continue
        ki = kidx[b][:len(si)]
        aa = np.take_along_axis(eA[si], ki[..., None], axis=1)
        bb = np.take_along_axis(eB[si], ki[..., None], axis=1)
        D = stat_D(aa, bb)
        out["D"][b] = D
        for c in C_STEPS:
            p, m = fdp[c]
            N = stat_N(aa, bb, p, m, c)
            out["N_%.2f" % c][b] = N
            out["r_%.2f" % c][b] = r_rms(N, D, d)
    return out


def boot_states_only(r, source, n_boot, seed, d):
    """Robustness check: resample states only, perturbations kept attached."""
    rng = np.random.default_rng(seed + 1)
    eA, eB, _ = errors(r)
    pw = np.where(source == "PW")[0]; wml = np.where(source == "WML")[0]
    fdp = {c: fd_index(r, c) for c in C_STEPS}
    out = {"r_0.10": np.empty(n_boot), "D": np.empty(n_boot)}
    for b in range(n_boot):
        si = np.concatenate([rng.choice(pw, len(pw), True), rng.choice(wml, len(wml), True)])
        D = stat_D(eA[si], eB[si])
        p, m = fdp[0.10]
        out["D"][b] = D
        out["r_0.10"][b] = r_rms(stat_N(eA[si], eB[si], p, m, 0.10), D, d)
    return out


def ci(x, lo=2.5, hi=97.5):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (np.nan, np.nan)
    return tuple(float(v) for v in np.percentile(x, [lo, hi]))


# ------------------------------------------------------------- horizon criterion
def horizon_check(r, d):
    """Criterion D on the fixed H=1000 subset (prereg 6 and 11)."""
    q1000 = r["q_oracle"]            # (2, R, S, K, Br)
    q500 = r["q_oracle_prefix"]
    tail = q1000 - q500              # same trajectories, so the MC noise largely cancels

    tA, tB = tail[0].mean(0)[..., 0], tail[1].mean(0)[..., 0]     # (S, K) base branch
    K = tA.shape[1]
    ca = tA - tA.mean(1, keepdims=True)
    cb = tB - tB.mean(1, keepdims=True)
    tail_deb = K / (K - 1) * float((ca * cb).mean())
    tail_naive = K / (K - 1) * float((0.5 * (ca ** 2 + cb ** 2)).mean())

    eA5 = r["q_phi"][None] - q500.mean(1)
    eA1 = r["q_phi"][None] - q1000.mean(1)
    D5 = stat_D(eA5[0], eA5[1])
    D1 = stat_D(eA1[0], eA1[1])
    p, m = fd_index(r, 0.10)
    N5 = stat_N(eA5[0], eA5[1], p, m, 0.10)
    N1 = stat_N(eA1[0], eA1[1], p, m, 0.10)
    return {
        "tail_rms_debiased": float(np.sqrt(max(tail_deb, 0.0))),
        "tail_rms_naive": float(np.sqrt(max(tail_naive, 0.0))),
        "tail_cross_raw": tail_deb,
        "e500_rms": float(np.sqrt(D5)) if D5 > 0 else np.nan,
        "D_500": D5, "D_1000": D1,
        "N_500": N5, "N_1000": N1,
        "N_ratio_1000_over_500": float(N1 / N5) if N5 != 0 else np.nan,
        "r_500": r_rms(N5, D5, d), "r_1000": r_rms(N1, D1, d),
        "mean_tail": float(tail[:, :, :, :, 0].mean()),
        "n_done_max": float(r["n_done"].max()),
    }


# ------------------------------------------------------------------- diagnostics
def diagnostics(r):
    eA, eB, se = errors(r)
    K = eA.shape[1]
    d = int(r["mu"].shape[1])
    sig = r["sigma"]
    act = r["action"]                                   # (S, K, Br, d)
    clipped_pt = np.abs(act) > CLIP                     # (S, K, Br, d)
    base_clip = clipped_pt[:, :, 0, :].any(-1)          # (S, K)
    any_clip = clipped_pt.any(-1).any(-1)               # (S, K) over all branches
    unc = float((eA[..., 0] * eB[..., 0]).mean())       # uncentered, debiased
    p, m = fd_index(r, 0.10)
    om2_num = stat_N(eA, eB, p, m, 0.10, sigma=sig, whiten=False)
    D = stat_D(eA, eB)
    return {
        "uncentered_e_rms": float(np.sqrt(unc)) if unc > 0 else np.nan,
        "centered_e_rms": float(np.sqrt(D)) if D > 0 else np.nan,
        "statewise_mean_e": (0.5 * (eA[..., 0] + eB[..., 0])).mean(1),   # (S,)
        "oracle_se_median": float(np.median(se[:, :, :, 0])),
        "oracle_se_mean": float(se[:, :, :, 0].mean()),
        "oracle_se_max": float(se[:, :, :, 0].max()),
        "sigma_mean_per_dim": sig.mean(0).tolist(),
        "sigma_rms": float(np.sqrt((sig ** 2).mean())),
        "sigma_min": float(sig.min()), "sigma_max": float(sig.max()),
        "sigma_anisotropy": float(sig.max(1).mean() / sig.min(1).mean()),
        "omega_rms": float(np.sqrt(om2_num / D)) if (om2_num > 0 and D > 0) else np.nan,
        "clip_rate_base": float(base_clip.mean()),
        "clip_rate_any_branch": float(any_clip.mean()),
        "clip_rate_coord": float(clipped_pt.mean()),
        "q_phi_mean": float(r["q_phi"][..., 0].mean()),
        "q_phi_min": float(r["q_phi"].min()), "q_phi_max": float(r["q_phi"].max()),
        "q_oracle_mean": float(r["q_oracle"][..., 0].mean()),
        "no_clip_mask": ~any_clip,
        "K": K, "d": d,
    }


# ------------------------------------------------- SECONDARY: support variants
VMIN, VMAX = 0.0, 150.0


def support_variants(r):
    """SECONDARY diagnostic: the critic-target support applied to the oracle.

    `src/jaxrl/utils.py:45` clips the regression target into [vmin, vmax] BEFORE the
    two-hot encoding, and the HL-Gauss head cannot represent a value outside that
    interval either. So Q_phi is fit to E[clip(G, 0, 150)], while the PRIMARY estimand
    -- unchanged in both preregistrations -- is the true unclipped soft Q^pi.

    Two variants are reported, and neither replaces the primary:

      clip_per_rollout   mean_r clip(G_r)   -- closest to how training clips each
                         sampled target before encoding it
      project_mean       clip(mean_r G_r)   -- the oracle mean projected onto the
                         representable interval

    Both APPROXIMATE the training target rather than matching it: training clips a
    lambda-return that bootstraps on Q_phi (itself >= 0), whereas these clip a
    500-step unbootstrapped MC return, so E[clip(G_lambda)] != E[clip(G_MC)].

    In pilot 1 this was computed post-hoc, after the WML output revealed the
    mismatch. In pilot 2 it is preregistered as a secondary.
    """
    q = r["q_oracle"]
    d = int(r["mu"].shape[1])
    out = {"frac_rollouts_below_vmin": float((q < VMIN).mean()),
           "frac_rollouts_above_vmax": float((q > VMAX).mean()),
           "frac_points_mean_below_vmin":
               float((q.reshape(-1, *q.shape[2:]).mean(0)[..., 0] < VMIN).mean())}
    for name, qc in (("clip_per_rollout", np.clip(q, VMIN, VMAX)),
                     ("project_mean", None)):
        if qc is None:
            mA = np.clip(q[0].mean(0), VMIN, VMAX)
            mB = np.clip(q[1].mean(0), VMIN, VMAX)
        else:
            mA, mB = qc[0].mean(0), qc[1].mean(0)
        eA, eB = r["q_phi"] - mA, r["q_phi"] - mB
        sub = {"D": stat_D(eA, eB)}
        for c in C_STEPS:
            p_, m_ = fd_index(r, c)
            sub["N_%.2f" % c] = stat_N(eA, eB, p_, m_, c)
            sub["r_%.2f" % c] = r_rms(sub["N_%.2f" % c], sub["D"], d)
        out[name] = sub
    return out


def sigma_error_link(r):
    """How the measured error concentrates in the heavy tail of the policy scale."""
    eA, eB, _ = errors(r)
    K = eA.shape[1]
    a, b = eA[..., 0], eB[..., 0]
    ca = a - a.mean(1, keepdims=True); cb = b - b.mean(1, keepdims=True)
    per = K / (K - 1) * (ca * cb).mean(1)              # per-state centred power
    srms = np.sqrt((r["sigma"] ** 2).mean(1))
    order = np.argsort(srms)
    lo, hi = order[:len(order) // 2], order[len(order) // 2:]
    return {"sigma_rms_median": float(np.median(srms)),
            "sigma_rms_p95": float(np.percentile(srms, 95)),
            "sigma_rms_max": float(srms.max()),
            "D_low_sigma_half": float(per[lo].mean()),
            "D_high_sigma_half": float(per[hi].mean()),
            "spearman_sigma_vs_D": float(np.corrcoef(
                np.argsort(np.argsort(srms)).astype(float),
                np.argsort(np.argsort(per)).astype(float))[0, 1])}


# ------------------------------------------------------------------------- main
def main(pw_p, wml_p, pw_h, wml_h, outdir):
    os.makedirs(outdir, exist_ok=True)
    runs = {"PW": load_run(pw_p), "WML": load_run(wml_p)}
    hor = {"PW": load_run(pw_h), "WML": load_run(wml_h)}
    source = runs["PW"]["source"]
    assert np.array_equal(source, runs["WML"]["source"])
    assert np.allclose(runs["PW"]["u"], runs["WML"]["u"]), "u must be shared"
    S, K = runs["PW"]["q_phi"].shape[0], runs["PW"]["q_phi"].shape[1]
    d = int(runs["PW"]["mu"].shape[1])

    sidx, kidx = boot_indices(source, K, N_BOOT, BOOT_SEED)
    pw_states = np.where(source == "PW")[0]
    wml_states = np.where(source == "WML")[0]

    res = {"n_states": int(S), "K": int(K), "d": d, "n_boot": N_BOOT,
           "boot_seed": BOOT_SEED}
    for tag, r in runs.items():
        pt = all_stats(r)
        bs = boot_stats(r, sidx, kidx)
        dg = diagnostics(r)
        hc = horizon_check(hor[tag], d)
        so = boot_states_only(r, source, 2000, BOOT_SEED, d)
        entry = {"point": pt,
                 "ci": {k: ci(v) for k, v in bs.items()},
                 "diagnostics": {k: v for k, v in dg.items()
                                 if not isinstance(v, np.ndarray)},
                 "horizon": hc,
                 "state_only_boot_r010_ci": ci(so["r_0.10"]),
                 "strata": {}}
        for sname, ss in (("PW_states", pw_states), ("WML_states", wml_states)):
            entry["strata"][sname] = {
                "point": all_stats(r, sel_states=ss),
                "ci": {k: ci(v) for k, v in boot_stats(r, sidx, kidx, states=ss).items()},
            }
        # preregistered no-clip sensitivity
        mask = dg["no_clip_mask"]
        if mask.sum() >= 2:
            eA, eB, _ = errors(r)
            aa, bb = eA.copy(), eB.copy()
            keep_rows = mask.all(1)
            if keep_rows.sum() >= 4:
                entry["no_clip_all_k"] = all_stats(r, sel_states=np.where(keep_rows)[0])
            entry["no_clip_frac_points"] = float(mask.mean())
        entry["support_variants"] = support_variants(r)
        entry["sigma_error_link"] = sigma_error_link(r)
        res[tag] = entry
        res[tag]["_boot"] = bs
        res[tag]["_diag_arrays"] = dg

    # paired PW - WML difference in r_RMS, same bootstrap indices
    for c in C_STEPS:
        k = "r_%.2f" % c
        diff = res["PW"]["_boot"][k] - res["WML"]["_boot"][k]
        res["paired_diff_%s" % k] = {
            "point": res["PW"]["point"][k] - res["WML"]["point"][k],
            "ci": ci(diff),
            "frac_positive": float(np.mean(diff[np.isfinite(diff)] > 0)),
        }

    # ---------------- feasibility criteria A-E, thresholds from the prereg
    verdicts = {}
    for tag in ("PW", "WML"):
        e = res[tag]
        p, C = e["point"], e["ci"]
        A = p["D"] > 0 and C["D"][0] > 0
        B = all(p["N_%.2f" % c] > 0 and C["N_%.2f" % c][0] > 0 for c in C_STEPS)
        rr = p["r_0.05"] / p["r_0.10"] if p["r_0.10"] else np.nan
        nr = p["N_0.05"] / p["N_0.10"] if p["N_0.10"] else np.nan
        Ccrit = (THRESH_STAB[0] <= rr <= THRESH_STAB[1]
                 and THRESH_STAB[0] <= nr <= THRESH_STAB[1])
        h = e["horizon"]
        tail_ok_deb = h["tail_rms_debiased"] < THRESH_TAIL * h["e500_rms"]
        tail_ok_nai = h["tail_rms_naive"] < THRESH_TAIL * h["e500_rms"]
        nrat = h["N_ratio_1000_over_500"]
        D_ok = tail_ok_deb and tail_ok_nai and THRESH_STAB[0] <= nrat <= THRESH_STAB[1]
        w = C["r_0.10"][1] - C["r_0.10"][0]
        E = w <= THRESH_WIDTH * p["r_0.10"]
        lo, hi = C["r_0.10"]
        place = ("BELOW BOUNDARY" if hi < 1 else
                 "ABOVE BOUNDARY" if lo > 1 else
                 "NEAR BOUNDARY - INTERVAL CROSSES 1")
        verdicts[tag] = {
            "A_centered_error_power": bool(A),
            "B_gradient_energy": bool(B),
            "C_step_size_stability": bool(Ccrit),
            "C_r_ratio": rr, "C_N_ratio": nr,
            "D_horizon_stability": bool(D_ok),
            "D_tail_ok_debiased": bool(tail_ok_deb),
            "D_tail_ok_naive": bool(tail_ok_nai),
            "D_N_ratio": nrat,
            "E_interval_precision": bool(E),
            "E_rel_width": float(w / p["r_0.10"]) if p["r_0.10"] else np.nan,
            "placement": place,
            "all_pass": bool(A and B and Ccrit and D_ok and E),
        }
    res["verdicts"] = verdicts
    res["overall"] = ("PASS TO SCALE"
                      if all(v["all_pass"] for v in verdicts.values())
                      else "NOT YET PRECISE ENOUGH TO SCALE")

    # ----------------------------------------------------------------- artifacts
    write_csvs(runs, res, outdir)
    clean = json.loads(json.dumps(
        {k: v for k, v in res.items()},
        default=lambda o: (o.tolist() if isinstance(o, np.ndarray)
                           else (None if isinstance(o, float) and not np.isfinite(o)
                                 else float(o) if isinstance(o, (np.floating,))
                                 else int(o) if isinstance(o, (np.integer,))
                                 else str(o)))))
    for t in ("PW", "WML"):
        clean[t].pop("_boot", None); clean[t].pop("_diag_arrays", None)
    with open(os.path.join(outdir, "mc_oracle_results.json"), "w") as f:
        json.dump(clean, f, indent=1)
    np.savez(os.path.join(outdir, "mc_oracle_boot.npz"),
             **{"%s_%s" % (t, k): v for t in ("PW", "WML")
                for k, v in res[t]["_boot"].items()})
    print(json.dumps({"overall": res["overall"], "verdicts": verdicts}, indent=1,
                     default=str))
    return res


def write_csvs(runs, res, outdir):
    import csv
    # 1. Q^pi summary at the base action points
    with open(os.path.join(outdir, "mc_oracle_qpi_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ckpt", "state", "state_source", "k", "q_phi", "qhat_A", "qhat_B",
                    "qhat_pooled", "se_A", "se_B", "traj_sd_pooled", "n_rollouts",
                    "horizon", "n_done", "base_clipped_coords"])
        for tag, r in runs.items():
            q = r["q_oracle"]; m, se = group_means(q)
            pooled = q.reshape(-1, *q.shape[2:]).mean(0)
            sd = q.reshape(-1, *q.shape[2:]).std(0, ddof=1)
            S, K = r["q_phi"].shape[0], r["q_phi"].shape[1]
            for s in range(S):
                for k in range(K):
                    w.writerow([tag, s, r["source"][s], k,
                                "%.6f" % r["q_phi"][s, k, 0],
                                "%.6f" % m[0, s, k, 0], "%.6f" % m[1, s, k, 0],
                                "%.6f" % pooled[s, k, 0], "%.6f" % se[0, s, k, 0],
                                "%.6f" % se[1, s, k, 0], "%.6f" % sd[s, k, 0],
                                int(q.shape[0] * q.shape[1]), int(r["horizon"]),
                                "%.3f" % r["n_done"][:, :, s, k, 0].mean(),
                                int((np.abs(r["action"][s, k, 0]) > CLIP).sum())])
    # 2. per-state centered error
    with open(os.path.join(outdir, "mc_oracle_error_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ckpt", "state", "state_source", "mean_e", "centered_e_power_AB",
                    "uncentered_e_power_AB", "q_phi_mean", "sigma_rms", "clip_rate_k"])
        for tag, r in runs.items():
            eA, eB, _ = errors(r)
            K = eA.shape[1]
            a, b = eA[..., 0], eB[..., 0]
            ca = a - a.mean(1, keepdims=True); cb = b - b.mean(1, keepdims=True)
            per = K / (K - 1) * (ca * cb).mean(1)
            unc = (a * b).mean(1)
            clip = (np.abs(r["action"][:, :, 0, :]) > CLIP).any(-1).mean(1)
            for s in range(eA.shape[0]):
                w.writerow([tag, s, r["source"][s],
                            "%.6f" % (0.5 * (a[s] + b[s])).mean(),
                            "%.6f" % per[s], "%.6f" % unc[s],
                            "%.4f" % r["q_phi"][s, :, 0].mean(),
                            "%.5f" % np.sqrt((r["sigma"][s] ** 2).mean()),
                            "%.4f" % clip[s]])
    # 3. gradient energy per state, step size and coordinate
    with open(os.path.join(outdir, "mc_oracle_gradient_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ckpt", "c", "state", "state_source", "N_state"] +
                   ["z2_dim%d" % j for j in range(int(runs["PW"]["mu"].shape[1]))])
        for tag, r in runs.items():
            eA, eB, _ = errors(r)
            for c in C_STEPS:
                p, m = fd_index(r, c)
                zA = (eA[..., p] - eA[..., m]) / (2 * c)
                zB = (eB[..., p] - eB[..., m]) / (2 * c)
                per = (zA * zB).mean(1)                    # (S, d)
                for s in range(per.shape[0]):
                    w.writerow([tag, c, s, r["source"][s], "%.6f" % per[s].sum()] +
                               ["%.6f" % v for v in per[s]])


if __name__ == "__main__":
    main(*sys.argv[1:6])
