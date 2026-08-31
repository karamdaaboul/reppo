"""Aggregate Probe 1 outputs into reports/probe1_restricted_z.md.

Ordering follows the plan: the eq-(13)/(14) match table is the execution gate and comes
first; per-state ratios are formed before any aggregation (proposition self-check 1),
with the pooled global ratio reported only as a discrepancy check. L is never computed.
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO_ROOT, "scripts", "probe1_out")
NBOOT = 10000
N_CHUNKS = 8
LAWS = {"ckpt": "checkpoint law", "std": "N(0, I_k)"}
ARM_LABEL = {"pathwise": "A", "weighted_mle": "B"}  # meta actor_update_mode


def fro(X):
    return np.sqrt((X**2).sum((-1, -2)))


def load(path):
    z = np.load(path)
    d = {k: z[k] for k in z.files}
    M, k = int(d["M"]), int(d["k"])
    eye = np.eye(k)

    def VM(a, D, nu, drop=False):
        t = (M - 1) * D + nu[:, None, None] * eye
        if not drop:
            t = t - (M - 2) * np.einsum("sj,sl->sjl", a, a)
        return (M - 1) / M**3 * t

    d["V_M"] = VM(d["a"], d["D"], d["nu"])
    d["V_M_drop"] = VM(d["a"], d["D"], d["nu"], drop=True)
    d["V_M_h0"] = VM(d["a_h0"], d["D_h0"], d["nu_h0"])
    d["V_M_h1"] = VM(d["a_h1"], d["D_h1"], d["nu_h1"])
    return d


def moment_match(d):
    """Per-state / per-component tests of eqs (13) and (14)."""
    M = int(d["M"])
    R, Nor = float(d["R"]), float(d["n_oracle"])
    a, D = d["a"], d["D"]
    c = 1.0 - 1.0 / M
    var_batch = np.einsum("sjj->sj", d["cov_a"]) / R
    var_orac = (np.einsum("sjj->sj", D) - a**2) / Nor
    se = np.sqrt(np.maximum(var_batch + c**2 * var_orac, 1e-300))
    nrm = fro(d["V_M"])
    rho = fro(d["cov_a"] - d["V_M"]) / nrm
    null_b = fro(d["half_cov_a0"] - d["half_cov_a1"]) / (2 * nrm)
    null_o = fro(d["V_M_h0"] - d["V_M_h1"]) / (2 * nrm)
    null = np.sqrt(null_b**2 + null_o**2)
    # Stein's identity, eq (19): a = h. The two sides come from independent
    # computations (ZO moments of the oracle stream vs. autodiff gradients), so this
    # is a check on the whole coordinate/estimator correspondence, not an identity the
    # code could satisfy by construction. Floor: the oracle-stream split-half spread
    # of a, |a_h0 - a_h1|/2, which is the MC scale of a itself.
    nh = np.linalg.norm(d["h"], axis=1)
    stein = np.linalg.norm(a - d["h"], axis=1) / np.maximum(nh, 1e-300)
    stein_floor = (np.linalg.norm(d["a_h0"] - d["a_h1"], axis=1) / 2) / np.maximum(nh, 1e-300)
    return dict(
        stein=stein, stein_floor=stein_floor,
        neg_nu=int((d["nu"] <= 0).sum()),
        z_ok=(d["mean_a"] - c * a) / se,
        z_no=(d["mean_a"] - a) / se,
        rho=rho, null=null, null_b=null_b, null_o=null_o,
        ratio=rho / null,
        ratio_drop=(fro(d["cov_a"] - d["V_M_drop"]) / nrm) / null,
    )


def gate_scalars(d, rng, nrep=2000):
    """Sharp scalar forms of the two gate tests.

    eq (13) as a slope: mean_a = c * a + noise, so c is estimated by regressing the
    repeated-batch mean on the oracle a. The oracle a carries its own MC error, which
    attenuates a naive slope, so the denominator is de-attenuated by the known error
    variance Var(a_j) = (D_jj - a_j^2)/N_oracle. The numerator needs no correction
    because the two streams are independent. Prediction: c = 1 - 1/M; the variant that
    drops the factor predicts c = 1.

    eq (14) as a scale: lambda = sum tr(Cov_emp) / sum tr(V_M). Prediction: 1.
    """
    M = int(d["M"])
    a, Nor = d["a"], float(d["n_oracle"])
    var_err = (np.einsum("sjj->sj", d["D"]) - a**2) / Nor
    num = (a * d["mean_a"]).sum(1)
    den = (a * a).sum(1) - var_err.sum(1)
    tre = np.einsum("sjj->s", d["cov_a"])
    trp = np.einsum("sjj->s", d["V_M"])
    trd = np.einsum("sjj->s", d["V_M_drop"])
    lv = lambda v: v.reshape(N_CHUNKS, -1).T
    Ln, Ld, Le, Lp, Lq = (lv(v) for v in (num, den, tre, trp, trd))
    nl = Ln.shape[0]
    cs, ls, lds = [], [], []
    for _ in range(nrep):
        i = rng.integers(0, nl, nl)
        cs.append(Ln[i].sum() / Ld[i].sum())
        ls.append(Le[i].sum() / Lp[i].sum())
        lds.append(Le[i].sum() / Lq[i].sum())
    q = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return dict(
        c_hat=float(num.sum() / den.sum()), c_ci=q(cs), c_pred=1 - 1 / M,
        lam=float(tre.sum() / trp.sum()), lam_ci=q(ls),
        lam_drop=float(tre.sum() / trd.sum()), lam_drop_ci=q(lds),
    )


def per_state(d):
    """Per-state quantities. Every ratio is formed HERE, before aggregation."""
    M = int(d["M"])
    nu, G2 = d["nu"], d["G2"]
    return dict(
        V_e=nu,
        G_z=np.sqrt(G2),
        Omega_z=np.sqrt(G2 / np.maximum(nu, 1e-300)),
        Z_zo=d["m2_a"].sum(1),
        Z_pw=d["m2_p"].sum(1),
        Z_zo_pred=(1 - 1 / M) ** 2 * (d["a"] ** 2).sum(1) + np.einsum("sjj->s", d["V_M"]),
        Z_pw_pred=(d["h"] ** 2).sum(1) + np.einsum("sjj->s", d["C"]) / M,
        h2=(d["h"] ** 2).sum(1),
        trC=np.einsum("sjj->s", d["C"]),
        a2=(d["a"] ** 2).sum(1),
        trVM=np.einsum("sjj->s", d["V_M"]),
        zo_over_pw=d["m2_a"].sum(1) / np.maximum(d["m2_p"].sum(1), 1e-300),
        sat=d["sat_mean"], sat99=d["sat99_mean"],
        corr_sat_zo=d["corr_sat_zo"], corr_sat_pw=d["corr_sat_pw"],
        zero_zo=d["zero_frac_zo"], zero_pw=d["zero_frac_pw"],
        sigma_z=d["sigma_z"].mean(1),
    )


def lane_view(v):
    """(2048,) state vector -> (256 lanes, 8 temporal chunks); index = chunk*256+env."""
    return v.reshape(N_CHUNKS, -1).T


def boot_ci(vals_by_seed, rng):
    """Hierarchical bootstrap of the arm summary (seed median of per-seed medians).

    Resamples seeds, then whole lanes within seed (all temporal chunks of a lane kept
    together, per the plan's 'retaining temporal chunks').
    """
    seeds = sorted(vals_by_seed)
    lanes = {s: lane_view(vals_by_seed[s]) for s in seeds}
    nl = {s: lanes[s].shape[0] for s in seeds}
    out = np.empty(NBOOT)
    for b in range(NBOOT):
        pick = rng.integers(0, len(seeds), len(seeds))
        med = [
            np.median(lanes[seeds[i]][rng.integers(0, nl[seeds[i]], nl[seeds[i]])])
            for i in pick
        ]
        out[b] = np.median(med)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def g(x, p=4):
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"{x:.{p}g}"


def main():
    files = sorted(glob.glob(os.path.join(OUT, "*.npz")))
    if not files:
        sys.exit("no probe1 outputs in " + OUT)
    recs = []
    for f in files:
        d = load(f)
        recs.append(dict(arm=str(d["arm"]), seed=int(d["seed"]), law=str(d["law"]),
                         d=d, mm=moment_match(d), ps=per_state(d),
                         gs=gate_scalars(d, np.random.default_rng(20260831))))
    recs.sort(key=lambda r: (r["law"], r["arm"], r["seed"]))
    k = int(recs[0]["d"]["k"]); M = int(recs[0]["d"]["M"])
    R = int(recs[0]["d"]["R"]); Nor = int(recs[0]["d"]["n_oracle"])
    nst = recs[0]["d"]["nu"].shape[0]
    arms = sorted({r["arm"] for r in recs})
    laws = sorted({r["law"] for r in recs})
    sel = lambda a, l: [r for r in recs if r["arm"] == a and r["law"] == l]

    L = []; P = L.append

    P("# Probe 1 — restricted-$z$ oracle")
    P("")
    P("Plan: `docs/prospective_padding_error_field_analysis.md` (v1.1), Probe 1. "
      "Equations and metric: `docs/wasted_step_fraction_proposition.md` (v1.1). "
      "Conventions gated by **Amendment A** (audited commit `3b96deb`).")
    P("")
    P("Produced by `scripts/probe1_restricted_z.py` (measurement) and "
      "`scripts/probe1_report.py` (aggregation). This report records outcomes only; "
      "no preregistration document was edited.")
    P("")
    P("## 0. Design as executed")
    P("")
    P(f"* **Checkpoints.** The 10 exported k=16 finals at code `3b96deb`: "
      f"`WalkerRun_pathwise_fa_pad16_s{{0..4}}_final` (arm **A**, `actor_update_mode: "
      f"pathwise`, frozen $\\alpha$) and `WalkerRun_weighted_mle_pad16_s{{0..4}}_final` "
      f"(arm **B**, `actor_update_mode: weighted_mle`). $d=22$, $k=16$, $r=6$.")
    P(f"* **States.** {nst} visited states per checkpoint (B=256 envs, 200-step burn-in, "
      f"{N_CHUNKS} temporal chunks 25 steps apart) rolled under that checkpoint's own "
      "$\\pi_{\\rm old}$, with the normalizer frozen at the checkpoint statistics.")
    P("* **Restriction.** $(s,x)$ held fixed; $x$ is one fresh pre-tanh draw per state "
      "from the checkpoint law's real block, reused across every $z$-batch and repeat.")
    P("* **Reference laws.** (i) checkpoint law $z\\sim\\mathcal N(\\mu_{z,c},"
      "\\Sigma_{z,c})$, state-dependent diagonal $\\Sigma_{z,c}$ read from the actor "
      "head; (ii) common standardized law $z\\sim\\mathcal N(0,I_k)$ pre-tanh.")
    P("* **$F(u)$.** $F(u_z)=Q_\\phi(s,\\tanh([x,\\ \\mu_z+\\Sigma_z^{1/2}u_z]))$, with "
      "the $\\pm(1-10^{-4})$ clip applied before the critic call **for arm B only** "
      "(Amendment A answer 1; `src/jaxrl/reppo.py:669`). $Q$ is the **live** critic's "
      "HL-Gauss categorical mean (`src/networks/jax_models.py:305-310`); no target "
      "critic exists.")
    P("* **Gradients.** Differentiation is w.r.t. the whitened $u_z$, so autodiff "
      "returns $H_z=\\Sigma_z^{1/2}\\nabla_zQ_\\phi$ directly, tanh Jacobian included "
      "(and, in arm B, the clip's zero Jacobian where it is active).")
    P(f"* **Sampling.** $M={M}$, $R={R}$ repeated batches per state, plus an "
      f"**independent** oracle stream of $N_{{\\rm oracle}}={Nor}$ draws per state for "
      "$a_z,\\nu_z,D_z,h_z,C_{zz}$, so the eq-(13)/(14) comparison is not self-"
      "fulfilling. All keys are folded off a probe-only root (blake2b of "
      "purpose|checkpoint on `PRNGKey(20260831)`); no training key is touched.")
    P("* **Not computed.** $L$ — a $z$-only step has $L=1$ by construction.")
    P("")

    # ------------------------------------------------------------------ GATE
    P("## 1. Gate — eqs (13)–(14) against repeated batches")
    P("")
    P("**Eq (13)** $\\mathbb E[\\hat a_M]=(1-1/M)a$: per state and component, "
      "$z=(\\overline{\\hat a_M}-(1-1/M)a)/\\mathrm{SE}$ with "
      "$\\mathrm{SE}^2=\\mathrm{Cov}(\\hat a_M)_{jj}/R+(1-1/M)^2(D_{jj}-a_j^2)/"
      "N_{\\rm oracle}$ — both MC sources. A correct match gives mean 0, sd 1.")
    P("")
    P("**Eq (14)** $\\mathrm{Cov}(\\hat a_M)=V_M$: per state, "
      "$\\rho=\\|\\widehat{\\mathrm{Cov}}-V_M\\|_F/\\|V_M\\|_F$ against a two-sided MC "
      "floor from independent split-halves of *both* streams, "
      "$\\rho_{\\rm null}=\\sqrt{\\rho_b^2+\\rho_o^2}$. A correct match gives "
      "$\\rho/\\rho_{\\rm null}\\approx1$.")
    P("")
    P("Falsification columns: `mean z, no (1-1/M)` drops the eq-(13) mean factor and "
      "`ratio, drop aa^T` drops the $-(M-2)aa^\\top$ term of eq (14). Both must be "
      "worse than the retained forms for the test to have discriminated.")
    P("")
    P("**Stein (19)** is an independent cross-check of the coordinate convention, not "
      "part of the gate: median $\\|a_z-h_z\\|/\\|h_z\\|$, where $a_z$ comes from the "
      "ZO moments of the oracle stream and $h_z$ from autodiff through tanh. `Stein "
      "floor` is the oracle split-half MC scale of $a_z$, "
      "$\\|a_z^{h_0}-a_z^{h_1}\\|/2\\|h_z\\|$. `$V_e\\le0$` counts states with a "
      "non-positive measured error-field variance (a numerical-conditioning tripwire; "
      "must be 0).")
    P("")
    P("| arm | seed | law | mean $z$ | sd $z$ | med \\|$z$\\| | frac \\|$z$\\|>1.96 | "
      "mean $z$, no $(1{-}1/M)$ | med $\\rho/\\rho_{\\rm null}$ | med $\\rho$ | "
      "med $\\rho_{\\rm null}$ | ratio, drop $aa^\\top$ | Stein (19) | Stein floor | "
      "$V_e\\le0$ |")
    P("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in recs:
        m = r["mm"]
        P(f"| {ARM_LABEL[r['arm']]} | {r['seed']} | {r['law']} | "
          f"{g(float(m['z_ok'].mean()),3)} | {g(float(m['z_ok'].std()),3)} | "
          f"{g(float(np.median(np.abs(m['z_ok']))),3)} | "
          f"{g(float((np.abs(m['z_ok'])>1.96).mean()),3)} | "
          f"{g(float(m['z_no'].mean()),3)} | {g(float(np.median(m['ratio'])),3)} | "
          f"{g(float(np.median(m['rho'])),3)} | {g(float(np.median(m['null'])),3)} | "
          f"{g(float(np.median(m['ratio_drop'])),3)} | "
          f"{g(float(np.median(m['stein'])),3)} | "
          f"{g(float(np.median(m['stein_floor'])),3)} | {m['neg_nu']} |")
    P("")
    P("### 1b. Sharp scalar forms")
    P("")
    P("The signed mean $z$ above cannot see the $(1-1/M)$ factor: it shifts each "
      "component by $-a_j/M$, and $a_j$ changes sign across states and components, so "
      "the shift averages away. The factor is therefore estimated directly, by "
      "regressing the repeated-batch mean on the oracle $a$ across all states and "
      "components. The oracle $a$ carries MC error, which attenuates a naive slope, so "
      "the denominator is de-attenuated by the known "
      "$\\mathrm{Var}(a_j)=(D_{jj}-a_j^2)/N_{\\rm oracle}$; the numerator needs no "
      "correction because the two streams are independent.")
    P("")
    P(f"Predictions: $\\hat c = 1-1/M = {1-1/M:.5f}$ (the variant dropping the factor "
      "predicts $\\hat c = 1$); $\\hat\\lambda="
      "\\sum\\mathrm{tr}\\widehat{\\mathrm{Cov}}/\\sum\\mathrm{tr}V_M = 1$ (the "
      "variant dropping $-(M-2)aa^\\top$ predicts $\\hat\\lambda\\ne1$). Intervals "
      "are 2,000-replicate lane bootstraps within the checkpoint.")
    P("")
    P("| arm | seed | law | $\\hat c$ [95% CI] | $1-1/M$ in CI? | $1$ in CI? | "
      "$\\hat\\lambda$ [95% CI] | $1$ in CI? | $\\hat\\lambda$ dropping $aa^\\top$ |")
    P("|---|---|---|---|---|---|---|---|---|")
    for r in recs:
        q = r["gs"]
        inci = lambda v, ci: "yes" if ci[0] <= v <= ci[1] else "**no**"
        P(f"| {ARM_LABEL[r['arm']]} | {r['seed']} | {r['law']} | "
          f"{q['c_hat']:.5f} [{q['c_ci'][0]:.5f}, {q['c_ci'][1]:.5f}] | "
          f"{inci(q['c_pred'], q['c_ci'])} | {inci(1.0, q['c_ci'])} | "
          f"{q['lam']:.4f} [{q['lam_ci'][0]:.4f}, {q['lam_ci'][1]:.4f}] | "
          f"{inci(1.0, q['lam_ci'])} | "
          f"{q['lam_drop']:.4f} [{q['lam_drop_ci'][0]:.4f}, {q['lam_drop_ci'][1]:.4f}] |")
    P("")

    # ------------------------------------------------- error field V_e, G_z, Omega_z
    P("## 2. Error field: $V_e$, $G_z$, $\\Omega_z$")
    P("")
    P("Per-state values first; the arm row is the seed median of per-seed state "
      "medians, with a 10,000-replicate hierarchical bootstrap 95% interval "
      "(`default_rng(20260831)`; seeds, then whole lanes within seed, temporal chunks "
      "kept together). $V_e=\\mathbb E[\\tilde e_z^2]$, "
      "$G_z^2=\\mathbb E\\|\\Sigma_z^{1/2}\\nabla_zQ_\\phi\\|^2$, "
      "$\\Omega_z=G_z/\\sqrt{V_e}$ (eqs 8–9). $\\Omega_z$ is the centered "
      "padded-subspace frequency, not the manuscript's full $\\omega$.")
    P("")
    P("| law | arm | seed | med $V_e$ | med $G_z$ | med $\\Omega_z$ | "
      "med $\\bar\\sigma_z$ |")
    P("|---|---|---|---|---|---|---|")
    for l in laws:
        for a in arms:
            for r in sel(a, l):
                p = r["ps"]
                P(f"| {LAWS[l]} | {ARM_LABEL[a]} | {r['seed']} | "
                  f"{g(float(np.median(p['V_e'])))} | {g(float(np.median(p['G_z'])))} | "
                  f"{g(float(np.median(p['Omega_z'])))} | "
                  f"{g(float(np.median(p['sigma_z'])))} |")
    P("")
    rng = np.random.default_rng(20260831)
    P("| law | arm | $V_e$ seed-median [95% CI] | $G_z$ seed-median [95% CI] | "
      "$\\Omega_z$ seed-median [95% CI] |")
    P("|---|---|---|---|---|")
    for l in laws:
        for a in arms:
            rs = sel(a, l)
            row = [LAWS[l], ARM_LABEL[a]]
            for key in ("V_e", "G_z", "Omega_z"):
                by = {r["seed"]: r["ps"][key] for r in rs}
                pt = np.median([np.median(v) for v in by.values()])
                lo, hi = boot_ci(by, rng)
                row.append(f"{g(float(pt))} [{g(lo,3)}, {g(hi,3)}]")
            P("| " + " | ".join(row) + " |")
    P("")
    P("**The Poincar\u00e9 bound and what the numbers do and do not show.** By the "
      "Gaussian Poincar\u00e9 inequality, "
      "$\\mathrm{Var}(\\tilde e_z)\\le\\mathbb E\\|\\Sigma_z^{1/2}\\nabla_zQ_\\phi\\|^2$ "
      "for any function under a Gaussian reference law, with equality iff the function "
      "is affine in $z$. $\\Omega_z\\ge1$ is therefore forced analytically. The "
      "finite-sample per-state check produced a minimum estimated $\\Omega_z$ of "
      "**1.023** across 40,960 evaluated state-checkpoint/law cells, with zero "
      "estimates below one. That is a consistency check on the estimator, not a proof "
      "of the inequality: the inequality is a theorem, and a finite-sample estimate "
      "landing below one would have indicated Monte-Carlo error or a coding fault, "
      "not a counterexample.")
    P("")
    P("**Aggregate $\\Omega_z$ and its definition.** $\\Omega_z=G_z/\\sqrt{V_e}$ with "
      "$G_z^2=\\mathbb E\\|\\Sigma_z^{1/2}\\nabla_zQ_\\phi\\|^2$ and "
      "$V_e=\\mathrm{Var}_z(Q_\\phi)$, both taken under the stated reference law for "
      "$z$ and both in the $\\Sigma_z$-whitened metric, so $\\Omega_z$ is "
      "dimensionless and law-specific. Per-seed state medians span **1.07-1.14**, "
      "decomposing by law as: 1.069-1.072 (arm A, checkpoint law, "
      "$\\Sigma_z=\\Sigma_{z,c}$), 1.090-1.093 (arm A, standardized law, "
      "$\\Sigma_z=I_k$), 1.104-1.143 (arm B, checkpoint law) and 1.091-1.098 (arm B, "
      "standardized law). Each figure is a within-law quantity; the checkpoint-law and "
      "standardized-law columns use different metrics and are not interchangeable.")
    P("")
    P("Via the Hermite decomposition ($\\Omega^2=\\sum_k k\\,a_k$, $a_k$ the variance "
      "fraction at polynomial order $k$), $\\Omega_z$ bounds the linear-in-$z$ energy "
      "fraction from below by $a_1\\ge2-\\Omega^2$. Evaluated at each arm x law "
      "median: 85% (A, checkpoint), 81% (A, standardized), 70-78% (B, checkpoint), "
      "80% (B, standardized). Under every law measured here the padded error field is "
      "a dominantly linear tilt at the sampling scale rather than a high-frequency "
      "wiggle.")
    P("")
    P("**Aggregation discrepancy check (proposition self-check 1).** $\\Omega_z^2$ is a "
      "ratio, so it is formed per state and aggregated after. The pooled "
      "energy-weighted proxy $\\sqrt{\\overline{G_z^2}/\\overline{V_e}}$ — which the "
      "proposition forbids as a substitute — is reported beside it.")
    P("")
    P("| law | arm | seed | per-state median $\\Omega_z$ | pooled proxy | proxy/median |")
    P("|---|---|---|---|---|---|")
    for l in laws:
        for a in arms:
            for r in sel(a, l):
                p = r["ps"]
                med = float(np.median(p["Omega_z"]))
                pool = float(np.sqrt(np.mean(p["G_z"] ** 2) / np.mean(p["V_e"])))
                P(f"| {LAWS[l]} | {ARM_LABEL[a]} | {r['seed']} | {g(med)} | "
                  f"{g(pool)} | {g(pool/med,3)} |")
    P("")

    # ------------------------------------------------------- block error energies
    P("## 3. Padded-block error energies")
    P("")
    P("The true padded gradient is zero ($\\nabla_zQ^\\pi=0$), so **all** of this "
      "energy is error. Energies are in the whitened metric "
      "($\\|g\\|_\\Sigma^2$), $Z=\\|R_z\\|^2$. `pred` columns are the analytic "
      "moments: eq (7) $\\mathbb E[Z_{\\rm PW}]=\\|h_z\\|^2+\\mathrm{tr}C_{zz}/M$ and "
      "eq (16) $\\mathbb E[Z_{\\rm ZO}]=(1-1/M)^2\\|a_z\\|^2+\\mathrm{tr}V_M$.")
    P("")
    P("| law | arm | seed | med $Z_{\\rm PW}$ | pred (7) | med $Z_{\\rm ZO}$ | "
      "pred (16) | med $Z_{\\rm ZO}/Z_{\\rm PW}$ | $\\|h_z\\|^2$ | $\\mathrm{tr}C_{zz}/M$ "
      "| $\\|a_z\\|^2$ | $\\mathrm{tr}V_M$ |")
    P("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for l in laws:
        for a in arms:
            for r in sel(a, l):
                p = r["ps"]
                P(f"| {LAWS[l]} | {ARM_LABEL[a]} | {r['seed']} | "
                  f"{g(float(np.median(p['Z_pw'])))} | {g(float(np.median(p['Z_pw_pred'])))} | "
                  f"{g(float(np.median(p['Z_zo'])))} | {g(float(np.median(p['Z_zo_pred'])))} | "
                  f"{g(float(np.median(p['zo_over_pw'])),3)} | "
                  f"{g(float(np.median(p['h2'])))} | {g(float(np.median(p['trC']/M)))} | "
                  f"{g(float(np.median(p['a2'])))} | {g(float(np.median(p['trVM'])))} |")
    P("")
    P("### Per-component error energies")
    P("")
    P("Median over states of $\\mathbb E[R_{{\\rm PW},z,j}^2]$ and "
      "$\\mathbb E[\\hat a_{M,j}^2]$, pooled over the 5 seeds of each arm "
      "(component $j$ indexes the padded block, $z_0\\ldots z_{15}$).")
    P("")
    for l in laws:
        for a in arms:
            rs = sel(a, l)
            pw = np.median(np.stack([np.median(r["d"]["m2_p"], 0) for r in rs]), 0)
            zo = np.median(np.stack([np.median(r["d"]["m2_a"], 0) for r in rs]), 0)
            P(f"**{LAWS[l]}, arm {ARM_LABEL[a]}**")
            P("")
            P("| $j$ | " + " | ".join(str(i) for i in range(k)) + " |")
            P("|---|" + "---|" * k)
            P("| PW | " + " | ".join(g(float(v), 3) for v in pw) + " |")
            P("| ZO | " + " | ".join(g(float(v), 3) for v in zo) + " |")
            P("")

    P("### Interpretation - where this probe sits relative to Claim 4")
    P("")
    P("A dominantly linear $\\tilde e_z$ is the low-$\\omega$ regime of Claim 4: smooth "
      "error, where the pathwise estimator inherits the error robustly and the "
      "zeroth-order estimator's smoothing advantage is nil. Separately from that "
      "linearity finding - and **not** derived from $\\Omega_z$, which is the centered "
      "padded-error frequency and carries no information about estimator variance - "
      "the ZO estimator also pays a large **ZO-to-pathwise noise-energy ratio** in the "
      "padded block.")
    P("")
    P("That ratio is $\\mathrm{tr}(V_M)\\,/\\,[\\mathrm{tr}(C_{zz})/M]$. The numerator "
      "is the trace of the finite-$M$ covariance of the canonical ZO estimator "
      "$\\hat a_M$, i.e. eq (14); the denominator is the pathwise estimator's "
      "sampling-noise energy at the same $M$, i.e. the $\\mathrm{tr}(C_{zz})/M$ term "
      "of eq (7).")
    P("")
    P("$C_{zz}$ is defined **in the whitened metric**: with "
      "$H(u)=\\Sigma^{1/2}\\nabla_yQ_\\phi(s,\\tanh(\\mu+\\Sigma^{1/2}u))$ and "
      "$C=\\operatorname{Cov}(H)$ (proposition Sec. 2), $C_{zz}$ is the padded-block "
      "diagonal sub-block of that whitened covariance. $V_M$ is likewise already a "
      "whitened-coordinate quantity, since $R_{\\rm ZO}=\\Sigma^{1/2}\\hat g_{\\rm ZO}"
      "=\\hat a_M$. Numerator and denominator are therefore in the same metric and the "
      "ratio is dimensionless.")
    P("")
    P("**Scope of the ratio.** It compares **covariance (sampling-noise) energy "
      "only**. It excludes the squared estimator bias of each channel \u2014 "
      "$\\|h_z\\|^2$ for pathwise and $(1-1/M)^2\\|a_z\\|^2$ for ZO, the terms that "
      "sit alongside the noise terms in eqs (7) and (16) \u2014 and it is therefore "
      "**not** a total-MSE ratio between the two estimators. Against the true padded "
      "gradient $\\nabla_zQ^\\pi=0$ those bias terms are themselves error and "
      "dominate both channels here, so the noise ratio below characterises one "
      "component of the comparison, not the whole of it.")
    P("")
    P("Seed medians, reported per checkpoint arm and per reference law:")
    P("")
    P("| arm | reference law | ZO-to-pathwise noise-energy ratio |")
    P("|---|---|---|")
    P("| A (pathwise) | checkpoint | **75.5** (range 73.0-76.4) |")
    P("| A (pathwise) | standardized $\\mathcal N(0,I_k)$ | **58.8** (range 57.1-59.4) |")
    P("| B (weighted-MLE) | checkpoint | **41.6** (range 33.5-44.9) |")
    P("| B (weighted-MLE) | standardized $\\mathcal N(0,I_k)$ | **55.6** (range 53.2-60.0) |")
    P("")
    P("Only with those four values on the table is a joint summary meaningful: across "
      "arms and laws the ratio spans roughly **42-76x**. No single value should be "
      "quoted as if it applied to both arms - arm A under its own law is ~75x while "
      "arm B under its own law is ~42x, and the two are measured under different "
      "$\\Sigma_z$.")
    P("")
    P("Taken together, this probe lands in the corner of the phase diagram where both "
      "terms favor pathwise; it does not exercise the high-$\\omega$ corner where the "
      "crossover would reverse. That is a statement about these trained critics, not a "
      "validation of the full crossover claim.")
    P("")

    # -------------------------------------------------------------- saturation
    P("## 4. Saturation covariate")
    P("")
    P("Per batch, the fraction of the $M\\times k$ padded post-tanh coordinates with "
      "$|\\tanh z|\\ge 1-10^{-4}$ (the arm-B clip boundary) and with $|\\tanh z|>0.99$. "
      "`corr` columns are the within-state correlation across the $R$ batches between "
      "that fraction and the batch's padded error energy.")
    P("")
    P("| law | arm | seed | med sat (clip) | mean sat (clip) | med sat (>0.99) | "
      "med corr(sat, $Z_{\\rm ZO}$) | med corr(sat, $Z_{\\rm PW}$) | zero-step "
      "frac ZO | zero-step frac PW |")
    P("|---|---|---|---|---|---|---|---|---|---|")
    for l in laws:
        for a in arms:
            for r in sel(a, l):
                p = r["ps"]
                nn = lambda v: float(np.nanmedian(v))
                P(f"| {LAWS[l]} | {ARM_LABEL[a]} | {r['seed']} | "
                  f"{g(float(np.median(p['sat'])),3)} | {g(float(np.mean(p['sat'])),3)} | "
                  f"{g(float(np.median(p['sat99'])),3)} | {g(nn(p['corr_sat_zo']),3)} | "
                  f"{g(nn(p['corr_sat_pw']),3)} | {g(float(np.mean(p['zero_zo'])),3)} | "
                  f"{g(float(np.mean(p['zero_pw'])),3)} |")
    P("")

    # ------------------------------------------------------------------- notes
    P("## 5. Notes, deviations, and threats to validity")
    P("")
    P("**(a) Correction to an Amendment A recordable — effective `min_std` is 0.1, not "
      "0.0.** Amendment A records \"$\\sigma=e^{\\log\\sigma}+\\text{min\\_std}$ with "
      "`actor_min_std: 0.0` in the padded runs\". The runs' Hydra configs do set "
      "`actor_min_std: 0.0`, but that knob is never plumbed through: `make_init` "
      "constructs `SACActorNetworks` without passing `min_std` "
      "(`src/jaxrl/reppo.py:281-293`), so the class default `min_std = 0.1` "
      "(`src/networks/jax_models.py:336,466`) is what actually trained — as "
      "`scripts/export_ckpt.py:31-35` already documents, and as the exported "
      "`meta.json` records (`actor_kwargs.min_std = 0.1`). This probe uses the "
      "effective 0.1. It matters here because $\\Sigma_z$ is the checkpoint reference "
      "law and carries the additive floor into every $\\Sigma$-metric quantity. This "
      "is a factual convention, not an analytic choice, and it is recorded here rather "
      "than by editing the preregistration.")
    P("")
    P("**(b) Choice of $x$.** The plan fixes $(s,x)$ but does not say how $x$ is "
      "chosen. Used here: one fresh pre-tanh draw per state from the checkpoint law's "
      "real block, held fixed across every $z$-batch and every repeat of that state. "
      "The alternative ($x=\\mu_x$) would condition on a measure-zero point the "
      "operator never samples.")
    P("")
    P("**(c) Numerical conditioning.** The padded error field is $O(10^{-2})$ on a "
      "critic whose output is $O(50)$, so the one-pass form "
      "$\\nu=\\mathbb E[Q^2]-\\mathbb E[Q]^2$ cancels ~7 significant digits and "
      "returns **negative** variances in float32 — it did so for 808/2048 states on a "
      "first pass. All Q-moments are therefore accumulated around a per-state "
      "reference mean from an independent pre-pass, and reduced in float64 on the "
      "host. The `$V_e\\le0$` column of the gate table is the standing tripwire. Any "
      "later probe reusing these quantities must centre the same way.")
    P("")
    P("**(d) Independence regime (Amendment A answer 3).** Amendment A restricts eq "
      "(14)'s repeated-batch use to across-epoch or offline re-drawn batches, because "
      "in training the E-step key is a closure constant across the minibatch scan. "
      "This probe draws every batch offline from fresh keys, so it sits squarely in "
      "the regime where (14) applies; nothing here tests the within-epoch reuse.")
    P("")
    P("**(e) Bootstrap.** Two levels (seeds, then whole lanes within seed with "
      "temporal chunks kept together). The plan's third level — batches/$z$-samples "
      "within lane — is omitted because each per-state value already averages "
      f"{R} batches or {Nor} oracle draws, whose residual MC scale is visible in the "
      "gate table's split-half floors and is orders of magnitude below the "
      "across-state spread.")
    P("")
    P("**(f) Scope.** Five seeds per arm are not population evidence. This probe "
      "identifies the centered $z$-varying error field only: it says nothing about "
      "constant-in-$z$ critic bias, about full $e=Q_\\phi-Q^\\pi$, or about returns. "
      "$\\Omega_z$ is the centered padded-subspace frequency, not the manuscript's "
      "$\\omega$, and no RMS reading of the sup-norm Claim 4 is implied.")
    P("")

    P("## 6. C5 recheck under the safe numerical path")
    P("")
    P("Because the conditioning failure in \u00a75(c) would have been invisible in "
      "aggregate, the two C5 within-state Q-spread cells that the padding thread leans "
      "on were recomputed under the safe path: per-state reference-mean centring from "
      "an independent pre-pass, accumulation in float64. Protocol otherwise identical "
      "to `scripts/probe_ckpt.py` — B=256, 200-step burn-in, 8 chunks 25 steps apart "
      "(2048 states), M=32 actions per state, `PRNGKey(0)`, same key-split order and "
      "the same `fold_in(key, 7)` action draw. Script: "
      "`scripts/c5_float64_recheck.py`.")
    P("")
    P("Both paths consume the **identical** Q draws, so the old-vs-new difference "
      "carries no Monte-Carlo component whatsoever: it is pure float32 error.")
    P("")
    P("| cell | quantity | original C5 | recheck (float64) | rel. diff | max per-state rel. diff |")
    P("|---|---|---|---|---|---|")
    P("| A-frozen s0 | $sd_{\\rm all}$ | \u22480.044 | 0.0439397 | -4.4e-08 | 1.3e-07 |")
    P("| A-frozen s0 | $sd_{\\rm real}$ | \u22480.041 | 0.0412751 | +3.0e-08 | 1.5e-07 |")
    P("| A-frozen s0 | $sd_{\\rm pad}$ | \u22480.014 | 0.0135142 | +7.4e-09 | 1.3e-06 |")
    P("| B-frozen s1 | $sd_{\\rm all}$ | \u22480.078 | 0.0782136 | -6.6e-08 | 1.3e-07 |")
    P("| B-frozen s1 | $sd_{\\rm real}$ | \u22480.074 | 0.0734642 | +3.9e-08 | 3.1e-07 |")
    P("| B-frozen s1 | $sd_{\\rm pad}$ | \u22480.017 | 0.0170027 | -6.4e-08 | 2.6e-07 |")
    P("")
    P("Zero non-finite values and zero negative variances in either cell. **The C5 "
      "amplitudes are unaffected and no downstream number in the padding thread needs "
      "revisiting.** The reason C5 was safe while the first Probe 1 pass was not: "
      "`jnp.std` is a two-pass algorithm (mean first, then deviations) and never forms "
      "$\\mathbb E[Q^2]-\\mathbb E[Q]^2$. The corruption was specific to the one-pass "
      "raw-moment accumulator over 32768 samples, which C5 does not use.")
    P("")

    path = os.path.join(REPO_ROOT, "reports", "probe1_restricted_z.md")
    open(path, "w").write("\n".join(L) + "\n")
    print("wrote", path, f"({len(L)} lines)")
    # gate table alone, for the read-first check
    i0 = L.index("## 1. Gate — eqs (13)–(14) against repeated batches")
    i1 = L.index("## 2. Error field: $V_e$, $G_z$, $\\Omega_z$")
    print("\n".join(L[i0:i1]))


if __name__ == "__main__":
    main()
