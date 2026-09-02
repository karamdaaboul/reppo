"""Phase-7 precision decomposition: which noise source dominates, and by how much.

Not an estimator of anything in the preregistration. This answers the two questions
phase 7 requires when the pilot fails: WHICH of MC rollout variance, finite-difference
bias, horizon truncation, state heterogeneity, perturbation heterogeneity, a near-zero
denominator or an implementation issue dominates, and what the remedy would cost.

The A/B split makes the decomposition direct: e_A - e_B is PURE Monte-Carlo noise
(Q_phi cancels), so its variance measures the noise, while e_A * e_B measures the
signal. The same holds for z.

Usage: mc_oracle_precision.py <pw_pilot.npz> <wml_pilot.npz> <out.json>
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from scripts.analysis.mc_oracle_analyse import (  # noqa: E402
    C_STEPS, errors, fd_index, load_run, stat_D, stat_N,
)

R_PER_GROUP = 8      # prereg 5


def decompose(r):
    eA, eB, _ = errors(r)
    S, K = eA.shape[0], eA.shape[1]
    d = int(r["mu"].shape[1])
    out = {"S": S, "K": K, "d": d}

    # ---- base error: signal vs MC noise -------------------------------------
    a, b = eA[..., 0], eB[..., 0]
    ca = a - a.mean(1, keepdims=True); cb = b - b.mean(1, keepdims=True)
    D = K / (K - 1) * float((ca * cb).mean())
    noise_e = 0.5 * float(((a - b) ** 2).mean())        # per-group-mean noise variance
    out["D_signal"] = D
    out["e_noise_var_per_group"] = noise_e
    out["e_snr"] = D / noise_e if noise_e > 0 else np.nan
    out["D_naive_would_be"] = K / (K - 1) * float((ca ** 2).mean())

    # ---- gradient: signal vs MC noise, per step size ------------------------
    for c in C_STEPS:
        p, m = fd_index(r, c)
        zA = (eA[..., p] - eA[..., m]) / (2 * c)
        zB = (eB[..., p] - eB[..., m]) / (2 * c)
        N = float((zA * zB).sum(-1).mean())
        nz = 0.5 * float(((zA - zB) ** 2).sum(-1).mean())
        out["N_%.2f_signal" % c] = N
        out["N_%.2f_noise_var" % c] = nz
        out["N_%.2f_snr" % c] = N / nz if nz > 0 else np.nan
        out["N_%.2f_naive_would_be" % c] = float((zA ** 2).sum(-1).mean())
        # empirical standard error of N over the S independent states
        per_state = (zA * zB).sum(-1).mean(1)           # (S,)
        out["N_%.2f_se_states" % c] = float(per_state.std(ddof=1) / np.sqrt(S))
        # split that SE into a state-heterogeneity floor and an MC-noise part.
        # Var(z_A z_B) = signal^2-ish + signal*noise + noise^2; the noise pieces
        # scale as 1/R and 1/R^2 in the per-group mean, so scaling R rescales them.
        per_state_noise = 0.5 * ((zA - zB) ** 2).sum(-1).mean(1)
        out["N_%.2f_se_from_noise_proxy" % c] = float(
            per_state_noise.std(ddof=1) / np.sqrt(S))

    # ---- how many rollouts would criterion B need? --------------------------
    # Noise variance in a per-group mean scales as 1/R. Write the empirical SE of N
    # as se(R) with the noise-driven part shrinking like 1/R (the noise^2 term
    # dominates when SNR < 1, giving se ~ 1/R) and the state-heterogeneity part
    # fixed. Criterion B needs the 95% lower bound above 0, i.e. se <= N/1.96.
    for c in (0.10,):
        N = out["N_%.2f_signal" % c]
        se = out["N_%.2f_se_states" % c]
        need = abs(N) / 1.96 if N != 0 else np.nan
        out["N_%.2f_se_needed_for_B" % c] = need
        out["N_%.2f_se_ratio" % c] = se / need if need and np.isfinite(need) else np.nan
        # pure-noise-limited scaling: se ~ 1/R  =>  R_needed = R * (se / need)
        out["N_%.2f_rollouts_needed_noise_limited" % c] = (
            R_PER_GROUP * se / need if need and np.isfinite(need) else np.nan)
        # state-limited scaling: se ~ 1/sqrt(S) => S_needed = S * (se/need)^2
        out["N_%.2f_states_needed_state_limited" % c] = (
            out["S"] * (se / need) ** 2 if need and np.isfinite(need) else np.nan)
    return out


def main(pw, wml, out):
    res = {}
    for tag, path in (("PW", pw), ("WML", wml)):
        res[tag] = decompose(load_run(path))
        print("=====", tag)
        for k, v in res[tag].items():
            print("  %-42s %s" % (k, ("%.5g" % v) if isinstance(v, float) else v))
    with open(out, "w") as f:
        json.dump(res, f, indent=1, default=lambda o: None if not np.isfinite(o) else float(o))


if __name__ == "__main__":
    main(*sys.argv[1:4])
