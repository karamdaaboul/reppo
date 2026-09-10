# Code-review addendum — independent review, verification, and fixes

An independent read-only review of the five analysis files was run after the results existed.
Every serious finding was verified personally before anything was changed. 2026-09-10.

## Findings that affect the integrity of the gates (confirmed, fixed)

**F5 — the decomposition check was vacuous. CONFIRMED.**
`resid = total - (C_sigma + C_mu)` is identically zero for *any* `F` by algebra:
`½(f_AA−f_AB+f_BA−f_BB) + ½(f_AB−f_BB+f_AA−f_BA) = f_AA − f_BB`. So V5 and the run-time
`max |decomp resid| = 1.1e-16` confirmed nothing beyond float rounding, and would have passed
with `F` wired to the wrong metric or the wrong arm. Replaced with two non-vacuous checks:
`C_sigma` against a direct one-factor-at-a-time evaluation, and `F_AA` against the
independently computed metric. Both PASS at `0.000e+00`.

**F10 — validation gate V7 could not fail. CONFIRMED.**
The old V7 hashed the same `obs` array three times, never loaded a checkpoint, and never used
`tag`. It would have passed even if the arms had been fed different banks. Replaced: each arm's
checkpoint is now loaded and the array it is actually evaluated on is hashed, plus a new
check that the three arms produce *distinct* outputs (3 distinct sigma hashes). Both PASS.

**F1 — `max_clamp` reported the smallest, not largest, clamp magnitude. CONFIRMED.**
`-v[neg].max()` returns the least-negative element. Fixed to `np.abs(v[neg]).max()`. Inert in
this run (`n_clamped = 0` at N=128), but the guard was not doing its job.

**F6, F7 — silent NaN paths. CONFIRMED as latent.** An empty rollout-bank mask (e.g. if `source`
were stored as bytes) would have produced NaN through to the CSV with only a RuntimeWarning,
and `max(nan, 1e-300)` makes `rel` NaN so the quadrature gate would report such a cell as
*clean*. Asserts on non-empty banks, finite observations and finite metrics added, plus an
explicit `isfinite` guard in the flag. Not triggered in this run.

**F9 — `excludes_1` meant "excludes 0" on saturation rows. CONFIRMED.** Renamed `excludes_null`.

## Finding that changes reported intervals (confirmed, fixed, immaterial)

**F8 — bootstrap RNG reuse. CONFIRMED.** The median rows reconstructed `default_rng(20260910)`
inside the loop, so all 27 shared one resample matrix; the mean rows drew from a single
advancing generator, so every interval depended on loop order. Replaced with deterministic
per-row child streams keyed by `(task, bank, contrast, statistic)`.

Effect on the primary bank, all nine contrasts:

* **point estimates: unchanged to `0.0e+00`** — the estimator does not use the RNG;
* interval endpoints moved by at most `1.4e-02` (worst case, walker `WML0.1/PW` upper bound
  9.109 -> 9.124, on an interval of width 3.2) — bootstrap Monte-Carlo noise;
* **every conclusion unchanged.**

## The serious numerical finding (confirmed, consequence measured, NOT concealed)

**F2 / F3 — the quadrature is not converged at extreme sigma, and the validation grid never
went there. CONFIRMED, and this is the most important item in the review.**

The V3 grid capped `sigma` at 10. The data reaches far higher:

| task | arm | max sigma | elements with sigma > 100 |
|---|---|---|---|
| walker | WML0.5 | **3.75e7** | 20387 / 147456 (13.8%) |
| leap | WML0.1 | 4.69e6 | 43021 / 393216 (10.9%) |
| leap | WML0.5 | 3.06e6 | 23720 / 393216 (6.0%) |
| walker | WML0.1 | 3.88e3 | 2365 / 147456 (1.6%) |

The grid was extended to `sigma = 1e7`. **V3 now fails**: `max |N128 − N512| = 1.213e-04`,
against a `1e-9` tolerance. The gate is left failing; it has not been weakened.

**Measured consequence for the reported quantity.** The reported metric is a *bank mean*, not a
single element. Recomputing real seed-level aggregates at N = 128 / 512 / 2048 on the four
highest-sigma cells:

| task | arm | seed | V(N=128) | V(N=2048) | relative |
|---|---|---|---|---|---|
| walker | WML0.5 | 305 | 0.669152660187 | 0.669151050136 | **2.4e-06** |
| leap | WML0.5 | 301 | 0.563273546260 | 0.563272224828 | 2.3e-06 |
| leap | WML0.1 | 301 | 0.605039524969 | 0.605039192273 | 5.5e-07 |
| walker | PW | 301 | 0.043167036100 | 0.043167035966 | 3.1e-09 |

**Worst relative movement of any real seed aggregate: `2.4e-06`**, against reported ratios of
2.7 to 15.0. The reason the scheme survives its own single-element inaccuracy is structural:
the panel boundary sits exactly at the tanh transition, so at large sigma both panels carry a
near-constant integrand (+1 and −1) and the unresolved layer holds negligible probability
mass.

**Status: the conclusions stand on this measured basis, not on a passing gate.** Anyone
reporting a *single-state* post-tanh variance at `sigma > 100` from this code should raise N.

**F4 — the in-run N64-vs-N128 check is a weak detector. CONFIRMED in principle.** Both panels
share the same split and under-resolve the same layer, so they can agree while both are off,
and bank-mean averaging cancels opposite-signed per-element errors. The direct N=128 vs
N=2048 comparison on real cells above is the stronger evidence and is what the conclusion
now rests on.

## Accepted, not fixed (low, no reported number affected)

* **F11** `matched_seeds.json` is written but `analysis.py` hardcodes seeds 301-308. All 72
  cells are complete so the sets coincide; the matched-seed design is documentation rather
  than an enforced constraint.
* **F12** `manifest.py` declares eight defaults but applies `eff()` to only two.
* **F13** section 2.5 would raise `FileNotFoundError` on a *deleted* baseline rather than
  reporting it — the most severe form of the modification it exists to detect.
* **F14** `make_figure.py` hardcodes `set_xlim(0.15, 60)`; the current data minimum is 0.199,
  within 25% of the floor, so a more extreme future seed could be clipped silently.

## Findings the reviewer raised that I checked and did not change

The Gauss-Legendre substitution, Jacobian, density factor and panel endpoints are correct;
chunking broadcasts and reshapes correctly; `T = 15` truncation discards `7.3e-51` of mass;
panel degeneration at `|mu/sigma| > 15` contributes exactly 0 and is harmless (verified
separately in `NUMERICS_NOTE.md` against a 2e7-sample Monte Carlo, agreement `6.6e-15`);
`paired_bootstrap` is genuinely paired with the seed as the unit and the percentile interval
is correct; no late-binding closure bug; `n` cannot silently disagree with the analysed
sample; the asymmetric bank construction matches `PROTOCOL.md` sections 46-47 by design.

## Net effect on the audit

No point estimate changed. No interval endpoint moved by more than `1.4e-02`. No conclusion
changed. Two validation gates that could not fail now can, and one gate now fails honestly
with its consequence quantified.
