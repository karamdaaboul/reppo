# AMENDMENT 1 to the frozen protocol — quadrature scheme

Written 2026-09-09, **before any post-tanh outcome value was computed**. The original
protocol is preserved unmodified as `PROTOCOL.md` (sha256
`20d7ad43378d2966eba74470435a083efa27ee08312343f1f1ea9d0434f2b04d`) and its first draft as
`PROTOCOL.md.orig`. Only sections 4 and 8 are amended. Sections 1-3, 5-7, 9-11 stand
unchanged.

## Why the amendment is unavoidable

Phase 5 validation gate V3 **failed**, and investigation showed the failure is real, not a
test artifact. Evidence in `diag_quad.out`:

Gauss-Hermite does not converge for `E[tanh(z)]`, `E[tanh(z)^2]` at large `sigma`. At
`mu = -4, sigma = 10` the post-tanh variance does not settle, it wanders:

| GH order | 32 | 64 | 96 | 128 |
|---|---|---|---|---|
| `v` | 0.787058 | 0.883693 | 0.848872 | 0.805340 |

Cause: after the substitution `z = mu + sqrt(2) sigma x`, the `tanh` transition occupies a
width `~1/sigma` in `x`, which at `sigma = 10` is far narrower than the Gauss-Hermite node
spacing. The rule cannot resolve it, and raising the order does not help — `numpy`'s
`hermgauss` additionally returns non-finite weights above `n ~ 200` (`n = 400` gives
`sum(w) = nan`), so the order cannot simply be increased.

**This regime is where the data lives.** The canonical WML arm on Walker has a median
pre-tanh sigma of 7.31 on the neutral bank, with much larger values in the tail. Reporting
64-point Gauss-Hermite would have produced a systematically wrong primary metric.

## Amended section 4 — quadrature

Replace Gauss-Hermite with **two-panel Gauss-Legendre split at the tanh transition**.
Writing `z = mu + sigma t` with `t` standard normal, the panels are `[-T, c]` and `[c, T]`
with `c = clip(-mu/sigma, -T, T)` (the point where `z = 0`, i.e. the transition centre) and
`T = 15`. Each panel uses `N`-node Gauss-Legendre against the standard-normal density.
Placing the transition exactly on a panel boundary makes the integrand smooth within each
panel; Gauss-Legendre nodes additionally cluster towards panel ends, which is where the
remaining structure is.

**Reported result uses `N = 128` per panel. `N = 64` per panel is the sensitivity check.**

Convergence, same grid (`diag_quad.out` D3): `|N=128 - N=256|` and `|N=256 - N=512|` are at
most `1.3e-14` at every tested `(mu, sigma)`, including `mu = -4, sigma = 10`, where the
converged value is `v = 0.830720629489` — the 64-point Gauss-Hermite answer `0.8837` is wrong
by 6.4%.

Everything else in section 4 is unchanged: float64, deterministic, `v = m2 - m1^2`, negative
values clamped with count and magnitude recorded, computed in normalized tanh action space,
`V = mean over states and dims`, analytic saturation via the Gaussian CDF.

## Amended section 8 — accuracy check

The 32-versus-64 Gauss-Hermite comparison is replaced by **64-versus-128 nodes per panel**
under the new scheme. The flag criterion is unchanged: relative difference above `1e-3` AND
absolute difference above `1e-8`. The higher-accuracy reference used in validation is
`N = 512` per panel rather than 128-point Gauss-Hermite.

## Corrected validation gate V4 (test defect, not a method defect)

V4 also failed, for an unrelated and purely test-side reason: the "high-accuracy numerical
reference" used `hermgauss(400)`, which returns NaN weights, so the comparison was `nan`.
Against a valid Gauss-Legendre reference the residual is `1.6e-2`, which is the *reference's*
error — Gauss-Legendre converges slowly on the discontinuous indicator `|tanh(z)| > 0.95`.

The analytic saturation probability is closed-form exact by construction (a Gaussian CDF
evaluation), so the correct gate tests the **implementation**, not the formula. V4 is
restated as: `sat_prob` must agree with an independent Gaussian survival-function
implementation (`scipy.stats.norm.sf`) to `1e-12`.

## Scope of this amendment

No change to run selection, populations, contrasts, metrics definition, aggregation order,
bootstrap, missing-data policy, or interpretation gates. No equivalence margin is introduced;
conclusion 3 remains unavailable.
