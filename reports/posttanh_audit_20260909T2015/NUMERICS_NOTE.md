# Supplementary numerical check: degenerate panel at large |mu/sigma|

Run 2026-09-10, after the results, as a robustness check. It changes no reported value.

## The concern

`tanh_moments` splits the integration range at `c = clip(-mu/sigma, -T, T)` with `T = 15`.
When `|mu/sigma| > 15` the clip saturates, one panel collapses to zero width, and the
integral is effectively evaluated on the single panel `[-15, 15]`.

This is not rare in the data:

| task | arm | max abs(mu/sigma) | elements with abs(mu/sigma) > 15 |
|---|---|---|---|
| walker | PW | 131.73 | 19395 / 147456 (13.2%) |
| walker | WML0.5 | 105.33 | 2907 / 147456 (2.0%) |
| walker | WML0.1 | 69.92 | 8963 / 147456 (6.1%) |
| g1 | PW | 54.43 | 42523 / 712704 (6.0%) |
| g1 | WML0.5 | 61.71 | 11126 / 712704 (1.6%) |
| g1 | WML0.1 | 29.53 | 17138 / 712704 (2.4%) |
| leap | PW | 45.08 | 2819 / 393216 (0.7%) |
| leap | WML0.5 | 48.68 | 4142 / 393216 (1.1%) |
| leap | WML0.1 | 57.70 | 13364 / 393216 (3.4%) |

## Why it is benign

When `|mu/sigma| > 15` the tanh transition lies more than fifteen standard deviations from
the mean, so `tanh(z)` is saturated across the entire integration range. The integrand is
then smooth and a single panel integrates it to machine precision — the panel that collapses
is exactly the panel that is no longer needed.

Convergence at extreme ratios, `v` at N = 128 / 512 / 2048 per panel:

| mu/sigma | sigma | v (N=128) | v (N=2048) | abs diff |
|---|---|---|---|---|
| 16 | 0.05 | 7.843133e-04 | 7.843133e-04 | 7.2e-14 |
| 20 | 0.05 | 4.438466e-04 | 4.438466e-04 | 9.4e-14 |
| 50 | 0.05 | 1.793628e-06 | 1.793627e-06 | 1.6e-13 |
| 132 | 0.05 | 1.455502e-13 | 0.0 | 1.5e-13 |

Independent Monte Carlo at the most extreme cell actually present in the data
(`mu/sigma = 131.7`, 2e7 samples): quadrature `1.534328e-13`, MC `1.468528e-13`,
difference `6.6e-15`.

Truncation mass discarded by `T = 15` under the standard normal: `2*sf(15) = 7.3e-51`.

## Conclusion

Every discrepancy in this regime is between `1e-13` and `1e-15`, on quantities whose
seed-level aggregates are of order `1e-1`. The degenerate panel cannot move any reported
figure. **No reported value changes; no amendment is required.**
