# Planted-error phase diagram: Claim 4 with known `Q^pi` and known critic error

**Status.** Separate from Probe 4 and from the 64-run experiment. This tests whether
Claim 4's mathematical mechanism appears where `Q^pi` and `e` are known exactly. It
is **not** evidence about learned critics, and nothing here may be used to
reinterpret Probe 4 or the return results.

**Provenance.** Sweep configuration `scripts/planted/planted_sweep_config.json`,
committed before the sweep ran. Analysis commit `7534b77`, branch `estep-study`.
Python 3.12.14 / NumPy on CPU (login node); no GPU, no training. Runtime 9 min 52 s.

---

## 1. Claim 4 as stated, and what it actually asserts

Quoted from `blurring_the_critic.tex`, `claim:crossover`:

> Let `Sigma = sigma^2 I` and `omega := ||grad e||_inf / ||e||_inf`. Comparing
> error-induced variances,
> `Var[g_PW]_e <~ (1/M) ||grad e||_inf^2`, `Var[g_ZO]_e <~ (d/M) ||e||_inf^2/sigma^2`,
> so the zeroth-order estimator has the smaller error-induced variance when
> `sigma omega > sqrt(d)`.

Three things this does **not** say, and which the experiment therefore separates:

1. It is about the **error-induced** variance component, not total estimator error
   and not update quality. The manuscript itself notes that on the smooth part
   `Q^pi`, `g_ZO` pays the classical factor of `d`.
2. It uses **sup-norms** over the domain, not RMS norms.
3. It is about `g_ZO` as defined in `eq:estimators`, which is **not** the implemented
   weighted-MLE operator. Both are measured here.

## 2. Construction

* `Q^pi(a) = -0.5 (a-a*)^T H (a-a*)`, `H = I`, `mu = 0`, `a* = G v` with `G = 1`
  fixed for every `d`, so the true signal energy does not change with dimension.
  Because `Q^pi` is quadratic its Gaussian blur has the **same** gradient, so the
  oracle update is exact rather than Monte Carlo (verified,
  `test_quadratic_blur_is_gradient_invariant`).
* `e(a) = eps sin(omega v_e^T a + phase)`, `||v_e|| = 1`, `eps = 1`. A single
  sinusoid is used for the primary grid **because for `J=1` the theorem's sup-norms
  are attained analytically** — `sup|e| = eps`, `sup||grad e|| = eps*omega` — so
  `omega` is exact and independently checkable rather than assumed.
* Trust-region step and metric taken verbatim from the proposition:
  `Delta_mu = sqrt(2 eps_TR) Sigma ghat / ||ghat||_Sigma`, `||g||_Sigma^2 = g^T Sigma g`,
  `||v||^2_{Sigma^-1} = v^T Sigma^-1 v`. `eps_TR` multiplies every estimator
  identically and cancels from all reported ratios.
* Operators: `g_PW` and `g_ZO` exactly as in `eq:estimators`; plus the **actual**
  weighted-MLE mean update `Delta_mu = sum_i w_i (a_i - mu)`, `w_i = softmax(Q_i/eta)`
  with `eta` solved per batch from the MPO dual (`test_eta_dual_is_solved_not_guessed`).
* **Equal budget:** all three consume `M = 32` critic forward evaluations per batch.
  PW additionally requires `M` backward passes; ZO and WML require none. That
  asymmetry favours the sampling estimators and is **reported, not corrected for**.
* Grid: `d` in {2,4,8,16,32,64} x `sigma` in {0.05,0.1,0.2,0.4,0.8} x `omega` in
  {0.5,...,64}, 8 random directions/phases per cell, 4096 batches, seed 20260901.
  240 cells, `r = sigma*omega/sqrt(d)` spanning 0.0044 to 36.2.

**Isolation of the error-induced component.** With common random numbers, each
estimator is evaluated on `Q^pi` and on `Q^pi + e` using the *same* `u_i`; the
difference is exactly the error-induced component, and `Var[.]_e` is the trace of its
covariance. This is an exact isolation, not a decomposition assumption.

---

## 3. Primary result — Claim 4's own quantity is confirmed, sharply

| `r` decade | n cells | median `r` | median `Var[ZO]_e/Var[PW]_e` | ZO better |
|---|---:|---:|---:|---:|
| 0.003–0.01 | 7 | 0.0063 | 20922 | 0% |
| 0.01–0.03 | 21 | 0.0177 | 2571 | 0% |
| 0.03–0.1 | 47 | 0.0707 | 273.8 | 0% |
| 0.1–0.32 | 44 | 0.200 | 25.24 | 0% |
| 0.32–1.0 | 44 | 0.566 | 3.164 | 0% |
| **1.0–3.2** | 38 | 1.600 | **0.379** | **100%** |
| 3.2–10 | 30 | 4.525 | 0.0473 | 100% |
| 10–32 | 7 | 18.10 | 0.00297 | 100% |
| 32–100 | 1 | 36.20 | 0.00075 | 100% |

**`r < 1`: ZO better in 0 of 164 cells. `r > 1`: ZO better in 76 of 76 cells.
Zero misclassifications in 240 cells.**

The fitted slope of `log(ratio_e)` on `log r` is `-2.03`, and `ratio_e * r^2` has
median 0.974 (5–95%: 0.68–1.72) — i.e. `ratio_e ~ r^-2`, which is exactly what the
two quoted bounds give when their constants match. The crossover is therefore not
merely *near* 1; the two bounds' constants agree well enough that it lands on 1.

**Non-collapsed views (the collapse is not hiding a confound).** Fitting the crossover
*within* each slice separately:

| slice | fitted `r*` | misclassified |
|---|---:|---:|
| d = 2, 4, 8, 16, 32, 64 | 1.097, 1.009, 1.015, 0.978, 0.993, 0.975 | 0 each |
| sigma = 0.05 … 0.8 | 1.043, 1.006, 1.005, 1.011, 0.986 | 0 each |
| omega = 0.5 … 64 | 1.128, 1.050, 1.053, 1.001, 1.013, 1.011, 0.989, 0.983 | 0 each |

**Median `r* = 1.0085`, full range [0.974, 1.128] over 19 independent slices.** The
boundary was marked at 1 before results were inspected and was not fitted to it. The
only systematic deviation is a mild upward bias at the smallest `d` and `omega`
(`r* ~ 1.10–1.13`), where a single sinusoid barely completes an oscillation across the
sampled region — the regime the claim itself calls "fewer than one oscillation".

---

## 4. The advantage does not transfer to the update — and why

The **operational** outcome is the trust-region update error against the exact oracle
update. Here the picture is much weaker:

| `r` decade | median MSE `ZO/PW` | median MSE `WML/PW` | `P(WML better)` | cos PW | cos WML |
|---|---:|---:|---:|---:|---:|
| 0.0063 | 3.49 | 5.32 | 0.000 | 0.935 | 0.509 |
| 0.0707 | 1.48 | 1.78 | 0.140 | 0.715 | 0.357 |
| 0.200 | 1.23 | 1.32 | 0.280 | 0.601 | 0.311 |
| 0.566 | 1.38 | 1.49 | 0.261 | 0.557 | 0.234 |
| **1.600** | **1.13** | **1.23** | **0.372** | 0.471 | 0.287 |
| 4.525 | 0.680 | 0.809 | 0.556 | 0.357 | 0.504 |
| 18.10 | 0.249 | 0.321 | 0.742 | 0.257 | 0.797 |

* ZO beats PW on update MSE in **43 of 240** cells; actual WML in **38 of 240**.
* Restricted to `r > 1`, WML beats PW in only **34 of 76** (45%).
* The fitted operational crossover is `r* = 1.03` for `g_ZO` but **`r* = 1.67` for the
  actual WML operator**, against `r* = 1.0085` for the error-induced variance.

The reason is in the claim's own text: the sampling estimators pay the classical `d`
factor on the *smooth* part `Q^pi`, which the error-induced isolation removes by
construction but the operational measure does not. Claim 4 is a statement about one
variance channel; it is **not** a prediction that the E-step produces a better update
whenever `sigma omega > sqrt(d)`, and this sweep shows the two questions have
different answers over a wide band above `r = 1`.

---

## 5. A discrepancy between the manuscript's `g_ZO` and the implemented E-step

`test_wml_equals_zo_plus_a_live_ubar_term` establishes, and the sweep quantifies, that
the two are not the same operator. Expanding the raw self-normalised softmax,

```
w_i = (1/M)(1 + (Q_i - Qbar)/eta) + O(eta^-2)
sum_i w_i u_i = ubar + (1/eta) * m_hat + O(eta^-2),      ubar = (1/M) sum_i u_i
```

`m_hat` is exactly the manuscript's (unwhitened) `g_ZO` numerator, but the `ubar` term
is **absent from `g_ZO`**, because `g_ZO`'s coefficients `(Q_i - Qbar)` sum to zero.
In the implemented E-step the mean is moved to the weighted *sample* mean, so even
with perfectly uniform weights the update is displaced by `Sigma^{1/2} ubar`, an
irreducible `O(sqrt(d/M))` mean-estimation noise that the pathwise operator does not
pay. Measured: after removing `ubar`, the WML direction matches `g_ZO` with median
cosine `> 0.999`; the raw WML direction has median cosine `> 0.99` with `ubar` and
`|cos| < 0.2` with `g_ZO`.

This is precisely the `\bar u` term that Amendment A answer 2 records as **live** in
this codebase (raw softmax, no centring, no baseline, no antithetic pairing). It is
the most likely explanation for the systematic gap between the ZO and WML columns
above (median MSE ratio WML/ZO ~ 1.13 at `r > 1`), and it is a property of the
implementation, not of Claim 4.

---

## 6. Validation

`scripts/planted/run_tests.py` — **12/12 pass**:

| test | what would fail it |
|---|---|
| `error_is_exactly_the_planted_field` | `Q_phi - Q^pi != e` |
| `omega_matches_the_theorem_definition` | measured sup-norm ratio != nominal `omega` |
| `oracle_gradient_by_finite_differences` | wrong oracle gradient |
| `quadratic_blur_is_gradient_invariant` | oracle not exact under blurring |
| `pw_unbiased_for_blurred_gradient_large_M` | PW biased for the smoothed gradient |
| `zo_unbiased_up_to_one_minus_one_over_M` | the `1-1/M` shrinkage wrong |
| `wml_equals_zo_plus_a_live_ubar_term` | first-order relation or `ubar` claim wrong |
| `rotation_invariance` | construction not isotropic |
| `error_induced_variance_ratio_is_amplitude_invariant` | `eps` a tuned parameter |
| `equal_query_budget` | unequal `M` |
| `no_weight_collapse_explains_the_ordering` | softmax collapse driving the result |
| `eta_dual_is_solved_not_guessed` | dual stationarity violated |

**Two tests failed on first run and both were errors in the tests, corrected before
the suite was accepted**: an over-strict bit-exactness assertion, and
`np.linalg.norm(p, -1)` which passes `ord=-1` rather than `axis=-1`. A third test was
rewritten because its premise was wrong: it asserted that WML and `g_ZO` agree as
`eta -> infinity`, which is false — that is the limit in which `ubar` dominates. The
corrected version is the one reported in Section 5, and finding this was the most
useful thing the suite did.

The `omega` calibration in the sweep's own CSV column (`om_meas`) is **not** valid and
is not used: it sampled from the policy region rather than the sup-norm domain, so it
measures a local slope and deviates from nominal by up to 2.39x at small `sigma*omega`.
The sup-norm calibration in the test suite is the one that supports the `omega` claim.

---

## 7. What this does and does not establish

**Establishes.** In a setting satisfying Claim 4's assumptions with `Q^pi` and `e`
known exactly, the predicted crossover in **error-induced variance** exists, is sharp,
scales as `r^-2`, and sits at `r = 1` within [0.97, 1.13] across every `d`, `sigma`
and `omega` slice independently. Both estimators target the same blurred gradient and
agree after trust-region normalisation as sampling error vanishes.

**Does not establish.** Nothing about learned critics, where `omega` is unmeasured.
Nothing about returns. Nothing about the 64-run experiment, whose `omega` is not
identifiable (see `reports/g1_kl_readonly_audit.md` Sec. 6). And specifically **not**
that the E-step yields a better update whenever `r > 1` — measured operationally it
does so in only 45% of the `r > 1` cells, with its own crossover at `r ~ 1.67`.

## 8. Reproduction

```bash
cd ~/repos/reppo
./.venv/bin/python scripts/planted/planted_error_sweep.py \
    scripts/planted/planted_sweep_config.json reports/artifacts
./.venv/bin/python scripts/planted/run_tests.py
./.venv/bin/python scripts/planted/analyse_sweep.py
./.venv/bin/python scripts/planted/make_figures.py
```

Outputs: `reports/artifacts/planted_sweep.csv` (240 cells),
`fig_phase_diagram.png`, `fig_phase_noncollapse.png`, `fig_phase_crossover.png`.
