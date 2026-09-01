# Mechanism evidence synthesis

Four bodies of evidence, deliberately kept apart. They were produced under different
designs with different identification properties, and combining their conclusions
would manufacture support that none of them individually provides.

Analysis commits `7534b77`, `7663d03`, `6da5ad5`, branch `estep-study`. Sources:
`reports/g1_kl_readonly_audit.md`, `reports/probe4_padding_error_field_results.md`,
`reports/planted_error_phase_diagram.md`.

---

## 1. What the existing 64 runs establish

4 tasks x 2 arms x 8 seeds at `07319d4`, 14/14 integrity checks PASS. Only g1 (d=29)
shows a detected difference, in favour of arm A (pathwise), with a paired CI excluding
zero. Hopper (d=4), walker (d=6) and leap (d=16) show none.

They establish that **under this protocol, on these four tasks, at these budgets, a
difference was detected on one task and not on three**. They are protocol-consistent
and reproducible.

They do **not** establish an action-dimension trend. The ordering is not monotone in
`d`, and the registered cross-task statistic is not evaluable because its
normalisation anchors were never registered before outcomes were observed. Those
anchors must not be constructed now, and no post-hoc replacement dimension test is
offered here.

They also carry a construct-validity problem that was not visible when they were
called clean (Sec. 2).

## 2. What the read-only KL audit establishes

Proved from the code trace, not inferred: the logged `train/kl` is `kl.mean()`
(`src/jaxrl/reppo.py:1002`) of exactly the `(B,)` per-state tensor that the defective
branch tests elementwise (`850-855`), which above `kl_bound = 0.1` **replaces** the
policy objective rather than penalising it. Since a finite mean never exceeds its
maximum, a logged value at or above the bound **proves** at least one state took the
gated branch.

* The branch provably fired in **64 of 64 runs**, at a median **83%** of logging
  points, in every arm on every task — including all three tasks where no difference
  was detected. Median of the per-run median KL is 0.1042 against a bound of 0.1.
* The KL multiplier is **35–112x larger in arm B on every task**, so a large dual is
  not g1-specific either.
* What *is* g1-specific is the **direction** of the dispersion contrast: arm B's KL
  IQR is 0.48–0.82x arm A's on hopper, walker and leap, and **6.9x** on g1; g1-B also
  has the largest overshoot of all eight cells and g1-A the smallest.
* **Within** g1 arm B (n=8), KL dispersion and overshoot correlate **positively** with
  final return (rho = +0.60, +0.67). The seeds that violated the trust region most
  returned the most. The pooled correlation is negative only because it re-expresses
  the arm contrast.

Recoverability: `omega` is **not identifiable** by any read-only route — `Q^pi` was
never stored and recovering it needs new environment interaction. The elementwise
gate rate is **not identifiable**. The KL multiplier trajectory **is** recoverable
(21 points/run, plus the raw parameter at three checkpoint fractions); its optimizer
state is not.

Established: the g1 return comparison is **confounded**. The two arms did not optimise
the same objective on the states where the branch was active. Not established: that
the defect *caused* the gap — three of the four arguments a causal story would need
point the other way.

## 3. What Probe 4 establishes about learned-critic error in inert coordinates

Executed as committed, on 10 regenerated checkpoints whose training is **byte-verified
identical** to the original commit (35/35 arrays, max abs difference 0.0), so all
three defects are preserved.

Primary `D_WML - D_PW`, seed medians, n=5, both reference laws:

| critic | checkpoint law | standardized law | sign | committed verdict |
|---|--:|--:|--:|---|
| A-trained | −0.0019009 | −0.0018309 | 0/5 positive | **FALSIFIES** |
| B-trained | +0.013698 | +0.0222636 | 5/5 positive | **SUPPORTS** |

**Class B — mixed.** The committed prediction holds on one critic source and reverses
on the other, identically under both laws. `p = 0.0625` is the floor of an exact sign
test at n=5; neither direction reaches conventional significance. Reported as mixed,
not resolved in favour of the supporting half.

The largest effect in the probe is not the operator: critic source moves `D` by 4–8x,
and each critic is hurt more by **its own training operator**. Exploratory: the WML
operator spends **46% (A) and 68% (B)** of its trust-region step on the `k=16`
coordinates the simulator provably discards, against 5% and 10% for PW — close to the
isotropic `k/d = 0.727`.

Not established: anything about returns. All 10 checkpoints trip the padding plan's own
contamination rule (`sigma_pad > 1.5 sigma_real`; ratios 1.74–1.93 for A, 3.64–5.58
for B), and return-level rehabilitation is out of scope by the plan's own text.

Departure from the committed order: Probes 2 and 3 have not been run.

## 4. What the planted-error experiment establishes about Claim 4

With `Q^pi` quadratic and `e` a planted sinusoid, so `e = Q_phi - Q^pi` holds by
construction and `omega = ||grad e||_inf/||e||_inf` is analytic and independently
verified. 240 cells, `d` in {2…64}, `sigma` in {0.05…0.8}, `omega` in {0.5…64}. 12/12
validation tests pass.

**Claim 4's own quantity is confirmed sharply.** The error-induced variance ratio
`Var[g_ZO]_e/Var[g_PW]_e`, isolated exactly by common random numbers, crosses 1 at
`r = sigma*omega/sqrt(d) = 1` with **zero misclassifications in 240 cells** (0/164
below, 76/76 above), scaling as `r^-2`. Per-slice fitted crossovers: median
`r* = 1.0085`, range **[0.974, 1.128]** across 19 independent `d`/`sigma`/`omega`
slices, so the collapse is not hiding a confound. The boundary was marked before
results were inspected and was not fitted.

**The operational consequence does not follow.** On trust-region update MSE against
the exact oracle, `g_ZO` beats PW in 43/240 cells and the actual WML operator in
38/240; **restricted to `r > 1`, WML wins only 34/76 (45%)**, with its own crossover at
`r* ≈ 1.67`. The reason is in the claim's own text: the sampling estimators pay the
classical `d` factor on the smooth part `Q^pi`, which the error-induced isolation
removes by construction and the operational measure does not.

**A separate finding about the implementation.** The actual E-step is not the
manuscript's `g_ZO`. Expanding the raw self-normalised softmax,
`sum_i w_i u_i = ubar + (1/eta) m_hat + O(eta^-2)`, where `m_hat` is `g_ZO`'s numerator
but `ubar = (1/M) sum_i u_i` is **absent from `g_ZO`** because its coefficients sum to
zero. The implemented E-step therefore carries an irreducible `O(sqrt(d/M))`
mean-estimation noise the pathwise operator does not pay. This is the `ubar` term
Amendment A answer 2 records as live. It is the likely source of the systematic
WML-worse-than-`g_ZO` gap (median MSE ratio ≈ 1.13 above `r = 1`).

## 5. What remains unidentified

* `omega` on any learned critic, by any route available in the retained artifacts.
* `Q^pi`, hence `e`, hence whether the manuscript's crossover condition is anywhere
  near satisfied in the actual experiments.
* Whether the KL defect caused the g1 gap. Requires corrected reruns; the retained
  data cannot answer it and the within-arm association points the wrong way.
* The elementwise gate firing rate, and the KL multiplier's per-update trajectory.
* Any action-dimension trend: not evaluable, and the anchors may not be built now.
* Whether the Probe 4 critic-source asymmetry ("each critic is hurt more by its own
  training operator") is real. Post-hoc, n=5.
* Whether Probe 1 and Probe 4 agree on a common checkpoint set — Probe 1's published
  numbers were computed on the lost originals, Probe 4's on the regenerations.

---

## 6. Paper-level decision table

| Claim | Current evidence | Evidence after this work | Status | Allowed wording | Must NOT be used |
|---|---|---|---|---|---|
| **Same smoothed-critic target** | Proposition `prop:estimand`, proof only | Verified numerically: PW unbiased for the blurred gradient; `g_ZO` unbiased up to `1-1/M`; both agree as sampling error vanishes (2 validation tests) | **Supported** | "Both estimators target the blurred gradient; we verify this numerically." | Do not extend the verification to the *implemented* E-step without the `ubar` caveat. |
| **First-order natural-gradient equivalence** | Proposition `prop:estep`, proof only | Holds for `g_ZO`. For the **actual** raw-softmax E-step it holds only after removing `ubar`; the raw WML direction has median cosine >0.99 with `ubar` and \|cos\|<0.2 with `g_ZO` | **Mixed** | "The operators coincide to first order. The implementation's uncentred weights add a `ubar` term absent from the idealised estimator." | Do not say the implemented E-step *is* `g_ZO`. |
| **Estimator variance difference** | Same-critic probe: ZO-like 4.5–7x noisier than PW on frozen learned critics | Unchanged; plus the planted result that the error-induced ratio is `r^-2` | **Supported, narrowly** | "Relative to `Q_phi`, the score-based estimator is noisier on these critics." | Do not describe this as measuring critic quality or as ruling it out. It is variance w.r.t. `Q_phi`, not w.r.t. `Q^pi`. |
| **`sigma`–`omega`–dimension crossover (Claim 4)** | Claim with unmodelled constants; no measurement | Confirmed in the planted setting for the **error-induced variance**: crossover at `r*` in [0.974, 1.128] over 19 slices, 0/240 misclassified. **Not** confirmed operationally: WML wins 34/76 above `r=1`, crossover `r* ≈ 1.67` | **Supported for the stated quantity; unsupported as an operational rule** | "In a controlled setting with known error field, the predicted crossover in error-induced variance appears at `sigma omega = sqrt(d)`. The advantage does not transfer one-for-one to update quality, because the sampling estimators still pay the classical `d` factor on the smooth part." | Do not claim the E-step produces better updates whenever `sigma omega > sqrt(d)`. Do not present the planted result as evidence about learned critics. |
| **Learned-critic padded-coordinate robustness** | Probe 1 (restricted-z) on the lost originals | Probe 4 crossed table: prediction holds on B-trained critics 5/5, reverses on A-trained 0/5, both laws | **Mixed** | "The prediction that the E-step is more affected holds for critics trained by the E-step and reverses for pathwise-trained critics." | Do not report only the supporting half. Do not draw returns from these checkpoints — all 10 trip the registered contamination rule. |
| **Action-dimension return trend** | Non-monotone across d=4,6,16,29; registered statistic not evaluable | Unchanged. Nothing here restores it | **Unidentified** | "The registered cross-task statistic is not evaluable; its anchors were not registered before outcomes were observed." | Do not claim a dimension trend. Do not define anchors post hoc or substitute a replacement test. |
| **Explanation of the g1 return gap** | Attributed to the estimator contrast | Confounded: the defective branch provably fired in 64/64 runs; g1-B has 6.9x arm A's KL dispersion and 12.6x its overshoot; but within g1-B, dispersion correlates **positively** with return | **Unidentified; the comparison is confounded** | "The g1 comparison is confounded by KL-gate behaviour that differs sharply between arms on that task; the confound cannot be removed from the retained data." | Do not say the KL defect caused the g1 result. Do not call the 64 comparisons "clean" without qualification. Do not present g1 as a clean operator contrast. |
| **Entropy-dual observation** | `sigma` held at a target throughout training, so it cannot be raised by choosing an earlier checkpoint | Reinforced from a new direction: with the dual frozen, the entropy bonus inflates `sigma` in coordinates where `Q` is flat — `sigma_pad/sigma_real` = 1.74–1.93 (A) and 3.64–5.58 (B) on 10/10 padded checkpoints | **Supported** | "The entropy term drives policy width in directions the critic cannot rank; with a frozen dual this shows up as width inflation on inert coordinates, 10/10 checkpoints." | Do not use the padded runs' returns as evidence for anything — the registered contamination rule fires on all of them. |

---

## 7. Required language

* The 64 comparisons are **protocol-consistent and reproducible**, with
  **construct-validity concerns** from the implementation defects. Not "clean".
* **No dimension trend** is claimed and **no normalisation anchors** are defined post hoc.
* Estimator variance relative to `Q_phi` is **not** a measure of critic quality, and
  critic quality is **not** eliminated as an explanation.
* The KL defect is **not** claimed to have caused g1; that needs a causal intervention.
* The g1 performance result is treated as **confounded until corrected reruns exist**.
* Probe 4 and the planted phase diagram are reported in full including where they
  contradict, or fail to support, Claim 4.
