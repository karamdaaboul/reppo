# Mechanism evidence synthesis

Four bodies of evidence, deliberately kept apart. They were produced under different
designs with different identification properties, and combining their conclusions
would manufacture support that none of them individually provides.

Analysis commits `7534b77`, `7663d03`, `6da5ad5`, `0fd0a30`, branch `estep-study`;
final analysis HEAD `39c5171`. Sources: `reports/g1_kl_readonly_audit.md`,
`reports/probe4_padding_error_field_results.md`,
`reports/planted_error_phase_diagram.md`, and — superseding the earlier reading of the
implemented E-step wherever they conflict — `reports/ubar_code_trace.md` and
`reports/ubar_ratio.md`, preregistered at `docs/prereg_ubar_ratio.md` (`5912170`).

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
zero. The implemented E-step therefore carries an `O(sqrt(d/M))` mean-estimation
displacement, in this synthetic setting, that the pathwise operator does not pay. This
is the `ubar` term Amendment A answer 2 records as live, and it is the likely source of
the systematic WML-worse-than-`g_ZO` gap here (median MSE ratio ≈ 1.13 above `r = 1`).

> **Superseded in scope by Sec. 5.** The word "irreducible" appeared here in the
> earlier version and is withdrawn: within a training minibatch the term does average
> down across states, and the frozen-checkpoint audit finds it neither
> dimension-amplified nor dominant in the trained system. The algebra above stands; the
> extrapolation from it to the benchmark does not, and Sec. 5 replaces it.

## 5. What the uniform empirical-mean audit establishes

74 frozen checkpoints, both arms, `d in {4, 6, 16, 22, 29}`, `M = 32`, preregistered
before any checkpoint number was computed. **This section supersedes Sec. 4's closing
paragraph**, which inferred from the planted setting that the implemented E-step
"carries an irreducible `O(sqrt(d/M))` mean-estimation noise". The algebra is right;
the inference about what it does in the trained system was untested, and is now tested.

* **The decomposition is exact.** The whitened mean score equals `sum_i w_i u_i_fit`
  (autodiff agreement `1.1e-15`), the weights are fully stop-gradiented, and
  `v = ubar + c` holds identically. This is the complete mean-score decomposition.
* **The first-order identification of `c` fails at operating logits.** `c ~= m_hat/eta`
  meets its preregistered adequacy criteria in **1 of 10** conditions, with residuals to
  **5.364**. It holds only where the softmax logit spread is below about 0.6. In two
  conditions the E-step has collapsed onto a single sample (median ESS **1.07** with
  max weight 0.968, and **1.36** with 0.848, out of M=32).
* **`ubar` and `c` are orthogonal** in every condition (cosines within
  `[-0.029, +0.013]`), so `ubar` rotates the score rather than reinforcing or
  cancelling it. State-level energy ratio `R2_exact` is **0.20-0.71**.
* **Not dimension-amplified.** In the only matched within-task manipulation of
  estimator-visible `d` (WalkerRun 6 -> 22), all-pairs `Rho` medians are **1.035** and
  **0.485**, both below the registered 1.3 threshold:
  **DIMENSION AMPLIFICATION REFUTED IN THE MATCHED ESTIMATOR PROBE.** Unpaired in seed,
  differing `alpha`, contaminated padded cohort; no return conclusion.
* **Not dominant where the return gap is.** At `d=29` the batch actor-gradient ratio is
  **0.085** (arm A) and **0.452** (arm B) in the mean-output and empirical-KL metrics:
  the centred critic-dependent component is larger in both. `ubar` is material but
  non-dominant only at WalkerRun `d=6` (1.90, 2.63) and padded `d=22` arm A (1.45).
* **Clipping, not tanh, is the transformation-level source.** The tanh Jacobian is
  mu-independent and cancels exactly; the clip binds on a condition-median up to
  **22.1%** of samples (43.3% at one checkpoint), inflates RMS `||ubar||` up to 3.0x,
  and gives `ubar_fit` a systematic mean vector up to **18.4x** the raw one.
  `ubar_fit` therefore carries transformation-induced bias, not Gaussian noise, and
  `d/M` does not predict it.

**Consequence.** `ubar` is a real, sometimes material finite-sample difference between
the implemented E-step and the centred estimator the manuscript analyses. It is
**common across tasks, largest in several low-dimensional conditions, and least
important at `d=29`**. It therefore does not explain why only one task has a detected
return gap, and it is closed as a dimension-based explanation of it.

## 6. What remains unidentified

* `omega` on any learned critic, by any route available in the retained artifacts.
* `Q^pi`, hence `e`, hence whether the manuscript's crossover condition is anywhere
  near satisfied in the actual experiments.
* Whether the KL defect caused the g1 gap. Requires corrected reruns; the retained
  data cannot answer it and the within-arm association points the wrong way.
* The elementwise gate firing rate, and the KL multiplier's per-update trajectory.
* Any action-dimension trend: not evaluable, and the anchors may not be built now.
* Whether the Probe 4 critic-source asymmetry ("each critic is hurt more by its own
  training operator") is real. Post-hoc, n=5.
* What the nonlinear centred component `c` is, mechanistically, outside the small-logit
  regime. The audit measures it but does not model it, and in two conditions the
  weighting has collapsed to a near-argmax.
* Which of the distinctions between the analysed estimator and the executed algorithm —
  nonlinear softmax weighting, action clipping, covariance updating, shared trunk
  parameters, or the trust-region gate — carries the g1 difference, if any does.
  `ubar` is now excluded as a dimension-based explanation, which narrows the list
  without resolving it.
* Whether Probe 1 and Probe 4 agree on a common checkpoint set — Probe 1's published
  numbers were computed on the lost originals, Probe 4's on the regenerations.

---

## 7. Paper-level decision table

Every row: allowed wording, forbidden wording, exact evidence source.

| # | Claim | Status | Allowed wording | Must NOT be used | Evidence |
|--:|---|---|---|---|---|
| 1 | Same blurred-critic estimand | **SUPPORTED** | "Both estimators target the gradient of the same Gaussian-smoothed critic." | Do not extend to the *implemented* E-step without the `ubar` caveat. | `planted_error_phase_diagram.md` §6 (2 validation tests); `blurring_the_critic.tex` `prop:estimand` |
| 2 | Population first-order natural-gradient identity | **SUPPORTED** under the stated idealised assumptions | "Under the stated assumptions and matched KL budgets the idealised population mean updates coincide to first order." | Do not say the *implemented* operators coincide. | `prop:estep`, `cor:identical`; unaffected by the audit |
| 3 | First-order description of implemented WML | **UNSUPPORTED GENERALLY AT OPERATING LOGITS**; local expansion only | "`c ~= m_hat/eta` is a small-logit expansion; it fails its adequacy criteria in 9 of 10 audited conditions (residuals to 5.364)." | Do not call `m_hat/eta` the empirical centred WML direction. Do not use it to explain the benchmark. | `ubar_ratio.md` §5 |
| 4 | Exact implemented mean-score decomposition `v = ubar + c` | **SUPPORTED** | "The implemented mean score decomposes exactly into a uniform empirical-mean term and a nonlinear centred term." | Do not call `v` the full implemented actor update. | `ubar_code_trace.md` §0.8 (T4, T5); `ubar_ratio.md` §2 |
| 5 | `ubar` as a dimension-amplified mechanism | **REFUTED IN THE MATCHED ESTIMATOR PROBE** | "`Rho` medians 1.035 and 0.485, both below the registered 1.3 threshold." | Do not say dimension amplification is observed. Do not say it cannot affect returns. | `ubar_ratio.md` §9 |
| 6 | `ubar` as the dominant g1 gradient component | **NOT SUPPORTED**; centred component larger | "At `d=29` the ratio is 0.085 (arm A) and 0.452 (arm B); the centred critic-dependent component is larger in both." | Do not say `ubar` dominates, caused g1, or that removing it would improve return. | `ubar_ratio.md` §11b; `ubar_batch_gradient.csv` |
| 7 | Clipping-induced fitted-score bias | **SUPPORTED** at frozen checkpoints; no return-level claim | "The hard clip gives `ubar_fit` a transformation-induced bias; the tanh transform itself cancels exactly." | Do not attribute it to tanh. Do not apply `d/M` to `ubar_fit`. Do not infer returns. | `ubar_ratio.md` §6, §12 |
| 8 | Claim 4 error-induced variance crossover | **SUPPORTED** in the planted ground-truth setting | "Under the assumptions of Claim 4, the critic-error-induced variance of the centred value estimator becomes smaller than the pathwise one when `sigma*omega > sqrt(d)`; 0/240 cells misclassified, per-slice `r*` in [0.974, 1.128]." | Do not state it for total variance, exact `c`, `v`, the actor gradient, update MSE, or return. | `planted_error_phase_diagram.md` §3 |
| 9 | Claim 4 as an operational WML-selection rule | **UNSUPPORTED** | "The advantage does not transfer: above `r=1` the actual WML operator wins only 34 of 76 cells, with its own crossover near 1.67." | Do not write "the E-step wins when `sigma*omega > sqrt(d)`". | `planted_error_phase_diagram.md` §4 |
| 10 | Learned-critic padded-coordinate prediction (Probe 4) | **MIXED** | "Holds on B-trained critics (5/5, +0.0137) and reverses on A-trained critics (0/5, -0.0019), identically under both reference laws." | Do not report only the supporting critic source. Do not infer returns; all 10 checkpoints trip the width gate. | `probe4_padding_error_field_results.md` §3 |
| 11 | Action-dimension return trend | **UNIDENTIFIED** | "The ordering is not monotone in `d`; the registered cross-task statistic is not evaluable." | Do not claim a trend. Do not define anchors post hoc. | `mechanism_evidence_synthesis.md` §1 |
| 12 | Cause of the g1 return gap | **UNIDENTIFIED AND CONFOUNDED** | "The comparison is confounded by KL-gate behaviour that differs sharply between arms on that task; the confound cannot be removed from the retained data." | Do not say the KL defect caused it, that `ubar` caused it, or that the runs are clean. | `g1_kl_readonly_audit.md` §7; `ubar_ratio.md` §14 |

Two further rows retained from the earlier synthesis, unchanged in status:

| # | Claim | Status | Note |
|--:|---|---|---|
| 13 | Estimator variance relative to `Q_phi` | Supported, narrowly | It is variance conditional on `Q_phi`, **not** error relative to `Q^pi`. Critic quality is **not** eliminated; `omega` remains unidentified on learned critics. |
| 14 | Entropy-dual width inflation on inert coordinates | Supported | 10/10 padded checkpoints, ratios 1.74-1.93 (A) and 3.64-5.58 (B). Returns from those runs remain unusable. |

## 8. Required language

* The 64 comparisons are **protocol-consistent and reproducible**, with
  **construct-validity concerns** from the implementation defects. Not "clean".
* **No dimension trend** is claimed and **no normalisation anchors** are defined post hoc.
* Estimator variance relative to `Q_phi` is **not** a measure of critic quality, and
  critic quality is **not** eliminated as an explanation.
* The KL defect is **not** claimed to have caused g1; that needs a causal intervention.
* The g1 performance result is treated as **confounded until corrected reruns exist**.
* Probe 4 and the planted phase diagram are reported in full including where they
  contradict, or fail to support, Claim 4.
