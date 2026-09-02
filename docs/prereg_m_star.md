# Pre-registration: what sets the E-step's sample requirement (M*)

**Status.** Committed before any Track A or Track B run. The LQR M-sweep of commit
`8e41e64` is outcome-seen and is the motivation; nothing in Sec. 4 or 5 below was chosen
after seeing an M* measurement.

**Revision history.** v1.0 (2026-09-01): as committed.

**Companion.** `docs/prereg_m_sweep_dmc.md` (Track C, the DMC arm).

---

## 1. The question, and why two published answers disagree

Chatterjee & Diaconis (*Ann. Appl. Prob.* 28(2):1099-1135, 2018) prove that
self-normalised importance sampling needs a sample of size ~`exp(D(target||proposal))`,
necessary and sufficient. In MPO's E-step the proposal is `pi_old`, the target is
`q* ∝ pi_old exp(Q/eta)`, and the KL is pinned at `eps_E` by the dual. That gives

    M* ~ exp(eps_E),   INDEPENDENT of d.

At `eps_E = 0.5` that is ~1.6, so `M = 32` would be enormous and the arm healthy. The
measured `KL(w||uniform) = 0.5021` and `ESS ~ 19` agree with that reading.

The LQR sweep disagrees: at `d = 21` the ESTEP cosine to the exact estimand is 0.550 at
`M = 32` and still improving at `M = 2048`.

**The candidate resolution, registered as a hypothesis, not a conclusion.** CD governs a
SCALAR expectation. `g_ZO` and `d_ESTEP` are d-VECTORS. The KL budget controls the
weights; it says nothing about resolving `d` directions. If that is right, `eps_E` is not
a sufficient statistic for the E-step's sample requirement and dimension is.

---

## 2. What is already determined and must NOT be reported as a finding

E0a (commit `33c7632`) measured `(M-1) MSE_ZO / ||g*||^2 = d + 1` to three digits across
seven dimensions. For the centred ZO estimator with no planted error, `M* - 1` is
therefore proportional to `d + 1` **analytically**.

**A fitted dimension exponent `b ~ 1` for the centred ZO arm is D1 -- determined before
the run.** It will be reported in the same sentence that says so.

Further, and registered here so it cannot be claimed later: to leading order BOTH
estimators carry an orthogonal noise energy of order `d/M`, since

    d_ESTEP = sum_i w_i u_i ~ ubar + (1/eta) a_hat_ZO      (expansion for large eta)

and `ubar` has energy `d/M`. So `b ~ 1` may be largely determined for the ESTEP arm too.
**The genuinely open quantity is the RATIO `M*_ESTEP / M*_ZO` and whether it grows with
`d`** (Q1), not the bare exponent.

---

## 3. Registered mechanistic prediction on the eps_E axis (this can fail)

`eps_E` enters only through the dual for `eta`. From the expansion above, at small
`eps_E` (large `eta`) the signal term is divided by `eta` while `ubar` is pure O(1/sqrt M)
noise carrying no signal, so the cosine DEGRADES and `M*` RISES as `eps_E -> 0`. At large
`eps_E` the weights collapse toward a single sample (`ESS -> 1`) and `M*` rises again.

**Registered prediction: `M*(eps_E)` is U-shaped with an interior minimum, i.e.
non-monotone.** CD predicts monotone increase. These differ in sign over the lower half of
the swept range, which is what makes the axis informative.

**Consequence for the fit, registered in advance.** A linear `a` fitted across a U-shaped
curve can land near zero BY CANCELLATION, which would put it in the CD null band for
entirely the wrong reason. Therefore:

- the linear fit of Sec. 4 is reported as specified, AND
- a monotonicity test is run: Spearman rho of `M*` vs `eps_E` at fixed `d`, plus whether
  the minimum of `M*(eps_E)` is interior to the swept range;
- **if the curve is non-monotone, `a` is reported but explicitly flagged as not
  interpretable, and the curve is the result.**

If instead `M*` falls monotonically with `eps_E` across the whole range, the mechanism
above is WRONG and that is reported as such.

---

## 4. Track A design

`M*(tau; d, eps_E, arm, regime)` := the smallest `M` at which the mean cosine between the
estimator and the EXACT estimand `g*` reaches `tau`, interpolated on `log M`. Cosine
rather than MSE because it is scale-free and therefore comparable across `d`; MSE is
recorded too.

- `d` in {2, 4, 8, 16, 21, 32, 64}. 21 is HumanoidRun.
- `M` in {4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048}.
- `eps_E` in {0.05, 0.1, 0.5, 1.0, 2.0, 4.0} (`exp(eps_E)` spans 1.05 to 54.6, so if CD
  governs, `M*` must move by ~50x across this axis).
- `tau` in {0.90, 0.95, 0.99}.
- Regimes: low `c = sigma*omega = 8.15`, high `c = 404` (matching commit `8e41e64`).
- Arms: PW (pathwise), ZO (centred, no `eps_E` dependence), ESTEP (softmax, shipped).

`eps_E` is an ESTEP-only axis by construction. ZO is run across the `d` axis only. PW is
recorded at every `M` as the reference; it is not `eps_E`-dependent.

One draw of `u` per cell serves all three arms and all six `eps_E` values -- `q` and `H`
are computed once and only the weights `w = softmax(q/eta)` change with `eps_E`. This is
required, not an optimisation: it makes the arms paired on identical samples.

### Fit (ESTEP arm)

    log M* = a * eps_E + b * log d + c

- **Headline `tau` = 0.95.** All three `tau` reported regardless.
- **Unweighted** ordinary least squares, matching the convention of
  `docs/prereg_lqr_crossover.md`.
- Fit performed separately per regime; both reported.

### Decision rule (committed)

- **CD supported:** `a` in [0.7, 1.3] AND `b` in [-0.2, 0.2].
- **Dimension hypothesis supported:** `b` in [0.7, 1.3] AND `a` in [-0.2, 0.2].
- **Both matter:** both outside their null bands.
- **Neither:** report `M*` descriptively and make no claim about what sets it.

Adjudication is void, and the descriptive branch is taken, if the non-monotonicity flag
of Sec. 3 fires.

### Censoring (committed)

If the cosine never reaches `tau` by `M = 2048`, `M*` is **right-censored**. Censored
cells are recorded as such, excluded from the fit, and counted in the report. If more than
20% of a given `d`'s cells are censored, that `d` is dropped from the fit and the drop is
reported with its count. `tau = 0.99` at `d = 64` is expected to censor.

### Linearity in log d (committed)

The residual of the `log d` fit is inspected. If `log M*` is not linear in `log d`, the
curve is shown and no exponent is forced.

---

## 5. Track B design

Same sweep cells, different analysis. At every cell record `ESS`, `ESS/M`,
`KL(w||uniform)`, logit spread (sd of `q/eta` across the `M` samples), and the cosine
deficit of ESTEP against PW.

- Restrict to cells where the deficit varies by a factor >= 2.
- Report Spearman and Pearson correlation of each diagnostic with the deficit, and the
  RANGE of each diagnostic within the worst-deficit decile.
- **If ESS is flat while the deficit varies, that is the result**, and it would explain
  how Section 7.6 could observe `ESS in [18.8, 19.4]` at `d = 21` and conclude the arm was
  not misconfigured. The report states plainly whether the data supports or refutes that
  reading.

**Candidate non-blind diagnostic, registered before testing:** `ESS / d` rather than
`ESS / M`. It is tested OUT OF SAMPLE: fitted on the low-frequency regime and evaluated on
the high-frequency regime. A diagnostic that only separates post hoc, or only within one
regime, is reported as failing. No diagnostic invented after seeing the correlations will
be presented as predictive.

---

## 6. Out of scope

- This is an LQR with a PLANTED rank-one error field and a quadratic `Q^pi`. It licenses
  nothing about DMC on its own; Track C is the DMC test.
- Cosine to the estimand is not return.
- `M*` is measured at a single `(sigma, omega)` pair per regime, not integrated over the
  grid.
- CD's theorem is not being tested. Its hypotheses are about a scalar functional; showing
  that a d-vector estimator needs more samples does not contradict it. The report must say
  whether the data SUPPORTS the scalar-vs-vector explanation or is merely CONSISTENT with
  it -- those are different claims, and only the second is licensed by a `b` measurement
  alone.

---

## 7. Seeds and provenance

`SEED_ROOT = 20260902`, `analysis` namespace (`ledger/README.md`). No confirmatory seed
(101-108) is consumed. The commit hash of this file is printed in every Track A run log.
