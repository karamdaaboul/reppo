# Pre-registration: LQR estimator-crossover experiment

**Status.** Committed before any E1a run. E0 and E0a were executed first, by design (they
gate the sweep); their outcomes are recorded in Sec. 8 below and were known at commit
time. Nothing in Sec. 5 was chosen after seeing an E1a, E1b, or rank-ladder result.

**Revision history.** v1.0 (2026-09-01): as committed.

**Companions.** `docs/wasted_step_fraction_proposition.md` (eq. (13)/(14), the
`(1 - 1/M)` attenuation and its covariance); `docs/prereg_dimension_ladder.md` (the
DMC-side ladder this does not replace).

---

## 1. Provenance

**Committed after:** the E0 gates and E0a (G1-G11) had run and passed; the closed-form
derivation of Sec. 3 had been carried out and checked against Monte Carlo; the sigma and
omega grid bounds had been placed using E0a's measured dimension factor.

**Committed before:** any E1a run, any E1b run, any rank-ladder run, any E2 run.

**The leakage path, named explicitly.** E0a measures `(M-1) MSE_ZO / ||g*||^2 = d + 1`,
and that result is the natural input to placing the grid so the crossover is interior at
every `d`. Using it that way is legitimate -- grid placement is not the decision rule --
but the two must not be conflated afterwards. Grid bounds: derived from E0a. Decision
rules (Sec. 5): fixed independently, and before, and not a function of any measured
crossover.

**Seed namespace.** `SEED_ROOT = 20260902`, a NEW `analysis` namespace for CPU-only
offline analysis. These runs must never consume the confirmatory seeds 101-108 reserved
in `ledger/README.md`, and must never be relabelled as confirmatory evidence.

---

## 2. Setup

Discounted LQR, `s' = A s + B a + w`, `r = -(s^T Qc s + a^T Rc a)`, `gamma = 0.99`,
`n = 2d`, `W = I`. Policy `pi(a|s) = N(-K s, sigma^2 I)` with `K` a scaled DARE gain
(`k_perturb = 1.15`), so the comparison happens away from the optimum where the
action-gradient is nonzero. Systems are rejection-sampled on `rho(A_K) <= 0.99`,
`cond(H) <= 50`, and a controllability floor; every accepted system's diagnostics are
logged with its results.

`Q^pi`, `grad_a Q^pi`, and the Gaussian-blurred estimand are closed form. The critic is
`Q_phi = Q^pi + e` with a PLANTED error field

    e(s, a) = (eps / sqrt(r)) * sum_{j=1..r} sin( omega v_j^T a + phi_j )

`V` orthonormal rows, `phi_j ~ U[0, 2pi)` per state, `eps` set as a fixed fraction of the
closed-form within-state Q spread so the amplitude is scale-matched across `d`.

**Operators.** Three arms, all driven through `src/jaxrl/estimators.py`, the same module
the training diagnostics and `scripts/probe2_full_estimator.py` call:

1. `g_PW`  -- pathwise, `(1/M) sum_i grad_a Q_phi(s, a_i)`.
2. `g_ZO`  -- centred zeroth-order, `(1/(sigma M)) sum_i (Q_i - Qbar) u_i`, WITH the
   `M/(M-1)` correction of Appendix A.2. This is the estimator Claim 4 and
   Proposition 1 are stated about.
3. `d_ESTEP` -- `sum_i w_i u_i`, `w = softmax(Q/eta)`. This is what the shipped
   `actor_update_mode="weighted_mle"` arm actually moves the mean by, and it is NOT
   estimator 2. Recording both is what keeps the paper's claim and the paper's code
   from being silently assumed equivalent.

`M = 32` primary. `N = 10^4` replicates over 64 states drawn from the closed-loop
stationary law at a single reference width `sigma_ref = 0.1`, shared across the whole
sigma grid so the comparison is paired.

**Grid.** `sigma` and `omega` share a common log ratio `r_g = 300^(1/19) ~ 1.3502`;
`sigma` 20 points from 0.01 (to 3), `omega` 34 points from 0.1 (to ~2005). Hence
`c = sigma*omega` takes 53 distinct values and EVERY sigma column brackets the crossover
at every `d`. The spec's `omega <= 300` does not: at `sigma = 0.01` it reaches only
`c = 3`, below `c*(64) = 8.13`, so the low-sigma columns could not have bracketed.

---

## 3. What is already known analytically, and must not be presented as a test (D1)

Decompose the whitened draw as `u_i = t_i v + w_i`, `w_i` orthogonal to `v`. For the
rank-one field, with `c = sigma*omega` and `theta = omega v^T mu + phi`:

    Var[g_PW]_e = (eps/sigma)^2 * c^2 * Vcos(c,theta) / M               <- d-INDEPENDENT
    Var[g_ZO]_e = (eps/sigma)^2 * [ (d-1) Vsin(c,theta)/(M-1) + Lambda(c,theta,M) ]

    Vcos = 1/2 + (1/2)e^{-2c^2}cos 2theta - e^{-c^2}cos^2 theta
    Vsin = 1/2 - (1/2)e^{-2c^2}cos 2theta - e^{-c^2}sin^2 theta

`Lambda` is the variance of the unbiased sample covariance of `(sin(theta + c t), t)`
over `M` draws, and is exactly `d`-independent. Consequences, all known before E1a:

1. Both terms carry the same `(eps/sigma)^2`, so the E1a contour is exactly the
   hyperbola `sigma*omega = c*(d)` and is **independent of eps**.
2. As `c` grows, `Vcos, Vsin -> 1/2` and `Lambda -> 1/(2(M-1))`, so
   `c^2/(2M) = d/(2(M-1))` and `c* -> sqrt(d M/(M-1))`.
3. Therefore **`p -> 1/2` by construction**: the pathwise error variance is
   `d`-independent because a rank-one error field makes it so, and the zeroth-order one
   is `(d-1)*const + const`. Solved exactly: `p = 0.4572` (d=1..64), `0.4826` (d=2..64),
   `0.4999` (d>=8).

**A rank-one E1a therefore cannot refute Claim 4.** It is a verification that the
pipeline and the production estimator core reproduce a closed-form identity. It is
recorded here in advance so it can never be reported as a test that could have failed;
a reviewer re-derives the above in ten minutes.

---

## 4. Prediction

**Rank-one E1a.** `c*(d)` matches the closed form of Sec. 3 to Monte Carlo error at all
seven `d`; the fitted `p` lands in `[0.35, 0.65]`; the contour collapses onto
`sigma*omega`.

**Rank ladder (the falsifiable arm).** Claim 4 as stated predicts a `sqrt(d)` threshold
in the MEASURED frequency `omega_inf`, whatever the error field's rank. Under the
registered norm convention the prediction is `p = +0.5` at every rank. The closed forms
of Sec. 3 do not determine this: for `r > 1` the pathwise error variance is no longer
`d`-independent, and the relation between the nominal `omega` and the realised
`omega_inf` changes with `r`.

**E1b.** The full-estimator crossover sits at higher `sigma*omega` than E1a's, by the
smooth-part variance the zeroth-order operator pays on `Q^pi` -- quantitatively E0a's
`(d+1)/(M-1)` factor. If E1a and E1b disagree, that gap is the result and is reported as
such, not resolved in favour of whichever number is more convenient.

---

## 5. Decision rules (committed)

### 5.1 The norm convention, registered first

Claim 4 writes `omega = ||grad e||_inf / ||e||_inf` without saying which norm
`||grad e||_inf` is, and the reading decides the exponent's SIGN. For the full-rank field
`||e||_inf = eps sqrt(r)`, `sup_a ||grad e||_2 = eps omega`, and
`max_j sup_a |d e/d a_j| = eps omega / sqrt(r)`, so the realised frequency is
`omega/sqrt(r)` under the L2 reading and `omega/r` under the componentwise one -- fitted
exponents of `0` and `-0.5` against `+0.5`.

**Registered primary: `omega_inf = sup_a ||grad e||_2 / ||e||_inf`**, the
Lipschitz-constant reading, which is the object Mohamed et al. (2020) bound the pathwise
variance by. All four combinations of `grad_norm in {2, inf}` and
`val_norm in {inf, rms}` are recorded for every row and tabulated in the report.

### 5.2 Rule A -- rank-one E1a (VERIFICATION, not adjudication)

Fit `log c*(d) = const + p log d` over the registered `d`-set.

- `p in [0.35, 0.65]` AND the per-`d` `c*` within 4 Monte Carlo standard errors of the
  Sec. 3 closed form: **VERIFIED**. The harness and the production estimator core
  reproduce the closed-form identities.
- Outside that band, or any `c*` outside 4 SE of the closed form: **IMPLEMENTATION
  BUG**. Investigate and fix. This is explicitly NOT a refutation of Claim 4, because
  Sec. 3 shows the rank-one answer is algebraic.

### 5.3 Rule B -- rank ladder (FALSIFIABLE)

Ranks `r in {1, 2, ceil(sqrt(d)), d}`, under the registered primary norm convention, at
`M = 32`. Fit `p` as above at each rank, and fit the joint exponent across the ladder.

- `p in [0.35, 0.65]` at every rank, with 95% bootstrap CIs excluding 0 and excluding 1:
  **CONFIRMED**. Claim 4's `sqrt(d)` scaling holds beyond the rank-one construction.
- `p in [0.8, 1.2]`: the scaling is **linear in `d`, not `sqrt(d)`**. Claim 4 must be
  restated. Check whether the sup-norm-to-RMS gap explains the discrepancy before
  rewriting.
- `p` outside both bands, or `c*(d)` not monotone in `d`, or `p` differing across ranks
  by more than the CIs allow: **Claim 4's dimensional prediction is wrong as stated and
  must be withdrawn from the abstract**, and the paper reports the rank dependence
  instead.
- **CONTAMINATED** (report separately, claim neither) if any contributing configuration
  has `rho(A_K) > 0.99` or `cond(H) > 50`, or if the realised `eps/q_spread` departs
  from its nominal by more than 10%.

### 5.4 Statistics (committed)

- Registered `d`-set for the primary fit: **`d in {2, 4, 8, 16, 32, 64}`**. `d = 1` is
  excluded because the orthogonal complement of `v` is empty there, the `(d-1)` term
  vanishes identically, and the crossover is set entirely by the along-`v` term -- a
  structurally different regime. Both fits land inside the registered band (0.4826 vs
  0.4572), so nothing turns on the choice; it is registered so it cannot be made after
  the fact. `d = 1` is reported as a secondary fit.
- Root-find in `log c` at fixed `d`, per state before any aggregation. Monotone PCHIP
  interpolation, `brentq`. **Interior brackets only**: a root at a grid edge means the
  range was inadequate, not that a crossover was found.
- **No-bracket rule:** if more than 10% of a given `d`'s states lack an interior
  bracket, that `d` is dropped from the fit and the drop is reported with its count.
- **Collapse threshold:** the crossover is solved independently in each of the 20 sigma
  columns; the residual scatter of `log(sigma*omega*)` about `log c*(d)` must have
  sd < 0.05 in log units. Larger means an implementation error or an unmodelled sigma
  dependence, and must be resolved before the exponent is reported.
- Bootstrap: hierarchical, two levels -- 64 states with replacement, then 40 batch means
  within each drawn state. `np.random.default_rng(20260902)`, **10 000 resamples**,
  percentile 95% CI. The variance decomposition BY LEVEL is reported: at `d >= 8` the
  per-state spread is negligible, so a CI driven entirely by replicate count is an
  artefact of `N`, a knob, and must be described as such rather than as evidence.
- Never subtract two variances; all comparisons are log-ratios.
- **Cross term (spec Sec. 5.2):** computed from the same pass. If
  `|2 Cov| > 0.25 * Var_e` anywhere in the swept region, Claim 4's variance
  decomposition is flagged as incomplete in the paper. It is reported either way.

### 5.5 E2 (follow-on; registered now so it cannot be chosen later)

**Metric `S`: the undiscounted stationary covariance**
`S = A_K S A_K^T + W + sigma^2 B B^T`. Registered because it is what REPPO actually
computes -- a batch average over visited states, with no `gamma^t` weighting.

This is a **robustness choice, not a validity one**: `S` is shared between both arms, so
it shifts them identically and cannot favour either operator.

**Initial state distribution `s_0`: the closed-loop stationary law.** This is the thing
that actually decides whether the two metrics differ. Under stationary `s_0`,
`E[s_t s_t^T] = S_inf` for all `t`, so `S_gamma = S_inf / (1 - gamma)` -- proportional,
and the scalar is absorbed by the trust-region budget `eps`. **Under stationary `s_0` the
metric choice is a step-size rescaling and nothing more.** The two metrics differ in
SHAPE only under a transient state distribution. Stated explicitly so a reviewer does not
read the choice as an oversight. A coarse `gamma`-discounted-occupancy arm is run on a
subgrid; if the return crossover moves between metrics, that is reported in one sentence.

Implementation requirements, registered: re-solve the Lyapunov equation for `S` at every
iteration (`K` changes, so `A_K` changes, so `S` changes; a stale `S` silently drifts the
step geometry as the policy improves); assert `rho(A_K) < 0.99` at every iteration, not
only at setup, because the E2 loop can walk `K` into regions the E1 grid never covered;
log `S`, its condition number, and `tr(G S^{-1} G^T Sigma)` per iteration, and discard
rather than rescue any run where `S` becomes ill-conditioned.

---

## 6. Out of scope

This experiment licenses **nothing** about DeepMind Control.

- `Q^pi` here is quadratic and globally smooth; a neural critic is neither.
- The error field is planted, not learned. Whether real critic errors have this
  structure -- in particular whether on-policy critics have larger `omega` than
  replay-trained ones -- is untested here (that is E3, not run).
- A positive result is a consistency check on the theory and evidence that the shipped
  estimator core reproduces the stated identities. It is not evidence that the algorithm
  works, and the abstract must not blur the two.
- `sigma < 0.1` is out of reach for the DMC arms: the effective `min_std` is 0.1,
  hardcoded in `src/networks/jax_models.py` (`ReppoConfig.actor_min_std` is dead config).
  Figures shade that region and captions say so. A crossover located at `sigma = 0.02`
  says nothing about the fork.
- The rank-one arm cannot detect the norm ambiguity of Sec. 5.1, because at `r = 1` every
  convention coincides. Only the rank ladder can.

---

## 7. What would make this experiment misleading

Registered so the guards cannot be quietly dropped.

- A near-marginal closed loop inflates the state covariance and moves the effective `eps`
  without moving the nominal one. Guard: rejection sampling plus per-configuration
  logging of `rho_closed`, `cond_H`, `tr_H2`, `||P||_F`, retries.
- `eps` not scale-matched across `d`. Guard: `eps` set from the closed-form Q spread; the
  `unit_H` normalisation holding `||H||_2` and `E_s||g*||` fixed so the `sigma^4`
  curvature term (which grows like `d^3`) does not drive E1b's `d` axis; and the
  eps-invariance arm at `eps_frac in {0.05, 0.20}`.
- float64 throughout, forced before any array is created, asserted at import and inside
  the kernel. The crossover is a ratio of two variances that are equal by construction;
  float32 does not resolve it. `dlog R / dlog c ~ -2`, so a 2% error in the ratio gives
  1% in `c*`, and the report states this.
- Squared errors accumulate about the EXACT estimand, never a sampled mean.

---

## 8. Reference points already measured (E0 / E0a, before this commit)

| gate | result |
|---|---|
| G1 Lyapunov relative residual | max 7.05e-15 over all `d`, both cost arms (bound 1e-12) |
| G2 closed-form `V^pi` vs MC rollout | max abs z 1.61 (bound 3.4), `n_traj = 4096`, horizon 1375 |
| G3 Stein identity, corrected `g_ZO` | max abs z 3.64 (bound 4.5); `sum z^2` = 0.3, 0.2, 0.2, 5.3, 21.1, 31.2, 81.9 against `d` = 1..64 |
| G4a trust-region step equality | max rel 8.88e-16 (bound 1e-13) |
| G4b K-space `E_s[KL] = eps` | max abs 9.99e-16 (bound 1e-12) |
| G5a blurred error gradient vs quadrature | rel 1.6e-15 / 2.0e-14 / 5.9e-13 at `c` = 0.5/1.5/3.0 (bound 1e-12); abs 8.0e-15 at `c = 6` |
| G5b blurred gradient parallel to `V` | 2.95e-16 orthogonal fraction (bound 1e-14) |
| **G6a production bit-identity** | `max abs(new - base) = 0.000e+00` over every actor and critic parameter; identical eval return |
| **G6b probe2 regression** | 34 arrays, **0 differ**, bit-identical to the archived `scripts/probe2_out/WalkerRun_pathwise_fa_pad16_s0_final__std.npz`; `V_e<=0` tripwire 0/256 |
| G6c core vs hand-written reference | max rel 1.53e-15 (bound 1e-14) |
| G6d linearity `g[Q+e] - g[Q] == g[e]` | max rel 6.81e-14 (bound 1e-13) |
| G7 analytic tripwire | 0 of 24 cells outside 4 sigma; worst rel/tol 0.77 |
| **G8 / E0a Nesterov-Spokoiny** | `(M-1) MSE_ZO/\|\|g*\|\|^2` = 2.000, 3.011, 4.980, 9.009, 16.960, 32.974, 65.059 at `d` = 1..64; want `d+1` |
| G10 phase uniformity | min KS p = 0.548 over `omega` in {0.5, 20, 300} |

All blocking gates pass. G6a and G6b together are what make this experiment evidence
about the operators running in the fork rather than about this harness alone: the
refactor that put the estimators in `src/jaxrl/estimators.py` left both the training path
and the shipped probe outputs bit-for-bit unchanged.

System table at commit time (`seed = SEED_ROOT + d`, identity cost):
`rho_closed` 0.45-0.69, `cond_H` 1.0-16.2, zero rejections at every `d`.

**Note on G8.** The spec predicted `MSE_ZO/MSE_PW` grows linearly in `d`. It does not,
and cannot: `MSE_PW = 4 sigma^2 tr(H^2)/M` is nonzero and itself grows with `d`, so that
ratio tends to a constant. The classical Nesterov-Spokoiny factor is `MSE_ZO` relative to
the exact gradient's zero variance, which is the `(d+1)/(M-1)` form asserted above. The
ratio is reported descriptively only.

---

## 9. If Rule B is refuted

The `sqrt(d)` term is not doing the work claimed for it, the dimensional prediction comes
out of the abstract, and the paper reframes around `omega` alone -- while stating, from
Sec. 5.1, that `omega` itself is not well-posed for a full-rank error field until the
norm in Claim 4 is fixed. That statement is a contribution whether Rule B confirms or
refutes.

---

## Addendum A1 (2026-09-03): post-hoc d = 6 arm

**Appended after all Sec. 5 results were seen; nothing above this line is altered.**

**Why.** The Sec. 5.4 cross-term flag, as reported in `reports/lqr_crossover.md` Sec. 5,
trips at `d in {1, 2, 4}` and clears at `d >= 8`. Walker's action dimension is 6, which
sits inside that gap and was never run. A single `d = 6` rank-one arm is added to
resolve it. It is a post-hoc addition and is labelled as such wherever it appears.

**Design, identical to the registered rank-one arm.** `kind = rank1`, `M = 32`,
`eps_frac = 0.05`, `normalize = unit_H`, `cost = identity`, the registered
`sigma` x `omega` grid (20 x 34, common log ratio `300^(1/19)`), `n_states = 64`,
`n_batch = 40`, `r_batch = 250` (`N = 10^4`), seeds by the registered convention
(`SEED_ROOT + d` for the system, `+ 2000 + d` for states, `+ 3000 + d` for the kernel
key), CPU only. Command: `python scripts/lqr_crossover/sweep.py --d 6`.

**What is computed and reported for d = 6, fixed here.**

- The cross term with **both denominators**: `|2 Cov| / Var_e` and `|2 Cov| / MSE_total`.
- Under **both** evaluation rules: the registered Sec. 5.4 rule (maximum **anywhere on
  the swept grid**, `|2 Cov| > 0.25 Var_e`) **and** the report's at-crossover variant.
- Each **unrestricted** and **restricted to `sigma >= 0.1`** (the DMC-reachable band,
  Sec. 6).
- `c*(6)` by the same estimator as Sec. 5.4, for completeness only.

**What d = 6 does not do.** It does not enter the registered primary `d`-set
`{2, 4, 8, 16, 32, 64}` of Sec. 5.4, does not change Rule A or Rule B, and is not
evidence about DMC. Its single purpose is to say on which side of the cross-term flag
`d = 6` falls under each rule.
