# Preregistration: Monte-Carlo `Q^pi` oracle feasibility pilot on corrected WalkerRun

**Status: IMMUTABLE once committed.** Written and committed before any oracle value
is computed for either checkpoint. Amendments, if any, are appended below the design
lock as dated entries with the original text left in place, never by editing it.

---

## 0. What this pilot is and is not

**It is** a precision and stability test of a measurement procedure. The question is:

> Can the centered learned-critic error and its RMS error-frequency analogue be
> estimated precisely enough on real learned critics to locate them relative to the
> characteristic boundary `r_RMS = 1`?

**It is not** an explanation of returns. Whether `r_RMS` comes out above or below 1
is irrelevant to the feasibility verdict, and nothing here licenses a statement about
which training arm should win.

The following are already settled elsewhere and are neither re-tested nor
reinterpreted here: the population-level agreement of PW and the centered Gaussian
score estimator; the local natural-gradient equivalence; the planted-error phase
diagram (commit `7663d03`: 240 cells, 12/12 validation tests, 0/240 error-channel
crossover misclassifications, median fitted `r* = 1.0085`, range `[0.974, 1.128]`);
the corrected end-to-end replication (`PW-1 - WML-32 = +135.58` on WalkerRun `d=6`,
`+9.51` on G1JoystickFlatTerrain `d=29`, 8/8 positive in both); the uniform
empirical-mean audit, which does not identify learned-critic `omega`; and Probe 4,
which is mixed and identifies only a centered error component in an inert subspace.

The planted-error result also establishes that the same boundary does **not** predict
actual WML update quality. That finding stands, and this pilot cannot overturn it.

---

## 1. Fixed checkpoints (section 4.1)

Exactly two, fixed now and not replaceable after seeing any oracle output:

| Role | Path | `actor.npz` sha256 (first 16) | `critic.npz` sha256 (first 16) |
|---|---|---|---|
| PW-trained | `exports/WalkerRun_pathwise_fa_s301_final` | `e0cbb41c2bbd92ac` | `b8f22a654a53e8f7` |
| WML-trained | `exports/WalkerRun_weighted_mle_s301_final` | `0e68d7ecc210cb99` | `c56c4d27a3265ef1` |

Both are `WalkerRun`, seed 301, faithful-repair arm, final checkpoint, `d = 6`,
`time_steps = 52297728`, `iteration = 399`. Full checksums are recorded in the report.
No third checkpoint is added. No checkpoint is swapped.

## 2. The estimand (sections 4.3 and 1)

Traced from source in `reports/mc_oracle_code_trace.md` and **verified**, not assumed:

```
Q_soft^pi(s0, a0) = E[ r_0 + sum_{t>=1} gamma^t ( r_t - alpha log pi(a_t|s_t) ) ]
```

* `gamma = 0.99`; `alpha = 0.014509915374219418`, read programmatically as
  `float(actor.temperature())` from each checkpoint (identical for both arms) and
  frozen during training (`update_entropy_lagrangian=false`).
* The externally fixed `a0` carries **no** entropy term. The implementation stores
  `r_tilde_t = r_t - gamma * alpha * log pi(a_{t+1}|s_{t+1})`; the `gamma` inside that
  expression is what shifts the entropy series to start at `j = 1`, and the unrolling
  has been checked for the off-by-one.
* `pi` is the **exported online actor** at `scale = 1.0`. There is no separate critic
  target network in this codebase, and the rollout exploration scale is `1.0`, so no
  behaviour/target ambiguity arises. The TD(`lambda = 0.95`) operator the code
  iterates has this `Q_soft^pi` as its fixed point.
* `Q_phi(s,a)` is the **expectation of the 151-bin HL-Gauss categorical head** over
  `linspace(0, 150, 151)`, at the **online** critic parameters, single head, no
  ensemble or min reduction, with its own observation normalizer, evaluated at the
  **unclipped** `tanh(y)` exactly as `reppo.py:741-744` does.

**No `Q_phi` bootstrap enters the oracle.** The realised discounted soft-reward sum is
used alone. A `gamma^H Q_phi` tail would make `e = Q_phi - Q_oracle` partly a
comparison of `Q_phi` with itself. This is a deliberate departure from
`scripts/critic_fidelity/common.py:soft_return`.

### Centered error and the primary statistic

```
e(s,y)        = Q_phi(s, tanh(y)) - Q_oracle^pi(s, tanh(y))
e_tilde(s,y)  = e(s,y) - E_{u~N(0,I)}[ e(s, mu(s) + Sigma(s)^{1/2} u) ]
```

Centering is essential because the centered value estimator subtracts `Qbar`, so a
per-state constant critic offset never reaches it. The numerator is unchanged by
centering: the subtracted quantity is constant in `y` at fixed `s`.

```
                E_{s,u}[ || Sigma(s)^{1/2} grad_y e(s,y) ||_2^2 ]
r_RMS^2  :=   ----------------------------------------------------
                        d * E_{s,u}[ e_tilde(s,y)^2 ]
```

with `y = mu(s) + Sigma(s)^{1/2} u`. For isotropic `Sigma = sigma^2 I` this is
`r_RMS = sigma * omega_RMS / sqrt(d)`, so the characteristic boundary is `r_RMS = 1`.

**Naming, fixed now.** This quantity is called *the RMS analogue of the Claim-4
error-frequency ratio*. It is **not** "the measured theoretical `omega_infinity`".

### Action clipping: a decision made before any measurement

The actor operator queries the critic at the **unclipped** `tanh(y)`
(`reppo.py:741-744`), while `ClipAction` gives the simulator `clip(a, -0.999, 0.999)`
(`jax_wrappers.py:254-265`). The pilot follows both: `Q_phi` at `tanh(y)`, environment
at its native clip, matching each side of training. `Q^pi(s,a)` is then genuinely
constant in `a` beyond the clip, because the simulator cannot distinguish those
actions, so `e(s,y)` genuinely varies there.

The two policies differ sharply in how much this bites: the PW checkpoint has
pre-squash `sigma ~ 0.518` and the WML checkpoint `sigma ~ 2.203` (final
`pi_sigma_mean_curve`), and the clip corresponds to `|y| > atanh(0.999) = 3.800`. A
non-trivial saturated fraction is therefore expected on the WML side.

**Preregistered secondary:** the clip rate is reported per checkpoint, and `D`, `N_c`
and `r_RMS` are recomputed on the subset of base points where **no** coordinate of the
base action or of any of its finite-difference branches is clipped. This is a
sensitivity analysis, reported alongside the primary, never substituted for it.

## 3. Common state distribution (section 4.2)

`rho_mix = 0.5 rho_PW301 + 0.5 rho_WML301`. States from one arm are **not** called
neutral.

* 32 full MJX states rolled under the corrected PW-301 policy, 32 under WML-301.
* `S = 64` total, `burn_in = 50` stochastic-policy steps from reset with actions
  clipped to `+-0.999`, matching `scripts/analysis/ubar_ratio.py:27` and
  `scripts/analysis/fr_samecritic.py:24`. This is the existing state-distribution
  probe convention and it is compatible with full-state capture.
* RNG root `20260902`, purpose-separated by blake2b fold-in on a string tag.
* The **complete environment state** is saved (every leaf of the wrapped
  `LogEnvState` pytree), not the observation alone, together with the raw observation,
  the source arm, and a sha256 of the archive.
* **Both critics are evaluated on the same 64 raw environment states.** Each
  checkpoint applies its own observation normalizer, policy, `Sigma`, critic, and its
  own continuation policy for `Q^pi`.
* The primary result weights the two 32-state strata equally. Strata are also reported
  separately as *PW-state stratum* and *WML-state stratum*. **No selection between
  strata after seeing results.**

## 4. Action perturbations (section 4.3)

`y_sk = mu(s) + Sigma(s)^{1/2} u_k`, `u_k ~ N(0, I_d)`, `K = 16` per state.

The **same standardized `u_k`** are used for both checkpoints: they are drawn once off
the state index from `fold("u")` and shared. `a_sk = tanh(y_sk)`, no hard clipping
applied by the pilot. The finite-`K` centering factor `1 - 1/K = 15/16` is corrected
explicitly by the `K/(K-1)` factor in section 6.

## 5. Monte-Carlo continuation batches (section 4.4)

Two independent rollout groups per action point, **8 continuation rollouts each**:
`Qhat_A^pi` and `Qhat_B^pi`. Independent root RNG streams
(`fold("roll", tag, group, replicate, phase)`).

**Common random numbers.** The continuation innovations are drawn with shape
`(H, n_base, 1, d)` and broadcast over the 25 finite-difference branches of a base
point, so `y + h_j e_j` and `y - h_j e_j` receive bit-identical standard-normal draws
at every step. Innovations are **independent** across perturbations `k`, states,
replicates and the A/B groups. Sharing across `k` would further reduce variance and is
deliberately *not* used: it is not required by the design, and adding unrequested
variance reduction would make a feasibility verdict optimistic relative to the
registered procedure.

The continuation policy is sampled by hand (`mu + sigma * eps`, then `tanh`) rather
than through distrax, because distrax draws independently per batch element and would
destroy the common random numbers. The hand-written tanh-Normal log density is
validated against distrax (T7b).

The first action is fixed externally; continuation randomness begins at `t = 1`. For
every action point the MC mean, trajectory variance, standard error, rollout count,
horizon and the realised return are stored.

## 6. Horizon (section 4.5)

Primary `H = 500`. **This is truncation, not an exact infinite-horizon oracle**, and
the language used throughout is *finite-horizon Monte-Carlo estimate of soft `Q^pi`*.

`gamma^H` is **not** claimed as the truncation-bias bound. A rigorous bound
`|bias_H| <= gamma^H B / (1 - gamma)` requires a finite per-step bound `B` on the soft
reward. The environment reward is bounded (`r in [0,1]` for WalkerRun), but
`- gamma * alpha * log pi(a|s)` is **unbounded above** for a tanh-Gaussian policy, so
**no global finite `B` exists** and no such bound is stated. The empirical maximum of
`|r_tilde|` is reported instead and labelled empirical, not a proof.

**Horizon doubling is mandatory.** Fixed subset, chosen now: the **first 8 states of
the PW-source stratum and the first 8 of the WML-source stratum** (16 states), and for
each the **first 8 perturbations**, run at `H = 1000` with the base action and the
`c = 0.10` finite-difference branches for all 6 coordinates. The `H = 500` and
`H = 1000` estimates come from the **same rollout**: innovations are always drawn at
length `H_LONG = 1000` and sliced, and the accumulator is frozen at step 500, so the
shorter estimate is literally the prefix of the longer one (T13).

For every state/perturbation, `tail_500_to_1000 = Qhat_1000 - Qhat_500`, centered
within state over `u`. **The assumption that truncation becomes a per-state constant
is tested, not assumed.** Reported: `RMS_u[centered(Q_1000 - Q_500)]` relative to
`RMS_u[e_tilde_500]`, and the gradient-energy numerator at `H = 500` versus `H = 1000`
on the subset.

Episodes that reach the 1000-step time limit stop accumulating (the `alive` mask).
Bank states sit at step 50, so no `H = 500` rollout reaches the limit at all, and an
`H = 1000` rollout reaches it around step 950 where `gamma^950 = 7.0e-5` bounds the
affected contribution by about `0.007` in value units.

## 7. Finite differences (section 4.6)

`d = 6`, so **full coordinate** differences, no Hutchinson approximation.
`h_j(s) = c * sigma_j(s)` at two fixed relative steps `c in {0.10, 0.05}`.

```
z_j := sigma_j(s) * grad_{y_j} e  =  [ e(y + c sigma_j e_j) - e(y - c sigma_j e_j) ] / (2c)
```

computed directly in the whitened form, avoiding a division by `sigma`. Computed
independently from oracle batches A and B, giving `z_A` and `z_B`.

## 8. Noise-debiased squared quantities (section 4.7)

Squared quantities are **never** formed by squaring one noisy estimate.

```
e_A = Q_phi - Qhat_A^pi ,   e_B = Q_phi - Qhat_B^pi
e~_A,sk = e_A,sk - mean_k e_A,sk        (centered within state, per group)
e~_B,sk = e_B,sk - mean_k e_B,sk

D   = K/(K-1) * mean_{s,k}[ e~_A,sk * e~_B,sk ]
N_c = mean_{s,k}[ sum_j z_A,skj * z_B,skj ]
r_RMS,c^2 = N_c / (d * D)
```

`Q_phi` is deterministic, so the A/B noise is independent by construction and the
cross-products are unbiased for the squared signals.

**The square root is taken only when `D > 0` and `N_c > 0`.** If either cross-product
is negative because MC noise dominates, the result is reported as

```
UNRESOLVED - ORACLE PRECISION INSUFFICIENT
```

A negative estimate is **not** clamped to zero and carried forward.

## 9. Bootstrap and statistical unit (section 4.9)

**The state is the outer independent unit.** Perturbations and rollout replicates are
repeated measurements. `64 * 16` perturbations are **not** treated as 1024 independent
states.

Paired stratified hierarchical bootstrap, 10,000 replicates,
`np.random.default_rng(20260902)`:

* resample 32 PW-source states with replacement within their stratum;
* resample 32 WML-source states with replacement within their stratum;
* equal 50/50 stratum weighting preserved;
* resample the 16 perturbations with replacement within each selected state;
* **the same bootstrap indices are applied to the PW and WML checkpoints**, so the
  difference is paired.

The estimator, including the `K/(K-1)` factor, is computed identically in the point
estimate and in every replicate. Reported 95% percentile intervals: `D`, `N_0.10`,
`N_0.05`, `r_RMS,0.10`, `r_RMS,0.05`, the paired `PW - WML` difference in `r_RMS`, and
each of the two state-source strata separately.

A state-only bootstrap (perturbations kept attached to their state) is reported as a
robustness check, not as the primary.

## 10. Validation before the pilot (section 4 / phase 5)

`scripts/analysis/test_mc_oracle.py` must pass in full. T1 and T1b are the **phase-2
gate**: if exact full-state restoration and branching cannot be established, the pilot
STOPS and `Q^pi` is not approximated from observations alone.

T1 full MJX state clone/replay determinism over 100 steps (states, rewards,
observations, done flags, actions, accumulators). T1b branching: same state, different
first action, same continuation innovations. T2 forced `a0` equal to the policy's own
sample reproduces a plain rollout. T3 soft return equals a hand-unrolled short
trajectory. T4 no entropy term on the externally fixed `a0`. T5 discount indexing
matches the critic target at every prefix length. T6 normalizers reproduce the saved
statistics. T7 `Q_phi` matches the existing `load_ckpt` evaluator. T7b the
hand-written tanh-Normal log density matches distrax. T8/T8b common random numbers.
T9 groups A and B independent. T10/T10b finite differences exact on a known quadratic
and `O(c^2)` on a cubic. T11 cross-product debiasing recovers `E[e~^2]` where the
naive square is biased up. T12 the `K/(K-1)` factor. T13 the shorter horizon is the
exact prefix of the longer rollout. T14 the state bank round-trips exactly.

**No test is weakened after inspecting the pilot outcome.**

## 11. Feasibility decision rule (section 4 / phase 7) — fixed now

For **each** of the two checkpoints:

| | Criterion | Threshold |
|---|---|---|
| A | Centered error power | `D > 0` and bootstrap 95% lower bound on `D > 0` |
| B | Gradient energy | `N_0.10 > 0`, `N_0.05 > 0`, both bootstrap lower bounds `> 0` |
| C | Step-size stability | `r_RMS,0.05 / r_RMS,0.10 in [0.8, 1.25]` **and** `N_0.05 / N_0.10 in [0.8, 1.25]` |
| D | Horizon stability | on the fixed subset, `centered RMS(Q_1000 - Q_500) < 0.25 * centered RMS(e_500)` **and** the `H=1000 / H=500` gradient-energy ratio at `c = 0.10` in `[0.8, 1.25]` |
| E | Interval precision | `width(95% CI for r_RMS) <= 0.60 * point estimate` |

**PASS TO SCALE** only if **both** checkpoints satisfy A-E, in which case the verdict
is `ORACLE PILOT FEASIBLE - SCALE TO ALL 16 CORRECTED WALKER CHECKPOINTS`, and that
scaling job is **not** launched in this task.

Otherwise `ORACLE PILOT NOT YET PRECISE ENOUGH TO SCALE`, naming which of MC rollout
variance, finite-difference bias, horizon truncation, state heterogeneity,
perturbation heterogeneity, a near-zero centered-error denominator, or an
implementation/restoration issue dominates, and estimating without running it whether
the remedy is more rollouts, more states, more perturbations, a longer horizon, or a
different finite-difference step.

If C fails, finite-difference bias is reported as **unresolved**. If D fails,
truncation remains material, and it is **not** said that centering solved truncation.

Placement is reported as three-valued, never forced binary:
`BELOW BOUNDARY` / `NEAR BOUNDARY - INTERVAL CROSSES 1` / `ABOVE BOUNDARY`. A
checkpoint genuinely near `r_RMS = 1` may have an interval crossing 1 and still PASS,
provided E holds.

**The pilot specification is not changed after seeing results and rerun under a new
definition. A second pilot would require a new preregistration.**

## 12. Secondary diagnostics, reported but never primary (section 8)

Uncentered error RMS (showing how much a statewise offset would have changed the
frequency estimate); statewise mean critic error `mean_u e(s,y)`; oracle MC variance
and standard error by state and action; critic-error amplitude by state-source
stratum; `sigma` statistics (per-dimension mean, RMS, min/max, anisotropy); the scalar
`omega_RMS = sqrt(E||grad_y e||^2 / E e_tilde^2)`; and the PW-trained versus
WML-trained critic comparison on the identical mixed state bank.

A scalar `sigma * omega` point is **not** forced onto the isotropic
`(sigma*omega, sqrt(d))` plane if the learned policy is materially anisotropic. The
primary placement is `r_RMS` against 1.

## 13. Prospective power notes (recorded, not used to tune anything)

Every design parameter above is fixed by the task specification; nothing here is
tuned. These quantities were read from the checkpoint metadata before launch and are
recorded so the feasibility verdict can be read against a prior expectation:

* PW-301 final `pi_sigma_mean = 0.5185`; WML-301 final `pi_sigma_mean = 2.2030`.
* WML-301 final `q_spread = 5.975` (the training-time spread of `Q` over the 32 E-step
  action samples). The PW arm logs `q_spread = 0` because it draws no E-step sample
  set, so no comparable prior is available on the PW side.
* Final evaluation returns 911.15 (PW) and 775.09 (WML).

**Registered expectation of the analyst, to be scored afterwards.** I expect the WML
checkpoint to satisfy A and B comfortably and the PW checkpoint to be the binding
case, because its much narrower policy should produce a small within-state spread of
`Q` relative to the MC noise of an 8-rollout mean. I expect criterion E to be the one
that fails, if any does, and I expect it to fail on the PW checkpoint first. I also
expect criterion C to pass and criterion D to pass comfortably given
`gamma^500 = 6.6e-3`.

## 14. Interpretation rules — binding

**Allowed:** that the MC oracle estimates centered learned-critic error with
sufficient or insufficient precision; that the RMS error-frequency analogue lies
below, near, or above `r_RMS = 1`; that the PW- and WML-trained critics differ in
their measured centered error spectrum on the common WalkerRun state distribution;
that the learned WalkerRun critic lies in a regime consistent or inconsistent with the
characteristic error-channel boundary.

**Forbidden:** that Claim 4 predicts which training arm should win; that `r_RMS > 1`
means WML should have higher return; that the oracle explains the WalkerRun return
gap; that the oracle explains the g1 return gap; that this measures `omega_infinity`
exactly; that centering eliminates truncation bias; that PW states are neutral; that
`Q^pi` is exact while `H` is finite and MC estimated; and any action-dimension trend.

---

## Design lock

Everything above is fixed at the commit that adds this file. Checkpoints, state
count, burn-in, RNG root, perturbation count, rollout counts, group structure,
horizons, the horizon-doubling subset, finite-difference steps, the estimator
formulae, the bootstrap scheme, the decision thresholds and the interpretation rules
are all settled before any oracle value exists.
