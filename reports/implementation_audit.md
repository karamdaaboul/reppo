# Implementation audit: does the code match the claimed algorithm?

**Audited SHA `b48a6ed8ac0b3cce58a4077bfa86272923066567`**, branch `estep-study`,
tracked tree clean, 0 untracked. `src/` is identical to `918f82c`; the later
commits add reports and analysis only. No source was modified during this audit.

**Verdict: 2 — MINOR DOCUMENTATION CORRECTIONS REQUIRED — RESULTS REMAIN USABLE.**

---

## A. Intended mathematical algorithm

**E-step.** Non-parametric target `q(a|s) ∝ π_old(a|s)·exp(Q_φ(s,a)/η)`. With
`a_i ~ π_old` the proposal cancels the `π_old` factor, so the self-normalised
weight carries **no** `1/π_old` importance term:

```
w_i(s) = softmax_i( Q_φ(s,a_i)/η ),   i = 1..M
```

**Eta dual.** `g(η) = η·ε_E + η·E_s[ log (1/M) Σ_i exp(Q_i/η) ]`, minimised over
`η ∈ [1e-4, 10]`.

**WML M-step.** `max_θ E_s Σ_i w_i(s) log π_θ(a_i|s)`.

**PW M-step.** `max_θ E_s[ Q_φ(s, tanh(μ_θ + σ_θ⊙u)) − α log π_θ ]`, one
reparameterised sample.

**M-step constraint.** `KL(π_old ‖ π_θ)`, forward, on the pre-squash diagonal
Gaussians, summed over action dimensions, evaluated per state, bounded by
`kl_bound = 0.1`.

## B. E-step implementation

| Quantity | Source | Actual | Match |
|---|---|---|---|
| candidates `a_i` | `reppo.py:879-891` | drawn from `actor_target_model.actor`, `sample_shape=(n_estep,)`, independent across M | yes |
| pre-tanh retained | `:879` | `distribution.sample_and_log_prob` then `bijector.forward_and_log_det` | yes |
| stop-gradient | `:939-950` | `w_i` fully detached; `η` detached inside the weights | yes |
| M | `:812-816` | `estep_num_samples` = 32 for WML, hard-coded 16 for PW | yes, asymmetry recorded in G |
| `Q_i` | `:944-947` | single critic target network; no ensemble, no min, no rescaling before the exponential | yes |
| weights | `estep_weights`, `:197-217` | `softmax(q_i/η, axis=0)`, over M, per state | yes |
| objective | `:985` | `−Σ_i w_i·logp_i`; `w` sums to 1 so this is a weighted mean | yes |

Numerically verified: weights equal `softmax(Q/η)` to 2e-31, sum to 1 per state,
total mass equals the batch size, i.e. normalised over M and not across the batch.

**S0.**

## C. Eta dual

`η = clip(softplus(η_param), 1e-4, 10.0)` (`jax_models.py:451-455`). Global, not
state-specific. Detached where used in the weights; gradient reaches it only
through `eta_loss`, which is added to the same actor loss and stepped by the same
optimizer.

`eta_dual_loss` (`:219-234`) computes `η·ε_E + mean_s(η·lse_s + qmax_s)`, which is
algebraically `η·ε_E + η·E_s log mean_i exp(Q/η)`. Verified against an independent
implementation at η ∈ {0.02, 0.1, 1.0} to 1e-8.

`ε_E = 0.5` constrains the state-averaged KL in nats over the full action vector,
not normalised by M or by `d`. These units differ from the M-step bound, which is
per state.

**S0.**

## D. M-step trust region — it is not a trust region

Direction established numerically: `gaussian_kl_diag(a,b) = KL(a‖b)` (12.598,
against 17.210 for the reverse), and the sampled `kl` with `a_i ~ π_old` converges
to `KL(old‖new)` (MC 12.600 over 4e5 samples). **Forward KL, `KL(π_old‖π_θ)`**
(`reppo.py:920`).

The tanh log-determinant appears in both densities and cancels exactly (verified to
1.8e-15), so **the constraint is imposed in pre-tanh Gaussian policy space.**

Reduction: summed over action dimensions — verified, `KL(d=12)/KL(d=3) = 4.0`
exactly — meaned over the M samples, and kept per state.

Enforcement (`reppo.py:1080-1086`, `actor_kl_clip_mode="clipped"`):

```python
actor_loss = jnp.where(kl < cfg.kl_bound,
                       objective,
                       kl * stop_gradient(lagrangian) * cfg.reduce_kl)
```

This is a hard switch, not a constrained optimisation and not a penalty added to
the objective. Below the bound there is no KL term at all; at or above it the
improvement objective is replaced outright by a KL-reduction term. `λ = exp(param)`
is unbounded above and is updated by `lagrangian_loss = −λ·sg(kl − 0.1)`.

Runtime behaviour, from existing logs:

| task | arm | objective used | objective replaced | median sampled KL | λ_eff |
|---|---|---|---|---|---|
| walker | PW-1 | 54.9% | 45.1% | 0.0864 | 0.026 |
| walker | WML-32 | 55.3% | 44.7% | 0.0874 | 0.346 |
| g1 | PW-1 | 52.4% | 47.6% | 0.0934 | 0.0057 |
| g1 | WML-32 | 47.2% | 52.8% | 0.1161 | 0.235 |

LEAP, from `train/fr_gate_operator` (recorded under `log_faithful_diag=true`):
objective used 50.93% (PW) and 47.26% (WML); objective replaced 49.07% and
52.74%. An earlier draft stated the LEAP fire fraction was unobserved; that was
wrong — it read `train/kl_gate_fire`, which sits behind `log_cov_diag=false`,
rather than `train/fr_gate_operator`, which LEAP does record.

## E. Covariance handling

| | PW | WML |
|---|---|---|
| mean updated | yes | yes |
| `log_std` updated | yes | yes |
| covariance | diagonal | diagonal |
| separate `KL_μ` / `KL_Σ` | no | no (`mstep_decoupled=false`) |

`σ = exp(log_std) + min_std`, `min_std = 0.1`, an additive floor. In the primary
arms `freeze_sigma = null` and `sqrt_rho = 1.0`, both exact no-ops taking Python
branches. Nothing is frozen in the Walker, G1 or LEAP primaries; the
covariance-freeze and `sqrt_rho` machinery is ablation-only and inactive. **S0.**

## F. Hyperparameters

| | Walker | G1 | LEAP |
|---|---|---|---|
| M | 32 | 32 | 32 |
| `eps_e` | 0.5 | 0.5 | 0.5 |
| `kl_bound` | 0.1 | 0.1 | 0.1 |
| alpha, frozen | 0.014509912580251694 | 0.00020752247655764222 | 0.000782382907345891 |
| gamma | 0.99 | 0.99 | 0.99 |
| lambda | 0.95 | 0.95 | 0.95 |
| vmin / vmax | 0 / 150 | −10 / 10 | −10 / 60 |
| minibatches x epochs | 128 x 4 | 128 x 4 | 128 x 4 |
| learning rate | 3e-4 | 3e-4 | 3e-4 |
| episode length | 1000 | 1000 | 500 |
| action dimension | 6 | 29 | 16 |

Alpha: `temperature_log_param = log(ent_start)`, `α = exp(param)`. With
`update_entropy_lagrangian=false` the `target_entropy_loss` term is never added to
the loss, so alpha is frozen exactly at `ent_start`. It is used consistently in the
critic target `r̃ = r − γ·α·log π(a'|s')` (`:609`) and in the actor objective
`α·log π − Q` (`:1040-1042`); no double counting and no omission. Target clipping
is `hl_gauss`'s `clip(inp, vmin, vmax)` (`utils.py:44-45`), applied to the training
target only and identically for both arms within a task. **S0.**

## G. PW / WML fairness

| Component | PW | WML | Same | Classification |
|---|---|---|---|---|
| critic, states, architecture, optimizer, lr, minibatches, epochs, rollout budget, critic and actor update counts, alpha, gamma, lambda, vmin/vmax, policy init | identical | identical | yes | controlled/common |
| estimator/operator | reparameterised SAC, 1 sample | weighted MLE, M=32 | no | necessary by definition |
| candidate M | not applicable | 32 | no | necessary by definition |
| trust-region mechanism, bound, direction | same | same | yes | controlled/common |
| KL Monte-Carlo sample count | 16 | 32 | no | see K.3 |
| λ_eff when the gate fires | 0.006–0.026 | 0.235–0.346 | no | endogenous dual |
| realised gate fire rate | 45.1% / 47.6% | 44.7% / 52.8% | no | endogenous |

## H. Runtime diagnostics

WML ESS (run-wide mean, M=32): Walker 20.49, G1 20.22, LEAP 18.47, i.e. ESS/M
between 0.58 and 0.64. Eta: Walker ~0.039, G1 ~0.0095, LEAP ~0.0093. Achieved KL
sits at 0.086–0.122 against a 0.1 bound in every task and arm, which is what a
threshold produces.

## I. Numerical unit checks

Ten substantive checks pass against the committed source: WML softmax (2e-31),
per-state normalisation, normalisation over M rather than batch, the eta dual at
three values of eta (1e-8), KL direction, the sampled KL converging to
`KL(old‖new)`, the summed-over-dimensions reduction, tanh Jacobian cancellation
(1.8e-15), and ESS equal to M under uniform weights.

## J. Paper versus code

No manuscript is present in this repository, so wording could not be audited
directly. Corrections required wherever these phrases appear:

| Phrase | Verdict |
|---|---|
| "KL-constrained M-step" / "trust region" | wording too strong; it is a hard switch that replaces the objective above the bound |
| "same trust region for both arms" | wording too strong; same mechanism, but realised fire rates and λ differ by arm |
| "E_s KL ≤ 0.1" | mathematical mismatch; the bound is applied per state, not to the state average |
| "MPO E-step", "eps_E" | matches implementation |
| "mean update" / "covariance update" | matches implementation |
| a bare "0.1 KL bound" across tasks | wording too strong; summed over `d`, so 0.1 is about five times tighter per dimension at d=29 than at d=6 |

## K. Discrepancies

**K.1 — S1.** The M-step is a per-state hard switch, not a trust region or a
Lagrangian constraint. The code is internally coherent; the terminology is wrong.

**K.2 — S1.** `kl_bound = 0.1` is per state and summed over action dimensions, so
it is not comparable across d = 6, 16, 29 without saying so.

**K.3 — S2, with reasons.** The gate thresholds a 16-sample KL estimate in PW and a
32-sample estimate in WML. This was treated as a candidate S3 — a noisier gate
statistic could flip the gate independently of real policy movement — and tested:
the repository logs both the sampled and the exact analytic KL, and the bias is
−1e-5 to +7e-5 against a 0.1 bound in all four task/arm cells. The gate statistic
is essentially unbiased in both arms, so the hypothesis did not survive. Residual
unknown, stated rather than assumed: per-update KL variance is not logged, so
differential gate flipping cannot be excluded from these artifacts.

**K.4 — S2.** Approximately 45–53% of per-state actor-loss terms were routed
from the policy-improvement objective to KL reduction under the KL gate, at
different rates per arm, and λ_eff differs by roughly an order of magnitude
between arms. Shapes verified: `kl`, both objectives and the `jnp.where` output
are all `(B,)`, and `jnp.mean(actor_loss)` reduces to a scalar afterwards, so a
single optimizer step mixes improvement gradients from gated-open states with
KL-reduction gradients from gated-closed states. Whether any entire optimizer
step received no improvement contribution is NOT determinable from the stored
logs, which record only the per-minibatch mean of the gate indicator. Both are endogenous responses of a shared mechanism to each
operator's own KL trajectory rather than external confounds, but they mean the arms
do not receive equal amounts of improvement signal, which no fairness claim may
assume.

**K.5 — resolved, was S1, now S0.** LEAP's gate fire fraction IS observed, via
`train/fr_gate_operator`: 49.07% (PW) and 52.74% (WML) of per-state terms routed
to KL reduction. The earlier claim that it was unobserved was an error of mine,
reading the wrong logged field.

Nothing was found at S3 or S4. The E-step, the eta dual, the weighting, the KL
direction, the coordinate system, covariance handling, alpha and value support are
all verified correct.

## L. Verdict

**2 — MINOR DOCUMENTATION CORRECTIONS REQUIRED — RESULTS REMAIN USABLE.**

The implemented algorithm is mathematically sound and internally coherent, and the
two arms differ only where intended plus the endogenous consequences of that
difference. What is wrong is the description: the M-step enforcement is not a trust
region, the bound is per state and dimension-summed, and a large fraction of actor
updates carry no improvement signal. Those facts change how the results must be
worded, not whether they stand.

---

## Addendum A1 — 2026-09-04T16:30:53+02:00 — targeted follow-ups, and three corrections

Append-only below this line. Three statements **above** were corrected in place at
the same commit and are itemised in A1.6.

### A1.1 `actor_target_model` is a hard copy, and `polyak` is dead config

```
source: src/jaxrl/reppo.py:693-697
update: actor_target.params <- actor.params      (assignment, no interpolation)
interval: once per learn_step, i.e. once per training iteration
coefficient: none
```

`polyak` is declared at `reppo.py:79` and **appears nowhere else in `src/`**. It
does not participate in the actor-target update.

Order within one iteration (`train_step`, `:1395-1405`): `collect_rollout` ->
lambda-returns -> **hard copy actor_target <- actor** (`:693`) -> `update`, i.e.
`num_epochs x num_mini_batches = 512` inner minibatch updates. `actor_target_model`
is merged once at `:698`, outside `update`, and the four downstream references
(`:806, :824, :864, :995`) are reads only. It is therefore **frozen across all 512
inner updates**.

1. Policy generating the WML candidates? **Yes** (`:864`).
2. Reference distribution for the M-step KL gate? **Yes**, the same object.
3. Equal to the policy before the current actor optimisation phase? **Yes, exactly** --
   copied immediately before it and never refreshed during it.
4. Necessarily the policy that generated the rollout? **Yes**, in this call order:
   the rollout precedes the copy and the actor is not updated in between. Condition:
   the rollout applies an exploration scale, which equals 1.0 here because
   `exploration_noise_min = exploration_noise_max = 1.0`.

**`pi_old` is exact notation**, with one precision worth stating in the paper: it
denotes the **iteration-start** policy, held fixed for all 512 inner updates, not a
Polyak/EMA target and not the previous minibatch's policy.

### A1.2 Statewise gate-flip rate: not identifiable

Every logged diagnostic is shape **`(21, 1)`** -- one value per evaluation, already
reduced over states, over the 512 inner updates per iteration, and over the
iterations between evaluations. `fr_gate_operator` is `_gate_open.mean()` over the
minibatch; `fr_kl_analytic_med` is a median over states; the sampled-minus-analytic
fields are a median and a mean over states. **No statewise pairing survives.**

```
STATEWISE GATE-FLIP RATE NOT IDENTIFIABLE FROM STORED LOGS
```

Not computable: statewise disagreement count, disagreement fraction, gate
false-positive and false-negative rates, `std(delta_KL)`, median absolute error,
95th-percentile absolute error. These need per-state values that were never stored.

What is recoverable is the paired error already reduced over states, averaged over
each run (median over the 8 seeds):

| task | arm | KL samples | gate open | gate KL-only | delta_KL med-over-states | delta_KL mean-over-states |
|---|---|---|---|---|---|---|
| walker | PW | 16 | 0.5490 | 0.4510 | −4.70e-05 | −1.03e-05 |
| walker | WML | 32 | 0.5530 | 0.4470 | −2.83e-04 | −4.16e-06 |
| g1 | PW | 16 | 0.5240 | 0.4760 | −1.64e-04 | +3.38e-05 |
| g1 | WML | 32 | 0.4722 | 0.5278 | −1.18e-04 | +6.64e-05 |
| leap | PW | 16 | 0.5093 | 0.4907 | −1.71e-04 | −5.12e-05 |
| leap | WML | 32 | 0.4726 | 0.5274 | −1.68e-04 | +1.53e-05 |

These bound the **bias** of the gate statistic at <= 0.3% of the 0.1 bound in every
cell. They say nothing about its **variance**, which is what would flip a
per-state gate decision, and the variance was not stored.

**Classification of the 16-vs-32 asymmetry: `UNRESOLVED`.** Not `NEGLIGIBLE`: the
quantity that would establish that -- the statewise flip rate -- cannot be computed
from these artifacts. This supersedes the S2 reasoning in K.3, which rested on the
bias check alone.

### A1.3 Eta never saturates

`eta = clip(softplus(eta_param), 1e-4, 10)`. Across all 24 WML runs:

| task | eta min | eta max | fraction at 1e-4 | fraction at 10 |
|---|---|---|---|---|
| walker | 0.015395 | 0.132002 | 0.0000 | 0.0000 |
| g1 | 0.003154 | 0.055624 | 0.0000 | 0.0000 |
| leap | 0.004292 | 0.017125 | 0.0000 | 0.0000 |

**No boundary saturation occurs in any run.** The smallest value seen anywhere is
0.00315, a factor of 31 above the lower clip; the largest is 0.132, a factor of 76
below the upper clip. Granularity caveat: `train/eta` is `(21,1)`, already averaged
over the 512 inner updates, so a brief per-update excursion to a bound could be
averaged away.

### A1.4 Realised finite-M particle E-step KL: not recoverable

```
REALISED PARTICLE E-STEP KL NOT RECOVERABLE FROM STORED LOGS
```

`D_particle = sum_i w_i log(M w_i)` needs the per-candidate weights `w_i`. Only
`ESS = 1/sum_i w_i^2` and `w_max` were stored, both already reduced, and neither
determines `D_particle`. The conclusion is therefore restricted to what was
established: **the MPO eta-dual objective is mathematically and numerically correct
(Section C), and no claim of runtime satisfaction of the E-step constraint is made.**

### A1.5 ESS trajectories, final 20% by training iteration

Final window defined by iteration (> 319 of 399), which is 5 of the 21 logged
points. `M = 32`.

| task | full mean | /M | last-20% mean | /M | final | min | min/M | seeds with last20 < full |
|---|---|---|---|---|---|---|---|---|
| walker | 20.491 | 0.6403 | 20.793 | 0.6498 | 20.524 | 16.046 | 0.5014 | 4/8 |
| g1 | 20.221 | 0.6319 | 19.421 | 0.6069 | 19.481 | 17.814 | 0.5567 | **8/8** |
| leap | 18.524 | 0.5789 | 19.898 | 0.6218 | 20.118 | 14.770 | 0.4616 | 0/8 |

* Does ESS stay near its full-run level? **Yes**, everywhere.
* Systematic late concentration? **Only on G1**, in 8/8 seeds, and small: −0.31 to
  −1.10 ESS units, about −4% relative. Walker is mixed (4/8) and LEAP rises in 8/8.
* Consistent across seeds? Within each task, yes; the direction differs by task.
* Does any task approach one or few effective samples late? **No.** The smallest
  value anywhere, over all 24 runs and all logged points, is 14.77 of 32.

**The statement "the near-one E-step concentration observed in the legacy defective
runs is absent from the corrected runs" is NOT supported by these artifacts and
must not be made.** The legacy 1.07/32 figure in `reports/ubar_ratio.md` is a
post-hoc probe on frozen checkpoints, not a training-log aggregate. The two are not
like-for-like, and the size of that gap is demonstrated on the corrected runs
themselves: on the same corrected Walker WML checkpoint the checkpoint-style probe
gives `ESS_Qphi_pilot_K16 = 1.58/16 = 0.099` while the training log gives
`ESS_training_M32 = 20.5/32 = 0.64`, a factor of 6.4 between measurement modes. A
legacy-versus-corrected comparison requires the same measurement mode on both.

The three ESS quantities remain strictly separate:
`ESS_training_M32`, `ESS_Qphi_pilot_K16`, `ESS_MC_oracle_K16`.

### A1.6 Corrections made in place above

1. **K.4** -- "45–53% of updates" replaced by "45–53% of per-state actor-loss terms
   routed ... under the KL gate", with the shape trace. The original wording implied
   whole optimizer steps received no improvement signal; the routing is per state and
   the mean over states is taken afterwards, so a step mixes both gradient types.
2. **Section D** -- the LEAP note claiming the fire fraction was unobserved was
   wrong; it read `train/kl_gate_fire` (behind `log_cov_diag=false`) instead of
   `train/fr_gate_operator`, which LEAP records.
3. **K.5** -- resolved from S1 to S0 for the same reason.

K.3's classification changes from **S2 to UNRESOLVED** per A1.2. No other severity
changes; nothing reaches S3 or S4, and the Section L verdict is unchanged.
