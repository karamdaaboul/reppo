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

For LEAP `kl_gate_fire` is gated behind `log_cov_diag=false`, so the LEAP fire
fraction is **not directly observed**; mean `train/kl` is 0.1097 (PW) and 0.1219
(WML), above the bound in both arms.

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

**K.4 — S2.** The gate routes 45–53% of updates away from the improvement
objective, at different rates per arm, and λ_eff differs by roughly an order of
magnitude between arms. Both are endogenous responses of a shared mechanism to each
operator's own KL trajectory rather than external confounds, but they mean the arms
do not receive equal amounts of improvement signal, which no fairness claim may
assume.

**K.5 — S1.** LEAP's gate fire fraction is unobserved because `kl_gate_fire` sits
behind `log_cov_diag=false`. Runtime constraint satisfaction is therefore not
directly observed for LEAP; the KL level is observed and exceeds the bound in both
arms.

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
