# Preregistration: the uniform empirical-mean term in the implemented E-step

**Status.** Written after the Step 0 source and artifact audit
(`reports/ubar_code_trace.md`) and **before** any checkpoint-level numerical result.
Step 0 produced only source facts, a dimension/`eta` inventory, and the algebraic
verification suite; no `R`, no `Rho`, no batch quantity has been computed.

**Analysis Git SHA at preregistration:** `1d4b5de25b8bc6e3fd4a532bbcaa1299be569a4c`.

**Frozen-checkpoint, read-only.** No training run is launched, no checkpoint is
modified or written, no training log is altered. No return-level causal conclusion is
drawn under any outcome.

---

## 1. What Step 0 already settled, and what it forces here

1. `v` is the **standardized mean-score direction**, not the full implemented actor
   update: `Sigma` is not fixed and shares every trunk parameter with the mean
   (`src/networks/jax_models.py:522-524`). The audit's naming condition is not met.
2. The exact score space is the whitened mean coordinate and **coincides with
   `u_i_fit`**; the tanh Jacobian is `mu`-independent and drops out of the mean score
   (verified to `1.1e-15`). The **implementation-space** decomposition is primary; the
   raw-Gaussian one is a manuscript-level diagnostic. The two differ **only** through
   the `+-(1-1e-4)` clip.
3. The `d/M` analytic baseline applies to `ubar_raw` **only**.
4. The M perturbations are **shared across minibatches within an epoch** (one key per
   epoch, `reppo.py:1093-1098`, not split in the minibatch scan at `613`/`1085`).
5. `eta` is **measured** for `weighted_mle` and does not exist for `pathwise`.
6. Available estimator-visible `d` is **{4, 6, 16, 22, 29}**. **There is no d=21
   checkpoint**; `HumanoidRun` is absent from this repository. No condition is
   relabelled to supply one.
7. The WML objective is wrapped by the KL clip (`reppo.py:850-855`), which **replaces**
   it above `kl_bound = 0.1`. Every result below describes the WML **mean score**, and
   the actor loss only on ungated states.

## 2. Checkpoint list

All `_final` exports in the confirmatory seed namespace, plus the padded Walker set.
Exploratory seeds `201` (g1 pathwise) and `234` (hopper weighted_mle) are **excluded**;
they are outside the registered confirmatory namespace.

| condition | task | arm | d | pad | seeds | n |
|---|---|---|--:|--:|---|--:|
| C1 | HopperHop | pathwise | 4 | 0 | 101–108 | 8 |
| C2 | HopperHop | weighted_mle | 4 | 0 | 101–108 | 8 |
| C3 | WalkerRun | pathwise | 6 | 0 | 101–108 | 8 |
| C4 | WalkerRun | weighted_mle | 6 | 0 | 101–108 | 8 |
| C5 | LeapCubeRotateZAxis | pathwise | 16 | 0 | 101–108 | 8 |
| C6 | LeapCubeRotateZAxis | weighted_mle | 16 | 0 | 101–108 | 8 |
| C7 | WalkerRun | pathwise | 22 | 16 | 0–4 | 5 |
| C8 | WalkerRun | weighted_mle | 22 | 16 | 0–4 | 5 |
| C9 | G1JoystickFlatTerrain | pathwise | 29 | 0 | 101–108 | 8 |
| C10 | G1JoystickFlatTerrain | weighted_mle | 29 | 0 | 101–108 | 8 |

**74 checkpoints.** `M = 32` and `eps_e = 0.5` for all; `min_std = 0.1` effective for all.

**Missing-data policy.** If a checkpoint fails to load or produces non-finite values it
is reported as missing in the per-checkpoint table and excluded from that condition's
aggregate, with the count stated. No checkpoint is silently substituted for another.

## 3. State source and probe construction

Matched to the existing q-spread probe, `scripts/q_spread_from_ckpt.py:34-49`:

* `Harness(ckpt_dir, n_states)` (`scripts/critic_fidelity/common.py:52-83`), which
  applies `ActionPad` when `meta.action_pad > 0` and freezes the checkpoint normalizer;
* `h.reset(rk)`; then `burn_in = 50` steps of the **stochastic** policy with
  `jnp.clip(pi.sample(...), -ACTION_CLIP, ACTION_CLIP)`, `ACTION_CLIP = 0.999`
  (`q_spread_from_ckpt.py:39-42`);
* states taken at the single post-burn-in timestep, as in that script;
* `pi = h.ck.actor.actor(h.na(obs))`, `cobs = h.nc(obs)`.

**Deliberate deviation, stated in advance:** `q_spread_from_ckpt.py:50-53` uses
`alpha_entropy` as the softmax temperature. That is not the trainer's temperature
(`reppo.py:741` uses `eta`). This audit uses **`eta`**. The state source and
preprocessing are matched; the temperature is not, and must not be.

* **N = 2048 states** per checkpoint (2048 parallel envs).
* **M = 32** action samples per state, drawn as `a = tanh(mu + sigma*u)`,
  `u ~ N(0, I_d)`, then clipped to `+-(1-1e-4)` — the trainer's construction
  (`reppo.py:711-712`), with `u` drawn explicitly so `u_raw` is exact.
* **16 independent action clouds** on a fixed subset of **256 states**, to separate
  state heterogeneity from action-cloud variability.
* **Analysis RNG seed: 20260902**, a probe-only root disjoint from every training seed.
  Per-checkpoint keys are folded from `blake2b(purpose|checkpoint)`.
* All reductions and diagnostics in **float64** (`jax_enable_x64` on).

## 4. `eta` policy

| arm | source | column |
|---|---|---|
| `weighted_mle` | `eta_param` in `actor.npz`, i.e. the value the run ended with | `eta_measured` |
| `pathwise` | solved from the registered E-step dual `g(eta) = eta*eps_e + eta*mean_j log mean_i exp(q_ji/eta)` at `eps_e = 0.5`, as a **single batch-shared scalar**, per `reppo.py:168-181` and Amendment A answer 4 | `eta_recomputed_counterfactual` |

The two are never merged into one column and never plotted with the same marker. No
"actual training eta" is claimed for arm A.

## 5. Metrics

Per state and action cloud, on `u_fit` (primary) and `u_raw` (diagnostic), `eps = 1e-12`:

```
v      = sum_i w_i u_i           ubar = (1/M) sum_i u_i        c = v - ubar
m_hat  = (1/M) sum_i (Q_i - Qbar) u_i
norm_ratio      = ||c|| / (||m_hat/eta|| + eps)
cosine_linear   = cos(c, m_hat/eta)
residual_linear = ||c - m_hat/eta|| / (||c|| + eps)
R_exact  = ||ubar|| / (||c|| + eps)          R_linear = ||ubar|| / (||m_hat/eta|| + eps)
cos_ubar_c = cos(ubar, c)                     cos_v_c  = cos(v, c)
cross_fraction  = 2<ubar,c> / (||ubar||^2 + ||c||^2 + eps)
direction_change = 1 - cos(v, c)
logit_spread = sd_i(Q_i/eta)   ESS = 1/sum_i w_i^2   w_max   clip_rate
```

Checkpoint-level energy ratios (**primary magnitude comparison**):

```
R2_exact  = sqrt( sum ||ubar||^2 / sum ||c||^2 )
R2_linear = sqrt( sum ||ubar||^2 / sum ||m_hat/eta||^2 )
```

Small-denominator states are **not discarded**; their frequency is reported and they
are retained in the energy ratios. `eps` prevents division by zero only.

## 6. Preregistered thresholds

**P1 — first-order identity.** Adequate for a task-arm condition only if all three
checkpoint-level medians hold: `norm_ratio in [0.8, 1.25]`, `cosine_linear >= 0.95`,
`residual_linear <= 0.25`. All three are reported whether or not they hold. **If P1
fails, every primary ratio uses measured `c`; the linearized ratio stays a diagnostic.**
P1 is stratified by `logit_spread`. The previously observed spread range is prior
calibration, not a prospective hypothesis, and agreement with it is not a test.

**P2 — raw-Gaussian sanity.** `RMS ||ubar_raw||` vs `sqrt(d/M)`; empirical mean vs
`sqrt(2/M) Gamma((d+1)/2)/Gamma(d/2)`; empirical median vs `chi_d^{-1}(0.5)/sqrt(M)`.
The median is **not** compared to `sqrt(d/M)`. For `ubar_fit` the moments are reported
(mean vector, RMS, covariance, clip rate, `ubar_fit - ubar_raw`) but **not** compared to
`d/M`. A nonzero conditional mean from clipping is called **transformation-induced
bias**, not Gaussian sampling noise.

**P4 — dimension amplification.** Only for a matched within-task comparison. The single
available one is **WalkerRun d=6 (C3/C4) vs d=22 (C7/C8)**, per arm.
`Rho = R2_exact(d=22) / R2_exact(d=6)`.

> **Registered limitation.** The two Walker levels share task, arm, probe construction
> and training configuration, but **not seeds** (101–108 vs 0–4, zero overlap) and
> **not `alpha`** (0.014509915 vs 0.01528). `Rho` therefore cannot be formed per matched
> seed as an ideal design would require. It is computed as a ratio of condition-level
> `R2_exact`, with all seed values reported and an all-pairs (8x5 = 40) ratio
> distribution for an interval. This is an **unpaired** matched-task contrast and is
> labelled as such wherever it appears.

Decision rule, on the median of the all-pairs `Rho` distribution:

* `< 1.3` -> **DIMENSION AMPLIFICATION REFUTED IN THE MATCHED ESTIMATOR PROBE**
* `> 2.0` -> **DIMENSION AMPLIFICATION OBSERVED; UBAR IS A LIVE ESTIMATOR-LEVEL MECHANISM**
* `1.3`–`2.0` -> **DIMENSION AMPLIFICATION UNDECIDED**

Cross-task contrasts (e.g. Walker d=6 vs g1 d=29) are reported descriptively, and the
formal verdict for them is fixed in advance as
**CAUSAL DIMENSION SCALING NOT IDENTIFIABLE FROM THE AVAILABLE CHECKPOINTS.**
A walker-versus-g1 contrast is not called a controlled manipulation of `d`.

**P5 — weak-coupling calibration.** Compare measured `R_linear` against
`1/logit_spread` per state: ratio, and Spearman rank correlation. Treated as a
calibration approximation, never used to replace a measured `R_exact`.

**P6 — batch materiality.** Reported for the mean-output metric and, where available,
an empirical-KL/Fisher metric; Euclidean parameter norms reported with the
parameterization caveat.

* median `R_batch_theta > 3` in the high-dimensional condition -> report in the first
  three lines: **THE UNIFORM EMPIRICAL-MEAN TERM DOMINATES THE FROZEN BATCH ACTOR
  GRADIENT IN THE HIGH-DIMENSIONAL CONDITION.**
* `< 1` -> the centered critic-dependent component is larger in that metric.
* `1`–`3` -> **MATERIAL BUT NON-DOMINANT.**

This is an estimator/update statement, not a return statement.

## 7. Aggregation

The **independent statistical unit is the checkpoint seed.** States and action clouds
are repeated measurements within a checkpoint and are never pooled across seeds as
though independent.

1. Aggregate within each checkpoint: median and IQR of `R_exact`; `R2_exact`; median and
   IQR of `R_linear`; `R2_linear`; median `cos(v,c)`, `cos(ubar,c)`, linearization
   residual; RMS `||ubar_raw||`; the three chi sanity checks; median `logit_spread`;
   median ESS; clip rate.
2. Then summarize across seed-level values: all seed values, median, IQR, and a
   seed-level bootstrap interval (10 000 resamples, `np.random.default_rng(20260902)`),
   plus the exact sign pattern for any paired contrast.

## 8. Batch reconstruction policy

Per checkpoint, reproduce the trainer's actual layout
(`config/experiment_overrides/*`): a pool of `num_envs x num_steps = 1024 x 128 =
131072` states, shuffled and split into `num_mini_batches` minibatches — **B = 2048**
for the DMC tasks (`num_mini_batches: 64`) and **B = 8192** for g1
(`num_mini_batches: 16`); `num_epochs = 8`.

* The **shared-PRNG** pattern of Step 0.4 is reproduced exactly: one `(M, B, d)` draw
  per epoch, reused across every minibatch of that epoch.
* Shuffles are repeated until at least **512 minibatches per task-arm condition** are
  available (1 shuffle for the 8-seed DMC conditions; 2 for the 5-seed padded
  conditions; 4 for g1, whose epoch yields only 16).
* Action-space decompositions are computed on **all** reconstructed minibatches.
* Actor-parameter gradient decompositions are computed on **8 minibatches per
  checkpoint** (>= 64 per condition). This reduction is registered in advance because
  each gradient triple requires three full backward passes over `B x M` log-probs; the
  action-space analysis carries the full 512.
* For raw Gaussian `ubar` only: with independent perturbations,
  `E||mean_batch(ubar_raw)||^2 = d/(M B)`; with the actual shared pattern, the
  prediction is that the standardized `ubar` does **not** average down as independent
  noise across minibatches. Both are checked. Neither is a prediction about `R`.
* `mean_batch(c)` is **not** assumed to scale as `1/sqrt(B)`.

Exact loss decomposition, at `theta_old` (first minibatch of the first epoch, where
`actor_target.params == actor.params` by `reppo.py:603-607`):

```
L_full     = -mean_s sum_i w_si        log pi_theta(a_si_fit | s)
L_uniform  = -mean_s (1/M) sum_i       log pi_theta(a_si_fit | s)
L_centered = -mean_s sum_i (w_si-1/M)  log pi_theta(a_si_fit | s)
```

with `g_full = g_uniform + g_centered` verified numerically per minibatch, reported
separately for the mean head, the scale head, the shared trunk, and the full vector.

**Optimizer-state replay is not available**: no export contains actor optimizer state
(Step 0.5). The optional copied-state single-update replay is therefore **not
performed**, and the gradient decomposition is the exact primary analysis.

## 9. Interpretation commitments

Questions A–G are reported separately and never collapsed into one verdict. Forbidden
under every outcome: that `ubar` caused the g1 return gap; that removing it would
improve return; that WML is worse because of it; any action-dimension return law; any
causal reading of a cross-task `d` contrast; and calling `v` the full WML operator.

If `ubar` is similarly large in all tasks, the report states that it explains a general
difference between the implemented WML operator and the centered manuscript estimator
but **does not** explain why only one task has a detected return gap.
