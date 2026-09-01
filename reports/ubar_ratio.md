**DIMENSION AMPLIFICATION REFUTED IN THE MATCHED ESTIMATOR PROBE** (WalkerRun d=6 -> d=22, the only within-task manipulation of estimator-visible d that exists here; all-pairs Rho median 1.035 arm A, 0.485 arm B, both < 1.3; unpaired in seed).

**THE UNIFORM EMPIRICAL-MEAN TERM DOES NOT DOMINATE THE FROZEN BATCH ACTOR GRADIENT IN THE HIGH-DIMENSIONAL CONDITION.** At d=29 the centered critic-dependent component is the larger one (R_batch_theta = 0.085 arm A, 0.452 arm B in the mean-output and empirical-KL metrics, both < 1). The term is MATERIAL BUT NON-DOMINANT only on WalkerRun d=6 (1.90 / 2.63) and d=22 arm A (1.45).

**Clipping changes the decomposition materially and must not be folded into "Gaussian sampling noise."** The +-(1-1e-4) clip binds on up to 20.7% of samples (walker arm B), inflates RMS ||ubar|| by up to 3.0x over the raw-Gaussian value, and gives ubar_fit a systematic mean vector up to 18x the raw one. The tanh transform itself is innocent: it drops out of the mean score exactly.

---

# The uniform empirical-mean term in the implemented E-step

Frozen-checkpoint estimator audit. Read-only with respect to training: no run was
launched, no checkpoint modified or written, no training log altered. No return-level
causal conclusion is drawn anywhere in this document.

## 1. Provenance

| | |
|---|---|
| Branch | `estep-study` |
| HEAD at analysis | `5912170` (preregistration); measurement jobs recorded `591217087aa8e96c09e2ab281d09b61fdb81f376` |
| `git status --short` at Step 0 | *empty* — 0 modified, 0 untracked |
| Python / JAX / flax / numpy / scipy | 3.12.14 / 0.5.2 / 0.10.6 / 2.5.1 / 1.18.0, `jax_enable_x64` on in all analysis code |
| Accelerator | 1x NVIDIA H100 (SLURM `c23g` 94 GB and `c25g` 80 GB); Step 0 checks on CPU |
| Checkpoint root | `/rwthfs/rz/cluster/hpcwork/qzi10910/reppo_runs/exports` — 228 dirs, 76 `_final` |
| SLURM jobs | Step 2: `3432865`, `3434212`. Step 3: `3434280`, `3434719`, `3434800`, `3434802`, `3436699` |

HEAD is not `b92c89f`; that commit is an ancestor of `1d4b5de`, on which the
preregistration `5912170` was built.

## 2. Source trace

Full detail in `reports/ubar_code_trace.md`. The load-bearing findings:

* **The objective is exactly the stop-gradiented weighted likelihood.**
  `src/jaxrl/reppo.py:754` `objective = -jnp.sum(w_i * logp_theta_i, axis=0)` with
  `w_i` and the `eta` inside it both `stop_gradient` (`751-753`). So the proposed
  decomposition **is** the complete mean-score decomposition — no gradient flows
  through the weights, the values or the samples.
* **But the objective is not the actor loss.** It is wrapped by
  `jnp.where(kl < cfg.kl_bound, objective, kl*sg(lagrangian)*reduce_kl)`
  (`850-855`), which **replaces** it above `kl_bound = 0.1`. Everything below
  therefore describes the WML **mean score** in full, and the actor loss only on
  ungated states.
* **`v` is the standardized mean-score direction, not the full implemented actor
  update.** The audit's naming condition is not met: `Sigma` is not fixed and shares
  every trunk parameter with the mean (`src/networks/jax_models.py:522-524`,
  `loc, log_std = jnp.split(actor_module(obs), 2)`).
* **The exact score space coincides with `u_fit`.** For a tanh-transformed diagonal
  Gaussian the Jacobian term is `mu`-independent, so
  `Sigma^{1/2} d/dmu log pi(a_i) = (atanh(a_i) - mu)/sigma = u_i_fit` exactly. The
  tanh transformation drops out of the mean score entirely; only the **clip** makes
  `u_fit` differ from `u_raw`. Verified by autodiff to `1.1e-15`.
* **The known clip/log-prob mismatch does not touch the WML mean score.** It is
  between `old_pi_act_log_prob` (returned at the unclipped sample, `709-711`) and
  `logp_theta_i` (at the clipped action, `717`); the objective uses only the latter.
  The former enters the KL and the gate.
* **Weights**: `jax.nn.softmax(q_i/eta, axis=0)` (`146-165`) — per-state over the
  sample axis, `M = 32` in all 76 exports, no advantage normalisation, rank transform,
  top-k or weight clipping. `eta` is the E-step dual, not `alpha`.
* **PRNG**: one key per epoch (`1093-1098`), **not** split in the minibatch scan
  (`613`, `1085`). The `(M, B, d)` standard-normal array is therefore bit-identical
  across every minibatch of an epoch, and independent across states within one.
* **`theta == theta_old`** exactly at the first minibatch of the first epoch
  (`603-607`). Actor optimizer state is **absent from every export**, so the optional
  copied-state replay was not performed.

## 3. Dimension and checkpoint audit

**There is no d=21 checkpoint in this repository.** Estimator-visible dimensions
present: **{4, 6, 16, 22, 29}**, with `action_dim == actor head/2 == critic action
input` for all 76 exports. The apparent inconsistency resolves as: WalkerRun physical
`d=6`; padded WalkerRun `d = 6+16 = 22`; g1 (`G1JoystickFlatTerrain`) `d=29`;
**`d=21` is `HumanoidRun`, which is absent**. No condition was relabelled to supply it.

74 checkpoints analysed (10 conditions). Seeds `201` and `234` were excluded as
outside the confirmatory namespace; that is the entire missing-data record — **0
checkpoints failed to load and 0 produced non-finite values.**

`eta` is **measured** (serialized `eta_param`) for all 38 `weighted_mle` checkpoints,
median 0.0089, range [0.00104, 0.0556]. `pathwise` checkpoints have **no
`eta_param` at all** — `with_eta=False`, no E-step temperature ever existed — so for
the crossed E-step evaluation `eta` is **RECOMPUTED COUNTERFACTUAL**, solved from the
registered dual at `eps_e=0.5` as a single batch-shared scalar: median 0.0724, range
[0.00202, 0.520]. The two are kept in separate columns throughout and drawn with
separate markers.

## 4. Preregistration

`docs/prereg_ubar_ratio.md`, committed at **`5912170`** after Step 0 and **before any
checkpoint-level number**. It fixes the checkpoint list, verified `d` and `M`, the
state source, burn-in, `N`, cloud count, analysis RNG seed, `eta` policy per arm,
aggregation rules, metrics, all thresholds, the missing-data policy, the
transformation branch and the batch reconstruction policy.

## 5. P1 — is the first-order approximation valid? **No, in 9 of 10 conditions.**

Gate: `norm_ratio in [0.8,1.25]` **and** `cosine_linear >= 0.95` **and**
`residual_linear <= 0.25`, on checkpoint-level medians.

| condition | arm | d | n | norm_ratio | cosine_linear | residual_linear | logit spread | P1 |
|---|---|--:|--:|--:|--:|--:|--:|---|
| hopper | A | 4 | 8 | 0.981 | 0.9986 | **0.0894** | 0.293 | **PASS** |
| hopper | B | 4 | 8 | 0.713 | 0.959 | 0.689 | 1.877 | FAIL |
| walker | A | 6 | 8 | 0.938 | 0.986 | 0.253 | 0.685 | FAIL |
| walker | B | 6 | 8 | 0.174 | 0.741 | **5.073** | 15.10 | FAIL |
| leap | A | 16 | 8 | 0.937 | 0.949 | 0.383 | 0.909 | FAIL |
| leap | B | 16 | 8 | 0.172 | 0.559 | **5.364** | 19.67 | FAIL |
| walker-pad16 | A | 22 | 5 | 0.957 | 0.964 | 0.320 | 0.695 | FAIL |
| walker-pad16 | B | 22 | 5 | 0.587 | 0.680 | 1.298 | 4.186 | FAIL |
| g1 | A | 29 | 8 | 0.904 | 0.940 | 0.418 | 0.879 | FAIL |
| g1 | B | 29 | 8 | 0.655 | 0.714 | 1.080 | 3.197 | FAIL |

**1/10 pass.** As preregistered, **every primary ratio below uses measured `c`**, and
the linearized ratio is retained only as a diagnostic.

The stratification by logit spread is clean and is the mechanism (figure panel d):
the residual rises essentially linearly in the spread and crosses the 0.25 threshold
at a spread of roughly 0.6. The only condition with a spread below that (hopper A,
0.293) is the only one that passes. `walker B` and `leap B` sit at spreads of 15 and
20, where the expansion is wrong by a factor of 5.

## 6. P2 — raw-Gaussian sanity checks, and the clipping bias

**The probe is correct.** Across all 10 conditions, comparing `RMS ||ubar_raw||` to
`sqrt(d/M)`, the empirical mean to `sqrt(2/M) Gamma((d+1)/2)/Gamma(d/2)`, and the
empirical median to `chi_d^{-1}(0.5)/sqrt(M)`:

**worst `|ratio - 1|` over all 30 comparisons = 0.0053.**

The median is compared only to the chi median, never to `sqrt(d/M)`.

`ubar_fit` is **not** compared to `d/M`. Its moments, and the transformation-induced
bias:

| condition | arm | d | clip rate | RMS ‖ubar_fit‖ | RMS ‖ubar_raw‖ | ‖mean ubar_fit‖ | ‖mean ubar_raw‖ | RMS‖ubar_fit−ubar_raw‖ |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| hopper | A | 4 | 1.40% | 0.597 | 0.355 | 0.0841 | 0.0052 | 0.460 |
| hopper | B | 4 | 4.34% | 0.385 | 0.354 | 0.0385 | 0.0079 | 0.179 |
| walker | A | 6 | 1.76% | 0.768 | 0.435 | 0.0805 | 0.0096 | 0.646 |
| **walker** | **B** | 6 | **20.7%** | **1.292** | 0.433 | **0.171** | 0.0093 | **1.261** |
| leap | A | 16 | 1.35% | 0.711 | 0.708 | 0.0809 | 0.0162 | 0.120 |
| leap | B | 16 | 5.21% | 0.700 | 0.708 | 0.0966 | 0.0155 | 0.172 |
| walker-pad16 | A | 22 | 0.42% | 1.028 | 0.829 | 0.107 | 0.0169 | 0.614 |
| walker-pad16 | B | 22 | 4.93% | 1.018 | 0.829 | 0.0992 | 0.0222 | 0.660 |
| g1 | A | 29 | 0.10% | 0.978 | 0.953 | 0.0243 | 0.0198 | 0.219 |
| g1 | B | 29 | 0.94% | 1.386 | 0.952 | **0.205** | 0.0217 | 1.025 |

This is **transformation-induced bias, not Gaussian sampling noise.** The clipped
residual has a systematic conditional mean an order of magnitude above the raw one
(walker B: 0.171 vs 0.0093; g1 B: 0.205 vs 0.0217), and up to 3.0x the RMS norm. It is
concentrated in arm B, whose policies are much wider (walker B median sigma 3.39 vs
arm A 0.475), pushing `|tanh y|` against the clip.

## 7. Per-checkpoint state-level results (P3)

Primary magnitude comparison is the checkpoint-level energy ratio
`R2_exact = sqrt( sum ||ubar||^2 / sum ||c||^2 )`. Seed is the unit; the interval is a
10 000-resample seed-level bootstrap.

| condition | arm | d | n | **R2_exact** | 95% CI | median R_exact (IQR) | cos(v,c) | cos(ubar,c) | ESS | spread |
|---|---|--:|--:|--:|---|--:|--:|--:|--:|--:|
| hopper | A | 4 | 8 | **0.656** | [0.366, 0.904] | 1.502 (0.48) | 0.697 | −0.015 | 29.6 | 0.293 |
| hopper | B | 4 | 8 | **0.299** | [0.256, 0.373] | 0.344 (0.07) | 0.959 | −0.029 | 9.24 | 1.877 |
| walker | A | 6 | 8 | **0.712** | [0.583, 0.885] | 0.711 (0.21) | 0.853 | −0.008 | 22.6 | 0.685 |
| walker | B | 6 | 8 | **0.662** | [0.506, 1.376] | 0.224 (0.63) | 0.981 | +0.013 | **1.36** | 15.10 |
| leap | A | 16 | 8 | **0.577** | [0.544, 0.775] | 0.806 (0.12) | 0.790 | −0.005 | 17.3 | 0.909 |
| leap | B | 16 | 8 | **0.201** | [0.187, 0.232] | 0.203 (0.02) | 0.981 | −0.011 | **1.07** | 19.67 |
| walker-pad16 | A | 22 | 5 | **0.687** | [0.601, 1.266] | 1.053 (0.16) | 0.708 | −0.003 | 21.8 | 0.695 |
| walker-pad16 | B | 22 | 5 | **0.302** | [0.275, 0.631] | 0.304 (0.11) | 0.959 | −0.005 | 2.94 | 4.186 |
| g1 | A | 29 | 8 | **0.615** | [0.599, 0.623] | 0.921 (0.02) | 0.747 | −0.014 | 19.4 | 0.879 |
| g1 | B | 29 | 8 | **0.404** | [0.353, 0.614] | 0.357 (0.11) | 0.943 | −0.010 | 3.70 | 3.197 |

Tail frequencies (medians over seeds):

| condition | arm | R>0.5 | R>1 | R>3 | cos(v,c)<0.9 | cos(v,c)<0.5 |
|---|---|--:|--:|--:|--:|--:|
| hopper | A | 0.669 | 0.567 | 0.398 | 0.645 | 0.395 |
| hopper | B | 0.319 | 0.105 | 0.017 | 0.279 | 0.053 |
| walker | A | 0.638 | 0.359 | 0.079 | 0.592 | 0.183 |
| walker | B | 0.125 | 0.039 | 0.012 | 0.126 | 0.023 |
| leap | A | 0.762 | 0.353 | 0.023 | 0.756 | 0.136 |
| leap | B | 0.011 | 0.000 | 0.000 | 0.014 | 0.000 |
| walker-pad16 | A | 0.808 | 0.527 | 0.085 | 0.807 | 0.257 |
| walker-pad16 | B | 0.150 | 0.030 | 0.002 | 0.161 | 0.010 |
| g1 | A | 0.832 | 0.431 | 0.010 | 0.835 | 0.130 |
| g1 | B | 0.286 | 0.060 | 0.006 | 0.302 | 0.022 |

Two structural observations:

1. **`ubar` and `c` are essentially orthogonal** in every condition — `cos(ubar,c)`
   ranges over [−0.029, +0.013] and `cross_fraction` is negligible. So they neither
   reinforce nor cancel; `||v||^2 ~ ||ubar||^2 + ||c||^2`, and `R` translates directly
   into a direction change with no interference term. This is why `cos(v,c)` tracks
   `1/sqrt(1+R^2)` closely.
2. **In `leap B` and `walker B` the E-step has collapsed onto a single sample.**
   Median ESS is **1.07** (max weight 0.968) and **1.36** (0.848) out of `M = 32`.
   In those two conditions the "weighted MLE" is operationally a hard argmax over the
   32 draws, which is also why the first-order expansion fails there by a factor of 5.
   This is a property of the trained `eta`, not of the probe.

## 8. Task-arm seed summaries

All 74 seed-level `R2_exact` values are in `reports/artifacts/ubar_per_checkpoint.csv`.
The condition medians are the table in Sec. 7. There is **no monotone trend in `d`** in
either arm:

* arm A: d=4 → 0.656, d=6 → 0.712, d=16 → 0.577, d=22 → 0.687, d=29 → 0.615
* arm B: d=4 → 0.299, d=6 → 0.662, d=16 → 0.201, d=22 → 0.302, d=29 → 0.404

The consistent structure is by **arm**, not by dimension: arm A sits at 0.58–0.71
everywhere, arm B at 0.20–0.66, and arm A exceeds arm B in every task. That is
explained by `eta`: arm B's trained temperatures are 8x smaller than arm A's
recomputed counterfactual ones, giving sharper weights, a larger `c`, and hence a
smaller ratio.

## 9. Matched dimension contrast (P4)

The only within-task manipulation of estimator-visible `d` available is
**WalkerRun d=6 vs d=22 (padded)**, per arm.

| arm | d=6 `R2_exact` (seeds 101–108) | d=22 `R2_exact` (seeds 0–4) | all-pairs Rho median | IQR | verdict |
|---|---|---|--:|---|---|
| A | 0.583, 0.566, 0.781, 0.604, 0.885, 2.276, 0.662, 0.761 | 0.687, 0.807, 0.646, 1.266, 0.601 | **1.035** | [0.818, 1.248] | **REFUTED** |
| B | 1.376, 0.692, 1.110, 1.693, 0.325, 0.559, 0.632, 0.506 | 0.396, 0.275, 0.290, 0.631, 0.302 | **0.485** | [0.284, 0.727] | **REFUTED** |

> **DIMENSION AMPLIFICATION REFUTED IN THE MATCHED ESTIMATOR PROBE.**

`ubar` does not show the required relative amplification in this matched probe; in
arm B its relative importance *falls* by half as `d` goes from 6 to 22. This does
**not** mean `ubar` cannot affect returns, and no such claim is made.

**Registered limitations of this contrast, restated:** the two levels share task, arm,
probe construction and training configuration, but **not seeds** (101–108 vs 0–4, zero
overlap, so `Rho` cannot be formed per matched seed) and **not `alpha`**
(0.014509915 vs 0.01528). It is an *unpaired* matched-task contrast. Additionally the
padded checkpoints fail the padding preregistration's own policy-width gate, so they
are used here for the estimator-level comparison only, exactly as instructed.

For the cross-task contrasts the verdict was fixed before execution and stands:
**CAUSAL DIMENSION SCALING NOT IDENTIFIABLE FROM THE AVAILABLE CHECKPOINTS.** A
walker-versus-g1 contrast is not a controlled manipulation of `d`.

## 10. Weak-coupling calibration (P5)

`R_linear ~ 1/logit_spread` under weak coupling. Per state:

| condition | arm | median R_linear | median 1/spread | ratio | Spearman |
|---|---|--:|--:|--:|--:|
| hopper | A | 0.928 | 1.930 | 0.481 | +0.911 |
| hopper | B | 0.222 | 0.622 | 0.357 | +0.948 |
| walker | A | 0.628 | 1.447 | 0.434 | +0.906 |
| walker | B | 0.044 | 0.069 | 0.645 | +0.845 |
| leap | A | 0.763 | 1.123 | 0.679 | +0.885 |
| leap | B | 0.038 | 0.052 | 0.728 | +0.945 |
| walker-pad16 | A | 1.026 | 1.474 | 0.696 | +0.924 |
| walker-pad16 | B | 0.147 | 0.199 | 0.741 | +0.941 |
| g1 | A | 0.786 | 1.102 | 0.713 | +0.946 |
| g1 | B | 0.222 | 0.282 | 0.785 | +0.906 |

The approximation **ranks** states very well (Spearman +0.85 to +0.95) but is
**biased high in level by 25–65%**. It is used as a calibration approximation only;
no primary `R_exact` was estimated from summary logit statistics.

## 11. Trainer-faithful batch analysis and the actor-gradient decomposition (P6)

**5376 reconstructed minibatches**, `>= 512` per task-arm condition as registered
(640 for the padded conditions), plus **1056 gradient triples**. Batch shapes are the
trainer's own: `B = 2048` for the DMC tasks (`num_mini_batches: 64`) and `B = 8192`
for g1 (`16`), over a `1024 x 128 = 131072` pool, with one `(M, B, d)` draw per epoch
reused across every minibatch of that epoch.

### 11a. Action-space

| condition | arm | d | B | `R_batch_action` | ‖mean ubar‖ | ‖mean c‖ | ‖mean ubar_raw‖ | `sqrt(d/(MB))` | ratio |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| hopper | A | 4 | 2048 | 0.277 | 0.0739 | 0.195 | 0.00640 | 0.00781 | 0.819 |
| hopper | B | 4 | 2048 | 0.211 | 0.0382 | 0.189 | 0.00782 | 0.00781 | 1.001 |
| walker | A | 6 | 2048 | **1.562** | 0.160 | 0.106 | 0.00797 | 0.00957 | 0.833 |
| walker | B | 6 | 2048 | **1.328** | 0.223 | 0.196 | 0.00986 | 0.00957 | 1.031 |
| leap | A | 16 | 2048 | 0.210 | 0.0685 | 0.386 | 0.0150 | 0.0156 | 0.957 |
| leap | B | 16 | 2048 | 0.110 | 0.0927 | 0.780 | 0.0158 | 0.0156 | 1.010 |
| walker-pad16 | A | 22 | 2048 | **1.182** | 0.148 | 0.116 | 0.0171 | 0.0183 | 0.934 |
| walker-pad16 | B | 22 | 2048 | 0.438 | 0.108 | 0.206 | 0.0188 | 0.0183 | 1.025 |
| g1 | A | 29 | 8192 | 0.107 | 0.0157 | 0.143 | 0.0104 | 0.0105 | 0.988 |
| g1 | B | 29 | 8192 | 0.312 | 0.0949 | 0.293 | 0.0107 | 0.0105 | 1.013 |

The last column tests the registered raw-Gaussian prediction
`E||mean_batch(ubar_raw)||^2 = d/(MB)`: measured/predicted is **0.82–1.03**, so
**within** a minibatch the perturbations do behave as independent across states. The
sharing established in Step 0.4 is **across** minibatches, not within one — the same
`ubar` array recurs in every minibatch of an epoch, so it does not average down across
minibatches, but each minibatch's own `ubar` is already suppressed by `1/sqrt(B)`.
`mean_batch(c)` was not assumed to scale as `1/sqrt(B)` and does not: it is 10–70x
larger than `mean_batch(ubar_raw)`, because the statewise `c` directions carry a
systematic mean that survives averaging.

### 11b. Actor-parameter gradients

`g_full = g_uniform + g_centered` verified per minibatch: **max relative residual
`4.35e-08` over all 1056 triples** on the full actor vector (float32 parameters;
`3.6e-16` when the same parameters are cast to float64), and `1.8e-08` in the induced
mean-output. The KL metric was cross-checked against the repository's own
`decoupled_kls` at finite step size: median `|ratio − 1| = 1.22e-4`, max `3.7e-2`.
The max is the expected finite-difference error of a second-order quantity at step
`t = 1e-4` on the minibatches with the largest gradients, not a disagreement about
the metric; the JVP value is the one used throughout.

| condition | arm | d | n | `R_meanout` | `R_KL` | `R_euclid` (full) | mean head | scale head | trunk | cos(g_full,g_cent) | cos(g_unif,g_cent) |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| hopper | A | 4 | 64 | 0.671 | 1.086 | 1.373 | 0.608 | 2.439 | 1.732 | 0.663 | +0.131 |
| hopper | B | 4 | 64 | 0.180 | 0.235 | 0.328 | 0.232 | 0.501 | 0.297 | 0.946 | −0.224 |
| **walker** | **A** | 6 | 64 | **1.899** | 1.893 | 1.991 | 1.824 | 3.554 | 1.540 | 0.502 | +0.049 |
| **walker** | **B** | 6 | 64 | **2.630** | 2.648 | 3.134 | 3.024 | 2.909 | 3.100 | 0.290 | +0.010 |
| leap | A | 16 | 64 | 0.088 | 0.089 | 0.284 | 0.058 | 0.827 | 0.374 | 0.949 | −0.075 |
| leap | B | 16 | 64 | 0.146 | 0.418 | 0.582 | 0.054 | 0.926 | 0.698 | 0.847 | −0.131 |
| **walker-pad16** | **A** | 22 | 80 | **1.453** | 1.479 | 1.780 | 1.356 | 4.511 | 1.510 | 0.533 | +0.038 |
| walker-pad16 | B | 22 | 80 | 0.297 | 0.541 | 0.671 | 0.343 | 1.225 | 0.788 | 0.808 | −0.062 |
| g1 | A | 29 | 256 | 0.085 | 0.085 | 0.152 | 0.094 | 0.638 | 0.199 | 0.989 | +0.025 |
| g1 | B | 29 | 256 | 0.452 | 0.457 | 0.621 | 0.253 | 2.204 | 0.709 | 0.860 | +0.052 |

Preregistered verdicts, in the mean-output and empirical-KL metrics (the invariant
ones; Euclidean is parameterization dependent and is reported alongside for that
reason):

* **d=29 (highest): `R = 0.085` (A) and `0.452` (B), both `< 1` — the centered
  critic-dependent component is larger.** The dominance headline is **not** triggered.
* **MATERIAL BUT NON-DOMINANT** on walker d=6 (1.90 A, 2.63 B) and walker-pad16 d=22
  arm A (1.45).
* **Centered component larger** in the remaining six conditions.

Note the scale head is systematically the component where `ubar` matters most
(`R_scalehead` up to 4.51), which is expected: the uniform term is a pure
mean-estimation displacement and the scale head sees it only through the shared trunk.

## 12. Clipping and transformation analysis

Summarised in Sec. 6. Separating the two sources as instructed:

* **Raw Gaussian finite-M noise**: `ubar_raw` matches all three chi benchmarks to
  within 0.53%, has a negligible mean vector (0.005–0.022), and averages down within a
  batch exactly as `sqrt(d/(MB))` predicts. This component is fully characterised and
  benign.
* **Clipping/transformation-induced bias**: the `+-(1-1e-4)` clip binds on 0.10%–20.7%
  of samples, inflates `RMS ||ubar||` by up to 3.0x, and gives `ubar_fit` a systematic
  conditional mean up to 18x the raw one. On walker arm B — the worst case, with 20.7%
  clipping and median policy width 3.39 — clipping, not Gaussian sampling, produces the
  majority of the uniform term. **The tanh transform itself contributes nothing**: it
  drops out of the mean score exactly (verified to `1.1e-15`), so the clip is the sole
  transformation-level source.

## 13. Answers to A–G, kept separate

**A. Is the algebraic uniform component present in the implementation?** **Yes.**
`v = ubar + c` holds identically (`2.7e-15`), and the whitened mean score equals
`sum_i w_i u_i_fit` exactly. The weights are fully stop-gradiented, so this is the
complete mean-score decomposition, not an approximation to it.

**B. Is `c ~= m_hat/eta` accurate at operating logits?** **No, in 9 of 10 conditions.**
It holds only where the logit spread is below ~0.6 (hopper arm A). At the spreads
actually seen in arm B on walker and leap (15–20) the expansion is wrong by a factor
of 5.

**C. Is the uniform component large relative to the exact centered component?**
At the **state** level it is comparable: `R2_exact` is 0.20–0.71, i.e. the uniform term
carries 20–71% of the centered term's energy, and exceeds it on a substantial minority
of states in arm A (`R>1` on 35–57% of states). At the **batch** level it is generally
smaller (Sec. 11).

**D. Does it alter the direction of the complete mean score?** **Yes, substantially,
and more in arm A.** Median `cos(v,c)` is 0.70–0.85 in arm A and 0.94–0.98 in arm B;
`cos(v,c) < 0.9` on 59–84% of states in arm A. Because `ubar ⟂ c`, this is a genuine
rotation with no cancellation.

**E. Does it survive trainer-level batch aggregation?** **Partly.** Within a minibatch
it averages down as independent noise (`sqrt(d/(MB))`, confirmed to 0.82–1.03), so
`R_batch_action` falls to 0.11–0.44 in seven conditions. It does **not** average down
across minibatches, because the same draw recurs in every minibatch of an epoch. It
survives at `R_batch_action > 1` on both walker d=6 conditions and walker-pad16 arm A.

**F. Does it remain material after projection through the actor network?**
**In three of ten conditions.** `R_batch_theta` in the invariant metrics is `> 1` on
walker d=6 (both arms) and walker-pad16 arm A, and `< 1` elsewhere, including both
d=29 conditions.

**G. Does its relative magnitude increase in a matched higher-dimensional condition?**
**No.** `Rho = 1.035` (arm A) and `0.485` (arm B), both below the 1.3 threshold:
**REFUTED** in the matched estimator probe.

## 14. What this does and does not license

The uniform term **is present in the implemented mean score** and is absent from the
centered value-only estimator the manuscript analyses. That is a real and general
difference between the implemented WML operator and the manuscript's `g_ZO`.

It is **common across tasks, not specific to the one task with a detected return gap.**
Ordered by `R2_exact`, the largest values are on hopper A, walker A and walker-pad16 A
— not on g1. At d=29, where the return gap was detected, the term is among the *least*
important at batch-gradient level (`R = 0.085` arm A, `0.452` arm B). **It therefore
explains a general difference between the actual WML update and the centered
manuscript estimator, and does not explain why only one task has a detected return
gap.**

Not claimed anywhere, under any outcome: that `ubar` caused the g1 return gap; that
removing it would improve return; that WML is worse because of it; any
action-dimension return law; any causal reading of a cross-task `d` contrast; or that
`v` is the full WML operator.

## 15. Limitations and every failed check

1. **P1 fails in 9/10 conditions** — the first-order approximation is not usable at
   operating logits, so the manuscript-level linearised story does not describe these
   checkpoints. Reported as a failure, not worked around.
2. **The matched `d` contrast is unpaired in seed** (101–108 vs 0–4) and differs in
   `alpha` (0.014509915 vs 0.01528). `Rho` is an all-pairs ratio, not a paired one.
3. **No d=21 checkpoint exists**; `HumanoidRun` is absent from the repository.
4. **`eta` for arm A is a recomputed counterfactual**, not a trained value. Every
   arm-A number involving weights is conditional on that construction.
5. **The padded checkpoints fail the padding preregistration's policy-width gate**
   (`sigma_pad/sigma_real` 1.74–5.58 across all 10). Used for the estimator-level
   comparison only.
6. **In leap B and walker B the E-step has collapsed** (ESS 1.07 and 1.36 of M=32).
   The "weighted MLE" is effectively a hard argmax there, which limits how much those
   two conditions say about a weighted update.
7. **The WML objective is KL-gated** (`reppo.py:850-855`), so these results describe
   the mean score in full and the actor loss only on ungated states.
8. **No optimizer state exists in any export**, so the optional copied-state
   single-update replay was not performed. The gradient decomposition is the exact
   primary analysis; no Adam-step sensitivity result is offered.
9. **Gradient triples are 8 per checkpoint per shuffle**, a reduction registered in
   advance (64–256 per condition against 512 minibatches for the action-space analysis).
10. `train/actor_loss`-style per-state recovery was **not** used anywhere; all
    quantities come from fresh frozen-checkpoint evaluations.

### Bugs found in the analysis, and how they were resolved

* **T3 initially failed** (`4.19e-08`). Cause: the trained parameters are float32, so
  the gradient identity can only hold to float32 epsilon. Casting the same parameters
  to float64 gives `6.8e-16`, proving precision rather than a real discrepancy. Both
  are now reported.
* **T6 initially read 3–4% below every Gaussian benchmark.** This looked like a
  standardization error and was not: it was Monte-Carlo noise from evaluating a pure
  PRNG check on only 128 states (relative SE ~1.3%). At 200 000 draws all three
  benchmarks match to 0.06%. An intermediate hypothesis — that an `arctanh` round-trip
  through float32 was biasing `u_raw` — was **wrong**; the explicit standard-normal
  draw was kept anyway because it removes a genuine precision hazard as `|a| -> 1`.
* **The first Step 3 array requested a comma-separated partition list**, which the
  account is not permitted to use (`Access/permission denied`); it was split into two
  single-partition arrays.
* **One g1 arm-A checkpoint initially contributed 16 minibatches instead of 64**,
  because it had been used as the pilot before per-condition shuffle counts were wired
  in, leaving that condition at 464 < 512. It was re-run and the condition now meets
  the registered target.

## 16. Reproduction

```bash
cd ~/repos/reppo
# Step 0: source-level verification (CPU)
JAX_PLATFORMS=cpu ./.venv/bin/python scripts/analysis/test_ubar_decomposition.py \
    exports/WalkerRun_weighted_mle_pad16_s0_final

# checkpoint/dimension inventory and the registered list
./.venv/bin/python scripts/analysis/mk_ubar_ckpt_list.py

# Step 2: 74 frozen checkpoints
sbatch --account=rwth2182 --array=0-73 slurm/ubar.sh
./.venv/bin/python scripts/analysis/ubar_aggregate.py

# Step 3: trainer-faithful batches and actor gradients
sbatch --account=rwth2182 --array=0-73 slurm/ubar_batch.sh
./.venv/bin/python scripts/analysis/ubar_batch_aggregate.py

# figure
./.venv/bin/python scripts/analysis/ubar_figure.py
```

Artifacts: `reports/artifacts/ubar_per_state.csv`, `ubar_per_checkpoint.csv`,
`ubar_batch_action.csv`, `ubar_batch_gradient.csv`, `ubar_checkpoint_audit.csv`,
`ubar_gates.json`, `ubar/*.npz` (74), `ubar_batch/*.npz` (74),
`reports/figures/fig_ubar_ratio.{png,pdf}`.
