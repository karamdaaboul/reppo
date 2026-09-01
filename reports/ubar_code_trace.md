# Source and artifact audit for the uniform empirical-mean term

Step 0 of the frozen-checkpoint estimator audit. Read-only with respect to training:
no run was launched, no checkpoint modified, no training log altered.

## Provenance

| | |
|---|---|
| Branch | `estep-study` |
| HEAD | `1d4b5de25b8bc6e3fd4a532bbcaa1299be569a4c` |
| `git status --short` | *empty* — 0 modified, 0 untracked |
| Remote `origin/estep-study` | `1d4b5de…` (identical) |
| Python / JAX / flax / numpy / scipy | 3.12.14 / 0.5.2 / 0.10.6 / 2.5.1 / 1.18.0 |
| `jax_enable_x64` default | `False`; analysis scripts enable it explicitly |
| Checkpoint root | `/rwthfs/rz/cluster/hpcwork/qzi10910/reppo_runs/exports` (228 dirs, 76 `_final`) |
| Output root | `/rwthfs/rz/cluster/hpcwork/qzi10910/reppo_runs/outputs` |
| Accelerator for this step | CPU (`JAX_PLATFORMS=cpu`); Step 2/3 use one H100 94 GB, SLURM `c23g` |

HEAD is **not** `b92c89f`. That commit is an ancestor; `1d4b5de` is a merge of it with
`c3b5e05`, which had been pushed from another machine.

---

## 0.1 The weighted-MLE actor loss

All references are `src/jaxrl/reppo.py` at `1d4b5de` unless stated.

| Element | Location |
|---|---|
| E-step action sampling | `709-711` `actor_target_model.actor(minibatch.obs).sample_and_log_prob(sample_shape=(n_estep,), seed=key)` |
| Gaussian noise | inside `distrax.Transformed(Normal, Tanh).sample`; the actor is built at `src/networks/jax_models.py:522-526` |
| Action clipping | `712` `old_pi_action = jnp.clip(old_pi_action, -1 + 1e-4, 1 - 1e-4)` |
| Transformation | `distrax.Tanh()`, `src/networks/jax_models.py:525` |
| Critic evaluation point | `748` `q_i = critic_target_model.critic(critic_obs_i, old_pi_action)` — the **clipped, post-tanh** action |
| Old-policy log prob | `709-711` returned at the **unclipped** sample; reduced at `716` |
| New-policy log prob | `717` `logp_theta_i = pi.log_prob(old_pi_action).sum(-1)` — at the **clipped** action |
| E-step logits | `165` `jax.nn.softmax(q_i / eta, axis=0)` |
| Softmax axis | `axis=0` = the sample axis M |
| Temperature | `741` `eta = jnp.squeeze(actor_model.eta())`; `src/networks/jax_models.py:539-543` softplus, clipped to `[1e-4, 10.0]` |
| Stop-gradients | `751-753` weights **and** `eta` inside the weights are both `stop_gradient` |
| Actor objective | `754` `objective = -jnp.sum(w_i * logp_theta_i, axis=0)` |
| Covariance loss | none separate — `log_std` is half of the same head, `jax_models.py:522-524` |
| Minibatches / epochs | `1085-1086` scan over `num_mini_batches`; `1094-1098` scan over `num_epochs` |

The live configuration in every analysed run is `mstep_decoupled: false`, so the
decoupled M-step block at `761-800` is dead code here and the objective is `754`.

**Is the actor loss exactly `L_full = -mean_s sum_i sg(w_i) log pi_theta(a_i_fit|s)`?**

For the objective at line 754, **yes**: the weights are stop-gradiented at `751-753`,
`eta` is stop-gradiented inside them, `q_i` flows only through the detached weights,
and the sampled actions come from the target actor, which carries no gradient.

**But that objective is not the actor loss.** It is wrapped:

```
850: elif cfg.actor_kl_clip_mode == "clipped":
851:     actor_loss = jnp.where(
852:         kl < cfg.kl_bound,
853:         objective,
854:         kl * jax.lax.stop_gradient(lagrangian) * cfg.reduce_kl,
855:     )
```

Elementwise over states, and **above `kl_bound = 0.1` the objective is replaced**, not
penalised. So `L_full` is the actor loss only on the subset of states below the bound.
`reports/g1_kl_readonly_audit.md` establishes that the gated branch provably fires in
64/64 runs at a median 83% of logging points. **The decomposition below therefore
describes the WML mean score itself, and the actor loss only on ungated states.** This
is stated here so no later table is read as covering the whole update.

No entropy term enters the WML objective: `879` adds `target_entropy_loss` only under
`update_entropy_lagrangian`, which is `false` in every frozen-alpha run analysed.

## 0.2 The E-step weights

`estep_weights` (`146-165`) is exactly `jax.nn.softmax(q_i / eta, axis=0)`. Verified:

* normalised over the **sample** axis separately per state — `axis=0` with `q_i` shaped
  `(M, B)`; test T1a gives `max |sum_i w_i - 1| = 5.6e-16`;
* `M = 32` in every analysed condition — `estep_num_samples: 32` in `config/reppo.yaml:67`,
  and `meta.json` reports 32 for **all 76** exported finals;
* the logits are `q_i / eta` with no further operation: no advantage normalisation, no
  rank transform, no top-k, no weight clipping. The only additional operation is the
  per-state max subtraction inside `jax.nn.softmax`, which is exact;
* `eta` is the E-step dual (`741`, `jax_models.py:539-543`), solved against `eps_e = 0.5`
  by `eta_dual_loss` (`168-181`). It is **not** the entropy coefficient `alpha`
  (`temperature()`, `jax_models.py:533-534`). Note that `scripts/q_spread_from_ckpt.py:50-53`
  uses `alpha` as the temperature; that script's `q_spread` is therefore a different
  quantity from the trainer's, and this audit uses `eta`;
* no flattening across states occurs before the softmax.

## 0.3 Action representation

```
u_i ~ N(0, I_d)                         standard normal, d = estimator-visible dim
  |
y_i = mu + sigma * u_i                  pre-squash Gaussian  (diagonal sigma)
  |
tanh                                    jax_models.py:525
  |
a_i_raw = tanh(y_i)                     post-squash, UNclipped
  |
clip to +-(1 - 1e-4)                    reppo.py:712
  |
a_i_fit = clip(a_i_raw)
  |-----> critic input          reppo.py:748   q_i = critic(critic_obs_i, a_i_fit)
  |-----> actor likelihood      reppo.py:717   logp_theta_i = pi.log_prob(a_i_fit)
```

* `a_i_fit` is **post-squash and clipped**. It is not the raw Gaussian variable.
* The critic and the actor likelihood are evaluated at the **same** point, `a_i_fit`.
* **The known clipping/log-prob mismatch does not affect the weighted-MLE mean score.**
  The mismatch is between `old_pi_act_log_prob` (returned at the *unclipped* sample,
  `709-711`) and `logp_theta_i` (at the *clipped* action, `717`). The WML objective
  (`754`) uses only `logp_theta_i`. `old_pi_act_log_prob` enters solely the KL at
  `716-722`, i.e. the gate and the dual — not the E-step objective.
* `Sigma` is **diagonal**, `sigma = exp(log_std) + min_std` (`jax_models.py:524`),
  **state dependent** (both heads are functions of `obs`), with effective
  `min_std = 0.1` on all 76 checkpoints.
* `Sigma` **is** updated by the WML loss: `loc` and `log_std` are the two halves of one
  linear head over a shared trunk (`jax_models.py:522-523`).

**Consequence for naming.** The audit's condition for calling `v` the full implemented
actor update — only the mean optimised, covariance fixed — is **not met**. `v` is
therefore reported throughout as the **standardized mean-score direction**.

Because `a_i_fit != a_i_raw`, both branches are carried: the raw-Gaussian `u_i_raw`
decomposition as a manuscript-level diagnostic, and the implementation-space
`u_i_fit = (atanh(a_i_fit) - mu)/sigma` decomposition as primary.

**The exact score space is the whitened mean coordinate, and it coincides with
`u_i_fit`.** For a tanh-transformed diagonal Gaussian,
`log pi(a) = log N(y; mu, sigma) - sum_j log(1 - a_j^2)` with `y = atanh(a)`, and the
Jacobian term does not depend on `mu`. Hence
`d/dmu log pi(a_i) = (y_i - mu)/sigma^2`, and
`Sigma^{1/2} d/dmu log pi(a_i) = (y_i - mu)/sigma = u_i_fit` exactly. The tanh
transformation drops out of the mean score entirely; only the **clip** makes
`u_i_fit` differ from `u_i_raw`. Verified by autodiff to `1.1e-15` (T4 below).

## 0.4 PRNG reuse

| Location | What it does |
|---|---|
| `1093-1098` | `key, train_key = split(key)`; `xs=jax.random.split(train_key, cfg.num_epochs)` — **one key per epoch** |
| `612` | `def update(train_state, key)` — that epoch key |
| `1074` | `key, shuffle_key = jax.random.split(key)` — consumed for the permutation |
| `613`, `1085-1086` | `minibatch_update` is scanned over `minibatch_idxs`; the carry is `(idx, train_state)` — **the key is not in the carry and is not split per minibatch** |
| `711` | `sample_and_log_prob(sample_shape=(n_estep,), seed=key)` uses that closure key |

Therefore the M perturbations are:

* **independent across states within a minibatch** — one `(M, B_mb, d)` draw;
* **identical across all minibatches of an epoch** — same key, same shape, so the
  underlying standard-normal array is bit-identical from one minibatch to the next;
* **regenerated each epoch** (fresh epoch key), and regenerated per learn step;
* the same closure key also seeds the pathwise SAC sample at `684`.

Actual sampled tensor shape: `(M, B_mb, d)` = `(32, 2048, d)` for the DMC tasks and
`(32, 8192, 29)` for g1 (`num_mini_batches` 64 vs 16 over a `1024 x 128 = 131072`
batch). This is the **shared-across-minibatches** case, and it is the reason the batch
analysis in P6 cannot assume `ubar` averages down as independent noise across
minibatches: at a fixed position index the standardized draw repeats exactly.

## 0.5 Actor optimization semantics

* `603-607`: `actor_target.params = actor.params` at the **start of each learn step**.
  So `theta == theta_old` exactly at the **first minibatch of the first epoch** and
  nowhere else.
* The same sampled actions are reused by all `num_mini_batches` of an epoch only in the
  sense that the *noise* repeats; the *states* differ per minibatch. Actions are
  redrawn each epoch. `num_epochs = 8`.
* The actor changes across minibatches while `actor_target` stays fixed for the whole
  learn step, so the M-step targets are fixed while `theta` moves.
* Actor optimizer state: **not present in any export** — `actor.npz` holds
  `actor_module/*`, `eta_param`, `lagrangian_log_param`, `temperature_log_param` only.
  The optional copied-state replay of Step 3 is therefore **not available**.
* Mean and covariance **share all trunk parameters** and one output head.
* A separate target actor exists (`actor_target`, `218`), refreshed as above.

The u-space identity describes the **initial mean score at `theta_old`**. It is not
extended to the cumulative M-step displacement anywhere in this audit.

## 0.6 Verified dimensions — there is no d=21 checkpoint

Every `_final` export, from `meta.json` (`reports/artifacts/ubar_checkpoint_audit.csv`):

| task | arm | seeds | configured `action_dim` | actor mean/scale head | critic action input | pad | estimator-visible d |
|---|---|---|---|---|---|---|---|
| HopperHop | pathwise | 101–108 | 4 | 2x4 | 4 | 0 | **4** |
| HopperHop | weighted_mle | 101–108, 234 | 4 | 2x4 | 4 | 0 | **4** |
| WalkerRun | pathwise | 101–108 | 6 | 2x6 | 6 | 0 | **6** |
| WalkerRun | weighted_mle | 101–108 | 6 | 2x6 | 6 | 0 | **6** |
| LeapCubeRotateZAxis | pathwise | 101–108 | 16 | 2x16 | 16 | 0 | **16** |
| LeapCubeRotateZAxis | weighted_mle | 101–108 | 16 | 2x16 | 16 | 0 | **16** |
| WalkerRun (padded) | pathwise | 0–4 | 22 | 2x22 | 22 | 16 | **22** |
| WalkerRun (padded) | weighted_mle | 0–4 | 22 | 2x22 | 22 | 16 | **22** |
| G1JoystickFlatTerrain | pathwise | 101–108, 201 | 29 | 2x29 | 29 | 0 | **29** |
| G1JoystickFlatTerrain | weighted_mle | 101–108 | 29 | 2x29 | 29 | 0 | **29** |

`action_dim == actor head/2 == critic action input` for all 76. Simulator-consumed
dimension is 6 for padded Walker (`scripts/verify_action_pad.py 16`: "simulator
received dim 6, want 6") and equals `action_dim` elsewhere.

**Distinct estimator-visible d: {4, 6, 16, 22, 29}. No checkpoint has d = 21.**

Resolving the stated inconsistency:

* Walker's physical action dimension is **6**; that is the d of the unpadded Walker runs.
* Padded Walker adds `k = 16` inert coordinates: d = 6 + 16 = **22**.
* g1 (`G1JoystickFlatTerrain`) is d = **29**, not 21.
* **d = 21 is `HumanoidRun`**, the task the manuscript's crossover discussion refers to.
  There is **no HumanoidRun checkpoint in this repository.** No d=21 condition is
  preregistered and nothing is relabelled to stand in for it.

## 0.7 Checkpoint and `eta` availability

| arm | n `_final` | `with_eta` | `eta_curve[-1]` nonzero | eta range |
|---|--:|--:|--:|---|
| `weighted_mle` | 38 | 38 | 38 | 0.0010462 – 0.055114 |
| `pathwise` | 38 | 0 | 0 | — |

* **B-trained checkpoints**: `eta_param` is serialized in `actor.npz` and
  `eta_curve` is present in `meta.json` for all 38. `eta` is **measured**.
* **A-trained checkpoints**: `with_eta = False`; there is no `eta_param` and no E-step
  temperature ever existed. No "actual training eta" is invented. Where a crossed
  same-critic E-step is evaluated on an A-trained checkpoint, `eta` is obtained by
  solving the registered E-step dual at `eps_e = 0.5` and labelled
  **RECOMPUTED COUNTERFACTUAL ETA**, in a separate column from measured values.

## 0.8 Autodiff verification — `scripts/analysis/test_ubar_decomposition.py`

Run on `exports/WalkerRun_weighted_mle_pad16_s0_final` (d=22, M=32, eta=0.0410919
measured, median logit spread 1.307, clip rate 0.283%). **10/10 pass.**

| check | result |
|---|---|
| T1a weights normalise per state over axis 0 | PASS, `max|sum_i w_i - 1| = 5.6e-16` |
| T1b M = 32 | PASS, `w.shape = (32, 128)` |
| T2 `L_full = L_uniform + L_centered` | PASS, residual `8.9e-16` |
| T3a grad decomposition, native float32 params | PASS, rel residual `4.19e-08` |
| T3b same, params cast to float64 | PASS, rel residual `6.80e-16` |
| T4 whitened mean score by autodiff == `sum_i w_i u_i_fit` | PASS, rel err `1.09e-15` |
| T5 `v = ubar + c` identically | PASS, `2.7e-15` |
| T6a RMS `||ubar_raw||` == `sqrt(d/M)` | PASS, `0.8290` vs `0.8292`, ratio `0.9998` |
| T6b `E||ubar_raw||` == chi mean | PASS, ratio `0.9998` |
| T6c median `||ubar_raw||` == chi median | PASS, ratio `0.9994` |
| T7 `u_fit == u_raw` where the clip does not bind | PASS |

The loss-gradient decomposition holds on the **full actor parameter vector**, so the
implementation-level analysis of P6 is exact and does not depend on the u-space
identity.

### Two bugs found in the probe during this step, both fixed

1. **T3 initially "failed" at `4.19e-08`.** The trained parameters are float32, so the
   identity can only hold to float32 epsilon (`1.2e-07`) at native precision. Casting
   the *same* parameters to float64 drives the residual to `6.8e-16`, proving the
   residual is precision and not a real discrepancy. The check now reports both.
2. **T6 initially read 3–4% below every Gaussian benchmark** (ratios 0.966, 0.966,
   0.955), which looked like a standardization error. It was Monte-Carlo noise: T6 was
   being evaluated on the same 128 states used for the gradient checks, where the
   relative standard error on the RMS is ~1.3%. T6 is a pure PRNG/shape check with no
   model involvement, so it now draws 200 000 independent `ubar` vectors and matches
   all three benchmarks to within 0.06%. A first attempted fix — replacing the
   `arctanh` round-trip with an explicit standard-normal draw — did **not** change the
   ratios and was therefore not the cause, though it was kept because it removes a real
   precision hazard as `|a| -> 1`.

---

## Consequences carried into the preregistration

1. `v` is the **standardized mean-score direction**, not the full implemented actor
   update: covariance is not fixed and shares parameters with the mean.
2. The **implementation-space** decomposition on `u_i_fit` is primary; the raw-Gaussian
   one is a manuscript-level diagnostic. The two differ **only** through the clip.
3. The `d/M` analytic baseline applies to `ubar_raw` **only** and is not applied to
   `ubar_fit`.
4. Perturbations are **shared across minibatches within an epoch**, so P6 must not
   assume `1/sqrt(B)` averaging across minibatches.
5. `eta` is **measured** for arm B and **recomputed counterfactual** for arm A.
6. Available d values are **{4, 6, 16, 22, 29}**; the only within-task d manipulation is
   **WalkerRun 6 vs 22**.
7. The WML objective is gated by the KL clip, so the decomposition covers the mean score
   in full and the actor loss on ungated states only.
