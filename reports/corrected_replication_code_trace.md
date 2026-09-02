# Source trace for the corrected operator replication

Read-only. Produced **before** any implementation change. All references are
`src/jaxrl/reppo.py` at repository HEAD `4eff7ed` unless stated; the `src/` tree at
that HEAD is `fc2eae02`, **byte-identical to the original experiment commit
`07319d4`**, so this trace describes the code that produced every run in the paper.

## 0. Provenance

| | |
|---|---|
| Branch / HEAD | `estep-study` / `4eff7ed964d9bd7a1bbc19be3b4e4d212da52a87` |
| `git status --short` | *empty* — 0 modified, 0 untracked |
| Remote | `origin/estep-study` = `4eff7ed` (identical) |
| `src/` tree | `fc2eae02e780a1c2c3c9b7df85a2a3139d56aacf` = `07319d4:src` |
| `config/` tree | `122511a9179be3d92481c04994e964e072bfc34d` |
| Python / JAX / flax / numpy / scipy | 3.12.14 / 0.5.2 / 0.10.6 / 2.5.1 / 1.18.0 |
| Accelerators | SLURM `c23g` (H100 94 GB ×50 nodes), `c25g` (H100 80 GB ×29) |

### Commit-role disambiguation (requested)

| Role | Commit |
|---|---|
| Analysis evidence HEAD | `39c5171` "Track the ubar-ratio figure" |
| Report reconciliation | `5de9b3e` |
| Report corrections | `ff73f3f`, `4eff7ed` |
| **Actual final repository HEAD** | **`4eff7ed`** |
| Manuscript reconciliation (Overleaf) | `3059cda`, `ab60455` |
| **Actual Overleaf HEAD** | **`73adc6a`** (Phase 1 scope correction) |

`5de9b3e` appears in `reports/final_manuscript_claim_audit.md` because the audit was
written *at* `5de9b3e`; `ff73f3f` and `4eff7ed` are its own later corrections. Both are
correct at their respective times and nothing was overwritten.

---

## 1. The actor update, line by line

| Element | Location | Fact |
|---|---|---|
| PW actor sampling | `684` | `pred_action, log_prob = pi.sample_and_log_prob(seed=key)` — **no `sample_shape`** |
| **PW actor-objective samples** | `684`, `809-812` | **exactly 1** per state |
| PW objective | `809-812` | `objective = log_prob * sg(temperature()) - value` |
| PW critic point | `685-687` | `critic(critic_obs, pred_action)` — post-tanh, **unclipped** |
| WML action sampling | `709-711` | `actor_target_model.actor(obs).sample_and_log_prob(sample_shape=(n_estep,), seed=key)` |
| **WML sample count `M`** | `704-708`, `config/reppo.yaml:67` | `estep_num_samples = 32`; `n_estep = 32` for `weighted_mle`, **16** for `pathwise` |
| Pre-squash latent | *not materialised* | `distrax.Transformed(Normal, Tanh)` samples post-tanh directly; `y` is never held |
| tanh transform | `src/networks/jax_models.py:525` | `distrax.Transformed(Normal(loc,std), Tanh())` |
| Hard clip | `712` | `old_pi_action = jnp.clip(old_pi_action, -1+1e-4, 1-1e-4)` |
| WML critic point | `748` | `critic(critic_obs_i, old_pi_action)` — **clipped** post-tanh |
| Old-policy log prob | `709-711`, `716` | returned by `sample_and_log_prob` at the **unclipped** sample |
| New-policy log prob | `717` | `pi.log_prob(old_pi_action)` at the **clipped** action |
| KL calculation | `722` | `kl = old_pi_act_log_prob - pi_act_log_prob`, shape `(B,)` |
| KL reduction | `716-720` | `.sum(-1)` over dims, `.mean(0)` over the `n_estep` samples |
| KL bound | `config/reppo.yaml:60` + all `experiment_overrides/*` | `kl_bound = 0.1` |
| Actor objective wrapper | `850-855` | `jnp.where(kl < kl_bound, objective, kl*sg(lagrangian)*reduce_kl)` |
| KL multiplier parameterization | `jax_models.py:536-537` | `lagrangian() = jnp.exp(lagrangian_log_param)` — **unbounded bare exponential** |
| KL multiplier loss | `873-874`, `888-889` | `lagrangian_loss = -lagrangian * sg(kl - kl_bound)`; added to `loss` |
| `eta` parameterization | `jax_models.py:539-543` | `clip(softplus(eta_param), eta_min=1e-4, eta_max=10.0)` |
| `eta` loss | `168-181`, `759`, `891` | standard MPO dual at `eps_e = 0.5` |
| Optimizer | `383-401` | `optax.chain(clip_by_global_norm(max_grad_norm), adam(linear_schedule(lr, 0, num_updates)))` |
| Target actor refresh | `603-607` | `actor_target.params = actor.params` at the **start of every learn step** |
| Mean/covariance coupling | `jax_models.py:522-524` | `loc, log_std = split(actor_module(obs), 2)` — one head, **shared trunk**; `sigma = exp(log_std) + min_std` (effective `min_std = 0.1`) |

### PRNG

| Level | Location | Behaviour |
|---|---|---|
| learn step | `1109` | fresh `learn_key` per step |
| epoch | `1093-1098` | `xs=jax.random.split(train_key, num_epochs)` — **one key per epoch** |
| minibatch | `613`, `1085-1086` | `minibatch_update` scanned over indices; the carry is `(idx, train_state)` — **the key is a closure constant, never split per minibatch** |
| within minibatch | `684`, `711` | one `(M, B, d)` (or `(B, d)`) draw; independent across states |

**Consequence:** the standard-normal array is bit-identical across every minibatch of
an epoch, and independent across states within one minibatch.

---

## 2. The three facts the corrected experiment rests on

**PW-1.** The pathwise objective uses **one** actor sample per state. `684` has no
`sample_shape`; `809-812` consumes only `log_prob` and `value` from that single draw.

**WML-32.** The weighted-MLE objective uses **32** critic-scored samples
(`estep_num_samples: 32`, `704-708`, `754`).

**The PW-side 16 samples are not operator-objective samples — with one exception that
must be stated.** For `pathwise`, `n_estep = 16` (`704-708`) and those samples feed
only `kl` (`716-722`); they never enter `objective` (`809-812`). However `kl` is
differentiable in `theta` through `logp_theta_i = pi.log_prob(old_pi_action)` (`717`),
so under the replacement branch (`850-855`), when `kl >= 0.1` the operator objective is
**discarded** and the entire actor gradient for that state comes from the 16-sample KL
term. So the 16 draws are KL-term gradient samples, not operator samples, and they
become the *only* gradient source exactly where the branch fires.

**These arms are not sample-budget matched**, and no part of this replication describes
them as such. `PW-1` issues 1 critic evaluation per state for its objective (plus 16
actor log-prob evaluations for the KL); `WML-32` issues 32. The training comparison is
an **algorithmic operator comparison**. The equal-query question is answered separately
by the frozen same-critic diagnostic registered in Phase 5.

---

## 3. The four construct-validity defects, located

1. **Mismatched log-probability points** (`709-712`, `717`). The old-policy log
   probability is returned at the unclipped sample; the clip is applied at `712`; the
   new-policy log probability is evaluated at the clipped action. The two terms of
   `kl` (`722`) refer to different action points, so `kl` is not a KL of anything.
2. **Replacement rather than penalty** (`850-855`). Above the bound the policy
   objective is discarded instead of penalised.
3. **Unbounded multiplier** (`jax_models.py:536-537`). A bare `exp` under Adam, unlike
   `eta` (`539-543`) and `beta_mu` (`545-546`), which are softplus-and-clipped.
4. **Shared PRNG across minibatches** (`613`, `1085`). Identical innovations in every
   minibatch of an epoch.

Defect 1 does **not** affect the WML operator objective (`754` uses only
`logp_theta_i`); it affects the KL and therefore the gate and the dual. Defect 2
affects both arms. This trace does not attribute any past outcome to any one of them.

---

## 4. Recoverability of the candidate tasks

| task | env identifier | d | config | frozen `alpha` | horizon | eval protocol | seed namespace | overrides | verdict |
|---|---|--:|---|---|---|---|---|---|---|
| **WalkerRun** | `env=mjx_dmc env.name=WalkerRun` | 6 | `experiment_overrides=mjx_dmc_large_data` | `0.014509912580251694` (`ladder_matrix.sh:36`, seed-901 calibration L.1.31) | `52 297 728` steps | `num_eval=20` (21 points) | 101–108 | none beyond the override file | **RECOVERABLE** |
| **G1JoystickFlatTerrain** | `env=mjx_humanoid env.name=G1JoystickFlatTerrain env.asymmetric_obs=false` | 29 | `experiment_overrides=mjx_humanoid_large_data` | `0.00020752247655764222` (`ladder_matrix.sh:34`) | `52 297 728` steps | `num_eval=20` | 101–108 | `asymmetric_obs=false` | **RECOVERABLE** |
| **HumanoidRun** | `env=mjx_dmc env.name=HumanoidRun` | **unconfirmed** | — | — | — | — | — | — | **NOT RECOVERABLE** |

### HUMANOIDRUN CORRECTED REPLICATION NOT REPRODUCIBLE FROM RETAINED ARTIFACTS

Exhaustive search of `/hpcwork/qzi10910`, `~/repos` and `~`: **zero** HumanoidRun
exports, **zero** run directories, **zero** `meta.json`, **zero** Hydra configs, and no
ledger entry containing the string. The only artifacts are an upstream benchmark CSV
and the upstream 23-task sweep script `scripts/paper_experiments/slurm_dmc.sh:16`,
which is not our experiment (it runs learned `alpha` from a different repository path).

Missing required fields, itemised:

| required field | status |
|---|---|
| exact environment identifier | recoverable (`mjx_dmc` / `HumanoidRun`) |
| **exact action dimension** | **unconfirmed** — the manuscript's own open `\verify{}` records that dm_control reports 21 but flattened MJX variants report 24, and `d` enters the crossover display directly |
| **exact training configuration** | **absent** — no `meta.json`, no Hydra snapshot |
| **exact frozen alpha** | value `0.00329` is documented (`prereg_dimension_ladder.md:1452`) and reproduces from `HumanoidRun_pathwise_s2`, but that document states the seed selection behind it "is not itself evidenced" and the nine candidate medians span `0.00255`–`0.00441` |
| **original training horizon** | **not recorded** |
| **evaluation protocol** | **not recorded** |
| **original seed namespace** | **not recoverable**, and actively contradicted: `prereg_dimension_ladder.md:1485-1486` records `HumanoidRun_pathwise_fa_s3_*` carrying resolved override seed **7**, and `HumanoidRun_weighted_mle_s0_*` carrying seed **1** |
| **task-specific overrides** | **not recoverable** — the override/name disagreement above proves the resolved overrides differ from the names |

Seven of eight fields are missing, unconfirmed or self-contradictory. Nothing is
invented or inferred. **HumanoidRun is excluded from the confirmatory launch.**

**Recommendation.** Remove the `d=21` experiment from the abstract and main headline
evidence. Its checkpoints are gone, its seed labelling is provably unreliable, its
action dimension is unconfirmed, and it cannot be replicated under corrected code. It
may remain in an appendix, labelled retrospective and non-reproducible, exactly as
`docs/prereg_dimension_ladder.md` already labels it.

**Confirmatory design is therefore the mandatory two-task set: 32 runs.**

### Seed namespaces already consumed (must not be reused)

| task | used |
|---|---|
| WalkerRun | 0–4, 101–108 |
| G1JoystickFlatTerrain | 101–108, 201 |
| HopperHop | 101–108, 234 |
| LeapCubeRotateZAxis | 101–108 |
| calibration | 901 |

The corrected replication reserves a **fresh namespace, 301–308**, and smoke tests use
**401+**, disjoint from both.
