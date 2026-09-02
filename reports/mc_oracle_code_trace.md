# What `Q_phi` was trained to predict: a line-level trace

**Purpose.** The Monte-Carlo oracle is only meaningful if it estimates the *same*
quantity the critic was regressed onto. This document traces that quantity from
source before any oracle value is computed. Everything below is read off
`src/jaxrl/reppo.py`, `src/networks/jax_models.py`, `src/jaxrl/utils.py` and
`src/env_utils/jax_wrappers.py` at `HEAD = d731e63` (branch `estep-study`), which is
the commit the corrected WalkerRun checkpoints were produced at.

**Headline.** The textbook SAC target is *not* what this code trains. Three things
differ and all three matter:

1. the critic is a **151-bin HL-Gauss categorical head** on `[0, 150]`, and `Q_phi`
   is the **expectation of that categorical distribution**, not a regression output;
2. the target is a **TD(lambda) return with `lambda = 0.95`**, not a one-step target;
3. the entropy term is folded into the reward as
   `r_tilde_t = r_t - gamma * alpha * log pi(a_{t+1} | s_{t+1})`, which places **no
   entropy term on the externally fixed first action**.

Point 3 is the form the pilot specification anticipated, and it is confirmed rather
than assumed. Point 2 does not change the *fixed point* — `lambda` trades bias for
variance in the estimator, and the operator the code iterates has the soft `Q^pi` as
its fixed point — but it does change what a one-step argument would predict, and it
is why the oracle targets the soft `Q^pi` rather than a one-step backup.

---

## 1. The scalar the actor operator queries

`src/jaxrl/reppo.py:728-744`:

```python
def actor_loss(params):
    critic_target_model = nnx.merge(
        train_state.critic.graphdef,
        train_state.critic.params,          # <- ONLINE critic params
    )
    ...
    pred_action, log_prob = pi.sample_and_log_prob(seed=akey)
    value = critic_target_model.critic(
        minibatch.critic_obs, pred_action    # <- UNCLIPPED tanh sample
    )
```

Two facts that the variable name actively obscures:

* **`critic_target_model` is not a target critic.** It is built from
  `train_state.critic.params`, the online critic. There is no separate critic target
  network anywhere in this update. The name is misleading; the parameters are the
  online ones.
* **The action is not clipped.** `pred_action` is the raw `tanh` sample. The
  environment clips to `+-0.999` (section 6) and the *stored* transition action is
  clipped, but the action the actor operator differentiates the critic at is not.

`critic()` is `src/networks/jax_models.py:305-310`:

```python
def critic(self, obs, action):
    value_cat = jax.nn.softmax(self.critic_cat(obs, action), axis=-1)
    value = value_cat.dot(jnp.linspace(self.vmin, self.vmax, self.num_bins, endpoint=True))
    return value
```

with `critic_cat -> critic_head` (`jax_models.py:297-303`):

```python
def critic_head(self, features):
    cat = self.critic_module(features) + self.zero_dist.value * 40.0
    return cat
```

For these checkpoints `num_bins = 151`, `vmin = 0.0`, `vmax = 150.0`, read from
`meta.json["critic_kwargs"]`.

**So `Q_phi(s, a)` is a convex combination of the support `linspace(0, 150, 151)`.**
It is *structurally* confined to `(0, 150)`: no value transform, no clipping step, but
the representation itself cannot express a value outside the support. This is a real
constraint on the measurable critic error, and it is recorded as a limitation rather
than assumed harmless.

### Reduction, ensemble, normalizer, action representation

| Question | Answer | Evidence |
|---|---|---|
| Which critic parameters | **online** `train_state.critic.params` | `reppo.py:729-732` |
| Ensemble / twin / min | **none** — a single head | `jax_models.py:297-310`; one `critic_module` |
| Distributional reduction | **expectation** over 151 HL-Gauss bins | `jax_models.py:305-310` |
| Value transform / clipping | **none applied**; support confines to `(0,150)` | as above |
| Critic observation normalizer | **its own** running estimator `critic_mean`/`critic_var` | `jax_wrappers.py:271-272, 342` |
| Action representation at the critic | **raw `tanh(y)`, unclipped** | `reppo.py:741-744` |
| Action representation at the sim | **`clip(a, -0.999, 0.999)`** | `jax_wrappers.py:254-265` |

For WalkerRun `obs_dim == critic_obs_dim == 24` and `asymmetric_obs: false`, so the
two normalizers are fed the same observation and their statistics coincide. They are
still *separate* estimators in code and are kept separate here.

---

## 2. The regression target

`reppo.py:676-687`:

```python
def critic_loss_fn(params):
    critic_pred = critic_model.critic_cat(minibatch.critic_obs, minibatch.action).squeeze()
    if cfg.hl_gauss:
        target_cat = jax.vmap(utils.hl_gauss, in_axes=(0, None, None, None))(
            target_values, cfg.num_bins, cfg.vmin, cfg.vmax)
        critic_update_loss = optax.softmax_cross_entropy(critic_pred, target_cat)
```

`target_values` is the `lambda`-return `G_t` of section 3; `minibatch.action` is the
**clipped** rollout action `a_t` (`reppo.py:547`). So the head is fit by cross-entropy
to `hl_gauss(G_t)`, a soft two-hot encoding (`src/jaxrl/utils.py:43-59`, Gaussian
smoothing with `sigma = 0.75 * bin_width`, and `inp` clipped to `[vmin, vmax]` at line
45). Under cross-entropy against a soft two-hot target, the minimiser of the expected
loss is the categorical whose bin probabilities match `E[hl_gauss(G_t)]`, and its
expectation is `E[G_t]` up to the discretisation of `hl_gauss` — so **`Q_phi(s_t,a_t)`
estimates `E[G_t | s_t, a_t]`**.

The loss is masked: `reppo.py:712-715` multiplies by `(1.0 - minibatch.truncated)`, so
transitions whose successor is a time-limit truncation contribute nothing to the
critic update.

`critic_loss` at `reppo.py:707-711` is a *logging-only* squared error against
`value`; it does not enter `loss`.

---

## 3. The `lambda`-return, unrolled

`reppo.py:612-640`:

```python
def compute_nstep_lambda(carry, transition):
    lambda_return, truncated, importance_weight = carry
    done = transition.done
    reward = transition.soft_reward
    value = transition.value
    lambda_sum = (
        jnp.exp(importance_weight) * cfg.lmbda * lambda_return
        + (1 - jnp.exp(importance_weight) * cfg.lmbda) * value
    )
    delta = cfg.gamma * jnp.where(truncated, value, (1.0 - done) * lambda_sum)
    lambda_return = reward + delta
    truncated = transition.truncated
    return (lambda_return, truncated, transition.importance_weight), lambda_return
```

scanned with `reverse=True` from `(batch.value[-1], ones_like(truncated[0]), zeros_like(iw[0]))`.

Resolving the carry indices: at index `t` the carry holds `G_{t+1}`,
`batch.truncated[t+1]` and `batch.importance_weight[t+1]`, while `transition` supplies
`batch.soft_reward[t]`, `batch.value[t]` and `batch.done[t]`. Therefore

```
G_t = r_tilde_t + gamma * [ trunc_{t+1} ? V_{t+1}
                          : (1 - done_t) * ( lambda*rho_{t+1} * G_{t+1}
                                           + (1 - lambda*rho_{t+1}) * V_{t+1} ) ]
```

**`rho = 1` in these runs.** `importance_weight` (`reppo.py:546-554`) corrects the
rollout policy `pi(., scale=offset)` back to `og_pi = pi(., scale=1)`. But
`config/reppo.yaml:33-34` sets `exploration_noise_min = exploration_noise_max = 1.0`,
so `offset == 1.0`, `og_pi == pi`, the raw log-ratio is `0`, and
`rho = clip(exp(0), lmbda_min, 1) = 1`. The Retrace machinery is present but inert.
The `lambda`-return is therefore plain on-policy TD(0.95).

**`V_{t+1}` is a one-sample bootstrap.** `reppo.py:556-560`:

```python
next_action, log_prob = actor_model.actor(next_obs).sample_and_log_prob(seed=act_key)
next_emb, _, _, value = critic_model.forward(next_critic_obs, next_action)
```

so `transition.value[t] = Q_phi(s_{t+1}, a_{t+1})` at a *sampled* `a_{t+1} ~ pi(.|s_{t+1})`,
using the **online** critic at rollout time. It is an unbiased one-sample estimate of
`V^pi(s_{t+1}) = E_{a~pi} Q(s_{t+1},a)`, and it enters `G_t` linearly, so the fixed
point is unaffected by the sampling.

**Setting `lambda -> 1`, `rho = 1`, no `done`/`truncation`,** the recursion unrolls to
`G_t = sum_{k>=0} gamma^k r_tilde_{t+k}`, which section 4 expands into exactly the
soft return the pilot specifies.

### The truncation flag is off by one

`transition.truncated = next_env_state.truncated` (`reppo.py:574`), i.e.
`batch.truncated[t]` flags whether `s_{t+1}` is a time-limit boundary. The carry
therefore supplies `batch.truncated[t+1]`, which flags `s_{t+2}`, where the intended
condition is plainly the one for `s_{t+1}`.

Consequences, both confined to time-limit boundaries:

* At the boundary transition itself the `where` takes the `else` branch, and
  `done_t = 1` zeroes it, so `G_t = r_tilde_t` with **no bootstrap** — the time limit
  is treated as a true terminal.
* One step earlier the `where` takes the `truncated` branch and bootstraps
  `gamma * V(s_t)` rather than continuing.

WalkerRun episodes are 1000 steps, so this touches on the order of `1/1000` of
transitions, and those transitions are additionally masked out of the critic loss by
`reppo.py:713`. It is recorded here because it is a genuine deviation from the
intended operator, not because it is expected to be numerically important.

---

## 4. The soft reward, and where the entropy term sits

`reppo.py:556-564`:

```python
next_action, log_prob = actor_model.actor(next_obs).sample_and_log_prob(seed=act_key)
...
soft_reward = (
    reward
    - cfg.gamma * log_prob.sum(-1).squeeze() * actor_model.temperature()
)
```

so `r_tilde_t = r_t - gamma * alpha * log pi(a_{t+1} | s_{t+1})` with
`a_{t+1} ~ pi(.|s_{t+1})` drawn fresh, and `alpha = exp(temperature_log_param)`
(`jax_models.py:533-534`).

Unrolling `G_t = sum_k gamma^k r_tilde_{t+k}`:

```
G_t = sum_k gamma^k r_{t+k}  -  alpha * sum_{j>=1} gamma^j log pi(a_{t+j} | s_{t+j})
```

**The externally fixed action `a_t` receives no entropy term.** This is precisely the
`Q_soft^pi(s0, a0)` of the pilot specification, and the `- gamma * alpha * log pi` at
index `t` supplies the `j = 1` term, not a `j = 0` term. There is no off-by-one here:
the `gamma` factor inside `soft_reward` is what shifts the entropy series to start at
`j = 1`.

Note the same `act_key` seeds both the executed action (`reppo.py:540`) and the
bootstrap/entropy action (`reppo.py:558`). Both terms enter `G_t` linearly, so this
correlation does not bias the fixed point.

### alpha is frozen

The confirmatory runs pass `hyperparameters.update_entropy_lagrangian=false` and
`hyperparameters.ent_start=0.014509912580251694` (ledger
`ledger/runs_faithful_repair.jsonl`, field `command`). The value the code actually
uses is the `exp()` of the stored parameter, read programmatically from the
checkpoint:

```
alpha = float(actor.temperature())  ->  0.014509915374219418
```

identical for both arms. It differs from the requested `ent_start` in the eighth
decimal, which is the float32 log/exp round-trip; the oracle uses the value read from
the checkpoint, not the value in the launch command and not a value copied from prose.

---

## 5. Which policy defines `pi` in `Q^pi`

* The bootstrap value and the entropy log-probability both come from
  `actor_model = nnx.merge(train_state.actor...)` inside `collect_rollout`
  (`reppo.py:516`), i.e. the **online actor**.
* `actor_target` is used **only** for the KL constraint (`reppo.py:750-760` and the
  E-step branch), and `reppo.py:647-651` copies `actor_target.params = actor.params`
  at the top of every learn step, so it is never more than one learn step stale.
* `scripts/export_ckpt.py:160` exports `state.actor.params` — the online actor.
* The rollout policy exploration `scale` is `1.0` (section 3), so the behaviour
  policy and `pi` coincide.

**Therefore `pi` is the exported online actor, evaluated with `scale = 1.0`.** There
is no ambiguity to resolve between online and target here, and none is silently
chosen: they are equal at export up to the last learn step, and only the online one
is saved.

`pi` is `Tanh(Normal(mu(s), sigma(s)))` with `sigma = exp(log_std) + min_std`,
`min_std = 0.1` (`jax_models.py:519-526`, `meta.json`).

---

## 6. Environment: transformations, termination, time limits

Wrapper stack, from `reppo.py:497-501` and `scripts/train_and_export.py:58-70`:

```
MjxGymnaxWrapper  ->  LogWrapper(num_envs)  ->  ClipAction(-0.999, 0.999)  ->  NormalizeVec
```

| Question | Answer | Evidence |
|---|---|---|
| Reward transformation | `state.reward * reward_scale`, `reward_scale = 1.0` | `jax_wrappers.py:104`; `config/env/mjx_dmc.yaml:4` |
| Action before env step | `clip(a, -0.999, 0.999)` | `jax_wrappers.py:254-265` |
| Action stored in the transition | `clip(a, -0.999, 0.999)` | `reppo.py:547` |
| Termination | `done = state.done > 0.5`; DMC WalkerRun has no early termination, `done` fires at the 1000-step limit | `jax_wrappers.py:105`, `episode_length=1000` |
| Time-limit handling | `truncated = env_state.info["truncation"]`, used as in section 3 | `jax_wrappers.py:191` |
| Bootstrap mask | `(1 - done_t)` on the `lambda` branch | `reppo.py:622` |
| Observation normalization | running mean/var, **updated every step**, applied *after* the update | `jax_wrappers.py:345-374` |

`env.step` for MJX takes **no RNG key**: `MjxGymnaxWrapper.step` calls
`self.env.step(state, action)` (`jax_wrappers.py:103`). The key threaded through the
wrappers is discarded. WalkerRun dynamics and its auto-reset are therefore a pure
function of `(state, action)`, which is what makes exact state branching possible at
all; this is verified empirically rather than assumed (see
`reports/mc_oracle_walker_pilot.md`, state-restoration section).

---

## 7. The two deliberate deviations the oracle makes, and why

**(a) The observation normalizer is frozen** at the checkpoint statistics rather than
continuing its running update. `NormalizeVec.step` updates `mean`/`var` from the
current batch on every step and then normalises with the *updated* statistics. A
policy whose input transform drifts is not a fixed policy, and `Q^pi` is undefined for
a non-stationary `pi`. Freezing is what makes the estimand well posed. The saved
`count` is `52_298_752`, so the per-step drift of the training normalizer at this
point is negligible in any case.

**(b) `NormalizeVec` is dropped from the wrapper stack** and normalization applied by
hand, because its state mixes unbatched running statistics with batched env state,
which makes per-state cloning error-prone. This follows the existing
`scripts/critic_fidelity/common.py` convention.

Both deviations are inherited from the already-used critic-fidelity harness and are
restated here so they are not rediscovered later as surprises.

---

## 8. What the oracle must therefore compute

```
Q_soft^pi(s0, a0) = E[ r_0 + sum_{t>=1} gamma^t ( r_t - alpha * log pi(a_t | s_t) ) ]
```

with

* `gamma = 0.99`, `alpha = 0.014509915374219418` (read from the checkpoint),
* `a_0` fixed externally and **not** entropy-penalised,
* `a_t ~ pi(.|s_t)` for `t >= 1` from the exported online actor with `scale = 1.0`,
* the environment receiving `clip(a, -0.999, 0.999)` at every step, including `t = 0`,
* **no `Q_phi` bootstrap at the horizon.**

That last point is a departure from `scripts/critic_fidelity/common.py:soft_return`,
which adds `gamma^H Q_phi(s_H, a_H)`. Here the oracle is being differenced *against*
`Q_phi`, so putting `Q_phi` into the oracle would make `e = Q_phi - Q_oracle` partly a
comparison of `Q_phi` with itself. The pilot uses the realised term only
(`soft_return_parts(...)[0]`), and treats the missing tail as truncation bias to be
measured by horizon doubling, not argued away.

`Q_phi` is queried at the **unclipped** `tanh(y)`, matching `reppo.py:741-744`, while
the environment applies its native `+-0.999` clip. `Q^pi(s, a)` is genuinely constant
in `a` beyond the clip, because the simulator cannot distinguish those actions, so the
resulting `e(s, y)` genuinely varies there. This is a property of the learned critic
relative to the executed behaviour, not an artefact — but because the WML checkpoint
has a wide pre-squash policy the saturated fraction is expected to be non-trivial, so
the clip rate is measured and a no-clip-subset sensitivity is preregistered.

---

## 9. Reproduction

```
git -C ~/repos/reppo show d731e63:src/jaxrl/reppo.py            | sed -n "540,730p"
git -C ~/repos/reppo show d731e63:src/networks/jax_models.py    | sed -n "289,320p"
git -C ~/repos/reppo show d731e63:src/jaxrl/utils.py            | sed -n "43,59p"
git -C ~/repos/reppo show d731e63:src/env_utils/jax_wrappers.py | sed -n "100,200p;254,375p"
```
