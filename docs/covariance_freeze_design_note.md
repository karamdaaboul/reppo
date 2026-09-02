# Covariance freeze: what the intervention is, and what it is not

Generic design note for the `freeze_sigma` mechanism. It documents the mechanism
only. It fixes no freeze value, no experimental design and no analysis; those belong
to a preregistration.

---

## 1. The policy's covariance, traced

`SACActorNetworks` (`src/networks/jax_models.py:326-433`) is the continuous actor.
Everything below is that class; `SACDiscreteActorNetworks` (line 456) is a separate
class that WalkerRun never constructs and that this mechanism does not touch.

| Question | Answer | Evidence |
|---|---|---|
| Actor output head | one `FCNN` with `out_features = action_dim * 2` | `jax_models.py:348-361` |
| loc / log_std split | `loc, log_std = jnp.split(loc, 2, axis=-1)` | `jax_models.py:393` |
| Effective sigma | `sigma = exp(log_std) + min_std`, then `* scale` | `jax_models.py:394` |
| `min_std` | **0.1**, the constructor default | `jax_models.py:336` |
| Is `min_std` configurable? | **No.** `make_init` never passes it, so the config field `actor_min_std` is dead | `reppo.py:364-377`; `export_ckpt.py` records `EFFECTIVE_ACTOR_MIN_STD` for this reason |
| State-dependent? | **Yes** — `log_std` is an output of `actor_module(obs)` | `jax_models.py:392` |
| Shared trunk? | **Yes, and more than a trunk** — `loc` and `log_std` are two halves of the *same* final `nnx.Linear`, so they share every hidden layer *and* one output weight matrix | `jax_models.py:348-361`, `FCNN.output_layer` |
| `scale` | rollout exploration multiplier; `exploration_noise_min = max = 1.0`, so it is identically 1 | `config/reppo.yaml:33-34`, `reppo.py:519-530` |

Every place the covariance is used:

| Path | Site | Constructor |
|---|---|---|
| Rollout, base policy | `reppo.py:538` | `actor()` |
| Rollout, exploration policy | `reppo.py:539` | `actor(scale=offset)` |
| Bootstrap / entropy action at `s_{t+1}` | `reppo.py:557` | `actor()` |
| Online actor in the actor loss (PW sampling) | `reppo.py:736` | `actor()` |
| Target actor, reverse-KL branch | `reppo.py:754` | `actor()` |
| Target actor, legacy E-step branch | `reppo.py:803` | `actor()` |
| Same-point repair, old policy | `reppo.py:772` | `gaussian()` |
| Same-point repair, new policy | `reppo.py:777` | `gaussian()` |
| Analytic Gaussian KL diagnostic | `reppo.py:798` | via `gaussian()` |
| Decoupled M-step KLs | `reppo.py:865, 868` | `gaussian()` |
| Estimator diagnostics | `reppo.py:1003` | `gaussian()` |
| **Evaluation** | `reppo.py:275` | **`det_action()` — no sigma at all** |

The evaluation policy is `tanh(mu)` and never draws a sample, so there is no
evaluation covariance to freeze. This is stated explicitly because a list of "every
path must use the frozen sigma" would otherwise imply an evaluation path that does
not exist.

The target actor needs no separate treatment: `reppo.py:647-651` copies
`actor_target.params = actor.params` at the top of every learn step, and the freeze is
a property of the module, not of the parameters.

## 2. Why this is not the `beta_sigma_fixed` experiment

The manuscript already contains a "fixed `beta_Sigma`" experiment. It is a different
mathematical object, and the difference is not cosmetic.

`beta_sigma_fixed` (`reppo.py:155`, used at `reppo.py:898-908`) lives inside the
**decoupled M-step** branch, which runs only when `mstep_decoupled=True`. In that
branch:

* `mu_new, sg_new = actor_model.gaussian(minibatch.obs)` (`reppo.py:868`) — the
  covariance is **still learned and still state-dependent**;
* the objective is `-sum_i w_i (logp_mu + logp_sigma)` with
  `logp_sigma = gaussian_logp(u_i, mu_old, sg_new)` (`reppo.py:889-891`) — so
  **`sg_new` still receives gradient**;
* `kl_sigma` is a covariance KL constraint and `beta_sigma` is its Lagrange
  multiplier;
* `beta_sigma_fixed` holds **the multiplier** constant instead of solving it through
  its dual. The source comment is explicit: *"constant: no dual term, so
  beta_sigma_param gets no gradient"* — the parameter that stops moving is the dual,
  not the covariance.

So that experiment fixes **the price of changing the covariance**. This intervention
fixes **the covariance**. Formally: there, the policy family is unchanged and a
regulariser weight is held constant; here, the family is restricted from
`{N(mu(s), Sigma(s))}` to `{N(mu(s), Sigma*)}` with `Sigma*` constant in state and in
time, and `dL/d(log_std head) = 0` identically.

There is a further, decisive separation. **All 16 corrected WalkerRun runs have
`mstep_decoupled: false` and `beta_sigma_fixed: null`**, and `with_betas =
cfg.mstep_decoupled`, so `beta_mu_param` and `beta_sigma_param` **do not exist in
their parameter tree at all**. The earlier experiment is on a code path these runs
never enter.

## 3. The mechanism

A default-off configuration field:

```yaml
freeze_sigma: null      # null | scalar | length-d vector
```

Applied in one helper that every distribution constructor routes through
(`jax_models.py`, `SACActorNetworks.effective_std`):

```python
def effective_std(self, log_std):
    if self.freeze_sigma is None:
        return jnp.exp(log_std) + self.min_std      # corrected path, unchanged
    frozen = jnp.asarray(self.freeze_sigma, dtype=log_std.dtype)
    if frozen.size == 1:
        frozen = jnp.reshape(frozen, ())
    return jnp.broadcast_to(frozen, log_std.shape)
```

Three properties are deliberate:

**It is applied at the effective sigma, not at `log_std`.** `freeze_sigma = x` means
the pre-squash sigma *is* `x`. Implementing `log_std = x` would have meant
`sigma = exp(x) + 0.1`, which is a different number and invites exactly the
`min_std` ambiguity the design is meant to avoid.

**The frozen value is a plain Python tuple, not an `nnx.Param`.** It therefore stays
out of `nnx.state`, so the parameter tree, the optimizer state and the checkpoint
layout are unchanged. Checkpoints exported before this mechanism existed still load,
because `load_ckpt` passes `actor_kwargs` through and the missing key falls back to
the constructor default `None`.

**The learned scale head is still computed.** It remains available as a diagnostic —
what the policy *would* have done — but `effective_std` discards it, so it has no
effect on the distribution.

A guard refuses `freeze_sigma` together with a non-unit exploration `scale`
(`reppo.py`, before `collect_rollout`), because `scale` multiplies sigma and would
otherwise make the effective width differ from the value it is declared to be.

## 4. Gradient semantics — the part that must not be understated

In the frozen branch `log_std` is read **only** for its static `.shape` and `.dtype`.
No traced value flows from it into the returned sigma. Therefore:

* the gradient of the actor loss with respect to the **scale half** of the output
  layer (`kernel[:, d:]`, `bias[d:]`) is **exactly zero**, not merely small;
* under Adam those coordinates receive `m = v = 0` and so never move, at any step;
* the **mean half** and the **shared trunk** remain fully trainable through every
  surviving mean/operator loss;
* the shared trunk **no longer receives the gradient component that previously
  arrived through the learned covariance branch**.

Measured on a fresh WalkerRun actor: the scale-half gradient is `0` when frozen and
`946.9` when learned, on the same inputs and the same loss.

> **The intervention disables covariance adaptation and therefore removes both the
> changing covariance and its gradient pathway through the shared actor network.**

It is **not** "holding a logged sigma statistic constant". A positive result is
therefore an effect of the whole package — the fixed width, the removed scale
gradient, and everything downstream in exploration and data collection — and cannot
be attributed to any one of them.

## 5. What the mechanism does not decide

It does not choose a freeze value, a task, a seed set, an analysis or a claim. It
does not distinguish exploration effects from critic-training effects from
shared-trunk gradient effects. Any experiment using it needs its own preregistration
to fix those, and any positive result is an effect of covariance adaptation *through
the complete training loop*, not of a specific mediating pathway.
