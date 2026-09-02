# Design lock: faithful-repair operator replication

Frozen **before** any training-code edit and before any corrected return is seen. No
implementation detail below was chosen using return.

Repository HEAD at lock time: `a3d2352` (branch `estep-study`). `src/` tree
`fc2eae02` = `07319d4:src`, byte-identical to the original experiment.

## 0. Terminology, used consistently in every downstream report

| term | meaning | instances |
|---|---|---|
| **coding inconsistency** | two quantities that the mathematics requires to be evaluated at one point are evaluated at different points | the old- and new-policy log probabilities entering the sampled KL, separated by `jnp.clip` (`src/jaxrl/reppo.py:709-717`) |
| **published algorithmic design** | a deliberate choice in the released REPPO algorithm | the piecewise policy/KL gate (`850-855`); the exponential log-space KL multiplier (`src/networks/jax_models.py:536-537`) |
| **construct-validity concern** | a property that may make an operator comparison measure something other than the operator | the gate may suppress the compared operator at different rates across arms |

These are **not** collectively called "code defects".

### Provenance of all three, verified from git history

`git log -S` on each construct returns upstream commit **`c3b13de`** ("Run config and
fixed aux loss issues", Claas Voelcker, 2025-08-04), with `86fd47b` (Axel Brunnbauer,
2025-07-15) and the fork base `09a8df2` (2025-07-15) around it. The first commit
authored by this project is `de68825`, 2026-08-28. **All three constructs — including
the clip that sits between the two log probabilities — are upstream and unmodified at
HEAD.**

Consequence to disclose: the repair in §2 makes the faithful arm differ from *released*
REPPO on this point. **Corrected in  §1:
there are TWO behaviour-affecting differences, not one — the same-point likelihood
evaluation here, and the fresh minibatch innovations of §5. The wording "exactly one
point" in the original lock was wrong and the preregistration supersedes it.** It is a repair of an upstream implementation inconsistency,
not a repair of anything this project introduced, and not a redesign of the algorithm.

## 1. Latent and action representation (frozen)

For the E-step / old-policy samples, the pre-squash latent is **materialised and
retained**:

```
u_i  ~ N(0, I_d)                       explicit standard-normal draw
y_i  = mu_old + sigma_old * u_i        pre-squash latent, from the TARGET actor
a_i  = tanh(y_i)                       post-squash action
```

`(y_i, a_i)` is one object. The identical pair is used for **all five** consumers:

1. critic evaluation — `Q(s, a_i)`;
2. old-policy log probability — from `y_i`;
3. new-policy log probability — from the same `y_i`;
4. the WML likelihood being fitted — from the same `y_i`;
5. the sampled KL — from the two log probabilities above.

`y_i` is **never** recovered by `arctanh`. It is carried forward from the draw.

## 2. Same-point log probability (frozen)

For a tanh-transformed diagonal Gaussian,

```
log pi(a) = log N(y; mu, sigma) - sum_j log(1 - tanh(y_j)^2),     y = atanh(a)
```

The Jacobian term depends only on `y`, not on the policy parameters. Because both
policies are evaluated at the **same** `y_i`:

* it is **identical** in the old and new log probabilities and **cancels exactly** in
  their difference, so the sampled KL never needs it;
* it is a constant with respect to `theta` in the WML objective and therefore does not
  enter the actor gradient.

It is nevertheless computed, for logging, with the numerically stable identity
`log(1 - tanh(y)^2) = 2*(log 2 - y - softplus(-2y))`, which is finite for every finite
`y` and does not underflow as `|y|` grows. `log N` is evaluated in closed form from
`y`, never from `a`.

**No hard clip is applied in the corrected path.** The clip exists in legacy code to
keep `atanh` finite; carrying `y_i` forward removes the need for it entirely. The
corrected-path clip rate is therefore **exactly zero by construction**, not
empirically small, and there is no numerical safeguard left that could move the action
point of either log-probability term.

One behavioural consequence is disclosed rather than hidden: legacy WML evaluates the
critic at the **clipped** action (`748`), while the corrected path evaluates it at
`tanh(y_i)` unclipped. Both log probabilities and the critic now share one point;
previously the critic shared a point with the new-policy log probability but not with
the old-policy one.

## 3. Sampled KL orientation (frozen, unchanged)

Legacy `722`: `kl = old_pi_act_log_prob - pi_act_log_prob`, with samples drawn from
`pi_old`. That is

```
KL(pi_old || pi_theta)  estimated as  E_{a ~ pi_old}[ log pi_old(a) - log pi_theta(a) ]
```

the **forward** KL from old to new. The corrected path estimates the same quantity at
the same orientation on the same samples. **The orientation is not changed.**

Reduction is also unchanged: `.sum(-1)` over action dimensions, `.mean(0)` over the
`n_estep` samples, giving one scalar per state, shape `(B,)`.

## 4. Published constructs, preserved

**The gate is kept.** `actor_loss = where(kl < kl_bound, objective, kl * sg(lagrangian)
* reduce_kl)` with `kl_bound = 0.1`, exactly as published. It is **not** replaced by an
additive Lagrangian in this experiment. Above the bound the operator objective is
intentionally absent; Phase 4 tests that this is so, rather than treating it as a bug.

**The multiplier is kept.** `lagrangian() = exp(lagrangian_log_param)`, unbounded, as
published. No bound is introduced. An additive-KL or bounded-dual variant, if ever
built, gets a separate name and a separate preregistration and is **not** a correction
arm.

## 5. PRNG (frozen)

Corrected mode splits a fresh operator-sampling key **inside** the minibatch scan.
Purpose-separated streams, each folded from the run root by a fixed string tag so that
they cannot interfere:

| stream | consumer |
|---|---|
| `env` | environment reset and step |
| `rollout` | rollout policy actions |
| `actor` | actor-objective sampling (PW-1 draw, WML-32 draws) |
| `perm` | minibatch permutation |
| `eval` | evaluation rollouts |
| `diag` | diagnostics |

The `actor` stream is folded with `(learn_step, epoch, minibatch_index)`. Changing the
PW actor sample count consumes only the `actor` stream and therefore **cannot shift**
the environment, rollout or evaluation streams — this is required so that the two arms
see the same environment realisation for a given seed.

Legacy mode retains the original behaviour exactly: one key per epoch, closure-constant
across the minibatch scan.

## 6. Diagnostics (frozen list, computed not placeheld)

Gate: elementwise activation rate; fraction of states receiving the operator objective;
fraction receiving only the KL term. KL: sampled KL quantiles (q10/25/50/75/90/95/99,
min, max); **exact analytic pre-squash Gaussian KL** as an independent diagnostic;
sampled-minus-analytic discrepancy (mean, median, q95). Gradients: operator-gradient
norm, KL-gradient norm, complete actor-gradient norm. Multiplier: raw log parameter,
effective `exp`, its gradient, Adam first and second moments, max and quantiles, and a
finite/non-finite indicator. Operator: `eta`, within-state logit spread, ESS, maximum
WML weight. Policy: sigma per coordinate and mean, action saturation rate. Plus actor
step norm, post-update KL, critic loss, evaluation return.

Disabled diagnostics are **omitted or written as NaN**, never as arrays of zeros.

## 7. Inherited hyperparameters (unchanged, from `meta.json` of the 64-run study)

| | WalkerRun | G1JoystickFlatTerrain |
|---|---|---|
| env args | `env=mjx_dmc env.name=WalkerRun` | `env=mjx_humanoid env.name=G1JoystickFlatTerrain env.asymmetric_obs=false` |
| overrides | `experiment_overrides=mjx_dmc_large_data` | `experiment_overrides=mjx_humanoid_large_data` |
| `d` | 6 | 29 |
| frozen `alpha` | `0.014509912580251694` | `0.00020752247655764222` |
| `update_entropy_lagrangian` | false | false |
| `M` (`estep_num_samples`) | 32 | 32 |
| `eps_e` | 0.5 | 0.5 |
| `kl_bound` | 0.1 | 0.1 |
| horizon | 52 297 728 steps | 52 297 728 steps |
| `num_eval` | 20 (21 points) | 20 |
| `num_mini_batches` / `num_epochs` | 64 / 8 | 16 / 8 |
| `gamma`, `vmin`, `vmax` | 0.99, 0, 150 | 0.97, −10, 10 |
| effective `min_std` | 0.1 | 0.1 |

**Identical frozen `alpha` across the two arms within a task.** No learned-alpha arm.

## 8. Arms and sample counts (frozen)

| arm | actor-objective samples | critic evaluations per state |
|---|--:|--:|
| **PW-1** | 1 | 1 |
| **WML-32** | 32 | 32 |

These are **not** equal-sample or equal-query and are never described as such. The
primary training comparison is an algorithmic operator comparison under the published
REPPO gate, not an isolated-estimator comparison. The equal-query question is answered
only by the mandatory frozen same-critic diagnostic (PW-1, PW-32, centred ZO-32, exact
nonlinear `c`, full `v`) at the final corrected checkpoints.

`PW-32` is **not** a confirmatory training arm here and must not delay this launch.

## 9. Smoke-test acceptance rules (frozen before any smoke run)

Smoke seeds **401–404**, disjoint from the confirmatory namespace 301–308. A smoke run
is accepted iff, over its short horizon:

1. all losses finite at every logged step;
2. `sigma` mean stays within `[0.02, 5.0]` and does not monotonically diverge;
3. effective multiplier finite and `< 1e6`;
4. sampled-vs-analytic KL median relative discrepancy `< 0.25` in the non-saturated
   regime;
5. corrected-path clip rate exactly `0.0`;
6. gate activation rate in `[0, 1]` and actually computed (not a placeholder);
7. evaluation return finite and not NaN;
8. two consecutive minibatches receive different innovation arrays.

Failing any of these blocks the confirmatory launch. **None of these rules refers to
the level of the return**, and none may be relaxed after seeing corrected returns.

## 10. What this experiment can and cannot conclude

It can answer: *when both operators are implemented with a consistent
same-point log-probability under the published REPPO gate, does the task-level operator
difference reproduce?*

It cannot attribute any change to the log-probability repair specifically — that would
need a separate randomized ablation — and it licenses no dimension trend, no statement
about learned-critic `omega`, and no reinterpretation of the old runs, which stay in
separate tables, directories and ledgers.
