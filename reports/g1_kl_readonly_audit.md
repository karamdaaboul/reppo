# Read-only KL and recoverability audit of the 64 confirmatory runs

**Status.** Read-only. No training was run, nothing in the run tree was modified, no
threshold was chosen after seeing an outcome. Produced by
`scripts/analysis/kl_readonly_audit.py` and `scripts/analysis/kl_audit_part2.py`.

**Provenance.** Analysis commit `7534b77` on branch `estep-study`. Runs analysed:
the 64 confirmatory ladder runs at `07319d4`. Environment: Python 3.12.14,
JAX 0.5.2, flax 0.10.6; NVIDIA H100 94 GB (SLURM partition `c23g`, account
`rwth2182`). Source artifacts: `/hpcwork/qzi10910/reppo_runs/outputs/conf/*/metrics.npz`
(64 files) and `exports/*_final/meta.json` (66 exported finals).

---

## 1. Code trace: what the logged KL scalar actually is

Every line below is quoted from `src/jaxrl/reppo.py` at `07319d4`, whose `src/` tree
(`fc2eae02`) is **identical** to the commit the runs were launched from.

**Sampling and the two log-probabilities.** `reverse_kl: false` in every run, so the
`else` branch at line 703 is the live one:

```
709: old_pi_action, old_pi_act_log_prob = actor_target_model.actor(
710:     minibatch.obs
711: ).sample_and_log_prob(sample_shape=(n_estep,), seed=key)
712: old_pi_action = jnp.clip(old_pi_action, -1 + 1e-4, 1 - 1e-4)
716: logp_old_i = old_pi_act_log_prob.sum(-1)            # (M, B)
717: logp_theta_i = pi.log_prob(old_pi_action).sum(-1)   # (M, B)
719: old_pi_act_log_prob = logp_old_i.mean(0)
720: pi_act_log_prob = logp_theta_i.mean(0)
722: kl = old_pi_act_log_prob - pi_act_log_prob
```

`sample_and_log_prob` returns `old_pi_act_log_prob` evaluated at the **unclipped**
sample; the clip is applied at 712, *after* that return; and line 717 evaluates the
new policy at the **clipped** action. Defect 1 confirmed: the two terms of the
difference refer to different action points. `n_estep` is 32 in arm B and 16 in arm A
(704--708), so the arms average over different sample counts.

**Shapes and reductions.** `logp_old_i`, `logp_theta_i` are `(M, B)`; `.mean(0)`
reduces over the M samples; `kl` at 722 is **`(B,)` — one scalar per state**, not per
action dimension. It is a sample estimate of a KL and is **not** guaranteed
non-negative.

**The defective branch.** `actor_kl_clip_mode: "clipped"` and `kl_bound: 0.1`
(`config/reppo.yaml:60`, and re-asserted in all five `config/experiment_overrides/*`):

```
850: elif cfg.actor_kl_clip_mode == "clipped":
851:     actor_loss = jnp.where(
852:         kl < cfg.kl_bound,
853:         objective,
854:         kl * jax.lax.stop_gradient(lagrangian) * cfg.reduce_kl,
855:     )
```

Elementwise over the `(B,)` state axis, and above the threshold it **replaces** the
objective rather than adding to it. Defect 2 confirmed.

**The logged scalar.**

```
1002: kl=kl.mean(),
1003: lagrangian=lagrangian,
```

The logged `train/kl` is `kl.mean()` of **exactly the same tensor** the `where` at
851--852 tests. It is a mean — not a maximum, not a sum, not a separate KL
calculation, not an unrelated diagnostic.

**Further reductions before storage.** `metrics = jax.tree.map(lambda x: x.mean(0),
metrics)` (1089) averages over minibatches; `x[-1]` (1100) keeps the **last epoch**;
`x[-1]` (1127) keeps the last iteration of each eval block. So each stored
`train/kl` point is the mean of `kl` over `num_mini_batches x B` states from one
epoch of one iteration, and there are 21 such points per run (`num_eval: 20`).

**The multiplier.** `src/networks/jax_models.py:406-407`:

```
406: def lagrangian(self) -> jax.Array:
407:     return jnp.exp(self.lagrangian_log_param.value)
```

A bare unbounded exponential, unlike `eta()` (411--413) and `beta_mu()` (416), which
are softplus-and-clipped. Defect 3 confirmed. It **is** logged (`train/lagrangian`,
1003) and it **is** serialised (`lagrangian_log_param/value` in every `actor.npz`).

**Estimator diagnostics.** `901: if cfg.log_estimator_diag and not cfg.reverse_kl:`
gates the *computation*, not just the logging, and every confirmatory run set
`log_estimator_diag=false`. Verified empirically: `est_h_norm`, `est_a_norm`,
`est_cos`, `est_rel_l2`, `est_var_proxy`, `est_bias2_proxy`, `est_wdisp_*` are
identically zero in **all 64** runs, as are `grad_norm_actor`, `grad_norm_critic` and
every `eval/episode_return_iqm*` field. There is no omega and no estimator
diagnostic in this dataset.

---

## 2. What the retained artifacts contain

| Artifact | Content | Verdict |
|---|---|---|
| `metrics.npz` (64) | 21 points x 60 keys, incl. `train/kl`, `train/lagrangian`, `train/pi_sigma_{mean,min,max}`, `train/ess*`, `train/eta`, `train/entropy`, `train/q`, `train/value_loss`, `train/abs_pred_action` | retained |
| `train/actor_loss` | `(21, 1, 8192)` — per-position, but already **averaged over 16 minibatches** (line 1089), so position *b* mixes 16 different states | retained but mixed |
| `actor.npz` | `lagrangian_log_param`, `eta_param`, `temperature_log_param` + weights, at `p25`, `p50`, `final` | retained (3 points) |
| `critic.npz`, `normalizer.npz` | full critic and normalizer state | retained |
| optimizer state (Adam moments for any dual) | — | **not present** |
| target actor / target critic | — | **not present** (Amendment A: no target critic exists) |
| per-state KL, per-minibatch KL, `I[KL >= 0.1]` | — | **not present** |
| replay metadata, per-update logs | — | **not present** (only 21 eval-aligned points) |

Field-name search over every meta key and pytree path for `kl|lambda|multiplier|dual|
beta|raw|log|constraint|penalty|actor_kl` returns the rows above and nothing further.

---

## 3. Conservative gate-exposure diagnostics

Per run over the n=21 retained points, with the bound `T = 0.1`:

* `F_run = mean_t[ KL_t >= T ]`
* `overshoot_run = mean_t[ max(KL_t - T, 0) ]`

**The bound that the code trace licenses.** For any finite collection, the mean never
exceeds the maximum. The logged scalar is the mean of exactly the tensor the branch
tests elementwise. Therefore **`logged_KL_t >= 0.1` proves that at least one state
took the gated branch at that logging point.** This needs no non-negativity
assumption, which matters here because the estimator at line 722 can be negative. It
is a **lower bound**: it says nothing about logging points below the bound, and it
does **not** identify how many states fired.

| task | d | arm | F_run med | overshoot med | KL med | KL IQR med | KL q75 med | KL max | longest run above | lambda med |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hopper | 4 | A | 0.405 | 0.00366 | 0.0982 | 0.01371 | 0.1026 | 0.152 | 4.0 | 0.00460 |
| hopper | 4 | B | 0.643 | 0.01890 | 0.1021 | 0.00699 | 0.1047 | 0.465 | 5.5 | 0.3344 |
| walker | 6 | A | 0.881 | 0.00633 | 0.1058 | 0.00618 | 0.1087 | 0.136 | 12.5 | 0.01616 |
| walker | 6 | B | 0.952 | 0.00616 | 0.1043 | 0.00296 | 0.1056 | 0.172 | 17.5 | 0.5635 |
| leap | 16 | A | 1.000 | 0.00889 | 0.1095 | 0.00303 | 0.1107 | 0.117 | 21.0 | 0.01182 |
| leap | 16 | B | 1.000 | 0.01577 | 0.1081 | 0.00249 | 0.1092 | 0.341 | 21.0 | 0.5085 |
| g1 | 29 | A | 0.548 | 0.00207 | 0.1003 | 0.00447 | 0.1029 | 0.139 | 3.0 | 0.00451 |
| g1 | 29 | B | 0.833 | 0.02603 | 0.1045 | 0.03077 | 0.1326 | 0.293 | 12.0 | 0.5042 |

**The single most important number in this table is not about g1.** `F_run > 0` in
**64 of 64 runs**; `F_run >= 0.5` in 54 of 64; the median `F_run` over all 64 runs is
**0.833** and the median of the median KL is **0.1042** against a bound of 0.1. The
trust region is binding, and the defective branch provably fires, in **every arm on
every task** — including all four cells where no return difference was detected. The
defect is a property of the whole dataset, not of g1.

A mean pinned just above the bound with dispersion around it is what one would expect
if roughly half the batch is above threshold at any time, but that reading requires a
symmetry assumption about the per-state KL distribution that the retained data cannot
support. It is stated here as an expectation, not as a measurement.

---

## 4. The g1 anomaly, recomputed from raw curves

Recomputed from the raw `train/kl` arrays rather than from the previously supplied
summary; the summary is confirmed (g1-B IQR 0.0308 vs g1-A 0.0045; q75 0.133).

The **direction** of the arm contrast is g1-specific. B/A ratio of median KL IQR:

| task | d | B/A (KL IQR) | B/A (overshoot) |
|---|---:|---:|---:|
| hopper | 4 | 0.51 | 5.17 |
| walker | 6 | 0.48 | 0.97 |
| leap | 16 | 0.82 | 1.77 |
| **g1** | **29** | **6.89** | **12.6** |

Arm B is *tighter* than arm A on the three tasks with no detected difference, and
6.9x looser on the one task with a difference. g1-B has the largest overshoot and
g1-A the smallest of all eight cells.

**But the KL multiplier is not g1-specific at all.** `lambda` is 35--112x larger in
arm B on **every** task (g1 112x, hopper 73x, leap 43x, walker 35x). Whatever makes
arm B's dual large is general to the arm; only the *dispersion* of the KL is peculiar
to g1.

---

## 5. Associations, without any causal claim

n = 8 seeds per arm. Pooling arms is misleading here: pooled over both g1 arms,
`spearman(KL IQR, return) = -0.738`, but arm B has simultaneously the higher IQR and
the lower return, so the pooled statistic largely re-expresses the arm contrast
itself. **Within** arm the sign reverses:

| g1, arm B (n=8) | rho | p |
|---|---:|---:|
| KL IQR vs final return | **+0.595** | 0.120 |
| overshoot vs final return | **+0.667** | 0.071 |
| lambda median vs final return | -0.429 | 0.289 |
| policy sigma vs final return | -0.833 | 0.010 |

Within arm B on g1, the seeds with the **most** trust-region violation returned the
**highest**. The two best seeds (106: 27.0, 108: 23.1) have the two largest IQRs
(0.0404, 0.0405); the worst seed (101: 9.4) has one of the smallest. This is n=8 and
not significant, but it is the opposite of what a "the gate broke g1" story predicts,
and it is reported because it is what the data says.

The one nominally significant within-arm association is with policy width, not with
the KL: narrower final sigma goes with higher return in g1-B (rho = -0.833).

---

## 6. Recoverability matrix

| Target | Status | Basis |
|---|---|---|
| `d` (action dim) | **exactly recoverable** | `meta.json:action_dim`, all 64 |
| `sigma` (policy width) | **exactly recoverable** | `train/pi_sigma_{mean,min,max}`, 21 pts; actor weights at 3 fractions |
| `Q_phi` | **exactly recoverable** | `critic.npz` at `p25`/`p50`/`final` |
| `grad Q_phi` | **exactly recoverable** | differentiable from `critic.npz` offline |
| estimator variance w.r.t. `Q_phi` | **exactly recoverable** (offline) | already measured by the same-critic probe |
| true `Q^pi` | **not identifiable** | no MC return rollouts retained; would need new environment interaction |
| critic error `e = Q_phi - Q^pi` | **not identifiable** | follows from `Q^pi` |
| `grad e` | **not identifiable** | follows |
| **`omega = ||grad e|| / ||e||`** | **not identifiable** | follows. There is **no** strict read-only route to omega |
| `Omega_z` (centred padded-subspace frequency) | recoverable **as a different quantity**, only on padded runs | Probe 1; it is *not* omega |
| aggregate KL threshold exposure | **exactly recoverable** at the 21 logged points | `train/kl` + the mean-vs-max bound |
| **elementwise (per-state) gate rate** | **not identifiable** | see below |
| KL multiplier `lambda` trajectory | **recoverable at 21 points**; raw parameter at 3 checkpoint fractions | `train/lagrangian`; `lagrangian_log_param` |
| `lambda` Adam / optimizer state | **not present** | no `opt_state` in any export |
| counterfactual corrected-code performance | **not identifiable** | requires a causal intervention (corrected reruns) |

**Explicitly excluded as omega substitutes**, per the scientific constraint: critic
gradient norm, critic Hessian norm, Bellman residual, inter-critic disagreement,
target-network disagreement, cross-seed disagreement. None of them identifies
`||grad e||/||e||` without an independent estimate of `Q^pi`. None is computed here
as a proxy for omega.

**Why the elementwise rate is not identifiable.** One route was attempted and
rejected. In `clipped` mode a gated state contributes `lambda * kl_b` and an ungated
one contributes `objective_b`, so if `train/actor_loss` were per-state one could
partition by magnitude. It is not: line 1089 averages it over 16 minibatches first,
and position *b* mixes 16 different states. Even ignoring that, the bands overlap —
on g1 arm A at the first eval, 49.9% of positions fall inside the nominal
"gated" band `[0.1*lambda, 50*lambda]` purely because `lambda ~ 0.013` makes that band
`[0.0013, 0.63]`, which swallows most of the ordinary objective range. The test has no
discriminating power and its output is not reported as a gate rate.

---

## 7. The five questions

**1. Is there a strict read-only route to omega?**
**No.** `omega = ||grad e||/||e||` requires `e = Q_phi - Q^pi`, and `Q^pi` is not
recoverable from any retained artifact — no Monte-Carlo return estimates were stored
and no rollout data was kept. Obtaining it requires new environment interaction from
the saved checkpoints, which is not read-only. The padded-coordinate route (Probe 1/4)
identifies the centred *z*-varying error field `Omega_z`, which is a different object:
it says nothing about constant-in-*z* bias or about the real-coordinate error.

**2. Can nonzero activation of the defective branch be lower-bounded?**
**Yes, and it is nonzero everywhere.** Because the logged scalar is proven to be the
mean of the same tensor the branch tests, `logged_KL_t >= 0.1` implies `max_b kl_b >=
0.1`, so at least one state took the gated path. This holds at a median 83% of logging
points, and at >=1 point in **64/64 runs**. The bound is conservative in both
directions: points below the bound are uninformative, and the number of firing states
is not bounded.

**3. Can the elementwise firing rate be recovered?**
**No.** See section 6. `I[kl >= 0.1]` was never stored, the per-state `kl` was reduced
to a mean at line 1002, and the one surviving per-position array is pre-averaged over
minibatches and does not separate the two branches.

**4. Can the KL multiplier be recovered?**
**Partially, and better than expected.** Its *value* trajectory is recoverable at the
21 logged points for all 64 runs (`train/lagrangian`), and the raw
`lagrangian_log_param` is serialised at three checkpoint fractions. Its **Adam state
is not present**, and there is no per-update trajectory — only 21 points out of
~400 iterations x 8 epochs x 64 minibatches of actual updates.

**5. How strongly is the g1 return result confounded by the observed behaviour?**
The g1 return comparison is **confounded** and should be treated as such. Arm B on g1
runs with a 6.9x wider KL dispersion, 12.6x the overshoot and a 112x larger multiplier
than arm A, under a branch that is provably active and that *replaces* the policy
objective when it fires. The two arms therefore did not optimise the same objective on
the states where the branch was active, and a return difference measured under that
condition cannot be attributed to the estimator contrast the experiment was designed
to test. That is a construct-validity problem, and it is sufficient on its own to
withhold the causal claim.

It is **not**, however, positive evidence that the defect *caused* the gap, and this
audit does not support that statement:

* the branch is provably active in **all four tasks and both arms**, including the
  three tasks where no difference was detected, so activation alone does not
  discriminate g1;
* the large multiplier is common to arm B on every task, so it does not discriminate
  g1 either;
* within g1 arm B, greater dispersion and greater overshoot are associated with
  *higher* return (rho = +0.60, +0.67, n=8), which is the wrong direction for the
  causal story;
* only the *dispersion* asymmetry is g1-specific, and dispersion is not the quantity
  the branch tests.

The correct statement is: **the g1 result is confounded by KL-gate behaviour that
differs sharply between arms on that task, and the confound cannot be removed from the
retained data. Whether the defect caused the gap is unidentified and requires
corrected reruns.**

---

## 8. Reproduction

```bash
cd ~/repos/reppo
./.venv/bin/python scripts/analysis/kl_readonly_audit.py reports/artifacts
./.venv/bin/python scripts/analysis/kl_audit_part2.py    reports/artifacts
```

Outputs: `reports/artifacts/kl_per_run.csv` (64 rows), `kl_by_task_arm.csv`,
`within_arm_assoc.csv`, `kl_curve_<run>.npy` and `lag_curve_<run>.npy` (64 each),
and figures `fig_kl_curves.png`, `fig_lagrangian.png`, `fig_g1_dispersion.png`.
