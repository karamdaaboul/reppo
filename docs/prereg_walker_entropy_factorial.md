# Preregistration — Walker entropy factorial

Registered before any run of either new arm. Append-only: once committed, this text
is not edited. Two new arms, 16 jobs. The existing Walker PW-1 and WML-32 baselines
are **not** rerun; they supply the other two cells of the factorial.

## 1. Baseline of record

Re-verified from run artifacts, not from today's Hydra defaults.

| field | value |
|---|---|
| `BASE_SHA` | `1c6259ef959400eba617e1d1392bfaf58248688c` (unanimous over all 16 completed ledger rows) |
| Seeds | 301-308 |
| Task | `WalkerRun`, `vmin 0`, `vmax 150`, `num_bins 151`, `hl_gauss true` |
| Executed protocol | **128 minibatches x 4 epochs**, 400 outer iterations, 52,297,728 env steps |
| `ent_start` | `0.014509912580251694`, `update_entropy_lagrangian: false`, so alpha is frozen |
| `ent_loss_per_dim` | `false` |
| KL | `kl_bound 0.1`, `reduce_kl true`, `actor_kl_clip_mode clipped`, `reverse_kl false` |
| WML | `estep_num_samples 32`, `eps_e 0.5`, `mstep_decoupled false`, `sqrt_rho 1.0` |
| `min_std` reaching the model | **0.1** (the config's `actor_min_std: 0.0` does not reach it) |
| Optimizer | `lr 3e-4`, `max_grad_norm 0.5`, `anneal_lr false`, `gamma 0.99`, `lmbda 0.95` |
| Eval | `num_eval 20`, giving 21 logged evaluations |

The baseline config contains an `experiment_overrides` block specifying 64 minibatches
x 8 epochs which is **overridden at runtime** by the base block's 128 x 4, because
`_self_` is last in `defaults:`. Every scientific value is therefore pinned explicitly
on the launch command line for both new arms so Hydra inheritance cannot alter it
silently. LEAP previously failed in exactly this way.

## 2. The two interventions

Both sit behind new config flags that default to `false`. Both branch in Python, so
with the defaults no operation is emitted and the baseline is bit-identical.

### PW-H — `pw_drop_actor_entropy: true`

The pathwise objective is `log_prob * sg(temperature) - value`. PW-H drops the
`log_prob * sg(temperature)` contribution, leaving `-value`.

alpha is **not** set to zero and is untouched everywhere else: the critic soft target,
the `weighted_mle` `q_spread` diagnostic, and the logged value all keep it. Unchanged:
pathwise-Q term, KL gate, architecture, optimizer, lr, seeds, budget, evaluation,
critic, candidate generation.

### WML+H — `wml_add_actor_entropy: true`

**HYBRID ABLATION. This is neither standard MPO nor standard REPPO**, and is labelled
as such in the code, here, and in every report that uses it.

The weighted-MLE objective is `-sum_i w_i * logp_theta_i` over the M=32 E-step
candidates drawn from `pi_old`. WML+H adds the same entropy contribution the pathwise
arm carries, with the sign matching the repository's minimisation convention.

The added term uses **one fresh reparameterised sample from the current policy**, not
the E-step candidates and not their weights. Concretely it reuses the `log_prob`
tensor already computed from `pi.sample_and_log_prob(seed=akey)` before the arm
branch: the same tensor, same RNG key, and same squashed-policy log-probability
implementation the pathwise arm uses. Shape is `(batch,)` after the sum over action
dimensions, matching `objective`. Reusing the already-drawn sample consumes no new
randomness, so the RNG stream is unchanged.

Placement: the contribution enters `objective`, which is the **gate-open** branch of
the per-state switch. Gate-closed states keep the existing KL-reduction branch
untouched in both arms.

Unchanged: weighted MLE term, eta dual, `eps_e`, M, KL threshold, critic, alpha value,
optimizer, candidate generation, seeds, budget.

## 3. Primary return statistic — not redefined

Taken verbatim from `docs/prereg_corrected_operator_replication.md` section 4, commit
`7edb8e8c3d358eba04e569223a3860a3b4fba85d`:

```
per-seed scalar = mean of the final three logged evaluations (indices 18, 19, 20 of 21)
paired effect   = PW minus WML, paired within seed; positive means pathwise higher
bootstrap       = paired percentile over the 8 seed differences, 10,000 resamples
RNG             = np.random.default_rng(20260902)
CI              = 95% percentile interval
```

Recorded fact, established by recomputation rather than by choice: that
preregistration fixes the bootstrap but does not name the statistic it is applied to.
Bootstrapping the **paired median** reproduces its frozen intervals exactly on all
three tasks, while bootstrapping the mean reproduces none of them. The paired median
is therefore the statistic of record.

## 4. Primary mechanistic outcome

Final fixed-bank median pre-tanh sigma on the canonical neutral Walker bank,
`reports/artifacts/walker_fixed_state_bank.npz`, sha256
`8adfeb0bf70bddcdbd64a84b972b4dbd62617c64e1c948589a7adc30cf64aa21` (verified before
use). Reported **continuously** for all four cells. **No success threshold is
defined.**

## 5. Also reported

Paired seed-level width; `P(|a|>0.95)`; `P(|a|>0.99)`; return under the frozen
statistic above; gate diagnostics including `fr_gate_operator` and the `fr_kl`
quantiles; and ESS and eta for both weighted_mle cells.

## 6. Stated analytic expectation, recorded before results

With `F := -dL/dlog_sigma`, so `F > 0` widens:

```
F_H = alpha [ 1 - 2 sigma^2 E(sech^2 y) ]
```

Since `E[sech^2] <= 1`, we have `F_H >= alpha(1 - 2 sigma^2) > 0` for any
`sigma < 0.707` regardless of mu. Walker PW operates at 0.44-0.48, so **PW's entropy
term is widening there**.

Therefore this preregistration does **not** predict that PW-H widens. The local
derivative predicts the opposite: removing a widening term should narrow or leave
flat. PW-H narrowing, or not moving, is **not a failure**.

At WML's operating sigma the same term is strongly contracting, so **WML+H narrowing
is largely predetermined by the derivative** and is not by itself informative. The
informative outcomes are (a) the **return**, and (b) whether **WML+H remains wider
than PW+H**.

## 7. Stated caveat, recorded before results

alpha is **PW-calibrated by construction**. It was fixed from PW's learned runs
(`ent_start` frozen with `update_entropy_lagrangian: false`), and the weighted_mle arm
has never carried an entropy term. WML+H therefore tests matched entropy **geometry**,
not WML's own optimal temperature. A poor WML+H return is not evidence that entropy
cannot help that operator; it is evidence about this particular alpha.

## 8. Factorial questions

Reported continuously, with no pass/fail thresholds.

* **Q1** does the PW-WML width gap persist with entropy ON in both cells?
* **Q2** does it persist with entropy OFF in both cells?
* **Q3** how much does adding or removing entropy move each operator, in width and in
  return?

The four cells are: PW-1 (entropy on, baseline), PW-H (entropy off, new),
WML-32 (entropy off, baseline), WML+H (entropy on, new).

## 9. Execution

| arm | flag | seeds | partition | rationale |
|---|---|---|---|---|
| PW-H | `pw_drop_actor_entropy=true` | 301-308 | `c23g` | matches its own baseline, array `3444831_8..15` |
| WML+H | `wml_add_actor_entropy=true` | 301-308 | `c25g` | matches its own baseline, array `3444832_8..15` |

Partitions are matched to each arm's own baseline rather than standardised, because
the two baselines themselves ran on different partitions. A single partition is
passed per job; SLURM rejects a comma-separated list for this account.

The ledger records the **actual execution SHA** per run, taken from the completed
record, not the preregistration commit.

## 10. Stopping rules

No tuning after launch. No added seeds. No run cancelled because its return looks
bad. No inspection of one arm's early results to alter the other. Technical failures
may be retried only with an identical scientific config, preserving the failure logs
and recording the retry as an additional ledger row.
