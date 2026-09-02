# Preregistration: faithful-repair operator replication

Committed **before any confirmatory run is launched** and before any corrected return
exists. Immutable once committed.

| | |
|---|---|
| **Correction-code commit** | `cfbd8dd` "Repair the same-point log probability behind default-off flags" |
| Design lock | `796e66e` `docs/faithful_repair_design.md` |
| Source trace | `a3d2352` `reports/corrected_replication_code_trace.md` |
| Repo HEAD at preregistration | recorded in §11 below |
| Legacy parity evidence | job `3442080`; 36/36 arrays byte-identical, max abs diff `0.0`, all metric curves identical to `07319d4` |

## 1. What the repaired execution changes — **two** behaviour-affecting details

**Correction to `docs/faithful_repair_design.md` §0.** That document states the repair
"makes the faithful arm differ from *released* REPPO on exactly one point". **That is
wrong and is corrected here.** The confirmatory arms differ from released REPPO in
**two** behaviour-affecting ways:

1. **Same-point likelihood evaluation.** The pre-squash latent `y_i` is materialised;
   both the old- and new-policy log probabilities are computed from that same `y_i`; no
   clip is applied on this path. Released code evaluates the old-policy term at the
   unclipped sample and the new-policy term at the clipped action, and evaluates the
   WML critic at the clipped action.
2. **Fresh minibatch innovations.** The actor-sampling key is split inside the
   minibatch scan. Released code reuses one key for every minibatch of an epoch, so the
   standard-normal array is bit-identical across the whole epoch.

Both are enabled together in every confirmatory run. **This experiment cannot attribute
any outcome to either one individually**; separating them would require a randomized
ablation that is not part of this design.

Unchanged from released REPPO: the piecewise policy/KL gate, the exponential log-space
KL multiplier, the forward KL orientation, `M`, `eps_e`, `kl_bound`, the optimiser, the
target-actor refresh, and every hyperparameter in §3.

## 2. Design

| | |
|---|---|
| Tasks | `WalkerRun` (d=6), `G1JoystickFlatTerrain` (d=29) |
| Arms | `PW-1-faithful-repair`, `WML-32-faithful-repair` |
| Seeds | **301–308**, eight paired seeds, a fresh reserved namespace |
| Total | **32 runs** = 2 tasks × 2 arms × 8 seeds |
| Excluded | **HumanoidRun** — not reproducible from retained artifacts (`reports/corrected_replication_code_trace.md` §4). No replacement task is substituted. |

Required flags in every confirmatory run:
`faithful_same_point=true`, `fresh_minibatch_key=true`, `log_faithful_diag=true`.

**Arm sample counts.** `PW-1` uses **one** actor sample for its objective; `WML-32`
uses **32** critic-scored samples. These are **not** equal-sample or equal-query and
are never described as such. `PW-32` is not a training arm in this experiment.

## 3. Frozen configuration

| | WalkerRun | G1JoystickFlatTerrain |
|---|---|---|
| env args | `env=mjx_dmc env.name=WalkerRun` | `env=mjx_humanoid env.name=G1JoystickFlatTerrain env.asymmetric_obs=false` |
| overrides | `experiment_overrides=mjx_dmc_large_data` | `experiment_overrides=mjx_humanoid_large_data` |
| **frozen alpha (identical across arms)** | **`0.014509912580251694`** | **`0.00020752247655764222`** |
| `update_entropy_lagrangian` | false | false |
| horizon | **52 297 728** steps | **52 297 728** steps |
| `num_eval` | **20** (21 logged points) | **20** |
| `M`, `eps_e`, `kl_bound` | 32, 0.5, 0.1 | 32, 0.5, 0.1 |
| `num_mini_batches` / `num_epochs` | 64 / 8 | 16 / 8 |

No learned-alpha arm.

## 4. Primary performance analysis — fixed now

* **Final score.** Mean of the **final three logged evaluations** (indices 18, 19, 20
  of 21). Fixed before any curve is inspected.
* **Secondary score.** The last logged evaluation alone — the definition the 64-run
  study locked — reported for comparability with the old table.
* **Primary contrast orientation.** `PW-1 minus WML-32`, per task, paired within seed.
  A **positive** difference means the pathwise arm scores higher.
* **Independent unit.** The **seed pair**. Evaluations and environment lanes are never
  treated as independent replicates.
* **Uncertainty.** Paired percentile bootstrap over the 8 seed-level differences,
  10 000 resamples, `np.random.default_rng(20260902)`; 95% interval.
* **Sign count.** Exact two-sided paired sign test over 8 pairs (2^8 = 256; minimum
  attainable two-sided p is 0.0078).
* Also reported: all eight paired seed scores, both arms' IQMs, paired median, paired
  mean, and complete learning curves.
* **Forbidden:** no cross-task Spearman, no normalised dimension anchors, no
  action-dimension trend claim, no post-hoc normalisation, no equivalence margin
  introduced after the fact.

## 5. Sampled-versus-analytic KL comparison — preregistered, diagnostic only

The exact analytic pre-squash Gaussian KL is logged alongside the sampled `M`-sample
estimate. **It never enters training, the gate, the dual, or any exclusion.** Reported
per task and arm, aggregated over seeds:

* **bias** — mean of (sampled − analytic);
* **RMSE** — root mean square of (sampled − analytic);
* **correlation** — Pearson and Spearman between sampled and analytic;
* **sampled standard error** — the within-state SE of the `M`-sample mean, and its
  ratio to the analytic KL;
* **false-positive gate rate** — fraction of states with sampled `≥ 0.1` but analytic
  `< 0.1` (the operator objective was discarded when the true KL was inside the bound);
* **false-negative gate rate** — fraction with sampled `< 0.1` but analytic `≥ 0.1`;
* **fraction receiving the operator branch** and **fraction receiving the KL-only
  branch**, per arm.

**Registered expectation, stated before launch:** the M=32 sampled KL is a
high-variance estimate of the quantity the gate thresholds — a bench measurement gave a
standard error `1.75×` the KL's own median value. Non-trivial false-positive and
false-negative gate rates are therefore *expected* and are a property of the published
design, not a defect and not an outcome of the repair. This expectation does not gate
any decision.

## 6. Missing runs, failures, and exclusions

* **Every attempted seed stays in the ledger**, with its outcome, whatever happens.
* **Infrastructure failure** (node fault, pre-empt, OOM, filesystem error, walltime
  exhaustion) may be rerun **only with the identical seed and identical configuration**.
  The rerun is recorded as an additional ledger row referencing the failed attempt.
* **Algorithmic divergence is an outcome, not a failure.** NaN losses, entropy
  collapse, or return collapse are reported as results and are **never** silently
  replaced, reseeded, or excluded.
* **Exclusion policy:** none. No run is excluded from the primary analysis for any
  reason. If a run cannot produce a final score at all, that is reported explicitly and
  its pair is reported as incomplete rather than dropped.
* If fewer than 8 complete pairs exist for a task, the paired analysis is reported at
  the reduced pair count with the shortfall stated; the missing pairs are not imputed.

## 7. No interim comparison

Until all 32 runs are complete, monitoring is restricted to **infrastructure health and
integrity only**: job state, wall clock, non-finite detection, evaluation count,
checkpoint presence, diagnostic non-placeholder status. **No return comparison between
arms is computed, inspected, or reported before the full set is complete.**

## 8. Integrity tests after completion

Every expected run exists exactly once; no duplicate task–arm–seed; no output-directory
collision; correct commit; correct configuration hash; correct frozen alpha; correct
horizon; equal evaluation counts; no non-finite values; checkpoints reload; final
evaluation reproduces on reload; diagnostic arrays contain real values, not
placeholders; optimizer states present.

## 9. Frozen same-critic diagnostic (mandatory, after the runs)

At the final corrected checkpoints, on identical states with common random numbers:
`PW-1`, `PW-32`, centred `ZO-32`, the exact nonlinear centred WML component `c`, and
the full standardized WML mean score `v`. This separates the algorithmic operator
difference from the action-query budget, the nonlinear softmax weighting, and the
uniform component. `PW-1` is never treated as `PW-32`.

## 10. Allowed and forbidden conclusions

| outcome | permitted wording |
|---|---|
| corrected `g1` difference keeps its sign, interval excludes zero | "A task-level operator difference persists under the corrected common trust-region implementation." |
| corrected interval includes zero | "The original g1 difference does not reproduce as a detected difference under the corrected implementation." |
| sign reverses | report the reversal, with no mechanistic attribution |
| WalkerRun | report the interval; it is a no-difference control, and no equivalence margin is introduced |

**Forbidden under every outcome:** that Claim 4 caused any result; that a particular
old defect caused the original result; any attribution to the same-point repair or the
PRNG change individually; any dimension trend; any claim that learned-critic `omega`
was measured; any pooling of these runs with the old ones; any reinterpretation of the
old runs, which stay in separate tables, directories and ledgers.
