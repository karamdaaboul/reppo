# Prospective cross-task generalisation test of the M-sweep phenomenon

**Label: PROSPECTIVE, NOT BLIND.** The HumanoidRun M-sweep results and the
M-sample-count audit have been seen. The WalkerRun PW-1 and WML-32 outcomes at
seeds 301-308 have been seen. Nothing about WalkerRun at `M = 128` or `M = 512`
has been observed, and this document is committed before any such run starts.

---

## 1. Origin

The hypothesis was formed on **HumanoidRun**, seeds **201-218**, which
`ledger/README.md:13` places in the **201+ EXPLORATORY / ALGORITHM DEVELOPMENT**
namespace: *"may become confirmatory evidence: never."* Sources:
`reports/m_star_study.md` and `reports/m_sample_count_audit.md`.

The exploratory finding is that weighted-MLE return collapses as
`estep_num_samples` rises: arm-A control mean `738.614`, `M = 32` mean `666.174`,
`M = 128` mean `137.16`, `M = 512` mean `10.98`. The `M = 32` to `M = 128` drop is
**529.01** return points.

**This is a generalisation test on a different task, not a same-task
replication.** WalkerRun is `d = 6`; HumanoidRun is `d = 21`. **No exploratory
return is pooled into the WalkerRun statistic or CI; the exploratory HumanoidRun
result is used only to formulate the hypothesis and to fix `T = 100`
prospectively.**

The WalkerRun **PW-1** and **WML-32** control outcomes at seeds 301-308 are
already known and are reported in `reports/corrected_operator_replication.md`.
**The only new information this study produces is `M = 128` and `M = 512`.**

## 2. Design

* Task **WalkerRun**, `actor_update_mode = weighted_mle`.
* `estep_num_samples` in **{128, 512}**, seeds **301-308**, **16 runs**.
* **Controls are not rerun.** The existing PW-1 and WML-32 runs at seeds 301-308
  from `reports/corrected_operator_replication.md` are used as-is.
* Every other setting is identical to the WML-32 arm at 301-308, which is the
  faithful-repair arm. Per `ledger/runs_faithful_repair.jsonl`, that means:

  ```
  hyperparameters.actor_update_mode=weighted_mle
  hyperparameters.update_entropy_lagrangian=false
  hyperparameters.ent_start=0.014509912580251694
  hyperparameters.faithful_same_point=true
  hyperparameters.fresh_minibatch_key=true
  hyperparameters.log_faithful_diag=true
  ```

  with `eps_e = 0.5`, `kl_bound = 0.1`, `num_eval = 20`, `sqrt_rho = 1.0`,
  `freeze_sigma = null`. **All eight WML-32 controls share one frozen entropy
  coefficient**, verified across all eight ledger rows:
  `ent_start = 0.014509912580251694`, identical in every row. It is written here
  as that scalar rather than as a per-seed lookup, and is not recomputed.

* **Disclosed asymmetry, untouched.** The pathwise arm uses `n_estep = 16` for its
  KL estimate while the weighted-MLE arm uses `estep_num_samples`. This study does
  not change that. It means PW-1 and WML-`M` never share a sample count, so the
  contrast `G` confounds operator with sample count. Section 6 shows why the
  primary statistic is nevertheless free of the pathwise arm.

* **Hardware matching, read from the completion records and SLURM accounting.**
  The eight WML-32 controls did **not** all run on one machine class. From
  `ledger/runs.d.faithful_repair/fr-walker-WML32-s30*.json` and `sacct`:

  | Seed | SLURM job | Partition | GPU string | wall s |
  |---|---|---|---|---|
  | 301 | `3444832_12` | `c25g` | NVIDIA H100 80GB HBM3 | 1323 |
  | 302 | `3444831_12` | `c23g` | NVIDIA H100 | 1377 |
  | 303 | `3444832_13` | `c25g` | NVIDIA H100 80GB HBM3 | 1313 |
  | 304 | `3444831_13` | `c23g` | NVIDIA H100 | 1380 |
  | 305 | `3444832_14` | `c25g` | NVIDIA H100 80GB HBM3 | 1324 |
  | 306 | `3444831_14` | `c23g` | NVIDIA H100 | 1370 |
  | 307 | `3444832_15` | `c25g` | NVIDIA H100 80GB HBM3 | 1311 |
  | 308 | `3444831_15` | `c23g` | NVIDIA H100 | 1381 |

  The split is by seed parity: **odd seeds on `c25g`, even seeds on `c23g`**, four
  each. The two classes differ in wall-clock by about 4.5% (`c25g` 1311-1324 s,
  `c23g` 1370-1381 s), so they are not interchangeable machines.

  **Requirement.** There is no single "control hardware" to match. Each new run for
  seed `s` is submitted to **the same partition as that seed's WML-32 control**:
  odd seeds to `c25g`, even seeds to `c23g`. The primary statistic is a per-seed
  difference `R_{WML32,s} - R_{WML-M,s}`; if the `M` run used a different machine
  class from its own control, hardware would enter that difference directly. The
  smoke test prints the partition, the GPU string and `memory.total` for each arm
  and asserts the partition against this table.

  **`M = 128` follows this pairing exactly.** It carries the primary statistic, which
  is the within-seed difference `R_{WML32,s} - R_{WML128,s}`, so its runs go to the
  same partition as their own controls: odd seeds `c25g`, even seeds `c23g`.

  **`M = 512` is placed on `c23g` for all eight seeds, and the mismatch is
  disclosed.** At the time of committing this document `c25g` had no free capacity
  -- the memory probe queued there was given a scheduler start estimate about 11
  hours out, with 19 nodes `mixed-`, 4 `reserved`, 2 `down`, 4 `drained`. Rather
  than wait on it or assert an unmeasured card, the whole `M = 512` arm runs on
  `c23g`.

  Consequences, registered in advance:

  * **The primary statistic is unaffected.** `Delta_128` never reads an `M = 512`
    return, and every `M = 128` run stays paired to its own control's partition.
  * **Condition (iii) is hardware-mismatched on the odd seeds.** For seeds 301,
    303, 305 and 307 the `M = 512` run is on `c23g` while its `M = 128` counterpart
    is on `c25g`, so `G_{512,s} - G_{128,s}` on those four seeds contains a machine
    difference as well as an `M` difference. The even seeds 302, 304, 306, 308 are
    within-partition on both arms. This is reported with condition (iii) whenever it
    is evaluated, and condition (iii) is never presented as within-hardware.
  * The Section 8 infeasibility clause remains registered but is **not triggered**:
    `M = 512` is feasible under the Section 4.2 budget on `c23g`, as measured below.

* **Config-resolution note, disclosed not fixed.** `config/reppo.yaml` lists
  `_self_` last in its `defaults:`, so the base `hyperparameters` block overrides
  `experiment_overrides`. With `experiment_overrides=mjx_dmc_large_data` the
  resolved values are `num_mini_batches = 128` and `num_epochs = 4` (base), **not**
  the `64` and `8` written in `mjx_dmc_large_data.yaml`; only `kl_bound = 0.1`
  survives because the base block lacks that key. This applies identically to the
  existing controls and to every arm here, so it does not bias the contrast. It is
  recorded because the resolved numbers, not the override file, are what Section 4
  asserts. **No code change is made.**

## 3. Code equivalence

The corrected WML-32 runs at seeds 301-308 launched at
**`1c6259ef959400eba617e1d1392bfaf58248688c`**, identical across all eight
completion records in `ledger/runs.d.faithful_repair/fr-walker-WML32-s30*.json`.

*Ledger caveat.* The **pre-launch** rows in `ledger/runs_faithful_repair.jsonl`
carry `git_sha = 7edb8e8c`, which is the **preregistration** commit, not the launch
tree. The completion records carry the launch commit. `src/` is byte-identical at
both (`399411d5862ff0d6837fc33adb40706ac5a7ab2a`), so the distinction changes no
result, but the field name is misleading and the completion record is authoritative.

The launch SHA is **not** the parity parent at `4eb4713`, and `src/` has since
changed (`399411d5` -> `2e604b3c`). So the earlier 8/8 parity result is **not**
cited here. An equivalence run was made instead, at the arm's own configuration
rather than at defaults, because `faithful_same_point=true` selects a different
E-step branch than the defaults path that parity exercised.

**SLURM job 3516153**, `slurm/m_sweep_equivalence.sh`, one H100, seed 301,
`total_time_steps = 2,621,440` (20 iterations), all six faithful-repair flags set
as above:

| Comparison | `actor.npz` | `critic.npz` |
|---|---|---|
| `4eb4713` vs `4eb4713` (self-control) | IDENTICAL | IDENTICAL |
| `1c6259e` vs `4eb4713` | **IDENTICAL** | **IDENTICAL** |

The self-control runs first: without it a match proves nothing, because two runs
of identical code disagreeing on this GPU would void the comparison. Final return
`528.604`, `alpha_ent 0.01451`, `alpha_kl 0.20316`, `ess 16.85` in every run.

**No behavioural difference. The study proceeds.**

## 4. The intervention, and what is held fixed

### 4.1 `estep_num_samples` is an operational M intervention

This study changes one configuration field. It does **not** change a mechanism,
and **no mechanism is inferred** from any result: what is measured is the
end-to-end effect of `M` on return, not an attribution of that effect to any one
of the channels traced below.

**Source.** `config/reppo.yaml:73` (`estep_num_samples: 32`), bound to the
dataclass field `src/jaxrl/reppo.py:117` (`estep_num_samples: int = 32`).

**Single read in the training path.** The field is read exactly once,
`src/jaxrl/reppo.py:813`:

```python
n_estep = (cfg.estep_num_samples
           if cfg.actor_update_mode == "weighted_mle"
           else 16)
```

Everything downstream reads `n_estep`, never the config field. Every site, and the
quantity it sets:

| Line | Quantity that reads `n_estep` |
|---|---|
| `831` | pre-tanh cloud on the faithful path, `u_i = normal(akey, (n_estep, ...))` -> `y_i`, `old_pi_action` |
| `879` | retained pre-tanh draw, `sample_shape=(n_estep,)`, weighted-MLE branch |
| `891` | `sample_shape=(n_estep,)`, non-weighted-MLE branch |
| `914` | `_nkl = min(cfg.kl_num_samples, n_estep)` -- inert here, `kl_num_samples = null` |
| `944` | `critic_obs_i` broadcast to `(n_estep, ...)` -> `q_i`, the E-step critic queries |
| `1003` | decoupled M-step cloud, `u_i = mu_old + sg_old * normal(key, (n_estep, ...))` |
| `1061` | `_cobs` broadcast -> `q_spread` diagnostic |
| `1152` | `_cobs_i` broadcast -> centred-ZO diagnostic |
| `1169` | `zo_diagnostics(..., n_estep, ...)`, logs `est_M` |

Outside the training path: `scripts/train_and_export.py:231-232` (export tag) and
`:258` (recorded into `meta.json`).

**Every quantity that therefore changes with `M`:**

1. the number of pre-tanh action samples drawn per state;
2. the number of critic queries per state, `q_i`;
3. the self-normalised E-step weights `w_i`, and hence `ESS` and `w_max`;
4. the `eta` dual, which is solved against `q_i` over those `M` samples;
5. the weighted-MLE fit target;
6. **the sampled KL estimate.** `logp_old_i` and `logp_theta_i` are the per-sample
   log-probabilities of that **same** cloud, and the KL fed to the gate and to the
   dual is their mean over the sample axis (`reppo.py:916-919`,
   `kl = old_pi_act_log_prob - pi_act_log_prob`). So `M` sets the Monte-Carlo
   sample count of the KL estimator as well as of the E-step. Raising `M` lowers
   that estimator's variance, which can change when the KL gate fires and how the
   KL dual moves, independently of anything the E-step does;
7. the diagnostics `q_spread`, the centred-ZO decomposition, and `est_M`.

Channel 6 is called out because it is easy to describe this intervention as
"E-step only". It is not: the KL estimator shares the cloud. **This document draws
no inference about which of channels 1-7 produces any observed effect.**

### 4.2 Budget invariance

`estep_num_samples` enters no batch, epoch, or iteration count. The following must
be **byte-identical across `M = 32`, `128`, `512`**:

| Quantity | Value |
|---|---|
| states per iteration (`num_envs` x `num_steps`) | `1024 x 128 = 131,072` |
| states per minibatch (`131072 / num_mini_batches`) | `131072 / 128 = 1,024` |
| actor and critic updates per iteration (`num_epochs x num_mini_batches`) | `4 x 128 = 512` |
| iterations | `399` |
| total actor and critic updates | `399 x 512 = 204,288` |
| env steps | `52,297,728` |

The smoke test in Section 11 asserts every row for each arm against the WML-32
run's values and fails loudly on any mismatch.

**If `M = 512` does not fit in memory without changing any of these, STOP and
report. The batch is not shrunk, the minibatch count is not raised, and the
iteration count is not cut.** A memory failure ends the study at that arm; it does
not license a redesign.

**`M = 512` memory probe, run before this document was committed.** One iteration
at the full registered batch geometry above, arm configuration as Section 2,
`num_eval = 1` solely so `reppo.py:1464` keeps `eval_interval >= 1` at a single
iteration, outputs discarded. `slurm/memprobe.sh`:

| Partition | GPU | `memory.total` | peak `memory.used` | Verdict |
|---|---|---|---|---|
| `c23g` | NVIDIA H100 | **95,830 MiB** | **9,175 MiB** | **FITS**, exit 0, job `3517772` |
| `c25g` | -- | -- | not measured | job `3517771` cancelled unrun; see below |

Peak usage is **9.6%** of the `c23g` card, an 8.7x margin. `M = 512` therefore fits
without touching any quantity in the table above, and the STOP condition is not
triggered.

The `c25g` probe never ran: no capacity, ~11 h scheduler estimate. Because
Section 2 places the whole `M = 512` arm on `c23g`, `c25g` feasibility at
`M = 512` is **not required and is not claimed** here. `M = 128` does run on
`c25g` for the odd seeds; it allocates strictly less than `M = 512`, and Section 11
requires the smoke test to confirm the budget assertion on that partition before
launch. No `c25g` memory figure is asserted by this document.

## 5. Export tag

`scripts/train_and_export.py` appends `_m{M}` to the tag whenever
`estep_num_samples != 32`. Resolved tags:

```
M = 32   WalkerRun_weighted_mle_s{seed}          (existing control, untouched)
M = 128  WalkerRun_weighted_mle_m128_s{seed}
M = 512  WalkerRun_weighted_mle_m512_s{seed}
```

The tag uniquely encodes `M`, so no arm can overwrite another. This closes the
hazard registered in `docs/prereg_m_sweep_dmc.md` Sec. 3.1, which the two Track C
smoke runs confirmed live by both using seed 299 and overwriting one another. The
`M = 32` tag is unsuffixed by design, so every published export stays byte-stable.
**No stop condition is triggered.**

## 6. Primary statistic

For each seed `s`:

```
G_{M,s}      = R_{PW1,s} - R_{WML-M,s}
Delta_{128,s} = G_{128,s} - G_{32,s}
```

**This reduces exactly:**

```
Delta_{128,s} = (R_{PW1,s} - R_{WML128,s}) - (R_{PW1,s} - R_{WML32,s})
              =  R_{WML32,s} - R_{WML128,s}
```

`R_{PW1,s}` cancels identically, per seed, before any aggregation. **The pathwise
control does not carry the verdict.** It appears in the framing to keep this study
commensurable with `corrected_operator_replication.md`, and its `n_estep = 16`
asymmetry therefore cannot affect the primary result.

**Estimator, inherited verbatim from `scripts/analysis/fr_analyse.py`:**

* `R` = `score_window3` = the mean of the **final three** logged evaluations
  (indices 18, 19, 20 of 21).
* Unit of analysis = the seed.
* **Primary estimate:** the **median over seeds** of `Delta_{128,s}`.
* Uncertainty: **paired percentile bootstrap**, **10,000** resamples,
  `np.random.default_rng(20260902)`, 95% interval at percentiles `[2.5, 97.5]`.
* Exact two-sided sign test reported alongside.

**Differences are formed per seed first.** Separately aggregated medians are never
subtracted. `median(Delta)` is not `median(G_128) - median(G_32)` and the latter is
not computed.

The known `M = 32` controls, for reference only:

| Seed | PW-1 | WML-32 | `G_{32,s}` |
|---|---|---|---|
| 301 | 909.7893 | 784.5499 | 125.2394 |
| 302 | 898.4741 | 777.2048 | 121.2693 |
| 303 | 907.8732 | 798.9025 | 108.9707 |
| 304 | 910.9802 | 710.2243 | 200.7559 |
| 305 | 908.4419 | 762.5137 | 145.9282 |
| 306 | 909.2847 | 717.0112 | 192.2735 |
| 307 | 912.1907 | 887.0840 | 25.1068 |
| 308 | 915.0308 | 646.5341 | 268.4966 |

`median(G_32) = 135.58`.

## 7. Decision rule

**`T = 100` return points.** **Selected after the existing WalkerRun `M = 32` and
PW-1 outcomes were observed, and before any WalkerRun `M = 128` or `M = 512`
outcome.** That is the precise provenance: the control returns in Section 6 were
in hand when `T` was chosen; nothing from the arms under test was.

`T` is a deliberately conservative attenuation of the
exploratory HumanoidRun effect of `+529` from `M = 32` to `M = 128`: it requires
under one fifth of that effect to survive the change of task and of action
dimension.

* **CONFIRMED** if all three hold:
  1. `median(Delta_128) > 100`;
  2. the 95% paired-bootstrap CI for `median(Delta_128)` lies **entirely above
     zero**;
  3. `median` over seeds of `[G_{512,s} - G_{128,s}] >= 0`, computed on the
     seeds for which **both** the `M = 128` and the `M = 512` run are
     terminal-completed and non-NaN (Section 8). **Hardware-mismatched on the
     odd seeds**: `M = 512` runs on `c23g` for all seeds while `M = 128` runs
     on `c25g` for seeds 301, 303, 305, 307 (Section 2), so on those four the
     quantity mixes a machine difference with the `M` difference. This is
     reported alongside condition (iii) every time it is evaluated.
* **REFUTED** if `median(Delta_128) < 0` **and** its 95% paired-bootstrap CI lies
  **entirely below zero**.
* **INCONCLUSIVE** in every other outcome.

An arm adjudicates **only** when every one of its runs is terminal (Section 8), or
when it is classified `UNSTABLE`. No gap analysis is performed on an arm with a
non-terminal run. Any `UNSTABLE` arm, or an infeasible `M = 512` arm, forces the
overall classification to `INCONCLUSIVE` per Section 8.

## 8. Terminal status, NaN, and infeasibility

**Terminal status of a run.** A run is **terminal** when it has either (a) written
its `_final` export after completing all `399` iterations -- status `completed` --
or (b) stopped without one, for any reason: a NaN abort, an out-of-memory kill, a
walltime kill, or a node failure. Category (b) is recorded with its cause. A run
that is still queued or still executing is **not** terminal, and an arm in which
any run is non-terminal is **not** adjudicated.

**NaN and other non-completions.**

* **No replacement seeds.**
* NaN aborts are recorded per arm, per seed, with the step at which they occurred.
* Adjudication uses seeds whose `M` run is terminal-completed and non-NaN,
  **minimum 6 pairs**.
* **Three or more non-completions in an arm -> that arm is classified `UNSTABLE`**
  and is not adjudicated.

**Deliberate departure from `docs/prereg_m_sweep_dmc.md` Sec. 3.3**, which permits
replacement seeds. Reason: a replacement seed has **no paired control**. The
primary statistic is `R_{WML32,s} - R_{WML128,s}`, which exists only for a seed
that already has a WML-32 control at 301-308. A replacement seed outside that set
contributes no pair and would silently convert a paired estimate into an unpaired
one.

**Condition (iii) is evaluated on the intersection.** `median` over seeds of
`[G_{512,s} - G_{128,s}]` is computed **only over seeds for which BOTH the
`M = 128` and the `M = 512` run are terminal-completed and non-NaN**. It is never
computed across differing seed sets, and the intersection size is reported. If that
intersection has fewer than 6 seeds, condition (iii) is not evaluable and the
overall classification is `INCONCLUSIVE`.

**Escalation to the overall classification.**

* **Any arm classified `UNSTABLE` -> the overall classification is
  `INCONCLUSIVE`.** The instability is reported separately and prominently, with
  its per-seed causes. An `UNSTABLE` arm is never read as evidence for or against
  the hypothesis: a run that does not finish has no return, and absence of a
  return is not a low return.
* **`M = 512` infeasible under the registered budget -> the overall classification
  is `INCONCLUSIVE`.** "Infeasible" means the arm cannot run without changing a
  Section 4.2 quantity or breaking the Section 2 hardware pairing -- an
  out-of-memory at `M = 512`, on either partition, is the expected form. Section 4.2
  forbids shrinking the batch to rescue it. The `M = 128` result is still reported
  in full, and is still subject to conditions (i) and (ii), but with condition
  (iii) unevaluable the overall classification cannot be `CONFIRMED`.

**Missingness limitation, registered in advance.** Missing completed pairs can
bias the paired estimate **in either direction**, and **no direction is assumed**.
A seed drops out of `Delta_128` only as a whole pair, so the surviving set is not a
random subsample of the eight, and the sign of any resulting shift is not knowable
from the fact of the dropout. **All missingness and its cause are reported** --
per arm, per seed, with the step and the failure mode. The minimum of 6 pairs and
the `UNSTABLE` rule above are unchanged. A result reached with exclusions is
interpreted under this registered missingness limitation, and **is evidence
neither for nor against** the hypothesis beyond what the surviving pairs support.

## 9. Logging

Every field below is already emitted at `4eb4713` under the Section 2
configuration, with **no training-code change and no flag change**. That
constraint removed three quantities from an earlier draft of this section; the
removals are recorded in Section 9.1 rather than silently dropped.

**Definition of `sigma`, read from the implementation.** The pre-squash standard
deviation is an **additive floor, not a clip**
(`src/networks/jax_models.py:408-425`, `SACActorNetworks.effective_std`):

```python
sigma = jnp.exp(log_std) + self.min_std          # min_std = 0.1
```

with the same form at `:566` for the discrete head. With `freeze_sigma = null`
this is the learned path, and `sigma > min_std` **strictly** for every finite
`log_std` -- no coordinate is ever clipped to `0.1`, it is only approached from
above as `exp(log_std) -> 0`.

Logged per evaluation, for every run:

| Quantity | Source at `4eb4713` | Gated? |
|---|---|---|
| `ESS` mean | `reppo.py:1303` `ess=ess.mean()` | no |
| `ESS` p5 / p25 / median / p75 | `reppo.py:1308-1311` | no |
| `ESS` fraction `< 4` | `reppo.py:1312` `ess_frac_lt4` | no |
| `eta` | `reppo.py:1316` | no |
| pre-squash `sigma` mean / min / max | `reppo.py:1267-1269` | no |
| `w_max` (largest self-normalised E-step weight) | `reppo.py:1313` | no |
| `q_spread` | `reppo.py:1314` | no |
| NaN flag | `scripts/train_and_export.py:190, 351` `nan_in_eval` | no |

**Descriptive only. No decision rule attaches to any of them.** They characterise a
result the rule has already classified, and may not be used to reclassify it.

### 9.1 Removed, and why

Three quantities from an earlier draft are **not** logged here, because each would
require turning on `log_cov_diag`, which is `false` in the WML-32 controls and
whose own config comment reads *"breaks bit-identity if on"*. Enabling it would
both violate Section 2's requirement that every other setting match the control
arm, and change the numerics the study is measuring.

| Removed | Why |
|---|---|
| per-coordinate `sigma` | `pi_sigma_percoord`, `reppo.py:1274-1277`, returns `jnp.zeros(d)` unless `log_cov_diag`. Replaced by the ungated mean / min / max. |
| fraction of coordinates at the `min_std` floor | `frac_sigma_at_min`, `reppo.py:1283-1290`, returns `0` unless `log_cov_diag`. |
| saturation fraction `|tanh(y)| > 0.999` | `estep_clamp_rate`, fed by `estep_clamp` which is `jnp.zeros(())` unless `log_cov_diag` (`reppo.py:796, 894`). |

`KL(w || uniform)` is also **not** logged: no such metric exists in the training
path at `4eb4713`, and it is not derivable from what is logged -- `ESS` is
`1 / sum(w^2)` whereas the weight KL needs `sum(w log w)`, and neither determines
the other. `w_max` is retained as the ungated weight-concentration diagnostic that
does exist.

## 10. Cost

Measured anchor, `reports/artifacts/exports_manifest.csv`: WalkerRun WML-32 at
seeds 301-308 averages **1315.1 s/seed** (21.9 min; min 1296.7, max 1340.6, spread
3.4%).

Compute asymmetry measured on HumanoidRun in commit `f165518`: `M = 32` 11.0
s/iter, `M = 128` 19.4 s/iter, `M = 512` 32.6 s/iter -- a **16x rise in `M` costs
3.0x time**, sub-linear, because the E-step critic evaluation is one term among env
stepping, critic training and the M-step.

Applying those ratios to the WalkerRun anchor:

| Arm | ratio | expected s/seed | expected per seed | 8 seeds |
|---|---|---|---|---|
| `M = 32` (existing) | 1.00 | 1315 | 21.9 min | -- |
| `M = 128` | 1.76 | ~2320 | ~38.7 min | ~5.2 h |
| `M = 512` | 2.96 | ~3898 | ~65.0 min | ~8.7 h |

**Total expected ~13.8 GPU-hours for the 16 runs.** This is an extrapolation of a
HumanoidRun (`d = 21`) ratio onto WalkerRun (`d = 6`); the true ratio depends on how
much of a step is E-step work versus env stepping, and that mix differs by task.
The figure is a budget estimate, not a prediction under test.

**Critic-query asymmetry in the actor improvement step.** Per state, per update:

| Arm | critic queries in the actor improvement step |
|---|---|
| PW-1 | **1** |
| WML-32 | 32x |
| WML-128 | **128x** |
| WML-512 | **512x** |

The `512x` figure is critic rows, not seconds; wall-clock scales sub-linearly, as
the table above shows.

## 11. Smoke test, before any real submission

On a GPU node, on a tiny config, the 16-run array is smoke-tested and must print,
per arm, the **resolved** values of `estep_num_samples`, states per iteration,
states per minibatch, updates per iteration, total updates, `freeze_sigma`,
`sqrt_rho` and the frozen `alpha`, and must **assert** each against the WML-32
run's values. It must also print the partition, the GPU string and
`nvidia-smi --query-gpu=memory.total` for each arm, and assert the partition
against the Section 2 seed-to-partition table. It must confirm the export tags
and that the Section 4.2 budget assertion passes at `M = 512` on **both**
partitions. A failed assertion stops the launch.

## 12. Ledger

Every run is added to `ledger/runs.jsonl` with tier **confirmatory** and the commit
hash of this document. The commit hash of this document is recorded in every run
log.

## Design lock

Everything above is fixed at the commit that adds this file: the arms, the seeds,
the controls and that they are not rerun, the estimator and its window, the
bootstrap and its RNG, the primary statistic and its reduction, `T = 100`, the
three-part CONFIRMED rule, REFUTED, INCONCLUSIVE, the terminal-status definition, the NaN policy and its minimum of
6 pairs, the `UNSTABLE` threshold and its escalation to overall `INCONCLUSIVE`,
the seed-to-partition pairing and the `M = 512` exception to it with its
disclosed condition-(iii) mismatch, the budget-invariance quantities, and the
interpretation constraints.
