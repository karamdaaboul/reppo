# Covariance freeze, Phase 1: source trace, implementation, parity

**Status: PHASE 1 COMPLETE.** No confirmatory frozen-sigma run exists. No freeze
value has been chosen. No preregistration has been written.

---

## 1.1 Provenance

| Item | Value |
|---|---|
| Branch | `estep-study` |
| HEAD at Phase 1 start | `2409e015ebb42290c51655f71ad705cbadc9a903` |
| `git status --short` | clean |
| Remote `origin/estep-study` | `2409e015ebb42290c51655f71ad705cbadc9a903` |
| **Source-tree hash** `HEAD:src` | **`399411d5862ff0d6837fc33adb40706ac5a7ab2a`** |
| Correction code | `cfbd8ddfed1648f21731c8059287104d2ac19cfb` |
| Corrected launch | `1c6259ef959400eba617e1d1392bfaf58248688c` |
| Corrected pre-launch ledger | `b3e3c39d758bdf7dbdbf7ec0877ac6417d3f7512` |

**`src/` is byte-identical at all three:** `HEAD:src`, `1c6259e:src` and `cfbd8dd:src`
are all `399411d5`. Every commit since the corrected launch touched only `scripts/`,
`reports/`, `docs/` and `slurm/`. The training code that will run the frozen arms is
exactly the code that produced the learned arms.

**Environment.** Python 3.12.14; jax 0.5.2; jaxlib 0.5.1; jax-cuda12-plugin 0.5.1;
flax 0.10.6; distrax 0.1.5; optax 0.2.5; chex 0.1.90; brax 0.14.2; mujoco 3.10.0;
mujoco-mjx 3.10.0; `playground` 0.0.5 pinned to the fork
`younggyoseo/mujoco_playground@1699a6d3d9996f2e8391791d55c2d91eccbe33ba`;
gymnax 0.0.9; numpy 2.5.1; scipy 1.18.0. Accelerators: c23g = NVIDIA H100 94 GB,
c25g = NVIDIA H100 80 GB HBM3.

### The 16 existing learned-sigma cells

`reports/artifacts/covariance_freeze_existing_cells.csv`, one row per cell, carrying
operator, seed, partition, resolved config path and SHA-256, config hash, alpha,
horizon, `num_eval`, output path, export paths, and both the ledger checkpoint
checksum and a freshly recomputed one.

All 16 verified: `status=completed`, resolved configs present, and **16/16 checkpoint
checksums still match the values recorded at launch**, recomputed under the exact
definition in `slurm/fr_launch.sh:63-67`.

**The table contains no return values.** The frozen arms have not run, and putting
learned returns one join away from a blinded analysis would defeat the outcome-blinding
rule.

### Hardware pairing — interleaved, not blocked

| | c23g | c25g |
|---|---|---|
| PW-1 | 301, 303, 305, 307 | 302, 304, 306, 308 |
| WML-32 | 302, 304, 306, 308 | 301, 303, 305, 307 |

8 runs on each partition, with the two operators on **opposite** partitions for any
given seed. Confirmed against `sacct`: array `3444831` ran on c23g and `3444832` on
c25g, and every row's actual partition matches its ledger `gpu_architecture`.

**Exact hardware matching is possible for all 16 pairs.** Nothing needs to be
reported as unmatchable. Sending every frozen run to c23g would break 8 of the 16
pairings.

## 1.2 Provenance of the covariance-freeze hypothesis

**COVARIANCE-FREEZE HYPOTHESIS WAS GENERATED FROM PILOT OBSERVATIONS, NOT TRIGGERED
BY A PROSPECTIVE RULE.**

A trigger does exist, in `docs/prereg_mc_oracle_walker_pilot_2.md` §11, commit
`4e03c25`, worded verbatim:

> "The ablation is preregistered only if, on the pilot-2 bank, the WML checkpoint's
> per-state RMS `sigma` has a p95 exceeding five times its median **and** the base
> action clip rate exceeds 50%."

It fails as a prospective rule on two independent grounds.

**It was written after the values were known.**

| Commit | Time | Content |
|---|---|---|
| `63c2cd2` pilot-1 preregistration | 13:00:48 | **no covariance or freeze trigger of any kind** |
| `fd57fda` pilot-1 report | 14:20:12 | contains p95 90.05, median 4.26, saturation 87.3% |
| `4e03c25` pilot-2 preregistration | 14:55:37 | the trigger, **35 minutes later** |

**Its own condition can never be evaluated.** It conditions on *the pilot-2 bank*.
Pilot 2 was preregistered and then deliberately not executed (see the execution-status
entry appended to that file); no pilot-2 oracle value exists.

**Measured pilot-1 values, and whether the trigger would fire:** p95 `90.05` against
median `4.26`, a ratio of **21.1** against a threshold of 5; base clip rate **87.3%**
against a threshold of 50%. On pilot-1 data both conditions hold by a wide margin.

This does not block the experiment. It fixes the wording: the hypothesis was
**generated from observation**, and the manuscript must not describe it as
prospectively triggered.

## 1.3 The covariance implementation

Full trace in `docs/covariance_freeze_design_note.md`. Summary:

| Question | Answer | Evidence |
|---|---|---|
| Actor output head | one `FCNN`, `out_features = action_dim * 2` | `jax_models.py:348-361` |
| loc / log_std split | `jnp.split(loc, 2, axis=-1)` | `jax_models.py:393` |
| Effective sigma | `(exp(log_std) + min_std) * scale` | `jax_models.py:394` |
| `min_std` | **0.1**, constructor default; `make_init` never passes it, so the config field `actor_min_std: 0.0` is **dead** | `jax_models.py:336`, `reppo.py:364-377` |
| State-dependent | **yes** | `jax_models.py:392` |
| Shared trunk | **yes, and the same final `Linear`** — loc and log_std are two halves of one output layer, not separate heads | `jax_models.py:348-361` |
| Online / target / rollout actors | all via `actor()` or `gaussian()` | `reppo.py:538, 539, 557, 736, 754, 803, 772, 777, 865, 868, 1003` |
| **Evaluation actor** | **`det_action() = tanh(mu)` — no sigma at all** | `reppo.py:275` |
| PW actor sampling | `pi.sample_and_log_prob` on `actor()` | `reppo.py:741` |
| WML E-step sampling | same-point branch materialises `y_i` from `gaussian()` | `reppo.py:772-788` |
| Sampled KL | from the two same-point log probabilities | `reppo.py:790-796` |
| Analytic KL diagnostic | `gaussian_kl_diag(mu_old, sg_old, mu_new, sg_new)` | `reppo.py:798` |
| Target actor refresh | `actor_target.params = actor.params` every learn step | `reppo.py:647-651` |
| Optimizer | `optax.chain(clip_by_global_norm, adam(lr))`, elementwise | `reppo.py:429-440` |
| Separate covariance loss | **none** outside `mstep_decoupled` | — |
| `beta_sigma` dual | exists only when `mstep_decoupled=True` | `reppo.py:894-908` |

There is **no evaluation covariance to freeze**: evaluation is deterministic. Stated
explicitly so that "every path uses the frozen sigma" is not read as implying a path
that does not exist.

### Why this is not the manuscript's fixed-`beta_Sigma` experiment

Traced, not asserted. `beta_sigma_fixed` (`reppo.py:155`, used at `reppo.py:898-908`)
lives inside the decoupled M-step. In that branch:

* `mu_new, sg_new = actor_model.gaussian(...)` — **the covariance is still learned and
  still state-dependent** (`reppo.py:868`);
* `logp_sigma = gaussian_logp(u_i, mu_old, sg_new)` — **`sg_new` still receives
  gradient** (`reppo.py:890`);
* `kl_sigma` is a covariance KL constraint and `beta_sigma` its Lagrange multiplier;
* `beta_sigma_fixed` holds **the multiplier** constant. The source comment says so:
  *"constant: no dual term, so beta_sigma_param gets no gradient"*.

That experiment **retains covariance learning** and **uses a covariance KL constraint
with a dual**; it fixes the *price* of changing the covariance. The present
intervention fixes *the covariance*. Formally, that one leaves the policy family
`{N(mu(s), Sigma(s))}` intact and holds a regulariser weight constant; this one
restricts the family to `{N(mu(s), Sigma*)}` with `Sigma*` constant in state and time,
and makes `dL/d(log_std head)` identically zero.

They are also on different code paths. **All 16 corrected Walker runs have
`mstep_decoupled: false` and `beta_sigma_fixed: null`** (verified across all 16
resolved configs), and `with_betas = cfg.mstep_decoupled`, so `beta_mu_param` and
`beta_sigma_param` **are absent from their parameter tree entirely**.

## 1.4 Implementation

`freeze_sigma`, default `null`, applied at the **effective sigma** in one helper that
every distribution constructor routes through (`SACActorNetworks.effective_std`).
`freeze_sigma = x` means the pre-squash sigma **is** `x`; it is deliberately not
`log_std = x`, which would have meant `exp(x) + 0.1`.

The value is a plain Python tuple, not an `nnx.Param`, so it stays out of `nnx.state`
and leaves the parameter tree, optimizer state and checkpoint layout unchanged.
Pre-existing checkpoints still load: `load_ckpt` passes `actor_kwargs` through and the
missing key falls back to the constructor default `None`. The learned scale head is
still computed and remains available as a diagnostic, but `effective_std` discards it.

A guard refuses `freeze_sigma` together with a non-unit exploration `scale`, which
would otherwise multiply the frozen width and make the effective sigma differ from the
value it is declared to be.

### A configuration hazard that would have silently voided the experiment

`scripts/train_and_export.py:139` merges `experiment_overrides` **on top of**
`hyperparameters`, so any key defined in `mjx_dmc_large_data` **cannot be overridden
from the command line**. Verified directly:

```
freeze_sigma AFTER the experiment_overrides merge : 1.1     <- override survives
num_envs (known-shadowed control)                 : 1024    <- silently overwritten
freeze_sigma present in mjx_dmc_large_data?       : False
```

`freeze_sigma` is not in that file, so `hyperparameters.freeze_sigma=...` takes
effect. Had it been, all 16 frozen runs would have trained with **learned** sigma and
produced a null result for a purely mechanical reason. This check must be repeated if
the override file ever changes.

## 1.5 Gradient semantics

In the frozen branch `log_std` is read **only** for its static `.shape` and `.dtype`.
Therefore:

* the gradient with respect to the **scale half** of the shared output layer
  (`kernel[:, d:]`, `bias[d:]`) is **exactly zero**, not small — measured `0` against
  a learned-arm control of **946.9** on the same inputs and loss;
* under Adam those coordinates get `m = v = 0` and never move — verified bit-unchanged
  after three optimizer steps while the mean half moved by `9.0e-4`;
* the mean half and the shared trunk stay fully trainable (`68.97` and `6.34`);
* the shared trunk **no longer receives the gradient component that previously arrived
  through the learned covariance branch**.

> **The intervention disables covariance adaptation and therefore removes both the
> changing covariance and its gradient pathway through the shared actor network.**

It is **not** "holding a logged sigma statistic constant". A positive result is an
effect of the whole package and cannot be attributed to any single component.

## 1.6 Candidate freeze value A

**The initial `log_std` output is NOT provably zero.** The actor trunk ends in an
`nnx.Linear` whose kernel uses the default lecun-normal initialisation
(`jax_models.py:35-49`, `normed_activation_layer`), so the pre-squash `log_std` at
initialisation is a random, state-dependent, seed-dependent quantity.

Measured over 8 model seeds x 256 WalkerRun states x 6 coordinates
(`reports/artifacts/covariance_freeze_initial_sigma.json`):

| nominal `exp(0)+min_std` | min | median | mean | p95 | max | within 1% of nominal | exactly nominal |
|---|---|---|---|---|---|---|---|
| **1.1** | 0.216 | 1.096 | 1.314 | 2.928 | **9.200** | **1.33%** | **False** |

Standard deviation across states within a seed **0.795**; spread of the per-seed
medians 0.163. The median sits near 1.1 by symmetry of the `log_std` distribution, but
the range spans a factor of **42**.

Therefore candidate A is labelled:

```
sigma_A = exp(0) + min_std = 1.1000000000000001   ->  NOMINAL ZERO-LOGSTD REFERENCE
```

It is **not** the initial policy width and must not be called "the initialization".

## 1.7 Candidate freeze value B

```
CANDIDATE B NOT RETRIEVABLE AS A PRE-EXISTING FIXED COVARIANCE.
```

The retained seed-901 calibration set is exactly three rows in `ledger/runs.jsonl`:
`cal901-g1` (G1JoystickFlatTerrain, d=29), `cal901-leap` (LeapCubeRotateZAxis, d=16),
`cal901-hopper` (HopperHop, d=4).

Failing conditions:

1. **No WalkerRun seed-901 run exists**, so no `d=6` artifact can exist.
2. Those runs are `"A-learned-alpha (calibration)"` with
   `update_entropy_lagrangian: true` — they calibrated the **entropy coefficient**,
   not the covariance.
3. No `fixed_sigma` / `freeze_sigma` / `frozen_sigma` value or vector appears in any
   artifact in the repository.
4. No seed-901 export was retained.
5. The rows are incomplete: `return_metrics: null`, `estimator_diag: null`, status
   `launched` / `queued`.

No per-coordinate vector has been manufactured from state-dependent learned sigmas.

### A and B side by side

| | Candidate A | Candidate B |
|---|---|---|
| Value | `1.1` scalar, all 6 coordinates | — |
| Status | available | **not retrievable** |
| Honest label | NOMINAL ZERO-LOGSTD REFERENCE | — |
| Provenance | `exp(0) + min_std`, `min_std = 0.1` from `jax_models.py:336` | none |
| Caveat | not the initialization; measured initial sigma spans 0.216-9.200 | — |

No recommendation between them is offered, and none is based on any expected effect
on return.

## 1.8 Corrected-path parity with the flag off

**PARITY PASS — maximum absolute difference 0.0 everywhere.**

Seed **499**, verified unused: 0 ledger rows, no output directory, disjoint from the
301-308 confirmatory and 401-406 smoke/calibration namespaces.

Both arms ran in **one worktree with one venv and one command line**, differing only
in the checked-out source: `2409e01` (pre-freeze) versus `46264c4` (freeze mechanism,
`freeze_sigma` left at its `null` default). Job `3486677`, c23g, 26 min 10 s.

| Arm | Arrays compared | Identical | Max abs diff |
|---|---|---|---|
| PW-1 | **322** | 322 | **0.0** |
| WML-32 | **327** | 327 | **0.0** |

Each arm covers 3 exported checkpoints (`final`, `p25`, `p50`) x
(`actor.npz` + `critic.npz` + `normalizer.npz`) = 105 and 108 parameter/normalizer
arrays, plus 73 `meta.json` fields per checkpoint, plus **82 training metric curves**
from `metrics.npz`. Actor, critic, target, optimizer-derived and dual states
(`alpha_entropy`, `alpha_kl`, `eta`) and normalizer statistics are all included.

The only permitted differences are `hydra_run_dir`, `train_seconds`, and the new
`actor_kwargs.freeze_sigma` key, which is asserted to be `null` in the flag-off export
and **is** `None`.

The parity worktree exports through a symlink into `/hpcwork/qzi10910/frparity_head`,
a different inode from the main `/hpcwork/qzi10910/reppo_runs/exports`, so the run
could not touch the 16 corrected artifacts. The worktree was restored to `cfbd8dd`.

### Supplementary component parity (CPU)

`scripts/analysis/cf_signature.py`: **83/83 arrays byte-identical**, max diff 0.0,
covering both actors' full parameter trees, `mu`, `sigma`, distribution `loc`/`scale`,
`det_action`, `temperature`, `lagrangian`, `eta`, the critic tree, `Q`, and **every
gradient leaf**, for both operators.

Recorded because it nearly produced a false alarm. The first version of this probe
reported a parity **FAILURE** on 16 gradient arrays. A base-versus-base control
mismatched **the same 16 keys against itself**: the probe loss called `distrax
log_prob` on a `tanh` sample that reached exactly `+-1` in float32, so the gradients
were `NaN` and every comparison was vacuous. Without that control a source-equivalent
implementation would have been reported as broken.

## 1.9 Freeze-specific unit tests

`scripts/analysis/test_covariance_freeze.py` — **12/12 pass**.

| Test | Result |
|---|---|
| T1 scalar broadcasts to all 6 coordinates | PASS — single unique value `1.100000023841858` |
| T2 vector preserves each coordinate | PASS — max diff 0 |
| T3 effective sigma correct for arbitrary obs | PASS — 256 observations incl. `1e3` and zero |
| T4 online and target actors identical | PASS — max diff 0 across independent inits |
| T5 `actor()`/`gaussian()`/`effective_std()` agree | PASS — and `det_action` is sigma-free |
| T6 scale-head gradient zero when frozen | PASS — kernel 0, bias 0 |
| T7 scale-output params unchanged after 3 Adam steps | PASS — delta 0; mean half moved 9.0e-4 |
| T8 mean head and trunk still get gradient | PASS — 68.97 and 6.34 |
| T8b learned control drives the scale head | PASS — **946.9** |
| T9 analytic KL loses its covariance contribution | PASS — 1.19e-07 vs a 15.85 control |
| T10 flag off reproduces `exp(log_std)+min_std` | PASS — max diff 0 |
| T11 invalid freeze values rejected | PASS — 4/4 (zero, negative, wrong length, NaN) |

T9 concerns the **analytic** Gaussian KL only. The published gate uses a **sampled**
KL estimator, which remains stochastic under a frozen covariance; no claim is made
that it becomes deterministic or free of Monte-Carlo error.

## Commits

| Commit | Content |
|---|---|
| `46264c4` | the default-off freeze mechanism and its 12 unit tests |
| `b34ff97` | design note and parity harness |
| `aa86663` | CPU component-parity probe |
| (this) | Phase 1 report and parity result |

## Operational note for Phase 4

Measured training time for the 16 corrected Walker runs: PW-1 **14.6-15.6 min**,
WML-32 **21.9-23.0 min**, median 18.7. The inherited `slurm/fr_launch.sh` requests
`--time=05:00:00`, a 13-20x over-request. Observed queue waits today ranged from 0 to
**307 minutes**, and the worst case was array `3444832` — 5-hour request, 41 min used,
307 min waited — while its twin `3444831` with the same request waited 0. A shorter
request backfills; Phase 4 should request roughly 45-60 minutes per run.

---

**PHASE 1 COMPLETE — WAITING FOR FREEZE-VALUE SELECTION.**

No preregistration written. No frozen-sigma run submitted. Candidate B is not
available, so the decision is candidate A at `1.1` labelled as a nominal
zero-log-std reference, or a value supplied explicitly.
