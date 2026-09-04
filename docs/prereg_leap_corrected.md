# Preregistration: corrected LeapCubeRotateZAxis operator replication

**Dated 2026-09-04T11:45:52+00:00. Label: PROSPECTIVE, NOT BLIND.** The corrected Walker and G1 results
are known and motivate the directional prediction. No corrected LEAP run exists;
no LEAP outcome under the faithful-repair implementation has been observed.

## 1. Question

The legacy 64-run study reported LeapCubeRotateZAxis **favouring weighted-MLE**.
Those runs were produced under the defective code path, and LEAP carries that
path's strongest pathology on record: `reports/ubar_ratio.md` documents the E-step
collapsed onto a single sample (**ESS 1.07 of M=32**) and the `+-(1-1e-4)` clip
binding on **43.3%** of samples at leap arm B seed 106, the worst checkpoint in the
study.

**Does the LEAP WML-favouring sign survive the corrected implementation?**

## 2. Design

* Task `LeapCubeRotateZAxis`, `env=mjx_dmc`, `d = 16`, `obs_dim = 32`.
* Arms **PW-1-faithful-repair** and **WML-32-faithful-repair**, the same corrected
  implementations used for Walker and G1.
* Seeds **301-308**, both arms, **16 runs**. Confirmatory tier.
* All training executes **on the cluster**. The workstation is used only for code
  preparation, audit, preregistration and verification.

## 3. Optimization protocol, and why

```
num_envs 1024   num_steps 128   num_mini_batches 128   num_epochs 4
minibatch size 1024             optimizer steps / iteration 512
iterations 399                  env steps 52,297,728
M 32   eps_e 0.5   kl_bound 0.1   num_eval 20
alpha 0.000782382907345891 (frozen; update_entropy_lagrangian=false)
sqrt_rho 1.0   freeze_sigma null   log_cov_diag false   (explicit no-ops)
```

> LEAP uses the common optimization protocol **actually executed** by the corrected
> Walker and G1 comparisons, providing cross-task comparability. This choice is
> made prospectively, after documenting the configuration-composition deviation in
> `docs/execution_deviation_walker_g1.md`.

**This document does not claim `128/4` was the registered Walker/G1 protocol.** It
was not: Walker registered 64/8 and G1 registered 16/8, and both executed 128/4
because `config/reppo.yaml` composes `_self_` last. LEAP matches what ran, not what
was written.

**The Hydra ordering is deliberately NOT repaired here.** Repairing it would change
how historical reproduction commands resolve at a new source SHA. Instead every
scientifically relevant value is **pinned explicitly on the command line**, so the
composition order cannot affect what executes. The ordering is a separate
reproducibility decision, to be made in its own commit.

## 4. Alpha

`alpha = 0.000782382907345891`, fixed by the registered `alpha_901` protocol
(`docs/prereg_dimension_ladder.md:834-835`): `median(alpha_curve[2:])` of that
task's seed-901 calibration, frozen identically in both arms. Reproduced exactly
from `exports/LeapCubeRotateZAxis_pathwise_s901orig_final`. The other seed-901
export (`..._s901_final`) gives `0.000803965260274708` and is **not** the registered
calibration. **This choice is not reopened and no alpha sweep is run.**

## 5. Canonical configuration and hashes

Rendered by `scripts/analysis/leap_config.py`, artifact
`reports/artifacts/leap_canonical_config.json`:

```
LEAP_CONFIG_HASH[PW ]      = d4eb715180afb32fe981026aaa693acfc289d95237a209e514b6c9cc4238fe6e
LEAP_CONFIG_HASH[WML]      = 221dc60824d0aab4769037048547bb653059f008c02cf47817676c05d550280d
LEAP_CONFIG_HASH[combined] = c28b607725aa045d45811cdf2ca6d57047bb910f59ab5a06b97c7521b52b4d07
```

Each arm's hash is identical across all eight seeds. The renderer must be run on
**both** machines and produce these values before any job is submitted.

## 6. Analysis, inherited unchanged

Identical to corrected Walker/G1 (`scripts/analysis/fr_analyse.py`):

* Score `R` = **mean of the final three logged evaluations** (indices 18, 19, 20
  of 21).
* Secondary score: the last logged evaluation alone.
* Unit of analysis: the seed. Contrast **paired within seed**.
* Uncertainty: **paired percentile bootstrap, 10,000 resamples,
  `np.random.default_rng(20260902)`**, 95% interval at percentiles [2.5, 97.5].
* Exact two-sided sign test reported alongside. IQM reported per arm.
* Differences formed per seed first; separately aggregated medians are never
  subtracted.

## 7. Primary hypothesis

```
Delta_LEAP = R_PW - R_WML,  paired within seed
```

**Prospective directional prediction: `Delta_LEAP > 0`**, based on the corrected
Walker (+135.58) and G1 (+9.51) results.

Fixed before any corrected LEAP outcome exists:

* **PW-supported** - the point estimate is positive and the 95% paired-bootstrap
  CI lies entirely above zero.
* **Inconclusive** - the CI contains zero.
* **Strong falsifier** - the CI lies entirely **below** zero
  (`CI_upper < 0`): corrected WML wins.

**A WML win is a valid and potentially important result** - it is the outcome the
legacy LEAP result predicts, and the reason this task was chosen. It will be
reported as a genuine result. **LEAP will not be tuned until PW wins.**

## 8. Completion and NaN

An arm adjudicates only when all eight of its runs are terminal. No replacement
seeds. NaN aborts recorded per seed with cause. Missingness can bias the paired
estimate in either direction; no direction is assumed.

## 9. Run gate

No confirmatory LEAP job starts until: cluster-only commits are preserved on a
pushed branch; the 32 historical Walker/G1 run-local `.hydra/config.yaml` files
confirm the `128/4` reconstruction; workstation SHA == cluster SHA; both tracked
trees clean; this document committed; and the two machines render identical LEAP
config hashes.

## Design lock

Everything above is fixed at the commit that adds this file: the task, arms,
seeds, the `128/4` protocol and its justification, alpha, the estimator, the
bootstrap and its RNG, the primary statistic, the three-way decision rule, and the
completion policy.

---

## Addendum L1 — 2026-09-04T11:56:55+00:00 — LEAP-specific environment pins, task wording, hash status

Append-only. No text above this line is altered. Written **before any corrected
LEAP run exists** and before any corrected LEAP outcome has been observed.

### L1.1 Five pins were missing, and the omission was material

The canonical configuration as first rendered omitted every LEAP-specific
environment setting, and would therefore have taken the `mjx_dmc` **group
defaults** — which are Walker's values, not LEAP's. Recovered from the single
distinct LEAP launch command in `ledger/runs.jsonl`, now pinned:

```
env.asymmetric_obs=false
env.vmin=-10
env.vmax=60
env.max_episode_steps=500
hyperparameters.max_episode_steps=500
```

| parameter | legacy LEAP | corrected LEAP, as first rendered | corrected LEAP, now |
|---|---|---|---|
| `gamma` | 0.99 | 0.99 | **0.99** |
| `lmbda` | 0.95 | 0.95 | **0.95** |
| `vmin` | **-10** | 0 | **-10** |
| `vmax` | **60** | 150 | **60** |
| `max_episode_steps` | **500** | 1000 | **500** |

`gamma` and `lmbda` always matched. **`vmin`, `vmax` and `max_episode_steps` did
not.** Answering the three required questions:

1. **Prospectively intended? No.** This was an omission on my part, not a design
   choice. It is corrected here, before any LEAP job runs.
2. **Does it come from the common corrected Walker/G1 protocol? No.** Walker's
   natural support is `[0, 150]` at 1000 steps and G1 runs under `mjx_humanoid`
   with its own bounds. These three values are **task properties of LEAP**, not
   optimization-protocol choices, and they sit outside the `128/4` decision, which
   is unchanged.
3. **Does it change LEAP as a corrected replication? It would have.** Running LEAP
   with critic support `[0, 150]` instead of `[-10, 60]` and with double the
   episode length would not have been a corrected replication of the legacy result
   at all. The Walker audit already showed critic representational support can
   materially shape the critic-error field: 34.94% of Walker WML oracle values fell
   below that critic's floor. Leaving this unpinned would have confounded the one
   comparison LEAP exists to make.

**No value was chosen to favour any outcome, and none was selected after seeing a
LEAP return — no corrected LEAP return exists.** The optimization protocol
(`128/4`), `alpha`, seeds, arms, estimator, bootstrap and decision rule are all
unchanged from the body of this document.

### L1.2 Superseded configuration hashes

The hashes in section 5 were rendered before these pins and are **superseded**:

```
superseded:  combined c28b607725aa045d45811cdf2ca6d57047bb910f59ab5a06b97c7521b52b4d07
CANONICAL:   PW       605efbf07b77dbb2cd6756a678b6d9a87223920af5e0782d2419ff8b6ae46ffb
             WML      7890bf72389b1c1b06375b6ed1686199d59de32dcb2c1c1852e233e9f8c27b94
             combined a1e015ced93d99962479a789881170c51f60fb305dfe4ebf931794c4e735397c
```

Each arm's hash is identical across all eight seeds.

### L1.3 Task wording

**`LeapCubeRotateZAxis` is a mujoco_playground dexterous-manipulation task** — a
16-DoF hand rotating a cube about the z axis, `action_dim = 16`, `obs_dim = 32`,
500-step episodes. **It is not a DMC locomotion task.** It is routed through this
project's `mjx_dmc` **configuration group** purely because that group carries the
`type: mjx` plumbing; the group name does not describe the task. Every
LEAP-specific value that differs from the group default is pinned in L1.1.

### L1.4 Critic-support bounds are per task

`docs/critic_reference_convention.md` writes its secondary reference as
`clip(Q, 0, 150)`. Those are **Walker's** bounds. The convention is unchanged; its
bounds are read from the task's resolved `env.vmin`/`env.vmax`, which for LEAP are
**`[-10, 60]`**. No decision is reopened.

### L1.5 Hash status

```
CURRENT_LEAP_CONFIG_HASH_STATUS = PROVISIONAL UNTIL FINAL CANONICAL SHA
```

The hashes above are valid for the configuration rendered at the workstation SHA
recorded with this addendum. Preserving and integrating the three cluster-only
commits may produce a new canonical SHA. If it does, the resolved LEAP
configuration must be **re-rendered at that SHA on both machines**, the hashes
recompared, and an exact match required before launch. If the re-rendered
scientific configuration is identical, that fact is recorded in a further dated
execution note carrying: original prereg SHA, original config hash, final
execution SHA, final config hash, scientific config changed YES/NO, and the reason
for the update. **The old hash is not assumed valid merely because the intended
hyperparameters are unchanged.**
