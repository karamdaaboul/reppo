# Execution deviation: corrected Walker / G1 optimization configuration

**Dated 2026-09-04T11:44:02+00:00.** Append-only. `docs/prereg_corrected_operator_replication.md` is
**not** modified. This document records what the completed corrected jobs actually
executed, and how that differs from what was registered prospectively.

## 1. Evidence

Resolved with Hydra at the **recorded training commit `1c6259e`** (the `git_sha` in
all 32 completion records under `ledger/runs.d.faithful_repair/`) using the exact
launch commands stored in `ledger/runs_faithful_repair.jsonl`. The `defaults:`
ordering was verified byte-identical at `1c6259e` and at HEAD before resolving.

Evidence level: **historical launch command + exact historical source SHA.**
Run-local `.hydra/config.yaml` for the 32 jobs lives on the cluster and has not yet
been read; this document is superseded if any run-local artifact contradicts it.

## 2. Walker

| | registered | executed |
|---|---|---|
| `num_mini_batches` | 64 | **128** |
| `num_epochs` | 8 | **4** |
| minibatch size | 2048 | **1024** |
| optimizer steps / iteration | 512 | 512 |

## 3. G1JoystickFlatTerrain

| | registered | executed |
|---|---|---|
| `num_mini_batches` | 16 | **128** |
| `num_epochs` | 8 | **4** |
| minibatch size | 8192 | **1024** |
| optimizer steps / iteration | 128 | **512** |

## 4. Symmetry

Resolved scientific-config hashes, one per arm across all eight seeds 301-308:

```
Walker  PW  12482fda0c0f2dec      Walker  WML  d29f0df6f47e72c6
G1      PW  2f576beb50b2781f      G1      WML  dfe2a42c71c530cd
```

Within each task the two arms differ only in `actor_update_mode`. Every seed in an
arm resolves to a single hash.

**Classification, both tasks: SYMMETRIC EXECUTION DEVIATION.**

## 5. Cause

`config/reppo.yaml` lists `_self_` **last** in its `defaults:`, so the base
`hyperparameters` block composes **over** `experiment_overrides`.
`mjx_dmc_large_data.yaml` (64/8) and `mjx_humanoid_large_data.yaml` (16/8) are
therefore discarded for those keys; only `kl_bound: 0.1` survives, because the base
block has no such key. This ordering and the base 128/4 values are present in the
**root commit `69d04eb`** of this repository's history and have never been changed
here.

## 6. Interpretation

> The within-task PW-vs-WML comparisons remain internally symmetric under the
> executed optimization protocol, because both arms used the same relevant
> optimization settings. However, the executed optimization settings differed from
> those specified prospectively, so these results are **not literal executions of
> the registered optimization configuration.**

This is **not** written as "comparison unaffected". A symmetric change removes the
between-arm confound; it does not establish that the effect size would be the same
under the registered protocol. Minibatch size and data-reuse depth plausibly
interact with both operators, and for the weighted-MLE arm they also enter the
`eta` dual, which is solved against the same minibatch.

## 7. Status of the existing results

**Retained, not discarded.** The corrected Walker and G1 return results stand as
measured and are to be labelled **results under the executed 128/4 protocol**.
No rerun is required for validity. Whether to also run the registered 64/8 and
16/8 settings is a scope decision and is not made here.
