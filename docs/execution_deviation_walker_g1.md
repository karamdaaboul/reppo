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

---

## Addendum D1 — 2026-09-04T12:18:40+00:00 — two further discarded G1 settings, from run-local evidence

Append-only. Nothing above this line is altered.

### D1.1 Evidence upgraded

Section 1 recorded the evidence level as *historical launch command + exact
historical SHA*, and said this document would be superseded if a run-local
artifact disagreed. All **32** run-local `.hydra/config.yaml` files have now been
read from
`/hpcwork/qzi10910/reppo_runs/outputs/faithful_repair/*/`.

**They confirm the reconstruction.** 32/32 carry `num_mini_batches = 128` and
`num_epochs = 4`. Of 77 configuration keys per run, exactly **seven** differ across
the 32, and all seven are task, arm or seed identity: `seed`,
`actor_update_mode`, `ent_start`, `env.name`, `env.vmin`, `env.vmax`, and
`env.push_distractions` (`False` on G1, unset on Walker) — the last verified to
split by **task only**, identical across arms within each task. **No material
within-task scientific difference exists.**

### D1.2 Two more overrides were discarded, both on G1

The same `_self_` composition order discards more of
`mjx_humanoid_large_data.yaml` than section 3 listed:

| G1 override | requested | executed |
|---|---|---|
| `num_mini_batches` | 16 | **128** |
| `num_epochs` | 8 | **4** |
| `gamma` | **0.97** | **0.99** |
| `critic_hidden_dim` | **1024** | **512** |

So the corrected G1 runs used Walker's discount and **half** the intended critic
width. `env.vmin`/`env.vmax` were unaffected (`-10`/`10` on G1) because those come
from the `env` group, not from `experiment_overrides`.

### D1.3 Classification unchanged

Both settings were applied **identically to both arms** — confirmed by the
run-local files, where `gamma` and `critic_hidden_dim` do not appear among the
seven varying keys. The classification for G1 therefore remains

```
SYMMETRIC EXECUTION DEVIATION
```

and the interpretation of section 6 stands unchanged: the within-task PW-vs-WML
comparison remains internally symmetric, but these results are **not** literal
executions of the registered configuration. The deviation is now larger than first
documented — a different discount and half the critic capacity, not only a
different minibatch geometry — which strengthens rather than weakens the reason to
avoid describing G1 as a reproduction of any registered or published
hyperparameter set.

Walker is unaffected by D1.2: its `gamma` and `critic_hidden_dim` were already the
base values.
