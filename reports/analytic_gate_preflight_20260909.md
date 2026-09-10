# Analytic-KL-gate experiment — PREFLIGHT FAILED, nothing implemented or launched

2026-09-09. Read-only. No file edited, no branch changed, no job submitted.

## Verdict

**STOP at stage 1.** Two blockers prevent identifying eight comparable canonical Walker runs
per arm under the stated partition rule. Per the instruction, the design is not modified and
no code is repaired.

## 1.1 Repository identity — PASS

```
path            /hpcwork/qzi10910/estep_wt
branch          estep-study
HEAD            3be8c1117154449e05f0a1279b4a82bff28e9dce
tracked-clean   YES
reppo.py        sha256 2c4ff079d2a8f357c6f096cb0e416334a85f1a3deeed085b03d37856a810366b
                1758 lines
```

## 1.2 Canonical Walker runs — LOCATED

From the frozen analysis `/hpcwork/qzi10910/gates/ec_all.py:15-18`:

| arm | run directory | export tag | exports present |
|---|---|---|---|
| WML | `reppo_runs/outputs/faithful_repair/walker_WML32_s{301..308}` | `WalkerRun_weighted_mle_s*_final` | **8/8** |
| PW | `reppo_runs/outputs/entropy_factorial/walker_PW-H_s{301..308}` | `WalkerRun_pathwise_fa_noent_s*_final` | **8/8** |

Note the two arms come from **two different experiments** (`faithful_repair` vs
`entropy_factorial`). Their budgets and gate settings match (below), but they were not
launched together.

## 1.3 Seeds — PASS

301-308, verified independently from `ledger/runs_faithful_repair.jsonl` and from each run's
`.hydra/config.yaml`. No other seeds appear in either arm.

## 1.4 Configuration and budget — PASS

All eight WML seeds, from `.hydra/config.yaml`:

| field | required | observed (all 8) |
|---|---|---|
| `faithful_same_point` | True | **True** |
| `reverse_kl` | False | **False** |
| `actor_kl_clip_mode` | clipped | **clipped** |
| `mstep_decoupled` | False | **False** |
| `sqrt_rho` | 1.0 | `None` -> code default **1.0** (`src/jaxrl/reppo.py:171`) |
| `wml_add_actor_entropy` | False | `None` -> code default **False** (`src/jaxrl/reppo.py:183`) |
| `update_entropy_lagrangian` | False | **False** |

Two fields read `None` because they did not exist when those configs were written; the
dataclass defaults supply the required values, and the executed code therefore took the
`sqrt_rho == 1.0` branch with no entropy add-on. Flagged rather than silently accepted.

Budgets identical across both arms and all sixteen runs: `total_time_steps = 50000000`,
`num_eval = 20`, `kl_bound = 0.1`; WML additionally `estep_num_samples = 32`, `eps_e = 0.5`;
PW `pw_drop_actor_entropy = True`, `ent_start = 0.014509912580251694`, both
`actor_kl_clip_mode = clipped`, `reverse_kl = False`, `faithful_same_point = True`.

## BLOCKER 1 — the canonical WML arm is mixed-partition by preregistered design

`ledger/runs_faithful_repair.jsonl` records a per-seed GPU-architecture assignment, and
`slurm/fr_launch.sh:24` filters on it (`rows = [r for r in rows if r["gpu_architecture"] == arch]`):

| seed | 301 | 302 | 303 | 304 | 305 | 306 | 307 | 308 |
|---|---|---|---|---|---|---|---|---|
| **WML-32** | c25g | **c23g** | c25g | **c23g** | c25g | **c23g** | c25g | **c23g** |
| PW-1 | **c23g** | c25g | **c23g** | c25g | **c23g** | c25g | **c23g** | c25g |

**Only 4 of 8 canonical WML seeds ran on c23g.** The rule "only accept old runs produced on
the same approved GPU partition" and "exclude mixed-partition runs" cannot be satisfied at
n = 8.

This is not incidental: the alternation was written into the ledger before launch, and both
batches were submitted in the same second (`3444831` c23g, `3444832` c25g,
2026-09-02T04:38:44) from the same script with identical resources.

The `PW-1-faithful-repair` rows above are **a different arm** from the canonical PW arm --
they export `WalkerRun_pathwise_fa_s*` (entropy on), not `pathwise_fa_noent`.

## BLOCKER 2 — the canonical PW arm has no partition record at all

The PW arm used by the frozen analysis is `entropy_factorial/walker_PW-H_s*`. There is:

* **no ledger** covering the entropy factorial -- `ledger/runs.d.*` holds only
  `faithful_repair`, `faithful_repair.void_3443771_3443772`, `leap` and `pad16`; no row in
  `ledger/runs.jsonl` mentions `PW-H` or `entropy_factorial`;
* **no node or partition field** in `meta.json` (confirmed by the earlier rerun audit);
* **no scheduler record in the run directories** -- each holds only `metrics.npz` and
  `train_and_export.log`.

Partition therefore **cannot be established authoritatively for any canonical PW seed**. An
earlier mtime-to-sacct match suggested c23g x7 and c25g x1, but that matching is
non-injective (two exports matched one job) and was already recorded as unreliable in
`reports/walker_wml_anomaly_20260909.md`.

## Why this matters here specifically

The partition question is live and unresolved. A forward-evaluation probe
(jobs `3919251`, `3919252`) showed the two SKUs are bitwise identical on the inference path,
**refuting** the hardware explanation for inference -- but it did not test training kernels,
and the 8/8 association between assigned architecture and reproduction failure on Walker WML
remains unexplained. Comparing a new all-c23g analytic-gate arm against a baseline that is
half c25g would confound the gate change with that open effect.

## Options, for decision — I have chosen none

1. **Re-baseline.** Re-run the sampled-gate arms on c23g for all 8 seeds, then compare
   analytic vs sampled on matched hardware. Cleanest paired design; costs 16 extra runs.
2. **Restrict to c23g seeds.** Uses WML seeds 302/304/306/308 only -- n = 4, and PW partition
   is still unknown, so this does not actually resolve blocker 2.
3. **Proceed at n = 8 accepting mixed partitions**, recording the violation explicitly. This
   contradicts the stated preflight rule and would confound the gate contrast.

## What was NOT done

No `actor_kl_estimator` option added. No code diff produced. No validation tests run. No
smoke seed, no full launch. Stage 2 onward is untouched pending a decision on the above.
