# Rerun-to-canonical validity audit

**Read-only.** No training launched, no reruns, no seeds/configs/exclusions/results/artifacts
modified. Executed 2026-09-09T17:15:10+0200.

```
command:  ./.venv/bin/python scripts/analysis/audit_rerun_validity.py
script:   scripts/analysis/audit_rerun_validity.py
script sha256: d96df1eec99090844c715b9b8dddade8b07e18729fea87022923d77ff184bbc5
canonical tree: /hpcwork/qzi10910/estep_wt     HEAD 1ff064e5493ee2159f60629eb33116924cd9f07f  tracked-clean YES
rerun tree:     /hpcwork/qzi10910/logsplit_wt  HEAD 964b251a7403d2517a38f1bd63b8918ea5a37df7  tracked-clean YES
stdout:   /hpcwork/qzi10910/gates/audit_rerun_validity.out
per-seed: reports/artifacts/audit_rerun_validity.csv  (48 rows)
```

## 0. Why the previous check silently covered 2 of 6 cells

`scripts/analysis/klsplit.py` resolved canonical runs through
`reports/artifacts/exports_manifest.csv`. **That manifest contains zero rows tagged
`pathwise_fa_noent`** (it holds 71 `pathwise_fa` and 85 `weighted_mle` rows). All three PW
cells and LEAP WML therefore resolved to nothing and printed `canonical metrics not found`,
while the two that resolved printed numbers. No count was asserted, so the gap was invisible.

This audit ignores the manifest and resolves from `exports/*/meta.json`.

## 1. Commit provenance — the stated frozen commit is not what ran

The recorded frozen project commit `07319d4862f3` ("Ignore the bootstrap symlinks so the
ladder can actually submit", 2026-09-01) **is an ancestor of both working trees but is not
the commit of any run in this audit.**

Justification for the actual commit used: these reruns exist to log the forward-KL mean/width
decomposition, which enters the source at `0502b8f` — a *descendant* of `07319d4`. `07319d4`
cannot produce `fr_kl_mean_part_med` or `fr_kl_width_part_med` at all, so it could not have
produced these runs. The reruns ran at `964b251`, which carries the instrumentation plus the
`eps_e` export-tag fix.

**All 48 reruns record `git_commit = 964b251a7403d2517a38f1bd63b8918ea5a37df7`** — uniform.

The canonical side is not uniform. Recorded `git_commit` across the 48 canonical runs:

| canonical `git_commit` | runs |
|---|---|
| `e9a937395a0eca048cc0b848afa38e72e61b0689` | 8 |
| `7be5b05c3fbba374ee3419727bf4a8db44e34bc8` | 8 |
| `918f82c82a81935af905ef8f7d64fce1bfcf021e` | 8 |
| `4665b08de73cf19ef438cfc9c4d9a1c976cf5dc5` | 8 |
| **not recorded (`None`)** | **16** |

**The 48 canonical baselines were produced at four different commits, with 16 runs recording
no commit at all.** This is reported, not resolved; it is a provenance limitation of the
existing baselines, independent of this audit.

## 2. Metadata gap that forced an explicit key extension

`meta.json` does **not** record `update_entropy_lagrangian` anywhere — not at top level, not
in `actor_kwargs`. Verified further: `alpha_entropy`, `ent_start` and `alpha_curve` are
**identical** between the `_ent` and `_noent` exports of every task (e.g. Walker
`alpha_entropy = 0.0145099`, `alpha_curve` constant, in both).

Consequence: a metadata-only key resolves each rerun to **two** canonical runs. The first run
of this audit correctly reported 48 duplicate matches and `OVERALL: FAIL`.

Resolution: the export tag is added to the key. It is **config-derived** —
`scripts/train_and_export.py` builds it from the resolved config — so it is provenance stored
in the path, not a guessed directory name. It is the only carrier of that configuration bit.
This is the sole key extension and it is documented in the script.

## 3. Per-cell audit

All six cells: 8 expected, 8 found, 8 matched, 8 compared; no missing seeds, no duplicates;
both return series exactly 21 values; no NaN or non-finite values.

| task | arm | tag | identical | max abs | max rel | max Δ score_window3 | worst |
|---|---|---|---|---|---|---|---|
| Walker | PW | `pathwise_fa_noent` | **8/8** | **0** | **0** | **0** | — |
| Walker | WML | `weighted_mle` | **4/8** | 162.518 | 0.345 | 107.424 | s305, idx 13 |
| G1 | PW | `pathwise_fa_noent` | 0/8 | 7.687 | 10.244 | 6.438 | s307, idx 19 |
| G1 | WML | `weighted_mle` | 0/8 | 12.050 | 30.326 | 6.769 | s302, idx 11 |
| LEAP | PW | `pathwise_fa_noent` | 0/8 | 16.246 | 6.450 | 12.910 | s305, idx 16 |
| LEAP | WML | `weighted_mle` | 0/8 | 31.125 | 206.081 | 18.465 | s302, idx 7 |

Large relative differences on G1 and LEAP are inflated by returns near zero and should not be
read as proportional error.

Per-seed maximum absolute differences for the two Walker cells:

* Walker PW — all eight seeds exactly `0`.
* Walker WML — s302, s304, s306, s308 exactly `0`; s301 `62.3`, s303 `94.7`, s305 `163`, s307 `90.3`.

### Configuration discrepancies

Whitelisted as documented storage/export differences: `hydra_run_dir` (different worktree, by
design), `train_seconds` (wall clock).

Benign schema additions, reported not whitelisted: `freeze_sigma` — the **only** differing key
inside `actor_kwargs`, present in the newer export with value `None` on both sides; plus
`sqrt_rho`, `git_diff_sha256`, `git_dirty`, absent from older canonical exports.

Training-outcome fields that differ **only where the run diverged**, and are not configuration:
`alpha_kl`, `ess_final`, `final_eval_return`, `entropy_curve_full`. On Walker WML these differ
on exactly the 4 divergent seeds; on Walker PW only `git_commit` differs.

**No configuration discrepancy was found in any cell.**

## 4. Totals

```
EXPECTED CELLS:     6
COMPARED CELLS:     6
EXPECTED RUN PAIRS: 48
COMPARED RUN PAIRS: 48
UNMATCHED:          0
DUPLICATE MATCHES:  0
FAILED ASSERTIONS:  0
OVERALL: PASS
```

**`OVERALL: PASS` is a coverage verdict only** — it certifies that all 6 cells and all 48 seed
pairs were located, uniquely matched and fully compared, with no silent skip. It does not
certify that the instrumentation left training unchanged.

## 5. What rerun validity is and is not established for

**Established on 1 of 6 cells.** Walker PW is bitwise identical on all 8 seeds across all 21
evaluations. There, the KL-split instrumentation is demonstrably a training no-op.

**Not testable on 4 of 6 cells.** `reports/g1_nondeterminism.md` records
`G1_SAME_CODE_REPRODUCIBLE = NO` — same commit, config and seed diverge run to run. For both
G1 cells and both LEAP cells the comparison cannot separate "the instrumentation changed
training" from "the task is not reproducible". Marked **not testable**, not passed.

**Inconclusive and anomalous on 1 of 6 cells.** Walker WML is identical on 4 seeds and
differs on 4, by up to 162.5 return. The same document states Walker is bitwise reproducible
(`T1 = 0.000e+00`), and Walker PW confirms it 8/8 here on the same nodes. **The weighted-MLE
path on Walker is therefore reproducible on half its seeds and not the other half, which the
existing documentation does not explain.** This is a live open question, not a resolved one.

> **Follow-up 2026-09-09:** investigated in `reports/walker_wml_anomaly_20260909.md`. The four
> divergent seeds trace to canonical runs on the **c25g** partition while the four exact seeds
> trace to **c23g**, and all reruns ran on c23g — an 8/8 association. Causation is **not**
> established: partition is confounded with batch and wall-clock time. A cross-partition
> forward-evaluation probe with a within-partition control is registered and running
> (jobs `3919251`, `3919252`). This cell stays **inconclusive** until it returns.

Consequence, unchanged from `reports/reruns_and_standards.md` and now quantified: the reruns
are valid as **fresh samples of the same arm** — same code, config and seeds — and are **not**
instrumentation of the specific canonical checkpoints on any cell except Walker PW. Every KL
split statement should be read as a statement about the arm.

## 6. Limitations this audit does not address

Explicitly retained and **not** validated here:

* **Bank-rollout correctness.** Bank hashes, state counts, depth ladders and generator
  equivalence were checked elsewhere; that the rollouts are physically correct was not.
* **`policy_dist()`.** Used throughout the width analyses; never audited.
* **The §8 fixed-normalizer intervals.** The min–max normaliser is estimated once on the full
  sample and held fixed through the bootstrap, making those IQM intervals anticonservative.
