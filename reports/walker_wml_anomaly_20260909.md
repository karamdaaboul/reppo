# The Walker-WML reproduction anomaly: partition association, causation NOT established

**Read-only. No training launched.** 2026-09-09.

## Finding, and its limit

**The four Walker-WML seeds that fail to reproduce were originally trained on a different GPU
partition than the reruns.** The association is perfect across all eight seeds.

**Causation is NOT established.** Partition is completely confounded with batch identity and
wall-clock time: the c23g seeds came from job `3444831` finishing 05:18-05:29 and the c25g
seeds from job `3444832` finishing 10:24-10:28. Any unobserved difference between those two
submissions is equally consistent with the data. A forward-evaluation test on both partitions was run and **refuted** the
hardware-arithmetic mechanism for the inference path (see "Decisive test" below). The cause
remains **unexplained**, and the words *solved*, *resolved* and *explained* are not used.

| seed | canonical job | partition | node | result vs rerun |
|---|---|---|---|---|
| 302 | `3444831_12` | **c23g** | n23g0020 | **exact** |
| 304 | `3444831_12` | **c23g** | n23g0020 | **exact** |
| 306 | `3444831_14` | **c23g** | r23g0001 | **exact** |
| 308 | `3444831_15` | **c23g** | n23g0015 | **exact** |
| 303 | `3444832_13` | **c25g** | w25g0006 | diverge |
| 305 | `3444832_14` | **c25g** | s25g0008 | diverge |
| 307 | `3444832_15` | **c25g** | w25g0006 | diverge |
| 301 | `3730619` | **c25g** | s25g0001 | diverge |

All 48 reruns ran on **c23g**. Correspondence is 8/8: every c23g-canonical seed reproduces
exactly, every c25g-canonical seed diverges.

`reports/covariance_freeze_phase1.md` records the hardware: **c23g = NVIDIA H100 94 GB,
c25g = NVIDIA H100 80 GB HBM3** — different SKUs.

The canonical Walker-WML baselines were produced by two sibling array jobs submitted together
and scheduled to different partitions: `3444831` on c23g (finishing 05:18–05:29) and `3444832`
on c25g (finishing 10:24–10:28), five hours apart on 2026-09-02. Seed 301 is separate again:
its export traces to `restore_wt_weighted_mle` and a later c25g job on 2026-09-06.

## Evidence ruling out other causes

**Code.** All 48 reruns record `git_commit = 964b251a7403d2517a38f1bd63b8918ea5a37df7`,
uniform. All eight canonical Walker-WML runs record `git_commit = None`, uniform. No
per-seed code difference exists.

**Configuration.** Every non-outcome field in `meta.json` is identical across all eight
canonical seeds — compared field by field against seed 302, excluding curves, `seed`,
`seed_index`, `hydra_run_dir`, `train_seconds` and training outcomes. Result: *identical
config* for all eight.

**Launch.** `seed_index = 0` and `num_seeds = 1` on both sides for every seed. The rerun
launcher replays the baseline overrides with only `seed` substituted.

**First divergence point — decisive.** Divergence begins at **evaluation index 0** for all
four divergent seeds, on *every* logged curve simultaneously:

| curve | first differing index (seed 305) |
|---|---|
| `eval_return_curve` | **0** |
| `kl_curve` | **0** |
| `entropy_curve` | **0** |
| `eta_curve` | **0** |
| `ess_curve` | **0** |
| `pi_sigma_curve` | **0** |
| `alpha_curve` | none (α is frozen) |

Accumulated floating-point drift would show agreement early and divergence later. Divergence
at the very first logged evaluation is the signature of a different execution substrate from
the first forward pass. The exact seeds agree at index 0 and everywhere after.

## What this does NOT explain

Partition does **not** account for G1 or LEAP.

| cell | canonical partitions | diverging |
|---|---|---|
| Walker WML | c23g {302,304,306,308} / c25g {301,303,305,307} | 4/8 — **perfectly associated** |
| Walker PW | c23g ×7, c25g ×1 | 0/8 |
| G1 WML | c23g {302,304,306,308} / c25g {301,303,305,307} | **8/8** |
| G1 PW | c23g {301,303,305,307} / c25g {302,304,306,308} | **8/8** |
| LEAP WML | **c23g ×8** | **8/8** |
| LEAP PW | **c23g ×8** | **8/8** |

G1 has canonical seeds on c23g that still diverge from c23g reruns, and **both LEAP cells are
entirely c23g and diverge on all sixteen runs**. That is same-hardware nondeterminism,
consistent with the pre-existing `G1_SAME_CODE_REPRODUCIBLE = NO` in
`reports/g1_nondeterminism.md`, and this extends the same finding to LEAP.

**Walker PW remains partly unresolved.** It reproduces 8/8, yet seed 303 appears to trace to a
c25g job. Either that match is wrong or the pathwise path is stable across these two H100
SKUs where weighted-MLE is not. **This cannot be settled from the present evidence** — see the
limitation below.

## Limitation of the job attribution

Canonical runs carry no partition or node field in `meta.json`. Attribution is by matching
each export's mtime to the nearest SLURM job end time, with matches beyond 120 s reported
`UNMATCHED`. **The mapping is not injective**: where jobs finish within seconds of each other,
two exports can match the same job (seeds 307 and 308 of Walker PW both matched `3734158_6`).

For Walker WML the attribution is strong — the two batches are five hours apart and all
matches land within ±3 s (seed 301 within 49 s). For cells whose exports finish minutes apart,
individual assignments should be treated as indicative only.

## Arm-distribution reproduction test

Since a setup difference was found for Walker WML, this test is confirmatory rather than the
fallback. It is the primary evidence for G1 and LEAP, where no setup difference explains the
divergence. `score_window3`, n = 8 per side, 10000 resamples, rng 20260914.

| cell | canon med | rerun med | two-sample diff [95% CI] | paired diff [95% CI] | P(rerun>canon) |
|---|---|---|---|---|---|
| Walker WML | 769.859 | 783.184 | +13.324 [−67.539, +107.390] | +0.000 [+0.000, +10.123] | 0.562 |
| Walker PW | 898.456 | 898.456 | +0.000 [−28.189, +28.189] | +0.000 [+0.000, +0.000] | 0.500 |
| G1 WML | 14.473 | 9.647 | −4.826 [−6.791, +1.922] | −2.952 [−6.074, +0.037] | 0.312 |
| G1 PW | 11.884 | 15.312 | +3.428 [−4.863, +6.592] | +0.550 [−0.774, +4.884] | 0.625 |
| LEAP WML | 24.931 | 28.707 | +3.776 [−6.976, +11.436] | +4.192 [−7.551, +10.726] | 0.625 |
| LEAP PW | 21.961 | 24.051 | +2.090 [−7.982, +12.521] | +1.278 [−5.355, +12.570] | 0.641 |

**Every interval contains zero, on both estimators, in all six cells.** Probability of
improvement spans 0.31–0.64 against an exchangeable value of 0.5; Walker PW is exactly 0.500
with zero difference.

**This is "no detectable difference", not established equivalence.** With n = 8 per side the
intervals are wide relative to the effects — G1 WML's two-sample interval spans −6.8 to +1.9
against a median shift of −4.8. Failure to reject a difference is not evidence of equality,
and no equivalence region was preregistered for this comparison.

## Decisive test — RESULT: REFUTED

Jobs `3919251` (c23g x2, node n23g0015) and `3919252` (c25g x2, node s25g0001), all
COMPLETED. Identical checkpoints, identical 3072-state bank (sha256 verified in-run),
identical PRNG key, source `3be8c111` clean on both, `reppo.py` sha256 `2c4ff079...`,
jax 0.5.2 / jaxlib 0.5.1 / Python 3.12.14, driver 580.178.04 on both. Device kinds do
differ as expected: `NVIDIA H100` on c23g, `NVIDIA H100 80GB HBM3` on c25g.

| comparison | result |
|---|---|
| within c23g (control) | **IDENTICAL** on all 3 checkpoints |
| within c25g (control) | **IDENTICAL** on all 3 checkpoints |
| **cross c23g vs c25g** | **IDENTICAL** on all 3 checkpoints, both pairs |

Both controls pass, so the design discriminates. Under the rule registered in advance:

```
VERDICT: REFUTED -- c23g and c25g agree bitwise on identical inputs.
```

**The GPU-architecture explanation is refuted for the forward/inference path.** `mu`,
`sigma` and sampled actions are byte-identical across the two SKUs for the weighted-MLE
divergent seed (s305), the weighted-MLE exact seed (s302), and the pathwise seed (s305).

### What the refutation does and does not cover

This probe exercised a narrow kernel set: the actor MLP forward pass, `tanh`, and a normal
sample. Training exercises far more — backward passes, the Adam update, MJX physics
stepping, replay indexing, and the E-step softmax and dual solve. **The hypothesis is
refuted for inference kernels and remains untested for training kernels.**

## The association is a designed A/B, not scheduler chance

`slurm/fr_launch.sh` line 24 filters a preregistered ledger by architecture:
`rows = [r for r in rows if r["gpu_architecture"] == arch]`. Reading
`ledger/runs_faithful_repair.jsonl`:

| seed | 301 | 302 | 303 | 304 | 305 | 306 | 307 | 308 |
|---|---|---|---|---|---|---|---|---|
| assigned arch | c25g | c23g | c25g | c23g | c25g | c23g | c25g | c23g |

The alternation was **deliberate**, written into the ledger before launch. The two batches
were submitted in the same second (2026-09-02T04:38:44), from the same script, same array
range `0-15`, same resources (24 CPU, 120000M, 1 GPU); the submit lines differ only in
`--partition` and the derived `FR_ARCH`. Ledger commands are identical apart from `seed=`
and `hydra.run.dir=`.

That makes the partition contrast a clean designed A/B rather than a confound — which
*strengthens* the association while leaving its mechanism unexplained.

## Current status: UNEXPLAINED

The Walker-WML divergence is perfectly associated (8/8) with a deliberately assigned GPU
architecture, no other setup difference exists, and the architecture's arithmetic is
bitwise identical on the inference path. The cause is **not established**.

The one hypothesis still standing is that a *training-path* kernel differs across the two
SKUs. The test would be a short training run — a few hundred updates, identical config and
seed, one per partition, weights compared — which requires new training and is **not
launched**.

## Consequences

* The Walker-WML anomaly is **unexplained**. Its leading candidate was tested and refuted. What is
  already established independently of it: the instrumentation is a proven no-op on Walker PW
  (8/8 bitwise) and on the four same-partition Walker-WML seeds. No evidence anywhere
  indicates the instrumentation changed training.
* Rerun validity by bitwise identity is **established on Walker PW and on the four
  c23g-canonical Walker-WML seeds**; it is **not testable** on G1 and LEAP.
* The reruns reproduce canonical behaviour at arm-distribution level on all six cells, with
  the power caveat above. The §7 KL-split result should continue to be read as a statement
  about the **arm**, which is how it is currently written.
* **New, actionable:** the canonical baselines are split across two GPU SKUs within single
  cells. Any future comparison requiring bitwise reproduction must pin the partition. The
  held-out replication should pin one partition for all arms and seeds and record it.

Artifacts: `/hpcwork/qzi10910/gates/anom_distribution.out`, scripts `anom1.py`–`anom6.py` in
`/hpcwork/qzi10910/gates/`.
