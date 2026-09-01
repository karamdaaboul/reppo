# Analysis plan for the 64-run confirmatory dimension ladder

**Status: LOCKED POST-HOC ANALYSIS PLAN. This is not a pre-registration.**

Committed before the aggregation it governs is re-run, but *after* the
confirmatory outcomes were examined. It constrains threshold-shopping from this
point forward; it cannot and does not confer prospective status on anything below.

## 1. Blindness status at time of writing — stated honestly

The task instruction that prompted this file asked it to record "blind at time of
writing: walker (both arms, all seeds)". **That is not true and is not recorded
here.** The actual state is:

| task | status at time of writing |
|---|---|
| g1 | seen — group means *and* all 8 per-seed finals per arm |
| leap | seen — group means *and* all 8 per-seed finals per arm |
| hopper | seen — group means *and* all 8 per-seed finals per arm |
| **walker** | **seen — all 8 per-seed finals per arm, IQM per arm, paired stats, and the fresh-only n=5 sensitivity** |

All four tasks were reported at full per-seed granularity, together with paired
differences, exact permutation p-values, and unpaired IQM bootstrap CIs, before
this file existed. **No task is blind. No rule in this document is
pre-registered.** Rules R1--R5 in §5 were specified after those results were
visible and are therefore post-hoc by construction.

The prospectively registered content for this experiment lives in
`docs/prereg_dimension_ladder.md` (§§1--4 and Amendments L.0--L.1) and is
unaffected by this file. Where this plan and that one conflict, the earlier
document governs and the deviation is stated.

## 2. Relationship to the registered statistics

`prereg_dimension_ladder.md` §3 commits to **unpaired** resampling as primary,
reasoning that matched seed IDs do not implement common random numbers because
the two arms consume randomness through different code paths. It designates the
paired analysis a sensitivity check.

This plan's R1 makes the **paired** CI the decision rule. That inverts the
registered hierarchy. Both are reported; the registered unpaired IQM statistic is
reported first and labelled primary, and R1 is applied as a declared deviation.

## 3. Scope

- Per-task only. No pooling, no cross-task aggregation, no scaling fit in `d`.
- Task scales are not commensurable: DMC tasks are bounded [0, 1000]; g1 and
  leap have no registered upper anchor (L.1.13 leaves `U_t` **open**, and the
  window §2 allowed for setting it -- "before any confirmatory outcome is
  examined" -- has closed).
- Read-only. No config, seed, alpha, checkpoint or stopping rule is modified.
  No run is re-executed.

## 4. Extraction and statistics

Extraction: one committed script emitting one CSV row per run with task, arm,
seed, final return, final sigma, final ESS, final alpha_kl, collapse flag, NaN
flag and wall clock.

Two extraction limitations, both from the trainer's logging and neither fixable
read-only:

1. **`sigma` is a mean, not a per-coordinate median.** The log field is
   `train/pi_sigma_mean` (`scripts/train_and_export.py`), accompanied by
   `train/pi_sigma_min` / `train/pi_sigma_max`. Per-coordinate sigmas are never
   written, so the requested per-coordinate median cannot be recovered. The mean
   is emitted and the column is named `sigma_mean` to prevent misreading.
2. **Collapse has no pre-existing definition.** No collapse or exclusion rule was
   registered before outcomes. The flag below is post-hoc and labelled as such.

Statistics, per task, RNG seed **20260901** for every resample:

- Arm mean and median, 95% percentile bootstrap CI, 10^4 resamples.
- IQM per arm as the primary point statistic.
- Paired A-B differences on matched seed IDs, 95% paired percentile bootstrap CI,
  10^4 resamples, plus the exact 2^8 = 256 sign-flip permutation p-value.

## 5. Decision rules (fixed at this commit; not revisited)

- **R1.** A gap is "detected" on a task only if the paired-difference 95% CI
  excludes zero.
- **R2.** No operator claim is entered on a task where median final `sigma_mean`
  in the two arms differs by more than 1.5x. Policy width is then a candidate
  cause and is reported as such.
- **R3.** Hopper is excluded from the operator comparison and reported in full as
  a failure **iff** at least 4 of 8 seeds in **both** arms finish below 5% of the
  best return observed on that task **and** the seed-901 calibration run also
  finished below that floor. Otherwise hopper is reported as a result.
- **R4.** If the sign of the detected gap is not monotone in `d` across the four
  tasks, the paper does not claim a `d` trend; it reframes around omega and
  reports `d` as an ordering only.
- **R5.** Nothing here licenses a change to any run. Any anomaly is reported,
  never fixed.

## 6. Known anomaly recorded before aggregation

The ledger entry `cal901-hopper` -- the seed-901 calibration that produced the
frozen HopperHop `alpha = 0.00037288447492755949` in `slurm/ladder_matrix.sh` --
carries `status: "queued"` and `return_metrics: null`. Its numbers survive only
as prose in `prereg_dimension_ladder.md` L.1.12 (final-five calibration curve
43.47 -> 54.83 -> 145.38 -> 156.57 -> 163.07) and L.1.13 (re-eval mean 156.864532,
re-eval IQM 177.926046, per-episode p25 = 0.00). R3's calibration term is
evaluated against those recorded values, and the provenance gap is reported.
