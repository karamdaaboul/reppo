# Pre-registration: HumanoidRun (d=21) at larger estep_num_samples

**Status.** Committed before any run is launched. The control arms are outcome-seen (they
are the published cohort); nothing in Sec. 3-5 was chosen after seeing an M != 32 result.

**Revision history.** v1.0 (2026-09-01): as committed.

**Companion.** `docs/prereg_m_star.md` (Tracks A/B, the LQR M* surface).

---

## 1. What this tests

The LQR M-sweep (commit `8e41e64`) found that at `d = 21, M = 32` both zeroth-order
operators sit at cosine 0.55-0.66 to the exact estimand while pathwise sits at 0.975, and
that the deficit closes to ~0.99 by `M = 2048`. That predicts the published HumanoidRun
gap is **under-sampling, not an operator crossover**. This is the test in the real system.

It is a PREDICTION made before the run, recorded at `8e41e64`.

---

## 2. The control, reproduced exactly

From `exports/*/meta.json`, `final_eval_return`:

- **A** = `HumanoidRun_pathwise_fa_s{1..9}_final`, n = 9, mean **738.614**, sd 44.316
- **B** = `HumanoidRun_weighted_mle_s{0,1,2,3,5,6,7,8}_final`, n = 8, mean **666.174**, sd 61.333
- **gap A - B = 72.440**, pooled two-sample t = **2.816** (matches the published
  `-72.4 (t=-2.82)` at `docs/prereg_action_padding.md:35`, sign convention B-A)

**The control is NOT re-run.** Arm A is reused as-is at every M.

---

## 3. Design

`env=mjx_dmc env.name=HumanoidRun experiment_overrides=mjx_dmc_large_data`,
`actor_update_mode=weighted_mle`, `eps_e=0.5`, `mstep_decoupled=false`,
`update_entropy_lagrangian=false`, **`ent_start=0.00329`** (the frozen alpha of the
committed d=21 arm; class-A provenance, `docs/prereg_dimension_ladder.md:1452-1459`),
`total_time_steps` as shipped (50M configured, ~52.3M executed).

**Swept: `estep_num_samples` in {128, 512}.** Control is the existing M=32 arm.

### 3.1 Seeds, and why this arm can never be confirmatory

**Seeds 201+ only.** Two independent reasons, both registered:

1. `scripts/train_and_export.py:177-196` builds the export tag as
   `HumanoidRun_weighted_mle_s{seed}` -- **it does not encode M**. Re-running seeds 0-10
   would silently OVERWRITE the committed M=32 exports and destroy the baseline.
2. Seeds 101-108 are RESERVED confirmatory (`ledger/README.md`) and must not be spent
   here.

`ledger/README.md` marks 201+ as EXPLORATORY / ALGORITHM DEVELOPMENT, "never" confirmatory
evidence. **This arm is therefore permanently exploratory by construction, and so is its
control** (the published cohort used seeds 0-10, a retrospective namespace). The result may
inform the paper's framing; it may not be presented as confirmatory evidence.

Target n = 8 per M. **Minimum n = 5**; below that the arm is reported as underpowered and
NO adjudication is made.

### 3.2 The KL coupling -- a declared confound, not a plumbing gap

`estep_num_samples` sets `n_estep` (`src/jaxrl/reppo.py:719-723`), which feeds **three**
consumers: the shared action draw, the E-step proper, AND the forward-KL Monte-Carlo
estimate (`reppo.py:731-742`) that drives the KL clip and the KL Lagrangian.

**So raising M from 32 to 512 also raises the KL estimator's sample count from 32 to 512.**
The comparison moves two axes, not one.

**Registered choice: leave the coupling INTACT.** Reasons: it is the single knob a
practitioner actually turns; it matches the ladder's registered convention that the mode
also sets the KL sample count (`slurm/ladder_matrix.sh:12-14`, amendments L.1.14/L.1.17,
"deliberately left intact"); and the alternative (`kl_num_samples=32`) exists only in an
uncommitted working tree and is labelled DEVELOPMENT ONLY (`reppo.py:136-143`).

**Registered consequence:** if the gap closes, the result does NOT by itself attribute the
closure to the E-step sample count rather than to the improved KL estimate. Attribution
requires a follow-up arm at fixed `kl_num_samples=32`, which is named here as required
future work and is NOT run now.

### 3.3 NaN divergences

Historically **6 of 30** HumanoidRun `weighted_mle` run directories NaN-aborted
(`scripts/train_and_export.py:158-171` raises `SystemExit(2)`, exports nothing). Within the
two committed arms specifically it is 2 of 21 (A seed 0, B seed 4).

**Registered rule: REPLACE, and report.** On NaN, relaunch with the next unused seed in the
201+ block. Every NaN is reported with its seed and run dir. This matches what the
committed cohort did (seed 4 -> seed 10). **Cap: 3 replacements per arm**; beyond that the
arm is reported as unstable at that M and no adjudication is made for it.

---

## 4. Primary outcome and decision rule (committed)

Primary outcome: **final return, arm A minus arm B, at each M**, pooled two-sample t
against the unchanged A cohort (n=9). Current gap **+72.440**.

Let `G_M` be the point estimate at sample size M, with 95% CI from the pooled t.

- **CLOSED** if the 95% CI contains 0 AND `G_M < 24.1` (one third of 72.4).
- **PARTIAL** if `24.1 <= G_M < 48.3`, regardless of CI.
- **UNCHANGED** if `G_M >= 48.3` and the CI excludes 0.
- **WORSE** if `G_M > 96.5` (72.4 + one third).
- **INCONCLUSIVE** if none of the above applies, or n < 5, or the arm is unstable.

Conclusions, registered:

- CLOSED at either M -> the d=21 deficit is a finite-sample effect of `estep_num_samples`.
  Section 7.6 is rewritten. The LQR prediction transfers. Attribution vs the KL confound
  (Sec. 3.2) remains open.
- UNCHANGED at both M -> the deficit is NOT about M. The LQR M-sweep does not transfer to
  DMC, and the LQR result is then evidence about the estimator in an LQR only. Section 7.6
  stands, and `docs/prereg_m_star.md`'s dimension hypothesis is not sufficient to explain
  the DMC gap.
- PARTIAL -> report the fraction closed; both mechanisms are in play.

**Power, stated in advance.** Pooled sd ~53, so with n=9 vs n=8 the SE of the gap is ~25.8
(consistent with the observed t = 2.816 for a 72.4 gap). The design has good power to
detect the full 72-point gap, but **cannot resolve a residual gap of ~26 from zero**. A
CLOSED verdict therefore means "not distinguishable from zero at this n", not "zero".

---

## 5. Compute asymmetry -- registered as a reported outcome, not a footnote

Pathwise queries the critic on `2048` rows per minibatch; `weighted_mle` queries it on
`M x 2048` (`reppo.py:764-768`). At M=512 that is a **512x** critic-row ratio against
pathwise and **16x** against the committed M=32 arm.

**Registered: report, alongside return, for every arm --** `train_seconds` from
`meta.json`, critic rows evaluated per environment step, and the wall-clock ratio to arm A.
A win at M=512 is reported with its price. Recorded baseline: arm B at M=32 took ~4400 s
per seed; arm A took 1523-2397 s.

---

## 6. Smoke test before committing to full length

One seed per arm at `total_time_steps=131072`, `num_eval=1` (one iteration). Note a literal
2000-step smoke **crashes** with ZeroDivisionError (`reppo.py:1164-1170`: `eval_interval`
floors to 0). Confirm: the config takes; `estep_num_samples` is echoed into `meta.json`;
ESS and logit spread are in range; memory holds at M=512 on GPU 0 (32 GB -- GPU 1 has 20 GB
and `XLA_PYTHON_CLIENT_PREALLOCATE=true` takes the whole card).

---

## 7. Provenance caveat, declared

The working tree is dirty: `src/jaxrl/reppo.py` and `config/reppo.yaml` carry an
uncommitted `kl_num_samples` field plus a refactor of the flag-gated `log_estimator_diag`
block. With `kl_num_samples` unset (its default) and `log_estimator_diag` false, behaviour
matches HEAD -- the diagnostics refactor is verified bit-identical by
`scripts/verify_estep.py` check_a at `0.000e+00`. But **the `git_sha` recorded in
`meta.json` will not fully describe the code that ran**, and that is disclosed here rather
than discovered later.

---

## 8. Out of scope

- Only `estep_num_samples` is varied. No claim is made about other tasks, other `eps_E`,
  or the pathwise arm's own M dependence.
- Exploratory namespace (Sec. 3.1): this cannot become confirmatory evidence.
- The KL confound (Sec. 3.2) is not resolved by this design.
