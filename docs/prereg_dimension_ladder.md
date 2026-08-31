# Pre-registration: operator gap across an action-dimension ladder (v2)

**Status.** Commit before any confirmatory run launches. Companion to
`docs/prereg_action_padding.md` (d1ab422) and
`docs/prospective_padding_error_field_analysis.md`.
v2 incorporates external review: anchor/ceiling separation, mixed-provenance
Spearman handling, resampling specification, O1 tightening, mechanical
calibration definitions, exact-SHA governance, minimum-qualified-tasks rule.
No confirmatory outcome existed at either version.

**What this experiment is.** An associational test: across independently
calibrated environments spanning roughly d = 4 to d = 29, do task-level
A−B gaps exhibit the rank pattern predicted by the direct operator and
error-field measurements? Task identity and dimension are confounded by
design, so no outcome licenses a causal "dimension drives the gap" claim.
The mechanistic weight stays on Probes 1--7. This ladder is
external-validity evidence.

## 1. Tasks, arms, seeds

| task | suite | nominal d | provenance |
|---|---|---|---|
| HopperHop | MJX DMC | 4 | prospective |
| WalkerRun | MJX DMC | 6 | 3+3 outcome-seen; top-up to 8+8 |
| LEAP cube RotateZ | MuJoCo Playground | ~16 | prospective |
| HumanoidRun | MJX DMC | 21 | retrospective (A n=9, B n=8, frozen-α, already reported) |
| G1 joystick (default scale) | MuJoCo Playground | ~29 | prospective |

The d of every task is verified from the environment spec of the exact
build at launch and recorded in Amendment L.1; the verified value is used
in analysis.

Arms: pathwise (A) and zeroth-order weighted-MLE (B), α frozen in both,
M = 32, ε_E = 0.5, action_pad = 0, no per-task hyperparameter tuning.

**Code governance.** Every launch runs at the exact SHA recorded in
Amendment L.1. If that SHA differs from the audited baseline `3b96deb`,
the machine-precision parity check against the pristine snapshot is
re-run and its result recorded before any confirmatory run.

Seeds: confirmatory 101--108 (8 per arm per task). Calibration seed 901
per new task, excluded from all confirmatory comparisons. WalkerRun keeps
its existing 3 frozen-α seeds per arm and adds confirmatory seeds
104--108; pooled (n=8) and fresh-only (n=5) results are reported
separately. HumanoidRun enters as-is, labelled retrospective.

**α-freezing for new tasks (mechanical).** One learned-α calibration run
(seed 901) per task. α is frozen at the median of the logged α values
over all evaluation checkpoints excluding the first two (warm-up). The
computed value is recorded in Amendment L.1 before confirmatory launch.
Same single-calibration-seed provenance statement in the paper as for
WalkerRun/HumanoidRun.

## 2. Anchors and qualification (committed before confirmatory outcomes)

Normalization and ceiling qualification are separate devices.

**Normalization anchors.** S_t(R) = (R − L_t)/(U_t − L_t).
- DMC tasks: L_t = 0, U_t = 1000 (independent task scale).
- LEAP and G1: U_t = 1.1 × the best checkpoint evaluation IQM of the
  calibration run (never a best single episode); L_t = the spec minimum
  return. If the spec permits negative returns, L_t is taken from the
  spec, not assumed 0. Anchors are recorded in Amendment L.1 before any
  confirmatory outcome is examined. If anchors are judged indefensible at
  that point, the task is excluded from cross-task aggregation and
  reported at task level only (gap + probability of improvement, which is
  anchor-free).

**Qualification gates**, applied to the calibration seed only (never to
comparison-arm outcomes):
- *Ceiling-censored:* calibration final IQM ≥ L_t + 0.90 (U_t − L_t),
  **available only where U_t is independent of the calibration run** —
  i.e., DMC tasks, or a spec-defined maximum/success criterion. For LEAP
  and G1 without such a criterion, the ceiling gate is recorded as
  *unavailable*; those tasks cannot be ceiling-censored, and this is
  stated rather than manufactured from the calibration run itself.
- *Floor-uninformative:* calibration final IQM ≤ L_t + 0.05 (U_t − L_t)
  → not learnable in budget; disqualified.
- Recorded here: WalkerStand/Walk, HumanoidStand/Walk, CheetahRun were
  previously screened and are ceiling-censored; the same-morphology
  difficulty control is therefore ceiling-censored at the final
  checkpoint and is not run. Learning-speed comparisons on such tasks may
  be reported descriptively; they carry no final-return claim.

**B-arm health** (diagnostic gate, not a tuning license): a B run is
configuration-valid if median ESS ∈ [8, 28] and no entropy collapse under
the paper's frozen-α definitions. Unhealthy runs are reported with the
flag, never dropped, and no per-task tuning occurs in response.

## 3. Statistics (committed)

**Per task.** Δ_t = IQM(S_{t,A}) − IQM(S_{t,B}) over confirmatory seeds,
95% bootstrap CI (10,000 resamples, `np.random.default_rng(20260901)`),
plus rliable probability of improvement. **Resampling scheme: the two
arms are treated as unpaired.** Reason: matched seed IDs do not implement
common random numbers — the arms consume randomness through different
code paths — and HumanoidRun's 9-vs-8 retrospective arms require
unpaired handling; one convention is used for all tasks. Seeds are
resampled independently within each arm. A paired analysis over matched
seed IDs is reported as a sensitivity check where IDs match. The same
convention applies to probability of improvement. All attempted seeds
reported, including divergences. Raw t-tests to an appendix as a
secondary check.

**Across tasks.** Spearman rank correlation of Δ_t with verified d over
qualified tasks, with an exact permutation p-value by full enumeration of
all n! orderings. Two versions are reported:
1. *Main ladder summary:* all qualified tasks, explicitly labelled a
   mixed prospective/retrospective analysis (Walker partly outcome-seen,
   Humanoid retrospective).
2. *Prospective sensitivity:* Hopper + LEAP + G1 + Walker fresh-only
   seeds, excluding Humanoid (n ≤ 4; reported for transparency, not
   inference).
The permutation p is a descriptive association statistic over this fixed
task set, not population-level inference. **If fewer than four tasks
qualify, no cross-task correlation or p-value is interpreted; results
remain task-level only.** No fitted scaling exponent is claimed under any
outcome.

## 4. Interpretation map (committed)

- **O1.** Δ_t CIs exclude 0 in the pathwise direction in *both* qualified
  high-dimensional tasks (d ≥ 21) and include 0 in *both* qualified
  low-dimensional tasks (d ≤ 6) → the paper may state: "the operator gap
  is detected in both qualified high-dimensional tasks and not detected
  in both qualified low-dimensional tasks; the intermediate-dimensional
  task is reported separately." An overall positive cross-task
  association is described only if the rank statistic of §3 supports it.
- **O2.** Δ_t CIs exclude 0 in the pathwise direction at low d with
  magnitude comparable to high d → the dimension association is not
  supported; the paper leads with the measured error-field results and
  reports the ladder as such.
- **O3.** Mixed outcomes, or qualification leaves fewer than four tasks →
  per-task reporting only; no cross-task sentence.

"Not detected" is the licensed phrase; a CI including zero is never
described as evidence of no effect. No outcome changes the padding
contamination verdicts or promotes the ladder above the mechanistic
probes.

## 5. Compute and schedule

The campaign comprises ~61 relevant runs (Walker top-up 10; Hopper, LEAP,
G1 confirmatory 48; calibration 3). The number of *remaining launches*
depends on the provenance answers in §6: previously launched HopperHop
and G1 calibration outputs, if usable, reduce new launches to ~59. Before
scheduling, one full benchmark run per new task is timed and its
GPU-hours recorded in Amendment L.1; no day-level schedule is claimed
before those measurements exist. Submission cutoff: runs completed by
2026-09-14 enter the submission analysis; later completions are reported
as pending, not silently added or dropped.

## 6. Provenance

Committed after: all padding results (k = 0/6/16), Probe 1, the existing
WalkerRun (3+3) and HumanoidRun (9+8) frozen-α comparisons.
Committed before: any Hopper, LEAP, or G1 comparison outcome and before
the WalkerRun top-up seeds.

HopperHop and G1 calibration runs were launched previously. Fill in
before first confirmatory launch (Amendment L.1):
- [ ] Do HopperHop / G1 / LEAP calibration outputs exist on disk? (paths)
- [ ] Were any inspected before this commit? (yes/no, by whom)
- [ ] Were they learned-α runs usable for the §1 α-freezing procedure?
Whatever the answers: those runs are calibration-only and are excluded
from confirmatory comparisons; if inspected, the affected task's anchors
and qualification are labelled retrospective.

## 7. Out of scope

- Any causal claim that dimension drives the gap.
- Any fitted √d (or other) scaling law.
- Any reuse of padding-cohort runs in this ladder.
- Any per-task hyperparameter tuning in either arm.
- Any cross-task claim if fewer than four tasks qualify.

## Amendment L.1 (append at launch, before confirmatory outcomes)

Exact SHA per launch (+ parity-check result if ≠ 3b96deb); verified d per
task; LEAP/G1 reward range from spec and anchors (L_t, U_t) with the
calibration checkpoint-IQM used; evaluation-episode count per checkpoint;
frozen-α values per task; benchmark GPU-hours per task; calibration-output
provenance answers; exact launch commands.

---

## Pre-launch Amendment L.0 — provenance correction (2026-08-31)

**Append-only.** Everything above this rule is the registered v2 text as
committed in `f103642a35d11df0ac7b278bab76a4219fa2cb52` and is unchanged;
sha256 of the leading 183 lines is
`f57703af853332d917615ca3ab5eef004ecf5e167a9ab33ed58d6a6deb346e27`. This
amendment supersedes the wording identified below rather than editing it.
No confirmatory run has been launched. **No performance outcome was
inspected in producing this amendment**: `final_eval_return`,
`eval_return_curve` and every other evaluation-return field were left
unread, and only configuration, α-trajectory and filesystem metadata were
consulted.

### L.0.1 Existing frozen-α comparison cohorts are retrospective/pilot

Frozen-α A-arm and B-arm cohorts for three tasks existed on disk before
this preregistration was committed. They are recorded here as
**retrospective/pilot** cohorts.

| task | retrospective/pilot A (frozen-α, pathwise) | retrospective/pilot B (weighted-MLE) |
|---|---|---|
| HopperHop | s0, s1, s2 | s0, s1, s2 |
| LeapCubeRotateZAxis | s0, s1, s2, s3, s4 | s0, s1, s2, s3, s4 |
| G1JoystickFlatTerrain | s0, s1, s2 | s0, s1, s2 |

All 22 export directories carry filesystem modification times between
2026-08-29T22:53:46 and 2026-08-31T02:52:12, i.e. every one of them
predates the preregistration commit at 2026-08-31T15:09:37+00:00.

**Supersession.** The `provenance` column of the §1 table labels
HopperHop, LEAP cube RotateZ and G1 joystick as `prospective`, and §6
describes the previously launched HopperHop and G1 outputs as
*calibration runs* while omitting LEAP. Both are superseded. The task is
not the unit of provenance; the cohort is. The correct distinction is:

- **existing s0--s4 cohorts: retrospective/pilot** — for all three tasks,
  LEAP included;
- **seeds 101--108: the prospective confirmatory cohort.**

The retrospective/pilot Hopper, LEAP and G1 runs are **not** included in
the primary prospective IQM, bootstrap CI, probability of improvement, or
task-level gap Δ_t used for the prospective ladder. They may later be
reported separately, and labelled, as retrospective supporting evidence.

Unchanged by this amendment: WalkerRun keeps its existing outcome-seen
3+3 cohort plus fresh seeds 104--108, with pooled n=8 and fresh-only n=5
reported separately; HumanoidRun remains retrospective.

### L.0.2 LEAP α protocol deviation

Recorded without reference to any performance outcome. The §1 mechanical
rule freezes α at the median of the logged α values over all evaluation
checkpoints excluding the first two (warm-up). For LEAP:

- calibration-median α under the rule (excluding two warm-up
  checkpoints): **0.0010604216950014234**;
- α actually used by the existing LEAP frozen-α A/B runs: **0.00094**,
  which is instead the calibration run's *first* checkpoint value
  (0.000944976) rounded to two significant figures.

The existing LEAP cohort therefore does not conform to the ladder's
preregistered α-freezing rule and remains retrospective/pilot only on
that ground as well as on the provenance ground of L.0.1.

**Prospective frozen-α values.** For seeds 101--108 the mechanical rule
is applied verbatim to the calibration trajectory. The exact values, read
from the calibration metadata's α trajectory with no evaluation-return
field consulted, are:

| task | prospective frozen α (median of `alpha_curve[2:]`) |
|---|---|
| HopperHop | 0.00035225937608629465 |
| LeapCubeRotateZAxis | 0.0010604216950014234 |
| G1JoystickFlatTerrain | 0.00022178766084834933 |

These values are fixed by the rule and by the calibration trajectory
alone. α is not tuned against any existing A/B result, and no existing
A/B result was examined.

For reference, the α actually frozen in the retrospective/pilot cohorts
was 0.00035 (Hopper) and 0.00023 (G1) — consistent with the mechanical
rule to two significant figures — and 0.00094 (LEAP), which is not.

### L.0.3 Verified action dimensions

The §1 table's nominal `~16` and `~29` are superseded by the verified
values below, read from the `action_dim` field of each task's exported
checkpoint metadata. The historical text is not rewritten; these are the
values used in analysis, per §1's instruction that the verified value
governs.

| task | verified d |
|---|---|
| HopperHop | 4 |
| WalkerRun | 6 |
| LeapCubeRotateZAxis | 16 |
| HumanoidRun | 21 |
| G1JoystickFlatTerrain | 29 |

### L.0.4 Cross-task inference caveat

The task ladder is descriptive/associational and no binary scientific
conclusion is based on a cross-task p<0.05 threshold. With exactly four
qualified tasks, an exact two-sided Spearman permutation test has only
4! = 24 permutations and its smallest attainable two-sided p-value is
approximately 0.083. Therefore, if n=4 tasks qualify, report rho and the
exact permutation p-value descriptively but make no significance claim.
If fewer than four tasks qualify, retain task-level reporting only as
already preregistered.

For n=5, use exact enumeration of all 5! permutations rather than a
Monte-Carlo permutation approximation.

### L.0.5 Calibration provenance

Filesystem and run provenance only; no evaluation-return field was opened
to establish any of it.

| task | learned-α calibration output exists | export path | hydra run dir | export mtime |
|---|---|---|---|---|
| HopperHop | yes | `exports/HopperHop_pathwise_s0_final` | `outputs/2026-08-29/14-35-29` | 2026-08-29T15:01:49 |
| LeapCubeRotateZAxis | yes | `exports/LeapCubeRotateZAxis_pathwise_s0_final` | `outputs/2026-08-30/12-40-42` | 2026-08-30T13:29:14 |
| G1JoystickFlatTerrain | yes | `exports/G1JoystickFlatTerrain_pathwise_s0_final` | `outputs/2026-08-29/22-25-28` | 2026-08-29T23:16:22 |

All three hydra run directories are present on disk. Each calibration run
is `actor_update_mode: pathwise` with learned α (a non-constant
`alpha_curve` over 21 checkpoints), so each is usable for the §1
α-freezing procedure.

One deviation is recorded: §1 specifies calibration seed 901 per new
task, whereas all three existing calibration runs are **seed 0**. The
α-freezing rule itself is unaffected — it reads a single calibration
trajectory — but the seed identifier differs from the registered text,
and the calibration seed is in any case excluded from all confirmatory
comparisons.

**Inspection status.**

- This coding session has inspected performance outcomes: **NO.**
- Author prior inspection: **TO BE FILLED BY KARAM.**

If the author had previously inspected a task's calibration performance,
that task's anchor and qualification decision is labelled retrospective,
per §2 and §6 of the original preregistration.

### L.0.6 Outstanding before the first prospective launch (seeds 101--108)

Recorded here as a readiness checklist; each item is filled in Amendment
L.1 at launch. Anchors and qualification are deliberately **not**
computed in this amendment, because both require opening
performance-return fields.

- [ ] LEAP and G1 normalization anchors (L_t from spec, U_t from the
      calibration checkpoint IQM), and the DMC anchors confirmed.
- [ ] Qualification result per task, from permitted calibration
      information only.
- [ ] Exact prospective frozen α per new task — computed in L.0.2 above;
      to be restated in L.1 alongside the launch command that consumes it.
- [ ] Benchmark GPU-hours per new task, from one timed full run.
- [ ] Exact launch SHA, plus the machine-precision parity check against
      the pristine snapshot if that SHA differs from `3b96deb`.
- [ ] Exact launch commands.
- [ ] The author's calibration-inspection answers (L.0.5).
