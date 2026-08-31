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

---

## Pre-launch Amendment L.0b — calibration protocol finalization (2026-08-31)

**Append-only.** Everything above this rule is unchanged: the registered v2
text of `f103642a35d11df0ac7b278bab76a4219fa2cb52` plus Amendment L.0 of
`463605fec7e4af8a3526b30c5dabb579d4b15bb0`. sha256 of the leading 349
lines is
`3458a0b4d3f2d9feafd0a6f9985308cb88abb017cf7a4483618d0b9b69ae8dea`.
No calibration or confirmatory run has been launched. **No evaluation
return was inspected in producing this amendment** — neither seed-0
calibration returns nor any frozen-α A/B outcome. Only preregistration
text, environment and reward specifications, Hydra run configuration, and
committed repository configuration were read.

### L.0b.1 The registered seed-901 calibration rule governs

§1 specifies one learned-α calibration run at **seed 901** per new task.
The learned-α calibrations on disk are seed 0. The registered rule stands;
the seed-0 runs do not substitute for it.

- HopperHop seed-0 calibration = **retrospective/pilot calibration**.
- LeapCubeRotateZAxis seed-0 calibration = **retrospective/pilot calibration**.
- G1JoystickFlatTerrain seed-0 calibration = **retrospective/pilot calibration**.

The α values recorded in L.0.2 from those seed-0 trajectories
(0.00035225937608629465 Hopper, 0.0010604216950014234 LEAP,
0.00022178766084834933 G1) remain on the record as provenance and as the
basis for the L.0.2 finding that the existing LEAP cohort deviates from
the α rule. **They are not the prospective frozen-α values** and
supersede nothing in §1.

For the primary prospective experiment, one fresh learned-α calibration at
**seed 901** is run for each of HopperHop, LeapCubeRotateZAxis and
G1JoystickFlatTerrain. After those three runs finish, and before any
A/B seed 101--108 run launches,

    alpha_t = median(alpha_curve[2:])

from that task's own seed-901 calibration is frozen, **identically in arms
A and B**. LEAP, Hopper and G1 normalization anchors and qualification are
likewise derived only from permitted seed-901 calibration information
under the final registered rules, including the corrections in L.0b.2.

### L.0b.2 Ceiling censoring is decoupled from the calibration-derived anchor

**Status of the registered text.** §2 does set
U_t = 1.1 × the best calibration checkpoint IQM for LEAP and G1, and does
diagnose ceiling censoring at final IQM ≥ L_t + 0.90 (U_t − L_t). With
L_t = 0 that threshold is 0.90 × 1.1 = **0.99 × the calibration run's own
best checkpoint**, and a run ending near its own best does not establish
that the environment is near a genuine task ceiling. §2 already carries a
proviso — the ceiling gate is "available only where U_t is independent of
the calibration run" — which forecloses the circular application. This
amendment removes the remaining ambiguity by stating the corrected rule
outright and by resolving the spec question §2 left open, before any
seed-901 outcome exists.

**Corrected rule.**

*DMC tasks (HopperHop, WalkerRun, HumanoidRun).* L_t = 0, U_t = 1000, both
independent of any run of ours. Ceiling qualification may use the
registered task-scale threshold.

*LEAP and G1.* The seed-901-derived U_t may still be used as a **fixed
normalization anchor**, because it is fixed before any confirmatory
outcome exists. It must **not** by itself be used as evidence of
environmental ceiling censoring. A calibration-derived anchor normalizes;
it does not certify a ceiling.

**Spec determination — no independent defensible ceiling exists for either
task.** Read from the environment sources in
`mujoco_playground/_src` at the installed version, with no return field
opened:

- *LeapCubeRotateZAxis* (`manipulation/leap_hand/rotate_z.py`). The single
  active positive reward term is `angvel` at scale 1.0, defined as the
  cube's angular velocity projected on the z-axis, with the source comment
  "Unconditionally maximize angvel in the z-direction" and the class
  docstring "Rotate a cube around the z-axis as fast as possible without
  dropping it." This is a raw angular velocity: **unbounded above by the
  specification**, limited only by actuator and contact physics. The other
  active term is `termination = −100.0` on cube drop. There is no success
  criterion, no normalized reward, and no finite spec-defined maximum
  return.
- *G1JoystickFlatTerrain* (`locomotion/g1/joystick.py`). The active
  positive terms are bounded — `tracking_lin_vel` and `tracking_ang_vel`
  are `exp(−error/σ) ∈ (0, 1]` at scales 1.0 and 0.75, `feet_air_time` at
  2.0 and `feet_phase` at 1.0 are bounded by construction, and `alive` is
  scaled to 0.0 — while every other active term is a non-positive cost.
  A finite analytic supremum is therefore derivable. It is a **loose and
  simultaneously unattainable** bound, requiring perfect velocity tracking,
  perfect gait phase and maximal air time together with exactly zero
  orientation, pose, slip, collision, contact-force, joint-limit and
  stand-still cost. It is not a spec-defined maximum, not a success
  criterion, and not a task ceiling in the sense DMC's 1000 is one.

**Consequence.** For **both** LEAP and G1 the ceiling-censoring gate is
recorded as **unavailable**. Neither task can be declared ceiling-censored.
No ceiling is manufactured from the calibration run. The
calibration-derived normalization anchor is retained for both. This
limitation is reported explicitly wherever LEAP or G1 qualification is
discussed. No part of this determination rests on observed seed-0 or A/B
performance; it rests on the reward specification alone.

The floor-uninformative gate (final IQM ≤ L_t + 0.05 (U_t − L_t)) remains
as registered, but for LEAP and G1 its threshold is likewise
calibration-derived (0.05 × 1.1 = 0.055 × the calibration best) and is
therefore reported as a within-run learnability check, not as an
environment-level statement.

### L.0b.3 Prospective versus mixed cross-task analyses

**Primary per-task prospective comparisons.**

| task | prospective cohort |
|---|---|
| HopperHop | seeds 101--108 |
| LeapCubeRotateZAxis | seeds 101--108 |
| G1JoystickFlatTerrain | seeds 101--108 |
| WalkerRun (fresh-only sensitivity) | seeds 104--108 |

**Retrospective or mixed evidence.**

| source | status |
|---|---|
| HumanoidRun | retrospective |
| WalkerRun pooled n=8 | mixed: outcome-seen 3+3 plus prospective 104--108 |
| Hopper / LEAP / G1 seed-0 pilot A/B cohorts | retrospective only |

The five-task ladder over all qualified task-level gaps is therefore
explicitly labelled a **MIXED PROSPECTIVE/RETROSPECTIVE associational
summary**, not a fully prospective inferential test.

A **fresh-data sensitivity** analysis is preregistered here over the
qualified prospective tasks only: HopperHop, LeapCubeRotateZAxis,
G1JoystickFlatTerrain, and WalkerRun fresh-only. At n ≤ 4 this sensitivity
is descriptive; per L.0.4 no p<0.05 claim is possible or required, and
none is made.

No outcome under either analysis licenses a causal statement that action
dimension drives the gap.

### L.0b.4 Suite-default γ provenance

Verified from committed repository configuration and the recorded Hydra
configuration of each calibration run; no environment or hyperparameter was
modified, and no return field was read.

| task | γ | source |
|---|---|---|
| HopperHop | 0.99 | base default, `config/reppo.yaml` |
| LeapCubeRotateZAxis | 0.99 | base default, `config/reppo.yaml` |
| G1JoystickFlatTerrain | 0.97 | `config/experiment_overrides/mjx_humanoid_large_data.yaml` |

Hopper and LEAP were launched with `experiment_overrides=mjx_dmc_large_data`,
which sets no γ and so inherits the base 0.99. G1 was launched with
`experiment_overrides=mjx_humanoid_large_data`, whose committed first
hyperparameter is `gamma: 0.97`. Both values are present unchanged in
upstream commit `69d04eb` (2026-05-06), months before any work on this
ladder (padding preregistration `d1ab422`, 2026-08-30; ladder
preregistration `f103642`, 2026-08-31). They are group defaults selected by
choosing the override group appropriate to the robot, not values chosen for
this ladder.

Accordingly:

> Cross-task suite-default differences, including gamma, are preserved.
> Within each task, arms A and B use identical gamma. No gamma value is
> selected or changed in response to operator-comparison outcomes.

γ is not harmonized across environments.

**One adjacent manual setting is flagged rather than rationalized.** The
LEAP seed-0 calibration was launched under the `mjx_dmc` env group with
hand-specified critic value support on the command line,
`env.vmin=-10 env.vmax=60`, rather than a group default; Hopper (0/150 from
`config/env/mjx_dmc.yaml`) and G1 (−10/10 from `config/env/mjx_humanoid.yaml`)
both used group defaults. The accompanying LEAP override
`max_episode_steps=500` is not a free choice — it matches the environment's
own `episode_length=500` default in `rotate_z.py`. The value support is a
critic-representation choice and not an operator-comparison hyperparameter,
and it was fixed before any comparison outcome; it is nonetheless a
per-task manual selection and is recorded as one. The exact LEAP value
support used for the prospective seed-901 calibration and for seeds
101--108 is registered in Amendment L.1 before launch, and is identical
across arms A and B.

### L.0b.5 Outstanding before launch — revised

This supersedes the L.0.6 checklist by inserting the seed-901 calibrations
ahead of everything that depends on them.

- [ ] Run three fresh learned-α calibrations at **seed 901**: HopperHop,
      LeapCubeRotateZAxis, G1JoystickFlatTerrain. Nothing below may be
      filled from seed-0 data.
- [ ] Prospective frozen α per task = `median(alpha_curve[2:])` of that
      task's **seed-901** calibration, applied identically to arms A and B.
- [ ] LEAP and G1 normalization anchors from the seed-901 calibration
      (L_t from spec, U_t = 1.1 × best seed-901 checkpoint IQM), used for
      normalization only, never as ceiling evidence (L.0b.2).
- [ ] Qualification per task from permitted seed-901 calibration
      information; ceiling gate recorded **unavailable** for LEAP and G1.
- [ ] LEAP critic value support (`env.vmin`, `env.vmax`) registered
      explicitly (L.0b.4).
- [ ] Benchmark GPU-hours per new task, from one timed full run.
- [ ] Exact launch SHA, plus the machine-precision parity check against the
      pristine snapshot if that SHA differs from `3b96deb`.
- [ ] Exact launch commands.
- [ ] Author's calibration-inspection answers (L.0.5), still
      **TO BE FILLED BY KARAM**.

---

## Amendment L.1 (part 1) — pre-seed-901 launch record (2026-08-31)

**Append-only.** Everything above this rule is unchanged: the registered v2 text
of `f103642a…`, Amendment L.0 of `463605fe…`, and Amendment L.0b of
`ca9c1d8e79e98cbfd92d1ef7eb9e39677452ec6b`. sha256 of the leading 561 lines is
`2148ed7c6b33281ebad97aaddd2987509f3242690bc986743e0c75f9bbf40bf2`.

This is **part 1**, committed *before* the three seed-901 calibrations launch. It
records everything §1/§2 and L.0b.5 require in advance: build, resolved
configuration, parity, and the exact launch commands. Part 2 — the resulting α
values, anchors, qualification and timings — is appended after those runs
finish and before any frozen-α A/B run. **No evaluation return has been
inspected**; every fact below comes from configuration, source, package
metadata, or the filesystem.

### L.1.1 Code governance — parity against the pristine reference

| item | value |
|---|---|
| launch SHA | `ca9c1d8e79e98cbfd92d1ef7eb9e39677452ec6b` |
| audited baseline (§1) | `3b96deb` |
| pristine reference | `69d04eb93dd9415e8f54cbe995b6fb3b5ae883d4` (cvoelcker/reppo) |
| check run | `scripts/verify_estep.py` |
| files covered | `src/jaxrl/reppo.py` vs `tests/reppo_upstream_snapshot.py` |
| sha256 `src/jaxrl/reppo.py` | `92fc2a51393288ea8b6ff98b27880692fc8efa613755b593a8cc0f6aa24af19d` |
| sha256 snapshot | `5614ad94fe083a8c03e0886f27ef01172d9a465c272859530169bb87f6e1c105` |

The launch SHA differs from `3b96deb`, so §1 requires the parity check; it was
run and it **passed**.

- **(a) Pathwise arm bit-identical to the pre-patch implementation — PASS.**
  `max |new − base|` over all actor parameters = `0.000e+00`; over all critic
  parameters = `0.000e+00`; eval return `17.916548` under both modules,
  difference `0.000e+00`. This is exact bit-identity, not a tolerance.
- **(b) Uniform weights collapse the weighted MLE onto the KL estimator — PASS.**
  Identity residual `9.537e-07` (rel `3.44e-08`); `estep_weights` on a flat
  exponent gives `max|w − 1/M| = 0.000e+00` with ESS `32.000` of 32; α→∞ gives
  `max|w − 1/M| = 1.043e-07`. Dual stationarity on a real checkpoint:
  `KL(w‖uniform) = 0.4954` against `eps_e = 0.5`, ESS `17.15` of 32 — **holds**.
- **(c) Decoupled-M-step reduction — DID NOT EXECUTE.** The check aborts with
  `FileNotFoundError` on `exports/WalkerRun_weighted_mle_s0_final`, which is
  absent from disk (only `_s1_`, `_s2_`, `_s99_` exist). This is a missing input
  artifact, not a numerical discrepancy. It is recorded rather than worked
  around, and it is **not estimator-relevant to this ladder**: check (c)
  exercises `mstep_decoupled = true`, and every ladder run — pilot, calibration
  and confirmatory — sets `mstep_decoupled: false`. No ladder result depends on
  the decoupled path.

**Diff against the audited baseline.** `git diff 3b96deb..HEAD` over `src/`,
`config/` and `tests/` touches only four files, all additions:
`config/env/mjlab.yaml`, `config/env/mjlab_liftcube.yaml`,
`src/env_utils/torch_wrappers/mjlab_env.py`, `src/torchrl/envs.py`. The JAX
training path used by this ladder — `src/jaxrl/`, `src/networks/`,
`src/env_utils/jax_wrappers.py`, `src/env_utils/action_pad.py`,
`config/reppo.yaml`, `config/env/mjx_dmc.yaml`, `config/env/mjx_humanoid.yaml`,
`config/experiment_overrides/` — is **byte-identical to `3b96deb`** (empty
diff). The mjlab additions are torch-path files that no ladder run imports.

### L.1.2 Build

| component | version |
|---|---|
| mujoco / mujoco-mjx | 3.10.0 / 3.10.0 |
| playground (MuJoCo Playground) | 0.0.5 |
| jax / jaxlib | 0.5.2 / 0.5.1 |
| flax / optax / distrax | 0.10.6 / 0.2.5 / 0.1.5 |
| GPU 0 | NVIDIA RTX PRO 4500 Blackwell, 32623 MiB |
| GPU 1 | NVIDIA RTX 4000 Ada Generation, 20475 MiB |

### L.1.3 Two configuration facts that change how the record must be read

**(i) The saved Hydra config does not describe the run.** `config/reppo.yaml`
lists `_self_` last in its `defaults`, so the base `hyperparameters:` block wins
at compose time and Hydra writes that pre-merge state to `.hydra/config.yaml`.
The override group is then merged back **on top at runtime**
(`src/jaxrl/reppo.py:1221`, `scripts/train_and_export.py:76`). Effective values
must be obtained by re-applying that merge; they agree with the exported
`meta.json` in every case checked. A consequence worth registering: a CLI
override of any key the override group also sets would be **silently clobbered**
by the group. None of the three launch commands below does this.

**(ii) `eval_interval` is dead config and `num_eval` is not an episode count.**
`reppo.py:960-961` and `1015-1020` recompute
`eval_interval = (total_time_steps / (num_steps · num_envs)) // num_eval`,
ignoring `cfg.eval_interval` (=2). With `total_time_steps = 5e7`,
`num_steps = 128`, `num_envs = 1024`: 381 training iterations, `381 // 20 = 19`,
`381 // 19 + 1 = 21` evaluation checkpoints — matching the 21-entry
`alpha_curve` in all three seed-0 calibrations. Each checkpoint evaluates
`num_envs = 1024` parallel environments for a full `max_episode_steps` rollout
and averages over completed episodes, so **evaluation episodes per checkpoint is
~1024**, not 20.

### L.1.4 Resolved (post-merge) configuration for the seed-901 calibrations

| item | HopperHop | LeapCubeRotateZAxis | G1JoystickFlatTerrain |
|---|---|---|---|
| verified action dim d | 4 | 16 | 29 |
| env type / suite | mjx / DMC | mjx / Playground | mjx / Playground |
| total_time_steps | 50 000 000 | 50 000 000 | 50 000 000 |
| max_episode_steps | 1000 | 500 | 1000 |
| eval checkpoints | 21 | 21 | 21 |
| eval episodes/checkpoint | ~1024 | ~1024 | ~1024 |
| gamma | 0.99 | 0.99 | 0.97 |
| lmbda / lmbda_min | 0.95 / 0.5 | 0.95 / 0.5 | 0.95 / 0.5 |
| critic vmin / vmax / num_bins | 0 / 150 / 151 | −10 / 60 / 151 | −10 / 10 / 151 |
| hl_gauss | true | true | true |
| num_envs × num_steps | 1024 × 128 | 1024 × 128 | 1024 × 128 |
| num_mini_batches / num_epochs | 64 / 8 | 64 / 8 | 16 / 8 |
| lr / optimizer | 3e-4 / Adam | 3e-4 / Adam | 3e-4 / Adam |
| critic_hidden_dim | 512 | 512 | 1024 |
| actor_hidden_dim / layers | 512 / 3 | 512 / 3 | 512 / 3 |
| critic enc/head/pred layers | 2 / 2 / 2 | 2 / 2 / 2 | 2 / 2 / 2 |
| reward_scaling / terminate | 1.0 / false | 1.0 / false | 1.0 / false |
| kl_start / kl_bound / clip mode | 0.01 / 0.1 / clipped | same | same |
| M (`estep_num_samples`) | 32 | 32 | 32 |
| eps_E | 0.5 | 0.5 | 0.5 |
| mstep_decoupled | false | false | false |
| action_pad | 0 | 0 | 0 |
| α init (`ent_start`) | 0.01 | 0.01 | 0.01 |
| α learning rate | actor Adam, 3e-4 | same | same |
| `update_entropy_lagrangian` | **true** (learned α) | **true** | **true** |
| effective actor `min_std` | 0.1 | 0.1 | 0.1 |

Notes on three entries that are easy to misread:

- **No gradient clipping is active.** `anneal_lr = false` selects the bare
  `optax.adam(lr)` branch (`reppo.py:343-356`); the `clip_by_global_norm` chain
  is taken only when `anneal_lr = true`. `max_grad_norm: 0.5` is therefore
  **inert** in every run of this campaign.
- **α has no separate learning rate.** It is the actor network's
  `temperature()` parameter and is trained by the same actor Adam at 3e-4.
- **`actor_min_std` is 0.0 in config but 0.1 in effect** (Amendment A.1),
  re-confirmed here in the exported `actor_kwargs` of all three tasks.

**Estimator switches held at base defaults in all runs:** `reduce_kl: true`,
`reverse_kl: false`, `update_kl_lagrangian: true`,
`actor_kl_clip_mode: "clipped"`, `eps_mu: 0.1`, `eps_sigma: 5e-5`,
`ent_loss_per_dim: false`, `beta_sigma_fixed: null`, `log_q_spread: false`,
`aux_loss_mult: 1.0`, `exploration_noise_min = max = 1.0`,
`exploration_base_envs: 0`, `polyak: 1.0`, `normalize_env: true`.

**Config inheritance chain:**
`config/reppo.yaml` → `defaults: [env, platform, experiment_overrides, _self_]`
→ `_self_` wins at compose time → runtime merge of
`experiment_overrides.hyperparameters` over `hyperparameters` → CLI overrides
that the group does not set. Groups used: `env=mjx_dmc` (Hopper, LEAP),
`env=mjx_humanoid` (G1); `experiment_overrides=mjx_dmc_large_data` (Hopper,
LEAP), `experiment_overrides=mjx_humanoid_large_data` (G1).

### L.1.5 A and B differ in exactly one resolved key

A full flattened-config diff of the pilot A and B runs, per task, yields exactly
one differing key in each case:

| task | differing keys | A | B |
|---|---|---|---|
| HopperHop | 1 | `actor_update_mode = pathwise` | `weighted_mle` |
| LeapCubeRotateZAxis | 1 | `actor_update_mode = pathwise` | `weighted_mle` |
| G1JoystickFlatTerrain | 1 | `actor_update_mode = pathwise` | `weighted_mle` |

The B commands additionally pass `eps_e=0.5`, `estep_num_samples=32` and
`mstep_decoupled=false` explicitly, but those equal the base defaults, so the
resolved configurations are identical apart from the operator. The same
one-key-difference discipline governs the prospective seeds 101--108.

### L.1.6 Critic value support registered for seed 901

§2 requires the value support to be fixed before launch and identical across
arms. Two conflicts surfaced in the audit and are resolved here on provenance
grounds, not on performance.

| task | seed-0 calibration | pilot A/B cohort | **registered for seed 901 and for seeds 101--108** |
|---|---|---|---|
| HopperHop | 0 / 150 | 0 / 150 | **0 / 150** (`mjx_dmc` group default; consistent) |
| LeapCubeRotateZAxis | −10 / 60 | −10 / 60 | **−10 / 60** |
| G1JoystickFlatTerrain | −10 / 10 | **−4 / 4** | **−10 / 10** |

- **LEAP.** −10/60 is the pre-existing task-specific choice, used identically by
  the calibration and both pilot arms and recorded in the exported
  `critic_kwargs`. It is retained unchanged. It is *not* retuned, and no
  calibration or comparison return was consulted in retaining it. For the
  record, `0 / 150` is the `config/env/mjx_dmc.yaml` group default that Hopper
  uses; LEAP has never run at that support, and `vmin = 0` could not represent
  the negative returns that `rotate_z.py`'s `termination = −100.0` term and its
  cost terms permit.
- **G1.** The seed-0 calibration ran at the `config/env/mjx_humanoid.yaml` group
  default −10/10, while the pilot A/B cohort was hand-set to −4/4 on the command
  line *after* that calibration had run. Because −4/4 was set later, it cannot be
  shown not to have been informed by the calibration's observed return scale.
  Seed 901 and the prospective seeds 101--108 therefore use the **group default
  −10/10**, whose provenance is clean by construction. The consequence is
  recorded plainly: the prospective G1 cohort will run at a different value
  support from the retrospective/pilot G1 cohort, so the two are not
  configuration-comparable and the pilot cohort's status as retrospective-only
  (L.0.1) is reinforced.

### L.1.7 Exact launch commands (seed 901, learned α)

Run from the repository root at SHA `ca9c1d8`, with
`CUDA_DEVICE_ORDER=PCI_BUS_ID` and `JAX_PLATFORMS=cuda,cpu`.

GPU 0 — G1JoystickFlatTerrain:

    CUDA_VISIBLE_DEVICES=0 python scripts/train_and_export.py \
      env=mjx_humanoid env.name=G1JoystickFlatTerrain env.asymmetric_obs=false \
      experiment_overrides=mjx_humanoid_large_data \
      num_trials=1 num_seeds=1 wandb.mode=disabled seed=901

GPU 1 — LeapCubeRotateZAxis:

    CUDA_VISIBLE_DEVICES=1 python scripts/train_and_export.py \
      env=mjx_dmc env.name=LeapCubeRotateZAxis env.asymmetric_obs=false \
      experiment_overrides=mjx_dmc_large_data env.vmin=-10 env.vmax=60 \
      env.max_episode_steps=500 hyperparameters.max_episode_steps=500 \
      num_trials=1 num_seeds=1 wandb.mode=disabled seed=901

Whichever GPU frees first — HopperHop:

    CUDA_VISIBLE_DEVICES=<free> python scripts/train_and_export.py \
      env=mjx_dmc env.name=HopperHop \
      experiment_overrides=mjx_dmc_large_data \
      num_trials=1 num_seeds=1 wandb.mode=disabled seed=901

No `hyperparameters.ent_start` or `hyperparameters.update_entropy_lagrangian`
override appears, so α is **learned** from the 0.01 initialization, as §1
requires of a calibration run. Exports land under
`exports/<EnvName>_pathwise_s901_final` (naming per
`scripts/train_and_export.py`), which cannot collide with the seed-0 pilot
namespace. These three runs also serve as the §5 benchmark timing runs; GPU
type, wall-clock and GPU-hours are recorded in part 2.

### L.1.8 Still outstanding after this commit

- [ ] α_t = `median(alpha_curve[2:])` per task, from the **seed-901** run.
- [ ] LEAP and G1 normalization anchors from seed 901 (normalization only; the
      ceiling gate stays **unavailable** per L.0b.2).
- [ ] Hopper qualification against the independent 0--1000 DMC scale.
- [ ] Floor/learnability gate per task, as registered and as narrowed in L.0b.2.
- [ ] Wall-clock and GPU-hours per task.
- [ ] Author's calibration-inspection answers (L.0.5) — **TO BE FILLED BY KARAM**.

---

## Amendment L.1 (part 2) — seed-901 calibration results (2026-08-31)

**Append-only.** sha256 of the leading 804 lines is
`d3cede10b1e7b8cb8c748e18c0cdc6056da85f09c759506aecee736fcac3507c`.
Runs launched at `07af381751cdfd9b5690ba8eaa286cfee3d7fc6e`, whose *code* state is
byte-identical to `ca9c1d8` and to the audited baseline `3b96deb` over the whole
JAX training path (L.1.1). Reading seed-901 evaluation returns is authorised: §2
derives anchors and qualification from calibration information. **No pilot or
A/B comparison outcome has been inspected.**

### L.1.9 Calibration runs and timings

All three completed, `nan_in_eval = False`, 21 checkpoints, learned α confirmed
by a non-constant `alpha_curve`. These are also the §5 benchmark timing runs.

| task | GPU | wall-clock (s) | GPU-hours | in-loop `train_seconds` |
|---|---|---|---|---|
| LeapCubeRotateZAxis | 1 — RTX 4000 Ada | 2933 | 0.815 | 2914.2 |
| G1JoystickFlatTerrain | 0 — RTX PRO 4500 Blackwell | 3134 | 0.871 | 3111.5 |
| HopperHop | 1 — RTX 4000 Ada | 1596 | 0.443 | 1579.4 |

Total 2.13 GPU-hours. Wall-clock exceeds `train_seconds` by the XLA compile and
export overhead.

### L.1.10 Prospective frozen α (the registered mechanical rule)

α_t = `median(alpha_curve[2:])` of that task's **seed-901** calibration, to be
frozen identically in arms A and B for seeds 101--108.

| task | **α₉₀₁ (registered)** | α from the seed-0 pilot calibration | α actually frozen in the pilot cohort |
|---|---|---|---|
| HopperHop | `0.00037288447492755949` | 0.00035225937608629465 | 0.00035 |
| LeapCubeRotateZAxis | `0.000782382907345891` | 0.0010604216950014234 | 0.00094 |
| G1JoystickFlatTerrain | `0.00020752247655764222` | 0.00022178766084834933 | 0.00023 |

These α₉₀₁ values supersede, for all prospective use, the seed-0 values recorded
in L.0.2. Note the single-seed spread the procedure carries: LEAP's α moved
−26.2% between two calibration seeds, against +5.9% (Hopper) and −6.4% (G1).
That dispersion is a property of the registered single-calibration-seed rule and
is recorded, not corrected.

### L.1.11 Qualification

Evaluation-return curves (means over ~1024 episodes per checkpoint, 21
checkpoints):

| task | best checkpoint mean | at index | final checkpoint mean |
|---|---|---|---|
| HopperHop | 163.072235 | 20 (= final) | 163.072235 |
| LeapCubeRotateZAxis | 24.325214 | 18 | 20.357870 |
| G1JoystickFlatTerrain | 32.286522 | 18 | 32.153908 |

- **HopperHop** — the DMC scale is independent of our runs, so both gates apply.
  Floor threshold `0 + 0.05·1000 = 50`; ceiling threshold `0 + 0.90·1000 = 900`.
  Final 163.07 lies strictly between. **Not floor-uninformative, not
  ceiling-censored: HopperHop QUALIFIES.**
- **LeapCubeRotateZAxis** — ceiling gate **unavailable** (L.0b.2). Passes the
  floor gate: see the invariance argument below.
- **G1JoystickFlatTerrain** — ceiling gate **unavailable** (L.0b.2). Passes the
  floor gate.

**The floor verdict is invariant to the undefined L_t.** §2 sets
`L_t` = the spec minimum return, but neither `rotate_z.py` nor `joystick.py`
defines one: LEAP's sole active positive term is raw z-angular-velocity,
unbounded *below* as well as above, alongside `termination = −100.0`; G1's active
cost terms are likewise unbounded below. The floor threshold
`L + 0.05(U − L)` is increasing in `L`, so the most permissive admissible `L` is
the largest one. At `L = 0` the thresholds are 1.34 (LEAP) and 1.78 (G1) against
final returns of 20.36 and 32.15; any `L < 0` lowers the threshold further. Both
tasks therefore clear the floor gate under **every** admissible `L_t`, and the
qualification conclusion does not depend on resolving the ambiguity. The
ambiguity still matters for the *anchor*, and is left open in L.1.13.

### L.1.12 HopperHop is not converged at the budget

Recorded because it bears on how HopperHop functions as the low-dimensional
anchor of the ladder. Its calibration curve is still climbing steeply at the
final checkpoint: 43.47 → 54.83 → 145.38 → 156.57 → 163.07 over the last five.
The registered gates are mechanical and HopperHop qualifies on them, but its
final return is a snapshot of a rising curve, not a converged level, so a
task-level gap measured there is a gap between two incompletely-trained arms.
No rule is changed on this basis; it is recorded so the ladder's low-d end is
not read as a converged comparison.

### L.1.13 The registered IQM anchor is NOT computable — anchors left open

§2 sets `U_t = 1.1 ×` the best checkpoint evaluation **IQM** for LEAP and G1.
That quantity cannot be recovered from the calibration artifacts:

1. The trainer stores only `eval/episode_return` — a **mean** over ~1024
   episodes — and its std. Per-episode returns are never written, at any
   checkpoint.
2. Only three checkpoints per task are exported (`_p25`, `_p50`, `_final`), and
   the best checkpoint is index **18 of 20** for both LEAP and G1 — not among
   them. Its parameters no longer exist.

The gap is material, not cosmetic. Re-evaluating the exported **final**
checkpoints with per-episode returns, under the evaluation policy the trainer
actually uses (`make_policy` → `det_action`, deterministic, **not** a stochastic
sample), gives:

| task | re-eval mean | re-eval **IQM** | IQM/mean | stored mean at final | re-eval mean vs stored |
|---|---|---|---|---|---|
| HopperHop | 156.864532 | 177.926046 | 1.1343 | 163.072235 | −3.8% |
| LeapCubeRotateZAxis | 19.041908 | 24.261469 | 1.2741 | 20.357870 | −6.5% |
| G1JoystickFlatTerrain | 32.199657 | 33.978524 | 1.0552 | 32.153908 | **+0.14%** |

G1 reproduces the stored protocol to 0.14%, which validates the re-evaluation
setup; Hopper and LEAP differ by a few percent through evaluation-key variance
on strongly left-skewed return distributions (Hopper p25 = 0.00, LEAP
p25 = −3.28, G1 min = −19.22). The IQM sits **5.5% to 27.4% above the mean**, by
a task-dependent amount, because the IQM trims exactly that failed-episode tail.

Consequence: substituting the stored mean for the registered IQM would bias
`U_t` downward by a different factor per task. Since `Δ_t` is an affine rescale
by `1/(U−L)` with the same anchor in both arms, **no per-task conclusion is
affected** — the sign of `Δ_t`, whether its CI excludes zero, and probability of
improvement are all anchor-invariant. Only the **cross-task** Spearman of §3,
which compares `Δ_t` magnitudes across differently-scaled tasks, depends on the
anchor.

**Anchors are therefore left OPEN and no `U_t` is registered here.** §2 requires
them only "before any confirmatory outcome is examined", and no confirmatory run
has launched. They must be resolved before the cross-task summary is computed.
The admissible resolutions are: (i) re-run the three calibrations with
per-episode evaluation logging, ~2.1 GPU-hours, giving the anchor exactly as
registered; (ii) register a superseding anchor definition (for example the best
checkpoint **mean**, or the final-checkpoint IQM) with the deviation stated;
(iii) invoke §2's own escape hatch and exclude the affected tasks from
cross-task aggregation, reporting them at task level only — noting that this
would leave fewer than four qualified tasks and, by §3, no cross-task analysis
at all. No option is chosen here.

### L.1.14 M-audit: `n_estep` is coupled to the operator switch

Recorded from source. In the forward-KL branch (`reppo.py:659-663`):

    n_estep = cfg.estep_num_samples if actor_update_mode == "weighted_mle" else 16

**What `n_estep` feeds, by arm.**

- *Pathwise (arm A).* The training estimator does **not** use it. The pathwise
  objective is built from a **single** reparameterised sample
  (`pred_action, log_prob = pi.sample_and_log_prob(seed=key)`, `reppo.py:647`),
  giving `value = Q(s, pred_action)` and
  `objective = log_prob · stopgrad(α) − value`. The 16 samples feed **only** the
  Monte-Carlo KL estimate `kl = mean_i logp_old_i − mean_i logp_theta_i`, plus
  the optional `log_q_spread` and `log_estimator_diag` diagnostics.
- *Weighted-MLE (arm B).* The 32 samples feed **both** the KL estimate **and**
  the E-step estimator itself — `q_i`, the weights `w_i`, the objective
  `−Σ_i w_i logp_theta_i`, `ess`, `w_max`, `q_spread` and the η dual.

**The confound.** `n_estep` is derived from `actor_update_mode`, so the single
config key that separates the arms also changes the KL estimate's sample count
from 16 to 32. Both estimates are unbiased for the same forward KL, so this is a
**variance** difference — arm B's KL estimate has roughly half the Monte-Carlo
variance of arm A's — and `kl` drives the trust-region clip and the KL
Lagrangian. The A/B contrast is therefore *operator plus KL-estimator
resolution*, not the operator alone. L.1.5's "exactly one differing resolved
config key" remains literally true and is not retracted; what is recorded here is
that the one key carries a second behavioural consequence.

**Effect on the seed-901 calibrations: none internally.** All three are
single-arm pathwise runs at `n_estep = 16`, so no within-run confound exists and
α₉₀₁ is unaffected. One cross-arm consequence is recorded: α₉₀₁ is measured under
the 16-sample KL regime and is then frozen into arm B, which runs at 32.

This is recorded as a finding. **No code change is made and no confirmatory run
is launched on it**, because deciding whether to decouple `n_estep` from the
operator switch changes what the ladder measures and is not a change this
amendment is authorised to make.

---

## Amendment L.1 (part 3) — provenance labels, Walker α, coupling direction, rerun protocol (2026-08-31)

**Append-only.** sha256 of the leading 978 lines is
`66b1d44690da1663056f57f9f97c3d10cd6f8cd4e1a583d48e7282acf36d36c5`.
Written **before** any anchor-rerun output is opened. Wording, labels and
protocol only: no experiment, seed, setting or analysis method is changed by any
item in this part.

### L.1.15 Author calibration-inspection answer (the reserved L.0.5 field)

The reserved field is filled, verbatim as given by the author:

> POSSIBLY — treat as inspected. He does not remember checking pilot A/B final
> returns, was not shown them in the current session, but cannot rule out
> exposure in an earlier session.

The cautious labels follow, and are binding:

- The pre-existing pilot cohorts — **HopperHop 3+3, LeapCubeRotateZAxis 5+5,
  G1JoystickFlatTerrain 3+3** — are labelled **retrospective / possibly
  inspected** and remain **excluded from all confirmatory analysis**.
- The §1 provenance entry for HopperHop, LeapCubeRotateZAxis and
  G1JoystickFlatTerrain is superseded by, and must be reported as:

  > "design-stage possibly outcome-exposed (pilot cohorts existed and may have
  > been seen before this prereg); confirmatory seeds 101--108 are prospective
  > with respect to their own outcomes."

This supersedes the bare `prospective` label of the §1 table and the
task-level reading already corrected in L.0.1. It is a strictly more cautious
label than L.0.1's, and it is the one that governs.

### L.1.16 WalkerRun top-up: frozen α, and a blocking cohort-composition finding

**The α value is confirmed.** Every frozen-α WalkerRun run on disk — padded and
unpadded, both arms — was launched with

    hyperparameters.update_entropy_lagrangian=false
    hyperparameters.ent_start=0.01528

so **α_Walker = 0.01528**, and it is identical across the cohort. It is read
from the runs' own resolved Hydra overrides, not from any prose. Source path for
the arm-A member:
`outputs/2026-08-28/04-45-58/.hydra/overrides.yaml`
(export `exports/WalkerRun_pathwise_fa_s2_final`, `meta.json`
`alpha_entropy = 0.015279999934136868`). Seeds 104--108 use this value, **not**
α₉₀₁, in both arms.

**The pooling premise does not hold as registered.** §1 states WalkerRun "keeps
its existing 3 frozen-α seeds per arm". An inventory of every non-padded
WalkerRun export whose overrides carry `update_entropy_lagrangian=false` finds:

| arm | frozen-α seeds on disk | n |
|---|---|---|
| A — `pathwise_fa` | s2 | **1** |
| B — `weighted_mle` | s1, s2, s99 | **3** |

Arm A has **one** frozen-α seed, not three, and the seed identifiers do not
correspond between arms ({2} against {1, 2, 99}). Pooling to n=8 would therefore
give n=6 in arm A and n=8 in arm B, not the registered balanced n=8.

A 5+5 frozen-α WalkerRun cohort at α = 0.01528 does exist — the `pad0` cohort,
`WalkerRun_{pathwise_fa,weighted_mle}_pad0_s0..s4`. With `action_pad = 0` no
`ActionPad` wrapper is constructed at all (`scripts/train_and_export.py:66-69`:
`if k > 0`), so those runs are behaviourally identical to unpadded ones and
differ only in export namespace. **§7 nevertheless excludes "any reuse of
padding-cohort runs in this ladder"**, and that exclusion is not overridden here.

Consequence: the **WalkerRun top-up is held** and is not launched with the rest
of the confirmatory queue. The three prospective tasks are unaffected. Three
admissible resolutions, none chosen here: (i) supersede §7 to admit the `pad0`
cohort as the Walker frozen-α cohort, on the recorded ground that k=0 applies no
wrapper; (ii) run all eight Walker seeds 101--108 fresh in both arms and drop
pooling entirely, reporting Walker as fully prospective; (iii) pool unbalanced
(6 vs 8) and record the departure from the registered n=8.

One unrelated inconsistency is recorded while inventorying: the export
`WalkerRun_pathwise_s99_final` carries `actor_update_mode = "pathwise"` in its
`meta.json` while the `.hydra/overrides.yaml` of the run directory its meta
points at specifies `actor_update_mode=weighted_mle`. That run is not
frozen-α and is outside every cohort used here, so nothing in this ladder
depends on it, but the export is not self-consistent and should not be relied on.

### L.1.17 Direction of the `n_estep` coupling (L.1.14)

The coupling recorded in L.1.14 — that `actor_update_mode` also sets the KL
estimate's sample count, 16 in arm A against 32 in arm B — has a determinate
sign, recorded here.

The Monte-Carlo KL estimate feeds the trust-region clip and the KL Lagrangian.
Both are **convex** in the estimate, so by Jensen a noisier estimate raises the
*expected* constraint pressure at equal true KL. Arm A's 16-sample estimate is
the noisier of the two — roughly twice the Monte-Carlo variance of arm B's 32
samples — so the coupling applies **more** expected constraint pressure to the
pathwise arm. It therefore **handicaps arm A**, and the measured
pathwise-minus-zeroth-order gap is **conservative** with respect to it: any true
pathwise advantage is understated, not manufactured, by this asymmetry.

The coupling is identical in every prior cohort (pilot, padding and
retrospective), because it follows from `reppo.py:659-663`, which is unchanged
across all of them. **No code change is made.** Decoupling `n_estep` from the
operator switch would change what the ladder measures and is out of scope for
this amendment.

### L.1.18 HopperHop budget note — both framings

L.1.12 records that HopperHop's calibration curve is still climbing steeply at
the budget (43.47 → 54.83 → 145.38 → 156.57 → 163.07 over the last five
checkpoints). Both readings are registered, and neither is dropped in favour of
the other:

- **As a limitation.** HopperHop's final return is a snapshot of a rising curve,
  not a converged level, so its task-level gap is a gap between two
  incompletely-trained arms. It is **budget-truncated**, and any statement about
  the low-dimensional end of the ladder inherits that caveat.
- **As a positive expectation.** Corollary `cor:where` places operator
  differences **mid-learning**, not at convergence. A task measured on the rising
  part of its curve is therefore measured where the corollary predicts the
  operator gap is largest, so budget truncation is not purely a defect for this
  comparison.

The two framings are reported together wherever HopperHop's gap is discussed.
No gate, rule or budget is changed.

### L.1.19 Anchor reruns — protocol and agreement tolerance, fixed in advance

L.1.13 records that the registered `U_t = 1.1 ×` best-checkpoint **IQM** is not
recoverable from the original calibration artifacts. Two reruns are executed to
recover it:

**Protocol.** LeapCubeRotateZAxis and G1JoystickFlatTerrain, **seed 901**,
identical configuration and identical launch command to the original executions
recorded in L.1.7, with exactly one addition: `hyperparameters.log_eval_iqm=true`,
which computes the interquartile mean over completed episodes **at every one of
the 21 checkpoints** and stores it alongside the existing mean. The IQM is
computed inside the evaluation function on the same episode population as the
mean, under the same deterministic evaluation policy (`make_policy` →
`det_action`). HopperHop needs no rerun: its best checkpoint is its final one, so
its IQM is already recoverable from the exported final checkpoint.

**Agreement tolerance, fixed here before any rerun output is opened.** The added
logging lives in the evaluation branch and does not feed training, so training
should be unperturbed; XLA fusion may nevertheless differ. Two tiers:

- **Tier 1 — unperturbed.** Every checkpoint agrees to
  `max |Δα| / α ≤ 1e-9` on `alpha_curve` **and**
  `max |Δ eval_mean| / |eval_mean| ≤ 1e-9`. The rerun is then the same
  trajectory and inherits the original's provenance entirely.
- **Tier 2 — perturbed but acceptable.** Not Tier 1, but both
  `|Δα₉₀₁| / α₉₀₁ ≤ 0.10` and
  `|Δ eval_mean| / |eval_mean| ≤ 0.10` **at the final checkpoint**.
  Rationale for 10%: it sits below the seed-to-seed α dispersion already
  measured under the registered rule (−26.2% for LEAP between two calibration
  seeds, L.1.10) and below nothing that would change a qualification verdict,
  whose margins are an order of magnitude wider (L.1.11).
- **Material departure.** Anything outside Tier 2 → **stop and report**. No
  anchor is taken from a rerun that fails this, and no confirmatory conclusion
  is drawn from one.

**Two-execution split, registered explicitly.** The registered **α₉₀₁ values
remain those of the original execution** (L.1.10) and are *not* re-derived from
the reruns, whatever the reruns show — α was fixed before the reruns existed and
re-deriving it afterwards would let a second execution reselect a registered
quantity. The **anchors** `U_t = 1.1 ×` best-checkpoint IQM are taken **from the
reruns**, which are the only executions that record the IQM. Each quantity
therefore has one, and only one, source execution, and both are named here.

### L.1.20 `L_t = 0` registered by supersession, with its caveat

**What §2 actually says is not an "(or 0)" fallback.** §2 reads: "`L_t` = the
spec minimum return. If the spec permits negative returns, `L_t` is taken from
the spec, not assumed 0." For LEAP and G1 no spec minimum exists (L.1.11), so
the registered rule has no value to return and the "not assumed 0" clause points
away from the value now being adopted. `L_t = 0` is therefore recorded here as an
explicit **supersession**, not as an application of the registered text.

**`L_t = 0` for LeapCubeRotateZAxis and G1JoystickFlatTerrain.** It is a
**zero-return reference point**, not a claim about any environment minimum;
neither environment has one.

**Shift-invariance caveat.** `S_t(R) = (R − L_t)/(U_t − L_t)` is not invariant to
the choice of `L_t` when `U_t` is itself measured: shifting `L_t` rescales
`S_t` by `(U_t − L_t)` and so rescales `Δ_t`. Because reward is defined only up
to an additive constant per task, a task whose rewards are offset by a constant
receives a different normalized gap under a fixed `L_t = 0`. Cross-task
comparisons of `Δ_t` magnitude therefore carry a task-dependent scale that
`L_t = 0` fixes by convention rather than measurement. Per-task conclusions are
unaffected: `Δ_t` rescales by the same positive factor in both arms, so its sign,
whether its CI excludes zero, and probability of improvement are all invariant.

**Anchor-free sensitivity, registered.** Per-task **probability of improvement**
is registered as the anchor-free sensitivity analysis for the cross-task
comparison. It is computed per task on raw returns, is invariant to any positive
affine renormalization, and is reported alongside the `S_t`-normalized cross-task
summary wherever that summary is given. If the two disagree in rank pattern, the
anchor-free version is the one that constrains the claim.

## Amendment L.2 — execution venue moved to RWTH CLAIX-2023 (2026-08-31)

Recorded **before** any confirmatory run starts. At the time of writing
`outputs/conf/` is empty and `ledger/runs.jsonl` contains no `namespace:
confirmatory` record, so no seed in 101–108 has been consumed in either arm on
any machine. The whole ladder therefore moves venue at once; nothing is split.

### L.2.1 Hardware

The confirmatory ladder executes on the RWTH CLAIX-2023 ML segment (`c23g`), not
on the workstation registered in L.1.2.

| | L.1.2 (calibration, seed 901) | L.2 (confirmatory, seeds 101–108) |
|---|---|---|
| GPU 0 | NVIDIA RTX PRO 4500 Blackwell, 32623 MiB | — |
| GPU 1 | NVIDIA RTX 4000 Ada Generation, 20475 MiB | — |
| node | single workstation, 2 heterogeneous GPUs | `c23g`: 2× Xeon 8468 (96 c), 512 GB, 4× NVIDIA H100 94 GB |
| allocation | — | one GPU per run (24 cores, 122 GB), one run per array task |

The seed-901 calibration runs, and therefore the frozen α values of L.1.10 and
L.1.16, were produced on the L.1.2 workstation and are **not** re-derived here.
α enters the confirmatory runs as a fixed number, so its provenance is unaffected
by where the confirmatory runs execute.

### L.2.2 The GPU counterbalancing becomes inapplicable

The local driver `scripts/run_confirmatory_ladder.sh` froze an odd/even GPU
assignment (odd seeds: A→GPU 0, B→GPU 1; even seeds swapped) so that each arm
received four seeds on each of the two GPU models. That rule was specified in the
driver, **not in the registered protocol text**, and the driver never executed.

Its purpose was to balance a two-GPU-model confound. `c23g` is homogeneous — every
run lands on an H100 — so the confound the rule balanced **does not exist** on the
new venue, and the rule has nothing to assign. It is recorded here as
inapplicable rather than silently dropped. No arm-level or seed-level assignment
decision remains: the array index determines `(task, arm, seed)` deterministically
(`slurm/ladder_matrix.sh`), and Slurm chooses the physical node.

**Same hardware class within a task.** All eight seeds of both arms of any given
task run on `c23g`. Local and cluster runs are not mixed within a task, in either
arm.

### L.2.3 What is unchanged

Per-task frozen α (L.1.10, L.1.16); seeds 101–108; arms A = `pathwise`,
B = `weighted_mle` with M = 32; ε_E = 0.5; `action_pad = 0`; 50 M environment
steps; `update_entropy_lagrangian=false`; and
`log_estimator_diag = log_eval_iqm = false`, which keeps every confirmatory run
bit-identical to the pristine reference (parity check (a)). The command strings
are byte-identical to those the local driver would have issued — verified by
diffing all 48 — with one exception, below. The `n_estep` coupling to the operator
switch (L.1.14, direction L.1.17) remains intact; no code change accompanies this
amendment.

**The one command difference: `hydra.run.dir`.** Every run is given
`hydra.run.dir=outputs/conf/<task>_<arm>_s<seed>`, as the local driver already did.
This is required, not cosmetic: under a job array, tasks starting within the same
second would otherwise all resolve to Hydra's default `outputs/<date>/<time>/` and
overwrite each other's `metrics.npz`, the collision that motivated copying the
diagnostic curves into `meta.json` — here at 48× the scale. It affects only the
output path; the exported checkpoints are unaffected either way, since
`scripts/train_and_export.py` writes them to `exports/` relative to the repo
root, independently of the Hydra working directory.

### L.2.4 Bit-identity across venues

Results produced on an H100 are **not** expected to be bit-identical to results
produced on the L.1.2 workstation; floating-point reduction order differs across
GPU architectures. The registered parity check (a) is an *arm-versus-arm,
same-hardware* property — that arm A on the study code reproduces the pristine
reference exactly — and it is preserved, because both arms of every task now
execute on the same architecture. No confirmatory comparison in §3 is made across
venues.

### L.2.5 Unaffected open items

The WalkerRun cohort-composition question (L.1.16) and the `U_t` anchors (L.1.13,
being resolved by the reruns of L.1.19) are untouched by this amendment. The
confirmatory runs are anchor-invariant — per-task Δ, its CI, and probability of
improvement do not depend on `U_t`; only the cross-task Spearman summary does — so
the ladder may launch before the anchors are recorded. Walker enters the matrix
only if L.1.16 is resolved in favour of running seeds 101–108 fresh, which changes
the array size from 48 to 64 and nothing else.

---

## Amendment L.1 (part 4) — launch gates (2026-08-31)

**Append-only.** sha256 of the leading 1261 lines is
`28cd735e5ec27f4f96ca82110dc34dc03f2258128d45aa74737fc760804353a0`.
Contains the two corrections that **withdraw** earlier claims of this amendment
(L.1.23, L.1.24), the anchor-rerun results, and the frozen launch configuration.

### L.1.21 Concurrent runs silently shared one Hydra directory

Found while auditing export provenance. Hydra's default run directory is
`outputs/<date>/<HH-MM-SS>` and `hydra.job.chdir: True`, so **two runs launched
in the same second receive the same directory**, chdir into it, and both write
`metrics.npz` there; the later finisher overwrites the earlier.

This happened to the original seed-901 calibrations: G1 and LEAP both launched at
`16:06:54` and produced a **single** directory `outputs/2026-08-31/16-07-00`,
whose `.hydra/overrides.yaml` records the G1 command and whose `metrics.npz`
carries G1's completion time. **LEAP's `metrics.npz` was overwritten.** Four run
directories on disk are claimed by more than one export for this reason,
including `outputs/2026-08-30/12-40-42`, shared by exports of two different
environments (`LeapCubeReorient` and `LeapCubeRotateZAxis`).

**Nothing previously reported is invalidated.** Every quantity used in this
amendment — α₉₀₁, the α curves, the evaluation-return curves, the resolved
architecture and value support — is read from each export's own `meta.json`,
which `scripts/export_ckpt.py` writes into `exports/<tag>/` independently of the
Hydra directory. The corroboration is direct: LEAP's registered
α₉₀₁ = 0.000782382907345891 and G1's = 0.00020752247655764222 were re-read from
preserved copies after the discovery and are unchanged. What was lost is the
per-iteration `metrics.npz` of whichever concurrent run finished first, and the
`hydra_run_dir` pointer of the affected exports.

**Two fixes, applied before any confirmatory launch.**

1. Every launch now passes an explicit unique `hydra.run.dir` —
   `outputs/conf/<task>_<arm>_s<seed>` for confirmatory runs,
   `outputs/manual/rerun901-<task>` for the anchor reruns. Collisions are
   impossible by construction rather than by launch timing.
2. `scripts/train_and_export.py` now persists into `meta.json` the curves that
   previously existed only in `metrics.npz`: the evaluation IQM/q25/q75 and
   episode counts, all eleven `est_*` estimator diagnostics, actor and critic
   gradient norms, entropy, policy σ, and the actor and critic losses. Exports
   are self-sufficient.

The first anchor-rerun attempt was launched before this was found, collided the
same way, and was **terminated at 26 minutes and discarded** rather than allowed
to destroy LEAP's IQM curve. The results in L.1.22 come from the relaunched pair.

**Export namespace for the fresh WalkerRun cohort.** `env.action_pad` is left
**unset** rather than set to `0`. The two are behaviourally identical — `build_env`
computes `k = int(cfg.env.get("action_pad", None) or 0)` and constructs the
`ActionPad` wrapper only `if k > 0` (`scripts/train_and_export.py:66-69`), so no
wrapper exists either way — but an explicit `0` appends a `_pad0` suffix to the
export tag and would place the fresh confirmatory cohort inside the padding
namespace that §7 excludes. Leaving it unset yields
`WalkerRun_{pathwise_fa,weighted_mle}_s<seed>` and keeps the namespaces disjoint.

### L.1.22 Anchor reruns — tolerance outcome and the resulting anchors

Executed per L.1.19: identical configuration and seed, plus
`hyperparameters.log_eval_iqm=true`, each in its own Hydra directory.

| task | GPU | wall-clock (s) | GPU-hours |
|---|---|---|---|
| LeapCubeRotateZAxis | 1 — RTX 4000 Ada | 2984 | 0.829 |
| G1JoystickFlatTerrain | 0 — RTX PRO 4500 Blackwell | 3109 | 0.864 |

Applying the tiers exactly as committed, before which no rerun output was opened:

| task | max `alpha_curve` discrepancy | final eval-mean discrepancy | α₉₀₁ discrepancy | **tier** |
|---|---|---|---|---|
| LeapCubeRotateZAxis | 4.729072e+01 | 4.157190e-02 | 2.758541e-02 | **2** |
| G1JoystickFlatTerrain | 1.939171e-01 | 1.425717e-02 | 7.463896e-02 | **2** |

Both are **Tier 2**, not Tier 1. The added evaluation instrumentation is compiled
into the same `train_eval_step` as the training scan, so it perturbs XLA fusion
and hence float32 rounding, and the trajectories diverge chaotically thereafter.
The large maximum `alpha_curve` discrepancy for LEAP is a *relative* figure taken
where α is near zero in the first checkpoints and is not a Tier-2 criterion; the
Tier-2 criteria are the α₉₀₁ and final-eval-mean columns, both inside 10%.

Per the committed interpretation these executions are **prospectively designated
instrumentation executions used to recover the missing IQM anchor, and are NOT
reproductions**. They are not described as replications anywhere.

**Anchors, from the rerun IQM curves, with `L_t = 0` per L.1.20:**

| task | best ckpt index | best-checkpoint IQM | final-checkpoint IQM | **U_t = 1.1 × best IQM** |
|---|---|---|---|---|
| LeapCubeRotateZAxis | 18 | 29.371067 | 28.794039 | **32.308174** |
| G1JoystickFlatTerrain | 20 (= final) | 34.475914 | 34.475914 | **37.923505** |

HopperHop needs no rerun: it is a DMC task whose anchors are the independent
`L_t = 0`, `U_t = 1000`, and it never uses the IQM rule. WalkerRun is likewise
DMC and keeps `0`/`1000` regardless of its pending calibration.

**The registered α₉₀₁ values remain those of the original executions** —
LEAP `0.000782382907345891`, G1 `0.00020752247655764222` — exactly as committed
in L.1.10 and unchanged by these reruns, per the two-execution split of L.1.19.

**Floor gate re-checked against the new anchors.** Thresholds
`0.95·L + 0.05·U` are 1.615 (LEAP) and 1.896 (G1) against final IQMs of 28.794
and 34.476. **Both qualification verdicts are unchanged**: neither task is
floor-uninformative, and the ceiling gate remains unavailable for both.

### L.1.23 SUPERSESSION — the KL-direction claim of L.1.17 is WITHDRAWN

L.1.17 stated that the 16-versus-32 KL-resolution asymmetry "handicaps arm A" and
that the measured gap is therefore "conservative". **That claim is withdrawn.**
It is not retracted from the record — L.1.17 stands as written and is superseded
here — but it must not be used.

The correct statement:

> The pathwise arm uses a noisier 16-sample Monte-Carlo estimate of KL than the
> weighted-MLE arm's 32-sample estimate. Although the 32-sample estimate has
> lower Monte-Carlo variance for the same KL quantity, KL enters nonlinear
> clipping, trust-region and Lagrangian dynamics. Therefore the sign and
> magnitude of the effect on learned policy return are **not** analytically
> determined.

The error in L.1.17 was to carry a Jensen argument about *expected constraint
pressure* through to a directional conclusion about *learned return*, across the
clipping and dual dynamics that separate them. Convexity of the constraint in the
estimate does not transfer through `actor_kl_clip_mode="clipped"` and a learned
Lagrangian to a signed effect on return.

The historical coupling is therefore: **documented**, **identical in nature
across the historical and registered comparisons**, and **to be assessed
empirically** in the separate development ablation of L.1.30. No direction is
claimed until that ablation completes.

**Paper wording, binding.** The task-level A/B comparison is a comparison of the
**registered pathwise and weighted-MLE implementations**, which differ primarily
in actor operator but also carry the documented KL-resolution nuisance. The
ladder alone is **not** described as a pure causal intervention on the operator.

### L.1.24 CORRECTION — floor-threshold monotonicity wording in L.1.11

L.1.11 contains the clause "the floor threshold `L + 0.05(U − L)` is increasing
in `L`, so the most permissive admissible `L` is the largest one." The premise is
right and the conclusion is **wrong**, and it is corrected here.

    L + 0.05(U − L) = 0.95 L + 0.05 U

is increasing in `L`, so a **larger** `L` gives a **higher** threshold and a
**more stringent** floor check. Relative to a hypothetical negative lower anchor,
choosing `L = 0` therefore makes the floor check **more stringent, not more
permissive**.

LEAP and G1 use `L = 0` as the already-amended zero-return **reference** anchor
(L.1.20), not as a claim that the mathematical minimum return is zero — neither
environment has one.

**The qualification verdicts are unchanged**, and are in fact reached under the
most stringent admissible anchor rather than the most permissive: final IQMs of
28.794 (LEAP) and 34.476 (G1) exceed thresholds of 1.615 and 1.896 by more than
an order of magnitude. No other qualification rule changes.

### L.1.25 WalkerRun — fresh balanced cohort, and a fresh α calibration

The §1 premise that a balanced non-padding Walker 3+3 frozen-α cohort already
existed is false (L.1.16): the historical inventory is arm A seed 2, arm B seeds
1, 2 and 99. Accordingly:

- The historical Walker frozen-α runs are **not pooled** with new runs and remain
  **retrospective supporting evidence**.
- The `pad0` cohort is **not reused**, per §7.
- **No unbalanced top-up is constructed.**
- WalkerRun instead receives a **completely fresh outcome cohort**: arm A seeds
  101--108 and arm B seeds 101--108, `action_pad` unset (L.1.21), historical
  registered KL behaviour unchanged, otherwise identical Walker configuration.

**Walker's frozen α is not 0.01528.** Given the provenance finding of L.1.26,
`0.015279999934136868` is not carried into the fresh cohort. A fresh WalkerRun
**learned-α calibration at seed 901** is run under the identical mechanical rule
α = `median(alpha_curve[2:])`, with `log_eval_iqm=true`. Its value is frozen into
this amendment **before any Walker A/B run launches**. Walker's DMC anchors
remain `L_t = 0`, `U_t = 1000` regardless. Walker A/B is therefore scheduled last
and does not gate the other three tasks.

### L.1.26 α provenance audit (configuration and history only)

Classification requested: A independent learned-α calibration; B mechanically
fixed prior procedure; C manual selection before A/B outcomes; D selection
informed by operator-comparison outcomes; E provenance cannot be established.

**HumanoidRun, α = 0.00329 → class A.** It reproduces exactly:
`median(alpha_curve[2:])` of `HumanoidRun_pathwise_s2` is `0.0032859752`, which is
`0.00329` to three significant figures (ratio 0.9988). This is an independent
learned-α calibration under the same mechanical rule the ladder registers.
Caveat recorded for disclosure: **which** of the nine learned-α Humanoid seeds was
designated as the calibration is not documented, and the nine medians span
0.00255 to 0.00441, so the selection of seed 2 is not itself evidenced.
HumanoidRun remains retrospective regardless; this audit is for paper disclosure.

**WalkerRun, α = 0.01528 → class E (provenance cannot be established).**
`d1ab422:docs/prereg_action_padding.md:11` documents it as "WalkerRun's own
learned-alpha median", which would be class B. It does not reproduce: the only
Walker learned-α export with a usable `alpha_curve`
(`WalkerRun_pathwise_s0_final`) gives `median(alpha_curve[2:]) = 0.01574559`,
**3.05% away** and rounding to 0.0157 rather than 0.0153. The three remaining
Walker learned-α exports carry all-NaN `alpha_curve` and cannot be checked. No
evidence indicates class D — the frozen-α Walker runs date 2026-08-28 and the
documented rationale is a calibration median, not an outcome — but the value
cannot be verified from any surviving artifact.

Per the labelling rule, Walker's α provenance is recorded as **uncertain**, and
no stronger prospective status is claimed for it. This is the ground on which
L.1.25 replaces it with a fresh seed-901 calibration rather than reusing it.

### L.1.27 Export-directory / resolved-override mismatch inventory

Reported only; **nothing was modified**. This is a manuscript-verification item.
In every case the export name and its own `meta.json` agree; the disagreement is
with the `hydra_run_dir` the meta points at, which is the L.1.21 collision
signature.

| export directory (×3: `_final`, `_p25`, `_p50`) | name seed | meta seed | override seed | name mode | override mode |
|---|---|---|---|---|---|
| `HumanoidRun_pathwise_fa_s3_*` | 3 | 3 | **7** | pathwise | pathwise |
| `HumanoidRun_weighted_mle_s0_*` | 0 | 0 | **1** | weighted_mle | weighted_mle |
| `WalkerRun_pathwise_s99_*` | 99 | 99 | 99 | pathwise | **weighted_mle** |

Nine directories, three distinct runs. Additionally, four Hydra run directories
are each claimed by more than one export:
`2026-08-28/08-27-54` (`HumanoidRun_pathwise_fa_s3`, `_s7`),
`2026-08-27/23-51-55` (`HumanoidRun_weighted_mle_s0`, `_s1`),
`2026-08-30/12-40-42` (`LeapCubeReorient_pathwise_s0`,
`LeapCubeRotateZAxis_pathwise_s0` — two different environments), and
`2026-08-31/16-07-00` (the two original seed-901 calibrations).

No cohort used in this ladder depends on any of the mismatched exports. The LEAP
seed-0 value support recorded in L.1.6 is independently corroborated by that
export's own `critic_kwargs` (`vmin = −10.0`, `vmax = 60.0`) and does not rest on
the shared Hydra directory.

### L.1.28 GPU counterbalancing schedule — frozen before launch

The two GPUs are different architectures (GPU 0 RTX PRO 4500 Blackwell, GPU 1
RTX 4000 Ada Generation). The assignment below is fixed before launch and is
**never changed on observed performance**:

- odd seeds 101, 103, 105, 107: arm A → GPU 0, arm B → GPU 1
- even seeds 102, 104, 106, 108: arm A → GPU 1, arm B → GPU 0

Verified: within every task, arm A runs four seeds on each architecture and arm B
runs four seeds on each architecture. Every run records task, arm, seed, GPU
model, code SHA, full command, start time and wall-clock into
`ledger/runs.jsonl`.

### L.1.29 Parity / governance at the launch SHA

Re-run after the instrumentation changes, with `log_eval_iqm = false` and
`log_estimator_diag = false` — the confirmatory configuration.

| item | value |
|---|---|
| launch SHA | recorded in L.1.30 |
| reference SHA | `69d04eb93dd9415e8f54cbe995b6fb3b5ae883d4` |
| files covered | `src/jaxrl/reppo.py` vs `tests/reppo_upstream_snapshot.py` |
| check | `scripts/verify_estep.py` |

- **(a) PASS.** `max |new − base| = 0.000e+00` over all actor parameters and over
  all critic parameters; eval return `17.191229` under both modules, difference
  `0.000e+00`. Bit-identical.
- **(b) PASS.** Identity residual `9.537e-07` (rel `3.44e-08`); flat exponent
  gives `max|w − 1/M| = 0.000e+00`, ESS `32.000` of 32; dual stationarity
  `KL(w‖uniform) = 0.5064` against `eps_e = 0.5`.
- **(c) did not execute** — `FileNotFoundError` on the absent
  `exports/WalkerRun_weighted_mle_s0_final`, **identically before and after** the
  instrumentation changes. A pre-existing missing artifact, not a regression, and
  it exercises `mstep_decoupled = true`, which every ladder run sets false.

The default-off gating therefore preserves the registered numerical parity.

### L.1.30 Launch gate status and the confirmatory queue

| # | gate | status |
|---|---|---|
| 1 | IQM reruns pass the committed tolerance rule | **PASS — Tier 2 both tasks** (L.1.22) |
| 2 | LEAP/G1 anchors computable | **PASS** — U_t 32.308174 / 37.923505 (L.1.22) |
| 3 | parity / governance | **PASS** (L.1.29) |
| 4 | KL-direction supersession appended | **DONE** (L.1.23) |
| 5 | floor-anchor wording correction appended | **DONE** (L.1.24) |
| 6 | Walker fresh 8+8 plan appended | **DONE** (L.1.25) |
| 7 | Walker/Humanoid α provenance recorded | **DONE** (L.1.26) |
| 8 | GPU counterbalancing frozen | **DONE** (L.1.28) |
| 9 | exact commands and launch SHA recorded | **DONE** — below |

**Frozen α per task, for seeds 101--108:**
G1 `0.00020752247655764222`; LEAP `0.000782382907345891`;
Hopper `0.00037288447492755949`; Walker — pending its seed-901 calibration
(L.1.25), frozen here before any Walker A/B run.

**Exact command form** (`scripts/run_confirmatory_ladder.sh`, committed):

    CUDA_DEVICE_ORDER=PCI_BUS_ID JAX_PLATFORMS=cuda,cpu \
    CUDA_VISIBLE_DEVICES=<gpu> python scripts/train_and_export.py \
      hydra.run.dir=outputs/conf/<task>_<arm>_s<seed> \
      <env args> seed=<seed> num_trials=1 num_seeds=1 wandb.mode=disabled \
      hyperparameters.actor_update_mode=<pathwise|weighted_mle> \
      hyperparameters.update_entropy_lagrangian=false \
      hyperparameters.ent_start=<frozen alpha> \
      hyperparameters.log_estimator_diag=false hyperparameters.log_eval_iqm=false

Fixed task priority, set before outcomes and not changed on results:
**G1 → LEAP → Hopper → Walker**. During execution only completion status,
crash/NaN/divergence, registered configuration validity, B-health flags
(ESS/entropy) and hardware/logging integrity are inspected. Task-level A−B
outcomes are analysed only after all registered seeds of that task complete.

**Development ablation (L.1.30a), development namespace, does not alter the
confirmatory protocol.** G1, weighted-MLE, M = 32, comparing KL Monte-Carlo
resolution 32 against 16, development seeds 201, 202, 203 in both conditions, GPU
architectures counterbalanced across the six runs. It quantifies the L.1.14
nuisance empirically and is the only basis on which any direction may later be
claimed (L.1.23). If its effect is negligible relative to the task-level A−B
difference, that is reported as evidence the nuisance is unlikely to explain the
main result; if material, it is flagged as a substantive limitation and the
decision whether to add a clean cohort is reopened explicitly, as a **subsequent
experiment**, and the registered comparison is not silently repaired.

**Claim boundary, binding.** The native task ladder does not establish that
action dimension causally drives the operator gap: task identity and dimension
remain confounded by design. The dimension argument rests on convergence of the
native ladder, the action-padding intervention, the direct estimator and
error-field probes, and any later within-task effective-dimension experiment.
The ladder supplies external-validity, associational evidence only.
