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
