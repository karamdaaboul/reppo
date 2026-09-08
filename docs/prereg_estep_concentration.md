# Preregistration — E-step concentration sensitivity (WML_noent at eps_e = 0.1)

Committed **before** launch. Append-only.

**Amendment note.** This document supersedes `docs/prereg_budget_matched.md` at commit
`7410271`, which remains in history unaltered. Three things changed and the reasons are
recorded here rather than silently applied:

1. **The study is renamed and reframed.** It was written as a *budget-matched* arm whose
   purpose was to remove a mean-step confound. That framing overstated what setting
   `eps_e = 0.1` does: it does not construct a matched trust region (section 1.2), so the
   arm cannot license a mean-step-versus-covariance attribution. It is registered here as a
   **sensitivity check on E-step concentration**.
2. **The mean-step attribution is removed as the frame.** The superseded P1/P2 pair and the
   three-branch verdict built on "budget matching removes the mean-step confound" are
   withdrawn. The frozen interpretation in section 7 replaces them.
3. **Scope is LEAN: 24 runs.** The 48-job baseline-rerun addition is not registered.

---

## 0. The primary phenomenon this arm is a sensitivity check on

### 0.1 Headline

> Across Walker, G1 and LEAP, and under all three operator/entropy pairings, WML has larger
> median pre-tanh scale than PW on identical states, on both independently sourced halves of
> the common bank. Split-half validation: 18/18 comparisons above 1, every bootstrap CI
> excludes 1, every comparison 8/8 in direction.

This is a **matched-state** statement and is never to be written as an unconditional "WML is
wider." Reported in the same breath, always: on each policy's **own** rollout states the two
arms are near parity — WML has the larger median in **2 of 9 task-by-pair comparisons**, and
every on-policy median across all twelve cells lies between **0.12 and 0.48**.

### 0.2 The split-half gate backing it

Job `3845429`, node r23g0004, 2026-09-08T03:00:18+02:00, commit `bdb0a31`; script
`scripts/analysis/split_half_gate.py` sha256 `c051708813a022a5...53b160ff`; artifacts
`reports/artifacts/split_half_gate.{csv,json}`, committed at `c709b41`.

Each union bank is 1536 PW-sourced + 1536 WML-sourced states. **Both halves are
matched-state populations**: within a half every arm sees exactly the same states, and the
halves differ only in which arm's rollouts sourced them. Ratio convention: median over the
eight seeds of per-seed log ratios, exponentiated. Interval: paired percentile bootstrap
over seeds, 10000 resamples, `np.random.default_rng(20260908)`, 95%.

| task | pair | PW-sourced half | WML-sourced half |
|---|---|---|---|
| walker | `WML_noent / PW_noent` | **29.264** [17.136, 55.992] | **19.379** [9.949, 20.836] |
| walker | `WML_ent / PW_ent` | 4.978 [2.752, 5.934] | 4.235 [3.104, 6.026] |
| walker | `WML_noent / PW_ent` | 17.928 [11.393, 34.973] | 14.615 [7.140, 15.113] |
| g1 | `WML_noent / PW_noent` | **2.004** [1.581, 2.200] | **1.908** [1.694, 2.258] |
| g1 | `WML_ent / PW_ent` | 1.933 [1.632, 2.827] | 2.026 [1.787, 2.968] |
| g1 | `WML_noent / PW_ent` | 1.836 [1.604, 2.200] | 1.794 [1.314, 2.044] |
| leap | `WML_noent / PW_noent` | **3.450** [1.631, 5.479] | **2.061** [1.412, 3.568] |
| leap | `WML_ent / PW_ent` | 4.582 [4.199, 11.416] | 5.681 [3.026, 11.088] |
| leap | `WML_noent / PW_ent` | 8.127 [2.564, 10.226] | 4.767 [2.150, 5.272] |

Every cell is 8/8 on the per-seed sign count. The `WML_noent / PW_ent` figures reproduce
`reports/same_state_width.md` exactly; that report carried no intervals, and every interval
here plus both entropy-matched pairs is new.

### 0.3 Two measured facts that constrain what may be claimed

**Saturation dissociates from width.** Under the definition locked in section 4.3, medians
over seeds:

| task | arm | P(\|a\| > 0.95) | P(\|a\| > 0.99) |
|---|---|---|---|
| walker | PW_noent / WML_noent | 0.570 / **0.790** | 0.413 / **0.715** |
| g1 | PW_noent / WML_noent | **0.385** / 0.351 | **0.248** / 0.237 |
| leap | PW_noent / WML_noent | 0.581 / **0.681** | 0.440 / **0.581** |

On G1 the pathwise arm is the **more saturated** of the two while remaining the **narrower**
one. Saturation and width are not interchangeable readouts and no claim here treats them
as such.

**One scalar width does not describe these policies.** Walker `WML_noent`, existing bank,
no new rollouts: median pre-tanh sigma is **0.218** on each seed's own 192 states and
**9.116** on the other seven seeds' 1344 states, a ratio of **35.3x** (per-seed 18.9 to
184.1, 8/8). Every width figure names its state population.

---

## 1. Setup

### 1.1 The arm

`WML_noent` at `eps_e = 0.1`, `mstep_decoupled = false`. A **one-field change** from the
executed `eps_e = 0.5` `WML_noent` baseline. The config `eps_mu` field is not touched and
the decoupled M-step is not enabled: `eps_mu` is inert unless `mstep_decoupled` is true
(`src/jaxrl/reppo.py:935-941` zeroes `kl_mu`, `kl_sigma`, `beta_mu`, `beta_sigma`
otherwise), and enabling that path would replace the per-state KL gate entirely
(`:1105-1112`).

### 1.2 The one honesty sentence

> eps_e (E-step target budget, batch-averaged, solved by the eta dual) and kl_bound = 0.1
> (per-state hard gate on the realized forward KL) are numerically equal but are different
> objects, and the WML arm carries the gate on top of its E-step, so this is not a
> mathematically matched trust region.

`kl_bound = 0.1` is confirmed from source and from the resolved `.hydra/config.yaml` of
every executed run; the dataclass default at `reppo.py:95` is `1.0` and is overridden by
`config/reppo.yaml`.

### 1.3 `eps_e` does not reach the pathwise path

Its only consumers are `eta_dual_loss` (`reppo.py:231-245`), called at `:1021` and `:1044`,
both inside the `weighted_mle` branch. The pathwise branch sets `eta_loss = 0`
(`:1085-1086`); `loss += eta_loss` is guarded by the arm (`:1158-1159`); and
`with_eta = (actor_update_mode == "weighted_mle")` (`:421`, `:435`) means the `eta`
parameter is never created under pathwise.

**Demonstrated, not only argued.** A 2-seed Walker `PW_noent` rerun at `eps_e = 0.1` on the
KL-split source is bitwise identical to the canonical baselines on actor, critic and
normalizer, against a control rerun at `eps_e = 0.5` on the pre-split source that is also
identical. `PW_noent` s301-308 therefore stand as the pathwise comparison with **no rerun**.

---

## 2. Arm set — LEAN, 24 runs

`WML_noent`, `eps_e = 0.1`, tasks `WalkerRun`, `G1JoystickFlatTerrain`,
`LeapCubeRotateZAxis`, seeds 301-308. `log_faithful_diag = true`, on the KL-split source
(branch `kl-split`, whose split lands at commit `0502b8f`), so each run carries `fr_lag_eff`,
`fr_kl_mean_part_med`, `fr_kl_width_part_med` and `fr_kl_split_resid_max`. Every other
scientific field pinned to that task's executed `eps_e = 0.5` `WML_noent` baseline.

**Reused without rerun.** The existing `PW_noent` s301-308 and `WML_noent` (`eps_e = 0.5`)
s301-308 **final** checkpoints. No baseline reruns are registered.

**Logging is not changed.** No density change, no added forward pass, no extra PRNG draw.
Any future logging-density change requires its own bit-check first, because raising
`num_eval` is not covered by the bit-check that passed here.

**Launch SHA.** The runs do **not** execute at `0502b8f`: gate 2 of section 8 requires the
export-tag suffix to be added first, which creates a later commit. That commit is the launch
SHA, is recorded in every run's ledger row, and must match the verified source SHA on the
workstation.

---

## 3. What is testable about lambda, established before launch

All twelve existing (task, arm) cells log `fr_lag_eff`, 21 evaluation points per run, all
finite. Applying the criterion of section 5 to the existing data:

| cell | median final-window slope | 95% CI over seeds | equilibrated |
|---|---|---|---|
| walker PW_ent | -5.19e-04 | [-1.05e-03, 1.31e-04] | yes |
| walker PW_noent | -2.07e-04 | [-7.22e-04, 1.02e-03] | yes |
| walker WML_noent | 1.95e-03 | [-1.47e-02, 7.10e-03] | yes |
| g1 PW_ent | 2.20e-05 | [-1.19e-05, 4.73e-05] | yes |
| **g1 PW_noent** | -5.71e-05 | [-8.86e-05, -1.56e-05] | **no** |
| g1 WML_noent | -7.46e-03 | [-1.64e-02, 4.62e-03] | yes |
| leap PW_ent | -1.77e-04 | [-4.35e-04, 2.00e-04] | yes |
| leap PW_noent | -1.30e-04 | [-4.05e-04, 5.19e-05] | yes |
| **leap WML_noent** | -6.87e-03 | [-1.16e-02, -3.16e-03] | **no** |

**A rerun cannot repair the two failures.** Training on the KL-split source is bit-identical,
so the lambda trajectories would be reproduced exactly; the failures are genuine residual
drift, not a shortage of logged points.

Consequently the lambda prediction of section 6.2 is registered **for Walker only**. On G1
and LEAP lambda is reported **descriptively**, and its final value is **not** treated as an
equilibrium value.

Recorded for context: lambda separates the arms by one to two orders of magnitude in the
existing data, pathwise 0.004-0.023 against weighted-MLE 0.33-0.39.

---

## 4. Endpoints (locked)

### 4.1 Primary geometry — matched-state

Median pre-tanh `sigma = exp(log_std) + min_std`, over states and action dimensions, on the
**fixed common banks**, hash-verified before use: `walker_fixed_state_bank.npz`
`8adfeb0b...aa21`, `g1_fixed_state_bank.npz` `cf6f7880...35e9`, `leap_fixed_state_bank.npz`
`0053b0f3...b158e`.

**The registered contrast is `WML_noent` at `eps_e = 0.1` against `WML_noent` at
`eps_e = 0.5`**, paired by seed, aggregated as the median over seeds of per-seed log ratios
exponentiated, with a paired percentile bootstrap over the eight seeds, 10000 resamples,
**`np.random.default_rng(20260910)`** — a fresh key, distinct from the gate's 20260908 and
from the frozen 20260902. Mean is reported alongside median. Both bank halves are reported
separately as well as pooled.

**Additionally, descriptively:** the same quantity against the existing `PW_noent`, to record
whether the matched-state operator gap persists under the changed E-step concentration. This
is a description, not a registered test.

### 4.2 Secondary geometry — own-occupancy

Genuine own-rollout banks, one per new (arm, seed), to the spec frozen in
`docs/protocol_onpolicy_banks.md`: 192 states per (arm, seed), same depth stratification,
32 parallel envs, stochastic actions clipped to +-0.999, frozen normalizer, and **fresh
independent keys** derived by sha256 rather than by Python's salted `hash()`.

Reported: own-rollout median and mean sigma; the **common-to-own amplification**, the ratio
of common-bank width to own-rollout width; `sat95`; `sat99`.

**Matched-state and own-occupancy quantities are kept explicitly separate and are never
averaged, compared across, or quoted without their population label.**

### 4.3 Saturation definition

The **exact expectation over the stochastic policy**, not a Monte Carlo count:
`P(|tanh Y| > t) = Phi((-c-mu)/sigma) + 1 - Phi((c-mu)/sigma)` with `c = arctanh(t)` and
`Y ~ N(mu, sigma^2)` the pre-tanh Gaussian, averaged over every state and action dimension
of the population named, at `t = 0.95` and `t = 0.99`, median over the eight seeds. The
deterministic action `tanh(mu)` is not used.

### 4.4 E-step diagnostics

Logged `eta`, `ess` and its percentiles, `w_max`, `q_spread`, `fr_gate_operator`, and the
KL split `fr_kl_mean_part_med` / `fr_kl_width_part_med` with `fr_kl_split_resid_max`.

### 4.5 Return

The frozen statistic `score_window3`, the mean of the final three logged evaluations
(indices 18, 19, 20), frozen at commit `7edb8e8`, with the paired percentile bootstrap and
exact sign test defined there. **Reported, not predicted** (section 6.3).

---

## 5. Lambda-equilibration criterion (locked)

Lambda is **equilibrated** in an arm if, over the final 20% of updates, the per-evaluation
change in lambda is non-monotonic **and** a linear-fit slope has a bootstrap CI including
zero. With 21 evaluation points the window is the final 5; the slope is fitted per seed and
the CI is a percentile bootstrap over the eight seeds.

Precedent for insisting on it: the decoupled `beta_sigma` result, where `beta_sigma` climbed
monotonically from 3.6 to 59.9 and never equilibrated, so its final value was meaningless.

**No lambda gap is reported until this passes in both arms of the comparison.**

---

## 6. Predictions

### 6.1 Direct E-step effect — registered, all three tasks

At `eps_e = 0.1` against `eps_e = 0.5`:

* **`eta` increases**
* **`ESS/M` increases**
* **`w_max` decreases**

Prior evidence, from the offline counterfactual on identical frozen candidates recorded in
`docs/prereg_eps_e_theory_test.md` (Walker, mean over the eight seeds): `eta` 0.958 -> 5.495,
`ESS/M` 0.739 -> 0.909, `w_max` 0.153 -> 0.056. The prediction registered here is that the
same three directions hold in **trained** runs, on all three tasks.

### 6.2 Lambda — registered, WALKER ONLY

At `eps_e = 0.1`, the WML gate multiplier `lambda` **moves toward** the pathwise `lambda`.

This is registered as a statement about **dual behaviour**, not as evidence that the two
trust regions match — section 1.2 says they do not. It is permitted on Walker because both
Walker baselines satisfy the section-5 criterion. On G1 and LEAP `lambda` is reported
descriptively only and its final value is **not** treated as an equilibrium value.

### 6.3 Directions deliberately NOT registered

**No direction is registered for matched-state sigma, own-rollout sigma, return, or
saturation.**

The reason, stated in advance so it cannot be invoked post hoc. Section 3.4 admits two
readings that point in **opposite** directions on width against `eps_e`:

* the per-step displacement pressure `A_pop ~ 1/eta^2` **falls** as `eta` rises, and
* the balance-law stationary width `sigma* ~ eta / sqrt(ESS)` **rises** as `eta` rises.

Both move when `eps_e` moves, and nothing in this design separates them. There is therefore
no clean directional width prediction to register, and **these runs do not test the
displacement-versus-curvature mechanism.**

---

## 7. Interpretation, frozen

* **If `eps_e = 0.1` materially changes matched-state width:** E-step concentration
  moderates WML matched-state geometry.
* **If it barely changes width:** the WML matched-state geometry is robust to this large
  change in E-step concentration.
* **Either way, no claim about the displacement-versus-curvature mechanism follows from
  these runs.**

In every case the width result is stated as **matched-state**, never as an unconditional
"WML is wider," and the on-policy near-parity of section 0.1 is reported alongside it.

---

## 8. Hard gates before any job is submitted

1. **Five unit tests, all passing.**
   **T1** `tests/test_kl_split.py` — the split sums to `gaussian_kl_diag`; both parts vanish
   at `pi_theta == pi_old`; each vanishes when only the other differs; the forward
   orientation is distinguished from the reverse; both parts non-negative. *(Passing at
   `0502b8f`; re-run as a gate.)*
   **T2** the `eta` dual minimiser is larger at `eps_e = 0.1` than at `0.5` on real
   checkpoint Q samples, and `estep_weights` is correspondingly closer to uniform.
   **T3** export-tag test: the tag at `eps_e = 0.1` differs from the baseline tag, **and**
   the tag at `eps_e = 0.5` is byte-identical to the tag already on disk, so no published
   export is orphaned.
   **T4** fixed-batch gradient test: the pathwise arm's actor gradient is identical at
   `eps_e = 0.1` and `0.5` on an identical batch.
   **T5** the split is a training no-op: fixed-batch actor and critic gradients identical
   between `f459db7` and `0502b8f`.

2. **Export-tag collision check.** `eps_e` is **not** in the export tag: the variant
   suffixes are only `_pad{k}`, `_m{M}`, `_rho{r}`, `_noent`, `_ent`
   (`scripts/train_and_export.py`, variant construction). A `WML_noent` run at
   `eps_e = 0.1` would therefore write `<Task>_weighted_mle_s<seed>_final` — **the
   baseline's own path**, the failure mode that already destroyed two Walker checkpoints.
   Before launch an **`_eps01`** suffix is appended, **only when `eps_e` differs from the
   shipped default of 0.5**, following the precedent already set for `estep_num_samples` and
   `sqrt_rho`, so no existing baseline tag is orphaned. Authored on the workstation, covered
   by T3. The canonical baselines are hashed before and after the launch and
   `BASELINES_UNCHANGED` is required.

3. **Per-(arm, seed) partition matching.** Each new run is submitted to the partition its
   corresponding baseline (arm, seed) ran on, read from the run ledger's `partition` field
   (`ledger/runs.d.*/`), not assumed — the G1 entropy factorial in particular was split
   across `c23g` and `c25g` per seed (`slurm/g1_ef_*_c23g.sh`, `slurm/g1_ef_*_c25g.sh`).
   A ledger row with the same fields is written for every new run.

4. **Smoke test** on a tiny config before the real submission.

5. **No gap analysis** until every seed of a task is complete.

---

## 9. What this does not test

It moves `eta` through `eps_e` and observes the E-step diagnostics, `lambda`, width and
saturation. It does not construct a matched trust region (section 1.2). It does not
manipulate the M-step constraint machinery, does not touch `eps_mu` or the decoupled path,
and does not separate the `1/eta^2` route from the ESS route, since `eps_e` moves both. It
makes no causal claim about return.

The tilting mechanism an earlier document registered this arm to test is **not** revived: it
is NOT SUPPORTED under `5edaca3`'s decision rule (`reports/logsigma_decomp_result.md`), and
nothing here depends on it.
