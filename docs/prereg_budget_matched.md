# Preregistration — budget-matched WML_noent at eps_e = 0.1

Committed **before** launch. Append-only. Written on the footing confirmed by the
split-half gate computed before this document: job `3845429`, node r23g0004,
2026-09-08T03:00:18+02:00, at commit `bdb0a31`, script
`scripts/analysis/split_half_gate.py` sha256 `c051708813a022a5...53b160ff`, artifacts
`reports/artifacts/split_half_gate.{csv,json}`. The gate numbers are reproduced inline in
section 0.1 so this document does not depend on an external file to be checkable.

## 0. Headline framing this arm serves

The primary phenomenon is the **matched-state** width divergence: on a fixed union bank,
where every arm is evaluated on exactly the same states, the weighted-MLE arm's pre-tanh
scale exceeds the pathwise arm's. The near-parity seen when each policy is evaluated on its
own visited states is reported alongside it, never in place of it.

**No result from this arm, in any branch of the verdict, is to be written as an
unconditional "WML is wider."** Every width statement carries the population qualifier.

### 0.1 The gate this rests on

Each union bank is 1536 PW-sourced + 1536 WML-sourced states. **Both halves are
matched-state populations**: within a half every arm sees exactly the same states. The
paired ratio is the median over the eight seeds of per-seed log ratios, exponentiated;
the interval is a paired percentile bootstrap over seeds, 10000 resamples,
`np.random.default_rng(20260908)`, 95%.

| task | pair | PW-sourced half | WML-sourced half |
|---|---|---|---|
| walker | `WML_noent / PW_noent` | **29.264** [17.136, 55.992] | **19.379** [9.949, 20.836] |
| walker | `WML_ent / PW_ent` | 4.978 [2.752, 5.934] | 4.235 [3.104, 6.026] |
| walker | baseline `WML_noent / PW_ent` | 17.928 [11.393, 34.973] | 14.615 [7.140, 15.113] |
| g1 | `WML_noent / PW_noent` | **2.004** [1.581, 2.200] | **1.908** [1.694, 2.258] |
| g1 | `WML_ent / PW_ent` | 1.933 [1.632, 2.827] | 2.026 [1.787, 2.968] |
| g1 | baseline | 1.836 [1.604, 2.200] | 1.794 [1.314, 2.044] |
| leap | `WML_noent / PW_noent` | **3.450** [1.631, 5.479] | **2.061** [1.412, 3.568] |
| leap | `WML_ent / PW_ent` | 4.582 [4.199, 11.416] | 5.681 [3.026, 11.088] |
| leap | baseline | 8.127 [2.564, 10.226] | 4.767 [2.150, 5.272] |

**All 18 half-task cells are above 1, every interval excludes 1, and every cell is 8/8 on
the per-seed sign count.** The gate holds for all three pairs, so the matched-state effect
is not an artifact of which arm's rollouts sourced the states.

The baseline-pair figures reproduce `reports/same_state_width.md` exactly (Walker
17.928/14.615, G1 1.836/1.794, LEAP 8.127/4.767); that report carried no intervals, and
every interval here, plus both entropy-matched pairs, is new.

**Measured saturation on the same union banks**, under the definition locked in section 4,
medians over seeds:

| task | arm | P(\|a\| > 0.95) | P(\|a\| > 0.99) |
|---|---|---|---|
| walker | PW_noent / WML_noent | 0.570 / **0.790** | 0.413 / **0.715** |
| g1 | PW_noent / WML_noent | **0.385** / 0.351 | **0.248** / 0.237 |
| leap | PW_noent / WML_noent | 0.581 / **0.681** | 0.440 / **0.581** |

Recorded because it is a **dissociation**: on G1 the pathwise arm is the *more* saturated of
the two while remaining the narrower one. Saturation and width are not interchangeable
readouts, and no claim in this preregistration treats them as such.

**Sibling-seed diagnostic** (Walker `WML_noent`, existing bank, no new rollouts): each
seed's own 192 states give a median pre-tanh sigma of 0.218, the other seven seeds' 1344
states give 9.116, a ratio of **35.3x** (per-seed 18.9 to 184.1, 8/8). A single scalar width
does not describe these policies; the state population must always be named.

## 1. Setup

**Matched arm.** `WML_noent` at `eps_e = 0.1`, `mstep_decoupled = false`. This is a
**one-field change** from the executed `eps_e = 0.5` `WML_noent` baseline. The config
`eps_mu` field is not touched and the decoupled M-step is not enabled; `eps_mu` is inert
unless `mstep_decoupled` is true (`src/jaxrl/reppo.py:935-941` zeroes `kl_mu`, `kl_sigma`,
`beta_mu`, `beta_sigma` otherwise), and enabling that path would replace the per-state KL
gate entirely (`:1105-1112`), which would not be a budget match but a different M-step.

**Honesty sentence, recorded verbatim:**

> eps_e (the E-step target budget, solved by the eta dual against the batch mean) and
> kl_bound = 0.1 (the per-state hard gate on the realised forward KL) are numerically equal
> but are different objects, so "budget-matched" means the numbers coincide, not that the
> two constraints are the same quantity.

`kl_bound = 0.1` is confirmed from source and from the resolved `.hydra/config.yaml` of
every executed run; note the dataclass default at `reppo.py:95` is `1.0` and is overridden
by `config/reppo.yaml`.

**`eps_e` does not reach the pathwise path.** Its only consumers are `eta_dual_loss`
(`reppo.py:231-245`) called at `:1021` and `:1044`, both inside the `weighted_mle` branch;
the pathwise branch sets `eta_loss = 0` (`:1085-1086`), `loss += eta_loss` is guarded by the
arm (`:1158-1159`), and `with_eta = (actor_update_mode == "weighted_mle")` (`:421`, `:435`)
means the `eta` parameter is never created under pathwise. This was additionally
**demonstrated**, not only argued: a 2-seed Walker `PW_noent` rerun at `eps_e = 0.1` on the
KL-split source is bitwise identical to the canonical baselines on actor, critic and
normalizer, against a control rerun at `eps_e = 0.5` on the pre-split source that is also
identical. `PW_noent` s301-308 therefore stand as the pathwise comparison with no rerun.

## 2. Arm set

**Core (24 jobs).** `WML_noent`, `eps_e = 0.1`, tasks `WalkerRun`, `G1JoystickFlatTerrain`,
`LeapCubeRotateZAxis`, seeds 301-308. `log_faithful_diag = true`, on the KL-split source
(branch `kl-split`, whose split lands at commit `0502b8f`), so each run carries
`fr_lag_eff`, `fr_kl_mean_part_med`, `fr_kl_width_part_med` and `fr_kl_split_resid_max`.
Every other scientific field pinned to that task's executed `eps_e = 0.5` `WML_noent`
baseline.

**Launch SHA.** The runs do **not** execute at `0502b8f`: gate 2 of section 8 requires the
export-tag suffix to be added first, which creates a later commit. The launch SHA is that
commit, it is recorded in the ledger row of every run, and it must match the verified source
SHA on the workstation. `0502b8f` is named here only as the commit at which the split and
its tests landed.

**Reused without rerun.** The width verdict uses the existing `PW_noent` s301-308 and
`WML_noent` (`eps_e = 0.5`) s301-308 **final** checkpoints, on the frozen union banks.

**Baseline logged reruns (48 jobs, gated addition, explicitly stated).** `PW_noent` and
`WML_noent` at `eps_e = 0.5`, all three tasks, seeds 301-308, on the KL-split source, in a
dedicated git worktree so `exports/` is a real directory and the canonical tree is
unreachable. Their **metrics only** are retained; their checkpoints are bit-identical to the
canonical ones by the bit-check above and are never copied into the canonical tree. This
addition exists for one reason: **no existing run carries the KL split**, and verdict branch
(c) names the KL width part as the candidate residual asymmetry, so branch (c) is untestable
without it. A minimal subset of 16 jobs (the two cells named in section 3) tests less.

## 3. What Step-2 recon establishes about P1's testability, recorded before launch

All twelve existing (task, arm) cells log `fr_lag_eff`, 21 evaluation points per run, all
finite. The equilibration criterion of section 5 is therefore **computable on existing
data**, and applying it gives:

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

**Consequence, recorded now so it is not discovered after the fact.** Rerunning a baseline
on the KL-split source reproduces its training bit-identically, so a rerun **cannot** change
these outcomes: the two failures are genuine residual drift in lambda, not a shortage of
logged points. Densifying the trajectory would require raising `num_eval`, which is not
covered by the passed bit-check and would need its own.

Therefore, at the current run length:

* **P1 is testable on Walker.** Both baselines equilibrate.
* **P1 is not testable on G1** (`PW_noent` still drifting) **or on LEAP** (`WML_noent` still
  drifting). Per section 5 no lambda gap is reported for those tasks. The verdict in
  section 7 is applied **per task**, and on G1 and LEAP it enters branch (c) by default
  unless a longer run is separately registered.

Also recorded: lambda separates the arms by one to two orders of magnitude in the existing
data, pathwise 0.004-0.023 against weighted-MLE 0.33-0.39. That separation is what P1
predicts budget matching should reduce.

## 4. Width measurement (locked)

**Primary, matched-state.** All arms' final checkpoints evaluated on the **same** frozen
union banks, hash-verified before use:
`walker_fixed_state_bank.npz` `8adfeb0b...aa21`, `g1_fixed_state_bank.npz` `cf6f7880...35e9`,
`leap_fixed_state_bank.npz` `0053b0f3...b158e`. Pre-tanh `sigma = exp(log_std) + min_std`,
**mean and median** over states and action dimensions, paired WML/PW per seed, aggregated as
the median over seeds of per-seed log ratios exponentiated, with a paired percentile
bootstrap over seeds. Both bank halves reported separately as well as pooled.

**Secondary, on-policy.** One fresh rollout bank per new (arm, seed), to the spec frozen in
`docs/protocol_onpolicy_banks.md`: 192 states per (arm, seed), same depth stratification,
32 parallel envs, stochastic actions clipped to +-0.999, frozen normalizer, and
**fresh independent keys** derived by sha256 rather than by Python's salted `hash()`. The
same statistics are reported.

**Saturation, neutral bank, both arms.** Definition, fixed here: the **exact expectation
over the stochastic policy**, not a Monte Carlo count --
`P(|tanh Y| > t) = Phi((-c-mu)/sigma) + 1 - Phi((c-mu)/sigma)` with `c = arctanh(t)` and
`Y ~ N(mu, sigma^2)` the pre-tanh Gaussian, averaged over every state and action dimension
of the full 3072-state union bank for that task, reported at `t = 0.95` and `t = 0.99`, with
the median taken over the eight seeds. The deterministic action `tanh(mu)` is **not** used.
Because the states are the union bank, this figure inherits the matched-state qualifier.

## 5. Lambda-equilibration criterion (locked)

Lambda is **equilibrated** in an arm if, over the final 20% of updates, the per-evaluation
change in lambda is non-monotonic **and** a linear-fit slope has a bootstrap CI including
zero. With 21 evaluation points the window is the final 5; the slope is fitted per seed and
the CI is a percentile bootstrap over the eight seeds.

Precedent for insisting on this: the decoupled `beta_sigma` result, where `beta_sigma`
climbed monotonically from 3.6 to 59.9 and never equilibrated, so its final value was
meaningless.

**No lambda gap is reported until this passes in both arms of the comparison.**

## 6. Predictions

**P1 (mean-step equalization).** At `eps_e = kl_bound = 0.1` the equilibrated lambda of
`WML_noent` converges toward the equilibrated lambda of `PW_noent`, with the WML-minus-PW
gap substantially smaller than at `eps_e = 0.5`, at a gate-fire rate not materially
different between arms.

**P2 (covariance mechanism).** The matched-state union-bank width ratio `WML_noent` over
`PW_noent` survives budget matching, remaining above 1 with a paired bootstrap CI
excluding 1.

## 7. Verdict

Applied verbatim after all eight seeds per (task, arm) complete and lambda equilibration is
confirmed in both arms.

**(a) P1 and P2 both hold:** budget matching removes the mean-step confound and the
surviving matched-state width gap is attributed to the covariance (M-step scale) mechanism
of Section 3.4. Report as matched-state, with the on-policy caveat.

**(b) P1 holds, P2 fails:** the matched-state width gap was driven by the budget/mean-step
mismatch; Section 3.4 is demoted to a per-step observation and the width result is reported
without the covariance-mechanism attribution.

**(c) P1 fails:** budget matching alone does not make the arms comparable; a further
asymmetry remains (candidate: the width-part of the gated KL that WML pays and PW does not,
per the KL split). No mean/covariance attribution is entered until that asymmetry is
characterized.

In every branch the width result is stated as matched-state and never as unconditional
"WML is wider," and the on-policy near-parity (WML wider in 2 of 9 arm-seed comparisons, all
medians in 0.12 to 0.48) is reported alongside.

## 8. Hard gates before any job is submitted

1. **Five unit tests, all passing.**
   T1 `tests/test_kl_split.py` -- the split sums to `gaussian_kl_diag`, both parts vanish at
   `pi_theta == pi_old`, each vanishes when only the other differs, the forward orientation
   is distinguished from the reverse, both parts non-negative. *(Already passing at
   `0502b8f`; re-run as a gate.)*
   T2 the eta dual minimiser is larger at `eps_e = 0.1` than at `0.5` on real checkpoint Q
   samples, and `estep_weights` is correspondingly closer to uniform.
   T3 export-tag test: the tag at `eps_e = 0.1` differs from the baseline tag, **and** the
   tag at `eps_e = 0.5` is byte-identical to the tag already on disk, so no published export
   is orphaned.
   T4 fixed-batch gradient test: the pathwise arm's actor gradient is identical at
   `eps_e = 0.1` and `0.5` on an identical batch.
   T5 the split is a training no-op: fixed-batch actor and critic gradients identical
   between `f459db7` and `0502b8f`.

2. **Export-tag collision check.** `eps_e` is **not** in the export tag: the variant
   suffixes are only `_pad{k}`, `_m{M}`, `_rho{r}`, `_noent`, `_ent`
   (`scripts/train_and_export.py`, variant construction). A `WML_noent` run at
   `eps_e = 0.1` would therefore write `<Task>_weighted_mle_s<seed>_final` -- **the
   baseline's own path**. Before launch, an `_eps{value:g}` suffix is added, appended only
   when `eps_e` differs from the shipped default of 0.5, following the precedent already set
   for `estep_num_samples` and `sqrt_rho`. Authored on the workstation, covered by T3, and
   the canonical baselines are hashed before and after the launch and required unchanged.
   The tag is **not** `eps01`; job `3824112`, which used worktree isolation instead, is
   cancelled with zero exports written.

3. **Per-(arm, seed) partition matching.** Each new run is submitted to the partition its
   corresponding baseline (arm, seed) ran on, read from the run ledger's `partition` field
   (`ledger/runs.d.*/`), not assumed. The `g1` entropy factorial in particular was split
   across `c23g` and `c25g` per seed. A ledger row is written for every new run with the
   same fields.

4. **Smoke test** on a tiny config before the real submission, and **no gap analysis** until
   every seed of a task is complete.

## 9. What this does not test

It moves `eta` through `eps_e` and observes lambda, width and saturation. It does not
manipulate the M-step constraint machinery, does not touch `eps_mu` or the decoupled path,
and does not separate the `1/eta^2` route from the ESS route, since `eps_e` moves both. It
makes no causal claim about return. Return is reported under the frozen statistic
(`score_window3`, commit `7edb8e8`) and is **not** predicted in either direction.

The tilting mechanism this arm was once registered to test is **not** revived: it is
NOT SUPPORTED under `5edaca3`'s rule (`reports/logsigma_decomp_result.md`), and nothing here
depends on it.
