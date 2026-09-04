# Mechanism follow-ups: KL budget scope, gate estimator, and the ESS gap

**Dated 2026-09-04T17:35:41+02:00. Audited source SHA `b48a6ed8ac0b3cce58a4077bfa86272923066567`**
(`src/` identical at `918f82c`). Read-only: no training, no source modification.
Every claim below carries its support. Anything not directly supported by an
artifact in this repository is marked **UNVERIFIED**.

---

## 1. The KL budget is per outer iteration, not per gradient step

| Claim | Support |
|---|---|
| `actor_target` is a **hard copy**, `actor_target.params <- actor.params` | `src/jaxrl/reppo.py:693-697` |
| taken once per `learn_step`, i.e. once per outer iteration | `src/jaxrl/reppo.py:1395-1405` (`collect_rollout` -> `learn_step`) |
| merged once **outside** `update`, so frozen across all inner minibatch updates | `src/jaxrl/reppo.py:698-699`; the four downstream uses `:806, :824, :864, :995` are reads only |
| `polyak` is declared but used nowhere in `src/` | `src/jaxrl/reppo.py:79`; `grep -rn polyak src/` returns that line alone |
| the gate compares against this frozen reference | `src/jaxrl/reppo.py:920` (`kl = old_pi_act_log_prob - pi_act_log_prob`), `:1080-1086` |

**`pi_old` is exact notation**, with the precision that it denotes the
**iteration-start** policy held fixed for every inner update of that iteration --
not a Polyak/EMA target and not the previous minibatch's policy.

### 1.1 Exact width-only budget

`D_KL( N(mu, s^2 I) || N(mu, c^2 s^2 I) ) = (d/2)(c^-2 - 1 + 2 log c)`.
Reproduced by `scripts/analysis/mech_kl_width.py` ->
`reports/artifacts/mech_kl_width.json`.

| | value | support |
|---|---|---|
| `d` for HumanoidRun | **21** | live `mujoco_playground registry.load("HumanoidRun").action_size`; also `docs/prereg_dimension_ladder.md:26` |
| `c_obs = 0.850/0.279` | 3.0466 | **UNVERIFIED** -- transcribed from `sec:entfail`; the underlying logs are not in this repository |
| `KL_Sigma(c_obs)` at d=21 | **14.026** | computed; manuscript states 14.0 |
| max `c` per full 0.1 budget, exact | **1.073185**, `y = 7.32%` | computed |
| manuscript's `y` from `KL ~ d y^2` | 6.90% | small-`y` approximation, reproduced |
| applications needed, exact | **15.77 -> 16** | computed; manuscript states 17 |

### 1.2 The correction

The manuscript's `sec:entfail` states *"Nothing accumulates across steps, so a
per-update constraint offers no protection against sustained drift"* and prices the
threefold widening at *"seventeen gradient steps, against thousands of updates per
evaluation."* **Manuscript text is external to this repository** (Overleaf project
"Main Paper", `blurring_the_critic.tex`) and is therefore unversioned here.

Both parts need correcting:

1. **"Nothing accumulates across steps" is incorrect.** Within an outer iteration
   movement *does* accumulate across the inner updates; it is measured against a
   frozen reference and capped in total. What resets is the reference, once per
   outer iteration.
2. **The denominator is wrong.** The budget is per outer iteration, so the figure
   is ~16 **outer iterations**, not 17 gradient steps.

**The number of outer iterations per evaluation for that experiment is
NOT RECOVERABLE** -- `reports/corrected_replication_code_trace.md:129` marks
HumanoidRun's steps, `num_eval` and overrides unrecoverable, and `:133` records an
exhaustive search finding zero HumanoidRun artifacts. **UNVERIFIED, conditional
only:** if that run used the shipped `mjx_dmc` composition it would be 19 outer
iterations per evaluation and 512 inner updates each.

Corrected statement, suitable for the paper:

> The trust region is imposed against the policy at the start of each outer
> iteration and held fixed for every inner update in that iteration, so a budget of
> 0.1 caps total movement per outer iteration rather than per gradient step.
> Spending the entire budget on isotropic width permits a factor of 1.073 per
> iteration at `d = 21`, so the observed threefold widening requires at least
> sixteen consecutive outer iterations of near-maximal width-only expansion. This
> is a best-case bound: mean movement and anisotropic covariance change also
> consume the budget, so the true requirement is larger.

The manuscript's **conclusion** -- that the trust region fails to contain the
collapse -- survives; its reasoning and its margin do not. The margin is far
tighter than "seventeen against thousands".

---

## 2. KL-gate estimator at N=16 versus N=32

**This is a counterfactual estimator characterisation, not a recovery of the
historical gate-flip rate.** That rate is not identifiable: every logged
diagnostic is shape `(21,1)`, already reduced over states
(`reports/implementation_audit.md` addendum A1.2, commit `e9b9f00`). And **no
adjacent-iteration checkpoint pair exists** -- snapshots sit at iteration 4, 9 and
399 with `checkpoint_frac` 0.25/0.5/1.0
(`exports/WalkerRun_weighted_mle_s301_{p25,p50,final}/meta.json`).

Method: real `(mu, sigma)` from `exports/WalkerRun_weighted_mle_s301_final` on real
states from `reports/artifacts/cd_bank_walker_corrected.npz`; `pi_new` calibrated to
a target exact KL with half the budget in the mean and half in the covariance;
8000 replicates per state, 16 states. Script
`scripts/analysis/mech_gate_probe.py` -> `reports/artifacts/mech_gate_probe.json`.

`P[ 1(sampled >= 0.1) != 1(exact >= 0.1) ]`:

| exact KL | flip N=16 | flip N=32 | ratio |
|---|---|---|---|
| 0.020 | 0.0399 | 0.0057 | **7.03** |
| 0.050 | 0.2610 | 0.1733 | 1.51 |
| 0.080 | 0.4296 | 0.3901 | 1.10 |
| 0.099 | 0.5131 | 0.5061 | 1.01 |
| 0.101 | 0.4774 | 0.4837 | 0.99 |
| 0.120 | 0.4156 | 0.3898 | 1.07 |
| 0.150 | 0.3307 | 0.2769 | 1.19 |
| 0.300 | 0.1189 | 0.0519 | **2.29** |

Below the bound every error is false-closed; above it, false-open.

**Answer: near 0.1 the 16-vs-32 difference is `NEGLIGIBLE` (ratio 1.0-1.1);
overall `SMALL`.** N=16 is 2-7x worse only far from the bound, where absolute
rates are low.

**The larger finding is not the asymmetry.** At the operating point these runs
occupied -- median sampled KL 0.086-0.122
(`reports/artifacts/corrected_runs.csv`; `train/kl` for LEAP) -- the per-state gate
decision disagrees with the exact KL **40-51% of the time at both sample counts**.
Some of that is unavoidable, since any unbiased estimator is ~50% wrong when the
truth sits on the threshold, but the confusion band is wide: the flip rate falls
below ~10% only outside roughly `KL in [0.03, 0.30]`. **The gate controls the step
in aggregate, not per state.**

---

## 3. The training-versus-probe ESS gap is the state bank

Same checkpoint, same `eta = 0.024808276444673538`
(`reports/artifacts/cd_walker_corrected_diagnostics.csv`, field `eta_saved`), same
training generator throughout. Script `scripts/analysis/mech_ess_decomp.py` ->
`reports/artifacts/mech_ess_decomp.json`.

| | ESS/M mean | median | p5 | p95 |
|---|---|---|---|---|
| M=16, pilot bank | 0.2910 | 0.1099 | 0.0625 | 0.9334 |
| M=32, pilot bank | 0.2563 | 0.0625 | 0.0312 | 0.9289 |
| M=32, **episode-spanning** bank | **0.6103** | 0.7201 | 0.0336 | 0.9779 |
| `ESS_training_M32/M` reference | **0.6403** | | | |

Reference from `reports/implementation_audit.md` addendum A1.5 (commit `e9b9f00`).

Per rollout depth, M=32: 0.427 (step 50), 0.485 (150), 0.727 (300), 0.684 (500),
0.653 (700), 0.658 (900), 0.642 (1000).

**Attribution:**

* **State distribution: dominant.** Median 0.0625 -> 0.7201, a factor of 11.5. The
  episode-spanning bank reproduces the training value (0.610 vs 0.640); the pilot's
  burn-in-50 bank does not. Burn-in-50 states are early-episode and atypical.
* **Candidate count: small, and opposite in sign** -- M=16 gives 0.291 against
  M=32's 0.256 in normalised terms.
* **Generation semantics: identical** -- same pre-tanh Gaussian, sigma, tanh,
  `q_scalar`, single critic and eta. Residual: `actor_target` versus the final
  actor, negligible at the final checkpoint.
* **Sampling variability: negligible** -- sd 0.0025 across 20 draws.

The pilot-bank M=16 **median of 0.1099 reproduces the previously reported
`ESS_Qphi_pilot_K16` of 0.099**, confirming the probe was computed correctly and
that the number is a property of the bank, not of the E-step. The pilot bank is
also strongly bimodal (p5 0.031, p95 0.929), so its median is a poor summary.

**Consequences.** Probe ESS numbers must not enter the manuscript. The three ESS
quantities stay separate (`ESS_training_M32`, `ESS_Qphi_pilot_K16`,
`ESS_MC_oracle_K16`) and only the first is representative. The MC-oracle pilot's
64-state bank is early-episode, which is a limitation of **everything** measured on
it, including its `r_RMS` and centred-error-power estimates.

---

## 4. Severity

No new S3 or S4. Section 1 corrects a manuscript mechanism claim (S1, and it
tightens rather than removes the conclusion). Section 2 leaves the historical rate
`UNRESOLVED` and adds a counterfactual `SMALL`. Section 3 removes a claim rather
than adding one.
