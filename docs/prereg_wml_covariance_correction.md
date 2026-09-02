# Pre-registration: M32 + fixed covariance correction (HumanoidRun, d=21)

**Status.** Committed before any run of this arm is launched.

**Label: PROSPECTIVE, NOT BLIND.** The M-sweep results
(`docs/prereg_m_sweep_dmc.md`, Track C) and the full audit
(`reports/m_sample_count_audit.md`) were seen before this document was written. The
decision rule below is registered in advance of the arm; it is not registered in advance
of the evidence that motivated it.

**Arm name, used everywhere.** *M32 + fixed covariance correction.* Not "bias removed"
and not "bias restored". The intervention moves the weighted-MLE fit toward a
**probe-measured population second moment**. It does not establish that the residual
bias is zero, and no wording in this document or in any log, tag or commit message
should imply that it does.

**Companions.** `docs/prereg_m_sweep_dmc.md` (Track C, the M sweep),
`docs/prereg_m_star.md` (Tracks A/B), `reports/m_sample_count_audit.md` (the audit).

---

## 1. What this tests

The audit found that the M-step's weighted-MLE covariance fit is biased low by
approximately `1 - 1/ESS`, that the eta dual pins `ESS/M ~= exp(-eps_E)` so ESS scales
with M, and that at `M = 32` the resulting policy-width contraction is what produces the
published arm's return. The claim under test:

> **If the M=32 arm's return depends on the finite-sample covariance shrinkage, then
> removing that shrinkage at M=32 -- changing nothing else -- should reproduce the
> M=512 collapse.**

This is the intervention the audit named as its single next experiment.

---

## 2. The intervention

In the weighted-MLE objective, on the **retained pre-tanh** samples `y_j`, with the
E-step weights `w_j` unchanged and already detached:

    mu_w    = sum_j w_j * y_j
    y_tilde = mu_w + sqrt_rho * (y_j - mu_w)

and the fit is taken on `y_tilde` instead of `y_j`. The weighted mean is preserved
exactly; the weighted second moment scales by `rho = sqrt_rho^2`.

**Nothing else changes.** `estep_num_samples` stays 32, `eps_e` 0.5, `kl_bound` 0.1,
`ent_start` 0.00329 frozen with `update_entropy_lagrangian=false`, `mstep_decoupled`
false, `lr` 3e-4, `num_envs` 1024, `num_steps` 128, `num_mini_batches` 64, `num_epochs`
8, horizon 52,297,728 steps. The critic still sees the **original** actions, so the
E-step weights, `eta`, the dual and the KL constraint are all untouched.

### 2.1 rho, and its provenance

    rho = 1.0848      sqrt_rho = 1.0416

**Provenance, stated plainly: this constant was measured POST HOC.** It was computed
from the frozen probe *after* the M-sweep collapse had been observed, and then fixed
before this arm was launched. It is **not** a prospectively derived quantity, and the
arm must never be described as if it were.

`rho` is the `solved_on_prefix` second-moment ratio from the audit's Step 5 frozen
nested-prefix probe -- the weighted second moment per dimension at `M = 2048` divided by
the value at `M = 32`, in whitened units where `pi_old` has variance 1.0:

| checkpoint | trained M | return | 2nd moment M=32 | M=2048 | ratio |
|---|---|---|---|---|---|
| `s2` | 32 | 585.0 | 0.8605 | 0.9335 | **1.0848** |
| `s202` | 128 | 520.5 | 0.8505 | 0.9286 | 1.0918 |
| `s211` | 512 | 10.6 | 0.8455 | 0.9125 | 1.0793 |

Mean 1.0853, spread +-0.6% across a 37x range of critic scale (`Q ~= 44.19`, `41.0`,
`1.18`). The `s2` value is used because `s2` is a member of the control arm.

`solved_on_prefix` is the correct dual mode: it re-solves `eta` on each prefix, which is
what actually happens when M changes. The `eta_ckpt_fixed` mode of the same probe gives
1.0506 (`sqrt_rho` 1.0250) and is **not** used, because it holds `eta` at a value the
dual would not have chosen.

**The theoretical ratio is NOT used.** `(1 - 1/ESS_pop)/(1 - 1/ESS_32)` evaluates to
1.0635 at the probe's ESS of 16.75, which under-corrects the measured 1.0848 by 2%. The
measured value is used precisely because the theoretical one does not account for the
low-ESS tail (Sec. 2.2).

### 2.2 Why fixed, and not online

`1/(1 - 1/ESS)` is **convex** in `1/ESS`, so the state-integrated bias is dominated by
the low-ESS tail rather than by the mean. That tail is heavy in this arm: across the
control's 8 seeds and all 21 evals the ESS **p5** runs 1.1-4.4 while the mean ESS runs
16.5-23.3. A correction computed from a mean ESS would systematically under-correct.

The frozen probe already integrates that tail: it measures the realised second-moment
ratio across 1024 states, tail included, rather than evaluating a convex function at an
average. That is the reason to prefer the measured constant.

An **online** `rho_t = 1/(1 - 1/ESS_t)` is deliberately **not** used here. It would make
the intervention a function of the policy it is measuring, coupling the correction to the
very trajectory under test and making a null result uninterpretable. It is named in
Sec. 4 as the follow-up if this arm under-corrects.

**Registered limitation.** `rho` is fixed while the control's own `1 - 1/ESS` moves over
training, from 0.9301 to 0.9603 (a 3.2% spread, against a correction of 8.5%). The spread
is tightest over evals 9-20 (1.0%) and widest over evals 0-8, which is exactly the window
in which the control's width contraction occurs (`sigma` falls 25% below its running max
by eval 1-4 in 7 of 8 control seeds). **A fixed `rho` is therefore least accurate in the
phase that matters most.** This is registered as a known weakness of the design, not
discovered afterwards.

### 2.3 Implementation, and one declared deviation

Pre-tanh samples are **retained**, never recovered by `arctanh`. The E-step clips actions
to `+-(1 - 1e-4)`, whose `arctanh` ceiling is 4.9517, while `|y|` reaches 9.32 in
practice and ~9% of states carry at least one clamped coordinate in an M=32 cloud; an
`arctanh` round trip would have given this arm a different clamp rate than its control.

The draw is decomposed exactly as distrax's `Transformed._sample_n_and_log_prob` does it
and is **verified bit-identical** to the call it replaces, on both actions and log-probs,
0 of 43008 floats differing (`scripts/msweep_audit/bitcheck.py`). A hand-rolled
`scale*normal + loc` draw was rejected: it reproduces the samples exactly but 19902 of
43008 log-prob floats differ.

`sqrt_rho == 1.0` is an exact no-op -- a Python-level branch emitting no op -- and a full
training run at `sqrt_rho=1.0` is bit-identical to the pre-change module on all 66 metric
keys.

**Declared deviation.** The intervention branch evaluates the fit log-prob by *forward*
composition (`base.log_prob(y) - fldj(y)`), while the control uses the shipped *inverse*
path (`pi.log_prob(a)`, which inverts the bijector internally). The two agree to ~1e-6
relative on the 99.66% of samples that are unclamped, and differ by up to 2.95 nats on
the 0.34% that are clamped -- which is exactly where the inverse path returns the wrong
value. The arm therefore differs from its control in two ways, not one: the rescale, and
a log-prob evaluation that is correct on clamped samples where the control's is not. The
second difference is ~1e-6 relative on 99.66% of the data and is registered here rather
than left to be found later.

---

## 3. Decision rule (committed, read in this order)

### Stage 1 -- mechanism gate

**Does the intervention move what it targets?** Read `sigma_final`, the arm mean of
`pi_sigma_mean` at the final eval.

> **`sigma_final >= 0.40` PASSES.**

**This is a FLOOR, not a window. A higher value is success, not overshoot.** The
corrected M=32 fit shrinks the policy width by 3.38% per fit
(`sqrt(0.8605 * 1.0848) = 0.9662`), which is *less* than M=512's 3.67%
(`sqrt(0.9279) = 0.9633`). The predicted `sigma_final` is therefore **at or above**
M=512's 0.552, against the control's 0.235.

Failure branches, none of which yield a return verdict:

- **`sigma_final < 0.40` AND KL gate fire rate `<= 0.48`** -> **UNDER-CORRECTED.** The
  intervention did not move what it targeted and the trust region did not stop it. No
  return verdict is entered. Next test: online `rho_t = 1/(1 - 1/ESS_t)` (Sec. 2.2).
- **`sigma_final < 0.40` AND KL gate fire rate clearly above 0.48** -> **GATE-LIMITED.**
  The trust region absorbed the intervention. **This is NOT a refutation** of the
  covariance channel. No return verdict is entered. 0.48 is the control's own rate
  (0.4796 +- 0.0390, `reports/m_sample_count_audit.md` Sec. 7).
- **`sigma_final > 2.0`, or `sigma_max` climbing without equilibrating** -> **OVERSHOT.**
  No return verdict is entered.

### Stage 2 -- return

**Read ONLY if Stage 1 passes.** Arm mean `final_eval_return`, n=5, against the
unchanged n=8 M=32 control (mean 666.174, sd 61.333, range 585.0-786.2).

| outcome | verdict |
|---|---|
| return < 400 | **covariance channel CONFIRMED** |
| return > 585 | **REFUTED**; next test is the mean-displacement channel |
| 400 <= return <= 585 | **INCONCLUSIVE** |

585 is the control's minimum, so "REFUTED" means the corrected arm lands inside the
control's observed range.

---

## 4. Design, power and limitations

- **n = 5**, seeds **221-225**, full 52,297,728 steps each, ~1.23 h/seed, ~6.1 h
  sequential on one GPU.
- **UNPAIRED against the n=8 control.** Seeds 221-225 come from the 201+ EXPLORATORY /
  ALGORITHM DEVELOPMENT namespace (`ledger/README.md:13`) and **cannot be seed-matched to
  the control's s0-s8**, which live in a retrospective namespace. The comparison is
  between two independent seed draws, so seed-level variance is not differenced out. With
  n=5 against n=8 and a control sd of 61.3, this arm can resolve the ~270-point move
  Stage 2 asks about and cannot resolve anything near the control's own spread.
- **Permanently exploratory.** Per `ledger/README.md:13` this arm can never become
  confirmatory evidence, and neither can its control.
- **The control is not re-run.** It is the published cohort, trained on an older tree.
  The audit (Sec. 1) records that no M=32 control exists on the current tree, so a
  tree-level regression cannot be excluded by this arm's data alone.
- **Not tested here:** the mean-displacement channel (65% of the M=32 step energy is
  sampling noise), any other task, any other `eps_E`, and the pathwise arm.

---

## 5. Logged per eval

Enabled by `hyperparameters.log_cov_diag=true` (default off, since adding ops perturbs
XLA fusion and breaks bit-identity for confirmatory runs):

| quantity | key |
|---|---|
| mean ESS | `train/ess` |
| median ESS | `train/ess_median` |
| ESS p5 | `train/ess_p5` |
| sigma per coordinate | `train/pi_sigma_percoord` (d = 21) |
| fraction of coordinates at `min_std = 0.1` | `train/frac_sigma_at_min` |
| E-step clamp rate | `train/estep_clamp_rate` |
| rescaled-sample clamp rate | `train/tilde_clamp_rate` |
| KL gate fire rate | `train/kl_gate_fire` |
| KL Lagrangian multiplier | `train/lagrangian` |

`min_std` is an **additive** floor (`std = exp(log_std) + min_std`), so sigma approaches
0.1 from above and never reaches it; `frac_sigma_at_min` uses a 1%-above-floor test.

---

## 6. Provenance

Export tag: `HumanoidRun_weighted_mle_rho1.0416_s{seed}`. The `_rho` suffix is appended
only when `sqrt_rho != 1.0`, and an `_m{M}` suffix only when `estep_num_samples != 32`,
so every tag already on disk stays byte-stable. This closes the overwrite hazard
registered at `docs/prereg_m_sweep_dmc.md` Sec. 3.1.

Each run records `git_commit`, `git_dirty` and `git_diff_sha256` into `meta.json`, and
the launcher writes the commit hash and the full working diff into the run directory.
This closes the gap noted in `reports/m_sample_count_audit.md` Sec. 1, where the Track C
runs recorded no commit at all. Note for the record: `docs/prereg_m_sweep_dmc.md` Sec. 7
states that a `git_sha` would be recorded in `meta.json`; no such field exists in those
exports. That document is not modified here.

The working tree is expected to be dirty at launch (unrelated in-progress work under
`scripts/lqr_crossover/`). The diff is captured by hash in `meta.json` and in full in the
run directory, so the code that ran is recoverable.

---

**Append-only.** Everything above this rule is the registered text as committed before
launch. Amendments are appended below, dated, never edited in place.
