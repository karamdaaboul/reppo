# Audit: why weighted-MLE return collapses as `estep_num_samples` (M) increases

Read-only diagnostic audit. **No training was run, no hyperparameter was changed, no
replacement run was launched.** Every frozen measurement below was taken on CPU
(`JAX_PLATFORMS=cpu`) so the two in-flight jobs were not disturbed.

**Environment correction.** The audit request names *WalkerRun*. Every run in this sweep is
**HumanoidRun** (`env.name=HumanoidRun`, `action_dim=21`, `obs_dim=67`). The four means in the
request reproduce exactly against the HumanoidRun exports (738.61 / 666.17 / 137.16 / 10.98),
so this is a naming slip, not a different dataset. The docstring at
`scripts/train_and_export.py:14` uses `env.name=WalkerRun` as its usage example, which is the
likely source. **All findings below are HumanoidRun (d=21) and must be filed as such.**

---

## Verdict

```
1. STATE-BUDGET CONFOUND:                 NO
2. OPTIMIZER/TRAINING-BUDGET CONFOUND:    NO
3. WML NORMALISATION BUG:                 NO
4. EFFECTIVE GREEDINESS INCREASES WITH M: NO
5. COVARIANCE/SATURATION WORSENS WITH M:  YES   (as failure to CONTRACT, not explosion)
6. KL-GATE SUPPRESSION WORSENS WITH M:    NO
7. CRITIC INSTABILITY WORSENS WITH M:     NO
8. EVIDENCE THAT epsilon_E MUST BE RETUNED: INSUFFICIENT
9. CURRENT M-SWEEP INTERPRETATION:        VALID OPERATOR RESULT
```

Scope on (9): the sweep is not confounded and is not an implementation failure. It is a valid
result about the **finite-M estimator**, and the mechanism is identified and quantified below.
It has not yet been isolated by an intervention, and the M=512 arm is at n=2, below the
`docs/prereg_m_sweep_dmc.md` §3.1 floor of n=5, so that arm carries no adjudication of its own.

---

## Step 1 — Provenance and exact configuration

Effective config was recomputed by reproducing `reppo.py:1370`
(`cfg.hyperparameters = OmegaConf.merge(cfg.hyperparameters, cfg.experiment_overrides.hyperparameters)`).
**This matters: Hydra writes `.hydra/config.yaml` BEFORE that merge**, so the saved file shows
`num_mini_batches: 128, num_epochs: 4` while the run actually used `64` and `8` from
`mjx_dmc_large_data`. Command-line arguments and saved configs both mislead here; only the
post-merge value is real.

| M | states/update | minibatch states | grad steps/iter | iters | env steps | actor updates | critic updates | critic evals / actor update | GPU | completed | config differences |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 32 (baseline, n=8) | 131,072 | 2,048 | 512 | 399 | 52,297,728 | 204,288 | 204,288 | 65,536 | not recorded | 8/8 | — |
| 128 (n=5) | 131,072 | 2,048 | 512 | 399 | 52,297,728 | 204,288 | 204,288 | 262,144 | GPU1 RTX 4000 Ada | 5 done, 1 NaN, 2 pending | `estep_num_samples`; `kl_num_samples` key present (=None) |
| 512 (n=2) | 131,072 | 2,048 | 512 | 399 | 52,297,728 | 204,288 | 204,288 | 1,048,576 | GPU0 RTX PRO 4500 | 2 done, 1 NaN, 5 pending | as above |

Identical across all three arms: `num_envs=1024`, `num_steps=128`,
`num_mini_batches=64`, `num_epochs=8`, `total_time_steps=50e6` (52,297,728 executed),
`lr=3e-4` Adam with `max_grad_norm=0.5`, no gradient accumulation, `polyak=1.0`,
`gamma=0.99`, `lmbda=0.95`, `num_bins=151`, `eps_e=0.5`, `kl_bound=0.1`,
`ent_start=0.00329` with `update_entropy_lagrangian=false`,
`actor_kl_clip_mode=clipped`, `update_kl_lagrangian=true`, `mstep_decoupled=false`,
`actor_min_std=0.0` in config but the network's own default `min_std=0.1` is what the
parameterisation applies (`jax_models.py:394`) — identical in every arm either way.

`mini_batch_size = (num_steps * num_envs) // num_mini_batches` (`reppo.py:1065`) — **no term
in M**. Actor and critic are updated in the same minibatch body (`reppo.py:1028-1042`), so both
receive 399 x 8 x 64 = 204,288 optimizer steps in every arm.

> **DID INCREASING M CHANGE THE NUMBER OF STATES USED FOR AN UPDATE?**
> **NO.** 2,048 states per minibatch and 131,072 states per iteration, byte-identical across M.
>
> **DID INCREASING M CHANGE THE NUMBER OF OPTIMIZER UPDATES OR TRAINING HORIZON?**
> **NO.** 204,288 actor and 204,288 critic updates, 399 iterations, 52,297,728 environment
> steps, in every arm. Every completed run reached the full horizon.

**The sweep is therefore NOT flagged `SAMPLE-COUNT SWEEP CONFOUNDED BY TRAINING-BUDGET CHANGE`.**

Two genuine provenance gaps, neither able to produce the effect:

* **No git commit is recorded in any run directory or `meta.json`.** All sweep runs were
  launched from a dirty working tree (uncommitted `src/jaxrl/reppo.py`, `config/reppo.yaml`).
  The only functional delta is the `kl_num_samples` knob, `None` in every run, whose `None`
  branch emits no slice op. The M=32 baseline predates that key entirely.
* **Arm is perfectly confounded with GPU**: M=128 ran only on GPU1 (Ada), M=512 only on GPU0
  (Blackwell), and the baseline's device is unrecorded. This cannot carry the result — the two
  sweep arms sit on *different* architectures and both collapse, and every frozen measurement
  in Steps 2/5 reproduces the mechanism on CPU.

---

## Step 2 — Is the WML computation normalised in M?

Traced at `reppo.py:719-786`. `w_i = softmax(q_i / eta, axis=0)` over the sample axis only
(`estep_weights`, `reppo.py:180`); `objective = -sum_i w_i * logp_theta_i` (`reppo.py:774`);
`ESS = 1/sum_i w_i^2` (`reppo.py:227`); the dual is
`eta*eps_e + mean_j[eta*log mean_i exp((q-qmax)/eta) + qmax]` (`reppo.py:194-197`) — **mean over
the sample axis, correctly M-normalised**.

Verified numerically on frozen states/critic/policy (1,024 states, one common action cloud,
nested prefixes):

| quantity | M=32 | M=128 | M=512 | M=2048 | expected |
|---|---|---|---|---|---|
| `max_state abs(sum_i w_i - 1)` | 7.8e-16 | 1.7e-15 | 3.1e-15 | 5.8e-15 | machine eps |
| `objective` = `-sum_i w_i logp` | -28.261 | -28.218 | -28.185 | -28.176 | O(1), no M term |
| `KL(w \|\| uniform)` at solved eta | 0.5000 | 0.5000 | 0.5000 | 0.5000 | = eps_E |
| `ESS/M` | 0.523 | 0.524 | 0.529 | 0.535 | ~ exp(-eps_E) |

No quantity that should stay O(1) grows with M. Softmax is over axis 0 only; no double sum; no
axis change at M=128/512; float32 forward with float64 host reduction, no overflow (the softmax
subtracts the per-state max, and the dual pulls `qmax` out). One truncation path exists
(`kl_num_samples`, `reppo.py:736-739`) and is `None` in every run audited.

> **WML NORMALISATION BUG: NO.**

---

## Step 3 — Resource / memory confound

> **Hypothesis: "M=512 fits by using fewer states per actor update."**
> **FALSE.** `mini_batch_size = (128 * 1024) // 64 = 2048` states at every M, from a formula
> with no M term (`reppo.py:1065`). Environment transitions represented in every actor update:
> **2,048**, identical in all three arms.

* Peak GPU memory was never logged. Measured live on the in-flight jobs: M=512 holds 8,824 MiB
  of a 32,623 MiB card; M=128 holds 2,552 MiB of 20,475 MiB. Both far from the limit, consistent
  with the prereg §4.1 smoke test finding no OOM at M=512.
* **No OOM handler, retry path, or batch-size autoscaling exists anywhere in
  `reppo.py` or `train_and_export.py`** (grep for oom/retry/memory/donate/autoscale returns
  nothing). There is no code path by which memory pressure could silently change the
  computation.
* XLA recompiles per M because the sample axis changes shape; the **state axis does not**.
* No walltime limit; all completed runs reached iteration 399. Terminations: two NaN aborts
  (`SystemExit(2)`), M=128 seed 206 and M=512 seed 213, both at step 39,845,888.
* Critic evaluations per actor update = `2048 * M` = 65,536 / 262,144 / 1,048,576. **This is a
  critic-query budget, not a state budget; the state budget is identical.**

---

## Step 4 — E-step geometry as M increases (from training logs)

Per-seed means over evals 15-19, aggregated per arm (n=8 / 5 / 2):

| quantity | M=32 | M=128 | M=512 |
|---|---|---|---|
| mean ESS | 19.3 +/- 0.5 | 90.5 +/- 8.5 | 348.2 +/- 11.2 |
| median ESS (`ess_median`) | 21.1 | 100.5 | 384.8 |
| mean ESS/M | 0.603 +/- 0.016 | 0.707 +/- 0.066 | 0.680 +/- 0.022 |
| mean max weight | 0.1675 +/- 0.0040 | 0.0932 +/- 0.0097 | 0.0539 +/- 0.0003 |
| mean logit spread | 2.685 +/- 0.589 | 3.424 +/- 1.037 | 1.921 +/- 0.030 |
| eta | 0.0243 | 0.0133 | 0.0077 |
| fraction of states with ESS < 4 | 0.086 | 0.074 | 0.050 |

**ESS grows proportionally with M** (19.3 -> 90.5 -> 348.2 is 1.00x / 1.17x / 1.13x of the
M=32 ratio). ESS/M does not collapse; it rises slightly. The `ESS < 4` tail shrinks.

> **Effective selection does NOT become more extreme as M increases.**
> The claim `MORE CANDIDATES MAKE THE E-STEP EFFECTIVELY GREEDIER` is **not supported**.

One honest caveat: `w_max * M` rises 5.36 -> 11.93 -> 27.59. At fixed `KL(w||uniform)` the
*bulk* concentration is M-invariant but the single largest weight is an extreme-value statistic
of a larger draw, so it grows relative to 1/M. That is expected behaviour of the maximum, not a
change in the weight distribution's bulk.

**ESS against the action dimension is the number that matters, and it is not a diagnostic anyone
was logging:** d = 21, so ESS/d = **0.92 / 4.31 / 16.7** on the mean and
**1.01 / 4.79 / 18.3** on the median. At M=32 the E-step delivers about as many effective
samples as the policy has dimensions to fit per state -- and fewer, on the mean.

---

## Step 5 — Extreme-value / critic-selection test (frozen, common random numbers)

One common cloud of 2,048 actions per state, 1,024 frozen states, nested prefixes 32/128/512/2048,
the eta dual re-solved on each prefix by golden section over the network's own `[1e-4, 10]`
clip range. Repeated on three checkpoints spanning the outcome range.

Checkpoint `s2` (trained M=32, return 585):

| quantity | M=32 | M=128 | M=512 | M=2048 |
|---|---|---|---|---|
| eta* | 0.00882 | 0.00957 | 0.00989 | 0.01007 |
| ESS | 16.8 | 67.1 | 270.8 | 1095.5 |
| ESS/M | 0.523 | 0.524 | 0.529 | 0.535 |
| KL(w\|\|uniform) | **0.500** | **0.500** | **0.500** | **0.500** |
| max Q | 44.216 | 44.222 | 44.226 | 44.228 |
| mean Q | 44.189 | 44.189 | 44.189 | 44.189 |
| std Q | 0.0177 | 0.0191 | 0.0189 | 0.0191 |
| Q max - median | 0.0248 | 0.0301 | 0.0336 | 0.0362 |
| weighted Q | 44.208 | 44.210 | 44.210 | 44.211 |
| Q gain over mean | 0.0190 | 0.0208 | 0.0212 | 0.0214 |
| **\|\|d\|\| (whitened mean displacement)** | **1.418** | **1.017** | **0.871** | **0.821** |
| **weighted 2nd moment / dim** | **0.8605** | **0.9105** | **0.9279** | **0.9335** |

> **Does increasing M primarily expose more extreme critic-valued actions? NO.**
> Max Q moves 44.216 -> 44.228 across a 64x increase in M — **0.6 of one Q standard deviation**.
> Weighted Q is flat to 3 decimal places. The E-step's Q gain rises 13% over 64x M.
>
> **Does the eta dual compensate enough to keep the E-step geometry stable? YES — exactly.**
> `KL(w||uniform)` is 0.500 at every M on every checkpoint, to three decimals.

**Hypothesis E is refuted.** The dual absorbs the extreme-value growth completely.

**But two M-step quantities change systematically, and this is the finding of the audit:**

`||d||`, the whitened mean displacement the M-step fits toward, falls 1.418 -> 0.821 and is
fully explained as estimator noise on top of a fixed population signal:

```
||d_M||^2  ~=  ||d_inf||^2 + d/ESS        (d = 21)
  M=32    measured 1.4176   predicted 1.3886   err +2.1%   noise share 65%
  M=128   measured 1.0169   predicted 0.9938   err +2.3%   noise share 32%
  M=512   measured 0.8707   predicted 0.8673   err +0.4%   noise share 10%
  M=2048  measured 0.8214   predicted 0.8329   err -1.4%   noise share  3%
```

**At M=32, 65% of the E-step's step energy is sampling noise, not signal.**

The weighted second moment per dimension (in whitened units where pi_old = 1.0) rises
0.8605 -> 0.9335, and matches the textbook weighted-MLE variance bias `1 - 1/ESS`:

| M | ESS | measured 2nd moment | `1 - 1/ESS` x population | implied sigma_new/sigma_old |
|---|---|---|---|---|
| 32 | 16.8 | 0.8605 | 0.8777 | **0.9276** (7.2% shrink per fit) |
| 128 | 67.1 | 0.9105 | 0.9195 | 0.9542 (4.6%) |
| 512 | 270.8 | 0.9279 | 0.9299 | 0.9633 (3.7%) |
| 2048 | 1095.5 | 0.9335 | 0.9327 | 0.9662 (3.4%, population) |

Both effects reproduce on all three checkpoints, across a 37x range of critic scale
(`s2` Q~44.19, `s202` Q~41.0, `s211` Q~1.18) and both outcome regimes:

| checkpoint (trained M, return) | \|\|d\|\| M=32 -> 2048 | 2nd moment M=32 -> 2048 |
|---|---|---|
| s2 (M=32, 585) | 1.418 -> 0.821 | 0.8605 -> 0.9335 |
| s202 (M=128, 520) | 1.400 -> 0.773 | 0.8505 -> 0.9286 |
| s211 (M=512, 10.6) | 1.384 -> 0.818 | 0.8455 -> 0.9125 |

This is a property of the estimator, not of any particular policy or critic state.

---

## Step 6 — Policy covariance / saturation (trajectories, 21 evals, 0 -> 52.3M steps)

Per-arm mean `pi_sigma_mean`:

```
M=32   0.800 0.626 0.599 0.606 0.600 0.647 0.651 0.638 0.660 0.664 0.591
       0.493 0.416 0.359 0.317 0.296 0.281 0.274 0.257 0.251 0.235
M=128  0.849 0.633 0.616 0.658 0.665 0.596 0.623 0.625 0.625 0.603 0.603
       0.571 0.607 0.552 0.519 0.558 0.584 0.545 0.503 0.494 0.447
M=512  0.811 0.633 0.661 0.694 0.623 0.664 0.574 0.682 0.667 0.549 0.628
       0.546 0.630 0.645 0.583 0.616 0.629 0.606 0.573 0.586 0.552
```

Entropy: -29.1 / -5.9 / -1.4 final. Saturation proxy `|action|`: 0.597 / 0.479 / 0.512 —
**saturation does not increase with M; it is highest in the arm that works.** `sigma_max`
5.97 / 5.38 / 4.31 — no width explosion at any M.

> **Does increasing M systematically increase policy width? YES.**
> Final sigma 0.235 -> 0.447 -> 0.552, monotone in M.
>
> **Is the M=128/512 collapse the same covariance pathology as M=32, only worse? NO — it is the
> inverse.** M=32 contracts hard and wins; M=512 never contracts and never commits to a gait.

**Temporal ordering (descriptive only, no causal claim).** `sigma25` = first eval where sigma
falls 25% below its own running max; `ret25` = first eval reaching 25% of that seed's own max
return:

```
M32   s0 sig 3/ret 8   s1 3/8   s2 4/9   s3 10/8   s5 1/8   s6 2/6   s7 1/9   s8 1/6
M128  s201 1/2   s202 1/14   s203 2/5   s204 5/5   s205 1/3
M512  s211 1/0   s212 5/0
```

In 7 of 8 M=32 seeds the width contraction **precedes** the return rise (s3 is the exception).
Within the M=128 arm, the only seed that learned (s202, return 520) has the **lowest** final
sigma of its arm (0.381 vs 0.437-0.498). The ordering is consistent with contraction leading
performance; it is not proof of it.

---

## Step 7 — KL gate interaction

The elementwise KL was recovered **exactly**, not inferred from the logged mean.
`lagrangian_loss_i = -lambda * (kl_i - kl_bound)` (`reppo.py:893-895`) is logged per state
(shape `(21, 1, 2048)`), so `kl_i = kl_bound - lagrangian_loss_i / lambda`, and since
`lambda > 0` the gate fires iff `lagrangian_loss_i <= 0` — a sign test, free of any scale
assumption.

| quantity | M=32 | M=128 | M=512 |
|---|---|---|---|
| **gate-fired fraction** | **0.4796 +/- 0.0390** | **0.4736 +/- 0.0786** | **0.3976 +/- 0.1157** |
| lambda_KL | 0.795 +/- 0.089 | 0.210 +/- 0.124 | 0.317 +/- 0.013 |
| kl_i mean | 0.1021 | 0.1010 | 0.0977 |
| kl_i median | 0.0989 | 0.0987 | 0.0963 |
| **kl_i SD across states** | **0.0412** | **0.0320** | **0.0167** |
| kl_i p05 / p95 | 0.064 / 0.143 | 0.058 / 0.149 | 0.075 / 0.124 |
| fraction kl_i < 0 | 0.000 | 0.000 | 0.000 |

1. **Does larger M cause the gate to fire more often? NO.** 0.480 / 0.474 / 0.398 — slightly
   *less* often at M=512.
2. **Does larger M increase the covariance share of KL? NOT RECOVERABLE.** `kl_mu` and
   `kl_sigma` are hard-zeroed on the `mstep_decoupled=false` path (`reppo.py:747-752`) and log
   as identically 0 in every run. The analytic mean/covariance split was never computed in this
   arm. No analytic KL exists anywhere in this code path, so sampled-minus-analytic error is
   also unavailable.
3. **Does lambda increase with M? NO — it falls,** 0.795 -> 0.210 / 0.317.
4. **Is the operator suppressed for a greater fraction of states? NO.**

> **Required explicit statement:** gate frequency remains similar across M (0.40-0.48) and
> **lambda does NOT grow — it drops by a factor of 2.5-3.8**. The covariance share of KL cannot
> be evaluated from these runs.

What *does* change is the gate's **character, not its rate**: the cross-state SD of the KL
estimate falls 0.0412 -> 0.0167 while the mean stays pinned at 0.10. At M=32 the per-state gate
decision is largely driven by estimator noise around a mean sitting exactly on the bound; at
M=512 it tracks the true per-state KL. Same fraction of states gated, different states.

`train/grad_norm_actor` and `train/grad_norm_critic` are logged as identically zero in every run
(unpopulated placeholders), so actor-objective norm, KL-gradient norm and total actor-gradient
norm are **unavailable** from these runs.

---

## Step 8 — Critic training

| quantity | M=32 | M=128 | M=512 |
|---|---|---|---|
| Q | 46.3 +/- 6.5 | 5.4 +/- 8.4 | 2.2 +/- 0.3 |
| bootstrap target | 46.1 +/- 6.6 | 5.3 +/- 8.4 | 2.2 +/- 0.3 |
| value loss | 4.065 | 0.238 | 0.017 |
| nonfinite in eval | 0 in completed runs | 0 | 0 |

Q tracks its target to within 0.2 in every arm — **no Bellman divergence, no value-support
saturation, no overestimation blow-up.** Value loss is *lowest* where return is worst, which is
what an easy, low-variance return distribution looks like, not instability. NaN aborts are 2 of
9 completed sweep runs (22%), matching the 6-of-30 historical prior recorded in prereg §3.3, and
seed 214 passed straight through the step where both aborts occurred, so the step is not a
deterministic trigger.

Temporal evidence: in M=512, Q is flat at 2-3 from eval 0 to eval 20 and never rises. In M=32,
Q begins rising at eval 6 — the same eval range as the return. **The critic follows the policy's
competence; it does not lead the collapse.** The effect of M on the critic is indirect, through
the state distribution a non-running policy visits.

> **CRITIC INSTABILITY WORSENS WITH M: NO.**

---

## Step 9 — Does epsilon_E need to be rescaled with M?

**The null stated in the audit request holds for the E-step, and is confirmed, not assumed.**
`KL(w||uniform)` is **0.500 at every M on every checkpoint**. The eta dual re-solves to hold the
same population constraint as M grows, and `ESS/M`, weighted Q and the E-step's Q gain are all
M-invariant. Increasing M does exactly what correct Monte-Carlo should do: it approximates the
**same** eps_E-constrained population E-step more accurately.

The audit's own trigger for recommending an eps_E sweep requires **both** that finite-M geometry
change systematically **and** that the eta dual fail to compensate. Only the first half is met.
The dual compensates exactly. The systematic finite-M change is **not in the E-step at all — it
is in the M-step**, whose weighted-MLE variance carries the standard `1 - 1/ESS` bias.

> **EVIDENCE THAT epsilon_E MUST BE RETUNED: INSUFFICIENT.**

Retuning eps_E would amount to choosing a constraint that reproduces, at M=512, the covariance
shrinkage that M=32 was getting from estimator bias. That may turn out to be the right
engineering decision, but it is a **different decision from correcting a mis-set constraint**,
and per the audit it must not be made on return. The measured quantity that would motivate any
such change is `weighted 2nd moment / dim` (population value 0.9335 -> only 3.4% width
contraction per fit), not the return ordering.

---

## Single most likely explanation

**The M-step's weighted maximum-likelihood covariance fit is biased low by `1 - 1/ESS`, and the
eta dual pins `ESS/M ~= exp(-eps_E)`, so ESS is proportional to M. Policy-width contraction —
which is what actually produces the M=32 arm's return — is therefore a finite-sample artifact
that increasing M correctly removes.**

At M=32, ESS ~= 17-21 against an action dimension of 21: **the E-step supplies about as many effective
samples as the fit has dimensions.** The weighted MLE shrinks the policy width by 7.2% per
fit against a population value of 3.4%, and 65% of the mean-displacement energy is sampling
noise. The policy contracts hard (sigma 0.80 -> 0.235), commits to a gait, and scores 666. At
M=512 the same estimator is accurate: shrink 3.7% per fit, 10% noise, sigma settles at 0.552
against the KL gate's restoring pull, the policy never commits, and return sits at 11.

The E-step's genuine signal is small enough to make this decisive: at the baseline checkpoint the
critic's spread across 2,048 candidate actions is `std(Q) = 0.019` on `Q = 44.19` — **0.04%** —
and the E-step's weighted Q gain is 0.021. Once the finite-sample noise and shrinkage are removed,
almost nothing is left to drive the policy.

**Ranked alternatives, both still live:**

2. **Mean-displacement noise as an implicit exploration schedule.** The same root cause (finite
   ESS) but a separate channel: 65% of the M=32 step is noise, decaying to 3% by M=2048. The
   frozen probe measures both channels but cannot apportion their contributions to return.
3. **Change in the KL gate's selection character.** At equal firing rate (~0.4-0.48) the gated
   set shifts from noise-selected to true-KL-selected. Real and measured, but secondary: lambda
   moves the wrong way (0.795 -> 0.317) for this to be the driver.

Ruled out by measurement: state/optimizer budget change (Step 1), normalisation bug (Step 2),
memory-driven computation change (Step 3), increased E-step greediness (Step 4), critic
extreme-value selection (Step 5), covariance explosion or saturation (Step 6), increased gate
suppression (Step 7), critic instability (Step 8).

---

## One next experiment (NOT launched)

**Restore the M=32 covariance bias at M=512, and change nothing else.**

Run the M=512 arm with the M-step's fitted per-state variance multiplied by the measured bias
ratio `(1 - 1/ESS_32) / (1 - 1/ESS_512) = 0.9403 / 0.9963 = 0.9438`, applied as a fixed scalar
derived from the measured ESS values in Step 5. The E-step weights, the mean displacement, eta,
eps_E, kl_bound, learning rates, batch sizes and horizon are all untouched — this isolates the
covariance channel from the mean-noise channel, since the mean stays accurately estimated at
M=512.

* If return recovers toward the 600s, **channel 1 (covariance bias) is confirmed** as the
  mechanism, and the finding is that the published M=32 result rests on a finite-sample variance
  underestimate.
* If it does not, the mean-displacement noise channel (alternative 2) carries the effect, and the
  follow-up is the matching intervention on `||d||`.

Either outcome is decisive, and neither requires touching a tuned hyperparameter. Seeds must come
from the 201+ exploratory block per `ledger/README.md`, and the export tag must be extended to
encode M before launch — `HumanoidRun_weighted_mle_s{seed}` does not, which is the overwrite
hazard prereg §3.1 registered and the two smoke runs confirmed live.

---

## Reproduction

```
scripts/msweep_audit/step1.py          effective post-merge config for all 15 runs
scripts/msweep_audit/steps4678.py      per-arm aggregates, evals 15-19
scripts/msweep_audit/step7_gate.py     exact elementwise KL + gate branch from lagrangian_loss
scripts/msweep_audit/step6_temporal.py trajectories and temporal ordering
scripts/msweep_audit/frozen_estep.py   frozen nested-prefix E-step probe (Steps 2 and 5)
    run as: JAX_PLATFORMS=cpu .venv/bin/python scripts/msweep_audit/frozen_estep.py \
              exports/HumanoidRun_weighted_mle_s{2,202,211}_final <out.json> 128 150
```

Raw probe output is committed alongside as `frozen_M32ckpt.json`, `frozen_M128s202.json`,
`frozen_M512ckpt.json`, `step7_gate.json`, `steps4678.json`, `step1.json`.

All frozen probes: 1,024 states (128 envs, 150-step burn-in, 8 chunks x 20-step gap), one common
action cloud of 2,048 draws per state, nested prefixes, eta re-solved per prefix in float64.
No optimizer step, no parameter update, no environment training.

---

## Corrections

**2026-09-02, Step 4 table label.** The rows originally read "median ESS", "median ESS/M",
"median max weight" and "median logit spread". All four are **means**: `reppo.py:1001-1012`
logs `ess=ess.mean()`, `w_max=w_max.mean()` and `q_spread=q_spread.mean()`, with `ess_median`
kept as a separate key. The values were correct; only the labels were wrong. The table now
carries the correct labels and adds the true median ESS from `train/ess_median`
(21.1 / 100.5 / 384.8 over evals 15-19).

Consequently `ESS/d` at M=32 is **0.92 on the mean and 1.01 on the median**, against d = 21.
The Step 5 mechanism is unaffected: the frozen probe measures ESS directly (16.8 at M=32) and
never used the logged value, and the `1 - 1/ESS` shrinkage law is evaluated at the probe's own
ESS. **No conclusion in this report changes.**

**2026-09-02, M=512 arm size.** Seed 214 completed after this report was written, taking the
M=512 arm from n=2 to n=3. Recomputed over evals 15-19 the arm's mean ESS moves 348.2 -> 350.4
and median ESS is 384.8. The arm remains below the n>=5 floor in `docs/prereg_m_sweep_dmc.md`
Sec. 3.1 and still carries no adjudication of its own. Every other figure in this report is
from the n=2 snapshot and is left as written.
