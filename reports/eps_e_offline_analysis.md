# Offline analysis: is a Walker WML-32 `eps_E = 0.1` training arm justified?

Read-only. No training was started. No training implementation was modified.

Producing jobs, both at commit `5737be373a66ed48be737508f34a0ada75a66d7b`:

| job | script | node | elapsed | log |
|---|---|---|---|---|
| 3632962 | `scripts/analysis/fixed_bank_width_saturation.py` | n23g0001 | 00:56:16 | `slurm/logs/fbwidth_3632962.out` |
| 3633063 | `scripts/analysis/walker_eps_counterfactual.py` | w23g0003 | 00:00:38 | `slurm/logs/epscf_3633063.out` |

State banks (frozen, arm-major, 1536 PW + 1536 WML):

| task | path | sha256 | depths |
|---|---|---|---|
| Walker | `reports/artifacts/walker_fixed_state_bank.npz` | `8adfeb0bf70bddcdbd64a84b972b4dbd62617c64e1c948589a7adc30cf64aa21` | 50/150/300/500/700/900 |
| G1 | `reports/artifacts/g1_fixed_state_bank.npz` | `cf6f7880b3a5b59433727f7a245b5ce97749096981f03233fd434ab3371935e9` | 50/150/300/500/700/900 |
| LEAP | `reports/artifacts/leap_fixed_state_bank.npz` | `0053b0f361e45b9227d15b982ff667a5fc665469dd54e673c70c0682bcb158b2` | 50/100/200/300/400/480 |

The Walker bank was reused, not rebuilt; its sha256 matches the value recorded at `5737be3`.

---

## Phase 0 — provenance of the 0.89–0.91 shrinkage statistic

```
SOURCE_FILE   = docs/prereg_wml_covariance_correction.md, section 2.1
SOURCE_COMMIT = f709857636e1175033ebcb6d456f4b6f04d435ff
TASK          = HumanoidRun          ARM = weighted_mle
SEEDS         = checkpoints s2 / s202 / s211 (trained at M = 32 / 128 / 512)
STATE_BANK    = frozen nested-prefix probe, 1024 states
M             = M = 2048 prefix vs M = 32 prefix of one candidate set
ETA           = re-solved on each prefix (`solved_on_prefix`)
QUANTITY      = weighted second moment per dimension, whitened units (pi_old variance = 1)
ONE_STEP_TARGET_OR_REAL_POLICY_CHANGE = ONE-STEP E-STEP TARGET
PRE_TANH_OR_POST_TANH = PRE-TANH
RECONCILIATION = DIFFERENT_QUANTITY
```

| checkpoint | trained M | 2nd moment M=32 | M=2048 | ratio |
|---|---|---|---|---|
| s2 | 32 | 0.8605 | 0.9335 | 1.0848 |
| s202 | 128 | 0.8505 | 0.9286 | 1.0918 |
| s211 | 512 | 0.8455 | 0.9125 | 1.0793 |

Different task, different seeds and tier, different bank, and — decisively — a different
object: a one-step fitting target, not a trained policy width.

---

## Phase 1 — does the widening replicate on G1 and LEAP?

Per-coordinate pre-tanh sigma on the frozen bank, PW vs WML, paired within seed.

| task | d | PW median sigma | WML median sigma | median paired ratio | seed pairs | coords 8/8 | coords >=5/8 |
|---|---|---|---|---|---|---|---|
| Walker | 6 | 0.436 – 0.484 | 3.83 – 20.07 | **16.05x** | 8/8 | 6 of 6 | 6 of 6 |
| G1 | 29 | 0.260 – 0.361 | 0.387 – 1.031 | **1.81x** | 8/8 | 20 of 29 | 27 of 29 |
| LEAP | 16 | 0.419 – 0.742 | 1.41 – 7.25 | **6.38x** | 8/8 | 10 of 16 | 16 of 16 |

`WML_GT_PW_SEED_PAIRS = 8/8` on all three tasks.

The widening is a general property of the WML arm, not a Walker artifact. Its magnitude is
strongly task-dependent and does not track action dimension monotonically (G1, the widest
action space at d = 29, shows the smallest effect).

WML mean and max sigma are dominated by extreme outliers (Walker s304 max 3.75e7, s303 max
4.27e6). All claims here use the median, which is not driven by those tails.

## Phase 2 — tanh saturation fingerprint

Exact, via `P(|tanh Y| > t) = Phi((-c-mu)/s) + 1 - Phi((c-mu)/s)`, `c = atanh(t)`.

| task | PW P(abs a > .95) | WML P(abs a > .95) | dSat95 range | pairs |
|---|---|---|---|---|
| Walker | 0.210 – 0.261 | **0.729 – 0.844** | +0.513 … +0.601 | 8/8 |
| G1 | 0.145 – 0.236 | 0.330 – 0.469 | +0.156 … +0.233 | 8/8 |
| LEAP | 0.241 – 0.402 | 0.485 – 0.758 | +0.116 … +0.482 | 8/8 |

On Walker the WML policy places **73–84 % of its action mass beyond `|a| > 0.95`**, against
21–26 % for PW. The trained WML Walker policy is effectively bang-bang. This is the
mechanistically substantive form of the widening: pre-tanh sigma of 6–20 is not "a wider
policy", it is a saturated one, where the tanh Jacobian is near zero over most of the mass.

## Phase 3 — offline `eps_E` counterfactual (Walker, identical candidates)

M = 32, 512 states, 8 independent candidate draws per seed. The eta dual is re-solved at each
`eps_E` on the **same** candidate sets, so the two rows differ only through eta.

| eps_E | eta | ESS (arith.) | ESS/M | w_max | radial | rho_med | rho_mean | P(rho>1) | rho_p95 | mean(1-1/ESS) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.5 (current) | 0.958 | 23.65 | 0.739 | 0.153 | 6.016 | 0.866 | 0.838 | 0.327 | 1.401 | 0.880 |
| 0.1 | 5.495 | 29.10 | 0.909 | 0.056 | 5.979 | 0.925 | 0.941 | 0.388 | 1.410 | 0.957 |

Deltas: eta x5.73, ESS/M +0.170, w_max /2.7, rho_med **+0.0588**, rho_mean +0.1025.

`eps_E = 0.1` is a large, real intervention on the E-step operating point. It is not a no-op.

---

## Phase 4 — Channel A vs Channel B

**Channel A, extreme-candidate selection.** Measured by the weighted pre-tanh radial second
moment `sum_i w_i ||u_i||^2` in whitened units, where the unweighted expectation is exactly
d = 6.

- `eps_E = 0.5`: 6.0164, excess **+0.27 %**
- `eps_E = 0.1`: 5.9785, excess **-0.36 %**

The E-step weights are essentially radius-neutral at both settings. **Channel A is null.**
It is not merely small relative to Channel B; it is indistinguishable from zero and changes
sign between the two settings.

**Channel B, finite-sample contraction.** Against the per-state reference `mean(1 - 1/ESS)`:

- `eps_E = 0.5`: rho_med 0.8663 vs 0.8797 → gap **-0.0134**
- `eps_E = 0.1`: rho_med 0.9251 vs 0.9570 → gap **-0.0320**

```
DOMINANT_CHANNEL = B (finite-sample contraction)
```

Channel B accounts for the contraction to within 1.3–3.2 percentage points. The residual is
small, of consistent sign, and does not require Channel A to explain.

**The consequence that matters.** At *both* `eps_E` values the one-step E-step target is a
*contraction*: rho_med < 1. The E-step never produces a target wider than the current policy
in the median. Yet the trained WML Walker policy is 16x wider than PW on the same states.

**The observed widening is therefore not produced by the mean of the E-step target**, and
`eps_E` does not act on the channel that would have to be suppressed to remove it.

## Phase 5 — reconciling with `E[Sigma_fit] ~ (1 - 1/ESS) Sigma_old`

The prior expectation used `1 - 1/ESS` at the run-wide mean training ESS of 20.52, giving
0.951, and the measured 0.866 then looked like an unexplained 0.085 excess contraction.

That comparison was mis-specified. `1 - 1/ESS` is a **per-state** factor and must be averaged
per state, not evaluated at the mean ESS. The script does this correctly (line 86,
`ref = mean(1 - 1/ess)`), while the reported ESS column is `mean(ess)`. The two differ by
Jensen's inequality, and the gap is large:

| eps_E | arithmetic mean ESS | implied harmonic mean ESS | mean(1 - 1/ESS) |
|---|---|---|---|
| 0.5 | 23.65 | **8.31** | 0.880 |
| 0.1 | 29.10 | **23.26** | 0.957 |

At `eps_E = 0.5` the per-state ESS distribution is strongly right-skewed: a minority of states
with ESS of order 2–4 dominate the covariance contraction, and the harmonic mean (8.31) is a
third of the arithmetic mean (23.65). Against the correct reference the measured contraction
is 0.866 vs 0.880 — a gap of 0.013.

```
ESS_GAP_STATUS = RESOLVED (Jensen; harmonic vs arithmetic mean ESS)
```

The apparent anomaly was an artifact of evaluating a convex function at a mean. The
HumanoidRun band of Phase 0 (0.846–0.934) is consistent with the same mechanism.

Note the second-order effect: `eps_E = 0.1` compresses the per-state ESS distribution
(harmonic/arithmetic rises from 0.35 to 0.80). It removes the low-ESS tail, which is exactly
the population responsible for the noisiest E-step fits.

## Phase 6 — recommendation

```
TRAINING_ARM_SCIENTIFICALLY_WORTHWHILE
```

Stated against my own Phase 4 finding, which points the other way on the one-step statistic.

**Evidence that `eps_E = 0.1` will NOT remove the widening:**

1. It moves the one-step target *closer* to pi_old (rho_med 0.866 → 0.925), i.e. it *reduces*
   the only contractive force the E-step applies.
2. It *increases* the fraction of coordinates whose target is wider than the current policy
   (P(rho>1) 0.327 → 0.388).
3. Channel A is null at both settings, so `eps_E` does not act on a selection channel.
4. In the limit `eps_E → 0` the E-step degenerates to a no-op (uniform weights, target =
   pi_old), so lowering `eps_E` moves WML toward exerting no corrective force at all.

**Evidence that it might:**

1. It cuts weight concentration 2.7x (w_max 0.153 → 0.056) and nearly triples the harmonic
   mean ESS (8.31 → 23.26), eliminating the low-ESS tail.
2. The one-step statistic measures the *expectation* of a single update. The trained policy
   is the result of ~204,000 updates. If widening is driven by *variance* of the fitted
   covariance compounding across updates rather than by its mean, then `eps_E = 0.1` is a
   direct and strong intervention on precisely that quantity — and every measurement in
   Phase 3 is blind to it.

These two mechanisms make **opposite** directional predictions on the same cheap experiment.
That is what makes the arm worth running: the offline analysis cannot separate them, and it
has narrowed the question to a single binary that 2.9 GPU-hours resolves.

**Cost.** 1296–1341 s per seed (mean 1315 s) at 52,297,728 steps; 8 seeds = **2.92 GPU-hours**.

**Both outcomes are publishable.** If widening persists, `eps_E` is exonerated and the source
is localised to the M-step gate, the entropy term, or the `min_std` floor — which redirects
the mechanism claim. If it vanishes, the update-variance mechanism is confirmed and the
paper gains a controlling knob with a measured offline signature.

**Honest caveat.** My reading of the one-step evidence is that the *modal* outcome is B3
(widening persists). The arm is worth running because a preregistered null here is
load-bearing — it closes off the E-step explanation that the current draft's framing invites
— not because I expect it to succeed.

---

## What is NOT in this analysis

- No figures. Neither script calls `savefig`; the Phase 8 "figures" item has no content, and
  none were fabricated.
- No training was started; no Hopper run; no `eps_E = 0.1` run; no change to `reppo.py`,
  `jax_models.py`, or any config.
- The G1 and LEAP banks are newly built here. Their sha256 values are recorded above and are
  the reference for any future reuse.
