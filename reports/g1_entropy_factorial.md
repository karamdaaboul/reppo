# G1 entropy factorial — four cells

Read-only analysis. Job `3791118` at commit `e9a9373`. Launch SHA
`e9a937395a0eca048cc0b848afa38e72e61b0689`; preregistration `aa5533d` is an ancestor.
Bank `g1_fixed_state_bank.npz`, sha256 `cf6f7880...35e9`, verified in-run before use.

Cells, same naming as Walker. `WML_ent` is a **hybrid ablation**: neither standard MPO
nor standard REPPO.

| cell | arm | entropy | export tag | provenance |
|---|---|---|---|---|
| `PW_ent` | pathwise | ON | `pathwise_fa` | baseline, SHA `1c6259e` |
| `PW_noent` | pathwise | OFF | `pathwise_fa_noent` | new, arrays `3766654` / `3766655` |
| `WML_ent` | weighted_mle | ON | `weighted_mle_ent` | new, arrays `3766656` / `3766657` |
| `WML_noent` | weighted_mle | OFF | `weighted_mle` | baseline, SHA `1c6259e` |

**Pipeline validation.** The two baseline cells reproduce the frozen G1 result exactly:
paired median **9.5147** against the published **+9.51**, mean **9.1910** against
**+9.19**, **8/8** positive. A Walker regression run in the same job also reproduced its
committed values exactly (`+22.939`, `+127.111`, `4.533x`, `22.952x`).

## 1. Frozen return scalar

`score_window3` = mean of the final three of 21 logged evaluations, commit `7edb8e8`.

| seed | PW_ent | PW_noent | WML_ent | WML_noent |
|---|---|---|---|---|
| 301 | 26.982 | 17.837 | 13.408 | 15.624 |
| 302 | 29.598 | 17.362 | 18.051 | 17.408 |
| 303 | 18.786 | 11.586 | 11.274 | 14.528 |
| 304 | 11.921 | 10.877 | 9.040 | 9.141 |
| 305 | 23.589 | 18.386 | 10.582 | 15.918 |
| 306 | 11.148 | 10.715 | 4.359 | 9.328 |
| 307 | 30.008 | 7.123 | 11.329 | 11.835 |
| 308 | 29.696 | 12.181 | 8.524 | 14.418 |
| **mean** | **22.716** | **13.258** | **10.821** | **13.525** |

## 2. Paired return effects

Frozen convention: paired percentile bootstrap of the paired median over the 8 seed
differences, 10,000 resamples, `np.random.default_rng(20260902)`, 95 % interval.

| seed | Delta_ent | Delta_noent |
|---|---|---|
| 301 | +13.575 | +2.212 |
| 302 | +11.548 | -0.046 |
| 303 | +7.512 | -2.942 |
| 304 | +2.881 | +1.736 |
| 305 | +13.006 | +2.468 |
| 306 | +6.788 | +1.387 |
| 307 | +18.679 | -4.712 |
| 308 | +21.171 | -2.237 |

```
Delta_ent    paired median +12.277   95% CI [+6.788, +18.679]   8/8 positive
Delta_noent  paired median  +0.671   95% CI [-2.942,  +2.212]   4/8 positive
```

**With entropy removed from both arms the operator difference is not resolved on G1**:
the interval contains zero and the sign count is 4/8.

## 3. Fixed-bank width and saturation

Saturation is `P(|a| > t)` per **action coordinate**, computed exactly from the pre-tanh
Gaussian (not sampled), averaged over the 3072 bank states and 29 coordinates.

| cell | subset | med sigma | mean | p95 | max | sat95 | sat99 |
|---|---|---|---|---|---|---|---|
| PW_ent | full | 0.2896 | 0.341 | 0.714 | 5.5 | 0.1585 | 0.0660 |
| PW_ent | pw_half | 0.2909 | 0.340 | 0.700 | 3.5 | 0.1407 | 0.0542 |
| PW_ent | wml_half | 0.2881 | 0.342 | 0.735 | 5.4 | 0.1793 | 0.0757 |
| PW_noent | full | 0.2635 | 0.364 | 0.998 | 20.6 | 0.3846 | 0.2480 |
| PW_noent | pw_half | 0.2760 | 0.375 | 1.035 | 7.1 | 0.3953 | 0.2590 |
| PW_noent | wml_half | 0.2510 | 0.353 | 0.956 | 20.5 | 0.3720 | 0.2369 |
| WML_ent | full | 0.5737 | 1.040 | 3.224 | 169.2 | 0.3646 | 0.2369 |
| WML_ent | pw_half | 0.5644 | 0.981 | 3.042 | 100.9 | 0.3676 | 0.2385 |
| WML_ent | wml_half | 0.5837 | 1.088 | 3.297 | 149.7 | 0.3617 | 0.2354 |
| WML_noent | full | 0.4969 | 0.992 | 3.272 | 158.9 | 0.3510 | 0.2365 |
| WML_noent | pw_half | 0.5159 | 0.952 | 3.034 | 101.7 | 0.3591 | 0.2423 |
| WML_noent | wml_half | 0.4993 | 1.013 | 3.384 | 158.9 | 0.3462 | 0.2311 |

**The halves agree closely in every cell** — the largest median discrepancy is `PW_noent`
at 0.2760 vs 0.2510 (10 %), and no cell reverses ordering. G1's halves agree better than
Walker's did, so the pooled numbers are safe here.

## 4. Paired width effects

Convention: **median over seeds of per-seed log ratios, exponentiated**. Not a mean of ratios.

| subset | quantity | median log | ratio | 95 % CI |
|---|---|---|---|---|
| full | log_ratio_ent | +0.6864 | **1.986x** | [1.709x, 2.919x] |
| full | log_ratio_noent | +0.6927 | **1.999x** | [1.639x, 2.072x] |
| pw_half | log_ratio_ent | +0.6593 | 1.933x | [1.632x, 2.827x] |
| pw_half | log_ratio_noent | +0.6951 | 2.004x | [1.581x, 2.200x] |
| wml_half | log_ratio_ent | +0.7062 | 2.026x | [1.787x, 2.968x] |
| wml_half | log_ratio_noent | +0.6463 | 1.908x | [1.694x, 2.258x] |

The WML/PW width separation is **essentially unchanged by entropy** on G1: 1.986x with it
on in both, 1.999x with it off in both.

## 5. Factorial interactions — descriptive only

No interaction test was preregistered; no p-value, threshold or verdict is introduced.
Each interval resamples the eight seeds once per replicate and forms both deltas and
their difference inside that replicate.

```
I_return = Delta_ent - Delta_noent
  point +11.606   95% CI [+5.401, +20.916]

I_width  = log_ratio_ent - log_ratio_noent
  full      point -0.0063 -> 0.994x   95% CI [0.843x, 1.300x]
  pw_half   point -0.0358 -> 0.965x   95% CI [0.815x, 1.254x]
  wml_half  point +0.0599 -> 1.062x   95% CI [0.876x, 1.356x]
```

The 2x2, cell means:

| outcome | PW_ent | PW_noent | WML_ent | WML_noent |
|---|---|---|---|---|
| score_window3 | 22.716 | 13.258 | 10.821 | 13.525 |
| bank median sigma | 0.3002 | 0.2759 | 0.6388 | 0.5746 |
| sat95 | 0.1748 | 0.3918 | 0.3763 | 0.3669 |

## 6. Diagnostics

| cell | alpha_kl (final) | gate fire | logged pi_sigma_mean | bank median | bank/logged | eta | ESS |
|---|---|---|---|---|---|---|---|
| PW_ent | 0.0044 [0.0043, 0.0055] | 0.525 | 0.3068 | 0.2896 | 0.94 | — | — |
| PW_noent | 0.0037 [0.0035, 0.0040] | 0.565 | 0.2562 | 0.2635 | 1.03 | — | — |
| WML_ent | 0.3630 [0.2789, 0.3953] | 0.510 | 0.3910 | 0.5737 | 1.47 | 0.004 | 19.83 |
| WML_noent | 0.3745 [0.3239, 0.4111] | 0.510 | 0.3920 | 0.4969 | 1.27 | 0.004 | 19.43 |

On G1 the logged and bank medians agree far better than on Walker in every cell
(0.94-1.47, against Walker's 0.85-4.22), so the on-policy and neutral-bank widths are not
in serious tension here.

## 7. Integrity

```
BASELINE_MANIFEST_MATCH = PASS   16/16 G1 baselines match exports_manifest.csv sha256
NEW_EXPORTS_COMPLETE    = PASS   16/16
NEW_TAGS_NON_COLLIDING  = PASS   4 distinct tags
```

## 8. The preregistered prediction

`docs/prereg_g1_entropy_factorial.md`, commit `aa5533d`, predicted only that **entropy
will modify G1 less strongly than Walker in magnitude**, to be assessed on the
dimensionless width log scale and on within-task relative return change.

**On width: SUPPORTED, decisively.** Effect of the entropy term on each operator's bank
median sigma, as an absolute log ratio:

| operator | Walker | G1 | Walker / G1 |
|---|---|---|---|
| weighted_mle | 8.6869 -> 2.0576, \|log\| = 1.4404 | 0.5746 -> 0.6388, \|log\| = 0.1059 | **13.6x** |
| pathwise | 0.4548 -> 0.3396, \|log\| = 0.2919 | 0.3002 -> 0.2759, \|log\| = 0.0844 | **3.5x** |
| interaction I_width | \|-1.6221\| | \|-0.0063\| | **258x** |

**On return: NOT SUPPORTED.** Within-task relative change from adding or removing the
entropy term:

| operator | Walker | G1 |
|---|---|---|
| pathwise | 909.008 -> 894.794, **-1.6 %** | 22.716 -> 13.258, **-41.6 %** |
| weighted_mle | 760.503 -> 887.775, **+16.7 %** | 13.525 -> 10.821, **-20.0 %** |

Entropy moves G1's return **more** than Walker's, by 27x on the pathwise arm. The
prediction holds on the mechanism it was argued from and fails on the outcome.

The motivating premises were both correct — G1's alpha is 2.075e-04, 69.92x below
Walker's, and the baseline separation is 1.81x against 16.05x — but a 70x smaller alpha
produced a far smaller *width* effect and a far larger *return* effect. Smaller alpha did
not mean a smaller intervention overall.

## 9. What the four cells show — bounded

**The PW advantage on G1 exists only when the entropy term is present.** With entropy on
in both arms the pathwise arm leads by `+12.277 [+6.788, +18.679]`, 8/8 seeds. With it off
in both, the difference is `+0.671 [-2.942, +2.212]`, 4/8, an interval containing zero.

This bears directly on the published G1 headline. The frozen `+9.51` compares `PW_ent`
against `WML_noent` — entropy **on** in the pathwise arm and **absent** in the
weighted_mle arm, because that arm never carried the term. That contrast confounds the
operator with the entropy term. Separating them here, the operator difference survives
when both arms carry entropy and is not resolved when neither does.

Removing entropy from pathwise costs 41.6 % of return (22.716 -> 13.258) while barely
moving its width (0.3002 -> 0.2759). Adding it to weighted_mle costs 20.0 % of return
(13.525 -> 10.821) and slightly *widens* rather than narrows (0.5746 -> 0.6388). Per the
preregistration, no sign of the entropy force is inferred from median sigma alone.

`PW_noent` again shows the pattern seen on Walker: the **narrowest median of the four
cells** (0.2635) yet more than double `PW_ent`'s saturation (0.3846 vs 0.1585), with mean
and max blowing up (0.364 and 20.6 against 0.341 and 5.5). Median width and saturation are
not interchangeable summaries.

### What is and is not established

Both new cells differ from their baselines in exactly one config field, with identical
seeds, budget, critic, optimizer and candidate generation, and the flags-off path was
shown bitwise identical on fixed data at d=29 (T1b, 0.000e+00 over 31 leaves; T2-T5 pass).

1. **entropy term -> return**: supported as causal on G1, 8 paired seeds per cell.
2. **entropy term -> width**: supported as causal, and the effect is small.
3. **width -> return**: **not established and not claimed.** No mediation analysis was done
   or preregistered. On G1 the two dissociate sharply: pathwise loses 41.6 % of its return
   with almost no width change, which is direct evidence *against* reading return
   differences off width on this task.

The preregistered caveat stands: alpha is PW-calibrated by construction, frozen from the
pathwise arm's learned runs, so `WML_ent` tests matched entropy **geometry**, not the
weighted_mle operator's own optimal temperature.

### Reproducibility caveat

Exact same-seed trajectory reproducibility is absent on G1 (`reports/g1_nondeterminism.md`).
Seed pairing may therefore provide less common-random-number variance reduction than under
deterministic execution. This is not a reinterpretation of any frozen result, and the
magnitude at full budget is unmeasured.
