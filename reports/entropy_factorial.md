# Walker entropy factorial — four cells

`ANALYSIS_ONLY = YES`. Read-only. No training, no launches, no edits to experimental
source. Job `3740916` at commit `4665b08`, node n23g0020, 00:00:49.

Cells, named this way throughout. `PW-H` / `WML-H` are deliberately not used: the same
suffix would mean "minus entropy" in one arm and "with entropy" in the other.

| cell | arm | entropy | export tag | provenance |
|---|---|---|---|---|
| `PW_ent` | pathwise | ON | `pathwise_fa` | corrected baseline, SHA `1c6259e` |
| `PW_noent` | pathwise | OFF | `pathwise_fa_noent` | new, SHA `4665b08`, array `3734158` |
| `WML_ent` | weighted_mle | ON | `weighted_mle_ent` | new, SHA `4665b08`, array `3734159` |
| `WML_noent` | weighted_mle | OFF | `weighted_mle` | corrected baseline, SHA `1c6259e` |

Seeds 301-308 in every cell, 32 checkpoints. `WML_ent` is a **hybrid ablation**:
neither standard MPO nor standard REPPO.

## 1. Frozen return scalar

`score_window3` = mean of the final three of 21 logged evaluations (indices 18, 19, 20),
the preregistered primary definition from commit `7edb8e8`. Not redefined, and the final
evaluation alone is not substituted.

| seed | PW_ent | PW_noent | WML_ent | WML_noent |
|---|---|---|---|---|
| 301 | 909.789 | 907.770 | 873.461 | 784.550 |
| 302 | 898.474 | 889.141 | 894.779 | 777.205 |
| 303 | 907.873 | 929.904 | 852.495 | 798.903 |
| 304 | 910.980 | 832.328 | 886.399 | 710.224 |
| 305 | 908.442 | 911.309 | 887.146 | 762.514 |
| 306 | 909.285 | 883.104 | 925.694 | 717.011 |
| 307 | 912.191 | 887.481 | 901.208 | 887.084 |
| 308 | 915.031 | 917.315 | 881.020 | 646.534 |
| **mean** | **909.008** | **894.794** | **887.775** | **760.503** |

## 2. Fixed-bank width and saturation

Bank `reports/artifacts/walker_fixed_state_bank.npz`, sha256
`8adfeb0bf70bddcdbd64a84b972b4dbd62617c64e1c948589a7adc30cf64aa21`, **verified in-run
before use**; the script aborts on mismatch. The same 3072 states for all 32 checkpoints.

**Saturation definition.** `P(|a| > t)` for `t` = 0.95 and 0.99, per **action
coordinate** (not per action vector), computed **exactly** from the pre-tanh Gaussian
via `P(|tanh Y| > t) = Phi((-c-mu)/s) + 1 - Phi((c-mu)/s)` with `c = atanh(t)` — the
policy is not sampled. Averaged over bank states and coordinates. State distribution is
the frozen neutral bank.

Medians over the eight seeds. The bank is half PW-visited and half WML-visited states,
so every quantity is reported three ways.

| cell | subset | med sigma | mean | p95 | max | sat95 | sat99 |
|---|---|---|---|---|---|---|---|
| PW_ent | full | 0.4499 | 0.497 | 0.859 | 4.2 | 0.2242 | 0.0861 |
| PW_ent | pw_half | 0.4550 | 0.510 | 0.876 | 4.2 | 0.2133 | 0.0742 |
| PW_ent | wml_half | 0.4537 | 0.485 | 0.843 | 2.6 | 0.2370 | 0.0978 |
| PW_noent | full | 0.3375 | 0.701 | 1.923 | 202.0 | 0.5697 | 0.4134 |
| PW_noent | pw_half | 0.3172 | 0.748 | 1.850 | 151.2 | 0.5781 | 0.4186 |
| PW_noent | wml_half | 0.3540 | 0.731 | 1.895 | 50.2 | 0.5507 | 0.3910 |
| WML_ent | full | 2.0152 | 3.745 | 12.968 | 887.9 | 0.6332 | 0.5030 |
| WML_ent | pw_half | 2.2139 | 4.422 | 15.569 | 868.9 | 0.6555 | 0.5295 |
| WML_ent | wml_half | 1.9063 | 3.340 | 10.421 | 175.1 | 0.6016 | 0.4638 |
| WML_noent | full | 7.3096 | 318.358 | 463.064 | 456002.8 | 0.7895 | 0.7151 |
| WML_noent | pw_half | 8.8018 | 259.097 | 559.871 | 295101.1 | 0.8226 | 0.7511 |
| WML_noent | wml_half | 6.4271 | 270.124 | 311.533 | 157096.5 | 0.7520 | 0.6736 |

**Do the halves agree?** Qualitatively yes, in all four cells, and no cell reverses its
ordering between halves. The PW cells agree closely (PW_ent 0.4550 vs 0.4537, under 1 %;
PW_noent 0.3172 vs 0.3540, 12 %). The WML cells differ more (WML_ent 2.2139 vs 1.9063,
16 %; WML_noent 8.8018 vs 6.4271, 37 %), always with the PW-derived half wider —
consistent with each policy being evaluated partly off its own state distribution. The
gap is a magnitude difference, not a sign difference, so pooled numbers are reported
alongside rather than instead of the halves. This is unlike the earlier tanh-clip
finding, which reversed a sign off-distribution.

## 3. Paired return effects

`Delta_ent(s) = PW_ent(s) - WML_ent(s)`, `Delta_noent(s) = PW_noent(s) - WML_noent(s)`.
Positive means pathwise higher.

| seed | Delta_ent | Delta_noent |
|---|---|---|
| 301 | +36.328 | +123.220 |
| 302 | +3.696 | +111.937 |
| 303 | +55.379 | +131.001 |
| 304 | +24.581 | +122.104 |
| 305 | +21.296 | +148.795 |
| 306 | -16.409 | +166.093 |
| 307 | +10.983 | +0.397 |
| 308 | +34.011 | +270.780 |

Frozen convention, paired percentile bootstrap over the 8 seed differences, 10,000
resamples, `np.random.default_rng(20260902)`, 95 % percentile interval:

```
Delta_ent    paired median  +22.939   95% CI [  +3.696,  +36.328]   7/8 positive
Delta_noent  paired median +127.111   95% CI [+111.937, +166.093]   8/8 positive
```

## 4. Paired width effects

**Ratio convention, fixed:** the **median over seeds of the per-seed log ratios**,
exponentiated for display. Not a mean of ratios.

`log_ratio_ent(s) = log median_sigma_WML_ent(s) - log median_sigma_PW_ent(s)`, and
likewise for `noent`.

| subset | quantity | median log | ratio | 95 % CI |
|---|---|---|---|---|
| full | log_ratio_ent | +1.5113 | **4.533x** | [3.255x, 5.987x] |
| full | log_ratio_noent | +3.1334 | **22.952x** | [13.170x, 33.400x] |
| pw_half | log_ratio_ent | +1.6050 | 4.978x | [2.752x, 5.934x] |
| pw_half | log_ratio_noent | +3.3764 | 29.264x | [17.136x, 55.992x] |
| wml_half | log_ratio_ent | +1.4434 | 4.235x | [3.104x, 6.026x] |
| wml_half | log_ratio_noent | +2.9642 | 19.379x | [9.949x, 20.836x] |

## 5. Factorial interactions — descriptive only

No interaction test was preregistered. No p-value, threshold or verdict is introduced.
Each interval comes from resampling the eight seeds **once per replicate** and forming
both deltas and their difference **inside** that replicate — the two deltas are not
bootstrapped separately and subtracted.

```
I_return = Delta_ent - Delta_noent
  point -104.172   95% CI [-153.597, -86.892]

I_width  = log_ratio_ent - log_ratio_noent
  full      point -1.6221 -> 0.197x   95% CI [0.104x, 0.385x]
  pw_half   point -1.7714 -> 0.170x   95% CI [0.072x, 0.321x]
  wml_half  point -1.5208 -> 0.219x   95% CI [0.152x, 0.479x]
```

The 2x2, readable directly (cell means over seeds):

| outcome | PW_ent | PW_noent | WML_ent | WML_noent |
|---|---|---|---|---|
| score_window3 | 909.008 | 894.794 | 887.775 | 760.503 |
| bank median sigma | 0.4548 | 0.3396 | 2.0576 | 8.6869 |
| sat95 | 0.2293 | 0.5605 | 0.6138 | 0.7861 |

## 6. Diagnostics

Medians over seeds, with the per-cell range where shown. `eta` and `ESS` are meaningful
only in the two weighted_mle cells; the pathwise cells log 0 by construction.

| cell | alpha_kl (final) | gate fire | logged pi_sigma_mean | bank median | bank/logged | eta | ESS |
|---|---|---|---|---|---|---|---|
| PW_ent | 0.0120 [0.0101, 0.0246] | 0.562 | 0.5297 | 0.4499 | 0.85 | — | — |
| PW_noent | 0.0216 [0.0176, 0.0321] | 0.589 | 0.2958 | 0.3375 | 1.14 | — | — |
| WML_ent | 0.3776 [0.3092, 0.4264] | 0.563 | 0.4779 | 2.0152 | 4.22 | 0.020 | 20.29 |
| WML_noent | 0.3585 [0.2135, 0.4329] | 0.563 | 2.9005 | 7.3096 | 2.52 | 0.037 | 20.04 |

### Correction: alpha_kl and lambda_eff

They are the **same field**. `scripts/export_ckpt.py:204` defines
`alpha_kl = actor.lagrangian()`, the KL Lagrangian multiplier, which is what the
implementation audit at `b48a6ed` calls `lambda_eff`.

An earlier note in this project said PW_noent's multiplier "collapsed to 0.018-0.032
against PW_ent's much larger values". **That was wrong in both directions.** PW_ent's
own range is 0.0101-0.0246, which is *lower* than PW_noent's 0.0176-0.0321, and the
audit's recorded Walker PW-1 `lambda_eff` of 0.026 (range 0.006-0.026) sits at the top
of PW_ent's range, exactly where it should. Removing the entropy term left the KL
multiplier slightly **higher**, not collapsed, and PW_ent was never the larger of the two.

### Logged sigma versus bank median

The two measure different state distributions: `pi_sigma_mean` is logged over the
policy's own training states, the bank median over the frozen neutral bank.

The previously reported pattern — agreement in PW, disagreement in WML — **holds in the
new cells too**, but the magnitude here is far smaller than the 22x reported earlier.
Both PW cells agree within 15 % (ratios 0.85 and 1.14). Both WML cells disagree, by
4.22x and 2.52x.

The sharpest instance is `WML_ent`: its logged on-policy sigma is **0.4779**, slightly
*narrower* than PW_ent's 0.5297, while its bank median is **2.0152**, 4.2x wider. That
cell is narrow where it actually operates and wide off its own distribution. Any width
claim about it must say which state distribution it refers to.

## 7. Integrity

```
BASELINE_MANIFEST_MATCH = PASS   16/16 baseline export dirs match exports_manifest.csv sha256
NEW_EXPORTS_COMPLETE    = PASS   16/16 new dirs have actor, critic, meta, normalizer
NEW_TAGS_NON_COLLIDING  = PASS   4 distinct tags
TRAINING_FILE_MODIFIED  = PASS   none; this analysis touched no experimental source
GIT_TREE_CLEAN          = PASS   apart from the known untracked helper scripts
```

## 8. What the four cells show — bounded

**Q1, does the width gap persist with entropy ON in both?** Yes, and the return gap
does too, but both shrink. Width `4.533x` with CI `[3.255x, 5.987x]`, excluding 1.
Return `+22.939` with CI `[+3.696, +36.328]`, excluding 0, 7/8 seeds positive.

**Q2, with entropy OFF in both?** Both gaps are much larger. Width `22.952x`
`[13.170x, 33.400x]`; return `+127.111` `[+111.937, +166.093]`, 8/8 positive.

**Q3, how much does entropy move each operator?** Adding it to weighted_mle moves the
bank median from 8.6869 to 2.0576 and `score_window3` from 760.503 to 887.775. Removing
it from pathwise moves the bank median from 0.4548 to **0.3396 — narrower, not wider** —
and `score_window3` from 909.008 to 894.794.

That PW narrowed is what the preregistration recorded in advance as the expectation from
the local derivative, and recorded in advance as **not a failure**. It is confirmed here.

A detail that cuts against reading width as a single axis: `PW_noent` has the **narrowest
median of all four cells** (0.3375) yet its saturation more than doubles against
`PW_ent` (sat95 0.5697 vs 0.2242). Its mean and max blow up (0.701 and 202.0, against
0.497 and 4.2). Removing the entropy term narrowed the typical state while producing a
heavy tail of very wide states. Median width and saturation are not interchangeable
summaries.

### What is and is not established

Both new cells differ from their baselines in exactly one config field, with identical
seeds, budget, critic, optimizer and candidate generation, and the flags-off path is
bitwise identical to the pre-change code (T1). So:

1. **entropy term → width**: supported as causal. Single-field intervention, 8 paired seeds.
2. **entropy term → return**: supported as causal, same design.
3. **width → return**: **not established, and not claimed.** No mediation analysis was
   done, none was preregistered, and the entropy term enters the objective directly, so
   it can move return through paths that have nothing to do with width. That
   `WML_ent` is both narrower and higher-returning than `WML_noent` is an **association**
   between two outcomes of the same intervention, not evidence that the narrowing caused
   the return recovery.

The preregistered caveat stands and limits claim 2: alpha is **PW-calibrated by
construction**, frozen from PW's learned runs, and weighted_mle never carried an entropy
term before. `WML_ent` therefore tests matched entropy **geometry**, not that operator's
own optimal temperature. A different alpha could move both outcomes.
