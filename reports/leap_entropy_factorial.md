# LEAP entropy factorial — four cells

Job `3812970` at commit `7be5b05`, node n23g0020. Launch SHA
`7be5b05c3fbba374ee3419727bf4a8db44e34bc8`; amendment `a926e0d` is an ancestor. Bank
`0053b0f3...b158e` verified in-run and reused, not rebuilt.

Cells: `PW_ent`, `PW_noent`, `WML_ent`, `WML_noent`. `WML_ent` is a **hybrid ablation**,
neither standard MPO nor standard REPPO.

**Pipeline validation.** Walker and G1 were re-run in the same job and reproduced their
committed values exactly (Walker `+22.939` / `+127.111` / `4.533x` / `22.952x`; G1
`+12.277` / `+0.671` / `1.986x` / `1.999x`). The LEAP baseline pair also reproduces the
published LEAP result exactly: paired median `+4.296`, 95% CI `[-1.339, +10.528]`, against
the published `+4.30 [-1.34, +10.53]`.

## 1. Return — `score_window3`, frozen definition from commit `7edb8e8`

| seed | PW_ent | PW_noent | WML_ent | WML_noent |
|---|---|---|---|---|
| 301 | 35.340 | 32.026 | 32.994 | 26.597 |
| 302 | 29.288 | 26.098 | 15.556 | 17.765 |
| 303 | 16.024 | 27.087 | 29.561 | 15.691 |
| 304 | 30.046 | 14.887 | 25.785 | 21.787 |
| 305 | 35.036 | 27.050 | 31.744 | 35.768 |
| 306 | 33.793 | 17.824 | 27.733 | 23.265 |
| 307 | 28.920 | 17.440 | 29.543 | 36.039 |
| 308 | 32.522 | 15.196 | 13.584 | 33.861 |
| **mean** | **30.121** | **22.201** | **25.812** | **26.347** |

Paired effects, PW minus WML, paired percentile bootstrap of the paired median over the 8
seed differences, 10,000 resamples, `np.random.default_rng(20260902)`:

```
RETURN_ENT_ON   Delta_ent   = +3.776   95% CI [ -0.623, +13.733]   6/8 positive
RETURN_ENT_OFF  Delta_noent = -6.171   95% CI [-18.599,  +8.332]   3/8 positive
```

**Both intervals contain zero.** The amendment `a926e0d` recorded in advance that LEAP's
return half was expected to be uninformative, because the baseline gap was
`+4.30 [-1.34, +10.53]`. It is. Neither the sign nor the magnitude of `Delta_noent` is a
finding: 3/8 positive with an interval spanning roughly +-18 is consistent with no effect,
and it must not be read as the operator ordering reversing when entropy is removed.

## 2. Geometry — matched-state width on the frozen bank

Reported separately from return. Medians over seeds. Both policies evaluated on identical
frozen states; per `docs/protocol_width_populations.md` this is **matched-state scale**,
not experienced exploration scale.

| cell | subset | med sigma | mean | p95 | max | sat95 | sat99 |
|---|---|---|---|---|---|---|---|
| PW_ent | full | 0.5999 | 0.895 | 2.500 | 51.4 | 0.3079 | 0.1754 |
| PW_ent | pw_half | 0.5398 | 0.884 | 2.512 | 51.4 | 0.3064 | 0.1694 |
| PW_ent | wml_half | 0.6579 | 0.991 | 2.724 | 35.3 | 0.3280 | 0.1865 |
| PW_noent | full | 1.0146 | 2.297 | 6.414 | 396.6 | 0.5812 | 0.4395 |
| PW_noent | pw_half | 0.9836 | 2.077 | 6.604 | 254.5 | 0.5843 | 0.4333 |
| PW_noent | wml_half | 1.0505 | 2.317 | 6.314 | 186.4 | 0.5893 | 0.4464 |
| WML_ent | full | 3.0262 | 18.620 | 59.009 | 21383.2 | 0.6288 | 0.5201 |
| WML_ent | pw_half | 2.5744 | 16.838 | 48.994 | 2271.0 | 0.6335 | 0.5141 |
| WML_ent | wml_half | 2.9929 | 20.699 | 59.653 | 15205.8 | 0.6265 | 0.5100 |
| WML_noent | full | 3.2913 | 68.803 | 135.085 | 303204.7 | 0.6810 | 0.5813 |
| WML_noent | pw_half | 3.8881 | 44.208 | 143.804 | 70357.1 | 0.7015 | 0.6016 |
| WML_noent | wml_half | 2.7911 | 70.819 | 131.285 | 41463.3 | 0.6366 | 0.5377 |

### Paired log-width ratios, halves separately

Convention: median over seeds of per-seed log ratios, exponentiated.

| subset | entropy on both | entropy off both |
|---|---|---|
| full | **5.163x** [3.695, 8.390] | **2.611x** [1.533, 4.455] |
| pw_half | 4.582x [4.199, 11.416] | 3.450x [1.631, 5.479] |
| wml_half | 5.681x [3.026, 11.088] | 2.061x [1.412, 3.568] |

```
WIDTH_ENT_ON  = 5.163x full, 4.582x PW half, 5.681x WML half
WIDTH_ENT_OFF = 2.611x full, 3.450x PW half, 2.061x WML half
```

**Every ratio exceeds 1 and every interval excludes 1**, on both halves and in both
entropy conditions. On identical states the weighted-MLE policies have a larger pre-tanh
scale than the pathwise policies, matching Walker and G1.

### The 2x2

| outcome | PW_ent | PW_noent | WML_ent | WML_noent |
|---|---|---|---|---|
| score_window3 | 30.121 | 22.201 | 25.812 | 26.347 |
| bank median sigma | 0.5787 | 1.1348 | 3.4461 | 3.4466 |
| sat95 | 0.3079 | 0.5812 | 0.6288 | 0.6810 |

### Interactions, descriptive only

```
I_return           +9.947   95% CI [-5.029, +22.646]    (contains zero)
I_width  full      1.977x   95% CI [1.426x, 3.555x]
I_width  pw_half   1.328x   95% CI [0.892x, 3.204x]     (contains 1)
I_width  wml_half  2.757x   95% CI [1.785x, 5.237x]
```

No interaction test was preregistered; no p-value or verdict is introduced.

## 3. Saturation

```
SATURATION_RESULTS
  PW_ent    0.3079 / 0.1754      PW_noent   0.5812 / 0.4395     (sat95 / sat99, full bank)
  WML_ent   0.6288 / 0.5201      WML_noent  0.6810 / 0.5813
```

WML exceeds PW in both entropy conditions. Removing entropy from PW nearly doubles its
saturation, 0.3079 to 0.5812, which tracks that cell's width increase. Adding entropy to
WML lowers saturation slightly, 0.6810 to 0.6288, with essentially no change in median
width.

## 4. Diagnostics

| cell | alpha_kl (final) | gate fire | logged pi_sigma_mean | bank median | bank/logged | eta | ESS |
|---|---|---|---|---|---|---|---|
| PW_ent | 0.0132 [0.0093, 0.0198] | 0.515 | 0.2713 | 0.5999 | 2.21 | — | — |
| PW_noent | 0.0080 [0.0054, 0.0147] | 0.517 | 0.2475 | 1.0146 | 4.10 | — | — |
| WML_ent | 0.3520 [0.1624, 0.4406] | 0.497 | 0.3405 | 3.0262 | 8.89 | 0.006 | 19.88 |
| WML_noent | 0.3308 [0.2277, 0.4211] | 0.512 | 0.4901 | 3.2913 | 6.72 | 0.009 | 20.10 |

`pi_sigma_mean` is a mean after the `+min_std` floor over states and action dimensions at
the final iteration, while the bank column is a median; the two are not like-for-like and
the ratio column is reported for completeness rather than as agreement.

## 5. Integrity

```
NEW_EXPORTS_COMPLETE    = PASS  16/16
NEW_TAGS_NON_COLLIDING  = PASS  4 distinct tags
LEAP baselines intact after launch = PASS 16/16, verified with LEAP's own ledger digest
```

The script's generic `BASELINE_MANIFEST_MATCH` prints `FAIL (0/0)` for LEAP. That check is
**vacuous here**, not a failure: LEAP's baselines predate `exports_manifest.csv` and have
no rows in it, so zero rows were compared. The applicable check is the ledger digest from
`slurm/leap_launch.sh:79-82`, which passes 16/16 both before and after the launch.

## 6. MAIN_LEAP_INTERPRETATION

**Geometry.** On identical frozen states the weighted-MLE policies have a larger pre-tanh
scale than the pathwise policies, in both entropy conditions and on both independently
sourced bank halves, with every interval excluding 1. LEAP therefore agrees with Walker and
G1 on the matched-state ordering, completing that pattern across all three tasks.

**Return.** Uninformative, as recorded in advance. Both paired intervals contain zero.

**Entropy's effect on width is task dependent, and LEAP is where it differs.** Removing the
entropy term from the pathwise arm **widened** it on LEAP, 0.5787 to 1.1348, whereas the
same removal **narrowed** it on Walker (0.4548 to 0.3396) and on G1 (0.3002 to 0.2759).
Adding entropy to the weighted-MLE arm left LEAP's median width essentially unchanged,
3.4466 to 3.4461, against a 4.2x contraction on Walker and a 1.1x widening on G1. Actor
entropy is a moderator whose effect is task dependent; it is not a uniform narrowing force.

No directional entropy prediction is scored, because none was preregistered: the amendment
at `a926e0d` registered `NONE` for both LEAP arms after the entropy-force predictor failed
its prespecified gate at 3/4.

No causal mechanism is claimed. Nothing here asserts that width causes return, that
weighted-MLE exploration is wider, or that any population is the true width.
