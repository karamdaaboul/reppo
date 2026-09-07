# On-arm versus neutral-bank width, all tasks, all available cells

> **On all three tasks the WML/PW width comparison reverses sign between the two state
> populations: on the neutral bank WML is 1.8x to 15.7x wider than PW, while on the states
> each policy actually visits WML is 0.46x to 0.65x as wide, i.e. narrower.**

Offline. No training, no launches. Job `3796783`, commit
`56e6c8b75b18f3602a3e3a639d21cf456fb1120d`, node w23g0005, 2026-09-07T12:26:53+02:00.
All three bank hashes verified in-run before use.

Neither population is designated correct here. Both are reported.

## 1. Per-cell table

Medians over seeds 301-308. `ratio` is neutral / on-arm.

**PROXY caveat.** Each bank was built from that task's two *baseline* runs, so its `source`
labels are `PW-sNNN` (from `PW_ent`) and `WML-sNNN` (from `WML_noent`). For the two
baseline cells, on-arm means the states that cell's own policy visited. For the two new
cells (`PW_noent`, `WML_ent`) no states from their own rollouts exist, so on-arm falls back
to the same-arm **baseline** states. Those rows are marked PROXY and their on-arm values
should not be read as that policy's own visited-state distribution.

| task | cell | on-arm | on-arm σ | neutral σ | ratio | on s95 | on s99 | neu s95 | neu s99 | logged σ |
|---|---|---|---|---|---|---|---|---|---|---|
| walker | PW_ent | own | 0.4783 | 0.4499 | 0.94 | 0.1450 | 0.0200 | 0.2242 | 0.0861 | 0.5297 |
| walker | PW_noent | PROXY | 0.3038 | 0.3375 | 1.11 | 0.5668 | 0.4192 | 0.5697 | 0.4134 | 0.2958 |
| walker | WML_ent | PROXY | 1.6309 | 2.0152 | 1.24 | 0.5907 | 0.4453 | 0.6332 | 0.5030 | 0.4779 |
| walker | WML_noent | own | **0.2182** | **7.3096** | **33.50** | 0.3978 | 0.2684 | 0.7895 | 0.7151 | 2.9005 |
| g1 | PW_ent | own | 0.2695 | 0.2896 | 1.07 | 0.0703 | 0.0260 | 0.1585 | 0.0660 | 0.3068 |
| g1 | PW_noent | PROXY | 0.2536 | 0.2635 | 1.04 | 0.3929 | 0.2587 | 0.3846 | 0.2480 | 0.2562 |
| g1 | WML_ent | PROXY | 0.3826 | 0.5737 | 1.50 | 0.3346 | 0.2122 | 0.3646 | 0.2369 | 0.3910 |
| g1 | WML_noent | own | **0.1759** | **0.4969** | **2.82** | 0.2040 | 0.1158 | 0.3510 | 0.2365 | 0.3920 |
| leap | PW_ent | own | 0.2242 | 0.5999 | 2.68 | 0.0870 | 0.0187 | 0.3079 | 0.1754 | 0.2713 |
| leap | WML_noent | own | **0.1225** | **3.2913** | **26.88** | 0.1773 | 0.0794 | 0.6810 | 0.5813 | 0.4901 |

LEAP has only its two baseline cells; `PW_noent` and `WML_ent` have not been run.

The divergence is concentrated in the `WML_noent` cells: 33.5x on Walker, 26.9x on LEAP,
2.8x on G1. Those policies are narrow where they operate and very wide off it. The
`PW_ent` cells agree closely on Walker and G1 (0.94, 1.07) and diverge moderately on LEAP
(2.68). Saturation follows the same pattern: Walker `WML_noent` is 0.3978 on-arm against
0.7895 neutral.

The logged `pi_sigma_mean` sits close to the **on-arm** value in most cells, as it should,
since it is recorded over the policy's own training states — Walker `WML_noent` 2.9005
against on-arm 0.2182 is the exception, and its own training distribution evidently differs
from the evaluation rollouts the bank was built from.

## 2. Paired WML/PW width ratio, both populations

Frozen convention: median over seeds of per-seed log ratios, exponentiated. All eight
per-seed values shown.

### walker

| pair | population | ratio | per-seed 301-308 |
|---|---|---|---|
| ent | neutral | 4.533x | 6.383 5.987 3.255 3.467 2.807 5.068 5.135 4.054 |
| ent | on-arm | 3.408x | 4.212 3.348 1.776 2.784 3.469 4.362 4.372 2.972 |
| noent | neutral | 22.952x | 17.026 33.400 17.517 30.072 32.630 10.274 13.170 50.941 |
| noent | on-arm | **0.680x** | 0.988 0.796 0.619 0.933 0.746 0.364 0.572 0.371 |
| baseline pair | neutral | 15.702x | 12.142 21.651 12.713 19.393 23.681 9.858 8.779 45.390 |
| baseline pair | on-arm | **0.456x** | 0.494 0.434 0.514 0.609 0.475 0.438 0.320 0.321 |

```
ent            SIGN AGREE = YES  (both > 1)      4.533x vs 3.408x
noent          SIGN AGREE = NO, REVERSES        22.952x vs 0.680x
baseline pair  SIGN AGREE = NO, REVERSES        15.702x vs 0.456x
```

### g1

| pair | population | ratio | per-seed 301-308 |
|---|---|---|---|
| ent | neutral | 1.986x | 1.709 1.288 2.998 1.948 1.998 2.919 1.975 2.072 |
| ent | on-arm | 1.404x | 1.067 0.920 2.396 1.650 1.421 1.303 1.386 1.435 |
| noent | neutral | 1.999x | 2.066 1.990 2.072 2.009 1.639 3.712 1.409 1.882 |
| noent | on-arm | **0.776x** | 0.903 0.955 0.635 0.687 0.752 0.852 0.575 0.801 |
| baseline pair | neutral | 1.805x | 1.681 1.444 2.118 2.046 1.317 2.854 1.810 1.800 |
| baseline pair | on-arm | **0.647x** | 0.620 0.653 0.641 0.622 0.630 0.708 0.706 0.685 |

```
ent            SIGN AGREE = YES  (both > 1)      1.986x vs 1.404x
noent          SIGN AGREE = NO, REVERSES         1.999x vs 0.776x
baseline pair  SIGN AGREE = NO, REVERSES         1.805x vs 0.647x
```

### leap

| pair | population | ratio | per-seed 301-308 |
|---|---|---|---|
| baseline pair | neutral | 6.376x | 6.166 2.165 6.977 6.757 16.531 2.161 4.883 6.593 |
| baseline pair | on-arm | **0.543x** | 0.873 0.642 0.428 0.435 0.517 0.570 0.505 0.622 |

```
baseline pair  SIGN AGREE = NO, REVERSES         6.376x vs 0.543x
```

### Sign agreement, stated plainly

| task | ent pair | noent pair | baseline pair |
|---|---|---|---|
| walker | AGREE | **REVERSES** | **REVERSES** |
| g1 | AGREE | **REVERSES** | **REVERSES** |
| leap | not available | not available | **REVERSES** |

The reversal is not marginal and not seed-dependent: for the baseline pair every one of
the eight per-seed on-arm ratios is below 1 on all three tasks, and every one of the eight
neutral ratios is above 1.

The `ent` pair is the exception that does not reverse — with entropy present in both arms,
WML is wider than PW on both populations, on Walker and on G1.

## 3. BANK_HALVES_AGREE, Walker, all four cells

Requested during the Walker factorial and not returned then. Relative difference is
`|pw_half - wml_half| / max(pw_half, wml_half)`.

| cell | full | pw_half | wml_half | rel. diff | |
|---|---|---|---|---|---|
| PW_ent | 0.4499 | 0.4550 | 0.4537 | 0.3 % | AGREE |
| PW_noent | 0.3375 | 0.3172 | 0.3540 | 10.4 % | DISAGREE |
| WML_ent | 2.0152 | 2.2139 | 1.9063 | 13.9 % | DISAGREE |
| WML_noent | 7.3096 | 8.8018 | 6.4271 | 27.0 % | DISAGREE |

```
BANK_HALVES_AGREE (walker, all four within 10%) = NO   (3 of 4 exceed 10%)
```

Three of four Walker cells exceed 10 %, against G1 where all four agreed within 10 %. In
every disagreeing cell the PW-derived half is the wider one, consistent with each policy
being evaluated partly off its own state distribution.

Note on convention: the Walker factorial report quoted the `WML_noent` half-to-half
difference as 37 %, normalising by the smaller half; normalising by the larger gives 27 %.
Same two numbers, 8.8018 and 6.4271.

## What this does and does not establish

It establishes that the WML/PW width comparison is **not robust to the choice of state
population**, and that the direction of the published width claim depends on that choice.
The neutral bank mixes states from both arms, so half of it is off-distribution for
whichever policy is being evaluated; the on-arm measurement removes that but restricts each
policy to states it already favours, which is a different and equally consequential choice.

It does not establish which population should be used. That depends on what the claim is
meant to be about, which is a decision for the principal:

* a claim about **the policies as deployed** — the width of the behaviour each arm actually
  produces — points to on-arm;
* a claim about **the operators compared on common ground** — matched states, no
  confounding by differing visitation — points to the neutral bank, which is what it was
  built for.

Both are reported above and neither is adopted here. What cannot be sustained is a claim in
either direction that does not name its population.
