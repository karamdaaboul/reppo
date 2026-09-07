# Stage A — measuring the actor-entropy force F_H on existing checkpoints

Offline. No training, no launches. Job `3795539`, commit
`56d48fb85e42323c21d9982c4427cc2fc3bda434`, node n23g0020,
**measured at 2026-09-07T11:43:48+02:00**.

**Outcome: the instrument fails its own validation gate (3/4 on the primary
measurement). Per the stage A instruction, no LEAP prediction is preregistered.**

## A2. Convention

```
F := -dL/dlog_sigma        F > 0 widens, F < 0 contracts
F_H = alpha * [ 1 - 2 * sigma^2 * E(sech^2(y)) ],   y = mu + sigma*u,  u ~ N(0, I)
```

Each task uses its **own** frozen alpha, read from all 8 exports and confirmed identical
within each task: Walker `1.45099154e-02`, G1 `2.07522389e-04`, LEAP `7.82382907e-04`.
LEAP sits between the other two, about 18x below Walker and 3.8x above G1.

### Verification against autodiff

`F_H` reaches the pathwise derivative only through Stein's lemma,
`E[u*tanh(mu+s*u)] = s*E[sech^2(y)]`, so it agrees in expectation rather than per sample.
Verified against JAX autodiff on the repo's own `distrax.Transformed(Normal, Tanh)`
log-prob, with the action clipped to `+-(1-1e-4)` exactly as `reppo.py:901` does. Without
that clip `atanh` overflows and the check returns NaN at large sigma; the first run of this
measurement did exactly that and the clip is the repo-faithful fix, not a tolerance change.

```
AUTODIFF_MATCH  max abs error = 8.986e-03   (alpha factored out)
convergence at mu=0, sigma=1.5:
  N=1e4  3.61e-02     N=1e6  1.20e-03
  N=1e5  6.98e-03     N=1e7  1.96e-04
```

The error falls as `1/sqrt(N)`, confirming the residual is Monte-Carlo convergence of the
Stein step and not a discrepancy in the formula.

## A1, A3. F_H on six baseline cells, seeds 301-308

On-arm = the states that cell's own policy visited, addressed by the bank's per-seed
`source` label, so `Walker PW_ent` seed 303 uses exactly the 192 states that policy
visited. Neutral = the full 3072-state bank, reported alongside.

| task | cell | subset | median F_H | mean F_H | frac>0 | med sigma | mean 2s^2E[sech^2] | class |
|---|---|---|---|---|---|---|---|---|
| walker | PW_ent | on_arm | +1.210e-02 | +9.993e-03 | 0.9489 | 0.4737 | 0.3113 | WIDENS |
| walker | PW_ent | neutral | +1.230e-02 | +1.087e-02 | 0.9779 | 0.4542 | 0.2507 | WIDENS |
| walker | WML_noent | on_arm | +1.424e-02 | **-3.997e-02** | 0.8862 | 0.2182 | 3.7545 | WIDENS |
| walker | WML_noent | neutral | -1.002e-01 | -7.196e+00 | 0.2828 | 7.1815 | 496.9071 | CONTRACTS |
| g1 | PW_ent | on_arm | +1.914e-04 | +1.758e-04 | 0.9982 | 0.2824 | 0.1527 | WIDENS |
| g1 | PW_ent | neutral | +1.962e-04 | +1.788e-04 | 0.9936 | 0.2991 | 0.1384 | WIDENS |
| g1 | WML_noent | on_arm | +2.030e-04 | +1.575e-04 | 0.9678 | 0.1887 | 0.2409 | WIDENS |
| g1 | WML_noent | neutral | +1.868e-04 | -9.996e-06 | 0.7844 | 0.5560 | 1.0482 | WIDENS |
| leap | PW_ent | on_arm | +7.472e-04 | +7.057e-04 | 0.9970 | 0.2297 | 0.0981 | WIDENS |
| leap | PW_ent | neutral | +6.224e-04 | **-1.691e-04** | 0.8141 | 0.5584 | 1.2161 | WIDENS |
| leap | WML_noent | on_arm | +7.694e-04 | +6.780e-04 | 0.9797 | 0.1228 | 0.1334 | WIDENS |
| leap | WML_noent | neutral | -1.749e-03 | -1.917e-01 | 0.3530 | 2.8974 | 245.9996 | CONTRACTS |

Classification is on **seed-median sign stability**: WIDENS if all 8 per-seed medians are
positive, CONTRACTS if all 8 are negative, MIXED otherwise. **No cell is MIXED** — every
cell's per-seed medians are the same sign, and remarkably tight:

```
walker  PW_ent     +1.21e-02 +1.18e-02 +1.17e-02 +1.22e-02 +1.23e-02 +1.18e-02 +1.21e-02 +1.24e-02
walker  WML_noent  +1.43e-02 +1.43e-02 +1.42e-02 +1.41e-02 +1.42e-02 +1.43e-02 +1.42e-02 +1.43e-02
g1      PW_ent     +1.92e-04 +1.92e-04 +1.93e-04 +1.89e-04 +1.91e-04 +1.89e-04 +1.92e-04 +1.93e-04
g1      WML_noent  +2.03e-04 +2.03e-04 +2.03e-04 +2.03e-04 +2.03e-04 +2.03e-04 +2.03e-04 +2.03e-04
leap    PW_ent     +7.62e-04 +7.45e-04 +6.74e-04 +7.33e-04 +7.56e-04 +7.52e-04 +7.49e-04 +7.60e-04
leap    WML_noent  +7.70e-04 +7.67e-04 +7.67e-04 +7.70e-04 +7.72e-04 +7.68e-04 +7.71e-04 +7.72e-04
```

Sign stability across seeds is therefore not the failure mode. The failure is elsewhere.

## A4. VALIDATION — the instrument's track record

F_H measured at a baseline checkpoint should predict the direction the entropy term pushes
width, and the observed transition should agree.

**On-arm, the measurement A1 specifies as primary:**

| task | cell | transition | observed | implies | measured | |
|---|---|---|---|---|---|---|
| walker | PW_ent | removing entropy | 0.455 -> 0.340 | narrower | WIDENS | MATCH |
| walker | WML_noent | adding entropy | 8.687 -> 2.058 | narrower | WIDENS | **MISS** |
| g1 | PW_ent | removing entropy | 0.300 -> 0.276 | narrower | WIDENS | MATCH |
| g1 | WML_noent | adding entropy | 0.497 -> 0.574 | wider | WIDENS | MATCH |

```
VALIDATION_4_OF_4 = NO   (3/4)
```

**Neutral, recorded for comparison: 4/4.** Walker WML_noent measures CONTRACTS there,
matching the observed contraction.

### Why it missed, and why that matters more than the miss

Two things separate the miss from the hits, and each on its own would flip the verdict.

**1. The state population.** Walker `WML_noent` has on-arm median sigma **0.2182** against
neutral **7.1815**, a factor of 33. That policy is narrow on the states it actually visits
and enormously wide off them. The four validation targets are **neutral-bank medians**
taken from the factorial reports, so the commensurate comparison is the neutral row. The
on-arm row is predicting the force on a different population from the one the target
measures.

**2. The summary statistic.** On-arm, Walker `WML_noent` has median F_H **+1.424e-02** but
mean F_H **-3.997e-02**, with 88.6% of state-coordinate pairs positive. A minority of very
wide coordinates carries strongly negative F_H and dominates the mean. The observed
contraction 8.687 -> 2.058 is driven by exactly those coordinates, which the mean captures
and the median does not. Classifying on the **mean** instead of the median gives **4/4 on
the on-arm measurement too**.

So there are at least two distinct ways to reach 4/4 — neutral states with the median, or
on-arm states with the mean — and **both would be chosen after seeing which one validates.**
That is estimator selection on the validation set, and adopting either without saying so
would make the instrument's track record meaningless. Neither is adopted here.

## A5. LEAP PREDICTION — NOT MADE

A4 instructs: if the instrument misses any of the four, say so plainly and do not proceed
to A5. It missed one. **No directional prediction for LEAP is preregistered.**

This is not a formality. The three defensible readings give three different LEAP
predictions:

| instrument variant | LEAP PW_noent | LEAP WML_ent |
|---|---|---|
| on-arm, median (the specified primary) | narrower | wider |
| on-arm, mean | narrower | wider |
| neutral, median | narrower | **narrower** |
| neutral, mean | **wider** | narrower |

The LEAP `WML_ent` direction flips between on-arm and neutral, and the LEAP `PW_noent`
direction flips between neutral-median and neutral-mean. Preregistering any one of these
would be picking the variant, not measuring the mechanism.

What the measurements themselves are, recorded as measurements and not as a prediction:

```
LEAP PW_ent     on_arm   median F_H +7.472e-04  mean +7.057e-04  frac>0 0.9970  seeds +8/-0
                neutral  median F_H +6.224e-04  mean -1.691e-04  frac>0 0.8141  seeds +8/-0
LEAP WML_noent  on_arm   median F_H +7.694e-04  mean +6.780e-04  frac>0 0.9797  seeds +8/-0
                neutral  median F_H -1.749e-03  mean -1.917e-01  frac>0 0.3530  seeds +0/-8
```

Per A5, no sign is inferred from median sigma. LEAP `WML_noent` illustrates why: its
neutral median sigma is 2.8974, far above `1/sqrt(2)`, yet its on-arm value is 0.1228, far
below. The threshold argument alone would give opposite answers on the two populations.

## A6. Return direction — post-hoc hypothesis, NOT derived

Kept explicitly separate from the width measurement above, which is derived.

Observed on two tasks: entropy raised WML return where it contracted an over-wide policy
(Walker, neutral sigma 8.69, **+16.7%**) and lowered it where the policy was already narrow
(G1, neutral sigma 0.497, **-20.0%**). Two points, fitted after the fact, with no mechanism
connecting width to return — indeed the G1 factorial found the two dissociate, pathwise
losing 41.6% of return with almost no width change.

LEAP `WML_noent` neutral median sigma is **2.8974**, between Walker's 8.69 and G1's 0.497
and nearer the low end on a log scale. The pattern, read literally, would put LEAP's WML
return change between the two and closer to G1's negative direction. **This is a post-hoc
hypothesis on n=2 and is not preregistered as a prediction.**

It is also expected to be untestable here: LEAP's baseline return gap is
`+4.30 [-1.34, +10.53]`, an interval containing zero, so the return half of a LEAP
factorial is unlikely to resolve anything.

## What would make the instrument usable

Stated so the choice is made before, not after, seeing LEAP:

* fix the state population and the summary statistic **in advance**, and justify them from
  the mechanism rather than from the validation score;
* the target quantity is a neutral-bank median, so if the instrument is to predict that
  target, the neutral bank is the commensurate population and the mean is the statistic
  that tracks a change dominated by the widest coordinates;
* validate the fixed choice on the four known outcomes, and report that it was fixed
  beforehand.

That is a decision for the principal, not something to settle by trying variants.
