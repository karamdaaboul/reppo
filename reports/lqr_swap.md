# Where pathwise and zeroth order swap: error size, dimension, and a blind test

**Four of the five registered predictions are confirmed and one is refuted. The
component model predicts the one-step swap to within 3.2 per cent at every error size,
and the whole-path swap scales as the square root of the action dimension. What fails is
the assumption joining the two: the one-step swap is not the path swap, and at a small
critic error it is more than four times too high.**

Registered in `docs/prereg_lqr_swap.md`. Symbols, defined once. `d` is the action
dimension. `M = 32` is the number of action samples per estimator call. `sigma` is the
policy standard deviation. `omega` is the spatial frequency of the planted critic error
and `eps` is its amplitude. `m = eps / eps_study` is the error size, as a multiple of the
crossover study's amplitude. `PW` is the pathwise estimator and `ZO` the centred
zeroth-order estimator with the `M/(M-1)` de-attenuation. `g*` is the exact gradient of
the Gaussian-blurred critic. The swap point is the `sigma*omega` at which `ZO`'s error
equals `PW`'s. `J` is a path's mean distance from the noise-free reference path in units
of `sigma`, and `rho` is the ratio of the median `J` over 100 seeds, `ZO` over `PW`.

---

## Verdicts

**Graph A, P1a: CONFIRMED.** The measured swap falls at every step, 49.91, 16.56, 5.19,
2.25, 1.566, 1.493, and consecutive 95 per cent intervals are disjoint.

**Graph A, P1b: CONFIRMED.** `swap(300) / tie = 1.4925 / 1.4711 = 1.0146`, interval
[1.0053, 1.0240], inside the registered [0.90, 1.10].

**Graph A, P3a, blind: CONFIRMED.** The measured swap is within 3.2 per cent of the
prediction at every error size, and every interval lies inside [0.90, 1.10].

**Graph B, P2: CONFIRMED.** The slope of `log(path swap)` on `log d` is **0.5180**,
interval [0.4393, 0.5455], inside the registered [0.35, 0.65].

**Graph C, P3b, blind: REFUTED.** At half the predicted swap `rho = 0.4461`, interval
[0.4164, 0.4830]. The prediction was `rho > 1`, and the interval excludes 1 in the other
direction. At twice the prediction `rho = 0.2729`, which is the predicted side.

---

## Graph A: the swap against error size, at `d = 2`

**One-step mean squared error, `10^4` replicates on an independent draw block.**

| `m` | predicted | measured | ratio | 95 % interval of the measured swap |
|---|---|---|---|---|
| 1 | 49.3021 | 49.9113 | 1.0124 | [49.0285, 50.5905] |
| 3 | 16.3659 | 16.5561 | 1.0116 | [16.3436, 16.7678] |
| 10 | 5.0903 | 5.1862 | 1.0188 | [5.1265, 5.2448] |
| 30 | 2.1762 | 2.2455 | 1.0318 | [2.2114, 2.2827] |
| 100 | 1.5467 | 1.5659 | 1.0124 | [1.5509, 1.5822] |
| 300 | 1.4793 | 1.4925 | 1.0089 | [1.4790, 1.5064] |

The swap falls by a factor of 33 as the error grows by a factor of 300, and it settles
onto the error-only tie rather than continuing to fall. Dropping the cross term costs
between 0.9 and 3.2 per cent, so the model is a good description of the one-step
comparison at this state. P1a is adjudicated conservatively: the marginal intervals of
consecutive error sizes are disjoint by a wide margin, which implies the paired ratio
excludes 1 at least as strongly.

**The angular bridge.** Measured on this study's own draws with the path study's own
crossing rule, the angular swap is 39.93, 4.218 and 1.943 at `m = 1`, `10` and `30`,
against that study's 39.985, 4.1721 and 1.9345. The agreement is 0.1, 1.1 and 0.5 per
cent, so the two harnesses agree and the gap between the angular and the squared-error
swap, which is a factor near 1.2, is a difference of metric and not of implementation.

**Figure** `reports/figures/lqr_swap_vs_error.pdf`.

---

## Graph B: the whole-path swap against dimension, at `m = 30`

| `d` | predicted one-step swap | measured path swap | 95 % interval | path over predicted |
|---|---|---|---|---|
| 2 | 2.1762 | 1.8442 | [1.6265, 2.6866] | 0.847 |
| 4 | 2.9282 | 2.2573 | [2.0621, 2.3842] | 0.771 |
| 8 | 4.0557 | 3.2605 | [3.1652, 3.3622] | 0.804 |
| 16 | 5.6060 | 4.8464 | [4.7543, 4.9703] | 0.864 |
| 32 | 7.9731 | 7.0891 | [6.9766, 7.2025] | 0.889 |
| 64 | 11.0710 | 10.5874 | [10.4112, 10.7037] | 0.956 |

The exponent is 0.5180 with interval [0.4393, 0.5455], so the path swap carries the same
square-root scaling the error-only tie does. The path swap sits below the one-step swap
at every dimension, by 15 per cent at `d = 2` and by 4 per cent at `d = 64`, so the two
converge as the dimension grows. The interval at `d = 2` is much wider than the others
because the ratio curve there is shallow and not monotone at its first two points.

**Figure** `reports/figures/lqr_swap_vs_dim.pdf`.

---

## Graph C: the blind path test at `d = 2`, `m = 3`

| `sigma*omega` | ratio to the prediction | `rho` | 95 % interval |
|---|---|---|---|
| 4.0915 | 0.25 | 0.6894 | [0.6663, 0.7250] |
| 5.7862 | 0.354 | 0.5582 | [0.5303, 0.5978] |
| **8.1830** | **0.5** | **0.4461** | **[0.4164, 0.4830]** |
| 11.5725 | 0.707 | 0.4025 | [0.3840, 0.4185] |
| 16.3659 | 1 | 0.3316 | [0.3183, 0.3488] |
| 23.1449 | 1.414 | 0.2949 | [0.2797, 0.3158] |
| **32.7319** | **2** | **0.2729** | **[0.2606, 0.2878]** |
| 46.2899 | 2.83 | 0.2354 | [0.2247, 0.2538] |
| 65.4638 | 4 | 0.2144 | [0.2068, 0.2288] |

`rho` is below 1 at every point on the grid, including the lowest, which sits at a
quarter of the prediction. The path swap at `m = 3` is therefore below 4.09, while the
one-step swap is 16.37, a factor of more than four. The mechanism is the one the path
study's fast panel showed: in a fast error field the pathwise arm does not merely take
noisier steps, it stalls, and a stall is a path-level failure that a one-step mean
squared error at the starting point cannot see.

**Figure** `reports/figures/lqr_blind_test.pdf`.

---

## The cross term, opened only after the verdicts were written

`results/lqr_swap/crossterm.json` was committed unread in the blind commit. At each
swap, the dropped term `2 m C` as a share of that arm's total error:

| `d` | `m` | swap | `PW` | `ZO` |
|---|---|---|---|---|
| 2 | 1 | 49.30 | 0.0015 | 0.0005 |
| 2 | 10 | 5.09 | 0.0011 | 0.0004 |
| 2 | 30 | 2.18 | 0.0083 | 0.0008 |
| 2 | 100 | 1.55 | 0.0153 | 0.0055 |
| 2 | 300 | 1.48 | 0.0066 | 0.0066 |
| 4 | 30 | 2.93 | 0.0031 | 0.0116 |
| 64 | 30 | 11.07 | 0.0000 | 0.0003 |

The dropped term never exceeds 1.6 per cent of either arm's total error at any swap, so
it explains the 0.9 to 3.2 per cent miss in Graph A and it explains nothing about Graph
C. The crossover study's finding that the cross term can be large at `d <= 4` was
measured on the error channel alone and at the error-only tie. At the total-error swap,
which is where this study evaluates it, the smooth part dominates the denominator and
the share is small.

---

## Gate G1, by dimension and arm

The gate is a maximum over many coordinates, so a value near its bound of four is partly
what a maximum of that many standard normals does on its own. The mean of `z^2` is the
calibration statistic a maximum cannot give. It is 1 when the arm is unbiased and the
standard error is right, whatever the number of coordinates.

| `d` | arm | max abs z at `10^4` | where | mean `z^2` | count | max abs z at 4000 |
|---|---|---|---|---|---|---|
| 2 | PW | 1.90 | swap30, coord 1 | 1.708 | 6 | 0.63 |
| 2 | ZO | 2.76 | swap30, coord 1 | 2.032 | 6 | 2.22 |
| 4 | PW | 1.91 | tie, coord 1 | 1.024 | 12 | 1.96 |
| 4 | ZO | 1.84 | tie, coord 1 | 1.224 | 12 | 2.12 |
| 8 | PW | 2.08 | eps = 0, coord 2 | 0.800 | 24 | 2.34 |
| 8 | ZO | 1.73 | tie, coord 6 | 0.552 | 24 | 2.22 |
| 16 | PW | 2.73 | eps = 0, coord 8 | 1.343 | 48 | 3.15 |
| 16 | ZO | 2.05 | tie, coord 15 | 0.784 | 48 | 2.18 |
| 32 | PW | 2.42 | swap30, coord 24 | 0.822 | 96 | 2.45 |
| 32 | ZO | 2.71 | tie, coord 13 | 1.101 | 96 | 2.46 |
| 64 | PW | 2.77 | swap30, coord 16 | 1.008 | 192 | 3.79 |
| 64 | ZO | 2.67 | eps = 0, coord 41 | 0.965 | 192 | 3.32 |

**A deviation, reported.** The gate as executed used 4000 replicates and the
preregistration says `10^4`. The table gives both. At the registered count the largest
value anywhere is 2.77 against the bound of 4, so the gate passes at its registered
strength and the deviation made it weaker rather than stronger. The 3.79 that the 4000
replicate run produced at `d = 64` fell to 2.77 at `10^4`, which is what a fluctuation
does and not what a bias does. The mean of `z^2` sits between 0.55 and 1.34 at every
`d >= 4`. At `d = 2` it is 1.7 and 2.0, on six values, where the sampling standard
deviation of the statistic is 0.58, so that is about one and a half standard deviations
and not evidence of bias.

---

## Which route the error-only tie uses, and the 4.4 per cent gap

**Graph A's dashed line is 1.4711.** That is the root of `V_ZO^e - V_PW^e` on the
registered 41 point grid `logspace(-1, 3, 41)`, at this study's single state, at the
fixed width `sigma = 0.13945`, from `10^4` replicates, with the crossing located by
linear interpolation of the difference against `log(sigma*omega)`. It is the number
committed in the blind commit and it is what P1b is adjudicated against.

The path study's value for the same state is 1.5387, which is 4.4 per cent higher. The
attribution, from `results/lqr_swap/diag.json`:

| route | value |
|---|---|
| registered, this study's 41 point grid | 1.4711 |
| the same statistic and the same draws on a 161 point grid | 1.5336, interval [1.5271, 1.5398] |
| the harness kernel at this state, collapsed onto `sigma*omega` (the path study's route) | 1.5387 |
| the same kernel data, the single width column nearest `sigma` | 1.5278 |

**The gap is grid resolution, not a difference of quantity.** Refining this study's own
grid from 41 to 161 points, on the same draws and the same statistic, moves the tie from
1.4711 to 1.5336 and closes 92 per cent of the distance to the path study's number. The
remaining 0.5 per cent is inside the Monte Carlo interval and matches the kernel's own
single-column value. The cause is that the difference being rooted is strongly curved in
`log(sigma*omega)`, because `PW`'s error channel grows as the square of the frequency
while `ZO`'s saturates, and the registered grid steps by a factor of 1.259, so a straight
line drawn across one step misplaces the root downward.

**Nothing here is substituted for the registered number.** P1b is confirmed against
1.4711, ratio 1.0146. Against the refined 1.5336 the ratio is 0.9732, which is also
inside [0.90, 1.10], so the verdict does not turn on the choice. The same coarse grid
underlies both the predicted and the measured swap in Graph A, and the bias is common to
the two, so the ratio that P3a is adjudicated on is largely free of it.

---

## Limitations

1. One state per dimension. Six systems and six states, one each, not a population.
2. `sigma` is fixed within each `d` and there is no covariance update, so this is the
   mean-only dynamics and not the algorithm.
3. The error is planted and `Q^pi` is quadratic. A learned critic is neither, and nothing
   here concerns learned critics.
4. `m = 30` is an illustration size, 150 per cent of the within-state value spread.
5. Graph B fits an exponent to six points, one per dimension, each a single state.
6. The refuted prediction P3b says the one-step swap is not the path swap at `m = 3` and
   `d = 2`. It does not say by how much in general, and the factor of four measured here
   is one point in a two-dimensional space of error size and dimension.

---

## Provenance

Preregistration `c90fbf5`. Amendment A0, which names the denominator of every
equivalence gate, `2d3411d`, decided before any outcome data existed. The blind commit of
the predictions and the sealed cross term, `31f0c90`. The code and this report are added
by the commit that carries them.

Run on the FZI workstation, host `karam`, CPU, float64, JAX 0.5.2, NumPy 2.5.1. Wall
time: gates 42 s, components 7 min 15 s, measure 1 min 59 s, diagnostics 2 min 30 s,
report and figures under a minute. All gates pass. `results/lqr_swap/` holds
`gates.json`, `components.json`, `components_full.json`, `crossterm.json`,
`graph_a.json`, `summary.json`, `diag.json` and `env.json`; `paths.npz` is git-ignored as
every `.npz` in this repository is, and its sha256 is recorded in `env.json`.
