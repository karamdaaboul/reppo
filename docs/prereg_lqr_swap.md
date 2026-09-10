# Pre-registration: where pathwise and zeroth order swap, against error size and dimension

**Status.** Committed before any run of this experiment. Sections 1 to 10 are
append-only. The blind predictions go below the amendment line in Section 11 and are
committed before any total-error sweep and before any path run.

**Revision history.** v1.0 (2026-09-10): as committed.

**Companions.** `docs/prereg_lqr_paths.md` (`4606164`, amendments A1 to A3 at `a874a8b`
and later) and `reports/lqr_paths.md` register and report the path study this extends.
`docs/prereg_lqr_crossover.md` (`d2cb9f6`) registers the harness. The canonical report of
that study is `reports/lqr_crossover_corrected.md` (`00091c4`), and it is the source of
every published number quoted here.

**Status of the path study's amendment A2.** Accepted, with the record tightened: the
three registered comparator failures now appear in the opening verdict of
`reports/lqr_paths.md` and not only in its gate section. This experiment is built so that
none of the three can recur. There is no E-step arm, so the conditioning of the softmax
displacement never enters. No difference of `lqr.q_pi` is ever formed. No quadrature is
used anywhere.

---

## 1. What this is

The path study confirmed four panels and left three limits. This experiment makes one
graph per limit.

1. Where the two estimators swap depends on how large the critic error is. At the study's
   error size the swap sits at `sigma*omega` near 40, and at thirty times that error it
   sits at 1.93. **Graph A** measures that dependence.
2. The path study covers `d = 2` only, so it says nothing about a `sqrt(d)` scaling.
   **Graph B** measures the whole-path swap against `d`.
3. The fast settings of the path study were placed by a calibration, so that test could
   hardly fail. **Graph C** predicts a swap from separately measured parts, commits the
   prediction, and then runs paths on both sides of it.

Symbols, defined once. `d` is the action dimension. `M = 32` is the number of action
samples per estimator call. `sigma` is the policy standard deviation, one scalar for all
coordinates. `omega` is the spatial frequency of the planted critic error, and it is the
dial. `eps` is the amplitude of that error. `eps_study` is the amplitude the crossover
study used, 5 per cent of the closed-form within-state spread of the action value.
`m = eps / eps_study` is the error size. `PW` is the pathwise estimator. `ZO` is the
centred zeroth-order estimator with the `M/(M-1)` de-attenuation. `g*` is the exact
gradient of the Gaussian-blurred critic, which both arms are unbiased for. The **swap
point** is the value of `sigma*omega` at which `ZO`'s error equals `PW`'s. `mu` is the
policy mean and `mu_0` is its starting value. `a*` is the exact maximiser of the action
value at the chosen state. `N` is the number of update steps and `R` is the number of
path seeds. The shipped E-step is not part of this study.

**What this is not.** It is not evidence about learned critics, because the error field
is planted. It is not evidence about return, because no environment is stepped. Graph B
is a statement about one state per dimension, not about a population of states.

---

## 2. What is imported, and the one refactor

Nothing below is reimplemented. The path study's own per-step primitives are imported,
which is what makes a number here comparable to a number there.

| object | file |
|---|---|
| `Setup`, `setup`, `at_omega`, `g_lin_np`, `make_q_of_u`, `arm_direction`, `g_star_np`, `reference_path`, `crossing`, `Gate`, `record_env`, `sha256_file`, `git_sha`, `prereg_sha` | `scripts/lqr_paths.py` |
| `build_system`, `sample_states`, `q_coeffs`, `q_spread_closed_form`, `grad_a_q_pi`, `q_of_u_factory` | `scripts/lqr_crossover/lqr.py` |
| `draw_error`, `PlantedError`, `e_value`, `blurred_e_grad`, `theta` | `scripts/lqr_crossover/error_field.py` |
| `both_from_shared_u`, and through it `whitened_pathwise`, `centred_zo`, `pathwise_mean`, `deattenuation_factor` | `scripts/lqr_crossover/estimators.py`, `src/jaxrl/estimators.py` |
| `run_d`, `crossover_by_c` (gate G3 only) | `scripts/lqr_crossover/sweep.py`, `analyze.py` |
| `c_star_asymptote` | `scripts/lqr_crossover/reference.py` |

**The refactor, registered here.** `scripts/lqr_paths.py` hardcodes its module constant
`D = 2` inside `setup`, `make_runner`, `reference_path` and `cal_block`. Those four are
parameterised by the dimension, with the default left at 2, so that every call the path
study made returns exactly what it returned before. Gate G0c below is the regression
check on that claim, and it compares arrays rather than trusting the argument.

**Two numerical rules, registered because the path study learned them the hard way.**
First, no difference of `lqr.q_pi` is ever formed. Every action value enters through the
reduced form documented in the header of `scripts/lqr_crossover/lqr.py`, which drops the
additive constant `gamma v/(1 - gamma)` analytically rather than cancelling it in
floating point. Second, every blurred quantity is taken from the harness's closed form,
which returns exactly zero when the blur factor `exp(-(sigma*omega)^2/2)` underflows. No
Gauss-Hermite quadrature is used anywhere in this experiment.

---

## 3. Shared setup

**System and state, at every `d`.** `lqr.build_system(d, seed=SEED_ROOT + d)` with the
study's shipped defaults, where `SEED_ROOT = 20260902`. One random stream,
`np.random.default_rng(SEED_ROOT + 2000 + d)`, is consumed in the study's order: first
`lqr.sample_states(sys, rng, 32)`, then
`error_field.draw_error(rng, 32, d, kind="full", omega=1.0)`. The state used is the first
index with `||a* - mu_0|| >= 1e-3`. This is the path study's rule, so at `d = 2` it
returns that study's state, its phases and its width.

**Start, target, width.** `mu_0 = -K s` in the study's normalised coordinates, where `K`
is the study's suboptimal gain and `s` the chosen state. `a* = mu_0 + 0.5 H^{-1} g`, in
closed form, where `H` and `g` are the first two return values of `lqr.q_coeffs`.
`sigma = ||a* - mu_0|| / 10` at each `d`, so `sigma` varies with `d` and is fixed within
each `d`. There is no covariance update anywhere.

**Error field.** Full rank, `k = d`, the study's `kind="full"` arm, with the harness's
orthonormal rows and the phases from the stream above. `eps_study` is
`0.05 * lqr.q_spread_closed_form(s, sigma)` at that `d`, computed once and held fixed
along every path.

**Update rule.** `mu_{t+1} = mu_t + 0.2 sigma g_hat / ||g_hat||`, `N = 80` steps, with
the single doubling to 160 that gate G2 may require. Every arm takes the same step
length. Only the direction comes from the estimator.

**Common random numbers.** At step `t` of seed `r`, both arms use the same draws, from
`fold_in(fold_in(PRNGKey(SEED_ROOT + 8000 + d), r), t)`. This is the path study's stream,
unchanged, so at `d = 2` the paths here and there share their randomness. The same seeds
`0` to `99` are used at every grid point of every graph.

**Seed offsets, new to this experiment.** One-step component draws:
`PRNGKey(SEED_ROOT + 10000 + d)`. One-step total-error measurement draws:
`PRNGKey(SEED_ROOT + 11000 + d)`, an independent block, so the measurement of Section 6
is not the same randomness the prediction of Section 5 was built from. Both blocks are
shared across every grid point and every `m`, which makes those curves paired.

---

## 4. Metrics

**One-step metric.** `MSE = E||g_hat - g*||^2` over `10^4` replicates at `mu_0`. Both arms
are unbiased for `g*`, so this is a trace of a covariance and not a bias-plus-variance
mixture. This differs from the angular metric the path study calibrated with. As a
bridge, the angular swap point is also reported at `m = 1`, `10` and `30`, computed with
the path study's own crossing function, beside that study's 39.985, 4.1721 and 1.9345.

**Swap on a grid.** The grid is `sigma*omega` in `numpy.logspace(-1, 3, 41)`. The swap is
the first grid point at which `ZO`'s metric falls below `PW`'s and stays below for the
next three points, located between that point and its predecessor by linear interpolation
of the difference against `log(sigma*omega)`. The persistence requirement is what stops a
single noisy cell from being read as a crossing. If no such point exists, the outcome is
reported as no crossing.

**Path metric.** `J` is the mean over steps of the distance from the noise-free reference
path, in units of `sigma`, exactly as in the path study.
`rho = median_r(J_ZO) / median_r(J_PW)` over the `R = 100` seeds.

**Path swap.** The value of `sigma*omega` at which `rho = 1`, by linear interpolation of
`log(rho)` against `log(sigma*omega)` between the bracketing grid points. If no pair of
grid points brackets `rho = 1`, the grid is extended once by a factor of four on the
needed side. If there is still no crossing, the outcome is reported as no crossing.

**Bootstrap.** Paired percentile, 10 000 resamples, `numpy.random.default_rng(20260902)`.
For the one-step swap the resampled unit is the replicate, drawn jointly across every
grid point and every `m`. For the path swap the resampled unit is the seed identifier,
drawn jointly across every grid point and every `d`, so that a resample is one coherent
redraw of the experiment. Resampling replicates is a statement about Monte Carlo
precision at one state. It is not an inferential claim about a population of states, and
the report says so.

---

## 5. The component model, and what the blind prediction is

Both operators are exactly linear in the critic, which the crossover study's gate G6d
asserts to `6.8e-14`. Write the critic as `Q_phi = Q^pi + m e`, where `e` is the planted
field at amplitude `eps_study`. Linearity gives
`g_hat(Q^pi + m e) = g_hat(Q^pi) + m g_hat(e)`, and the same split holds for the exact
target. Therefore, with `D_s = g_hat(Q^pi) - g*(Q^pi)` and `D_e = g_hat(e) - g*(e)`,

    MSE_X(sigma*omega, m) = V_X^s + m^2 V_X^e(sigma*omega) + 2 m C_X(sigma*omega)

    V_X^s = E||D_s||^2      the error with no waves, independent of omega
    V_X^e = E||D_e||^2      the error channel at m = 1
    C_X   = E<D_s, D_e>     the cross term

**This decomposition is exact.** The prediction is what becomes approximate: it drops
`C`. All three parts are estimated from one pass over shared draws at `eps_study`, at
every point of the 41 point grid, for both arms.

**Predicted swap(m):** the value of `sigma*omega` at which
`V_ZO^s + m^2 V_ZO^e = V_PW^s + m^2 V_PW^e`, found by the crossing rule of Section 4.

**Error-only tie:** the value of `sigma*omega` at which `V_ZO^e = V_PW^e`, by the same
rule. It does not depend on `m`.

**Why the swap must fall toward the tie.** `V_ZO^s` exceeds `V_PW^s`, because the
zeroth-order operator pays the classical dimension factor on the smooth part. So the
crossing needs `V_ZO^e < V_PW^e`, which happens only above the tie, and the required
margin shrinks as `m^2` grows. With the textbook forms this reads
`swap^2 = tie^2 + (K/m)^2`, where `K` measures the zeroth-order operator's extra noise on
the smooth part.

**What the test checks.** Whether dropping `C` matters at the swap. The crossover study
found the cross term is not negligible at `d <= 4` on the error channel
(`reports/lqr_crossover_corrected.md` Sec. 5.3), so this test can fail. The measured `C`
at each swap is computed at the same time as everything else, and it is reported only
after every verdict is written.

`V^s`, `V^e` and `C` are computed at every `d` used in Graph B.

---

## 6. Graphs, predictions and decision rules

### 6.1 Graph A: the one-step swap against error size, at `d = 2`

`m` in `{1, 3, 10, 30, 100, 300}`. The total one-step MSE of both arms is measured on the
41 point grid with the independent draw block of Section 3, and the measured swap is
extracted by the rule of Section 4.

**P1.** The swap falls as `m` grows, and `swap(300)` lies within 10 per cent of the
error-only tie.

- P1a is confirmed when the point estimates are strictly decreasing in `m` and the 95 per
  cent interval of every consecutive ratio `swap(m_{i+1}) / swap(m_i)` excludes 1 from
  below. It is refuted when any such interval excludes 1 from above. Otherwise it is
  inconclusive.
- P1b is confirmed when the 95 per cent interval of `swap(300) / tie` lies inside
  `[0.90, 1.10]`. It is refuted when that interval lies entirely outside. Otherwise it is
  inconclusive.

**P3a, blind.** For every `m`, `|measured swap / predicted swap - 1| <= 0.10`. Confirmed
when the 95 per cent interval of the ratio lies inside `[0.90, 1.10]` at every `m`.
Refuted when it lies entirely outside at any `m`. Otherwise inconclusive. The predicted
swap carries no interval: it is a number committed before the measurement exists.

**Figure** `reports/figures/lqr_swap_vs_error.pdf`. Horizontal axis `m`, logarithmic.
Vertical axis the swap point `sigma*omega`, logarithmic. Navy `#1F4E79` points with 95
per cent interval bars. One brick `#C0504D` dashed horizontal line at the error-only tie,
labelled "error only".

### 6.2 Graph B: the whole-path swap against dimension

`d` in `{2, 4, 8, 16, 32, 64}`, at `m = 30` everywhere. At each `d` the grid is nine
values of `sigma*omega`, log-spaced from `0.25` to `4` times the component-model swap at
that `d`, that is, half-octave steps. The grid is fixed by the blind commit, before any
path runs. Paths use seeds `0` to `99` for both arms.

**P2.** The least-squares slope of `log(path swap)` on `log d` lies in `[0.35, 0.65]`,
the band the crossover preregistration used for the same exponent. Confirmed when the
95 per cent bootstrap interval of the slope lies inside that band. Refuted when it lies
entirely outside. Otherwise inconclusive. The predicted swap and the measured path swap
are tabulated side by side at every `d`.

**Figure** `reports/figures/lqr_swap_vs_dim.pdf`. Horizontal axis `d`, logarithmic with
powers of two as ticks. Vertical axis the path swap, logarithmic. Navy points with 95 per
cent interval bars. One brick dashed line at `sqrt(d M/(M-1))`, labelled "sqrt d".

### 6.3 Graph C: the blind path test at `d = 2`, `m = 3`

Nine values of `sigma*omega`, log-spaced from `0.25` to `4` times the predicted swap at
`m = 3`, in half-octave steps, so the grid contains `0.5` and `2` times the prediction
exactly. Paths use seeds `0` to `99` for both arms.

**Registered deviation.** The brief asks for seven grid points, log-spaced from `0.25` to
`4` times the prediction, including `0.5` and `2` times it. That is not possible. A
log-spaced grid spanning four octaves contains the one-octave marks only when the number
of intervals is a multiple of four, which allows five, nine or thirteen points and never
seven. Nine is chosen, which also makes the Graph B and Graph C grids the same shape.

**P3b.** `rho > 1` at `0.5` times the predicted swap and `rho < 1` at `2` times it, with
both 95 per cent intervals excluding 1. Confirmed when both hold. Refuted when either
interval excludes 1 in the other direction. Otherwise inconclusive.

**Figure** `reports/figures/lqr_blind_test.pdf`. Horizontal axis `sigma*omega`,
logarithmic. Vertical axis `rho`, logarithmic. Navy points with 95 per cent intervals
joined by a thin line. A thin gray horizontal line at `rho = 1`. One brick dashed
vertical line at the predicted swap, labelled "predicted".

### 6.4 Figure style

matplotlib, PDF, 3.4 by 2.6 inches, serif font (STIX), one panel per figure, at most two
data elements, minimal labels, no titles.

---

## 7. Gates

Every gate runs before the stage it protects. Tolerances are declared here and are not
revisited. **If a registered gate fails, the experiment stops. No gate is amended and
continued.**

**G0a.** `PW` and `ZO`, as called by this experiment, reproduce the harness composition
`lqr.q_of_u_factory` plus `error_field.e_value` fed to `both_from_shared_u`, on 100 fixed
draws in float64, to a maximum relative deviation of `1e-13`. Run at every `d`.

**G0b.** The moving-mean closure's gradient matches the harness's exact action gradient.
For ten means along the segment from `mu_0` to `a*` and 100 draws, the vector-Jacobian
product of the closure with respect to the whitened draw must equal
`sigma * alpha_Q * lqr.grad_a_q_pi(sys, alpha_s s, mu + sigma u)` to `1e-13`. This is
constant free, so it is not the comparison that failed in the path study.

**G0c.** The dimension refactor of Section 2 changes nothing at `d = 2`. The refactored
`scripts/lqr_paths.py` must reproduce the committed path arrays of
`results/lqr_paths/paths_*.npz` bitwise, and the setup values `sigma`, `eps_study`,
`mu_0` and `a*` exactly.

**G1.** At `mu_0`, over `10^4` replicates, the mean of each arm matches `g*` within four
standard errors per coordinate. Run at every `d`, at `m = 0`, at the error-only tie, and
at the component-model swap for `m = 30`.

**G2.** The reference path reaches a stationary point, as in the path study: the last 20
points lie inside a ball of radius `sigma` centred at their mean, with one doubling of
`N` from 80 to 160 allowed. Evaluated at every grid point of Graphs B and C before any
path runs there. A grid point that fails at `N = 160` stops the experiment, and the
report carries the per-point tail radius so the cause is visible.

**G3.** At `d = 2`, the study's full-rank arm rerun through `sweep.run_d` returns
`c* = 1.522` within 2 per cent, the tolerance the path study registered for the same
check. The artifact from that run is reused if it is present and its sha256 matches;
otherwise the run is repeated. Reported beside it, as a descriptive cross-check and not a
gate: this experiment's own error-only tie at `d = 2`, which is the same quantity
estimated from `V^e` rather than from the sweep kernel, against the path study's
single-state value of 1.5387.

---

## 8. Compute and reproducibility

`scripts/lqr_swap.py` has five stages, run as
`python scripts/lqr_swap.py --stage <stage> --out results/lqr_swap`, with `<stage>` in
`gates`, `components`, `measure`, `figures`, `report`. The harness pins the process to
CPU at import and sets float64, so this runs on CPU on every machine. Seeds are mapped
with `jax.vmap` and the per-step update is compiled with `jax.jit`, with the frequency
and the amplitude passed as traced arguments so that a grid of frequencies costs one
compilation rather than nine.

At large `d` the one-step component pass holds `10^4` replicates times `M = 32` samples
times `d` coordinates. Replicates are accumulated in chunks of 1000. Chunking changes the
memory and not the estimator.

Runs happen on the FZI workstation, at the same commit as the working tree.
`results/lqr_swap/env.json` records hostname, device list, library versions, git hash,
the script's own sha256 and the wall time of every stage. Any `.npz` output is git-ignored
repository-wide, so a manifest with sha256 per file is written beside it.
`scripts/lqr_swap.sbatch` copies its header from `scripts/lqr_paths.sbatch`: account
`rwth2182`, partition `c23g`, one GPU that the harness's CPU pin leaves unused, and the
repository `.venv`.

---

## 9. Limitations, registered in advance

1. One state per dimension. Graph B's exponent is fitted across six systems and six
   states, one each, not across a population.
2. `sigma` is fixed within each `d` and there is no covariance update, so this is the
   mean-only dynamics and not the algorithm.
3. The error is planted and `Q^pi` is quadratic. A learned critic is neither.
4. `m = 30` is an illustration size. It is thirty times the study's amplitude, which is
   150 per cent of the within-state value spread.
5. The component model drops the cross term by construction. A confirmed P3a says the
   dropped term does not move the swap by more than a tenth. It does not say the term is
   zero.

---

## 10. What would make this experiment misleading

- Reading Graph B's exponent as a statement about learned critics, or as a replacement
  for the crossover study's rank ladder. It is the same `sqrt(d)` question asked of the
  whole path rather than of one step, at one rank and one error size.
- Comparing `J` across grid points or across `d`. `J` is a divergence from that setting's
  own reference path. Only `rho`, formed inside a setting, is interpreted.
- Quoting the predicted swap as a measurement. It is a model output, committed blind, and
  its whole purpose is to be wrong in a measurable way.
- Letting the measurement reuse the component draws, which would make P3a a test of
  arithmetic rather than of the dropped cross term. Section 3 fixes two independent
  blocks.
- Reporting the cross term before the verdicts. It is the explanation for a miss and it
  is written down only after the verdicts are.

---

## 11. Amendment line

**Everything above this line is append-only and was committed before any run of this
experiment. Everything below is appended after the stage it reports, and never edits
anything above.**

### A0. The equivalence gates name their denominator (2026-09-10)

**Decided by Karam before any outcome data of this experiment existed.** The gates stage
had run. The components stage had not, so no prediction, no sweep and no path existed
when this was written.

**What failed, as registered.** `G0a` passed at `d = 2` and failed at every larger `d`.
`G0b` failed at every `d`. The registered sentence asks the arms to reproduce the harness
"to a maximum relative deviation of `1e-13`" and does not say relative to what. Scored
against each component, the worst values were:

| gate | d = 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|
| G0a | 2.2e-15 | 2.6e-13 | 7.7e-13 | 1.7e-11 | 4.5e-12 | 4.2e-11 |
| G0b | 1.0e-11 | 1.2e-10 | 8.3e-12 | 6.3e-12 | 2.7e-10 | 1.3e-9 |

**What the measurement shows.** The deviation itself is flat in `d` and sits at float64
rounding. What grows is the denominator's collapse toward zero.

| quantity | G0a `d = 2` ZO | G0a `d = 64` PW | G0b `d = 2` | G0b `d = 64` |
|---|---|---|---|---|
| max absolute deviation | 4.4e-15 | 2.2e-15 | 4.4e-16 | 3.9e-16 |
| divided by the vector norm | 1.8e-15 | 2.6e-15 | 1.9e-13 | 4.0e-15 |
| divided by the component | 2.0e-15 | 4.0e-11 | 1.0e-11 | 1.3e-9 |
| smallest component over vector | 7.1e-2 | 7.1e-6 | 1.5e-3 | 1.5e-7 |

The componentwise ratio tracks one over the smallest component. At `d = 64` one
coordinate of the harness's own output is `7e-6` of the vector, so a `2e-15` rounding
difference reads as `4e-11`. The vector-norm denominator fails as well, at `d = 2`, where
one draw produces a near-zero gradient vector. Both denominators measure how close
something is to zero. Neither measures whether the code reproduces the harness.

**The decision.** The tolerance stays at `1e-13`. Each equivalence gate is scored as
`max|x - y| / max|y|`, both maxima taken over that gate's own test set, so no single
near-zero item can inflate it. All three denominators are printed and stored for every
gate. The registered failure stays visible in the gate table and in the opening lines of
`reports/lqr_swap.md`. Under the scored reading the measured values are `1.4e-15` to
`6.4e-15` at every `d` for both gates, which is 15 to 75 times inside the bound.

**Two gate-code crash fixes, listed here because they are changes to this experiment's
code and not to any criterion.** First, `lqr_paths.make_runner` returns a pair, and `G0c`
compared the pair rather than its first element, which produced a spurious deviation of
12 in action units. Second, `G0c` compared `sigma` against a literal transcribed to seven
digits at a tolerance of `1e-6`, which the true value `0.13945420870161135` fails by
`1.5e-6`. The gate now compares bitwise against the values stored in the committed path
artifacts, which is the correct reference. With both fixed, `G0c` passes: all four
committed `d = 2` panels reproduce bitwise, maximum absolute difference `0.0`.

**A convention for every future preregistration in this project.** An equivalence gate
must name its denominator, and that denominator must be scaled to the largest value in
the gate's test set. A relative tolerance with an unnamed denominator is not a criterion.
This is the second study in a row in which one met a near-zero denominator.

---

### A1. The blind commit (to be appended after the gates and components stages)

To be filled, before any total-error sweep and before any path run, with: the predicted
`swap(m)` at all six values of `m`; the error-only tie; the nine point grid for Graph C;
the nine point grid at each of the six dimensions for Graph B; the setup values at every
`d`; and the gate table.
