# Why the error-channel crossover sits at `r = 1` and the update crossover does not

**Scope.** Controlled planted-error experiment only. `Q^pi` and `e` are known exactly by
construction. Nothing here is evidence about learned critics, about DeepMind Control
returns, or about Probe 4.

**Provenance.** Pre-registration `docs/prereg_planted_amplitude.md`, committed `4b88f03`
**before** either run. Scripts `scripts/planted/decompose_sweep.py`,
`amplitude_sweep.py`, `analyse_mechanism.py`, `make_mechanism_figure.py`. Artifacts
`reports/artifacts/planted_error_decomposition.csv` (240 cells, 101 columns),
`planted_amplitude.csv` (96 cells), `planted_amplitude_replicates.csv` (6144 blocks),
`fig_planted_mechanism.{pdf,png}` + `fig_planted_mechanism_data.csv`. CPU only, float64.

---

## 0. Answer, in one paragraph

`r = sigma*omega/sqrt(d)` locates the crossover on the **critic-error variance channel**
for the manuscript's centred estimators, sharply and robustly: refitted `r* = 1.009`
(`R^2 = 0.994`), `beta = -2.024` with 95% CI `[-2.046, -2.002]`, and `r*` within
`[0.986, 1.043]` across every `sigma` slice of the committed grid separately. It does **not** locate the
crossover of the policy update, and the reason is not that the smooth `Q^pi` channel is
hard to estimate. It is that **suppressing the error channel's variance does not
suppress the error channel's bias.** In the `1 < r < 1.67` band the E-step's
error-channel variance, measured against its own signal, is 7.5x smaller than pathwise's,
and its update error is still 1.32x larger. The decomposition shows why, in units that
are directly comparable because they are angular: pathwise loses **0.196 of its 0.212**
update error to error-channel *noise* — exactly the term `r` governs — while the E-step
loses **0.208 of its 0.293** to error-channel *bias*, a systematic displacement toward
the error field that `r` says nothing about and that the E-step's variance advantage does
not touch. The E-step's own error-channel noise contributes 0.006. On top of that it pays
an `A`-independent floor 23x pathwise's, which is mean-estimation noise (`ubar`), not the
cost of estimating `Q^pi`. Both extra terms scale differently from the one `r` controls,
so the operational boundary moves with the error amplitude (measured) and with the policy
scale `sigma` (measured), and `1.67` is neither universal nor a boundary.

---

## 1. Audit of the existing result

### 1.1 Reproduction

`decompose_sweep.py` re-runs the committed 240-cell grid with the original seed, the
original 8 x 4096 partition and the original RNG call order preserved exactly (including
the discarded `om_meas` draw, which must be kept or the stream diverges).

| column | max relative deviation from `planted_sweep.csv` |
|---|---|
| `ratio_e`, `var_pw_e`, `var_zo_e` | **0.000e+00** (bit-identical) |
| `mse_pw`, `mse_zo`, `mse_wml` | 4.8e-15, 3.2e-15, 3.4e-15 |

The committed result reproduces exactly.

### 1.2 Exact definitions, read from the implementation

| item | value in the code |
|---|---|
| `Q^pi(a)` | `-0.5 (a - a*)^T H (a - a*)`, `H = I`, `mu = 0`, `a* = G v_sig`, `G = 1` fixed for every `d` |
| `e(a)` | `A sin(omega v_e^T a + phase)`, `\|\|v_e\|\| = 1`, `phase ~ U[0, 2pi)` |
| amplitude `A_0` | `1.0` (`planted_error.eps`) |
| grid | `d in {2,4,8,16,32,64}` x `sigma in {0.05,0.1,0.2,0.4,0.8}` x `omega in {0.5,...,64}` = 240 cells |
| `M` | 32 (equal for all three operators; PW additionally uses `M` backward passes) |
| MC replicates | 8 direction blocks x 4096 batches = **32,768 batches per cell** |
| baseline | `Qbar` = sample mean of `Q_i` (centred ZO) |
| trust region | `Delta_mu = sqrt(2 eps_TR) Sigma ghat / \|\|ghat\|\|_Sigma`, `eps_TR = 0.1`, cancels from all ratios |
| target | `g* = grad Q^pi(0) = a*`, exact and blur-invariant (quadratic); excludes `e` entirely |
| `eta` | solved per batch from the MPO dual by 60-step bisection, bracket `[1e-4, 1e4]` |

**One discrepancy with the request.** The planted field is a **sine with a uniformly
random phase**, not `A cos(omega u^T a)`. Over the 8 direction blocks the two are
equivalent in distribution, but per block the phase is a live variable: the blurred error
gradient carries a `cos(phase)` factor, so the error channel's systematic component is
block-dependent and averages down across blocks. This matters for Sec. 2 and is stated
there.

### 1.3 "Error-induced variance" — the audit the request asked for

**The code computes the paired definition, not the subtraction.** In
`planted_error_sweep.py` the same `u` is used for the clean and contaminated critic and
the difference is formed before any variance is taken:

```python
pw0, zo0, wml0 = estimators(u, ..., with_error=False)
pw1, zo1, wml1 = estimators(u, ..., with_error=True)
dpw, dzo = pw1 - pw0, zo1 - zo0
acc["var_pw_e"].append(np.trace(np.cov(dpw.T)))
```

So the committed quantity is `Var[ghat(Q^pi + e; xi) - ghat(Q^pi; xi)]` under common
random numbers. The scalar measure is `V(ghat) = tr Cov(ghat) = E||ghat - E ghat||_2^2`,
as the request supposed.

Both definitions are now reported. Their difference is the cross term, and the identity
`V_sub - V_paired = 2 Cov(ghat(Q^pi), ghat(e))` holds to `1.4e-14`:

| operator | median \|`V_sub`/`V_paired` - 1\| | max | median \|cross\| / `V_paired` |
|---|---|---|---|
| PW | 1.42e-02 | 1.27e+01 | 1.42e-02 |
| ZO | 2.17e-02 | 2.98e+00 | 2.17e-02 |
| **WML (actual E-step)** | **8.42e-01** | 1.39e+00 | **8.42e-01** |

**Does the `r ~ 1` conclusion survive?** Yes, and the paired definition is what makes it
sharp:

| definition | cells with ZO better, `r < 1` | cells with ZO better, `r > 1` |
|---|---|---|
| paired (committed, primary) | **0 / 164** | **76 / 76** |
| subtraction | 12 / 164 | 76 / 76 |

The subtraction definition would have produced 12 misclassifications below the boundary.
The committed choice was the right one. For the E-step the two definitions differ by
84%, so any future statement about the E-step's error channel must say which is meant.

**Linearity, measured rather than assumed.** `max |paired difference - ghat(e)|` is
`1.6e-14` (PW) and `1.0e-14` (ZO) — the paired difference *is* `ghat(e)` — and `1.22`
for the E-step, which is nonlinear. This is what licenses the amplitude-invariance
statement in Sec. 3 for PW/ZO and forbids it for the E-step.

### 1.4 "Better update" is exactly a cosine

Every update is normalised to the same length, so the committed metric satisfies

```
Err = ||Delta_mu - Delta_mu*||^2_{Sigma^-1} = 4 eps_TR (1 - cos(ghat, g*))
```

verified to `1.7e-16` for all three operators. **The operational comparison is purely
directional; magnitude cannot contribute.** Any statement about "update MSE" in this
experiment is a statement about angle.

### 1.5 How the crossovers were estimated — and a correction

The committed report estimates every crossover by OLS of `log(ratio)` on `log r` over the
whole grid, then `r* = exp(-intercept/slope)`. Refitting:

| quantity | slope | `r*` | `R^2` |
|---|---|---|---|
| error-channel ZO/PW | -2.024 | 1.009 | **0.994** |
| error-channel WML/PW | -1.770 | 0.077 | 0.650 |
| operational ZO/PW | -0.261 | 1.028 | **0.500** |
| operational WML/PW | -0.261 | **1.668** | **0.514** |

The `1.67` reproduces, but it comes from a fit that explains half the variance. The
relation is not a power law, so the fitted line's crossing is a summary statistic, not a
boundary.

**New audit finding: the operational crossover is not `sigma`-invariant, and the
committed report never checked it.** Its non-collapse figure was run on the error-channel
ratio only. Fitting each `sigma` slice of the committed grid separately:

| `sigma` | 0.05 | 0.10 | 0.20 | 0.40 | 0.80 |
|---|---|---|---|---|---|
| `r*` error channel (ZO/PW) | 1.043 | 1.006 | 1.005 | 1.011 | 0.986 |
| **`r*` operational (WML/PW)** | **0.508** | **0.922** | **1.535** | **1.893** | **2.677** |

The error-channel crossover is flat to 6%. The operational one moves by **5.3x across
the committed grid's own `sigma` axis.** `1.67` is the pooled average of a quantity that
is not constant over the population it was pooled from.

---

## 2. Decomposition: what actually happens in `1 < r < 1.67`

26 cells of the committed grid fall in the band, spanning all six dimensions. The
hypothesis put to test was: (1) the E-step has lower variance on the planted-error
component; (2) it still has larger total update error; (3) the remainder comes primarily
from estimating the smooth `Q^pi` component.

**(1) TRUE, 26/26 cells.** Median raw `Var_e[E-step] / Var_e[PW] = 0.0035`.

**A units caveat that applies to this ratio throughout.** `Delta_mu_WML` is a
*displacement* and `ghat_PW` a *gradient*, so their variances are not in the same units
and the raw ratio carries an arbitrary scale — measured, it picks up roughly `sigma^2`
(regressing `log` of the raw ratio on `log sigma` within `r`-bins of the committed grid
gives slopes of +1.6 to +3.1, against +2 for a pure unit effect). The number 0.0035
therefore should not be read as "285x better". The dimensionless form divides each
operator's error-channel variance by its own clean-signal magnitude
`|E[ghat(Q^pi)]|^2`; band medians:

| operator | `Var_e / \|E ghat(Q^pi)\|^2` | vs PW |
|---|---|---|
| PW | 3.994 | 1.00 |
| ZO | 3.210 | 0.79 |
| **E-step** | **0.476** | **0.13** |

So the defensible statement is that the E-step's error channel is **7.5x smaller
relative to its own signal**, and conclusion (1) holds under both forms.

**(2) TRUE, 26/26 cells.** Median `Err[E-step] / Err[PW] = 1.32`.

**(3) FALSE.** The counterfactual attribution (prereg Sec. 2.7; medians over the band,
`Err` in the units of Sec. 1.4):

| operator | floor, error field off | + error-channel **mean** | + error-channel **noise** | = total | [clean-noise effect] |
|---|---|---|---|---|---|
| PW | 0.0033 | +0.0126 | **+0.1958** | 0.2118 | +0.0179 |
| ZO | 0.0616 | +0.0518 | +0.1503 | 0.2637 | +0.0452 |
| **E-step** | **0.0793** | **+0.2082** | +0.0056 | 0.2931 | +0.0507 |

**The two operators fail in different ways.** Pathwise loses 92% of its update error to
the error channel's *variance* — precisely the term Claim 4 bounds and precisely the term
the E-step suppresses. The E-step has almost no error-channel variance left (+0.0056) and
instead loses 71% of its error error to the error channel's *systematic mean*: a
displacement toward the structure of `e` that survives averaging.

Why the E-step and not pathwise: pathwise's error-channel mean is the blurred gradient
`A omega exp(-omega^2 sigma^2 / 2) cos(phase) v_e`, exponentially suppressed once
`omega*sigma` is large — which is exactly the `r > 1` regime. The E-step's weights
`softmax(Q/eta)` reweight samples toward high `Q`, and a planted bump in `Q` attracts
them; that mode-seeking displacement has no such suppression. Measured, in the band:

- `|E[Delta_mu_clean]| = 0.1713`, and its direction is essentially perfect,
  `cos(E[Delta_mu_clean], g*) = 0.9999` — absent critic error the E-step is not
  directionally biased;
- `|E[Delta_mu_err]| = 0.0782`, i.e. **the systematic error-channel displacement is 64%
  as large as the entire true-signal displacement.**

**There is also an `A`-independent floor, and it is not about `Q^pi` either.** With the
error field switched off entirely:

| region | `Err` floor PW | ZO | E-step | E-step / PW |
|---|---|---|---|---|
| `r < 1` (164 cells) | 0.0017 | 0.0848 | 0.1289 | **63.6x** |
| `1 < r < 1.67` (26) | 0.0033 | 0.0616 | 0.0793 | **23.0x** |
| `r > 1.67` (50) | 0.0055 | 0.0555 | 0.0578 | 9.2x |

and its `sigma`-dependence identifies it:

| `sigma` | 0.05 | 0.10 | 0.20 | 0.40 | 0.80 |
|---|---|---|---|---|---|
| PW floor | 0.0002 | 0.0007 | 0.0027 | 0.0106 | 0.0380 |
| E-step floor | 0.1026 | 0.0999 | 0.1013 | 0.1233 | 0.1841 |

Pathwise's floor scales as `sigma^2`; the E-step's is nearly flat. That is the signature
of the live `ubar` term the committed report identified in its Sec. 5 — the E-step moves
the mean to the weighted *sample* mean, so it is displaced by `Sigma^{1/2} ubar` even
with uniform weights, and the *direction* of `sigma*ubar` does not depend on `sigma`. So
the `A`-independent floor is mean-estimation noise, **not** the cost of estimating the
smooth `Q^pi`.

**Post-normalisation decomposition** (exact on the unit sphere,
`E||uhat - uhat*||^2 = ||E uhat - uhat*||^2 + E||uhat - E uhat||^2`; band medians):

| operator | `E\|\|uhat-uhat*\|\|^2` | bias² | variance | variance share |
|---|---|---|---|---|
| PW | 1.0588 | 0.3400 | 0.6499 | 61.4% |
| ZO | 1.3185 | 0.4935 | 0.8435 | 64.0% |
| E-step | 1.4655 | 0.5843 | 0.8881 | 60.6% |

Normalisation does not preserve the pre-normalisation ordering: pre-normalisation the
E-step holds a 7.5x advantage on the error channel relative to its own signal, and
post-normalisation it is nonetheless the worst of the three on every term — bias²,
variance and total.

---

## 3. The amplitude experiment

Design, bands and bootstrap fixed in `docs/prereg_planted_amplitude.md` before the run:
`A in {0.25, 1.0, 4.0}`, `d in {4,16,64}` at `sigma = 0.4` (plus a `sigma = 0.2` check at
`d = 16`), `r in {0.5,...,3.0}` with `omega = r sqrt(d)/sigma`, 64 direction blocks x 512
batches (the committed 32,768-batch budget re-partitioned), common random numbers across
amplitudes, bootstrap `B = 10,000` over blocks, seed 20260903.

**Pathology criteria (prereg Sec. 6) — all clear.** Min median ESS 13.94 against the
registered floor of 2; min single-batch ESS 10.56; `eta` never reached a bracket bound;
zero non-finite values; `omega` sup-norm calibration exact to 1.1e-16; CRN check exact
(clean-channel columns bit-identical across amplitudes, spread 0.000e+00).

### 3.1 Implementation check (not a finding)

As registered in prereg Sec. 1: the centred ZO and PW estimators are linear in `Q`, so
`ghat(e_A) = A ghat(e_1)` and the ZO/PW error-channel ratio must be exactly
amplitude-invariant. Measured `r*_var = 0.9934` at all three amplitudes, **max relative
spread 3.35e-16**, slope -2.031, `R^2 = 0.9993`. The implementation does what the algebra
says. This is reported as a check and carries no evidential weight.

### 3.2 H1 — NOT SUPPORTED

`r*_var(A)` for the **E-step**, the only place H1 has content:

| `A` | `r*_var` | 95% CI | in `[0.8, 1.25]`? |
|---|---|---|---|
| `A_0/4` | 0.326 | [0.321, 0.331] | **no** |
| `A_0` | 0.218 | [0.212, 0.223] | **no** |
| `4A_0` | 0.080 | [0.077, 0.082] | **no** |

**The registered quantity is unit-inconsistent (Sec. 2), and that is a flaw in the
pre-registration, not in the run.** The verdict is reported as registered above; the
dimensionless form, computed post-hoc and labelled as such in the analysis output, is the
one to read:

| `A` | `r*_var` dimensionless | 95% CI | in `[0.8, 1.25]`? | slope |
|---|---|---|---|---|
| `A_0/4` | 1.399 | [1.394, 1.403] | no | -2.023 |
| `A_0` | **0.876** | [0.867, 0.883] | **yes** | -2.116 |
| `4A_0` | <0.5 (extrapolated) | [0.317, 0.327] | no | -2.108 |

**H1 fails under both forms**, because it requires all three amplitudes to sit in the
band and they do not. But the dimensionless numbers say something the raw ones hide: at
the reference amplitude the E-step's error-channel boundary really does sit near the
theoretical one (0.876), and **amplitude moves it away in both directions, by a factor of
at least 4 across the range tested.** The `r`-scaling itself is intact — the fitted slope
is -2.02 to -2.12 in every case, matching the linear pair's -2.03. What amplitude changes
is the constant, and therefore the crossing.

None of this rests on extrapolation. At the lowest sampled `r = 0.5` the E-step is
already ahead on the raw error-channel comparison at every amplitude (0.482, 0.171,
0.019), and on the dimensionless one it is ahead at `4A_0` (0.477) and behind at
`A_0/4` (8.03) — the amplitude ordering is visible inside the sampled range.

### 3.3 H2 — not supported as registered; the direction is nonetheless decisive

| `A` | `r*_op` (E-step vs PW) | 95% CI |
|---|---|---|
| `A_0/4` | >3.0 (extrapolated) | [9.28, 10.15] |
| `A_0` | >3.0 (extrapolated) | [2.99, 3.12] |
| `4A_0` | >3.0 (extrapolated) | [3.33, 3.67] |

The ordering `r*_op(4A_0) < r*_op(A_0/4)` holds with disjoint intervals, but **both point
estimates fall outside the sampled range `[0.5, 3.0]`**, and prereg Sec. 4 forbids using
an extrapolated bound to satisfy a decision band. **H2 is therefore recorded as not
supported as registered.**

The registered fallback is descriptive reporting, and it is unambiguous. The operational
gap `Err[E-step] - Err[PW]`, entirely in-range, shrinks with amplitude at **every** `r`
with disjoint 95% intervals:

| `r` | gap at `A_0/4` | gap at `4A_0` | disjoint |
|---|---|---|---|
| 0.50 | 0.1507 [0.1501, 0.1513] | 0.0799 [0.0751, 0.0852] | yes |
| 1.00 | 0.1415 [0.1410, 0.1420] | 0.0728 [0.0699, 0.0755] | yes |
| 1.50 | 0.1263 [0.1259, 0.1267] | 0.0439 [0.0421, 0.0456] | yes |
| 2.00 | 0.1116 [0.1111, 0.1121] | 0.0240 [0.0217, 0.0262] | yes |
| 3.00 | 0.0844 [0.0837, 0.0851] | 0.0008 [-0.0012, 0.0029] | yes |

At `4A_0, r = 3` the gap has closed to within Monte-Carlo error of zero. **The amplitude
moves the operational comparison; it does not move the error-channel boundary.** That is
the intended contrast, obtained.

### 3.4 The `sigma` check, and a second mover

Registered as a confound check at `d = 16`, it found a second variable. The E-step's
operational ratio at `r = 1`, `A_0`, is 1.56 at `sigma = 0.2` against 2.15 at
`sigma = 0.4`; the same reordering appears at all three amplitudes. Together with
Sec. 1.5 this makes the point twice over: **the operational comparison depends on
`sigma` separately from `r`, in the new data and in the committed grid.**

By contrast the ZO/PW error-channel ratio at `d = 16` is 3.855 (`sigma = 0.2`) against
3.900 (`sigma = 0.4`) at `r = 0.5`, and 0.974 against 0.972 at `r = 1` — `r` collapses it
to within 1%. Both quantities compared there are gradients, so that comparison is
unit-consistent, and so is the operational one above (it is angular). **The E-step's raw
error-channel ratio must not be compared across `sigma`**, for the reason in Sec. 2: the
`sigma^2` unit factor accounts for roughly 4x of the ~9x difference between the two
`sigma` levels, and the remainder is not cleanly separable. The `sigma` conclusion rests
on the operational and ZO/PW numbers, which are clean.

### 3.5 Mismatch region

Cells where the E-step wins the error channel but loses the update:

| `A` | cells | fraction | 95% CI | `r` span |
|---|---|---|---|---|
| `A_0/4` | 24/24 | 100.0% | [100%, 100%] | [0.50, 3.00] |
| `A_0` | 22/24 | 91.7% | [91.7%, 95.8%] | [0.50, 3.00] |
| `4A_0` | 23/24 | 95.8% | [91.7%, 95.8%] | [0.50, 3.00] |

The mismatch is not a narrow band between two boundaries. Over the whole region sampled
at `sigma = 0.4` the E-step wins the channel Claim 4 is about and loses the update, and
what amplitude changes is the *size* of the operational deficit, not the extent of the
region.

The first criterion uses the raw `Var_e` comparison, whose threshold inherits the units
caveat of Sec. 2, so the region was recomputed under the dimensionless criterion:

| `A` | mismatch cells, raw | dimensionless | `r` span, dimensionless |
|---|---|---|---|
| `A_0/4` | 24/24 | 12/24 | [1.50, 3.00] |
| `A_0` | 22/24 | 16/24 | [0.75, 3.00] |
| `4A_0` | 23/24 | 23/24 | [0.50, 3.00] |

Under the stricter, dimensionally sound criterion the mismatch region **grows with
amplitude** — 12, 16, 23 cells — because raising `A` moves the E-step's error-channel
crossing to smaller `r` while the operational crossing stays beyond the sampled range.
This is the same effect as Sec. 3.3 seen from the other side, and it is the sharper
statement of the two: the more the critic error matters, the wider the region in which
winning the error channel fails to win the update.

---

## 4. The `r^-2` statement, quantified

Population: the committed 240-cell grid, error-channel ratio ZO/PW. Fit
`log(ratio) = alpha + beta log r`. Bootstrap over cells, 2000 resamples.

| population | n | `beta` | 95% CI | `R^2` | `r*` |
|---|---|---|---|---|---|
| all cells | 240 | **-2.0239** | [-2.0464, -2.0023] | **0.9937** | 1.0088 |
| `r in [0.5, 2]` | 55 | -2.0980 | [-2.1866, -2.0239] | 0.9862 | 1.0037 |
| `r in [0.25, 4]` | 106 | -2.1060 | [-2.1667, -2.0516] | 0.9877 | 1.0146 |
| `r in [0.8, 1.25]` | 28 | -1.9365 | [-2.0301, -1.8320] | 0.9816 | 1.0007 |

`beta` is within 0.11 of -2 on every population and the interval excludes -2 on two of
the four, so the agreement is close but not exact and should be reported as such.

**Language.** Claim 4 supplies two **upper bounds**,
`Var[g_PW]_e <~ (1/M)||grad e||_inf^2` and `Var[g_ZO]_e <~ (d/M)||e||_inf^2/sigma^2`.
Their ratio is `r^-2`. A ratio of upper bounds does not imply the ratio of the bounded
quantities, so the measured `beta ~ -2` is an **empirical property of this construction**
— where a single sinusoid attains both sup-norms exactly — and not a mathematical
consequence of the claim. The correct phrasing is that the measurement is *consistent
with* the bounds' ratio, not that the claim predicts it.

---

## 5. The figure

`reports/artifacts/fig_planted_mechanism.{pdf,png}`, source data
`fig_planted_mechanism_data.csv`. Panel A carries two operators because the contrast is
the result: the manuscript's `g_ZO`, one amplitude-invariant curve crossing 1 at
`r = 0.99`; and the E-step, three amplitude-ordered curves already below 1 at the left
edge. Panel B shows the operational gap, three separated curves approaching zero at
different rates. Bands are 95% bootstrap intervals over direction blocks; marker shape
encodes `d`, so identity is never carried by colour alone.

---

## 6. Reviewer-style assessment

**What exactly does the planted experiment establish?** That in a construction satisfying
Claim 4's assumptions exactly — quadratic `Q^pi`, a single planted sinusoid whose
sup-norms are attained analytically so `omega` is exact — the ratio of error-induced
variances between the manuscript's two centred estimators crosses 1 at
`r = sigma*omega/sqrt(d) = 1`, sharply (0 misclassifications in 240 cells), with
`beta = -2.024 [-2.046, -2.002]`, and with a per-slice `r*` in [0.97, 1.13] over 19
independent `d`/`sigma`/`omega` slices. It establishes nothing about learned critics,
returns, or any operator other than the two it measures.

**Is the `r = 1` transition robust?** For the manuscript's centred `g_ZO`, yes — it is
the most robust thing in the study. It is invariant across `d`, `sigma` and `omega`
slices, invariant to error amplitude (exactly, by linearity), and unchanged under either
variance definition on the `r > 1` side. For the **actual E-step it is not invariant**:
in the dimensionless form its error-channel crossing is 1.40 / 0.88 / <0.5 at
`A_0/4` / `A_0` / `4A_0`. It happens to sit near 1 at the reference amplitude, which is
worth saying plainly rather than dressing up — but it is a coincidence of `A_0`, not a
boundary, and it moves by at least 4x over the amplitude range tested.

**Why is the operational crossover different?** Because the update error contains two
large terms that `r` does not govern. The error channel's *bias* — a mode-seeking
displacement toward the planted field, contributing +0.208 of the E-step's 0.293 update
error against +0.006 from its error-channel noise — and an `A`-independent floor from the
live `ubar` mean-estimation term, 23x pathwise's in the band. Claim 4 bounds a variance;
these are a bias and a term with no critic error in it at all.

**Is `1.67` universal or configuration-dependent?** Configuration-dependent, and not
really a boundary. It is the crossing of an OLS line with `R^2 = 0.514` fitted over a
range where the relation is not a power law, and refitting the committed grid's own
`sigma` slices separately gives 0.51 / 0.92 / 1.54 / 1.89 / 2.68 — a 5.3x spread. It
should not appear in the paper as a number without those qualifications.

**Does amplitude explain the separation?** It explains part of it, and demonstrates the
mechanism. The operational gap shrinks monotonically with amplitude at every sampled `r`
with disjoint intervals, closing to within Monte-Carlo error of zero at `4A_0, r = 3`,
while the error-channel boundary does not move at all for the linear pair. But amplitude
is not the only mover: `sigma` shifts the operational crossover by 5.3x at fixed `r`.
The general statement the data support is that the operational comparison depends on the
relative strength of the true value signal and the critic error, of which amplitude is
one control and `sigma` another.

**What is the strongest remaining objection?** That the error field is a **single-mode
sinusoid with one random direction**, and both headline results may be artifacts of that
choice. Specifically: (i) the `r^-2` agreement is expected to be tightest exactly when a
single mode attains both sup-norms, which is the case constructed here; (ii) more
importantly, the mode-seeking bias identified in Sec. 2 is the mechanism most likely to
be special to a single coherent bump — with `J` superposed modes at random directions the
softmax is pulled in competing directions and the systematic displacement should partially
cancel, whereas the error-channel *variance* has no reason to change. If that cancellation
is large, the operational crossover would move toward `r = 1` for reasons unrelated to
amplitude, and the paper's central distinction would be weaker than stated.

The committed config already registers a `J = 4` secondary arm
(`planted_error.secondary`) that was never run. **Per instruction, no follow-up
experiment was run here.** The minimal one that would close this objection is that
`J = 4` arm restricted to the amplitude design's `(d, r)` cells at `A_0` — 24 cells, the
same budget, no new machinery — reporting only whether `|E[Delta_mu_err]|` and the
operational gap fall relative to `J = 1`. It is a one-run check, not a sweep.

---

## 7. What this does not establish

1. Nothing about learned critics: `omega` is not identifiable for a trained network, and
   `e` here is planted, not learned.
2. Nothing about returns, DMC, or the 64-run experiment.
3. `Q^pi` is quadratic and globally smooth; the oracle is exact only because of that.
4. The operational metric is purely directional (Sec. 1.4). A result about update
   *magnitude* would need a different metric.
5. `sigma < 0.1` is out of reach for the DMC arms (`min_std = 0.1`, hardcoded at
   `src/networks/jax_models.py:336`), and two of the five `sigma` slices behind the
   `r*_op` spread in Sec. 1.5 lie at or below it.
