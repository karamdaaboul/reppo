# Pre-registration: planted-error amplitude experiment

**Committed before the experiment is executed.** Design, hypotheses, decision bands,
crossover estimator, bootstrap procedure and seeds are all fixed here. Nothing in this
document may be revised after the outcome metrics are inspected; if it is revised, the
revision is a separate commit with its reason stated and the original left in history.

Branch `estep-study`. Prior work this extends: `scripts/planted/planted_error_sweep.py`,
`scripts/planted/planted_sweep_config.json` (committed `7534b77`, before its own sweep
ran), `reports/planted_error_phase_diagram.md` and `reports/artifacts/planted_sweep.csv`
(committed `7663d03`).

---

## 1. What the existing result is, and the one thing that is already known analytically

The committed sweep establishes, over 240 cells:

- The **error-induced variance** ratio `Var[g_ZO]_e / Var[g_PW]_e` crosses 1 at
  `r = sigma*omega/sqrt(d) = 1`, with zero misclassifications either side and per-slice
  fitted `r*` in [0.974, 1.128].
- The **operational** comparison (trust-region update error against the exact oracle)
  does not follow: the actual weighted-MLE E-step beats PW in only 34 of 76 cells with
  `r > 1`, and its globally fitted crossover is `r ~ 1.67`.

**Stated before this run, because it is a property of the code and not an outcome.**
The centred zeroth-order and pathwise estimators are *linear in Q*:

```
g_PW(Q^pi + e) = g_PW(Q^pi) + g_PW(e)
g_ZO(Q^pi + e) = g_ZO(Q^pi) + g_ZO(e)      (centring by Qbar is linear)
```

so under common random numbers the paired difference is exactly `g(e)`, and since
`e_A = A e_1` with both estimators linear, `g(e_A) = A g(e_1)`. The error-channel ratio

```
Var[g_ZO]_e / Var[g_PW]_e
```

is therefore **A-independent by construction**, exactly, for the PW/ZO pair. This is
already asserted by `test_error_induced_variance_ratio_is_amplitude_invariant`
(tolerance 1e-8).

**Consequence for H1.** For the PW/ZO pair, amplitude invariance is a *check that the
implementation does what the algebra says*, and will be reported as such. It is not a
finding and will not be presented as one. H1 has empirical content **only for the
softmax E-step (WML)**, whose weights `w_i = softmax(Q_i/eta)` and whose dual-solved
`eta` are nonlinear in `Q`, so no invariance is implied and the measurement can come
out either way.

---

## 2. Definitions fixed in advance

### 2.1 Scalar variance measure

For a vector estimator,

```
V(ghat) = E|| ghat - E ghat ||_2^2 = tr Cov(ghat)
```

the trace of the covariance. This is what the committed code computes
(`np.trace(np.cov(.T))`) and it remains primary.

### 2.2 Error-channel variance: two definitions, both reported

**Primary (paired, common random numbers)** — the committed definition:

```
V_e^paired(ghat) = tr Cov[ ghat(Q^pi + e; xi) - ghat(Q^pi; xi) ]
```

with the *same* draw `xi = {u_i}` in both terms.

**Secondary (subtraction)**:

```
V_e^sub(ghat) = tr Cov[ ghat(Q^pi + e) ] - tr Cov[ ghat(Q^pi) ]
```

**Reported difference.** For any estimator,

```
V_e^sub - V_e^paired = 2 * tr Cov( ghat(Q^pi), ghat(e) )
```

the cross term, reported as its own column. For PW and ZO this identity is exact by
linearity; for WML the "`ghat(e)`" slot is the paired difference and the identity is
definitional rather than algebraic, which is stated wherever the WML cross term is used.

### 2.3 Target

`g* = grad Q^pi(0) = a*`, exact and blur-invariant because `Q^pi` is quadratic. The
target is the gradient of the **true** value function alone; the planted error is
contamination and is never part of the target.

### 2.4 "Better update"

Unchanged from the committed code:

```
Delta_mu = sqrt(2 eps_TR) * Sigma ghat / ||ghat||_Sigma,     ||g||_Sigma^2 = g^T Sigma g
Err      = || Delta_mu - Delta_mu* ||^2_{Sigma^-1}
```

**Noted in advance** (verified against the committed CSV to 1.7e-16): because every
update is normalised to the same length, this metric is *exactly*

```
Err = 4 eps_TR (1 - cos(ghat, g*)) = 2 eps_TR E|| uhat - uhat* ||^2
```

with `uhat = ghat/||ghat||`. The operational comparison is therefore purely
**directional**; magnitude cannot contribute. `eps_TR = 0.1` cancels from every ratio.

### 2.5 Post-normalisation decomposition

Because normalisation is nonlinear, the pre-normalisation decomposition is not assumed
to carry through. On the unit sphere the decomposition is exact:

```
E|| uhat - uhat* ||^2  =  || E[uhat] - uhat* ||^2  +  E|| uhat - E[uhat] ||^2
                          \_____ bias^2 _____/       \_____ variance _____/
```

and `Err = 2 eps_TR` times this. Reported for PW, ZO and WML.

### 2.6 Pre-normalisation decomposition

For PW and ZO (gradient estimators, same units as `g*`): bias vector `E[ghat] - g*`,
squared bias, `tr Cov`, MSE, and cosine.

For WML: `Delta_mu_WML` is a *displacement*, not a gradient, so a bias against `g*` has
no scale convention and **will not be reported as one**. Reported instead, which fixes
the first two moments completely and imposes no bogus scale: `||E[Delta_mu]||`, the
angle of `E[Delta_mu]` to `g*`, `tr Cov`, and the scale-free noise-to-signal ratio
`tr Cov / ||E[Delta_mu]||^2`.

### 2.7 Channel attribution (non-additive-safe)

Additive attribution is invalid after normalisation, so attribution is by
**counterfactual intervention on the estimator output**, using the CRN-paired pair
`(ghat_0, ghat_1) = (ghat(Q^pi), ghat(Q^pi+e))`:

```
Err_total     = Err( ghat_1 )                              both channels noisy
Err_cleanoff  = Err( E[ghat_0] + (ghat_1 - ghat_0) )       smooth channel estimated perfectly
Err_erroroff  = Err( ghat_0 + E[ghat_1 - ghat_0] )         error channel estimated perfectly
Err_baseline  = Err( ghat_0 )                              error field switched off
```

Whichever of `Err_cleanoff` / `Err_erroroff` falls further below `Err_total` identifies
the dominant channel. For PW and ZO these are exact interventions on `g(Q^pi)` and
`g(e)`; for WML they are interventions on the paired difference, which is stated.

---

## 3. Design

### 3.1 Amplitudes

`A_0 = 1.0` (the committed `planted_error.eps`). Three log-spaced levels:

```
A in { 0.25, 1.0, 4.0 }   =   { A_0/4, A_0, 4 A_0 }
```

No amplitude is excluded in advance. Pathology criteria are in Sec. 6; if one triggers,
it is documented **before** the outcome metrics for that arm are examined.

### 3.2 Factorial over (d, r)

Dimensions, taken from the existing grid, low / middle / high:

```
d in { 4, 16, 64 }
```

Target dimensionless values, sampling the transition directly:

```
r in { 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 3.0 }
```

`sigma` is held at **0.4** (a value from the existing grid) and `omega` is solved from
`omega = r sqrt(d) / sigma`, which keeps every `omega` inside the swept range
[0.5, 64] of the committed grid:

| `d` | `omega = r sqrt(d)/0.4` | range over the `r` grid |
|---|---|---|
| 4  | `5r`  | 2.5 – 15 |
| 16 | `10r` | 5 – 30 |
| 64 | `20r` | 10 – 60 |

**Second `sigma` level (confound check).** At `d = 16` only, the same `r` grid is
repeated at `sigma = 0.2` (`omega = 20r`, range 10–60). This tests that a change in
`r` traced through `omega` at fixed `sigma` is not `sigma`-specific. It is a check, not
a separate hypothesis.

**Total: (3 x 8 + 1 x 8) x 3 = 96 cells.** Not a re-run of the 240-cell sweep.

### 3.3 Held fixed

Same `Q^pi` (`H = I`, `mu = 0`, `a* = G v`, `G = 1`), same `M = 32`, same `eps_E = 0.5`,
same MPO dual `eta` solver, same trust-region step and metric, same baseline (`Qbar` =
sample mean), same estimator implementations imported from `planted_error_sweep.py`,
same isotropic treatment of the random directions `v_signal`, `v_e` and phase.

### 3.4 Monte Carlo budget

**Identical total budget to the committed sweep, re-partitioned.** The committed sweep
used 8 directions x 4096 batches = 32,768 batches per cell. This experiment uses

```
n_directions = 64,  n_batches = 512     (= 32,768 batches per cell)
```

Re-partitioning is required, not cosmetic: the bootstrap replicate unit (Sec. 5) is the
independent direction block, and 8 blocks is too few to resample. The total number of
critic evaluations per cell is unchanged.

### 3.5 Common random numbers

Within a cell, `v_signal`, `v_e`, `phase` and the whitened draws `u` are drawn **once
per direction block and reused across all three amplitudes**, and within each amplitude
the same `u` is used for the clean and contaminated critic. CRN across amplitudes is
valid because the amplitude enters only as a multiplier on `e`, changing no distribution
that is sampled.

Seed: **`20260903`** (distinct from the committed sweep's `20260901` so the two runs are
independent draws; the reproduction check in Sec. 7 uses `20260901`).

---

## 4. Hypotheses and decision bands

Fixed here, applied without modification.

### H1 — the error-channel crossover is amplitude-invariant

`r*_var(A)`, estimated as in Sec. 5, for the **WML vs PW** error channel.

> **H1 supported** if `r*_var(A) in [0.8, 1.25]` for all three amplitudes.

The **ZO vs PW** error channel is reported alongside as an implementation check
(Sec. 1); by construction it must be amplitude-invariant to floating-point tolerance,
and a deviation there is a bug report, not evidence.

### H2 — the operational crossover is not universal in `r`

`r*_op(A)` for the **WML vs PW** update error.

> **H2 supported** if `r*_op(4 A_0) < r*_op(A_0/4)` with **non-overlapping 95% bootstrap
> intervals**.

Otherwise the result is reported **descriptively**, with the intervals shown, and H2 is
recorded as not supported. H2 is a diagnostic hypothesis about the direction of a shift.
It is **not** claimed to follow mathematically from the existing theory and will not be
called a theorem. For the linear PW/ZO pair a directional shift *is* implied by the
decomposition (the error channel scales as `A^2` while the smooth channel is
`A`-independent); for the nonlinear WML operator no such implication is available, which
is why it is measured.

`r*_op` for **ZO vs PW** is reported alongside.

### C — the mismatch region

The set of cells with

```
V_e[WML] < V_e[PW]      (E-step wins the error channel)
Err[WML] > Err[PW]      (E-step still loses the update)
```

reported as a fraction of cells and as an `r`-interval per amplitude, with bootstrap
intervals on the fraction.

### Out-of-range handling, fixed in advance

If a fitted `r*` falls outside the sampled range [0.5, 3.0] it is reported as
`"<0.5"` or `">3.0"` and flagged as extrapolated. It is **not** silently extrapolated,
and an extrapolated bound may not be used to satisfy a decision band.

---

## 5. Crossover estimator and uncertainty

**Estimator.** For a given amplitude and estimator pair, ordinary least squares of
`y = log(ratio)` on `x = log(r)` over the sampled cells, with

```
r* = exp(-intercept / slope)
```

Pooled across `d` (and across the `sigma` check, which is reported separately as well).
Reported with the fit's `R^2`, because the committed report's operational crossover was
obtained by this method over a range where the relation is not a power law: refitting
the committed CSV gives `R^2 = 0.9937` for the variance ratio but only **0.50 (ZO)** and
**0.51 (WML)** for the operational ratios. Any `r*` from a fit with low `R^2` is
reported as a **descriptive summary of where the fitted line crosses 1, not as a sharp
boundary**, and the binned medians are shown next to it.

**Bootstrap.** Replicate unit: the **direction block** (64 per cell), resampled with
replacement, `B = 10,000` resamples, bootstrap seed **`20260903`**. Cell statistics are
recomputed from the resampled blocks, the fit is repeated, and the 95% interval is the
[2.5, 97.5] percentile interval of `r*`. The same procedure gives intervals for `beta`,
for the mismatch fraction, and for every ratio reported with uncertainty.

---

## 6. Pathology criteria, declared before the run

Monitored and reported for every cell and amplitude regardless of outcome:

1. `eta` solver hitting either bracket bound (`lo = 1e-4`, `hi = 1e4`).
2. Median effective sample size `ESS = 1/sum_i w_i^2` below 2 (softmax weight collapse
   would make WML pure noise and the comparison meaningless).
3. Any non-finite value in any estimator output.
4. Measured `omega` departing from nominal (sup-norm calibration on a dense grid over a
   full period along `v_e`, as in `test_omega_matches_the_theorem_definition` — **not**
   the policy-region sampling in the committed sweep's `om_meas` column, which that
   report already records as invalid and unused).

If any criterion triggers for an amplitude, it is documented here in a follow-up commit
**before** that arm's outcome metrics are interpreted, and the affected cells are
reported, not silently dropped.

---

## 7. Reproduction check on the committed sweep

Separately from the amplitude experiment, the committed 240-cell grid is re-run at
`A = A_0` with the original seed `20260901`, the original 8 x 4096 partition, and the
original RNG call order preserved exactly (including the discarded `om_meas` draw, which
must be kept to hold the stream). This must reproduce the committed `ratio_e` column to
floating-point tolerance. It adds instrumentation only:

- `V_e` for **WML** (the committed sweep measured the error channel for PW and ZO only);
- both variance definitions and the cross term (Sec. 2.2);
- the pre- and post-normalisation decompositions (Secs. 2.5, 2.6);
- the channel attribution counterfactuals (Sec. 2.7).

A failure to reproduce is a blocking finding and is reported before anything else.

---

## 8. Deliverables fixed in advance

- `reports/artifacts/planted_error_decomposition.csv` — Sec. 7 grid, per cell.
- `reports/artifacts/planted_amplitude.csv` — Sec. 3 factorial, per cell.
- `reports/artifacts/planted_amplitude_replicates.csv` — per direction block, the
  bootstrap input.
- `reports/planted_error_mechanism.md` — definitions, audit, decomposition, results,
  uncertainty, interpretation, reviewer assessment.
- One main figure with source-data CSV.

**No further experiments.** Orientation, curvature, random-Fourier, multi-mode and
large-`M` sweeps are out of scope for this work and are not run.

---

## 9. What would make this experiment misleading

- Choosing `sigma = 0.4` to make the transition land conveniently. Mitigated by the
  second `sigma` level at `d = 16` and by the fact that the `r` grid, not `sigma`, was
  chosen to bracket the transition.
- Reading `r*_op` from a low-`R^2` power-law fit as if it were a sharp boundary. The
  committed report does this; Sec. 5 fixes the reporting rule that prevents it here.
- Presenting the PW/ZO amplitude invariance as evidence. Sec. 1 forecloses it.
- Bootstrapping over batches within a direction block, which would treat the shared
  `v_signal`, `v_e` and phase as if they were resampled and understate the interval.
  The replicate unit is the block.
