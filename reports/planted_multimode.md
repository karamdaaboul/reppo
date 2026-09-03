# Is the E-step's systematic response to critic error a single-mode artifact?

**Scope.** Controlled planted-error experiment. `Q^pi` and `e` are known exactly. Nothing
here is evidence about learned critics, DMC returns, or Probe 4.

**Provenance.** Pre-registration `docs/prereg_planted_multimode.md`, committed
`44c86a9` **before any J = 4 outcome was computed**. Prior: `4b88f03` (amplitude
prereg), `1b9e05a` (mechanism report), `3fde75c` (main figure). Scripts
`scripts/planted/multimode_sweep.py`, `analyse_multimode.py`,
`make_multimode_figure.py`. CPU only, float64.

```bash
./.venv/bin/python scripts/planted/multimode_sweep.py reports/artifacts
./.venv/bin/python scripts/planted/analyse_multimode.py
./.venv/bin/python scripts/planted/make_multimode_figure.py
```

Three objects kept separate throughout: **(1)** centred ZO vs PW critic-error variance;
**(2)** the actual nonlinear MPO E-step's response to critic error; **(3)** final
normalised policy-update quality.

---

## 0. Answer

**No — with a quantitative qualification, and one unexpected finding.**

The E-step's systematic error-induced displacement survives splitting the critic error
into four competing orthogonal modes. Per fixed error field it falls from
`B_J1 = 0.357` to `B_J4 = 0.172` of the clean signal — an attenuation of **2.08x**,
registered as **CASE B (partial attenuation)**. But most of that attenuation is not mode
competition: holding `||e||_inf` fixed, as Claim 4's norm requires, necessarily lowers
the field's RMS strength by `sqrt(J) = 2`, and calibrating against the committed
amplitude experiment shows **only a further 1.15x is attributable to competition between
modes**. At `J = 4` the E-step is still dominated by systematic displacement over noise
by 3.2 to 1 (5.7 to 1 at `J = 1`), while pathwise remains dominated by error-channel
noise by 100 to 1. The qualitative distinction that motivated the mechanism is intact.

**The unexpected finding** is what the displacement *is*. It is not a random
misdirection: `cos(delta_b, v_sig)` has median **-0.96** (`J = 1`) and **-0.98**
(`J = 4`). The critic error makes the E-step take a **smaller step in the right
direction** — a 36% shrinkage of the true-signal displacement at `J = 1`, 17% at
`J = 4` — while the `ubar` mean-estimation noise floor is untouched. Signal shrinks,
noise does not, so the normalised direction degrades.

**A second finding, secondary but consequential for the manuscript:** the centred ZO/PW
variance ratio crosses 1 at `r_eff = 0.4924` for `J = 4`, not at 1 — exactly
`1/sqrt(J)`. Claim 4's **sup-norm** definition of `omega` is not the convention that
organises the realised variance ratio when the error has several modes; the RMS
convention is, and under it the crossing returns to `0.9848`. This is the norm ambiguity
already flagged in `reports/lqr_crossover.md` Sec. 7.2, now measured.

---

## 1. Blocking audit

### 1.1 Was the previous "systematic bias" conditional, or an artifact of pooling?

**Conditional.** Read from `scripts/planted/decompose_sweep.py`:

1. **Was the action-batch expectation taken inside each fixed field block first?**
   Yes. `v_sig`, `v_e` and `phase` are drawn once per block, outside the batch axis;
   every statistic is computed from `X` of shape `(R, d)` whose `R` axis is
   action-sampling noise at fixed field, so `X.mean(0)` is `E_action[· | e_b]`.
2. **Was the norm taken per block?** Yes —
   `out["pre_wml_err_meannorm"] = ||mu||` with `mu = X.mean(0)`.
3. **Were per-field magnitudes then averaged across blocks?** Yes —
   `main()` does `row[k] = np.mean([b[k] for b in blocks])`.

No vectors were pooled across error fields before a norm was taken. The previously
reported `||E Delta_mu_err|| = 0.0782` and the counterfactual error-channel mean
contribution `0.2082` are conditional quantities and stand as reported.

**One refinement was required and is applied here.** The previously quoted ratio 0.636
was `E_field[||delta||] / E_field[||s||]`, a ratio of block-averages, not
`E_field[||delta||/||s||]`. The per-block ratio is primary in this experiment, so `B` is
computed per block and the `J = 1` value is measured fresh rather than carried over.

**Independent replication.** `B_J1` measured here (seed `20260904`) against the same
quantity recomputed from the committed amplitude experiment at `A = A_0` (seed
`20260903`): agreement is within 2% at **21 of 24** cells and within 5% at 22 of 24. The
two exceptions are `d = 4, r = 0.5` (13.7%) and `d = 4, r = 0.75` (13.0%), where the
across-block distribution of `B_b` is widest. Two independent seeds, same construction,
same answer.

### 1.2 What the existing "J = 4 configuration" was

`planted_sweep_config.json` contains one sentence —
`"J=4 superposed random directions with sup-norms calibrated numerically (robustness
only)"` — plus `"eps_secondary": [0.1, 1.0]`. There is **no implementation** in
`scripts/planted/`, and the sentence fixes no `J`, no `c_j`, no frequency rule, no
direction-sampling rule, no phase rule and no normalisation; it also proposes numerical
sup-norm calibration where an analytic value exists. **It is not a pre-registration and
was not treated as one.** The construction was specified in full in
`docs/prereg_planted_multimode.md` before the run.

---

## 2. Construction and normalisation

```
e(a) = sum_{j=1}^{J} c_j sin( omega v_j^T a + phi_j )
```

| item | value |
|---|---|
| `J` | 1 and 4 |
| `c_j` | `A_0 / J`, `A_0 = 1.0` |
| `{v_j}` | **orthonormal**, first `J` columns of the Q factor of a `d x J` Gaussian (Haar `J`-frame) |
| `phi_j` | `U[0, 2pi)`, independent |
| frequency | a **single** `omega` shared by all modes |

Because the `v_j` are orthonormal, `a -> (v_1^T a, ..., v_J^T a)` is onto `R^J`, so the
`J` phase arguments are independently settable and both sup-norms are attained
**exactly**:

```
A_eff = ||e||_inf      = sum_j |c_j|             = A_0
L_eff = ||grad e||_inf = omega sqrt(sum_j c_j^2) = omega A_0 / sqrt(J)
omega_eff = L_eff / A_eff = omega / sqrt(J)          (Claim 4's definition)
r_eff = sigma omega_eff / sqrt(d)
```

**These are exact formulae, not calibrations.** The sweep re-checks them per cell with
20-start L-BFGS on `-|e|` and `-||grad e||` plus a 200,000-point dense probe. Both
searches return lower bounds; measured numeric/analytic ratios are **1.000000 for both
`A_eff` and `L_eff` at every one of the 48 cells**.

**Normalisation, and the choice rejected.** `c_j = A_0/J` holds `A_eff = A_0` exactly
equal across `J`, so `J = 4` is not a stronger critic error. `c_j = A_0/sqrt(J)` was
rejected in the prereg because it would give `A_eff = 2 A_0` at `J = 4` and confound mode
count with error strength — the request's CASE 3 defect. Cells are matched on `r_eff`,
so `omega = r_eff sqrt(J d) / sigma` and nominal `omega` differs between arms by exactly
`sqrt(J) = 2`.

**What this normalisation does not hold fixed, and it matters (Sec. 5.2).** Averaging
over phases, `E_phi E_a[e^2] = sum_j c_j^2 / 2 = A_0^2/(2J)`, so the field's **RMS**
strength falls by exactly `sqrt(J) = 2` at `J = 4` even though its sup-norm is unchanged.
Matching the sup-norm — which is what Claim 4's `omega` requires — necessarily
un-matches the typical magnitude.

### 2.1 Design

`A = A_0 = 1.0`, `sigma = 0.4`, `d in {4, 16, 64}`,
`r_eff in {0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 3.0}`, `J in {1, 4}` — 48 cells.
64 error-field blocks x 512 action batches = 32,768 batches per cell, the committed
budget. Seeds: run and bootstrap `20260904`.

### 2.2 Validation

| check | result |
|---|---|
| CRN: `wml0` checksum, `J=1` vs `J=4` | **0.000e+00** (clean channel bit-identical across arms) |
| frame orthonormality `max \|VV^T - I\|` | 1.11e-15 |
| sup-norm numeric/analytic, `A_eff` and `L_eff` | **1.000000** at all 48 cells |
| min median ESS | 13.94 (`J=1`), 13.93 (`J=4`); floor is 2 |
| min single-batch ESS | 10.62 / 10.48 |
| `eta` bracket hits | 0 |
| non-finite values | 0 |
| `A_eff` realised | 1.0 in both arms |

---

## 3. Primary: conditional systematic displacement

`B_b = ||delta_b|| / ||s_b||` with both expectations taken over action batches inside the
fixed field block; `B_J = E_field[B_b]`; bootstrap over the 64 blocks, `B = 10,000`.

| | `B_J1` | `B_J4` | `B_J4/B_J1` (paired) |
|---|---|---|---|
| median across the 24 cells | **0.3572** | **0.1717** | **0.4816** |
| mean across cells | 0.3758 | 0.1858 | 0.4997 |
| per-cell range | 0.224–0.699 | 0.099–0.371 | 0.363–0.751 |
| cells whose paired 95% CI excludes 1 | — | — | **24 / 24** |

Across-block distribution of `B_b`, pooled over cells:

| arm | mean | median | q25 | q75 | min | max |
|---|---|---|---|---|---|---|
| `J = 1` | 0.3758 | 0.3571 | 0.2339 | 0.4609 | 0.169 | 1.502 |
| `J = 4` | 0.1858 | 0.1715 | 0.1052 | 0.2689 | 0.077 | 0.546 |

`B` is nearly flat in `r_eff` above ~1 and rises only at the smallest `r_eff` — the
systematic displacement is largely a property of the error amplitude, not its frequency.

**Registered verdict: CASE B, partial attenuation, factor 2.08x.** The median 0.4816
sits just below the registered CASE A threshold of 0.5 and the across-cell mean is
0.4997, so the A/B boundary is not cleanly resolved by this statistic; the registered
rule uses the median and gives CASE B. Section 5.2 is what actually settles the
interpretation.

---

## 4. Secondary: conditional error-channel noise

`N_b = tr Cov_action[Delta_err | e_b] / ||s_b||^2`.

| | `N_J1` | `N_J4` | ratio |
|---|---|---|---|
| median across cells | 1.0833 | 0.6079 | **0.5628** |

`N` grows steeply with `d` (0.154 at `d=4`, 1.08 at `d=16`, 7.9 at `d=64` for `J=1`),
as the `ubar` term predicts.

**Does `J = 4` change systematic displacement, stochastic noise, or both? Both, by
similar factors** — 0.48 for the systematic term against 0.56 for the noise term. There
is no selective suppression of the systematic channel.

---

## 5. Why the attenuation happens

### 5.1 The displacement is a shrinkage of the true signal, not a misdirection

`cos(delta_b, v_sig)` per block:

| arm | mean | median | q05 | q95 | fraction < -0.9 |
|---|---|---|---|---|---|
| `J = 1` | -0.8931 | **-0.9624** | -0.9996 | -0.6183 | 0.542 |
| `J = 4` | -0.8613 | **-0.9785** | -0.9996 | -0.5747 | 0.630 |

Median by dimension: -0.993 / -0.992 / -0.837 at `d` = 4 / 16 / 64 for `J = 1`
(-0.999 / -0.982 / -0.647 for `J = 4`). The error-induced displacement is
**anti-parallel to the true signal** — nearly exactly so at low `d`, with a growing
transverse component as `d` rises.

With `||s|| = 0.2437` (identical across arms by CRN) and `||delta|| = 0.0870` / `0.0418`,
the critic error shrinks the E-step's true-signal displacement by **36%** at `J = 1` and
**17%** at `J = 4`. The `ubar` noise floor is unchanged — `Err_baseline` is 0.1655 in
both arms, as CRN requires. Signal shrinks, noise floor does not, and the normalised
update direction degrades accordingly. That is the mechanism, stated more precisely than
the previous report could.

### 5.2 Most of the attenuation is the weaker field, not mode competition

Matching `||e||_inf` lowers RMS strength by `sqrt(J) = 2` (Sec. 2). Calibrating the
amplitude sensitivity of `B` against the **committed** amplitude experiment — no new run
— gives `B ~ A^p`:

| `d` | `p` | expected `0.5^p` | measured `B_J4/B_J1` | residual |
|---|---|---|---|---|
| 4 | 0.6116 | 0.6544 | 0.5848 | 0.894 |
| 16 | 0.8544 | 0.5531 | 0.4816 | 0.871 |
| 64 | 0.9597 | 0.5142 | 0.4441 | 0.864 |

Residual median **0.871**, tight across dimensions. So of the 2.08x attenuation, about
**1.8x is the amplitude drop the sup-norm normalisation implies** and only a further
**1.15x is attributable to competition between modes**. Splitting the error into four
competing directions attenuates the E-step's systematic displacement barely more than
simply weakening the field by the same amount would.

`p < 1` at every `d` (0.61–0.96) — the E-step's systematic response saturates with
amplitude, consistent with the softmax saturation seen in the amplitude experiment.

---

## 6. Counterfactual decomposition (existing procedure, unchanged)

Medians over the 24 cells per arm, `Err` in the existing angular units.

**`J = 1`**

| operator | floor | + error MEAN | + error NOISE | = total | systematic/noise |
|---|---|---|---|---|---|
| PW | 0.0143 | +0.0114 | **+0.1213** | 0.1470 | 0.09 |
| ZO | 0.1351 | +0.0004 | +0.0642 | 0.1998 | 0.01 |
| **E-step** | 0.1655 | **+0.0664** | +0.0116 | 0.2435 | **5.74** |

**`J = 4`**

| operator | floor | + error MEAN | + error NOISE | = total | systematic/noise |
|---|---|---|---|---|---|
| PW | 0.0143 | +0.0013 | **+0.1717** | 0.1873 | 0.01 |
| ZO | 0.1351 | +0.0001 | +0.0216 | 0.1568 | 0.00 |
| **E-step** | 0.1655 | **+0.0289** | +0.0089 | 0.2033 | **3.23** |

**The qualitative pattern survives.** Pathwise is dominated by error-channel noise at
both `J` (ratio 0.09 → 0.01, i.e. *more* noise-dominated at `J = 4`). The E-step is
dominated by systematic displacement at both `J` (5.74 → 3.23). The two operators still
fail in different ways, and the difference is if anything cleaner at `J = 4`.

---

## 7. Cross-field cancellation

| `J` | `E_field \|\|delta\|\|` (conditional) | `\|\| E_field delta \|\|` (pooled) | pooled/conditional | `E_field[delta . v_sig]` |
|---|---|---|---|---|
| 1 | 0.08703 | 0.00919 | **0.120** | -0.08609 |
| 4 | 0.04185 | 0.00408 | **0.114** | -0.04102 |

The pooled vector mean is 12% of the conditional magnitude at both `J`, so **88% of the
raw pooled mean cancels** — and the cancellation is essentially the same at `J = 1` and
`J = 4`, so it is not a multi-mode effect.

**But the cancellation is an artifact of the random signal direction, not of the
mechanism.** Projected on each block's own signal direction, `E_field[delta . v_sig]` is
-0.08609 (`J = 1`) — that is **98.9% of the full conditional magnitude 0.08703**. The
displacement points the same way relative to the policy in every block; it is only the
policy's orientation that is random, so a fixed-frame vector average destroys it.
Reporting `||E_field delta||` as "the bias" would have understated the effect by 8.3x.
The conditional quantity is the right one, and this table is the evidence for that.

---

## 8. Centred ZO/PW: the theory check, and a norm-convention finding

`R_var = Var_e[ZO]/Var_e[PW]`, paired CRN definition, fitted against `r_eff`:

| arm | `beta` | 95% CI | `R^2` | crossing in `r_eff` | ZO better below `r_eff=1` | above |
|---|---|---|---|---|---|---|
| `J = 1` | -2.0268 | [-2.0728, -1.9947] | 0.9990 | **0.9931** | 0 / 6 | 15 / 15 |
| `J = 4` | -2.0000 | [-2.0071, -1.9940] | **1.0000** | **0.4924** | 6 / 6 | 15 / 15 |

The scaling is **consistent with the ratio of the two bounds Claim 4 supplies**; Claim 4
supplies bounds and does not imply the measured ratio must equal `r^-2`.

The `J = 4` crossing is at 0.4924, not 1. This is not a failure of the organisation — it
is a statement about which norm convention `omega` should use, and it is predictable in
closed form. The realised variances depend on RMS statistics, not sup-norms:

```
Var_e[PW] ~ (1/M) E||grad e||^2 = omega^2 A_0^2 / (2 J M)
Var_e[ZO] ~ (d/M) E[e^2]/sigma^2 = d A_0^2 / (2 J M sigma^2)
```

The `A_0^2/(2J)` cancels, so the ratio is `r_nom^-2` with `r_nom = sigma omega/sqrt(d)`
— the **nominal** `r`, independent of `J`. The crossing must therefore sit at
`r_eff = 1/sqrt(J)`. Measured:

| arm | crossing in `r_eff` | predicted `1/sqrt(J)` | crossing in `r_nom` | predicted |
|---|---|---|---|---|
| `J = 1` | 0.9931 | 1.0000 | 0.9931 | 1.0 |
| `J = 4` | 0.4924 | 0.5000 | **0.9848** | 1.0 |

**The centred ZO/PW organisation survives `J = 4` intact — under the RMS convention.**
Under Claim 4's sup-norm convention it is displaced by exactly `sqrt(J)`.

**One caveat on the strength of that test.** Because `r_nom = sqrt(J) r_eff`, the
`J = 4` arm spans `r_nom in [1.0, 6.0]`: it has **no cells strictly below** the `r_nom`
boundary (3 cells sit exactly on it, with `R_var` = 0.989, 0.967, 0.964). So `J = 4`
confirms 21/21 cells on the *above*-boundary side and cannot test the below-boundary
side at all. The `J = 1` arm does test both (6/6 below, 15/15 above, 3 on the boundary).
The `sqrt(J)` shift itself is a prediction confirmed to three digits, but the two-sided
sharpness of the boundary is evidenced by `J = 1`, not by `J = 4`.

This is the
ambiguity `reports/lqr_crossover.md` Sec. 7.2 flagged as unresolved ("until the paper
states which norm `omega` denotes, Claim 4 is not well-posed for a full-rank error
field"), now measured on a two-point ladder: the data pick the RMS convention.

---

## 9. Operational result

`G = Err[E-step] - Err[PW]` against `r_eff`, bootstrap over blocks. **No crossover is
fitted or quoted.**

| `r_eff` | `G` (`J=1`) | 95% CI | `G` (`J=4`) | 95% CI | shrinks? |
|---|---|---|---|---|---|
| 0.50 | 0.1386 | [0.1280, 0.1489] | 0.1225 | [0.1154, 0.1296] | overlap |
| 0.75 | 0.1286 | [0.1213, 0.1355] | 0.0860 | [0.0807, 0.0913] | yes |
| 1.00 | 0.1151 | [0.1102, 0.1198] | 0.0546 | [0.0501, 0.0591] | yes |
| 1.25 | 0.0925 | [0.0878, 0.0972] | 0.0271 | [0.0226, 0.0316] | yes |
| 1.50 | 0.0746 | [0.0699, 0.0793] | 0.0044 | [-0.0004, 0.0092] | yes |
| 1.75 | 0.0591 | [0.0542, 0.0638] | **-0.0152** | [-0.0206, -0.0097] | yes |
| 2.00 | 0.0450 | [0.0397, 0.0501] | **-0.0322** | [-0.0383, -0.0262] | yes |
| 3.00 | 0.0064 | [0.0005, 0.0123] | **-0.0787** | [-0.0868, -0.0706] | yes |

The gap shrinks at 7 of 8 `r_eff` values with disjoint intervals and overlaps at the
eighth; it never grows. At `J = 4` it becomes **negative** from `r_eff = 1.75` — the
E-step produces the better update there. That is consistent with Sec. 5: a weaker
effective field means less signal shrinkage, so the E-step's fixed `ubar` penalty is
overcome sooner.

---

## 10. Limitations

1. `Q^pi` is quadratic and globally smooth; `e` is planted, not learned. Nothing here
   licenses a claim about learned critics.
2. `J = 4` with orthonormal directions and a **single shared frequency** is one point in
   a large space. A frequency spread was deliberately excluded (it would move `L_eff`
   inseparably from the mode count) and is untested.
3. Matching `||e||_inf` necessarily un-matches RMS strength. Sec. 5.2 separates the two
   using an amplitude calibration from a different experiment; that calibration assumes
   `B ~ A^p` holds locally, and `p` was fitted over `A in {0.25, 1, 4}`, a wide range.
4. The operational metric is purely directional; magnitude cannot contribute.
5. `sigma = 0.4` only, one amplitude only. The previous report showed the operational
   comparison depends on `sigma` separately from `r`; that dependence is not re-probed
   here.
6. `d >= J` is required for orthonormal directions, so `J = 4` at `d = 4` uses a complete
   basis — a boundary case, and the `d = 4` column is where `B` and its amplitude
   exponent are least typical.

---

## 11. Reviewer-style verdict

**1. Was the previous `J = 1` "systematic bias" genuinely conditional, or partly an
artifact of pooling?** Genuinely conditional. The action-batch expectation was taken
inside each fixed field block, the norm per block, and only then averaged across blocks
(Sec. 1.1). Pooling would have understated it by 8.3x (Sec. 7), and the previous report
did not pool.

**2. `B_J1`?** **0.357** (median across cells; mean 0.376; per-block IQR 0.234–0.461).
The systematic error-induced displacement is 36% of the clean E-step signal for a typical
fixed error field.

**3. `B_J4`?** **0.172** (mean 0.186; IQR 0.105–0.269).

**4. `B_J4/B_J1`?** **0.4816** by paired per-block ratio, per-cell range
[0.363, 0.751], and all 24 cells' paired 95% CIs exclude 1. Attenuation factor 2.08x.

**5. Does `J = 4` change systematic displacement, stochastic noise, or both?** **Both,
by similar factors** — 0.48 systematic against 0.56 noise. No selective suppression.

**6. Does the E-step still show low critic-error noise but large systematic
displacement?** **Yes.** Its systematic/noise ratio is 3.23 at `J = 4` (5.74 at
`J = 1`), while pathwise's is 0.01 (0.09 at `J = 1`). The two operators still fail in
different ways, more cleanly at `J = 4` than at `J = 1`.

**7. Does the operational gap shrink substantially?** Yes — it shrinks at 7 of 8
`r_eff` values with disjoint intervals and turns negative above `r_eff = 1.75`. But this
follows from the weaker effective field (Sec. 5.2), not from mode competition, and it is
the expected consequence of less signal shrinkage.

**8. How much cross-field cancellation?** 88% at both `J` (pooled/conditional = 0.120 and
0.114). It is essentially identical across arms, so it is not a multi-mode effect — and
it is an artifact of the randomly oriented policy, since projected on each block's own
signal direction the pooled mean retains 98.9% of the conditional magnitude.

**9. Does the centred ZO/PW `r` organisation survive `J = 4`?** **Yes, under the RMS
convention** — crossing 0.9848, `beta = -2.0000` [-2.0071, -1.9940], `R^2 = 1.0000`,
21/21 off-boundary cells correctly classified. Under Claim 4's sup-norm convention the
crossing moves to 0.4924, exactly `1/sqrt(J)`. Caveat: the `J = 4` arm spans
`r_nom in [1.0, 6.0]` and so tests only the above-boundary side; two-sided sharpness
comes from `J = 1`.

**10. Was the single sinusoid (a) a serious artifact, (b) an especially clean but
representative example, or (c) in between?** **(b), close to the (b)/(c) boundary.** The
mechanism survives with the same sign, the same qualitative operator asymmetry, and a
systematic term still 3.2x its noise term. The attenuation that does occur is 1.8x
amplitude and only 1.15x mode competition. The honest caveat is that `J = 4` is one
point, at a single shared frequency.

**11. Strongest one-sentence conclusion for the paper.**
*Splitting the planted critic error into four competing orthogonal modes at matched
`||e||_inf` leaves the E-step's systematic error-induced displacement intact in kind and
attenuated 2.08x in size — of which only 1.15x is attributable to mode competition
rather than to the lower RMS field strength that sup-norm matching imposes — so the
mechanism is not an artifact of a single coherent mode.*

**12. Should `J = 4` be main paper, appendix, or omitted?** **APPENDIX**, with two
sentences in the main text: one stating that the mechanism survives a four-mode error
field with the attenuation decomposed, one stating the norm-convention result. The
figure is a diagnostic and should **not** replace the current main figure
(`fig_planted_mechanism.pdf`). The one element with a claim on main-paper space is
Panel C's norm-convention finding, because it bears on whether Claim 4 is well-posed —
that belongs as a sentence in the main text pointing at the appendix.

**13. Is any further planted-error experiment necessary?** **No.** Nothing here uncovered
an implementation defect; every registered validation passed exactly (CRN 0.000e+00,
sup-norms 1.000000, ESS ≥ 10.5, no `eta` bracket hits, no non-finite values), and the
`J = 1` arm independently replicated the committed amplitude experiment under a different
seed. The stop rule applies.
