# Pre-registration: planted-error multi-mode (J = 1 vs J = 4) experiment

**Committed before any J = 4 outcome is computed.** Construction, normalisation, grid,
budget, seeds, primary and secondary statistics, pathology criteria and interpretation
rules are fixed here and are not revised after outcomes are seen.

Branch `estep-study`. Prior work: `docs/prereg_planted_amplitude.md` (`4b88f03`),
`reports/planted_error_mechanism.md` (`1b9e05a`), `reports/artifacts/planted_amplitude.csv`.

**One question only:** is the actual MPO E-step's *systematic* response to critic error
an artifact of using a single coherent sinusoidal mode? This is **not** an attempt to
show the E-step has a universal `r = 1` crossover; the previous experiment showed it does
not, and that result stands.

Three objects are kept separate throughout: **(1)** centred ZO vs PW critic-error
variance; **(2)** the actual nonlinear E-step response to critic error; **(3)** final
normalised policy-update quality.

---

## 1. Blocking audit, resolved before this document was written

### 1.1 Was the previous systematic-displacement measure conditional or pooled?

**Conditional.** Read from `scripts/planted/decompose_sweep.py`:

- In `measure_block`, every statistic is computed from `X` of shape `(R, d)` where `R` is
  the number of **action batches inside one fixed field block** — `v_sig`, `v_e` and
  `phase` are drawn once per block, outside the batch axis.
- `mu = X.mean(0)` is therefore `E_action[· | field]`, and
  `out["pre_wml_err_meannorm"] = ||mu||` takes the **norm per block**.
- `main()` aggregates with `row[k] = np.mean([b[k] for b in blocks])` — the **per-field
  magnitudes are averaged across blocks**.

The same holds for the counterfactual terms: `err_wml_erroroff` uses `dg.mean(0)`, the
within-block action mean. So `||E Delta_mu_err|| = 0.0782` and the error-channel mean
contribution `0.2082` are conditional quantities. **No vectors were pooled across error
fields before taking a norm**, and no recomputation is required on that account.

### 1.2 One refinement that *is* required

The previously reported ratio 0.636 was
`E_field[||delta_b||] / E_field[||s_b||]` — a ratio of block-averages — not
`E_field[ ||delta_b|| / ||s_b|| ]`, the average of per-block ratios. These differ. The
per-block ratio is the quantity specified as primary here, so **`B` is computed
per-block in this experiment for both J = 1 and J = 4**, and the J = 1 value is reported
fresh rather than carried over.

### 1.3 What the existing "J = 4 configuration" actually is

`scripts/planted/planted_sweep_config.json` contains exactly one sentence:

> `"secondary": "J=4 superposed random directions with sup-norms calibrated numerically (robustness only)"`

plus `"eps_secondary": [0.1, 1.0]`. There is **no implementation** anywhere in
`scripts/planted/`, and the sentence fixes no `J`, no coefficients `c_j`, no frequencies,
no direction-sampling rule, no phase rule, and no normalisation. It also proposes
*numerical* sup-norm calibration, which Sec. 14 of the request rules out where an
analytic value exists.

**It is therefore not a pre-registration and is not treated as one.** It states an
intention, not a design. The construction below is specified here in full, before any
outcome is computed, and the reason for each choice is given.

---

## 2. Construction

### 2.1 The field

```
e(a) = sum_{j=1}^{J} c_j sin( omega v_j^T a + phi_j )
```

- `J in {1, 4}`.
- `c_j = A_0 / J` for all `j`, with `A_0 = 1.0` (the committed amplitude).
- `{v_1, ..., v_J}` **orthonormal**, drawn as the first `J` columns of the Q factor of a
  `d x J` standard Gaussian (Haar-distributed `J`-frame). Requires `d >= J`; satisfied
  by `d in {4, 16, 64}`.
- `phi_j ~ U[0, 2pi)` independent.
- **All modes share the single frequency `omega`.** A frequency spread would change
  `L_eff` in a way that is not separable from the mode-count effect, which is the one
  thing this experiment is meant to isolate.

`J = 1` reduces to `e(a) = A_0 sin(omega v_1^T a + phi_1)`, exactly the committed field.

### 2.2 Effective quantities — analytic, not calibrated

Because the `v_j` are orthonormal, the map `a -> (v_1^T a, ..., v_J^T a)` is onto `R^J`,
so the `J` phase arguments can be set independently. Hence both sup-norms are attained
**exactly**:

```
A_eff = ||e||_inf      = sum_j |c_j|              = A_0
L_eff = ||grad e||_inf = omega sqrt(sum_j c_j^2)  = omega A_0 / sqrt(J)
```

(the gradient is `sum_j c_j omega cos(theta_j) v_j`, and for orthonormal `v_j`,
`||grad e||^2 = sum_j (c_j omega cos theta_j)^2`, maximised at `cos theta_j = ±1`).

Claim 4 defines `omega := ||grad e||_inf / ||e||_inf`, so

```
omega_eff = L_eff / A_eff = omega / sqrt(J)
r_eff     = sigma * omega_eff / sqrt(d) = sigma * omega / (sqrt(J) sqrt(d))
```

**Verified numerically before this commit** (60-start L-BFGS on `-|e|` and
`-||grad e||`, plus a 400,000-point dense probe, at `(d, J, omega)` =
(4,1,5), (4,4,10), (16,4,20), (64,4,40), (16,1,20), (64,4,120)): analytic and numerical
agree to all printed digits (`A_eff` 1.000000, `L_eff` exact). The numerical search is a
**check on an exact formula**, not the source of the value.

### 2.3 Normalisation, and why

Two quantities enter Claim 4: `||e||_inf` and `||grad e||_inf`. The design holds
**both** matched between `J = 1` and `J = 4`:

- `c_j = A_0 / J` holds `A_eff = A_0` **exactly equal across J**, so `J = 4` is not
  simply a stronger critic error. (Contrast `c_j = A_0/sqrt(J)`, which would give
  `A_eff = A_0 sqrt(J)` — a 2x larger error field at `J = 4`, confounding the comparison.
  This is the defect the request's CASE 3 warns about, and it is avoided by construction.)
- `omega` is solved per cell from the **target `r_eff`**:

```
omega = r_eff * sqrt(J) * sqrt(d) / sigma
```

so the two arms are compared at equal `r_eff`, not equal nominal `omega`. Nominal
`omega` differs between arms by exactly `sqrt(J) = 2` and is reported alongside.

Under this normalisation `J = 4` differs from `J = 1` in **one** respect: the error
energy is split across four competing orthogonal modes instead of concentrated in one.
That is the intended contrast.

This is the request's **CASE 2** with the construction supplied here: the comparison is
interpretable using measured `A_eff` and `r_eff`, both of which are analytic and matched.

---

## 3. Design — 48 cells, nothing more

```
A          = A_0 = 1.0                    (single amplitude)
sigma      = 0.4                          (single value)
d          in {4, 16, 64}
r_eff      in {0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 3.0}
J          in {1, 4}
```

24 cells per arm, 48 total. Resulting nominal `omega = r_eff sqrt(J d) / 0.4`:

| `d` | `J=1` | `J=4` |
|---|---|---|
| 4 | `5 r` (2.5–15) | `10 r` (5–30) |
| 16 | `10 r` (5–30) | `20 r` (10–60) |
| 64 | `20 r` (10–60) | `40 r` (20–120) |

The `J = 1` arm reproduces the amplitude experiment's `A_0` configuration exactly in
`(d, sigma, omega, M)`, differing only in seed.

**Budget, unchanged from the amplitude experiment:** 64 independent error-field blocks x
512 action batches = **32,768 batches per cell**.

**Held fixed:** same `Q^pi` (`H = I`, `mu = 0`, `a* = G v_sig`, `G = 1`), `M = 32`,
`eps_E = 0.5`, same MPO dual `eta` solver, same trust-region step and metric
(`eps_TR = 0.1`), same baseline `Qbar`, same estimator implementations imported from
`planted_error_sweep.py` / `decompose_sweep.py`.

**Not run:** other amplitudes, dimensions, `sigma` values, `M`, `Q^pi`, curvature,
orientation, `J = 8`, random-Fourier.

### 3.1 Common random numbers

Per field block, drawn once and shared by both arms: the signal direction `v_sig`, the
orthonormal 4-frame `V`, the four phases `phi_1..phi_4`, and the action draws `u`.
`J = 1` uses `(V[0], phi_1)`; `J = 4` uses all four rows. Within an arm the same `u`
serves the clean and contaminated critic.

`J = 1` is **not** nested inside `J = 4` (the coefficients and `omega` both differ by
construction), so this is coupling of the random inputs, not nesting. It is nonetheless
a valid pairing at block level — block `b` sees the same policy noise, the same true
signal and the same first error direction in both arms — so the paired ratio
`B_J4/B_J1` is computed per block and its interval is a **paired** bootstrap.

**Seeds:** run `20260904`; bootstrap `20260904`.

---

## 4. Statistics

### 4.1 Primary — conditional systematic E-step displacement

For each field block `b`:

```
delta_b = E_action[ Delta_mu_Estep(Q^pi + e_b) - Delta_mu_Estep(Q^pi) | e_b ]
s_b     = E_action[ Delta_mu_Estep(Q^pi) | e_b ]
B_b     = ||delta_b|| / ||s_b||
B_J     = E_field[ B_b ]
```

Reported: mean, median, 25th/75th percentiles, full across-block distribution, and a 95%
bootstrap CI over the 64 blocks. Cell-level `B_J` is the mean over blocks; the headline
per arm is the median across the 24 cells. `B_J4 / B_J1` is computed **per block** and
its 95% interval is a paired bootstrap over blocks.

### 4.2 Secondary — conditional error-channel noise

```
N_b = tr Cov_action[ Delta_mu_Estep(Q^pi + e_b) - Delta_mu_Estep(Q^pi) | e_b ] / ||s_b||^2
N_J = E_field[ N_b ]
```

`B` and `N` are both normalised by the same `||s_b||`, so they are dimensionless and
directly comparable — the normalisation the previous report established as necessary.

### 4.3 Operational

The existing normalised directional metric, unchanged:
`Err = 4 eps_TR (1 - cos(ghat, g*))`. Report `Err_Estep`, `Err_PW`, and
`G = Err_Estep - Err_PW` **as a function of `r_eff`**, with bootstrap intervals, for
each `J`. **No operational crossover is fitted or quoted as a threshold.** The reported
question is only whether `G` shrinks, stays similar, or grows from `J = 1` to `J = 4`.

### 4.4 Counterfactual decomposition — the existing procedure, unchanged

Exactly as `decompose_sweep.py` already implements (prereg_planted_amplitude Sec. 2.7):
`Err_baseline` (field off), `Err_erroroff` (`g0 + E[dg]`), `Err_cleanoff`
(`E[g0] + dg`), `Err_total`, all with the within-block action mean. Reported for PW, ZO
and the E-step at both `J`.

### 4.5 Cross-field cancellation

Both reported, per arm:

```
(A) conditional:  E_field ||delta_b||
(B) pooled:       || E_field delta_b ||
```

and the ratio (B)/(A). Also reported: the component of the pooled mean along the block's
own signal direction, `E_field[ delta_b . v_sig ]`, since an isotropic error direction
makes the raw pooled vector vanish by symmetry while a systematic signal-direction effect
would not.

### 4.6 Centred ZO/PW theory check — secondary

`R_var = Var_e[ZO]/Var_e[PW]` under the paired CRN definition, against `r_eff`. Fit
`log R_var = alpha + beta log r_eff`; report `beta`, 95% bootstrap CI, `R^2`, fitted
crossing, and counts either side of `r_eff = 1`. **Language rule:** the measured scaling
is reported as *consistent with the ratio of the two bounds Claim 4 supplies*; it is
never written as implied by the claim.

---

## 5. Pathology and validation criteria

Reported for every cell and arm regardless of outcome:

1. min / median effective sample size `ESS = 1/sum_i w_i^2`; floor 2.
2. `eta` solver hitting either bracket bound (`1e-4`, `1e4`).
3. Any non-finite value.
4. `A_eff`, `L_eff`, `omega_eff`, `r_eff`, nominal `omega` — analytic values, with the
   numerical sup-norm check of Sec. 2.2 re-run in the sweep and its deviation reported.
5. Orthonormality residual `||V V^T - I||_max` of the frame.
6. CRN check: `v_sig`, `V`, `phi`, `u` bit-identical across arms within a block; the
   clean-channel columns bit-identical across arms.
7. Consistency check: the `J = 1` arm's cell-level `Err` values must agree with
   `reports/artifacts/planted_amplitude.csv` at `A = 1.0`, `sigma = 0.4` within
   Monte-Carlo error (different seed, so not bit-identical).

If any criterion trips, it is documented before the affected arm's outcomes are
interpreted, and affected cells are reported rather than dropped.

---

## 6. Interpretation rules, fixed before the run

Applied to the headline `B_J4 / B_J1` (median across cells of the per-block paired
ratio) together with the counterfactual decomposition of Sec. 4.4.

**CASE A — multi-mode bias persists.** `B_J4/B_J1 >= 0.5`, and at `J = 4` the E-step's
error-channel systematic-mean contribution still exceeds its error-channel noise
contribution. → The systematic displacement is not a peculiarity of a single coherent
mode; the mechanism stays in the main paper.

**CASE B — partial attenuation.** `0.1 <= B_J4/B_J1 < 0.5`, systematic displacement
still material and the variance-vs-update mismatch still present. → Multiple modes
partially cancel the E-step's systematic displacement, but the distinction between
suppressing critic-error variance and producing a correct update survives. Report the
attenuation factor.

**CASE C — strong cancellation.** `B_J4/B_J1 < 0.1` **and** the operational gap `G` at
`J = 4` is within its 95% interval of zero over most of the `r_eff` range. → The large
`J = 1` systematic displacement is primarily a coherent single-mode phenomenon. The
paper must narrow the mechanism claim to a diagnostic example, while keeping (i) the
centred ZO/PW Claim-4 validation and (ii) the result that the actual E-step is not
governed by `r` alone. **This outcome is reported as prominently as any other.**

If the decomposition and `B` ratio point to different cases, both are reported and the
more conservative case is adopted.

---

## 7. Deliverables

`reports/artifacts/planted_multimode.csv`, `planted_multimode_replicates.csv`,
`fig_planted_multimode.{pdf,png}`, `fig_planted_multimode_data.csv`,
`reports/planted_multimode.md`.

The figure is **not** an automatic replacement for the current main-paper figure; the
report ends with an explicit main-paper / appendix / omit recommendation.

## 8. Stop rule

This is the final planted-error experiment. No `J = 8`, no random-Fourier sweep, no
further `M`, curvature, orientation, amplitude or `sigma` levels — unless this run
uncovers an implementation defect that invalidates the existing controlled evidence.
