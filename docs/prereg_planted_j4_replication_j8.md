# Pre-registration: J = 4 independent replication, and J = 8 extension

**Committed before either run is executed.** Seeds, grids, normalisation, decision rules
and correctness checks are fixed here and are not revised after outcomes are seen.

Branch `estep-study`. Prior work: `docs/prereg_planted_multimode.md` (`44c86a9`),
`reports/planted_multimode.md` (`d800a5d`), `reports/planted_error_mechanism.md`
(`1b9e05a`).

**These are appendix robustness checks. They do not change the main-paper result**
(`reports/planted_error_mechanism.md`, `fig_planted_mechanism.pdf`), and no outcome here
may be used to revise Claim 4.

Two questions only:

- **A.** Was the committed `J = 4` result reproducible, or could it have been a lucky
  draw?
- **B.** Does the same qualitative E-step / pathwise behaviour continue with eight waves?

---

## 1. Construction — unchanged, not redesigned

Exactly the field registered in `docs/prereg_planted_multimode.md` Sec. 2:

```
e(a) = sum_{j=1}^{J} c_j sin( omega v_j^T a + phi_j ),   c_j = A_0 / J,   A_0 = 1
```

`{v_j}` orthonormal (first `J` columns of the Q factor of a `d x J` Gaussian),
`phi_j ~ U[0, 2pi)` independent, one `omega` shared by all modes. Because the `v_j` are
orthonormal the sup-norms are attained exactly:

```
A_eff = ||e||_inf      = sum_j |c_j|             = A_0 = 1     for every J
L_eff = ||grad e||_inf = omega sqrt(sum_j c_j^2) = omega A_0 / sqrt(J)
omega_eff = L_eff / A_eff = omega / sqrt(J)
r_eff     = sigma omega_eff / sqrt(d)
=> omega  = r_eff sqrt(J d) / sigma
```

For `J = 8`: `c_j = 1/8`, `||e||_inf = 1`, `||grad e||_inf = omega/sqrt(8)`. **This
normalisation is fixed here and is not changed after results are seen.**

`d >= J` is required for orthonormal directions, so **`J = 8` uses `d in {16, 64}`
only**. No replacement `d = 4` arm is invented.

---

## 2. Experiment A — independent replication of J = 4

Identical to the committed run in every respect except the seed:

```
J in {1, 4}          (both arms, as in the committed run, CRN-paired per block)
A_0 = 1,  sigma = 0.4,  M = 32
d in {4, 16, 64}
r_eff in {0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 3.0}
64 error-field blocks x 512 action batches  = 32,768 batches per cell
run seed = 20260905      bootstrap seed = 20260905
```

Seeds `20260904` (committed run) and `20260905` are distinct, so directions, phases and
action samples are drawn afresh; nothing is reused.

## 3. Experiment B — J = 8

```
J in {1, 8}          (J = 1 carried alongside for CRN pairing and the trend)
A_0 = 1,  sigma = 0.4,  M = 32
d in {16, 64}
r_eff in {0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 3.0}
64 error-field blocks x 512 action batches
run seed = 20260906      bootstrap seed = 20260906
```

---

## 4. Statistics — unchanged from the committed experiment

All definitions are taken verbatim from `docs/prereg_planted_multimode.md` Sec. 4 and
the existing `measure_block` implementation. Nothing new is invented.

- **Primary** `B_b = ||delta_b|| / ||s_b||` with both expectations over action batches
  **inside** each fixed error-field block, the norm per block, then averaged across
  blocks. `B_J = E_field[B_b]`. Bootstrap over the 64 blocks, `B = 10,000`.
- **Noise** `N_b = tr Cov_action[Delta_err | e_b] / ||s_b||^2`.
- **Counterfactual decomposition**, existing procedure: `Err_baseline` (field off),
  `Err_erroroff` (`g0 + E[dg]`), `Err_cleanoff` (`E[g0] + dg`), `Err_total`, reported for
  PW, centred ZO and the E-step, with the summary ratio
  `systematic / noise = (Err_erroroff - Err_baseline) / (Err_total - Err_erroroff)`.
- **Update quality** `Err = 4 eps_TR (1 - cos)`, and `G = Err_Estep - Err_PW` against
  `r_eff`. **No crossover is fitted.**
- **Alignment** `cos(delta_b, v_sig)`, median and distribution.
- **Centred ZO/PW** `R_var = Var_e[ZO]/Var_e[PW]`, fitted against **both** `r_eff` and
  `r_nom = sigma omega / sqrt(d)`; report `beta`, `R^2` and the fitted crossing for each.
  Which convention collapses better is **not** assumed in advance.

---

## 5. Decision rules, fixed before the runs

### 5.1 Replication (Experiment A)

The committed value is `B_J4 = 0.172` (median across cells). Exact agreement is **not**
required.

> **Replication is declared successful if both hold:**
> 1. **Qualitative pattern preserved:** in the new run, pathwise has
>    `systematic/noise < 1` and the E-step has `systematic/noise > 1`.
> 2. **Magnitude consistent:** the new headline `B_J4` is within **20%** of 0.172,
>    i.e. in `[0.138, 0.206]`.

The 20% band is set from the seed-to-seed spread already observed: the `J = 1` arm of the
committed run agreed with the amplitude experiment within 2% at 21 of 24 cells and 13.7%
at the widest. Per-cell values and bootstrap intervals are reported either way.

If (1) holds and (2) fails, the result is reported as **qualitatively replicated,
quantitatively shifted**, with the shift quantified — not as a failure.

### 5.2 J = 8 (Experiment B)

> `J = 8` **supports the same story** if pathwise has `systematic/noise < 1` and the
> E-step has `systematic/noise > 1`, as at `J = 1` and `J = 4`.

The trend in `B` across `J in {1, 4, 8}` is reported descriptively. **A smaller `B_J8`
is not to be reported as mode cancellation.** Under this normalisation the RMS field
strength falls as `1/sqrt(J)`:

```
E_phi E_a[e^2] = sum_j c_j^2 / 2 = A_0^2 / (2J)
=> RMS_J / RMS_1 = 1 / sqrt(J);  RMS_8/RMS_1 = 0.354,  RMS_8/RMS_4 = 0.707
```

so part of any reduction is expected purely because the typical field is weaker. The
share is estimated using the **already-committed** amplitude experiment
(`reports/artifacts/planted_amplitude_replicates.csv`): fit `B ~ A^p` per `d` from
`A in {0.25, 1, 4}` at `J = 1`, then compare the measured `B_J/B_1` against the expected
`(1/sqrt(J))^p`. The residual is the part attributable to mode structure.
**No new amplitude sweep is run.**

---

## 6. Code-correctness checks — all must pass before J = 8 is interpreted

1. The `J` directions are orthonormal: `max |V V^T - I|` at machine precision.
2. Analytic `||e||_inf = 1` agrees with numerical verification (multistart L-BFGS plus
   dense probe; both give lower bounds, so the numeric/analytic ratio must be `<= 1` and
   is required to reach 1 to 6 decimal places).
3. Analytic `||grad e||_inf = omega / sqrt(J)` agrees with the same numerical check.
4. No NaN or infinite values anywhere.
5. No `eta` solver bracket hits (`1e-4`, `1e4`).
6. ESS healthy: min median ESS well above the floor of 2.
7. The clean-`Q^pi` results do not depend on `J`: the `wml0` checksum must be
   **bit-identical** across the two `J` arms within a block (CRN).
8. Experiment A reproduces the committed `J = 4` qualitative result (Sec. 5.1).

**If any of 1–7 fails, stop and report the defect before interpreting anything.**

---

## 7. Deliverables

`reports/artifacts/planted_j4_replication{,_replicates}.csv`,
`reports/artifacts/planted_j8{,_replicates}.csv`,
`reports/artifacts/fig_planted_j1_j4_j8.{pdf,png}` + `_data.csv`,
`reports/planted_j4_replication_j8.md`.

The report opens with eight one-line answers before any technical detail.

The figure is **appendix only** and does not replace `fig_planted_mechanism.pdf`.

## 8. Stop rule

After Experiments A and B, stop. No `J = 16`, no further extension, unless one of the
Sec. 6 checks reveals an implementation defect.
