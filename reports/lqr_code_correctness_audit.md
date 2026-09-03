# LQR crossover experiment: forensic correctness audit

**Gate order.** Parts 2 and 3 (the two estimators and their linearity) were run and
passed before any downstream number was interpreted; had either failed, this audit would
have stopped there with verdict D, because every later part is computed on those
objects. A precision check (TEST K, below) was run first of all, since a float32 sweep
would have invalidated every small-`sigma` cell before any rule could be read.

**Independence.** The test suite derives every quantity from first principles in numpy
and compares it to the production core, the sweep kernel's arithmetic, the saved `.npz`,
and the report. The first audit's artifacts (`lqr_audit_rms.csv`,
`lqr_audit_summary.json`) are used **only as comparison targets**, never as inputs or as
sources of expected values; where the two agree it is because both reach the same number
from different code. Both audits were written by the same hand, which is a residual
blind-spot risk this document cannot remove; it is why every check is committed and
rerunnable rather than asserted.

**Rerun.** `JAX_PLATFORMS=cpu python scripts/lqr_crossover/audit_correctness_tests.py`
regenerates `reports/artifacts/lqr_code_correctness_checks.csv` (132 checks) and
`lqr_code_correctness_part6.csv` from the committed script and the manifest-hashed
`.npz` in ~6 minutes on CPU.

## Verdict

**C. PARTIALLY CORRECT.** The central estimator computation — pathwise, centred
zeroth-order with one `M/(M-1)` de-attenuation, the error-channel variance
`tr Cov[g(Q^pi+e) - g(Q^pi)]` isolated under shared randomness, the crossover root, the
dimension fit, and the finite-`M` constant — is mathematically correct and was
independently reproduced from first principles at machine or Monte-Carlo precision.
Two analysis choices materially change claimed results: the cross-term diagnostic was
rewritten in code at the results commit from the registered rule to a different one
(the registered rule flags every `d`, the reported one flags `{1, 2, 4}`), and the
report's "c* analytic" column is evaluated with the wrong phase convention
(`theta = phi` instead of `theta = omega v^T mu + phi`) and a different aggregate, so it
is not the closed form the measurement should be compared to. No error forces a rerun.
No error touches the primary error-channel comparison.

Everything below was recomputed by `scripts/lqr_crossover/audit_correctness_tests.py`
(temporary audit code; numpy re-implementations compared against the production core
`src/jaxrl/estimators.py`, the sweep kernel, the saved `.npz`, and the report). Results:
`reports/artifacts/lqr_code_correctness_checks.csv` (132 checks) and
`lqr_code_correctness_part6.csv`. Nothing in production was modified; no GPU; no new
sweep.

---

## Summary table

| component | intended math | implemented math | independent test | verdict | scientific consequence |
|---|---|---|---|---|---|
| dtype / precision | float64 throughout; `jax_enable_x64` set before any array exists | `scripts/lqr_crossover/__init__.py` sets `JAX_PLATFORMS=cpu`, `jax_enable_x64=True`, and **asserts** `jnp.zeros(1).dtype == float64` at import; `sweep.py` imports it before `jax`; kernel inputs are `jnp.asarray` of float64; no `float32`/`astype` anywhere in the harness | TEST K: x64 on, `jax.random.normal` → float64, platform cpu; **all statistics in all 28 cited sweep `.npz` are float64**; independent recompute of the `sigma=0.01, omega=0.1` corner vs saved `s3`: PW 4.4 %, ZO 0.2 % (MC bound 8.7 %); corner power laws `s3_pw ∝ c^4.09`, `s3_zo ∝ c^1.92`, `s2_pw ∝ c^2.05`, `s2_zo ∝ c^0.96` (analytic 4, 2, 2, 1) | correct; corner is signal | the registered cross-term ratios of 1e5–1e6 are a mathematical property of that rule (`∝ c^-2`), not a precision artifact; no cell needs to be marked unreliable |
| LQR `Q^pi` | `Q = -a^T H a + b^T a + c(s)`, `H = Rc + gamma B^T P B`, `P` from the discounted Lyapunov equation; normalised critic `alpha_Q Q^pi` at states `alpha_s s` with `\|\|alpha_Q H\|\|_2 = 1` | `lqr.py`: as intended; sweep uses the reduced form `Q(mu+sigma u) - Q(mu) = -sigma^2 u^T H u + sigma u^T g*` | FD vs `grad_a_q_pi` **3.9e-8**; reduced form vs `alpha_Q grad_a Q^pi(alpha_s s, a)` **7.3e-16**; kernel `qq/ql` form vs `Q(mu+su)-Q(mu)` **2.8e-13**; `\|\|alpha_Q H\|\|_2 - 1` ≤ 1e-12 | correct | none |
| PW estimator | `(1/M) sum_i grad_a Q(mu + sigma u_i)` | kernel: `-2 sigma H ubar` (smooth, exact) and `(c/sqrt r)(mean cos - damp cos theta) V` (error); production core: `vjp` mean `/ sigma` | my numpy PW vs core **1.0e-14**; linearity in `Q` **4.0e-14** | correct | none |
| centred ZO | `(1/(sigma M)) sum_i (Q_i - Qbar) u_i`, `Qbar` = same batch's mean, centred **before** multiplying by `u` | kernel: `(qc[...,None]*u).mean(1) * M/(M-1) / sig`; core `centred_zo(..., deattenuate=True) / sigma` | my numpy ZO vs core **8.9e-16**; linearity **3.9e-14** | correct | none |
| de-attenuation | `E[plain] = (1-1/M) h` for a linear critic; multiply by `M/(M-1)` once | applied once, in the kernel; in the core only when `deattenuate=True` (both call sites checked) | `E[plain]/h = 31/32` and `E[deatt]/h = 1` within 0.44 SE (4e5 batches); `deatt/plain = 32/31` to 2e-16; `trCov` ratio `(32/31)^2` to 4e-15 | correct, once | mean unbiased, variance × `(M/(M-1))^2`; sets the constant in Part 6 |
| error-channel variance | `tr Cov[g(Q^pi+e;xi) - g(Q^pi;xi)]` | kernel computes `g(e;xi)` directly and stores `s3 = \|\|g(e) - E g(e)\|\|^2` with the **exact** blurred mean; equals the difference by linearity | `A = g(Q^pi+e)-g(Q^pi)` vs `B = g(e)` on identical `u`: **≤ 1.3e-15**; my `tr Cov(B)` vs saved `s3` (3 cells, `d=4`, state 0): 0.3–1.6 % (MC noise, bound 7.9 %); sample mean of `B` vs analytic blurred gradient ≤ 1.5 SE | correct; a variance about the exact mean, **not** `E\|\|g\|\|^2` | primary comparison is the right quantity |
| RMS frequency | `omega_RMS^2 = E\|\|grad e\|\|^2 / E[e^2]` with `theta_j = omega v_j^T mu + phi_j` | `lqr_audit_rms.csv` (closed form at realised phases) | my closed form vs 2e5-draw MC **4.9e-3** (rank 1/2/full); vs the CSV **1.1e-16**; `omega_RMS/omega` at crossover cells 1.0000–1.0034 | correct | nominal `omega` **is** the RMS frequency at every crossover cell |
| crossover extraction | root of `log Var_ZO_e - log Var_PW_e = 0` in `log c`, interior bracket | `solve_crossover`: sign change of the decreasing log-ratio, PCHIP + `brentq` on `[logc_k, logc_{k+1}]`, `xtol 1e-13`, rejects roots in the first/last interval and any non-bracketed case; first crossing taken | exactly one sign change at every `d`; my PCHIP root = report `c*` to **≤ 2.2e-16**; PCHIP vs linear interpolation differ 1.7e-6 (d=64) … **2.5e-3 (d=2)** | correct; interpolation error ≈ 0.1–0.25 % at `d ≤ 4` | affects Rule A's SE clause at `d=2` (SE 0.12 %) |
| state aggregation | prereg 5.4: "root … per state before any aggregation" | `crossover_by_c(z)` averages the per-state log-ratio over states, then roots once | root-of-mean vs median-of-per-state-roots: **8.7e-5 … 7.2e-4**; mean-of-roots within 1e-3; 0 % states without bracket | departure, numerically immaterial | none |
| dimension fit | OLS of `ln c*` on `ln d`, unweighted, `d in {2,4,8,16,32,64}` | `fit_p` on `FIT_DS`; `d=1` excluded; `d=6` cannot enter (`report.py` loads `DS` only) | my refit **0.4876 / 0.4870 / 0.4871** (rank1 / r=2 / full) = report to 5.6e-17; with `d=1`: 0.4680 | correct | none |
| rank normalisation | `\|\|e\|\|_inf = eps sqrt r`, `sup \|\|grad e\|\|_2 = eps omega`, `max_j sup \|d_j e\| = eps omega/sqrt r` (coordinate basis) | `error_field.omega_inf`: `omega/sqrt r` (L2/inf), `omega/r` (inf/inf) for any `V` | `V` orthonormal to ≤ 1e-13 for rank1/rank_r/full; `rank_r2` at `d=1` has `r=1` | correct for `V=I`; the componentwise reading is only exact in the coordinate basis (full arm), an upper bound for random `V` | none for the registered primary |
| full-rank exponent | fit of `ln c*_inf` where `c_inf = c/sqrt r`, `r=d` | `report.py`: `p + (-0.5)` hardcoded | refit of `ln(c*/sqrt d)` gives `p - 0.5` to 2.7e-16; **−0.0129 = 0.4871 − 0.5 exactly** | a deterministic relabelling, printed in a column headed "p (omega_inf …)" beside fitted values | the nominal/RMS crossover is unchanged at full rank; only the frequency normalisation changed |
| cross term | prereg 5.4: flag if `\|2Cov\| > 0.25 Var_e` **anywhere on the swept grid** | at `33c7632`, `analyze.cross_share` computed exactly that; at `98da377` (the results commit) it was replaced by an at-crossover ratio and a `frac_of_total`; `report.py` prints the at-crossover one | registered rule from saved `s2/s3`: PW 7.1e5…3.3e5, ZO 5.7e4…5.9e3 → **every `d` flagged**; at crossover `{1, 2, 4}`; `\|2Cov\|/MSE_total` (`sigma ≥ 0.1`) ≤ 0.21 PW / 0.026 ZO → none | **post-outcome replacement of the registered criterion**; the primary error-channel quantity needs no cross term | changes the reported flag set (a claimed result) |
| bootstrap | hierarchical: 64 states with replacement, then 40 batches within each; recompute `c*` and refit `p` per resample; 10 000 | `bootstrap_p`: as registered for `level=None`; 10 000 for both-level, **3 000** for the two single-level runs; CI only for rank1 | code read + `bootstrap_p.json` | correct procedure, short count on two levels, no CI for `r=2`/full | Rule B's "CIs excluding 0 and 1" unverified off rank1 |
| actual E-step | `w ∝ exp(Q/eta)`, M-step `argmin −Σ w_i log π_θ(a_i)` | LQR arm: `d = Σ w_i u_i`, `eta` grid-minimised on the same dual; production: one optimizer step on `−Σ w_i log π_θ(clip(tanh y_i))` with `eta` learned by the dual's gradient, `σ` fitted too, KL clip | `d_ES = ubar + sigma·zo_plain/eta + O(eta^-2)` holds (rel 7e-4 at `eta=1e3`); `cos(d_ES, g_ZO)` = −0.09 / +0.02 / +0.62 / +0.87 at `eta` = 1e3 / 10 / 1 / 0.1 | the "ZO" correspondence is first-order in `1/eta` and, at that order, dominated by `ubar`; qualitative alignment only at small `eta` | statements that the E-step "implements ZO" are asymptotic-and-weak, not exact; the LQR arm correctly measures cosine to `g*` instead |
| provenance | every number from `33c7632` code on `33c7632` data | all 34 sweep `.npz` and six `estep` files embed `git_sha = 33c7632`; `estep_d64` embeds `98da377` (regenerated twice); `analyze.cross_share` rewritten at `98da377`; `msweep_hi/hiw.py`, `probe2_full_estimator.py` untracked until `c0a1ff1`; `.npz` ignored, manifest only; `ceil(sqrt d)` rank arm and E2 never run; `reports/lqr_crossover.md` regenerates **byte-identically** from the current tree | — | reproducible, with the caveats listed | none on estimator-level numbers |

**Severity of each discrepancy**

| # | discrepancy | severity |
|---|---|---|
| 1 | Cross-term criterion replaced in code at the results commit; registered rule would flag all `d`, report flags `{1,2,4}` | **MAJOR** |
| 2 | Report's "c* analytic" column uses `theta = phi` (dropping `omega v^T mu`, median magnitude 13.5 at `omega=5`) and a median-of-per-state-roots aggregate; Rule A's 4-SE clause never computed and would fail at `d=2` (16 SE), `d=4` (4.7 SE) against that column; passes (≤ 1.8 SE) against the correctly aggregated closed form | **MAJOR** (a printed quantity is wrong; the verdict survives) |
| 3 | `p (omega_inf) = -0.0129` is `p - 0.5`, printed beside fitted exponents under a heading that reads as a fit | MINOR |
| 4 | Rule B ladder missing the registered `ceil(sqrt d)` rung; CI clause unverified for `r=2`; "CONFIRMED at fixed rank" is not a registered outcome | MINOR |
| 5 | §7.2 table prints `nan` (`ladders.get("full")` vs key `"full rank"`) | MINOR |
| 6 | `estep_d64_rank1_M32.npz` regenerated at `98da377` after the other six; two index rows were uncommitted | MINOR |
| 7 | `msweep_hi.py`, `msweep_hiw.py`, `probe2_full_estimator.py` untracked when §7.4 / G6b results were produced | MINOR |
| 8 | Bootstrap 3 000 resamples on two of three levels (registered 10 000) | MINOR |
| 9 | Two of four registered norm conventions stored as `nan` | MINOR |
| 10 | E2 (prereg 5.5) never run, not reported as not run | MINOR |
| 11 | Root of the state-averaged ratio instead of per-state roots (≤ 7e-4) | COSMETIC |
| 12 | `estep_arm.py` `eta` grid capped at 10 (msweep uses 1e4); saturation not checked | COSMETIC (ESS/M 0.46–0.57 implies interior) |
| 13 | `rank_r2` "componentwise" `omega_inf` assumes `V = I`; exact only for the full arm | COSMETIC (unused: fixed rank gets shift 0) |

---

## TEST K — dtype and the small-`sigma` corner

**Why it matters.** The registered cross-term rule reaches `7e5` at `sigma = 0.01,
omega = 0.1`, where `Var_PW_e ∝ c^4` with `c = 10^-3`. If the kernel had run in float32
those cells would be rounding noise and the registered rule's verdict would be
uninterpretable.

**Configuration path.** `scripts/lqr_crossover/__init__.py:29–41` sets
`JAX_PLATFORMS=cpu` and `jax_enable_x64=True` and **asserts** at import that
`jnp.zeros(1).dtype == float64` and the platform is CPU; `sweep.py` imports that package
before `jax` (line 14 vs 16); `src/jaxrl/__init__.py` sets only
`jax_default_matmul_precision="highest"`; the estimator core sets nothing; the kernel's
inputs are `jnp.asarray` of float64 numpy and `u = jax.random.normal(...)` (float64 under
x64); no `float32` or `astype` occurs in the harness.

**Realised precision.** In the audit process, after the same import: x64 on,
`jax.random.normal` emits float64, `jnp.asarray(np.float64)` stays float64, platform cpu.
Every one of `s1_pw, s1_zo, s2_pw, s2_zo, s3_pw, s3_zo, eps, sigmas, omegas` in all 28
cited sweep `.npz` is stored as float64 (numpy preserves the JAX dtype, so float32
arithmetic would have left float32 files).

**Is the corner real?** The kernel never forms the physical variance; it stores `s3` in
units of `(eps/sigma)^2` with the amplitude factored out, so at the corner the stored
`s3_pw = 1.4e-14` and `s3_zo = 9.1e-8` (state-averaged; physical `Var_PW_e = 4.5e-17`,
`Var_ZO_e = 2.9e-10` after the `3.3e-3` prefactor). The delicate operation is
`mean_i cos(theta + c t_i) − e^{−c²/2} cos theta`, an O(1)−O(1) subtraction that leaves
~2e-4, i.e. a loss of ~4 digits — negligible in float64's 16 and survivable even in
float32's 7. Independent recomputation of that cell (`d=4`, state 0, 2e4 fresh batches):
float64 reproduces the saved `s3_pw` to 4.4 % and `s3_zo` to 0.2 % (MC bound 8.7 %);
a float32 recompute reproduces them equally (4.4 %, 0.2 %). Along the `sigma = 0.01`
column the saved statistics follow the analytic power laws — `s3_pw ∝ c^{4.09}`,
`s3_zo ∝ c^{1.92}`, `|s2_pw| ∝ c^{2.05}`, `|s2_zo| ∝ c^{0.96}` against 4, 2, 2, 1 — which
rounding noise cannot do.

**Consequence.** No small-`sigma` cell is unreliable. The registered rule's ratios of
`1e5–1e6` are what the mathematics gives (`|2Cov|/Var_e ∝ c^{-2}` for PW, `c^{-1}` for
ZO), so the rule is unbounded *by construction*, not by precision — which is exactly why
its verdict ("every `d` flagged") and the report's post-hoc replacement must both be
stated rather than either being silently adopted.


## Part 0 — map

| quantity | equation | implementation | saved variable | analysis | reported |
|---|---|---|---|---|---|
| `Q^pi(s,a)` | `-a^T H a + b^T a + c` | `lqr.q_pi`, `grad_a_q_pi`, `q_coeffs` (normalised) | `alpha_Q`, `alpha_s`, `cond_H`, `tr_H2` | — | §0 table |
| `e(s,a)` | `eps/sqrt r Σ sin(omega v_j^T a + phi_j)` | `error_field.e_value`, `e_grad`; kernel `per_omega` | `phi`, `vtmu`, `rank`, `eps` | — | — |
| `theta_j` | `omega v_j^T mu + phi_j` | kernel `th = om*vtmu + phi`; `EF.theta` | `phi`, `vtmu` | `report.py` passes **`phi[:,0]`** to `reference.crossover_c_star` | §3 "c* analytic" ⚑ |
| `g_PW(Q^pi)` | `(1/M)Σ grad Q^pi` | kernel `-2 sig (ubar @ H)` | `s1_pw = \|\|D_pw\|\|^2` | `full_mse` | §4 |
| `g_ZO(Q^pi)` | `(M/(M-1))(1/(σM))Σ(q_i - qbar)u_i` | kernel `smooth()` | `s1_zo` | `full_mse` | §4 |
| `g_PW(e) - E` | error-only pathwise, exact mean subtracted | kernel `de_pw` | `s3_pw` (units `(eps/σ)^2`) | `err_only` → `log_ratio_by_c` | §3 |
| `g_ZO(e) - E` | de-attenuated centred ZO of `f`, exact blurred gradient subtracted | kernel `de_zo = b - tgt` | `s3_zo` | same | §3 |
| cross term | `2 (eps/σ) <D_sm, De_unit>` | kernel `s2_pw/s2_zo` `(S, sig, om)` | `s2_*` (diagonal in `sigma` is physical) | `cross_share` (rewritten at `98da377`) | §5 ⚑ |
| `c*(d)` | root of state-averaged `log(s3_zo/s3_pw)` collapsed on `c` | `crossover_by_c` | — | `solve_crossover` | §3 |
| `p` | OLS slope, `FIT_DS` | `fit_p` | `bootstrap_p.json` | — | §3, §6, §7.1 |
| `p_omega_inf` | — | `report.py`: `pn + (-0.5 if full else 0)` | — | — | §7.1 ⚑ (relabelling) |
| `omega_inf_factor` | 4 conventions | `EF.omega_inf` for `inf` only; `nan` for `rms` | `omega_inf_factor` | — | §7.2 (prints `nan`) ⚑ |
| E-step | `Σ w_i u_i / σ`, `eta` from dual | `estep_arm.py` (grid), `msweep*.py` | `estep_d*.npz`, `m_sweep_estep*.json` | — | §7.3, §7.4 |

Ambiguities flagged: the `theta` convention in the report's closed-form column; the
`p (omega_inf)` column's status; and the cross-term function's identity across commits.

---

## Part 1 — mathematics re-derived

**`Q^pi`.** With `s' = As + Ba + w`, `r = -(s^T Qc s + a^T Rc a)`, `pi = N(-Ks, σ^2 I)`,
`V^pi = -(s^T P s + v)`, `P = Qc + K^T Rc K + γ A_K^T P A_K` (`solve_P` passes
`sqrt(γ) A_K^T` to `solve_discrete_lyapunov`, which solves `X = aXa^T + q` — correct).
`Q^pi(s,a) = -s^T Qc s - a^T Rc a - γ E[(As+Ba+w)^T P (As+Ba+w) + v] + const` gives
`H = Rc + γ B^T P B`, `b = -2γ B^T P A s`, `grad_a Q = -2Ha + b`. FD: **3.9e-8** relative.
Under `unit_H`, `q_coeffs` returns `(alpha_Q H, alpha_Q grad_raw(alpha_s s), -K alpha_s s)`;
the kernel's `Q(mu+σu) - Q(mu) = -σ^2 u^T H u + σ u^T g*` holds to **2.8e-13**;
`||alpha_Q H||_2 = 1` to 1e-12. `q_spread^2 = σ^2||g*||^2 + 2σ^4 tr H^2`: MC agrees to 2e-3.

**`e` and `grad e`.** `grad e = (eps ω/sqrt r) Σ cos(·) v_j`; implementation vs an
independent sine field **2.7e-15**, FD **2.9e-10**. `V V^T = I` to 1e-13 (rank1, rank_r,
full); `rank1 → r=1`, `rank_r --rank 2 → r = min(2,d)` (so `d1_rank_r2` has `r=1`),
`full → r=d, V=I`. Saved `eps = 0.05 q_spread` exactly (0 deviation on replayed states);
saved `phi`, `vtmu` replay bit-exactly from the seeds.

**Phase.** The kernel uses `theta_j = ω v_j^T mu + phi_j` (correct; verified 7e-15). The
report's closed-form column uses `theta = phi`. They are not interchangeable: at `ω = 5`,
median `|ω v^T mu| = 13.5` radians. This is the source of discrepancy #2.

## Part 2 — the estimators, from one batch

Sampled: `u ∈ R^{M×d} ~ N(0, I)`, `M = 32`; the same `u` serves `Q^pi`, `e`, and `Q^pi+e`
(kernel: one `u` per batch, `T = u V^T` reused). `Qbar` is the same batch's mean; centring
precedes the multiplication by `u`; the factor `M/(M-1)` is applied once (kernel line
`* (M / (M - 1.0)) / sig`; the core's `deattenuate=True` at every LQR call site, `False`
only in probe2, which stores the uncorrected estimate by design); it scales the mean by
`M/(M-1)` and the variance by `(M/(M-1))^2` (verified 2e-16 / 4e-15); `/σ` once, at the
end; means over the sample axis (`mean(1)`), then squared Euclidean norm summed over `d`
= `tr Cov` when the subtracted mean is exact.

**Derivation.** For `q_i = σ u_i^T h` (linear critic):
`E[(1/M)Σ q_i u_i] = σ h`, `E[qbar ubar] = (1/M^2)Σ_{ij} E[q_i u_j] = σ h/M`, so
`E[plain] = (1 − 1/M) h`; de-attenuated: `h`. Measured: **31/32** and **1** within 0.44 SE.
The implemented estimator is the de-attenuated convention; the finite-sample centring
factor at `M=32` is `31/32` on the mean, `(32/31)^2 = 1.0656` on the variance after
correction.

## Part 3 — linearity

Both operators are linear in `q` (centring is a linear projection). With identical `u`:
`g(Q^pi+e) − g(Q^pi) − g(e)` ≤ **4.0e-14** (PW) / **3.9e-14** (ZO) in the production
core, and ≤ 4.3e-14 in my numpy versions; core vs mine ≤ 1.0e-14. The identity
`g(Q^pi+e;ξ) − g(Q^pi;ξ) = g(e;ξ)` **is true in the implementation** to float64 rounding.

## Part 4 — error-channel variance

The code stores `s3 = ||g(e;ξ) − E g(e)||^2` with the exact blurred gradient as the mean
(PW: `damp cos θ`; ZO: `tgt = (c damp/sqrt r)(cos θ @ V)`), so `mean_ξ s3 = tr Cov`. It is
**not** `E||g||^2`, and both operators subtract their exact means the same way.
Independent recomputation (numpy, fresh draws, `d=4`, state 0, three cells): `A = B` to
≤ 1.3e-15; `tr Cov(B)` vs saved `s3`: PW 1.6 %, 1.0 %, 0.1 %; ZO 0.8 %, 0.3 %, 1.3 %
(MC noise, bound 7.9 % at 5 combined SE); sample mean of `B` matches the analytic
blurred gradient within 1.5 SE for both. The de-attenuated centred ZO is exactly unbiased
for the blurred gradient of *any* `e` (Stein: `E[e(mu+σu) u] = σ E[grad e]`), which is why
subtracting `tgt` is legitimate.

## Part 5 — sine-field moments

With `a = mu + σu`, `v_j^T a = v_j^T mu + σ t_j`, `t_j` iid `N(0,1)` (orthonormal `V`):
`E[e^2]/eps^2 = (1/r)[Σ_j (½ − ½e^{−2c^2} cos 2θ_j) + e^{−c^2} Σ_{i≠j} sin θ_i sin θ_j]`,
`E||grad e||^2/eps^2 = (ω^2/r) Σ_j (½ + ½ e^{−2c^2} cos 2θ_j)`, `θ_j = ω v_j^T mu + φ_j`.
MC (2e5 draws, realised phases, ranks 1/2/`d`): **4.9e-3** max relative; vs
`lqr_audit_rms.csv`: **1.1e-16**. `ω_RMS/ω` at the crossover cells: 1.0000 (d=1), 1.0034
(d=2), 1.0001 (d=4), then 1.0000 — the claim holds; `θ = φ` was not assumed.

## Part 6 — the variance law

Independent derivation for the de-attenuated estimator: `Var_PW ≈ E||grad e||^2/M`;
`Var_ZO ≈ (M/(M−1))^2 (M−1)/M · d E[e^2]/(Mσ^2) = d E[e^2]/((M−1)σ^2)`; hence
`R r_RMS^2 → M/(M−1) = 32/31 = 1.03226`, and the crossover sits at `r_RMS = sqrt(32/31)
= 1.0160` (this is `reference.py`'s `sqrt(dM/(M−1))`). **31/32 would be right only for the
plain centred estimator** (as in the planted sweep). Measured medians over cells with
`r_nom ∈ [0.5, 3]`:

| arm | d=1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| rank1 | 1.438 | 1.142 | 1.037 | 1.0331 | 1.0320 | 1.0332 | 1.0323 |
| r=2 | 1.446 | 1.108 | 1.037 | 1.0363 | 1.0337 | 1.0332 | 1.0325 |
| full | 1.440 | 1.123 | 1.041 | 1.0315 | 1.0321 | 1.0324 | 1.0328 |

Relative error vs `32/31` at `d ≥ 8`: **≤ 3.9e-3** (every arm); vs `31/32` it would be
+6.5 %. Convergence to the analytic constant is established. Small-`d` excess is the
along-`v` term `Λ ≠ Vsin/(M−1)` whose share is `1/d`. Full table with `R`, `r_RMS`, min/max:
`lqr_code_correctness_part6.csv`.

## Part 7 — crossover extraction

Root-found: `log Var_ZO_e − log Var_PW_e` (ZO/PW; equality ⇔ 0), in `log c`, PCHIP on the
53-point collapsed grid, `brentq` on the bracketing interval, `xtol = 1e-13`; roots in the
first or last interval and non-bracketed cases return `nan`; the first sign change is
used (exactly one exists at every `d`). Aggregation: **the prereg registers per-state
roots; the report's values are the root of the state-averaged ratio.** Both from the
same artifacts:

| d | root of mean (report) | median of per-state roots | mean of roots | no bracket |
|---|---|---|---|---|
| 1 | 1.2014 | 1.2015 | 1.2027 | 0 % |
| 2 | 1.5182 | 1.5193 | 1.5185 | 0 % |
| 4 | 2.0495 | 2.0483 | 2.0496 | 0 % |
| 8 | 2.8746 | 2.8755 | 2.8747 | 0 % |
| 16 | 4.0603 | 4.0605 | 4.0605 | 0 % |
| 32 | 5.7465 | 5.7437 | 5.7468 | 0 % |
| 64 | 8.1306 | 8.1339 | 8.1307 | 0 % |

My PCHIP root equals the report's to 2.2e-16; linear interpolation differs by 0.1 % (d=1),
0.25 % (d=2), 0.16 % (d=4), <0.04 % beyond — the interpolation error is of the same order
as the bootstrap SE at `d ≤ 4`.

## Part 8 — exponent

Natural-log OLS, unweighted, `d ∈ {2,4,8,16,32,64}`; `d=1` excluded (secondary: 0.4680);
`d=6` cannot enter. Independent refits: rank1 **0.4876**, r=2 **0.4870**, full **0.4871**
(match to ≤ 5.6e-17); with per-state-median roots 0.4875 / 0.4869 / 0.4874. Direct fits.
`p_omega_inf`: `0.4876` (r=1), `0.4870` (r=2) are the same numbers with shift 0;
`−0.0129` is `0.4871 − 0.5` **exactly** (refit of `ln(c*/sqrt d)` gives the same to 2.7e-16).
The report's column header "p (omega_inf, registered primary)" and the sentence "Rule B:
REFUTED at full rank (p = −0.0129 …)" present a relabelling as if it were a measurement.

## Part 9 — rank and frequency

`||e||_inf = eps sqrt r` (orthonormal rows let all `r` sines peak together: the map
`a ↦ V a` is onto `R^r`); `sup ||grad e||_2 = eps ω` (`||Σ cos_j v_j||^2 = Σ cos_j^2 ≤ r`,
times `(eps ω/sqrt r)^2`); `max_j sup |∂_j e| = eps ω/sqrt r` **for `V = I`** (full arm;
for random `V` in `rank_r2` it is a bound, but that convention is unused there). Mappings:
`ω_inf/ω = 1/sqrt r` (L2) or `1/r` (componentwise); fixed rank → constant → no exponent
change; `r = d` → `−1/2` or `−1`. No `ceil(sqrt d)` arm exists. **The full-rank result
shows no change in the nominal/RMS crossover** (`c*` per `d`: 1.197, 1.522, 2.051, 2.871,
4.065, 5.748, 8.131 vs rank1 1.201, 1.518, 2.050, 2.875, 4.060, 5.747, 8.131; `p` 0.4871
vs 0.4876) — only the frequency normalisation changes.

## Part 10 — cross term

Full-estimator variance `Var[g(Q^pi+e)] = Var[g(Q^pi)] + Var[g(e)] + 2Cov(g(Q^pi), g(e))`
holds by linearity; the **error-channel** quantity `Var[g(Q^pi+e) − g(Q^pi)] = Var[g(e)]`
exactly, with no cross term. Claim 4 compares error-channel variances → **no cross term
enters the primary comparison.** The cross term matters only for total MSE (E1b).

The registered diagnostic, exactly as written, from saved `s2/s3` (`lqr_audit_crossterm.csv`):

| reading | d=1 | 2 | 4 | 6 | 8 | 16 | 32 | 64 | status |
|---|---|---|---|---|---|---|---|---|---|
| max over the entire grid, `\|2Cov\|/Var_e`, PW | 7.1e5 | 3.8e5 | 3.8e5 | 2.8e5 | 3.0e5 | 4.3e5 | 3.6e5 | 3.3e5 | **preregistered** → all flagged |
| same, ZO | 5.7e4 | 3.5e4 | 2.0e4 | 1.7e4 | 1.4e4 | 9.3e3 | 9.2e3 | 5.9e3 | preregistered → all flagged |
| at the measured crossover, PW / ZO | 0.51/6.64 | 0.17/14.7 | 0.07/3.79 | 0.33/0.86 | 0.10/0.23 | 0.01/0.16 | 0.00/0.11 | 0.01/0.09 | **post-hoc** (code changed at `98da377`) |
| `\|2Cov\|/MSE_total`, `σ ≥ 0.1`, PW / ZO | 0.21/0.03 | 0.11/0.02 | 0.07/0.01 | 0.06/0.01 | 0.05/0.01 | 0.03/0.01 | 0.01/0.01 | 0.01/0.00 | post-hoc, never printed |

`c → 0` analytically (rank 1, units `(eps/σ)^2`): PW error `∝ c(mean cos − E cos) ≈ −c^2
sin θ · tbar` so `Var_e ∝ c^4` while `Cov ∝ c^2` → ratio `∝ c^{−2}`; ZO error
`∝ c cos θ · (1/M)Σ(t_i − tbar)u_i` so `Var_e ∝ c^2`, `Cov ∝ c` → ratio `∝ c^{−1}`.
Measured slopes on the `σ = 0.01` column at the six smallest `ω`: PW **−2.05**, ZO
**−0.92**. The report's "`Var_e ~ c^2`, cross `~ c`" is the ZO case. The registered rule
is therefore unbounded by construction — but it is the registered rule, and its verdict
("flagged at every `d`") is the one the prereg returns; the report's flag set is post hoc.
None of this touches the error-channel comparison.

## Part 11 — Monte Carlo and bootstrap

Per `d`: 64 states (`sample_states`, `rng(SEED_ROOT+2000+d)`), per state `n_batch = 40`
scan steps × `r_batch = 250` draws of `u ∈ R^{32×d}` → `N = 10^4` batches per state;
keys `fold_in(PRNGKey(SEED_ROOT+3000+d), i)` split into 40 — independent across states
and batches. Common random numbers across the whole `σ×ω` grid and across PW/ZO are
**intentional** (paired design, stated in the prereg). Ladder/eps arms: 32 × 20 × 100.
Bootstrap (`bootstrap_p`): states with replacement, then batches within each drawn state,
`c*` re-rooted and `p` re-fit inside every resample, seed 20260902 — as registered; 10 000
for the both-level CI `[0.4869, 0.4883]`, **3 000** for each single-level run; **no CI
exists for `r=2` or full rank**, so Rule B's CI clause is unverified there.

## Part 12 — provenance

All 34 sweep `.npz` and six `estep` files: `git_sha = prereg_sha = 33c7632`.
`estep_d64_rank1_M32.npz`: `98da377`, regenerated twice (index rows uncommitted until
`c0a1ff1`); §7.3's `d=64` row comes from it. `d6_*.npz`: post hoc (addendum A1, `d2cb9f6`),
never enters the report. `m_sweep_estep_hi/hiw.json` (§7.4) were committed at `8e41e64`
with their scripts **untracked** until `c0a1ff1`; `probe2_full_estimator.py` (G6b) likewise;
G6b's reference `.npz` is git-ignored. `.npz` are ignored (`*.npz`, `.gitignore:177`);
`lqr_npz_manifest.csv` holds sha256. Between data (`33c7632`) and results (`98da377`):
`sweep.py` gained only `prereg_sha` and CLI flags (kernel unchanged); `analyze.py`:
`solve_crossover`, `crossover_by_c`, `log_ratio_by_c`, `fit_p`, `err_only`, `full_mse`
**unchanged**; `cross_share` **rewritten**; `group_matrix` added. `reports/lqr_crossover.md`
regenerates byte-identically from the current tree. Registered but never run: the
`ceil(sqrt d)` rung, E2.

## Part 13 — the shipped E-step

Production (`reppo.py`, `weighted_mle`): `y_i = mu_old + σ_old u_i` retained pre-tanh,
`a_i = clip(tanh y_i, ±(1−1e−4))`, `q_i = Q_target(s, a_i)`, `w = softmax(q/η)` over the
`M` axis (max-subtracted, normalised), `η` learned by gradient on `η ε_e + η mean log
mean exp((q−qmax)/η) + qmax`, objective `−Σ_i w_i log π_θ(a_i)` (tanh-Normal, `σ` fitted
too), one optimizer step under the KL clip/Lagrangian, `ESS = 1/Σ w_i^2`. The exact
weighted-MLE mean for fixed `σ` and no clip is `Σ w_i y_i = mu_old + σ Σ w_i u_i`, so the
LQR `d_ESTEP = Σ w_i u_i / σ` is the **M-step optimum's mean direction**, not the realised
one-step update; `η` there is grid-minimised on the same dual (`estep_arm.py` cap 10,
`msweep.py` cap 1e4). Relation to centred ZO: `d_ES = ubar + σ·zo_plain/η + O(η^{−2})`
(verified, rel 7e-4 at `η = 10^3`); at that order `ubar` dominates and
`cos(d_ES, g_ZO) = −0.09`; alignment appears only as `η` shrinks (0.62 at `η=1`, 0.87 at
`η=0.1`, ESS/M 0.16). **No statement that the E-step "implements ZO" is exact; the
first-order equivalence is real but the term that survives at first order is `ubar`, not
`g_ZO`.** The report keeps these separate (§7.3 measures cosine to `g*`), which is correct.

## Part 15 — silent-bug sweep

Checked and clear: axis reductions (`mean(1)` over samples, state axis kept), broadcasting
(`_align` not einsum-with-`s`, gate G5 history), state-then-batch order (batch mean inside
state, then state mean — same at every use), sample reuse (intentional CRN), covariance
denominator (`s3` is a mean of squared errors about the exact mean; `np.cov` ddof=1 only
in my audit), `M` vs `M−1` (once, correct), `σ` vs `σ^2` (kernel `smooth()` and `/sig`
verified), `ω` vs `σω` (`c = sig*om` per cell; `T` scaled by `c`), nominal vs `ω_inf`
(nominal in every `c*`; `ω_inf` only in the relabelling), rank normalisation (`/sqrt r`
in `f` and in `de_pw`, verified against my field), `φ` vs `θ` (**kernel correct; report's
closed-form column wrong**), median vs mean (report's "c* analytic" is a median of roots —
wrong aggregate; `c*` itself is root-of-mean), interpolation on a monotone log-ratio
(single sign change), no extrapolated roots (interior only), `d=6` cannot enter fits, `d=1`
excluded from `FIT_DS`, no stale artifacts (report regenerates identically), dictionary-key
error (§7.2 `nan`), report values not regenerable: none.

---

## Answers

1. **PW correct?** Yes. Analytic-gradient mean over the batch, `/σ` once; matches an
   independent implementation to 1e-14 and FD to 4e-8.
2. **Centred ZO correct?** Yes. Same-batch `Qbar`, centred before `u`, mean over samples,
   `/σ` once; matches an independent implementation to 9e-16.
3. **`M/(M−1)` correct and once?** Yes. Applied once in the kernel and once in the core
   (never both on the same value); makes the mean exact and scales the variance by
   `(32/31)^2`. Verified analytically and to 2e-16 numerically.
4. **Error-channel measurement = the theory's quantity?** Yes. `tr Cov[g(Q^pi+e) −
   g(Q^pi)]` under shared randomness, computed as `tr Cov[g(e)]` about the exact mean;
   the two are equal to 1e-15 by linearity, and the stored statistic is a variance, not
   `E||g||^2`.
5. **Is `c*` the equality point of the two error-channel variances?** Yes — of the
   state-averaged log-ratio, interior-bracketed, single crossing; per-state roots agree
   to ≤ 7e-4. Interpolation contributes ~0.1–0.25 % at `d ≤ 4`.
6. **Is `p ≈ 0.4876` correctly computed?** Yes: unweighted natural-log OLS on
   `{2,…,64}`, independently refit to 5.6e-17, bootstrap CI `[0.4869, 0.4883]` from the
   registered procedure.
7. **Discovery or construction?** Largely imposed. For a rank-one field `Var_PW_e` is
   `d`-independent and `Var_ZO_e = (d−1)·const + const`, so `p → 1/2` algebraically; the
   prereg says so in advance (Sec. 3) and the closed form gives 0.4920 on the same
   `d`-set. It is a verification of the pipeline, not a test of Claim 4.
8. **Full-rank `−0.0129`: fit or relabelling?** A relabelling: `0.4871 − 0.5` exactly.
   The full-rank arm's nominal crossovers are indistinguishable from rank-one's.
9. **Does full rank refute the variance mechanism or the sup-norm `sqrt d` reading?**
   Only the reading. The nominal/RMS crossover and the `32/31` law are unchanged at full
   rank (medians 1.0315–1.0328); what moves is `ω_inf = ω/sqrt r` with `r = d`.
10. **RMS calculations correct?** Yes: closed forms at the realised `θ = ω v^T mu + φ`
    verified against MC to 5e-3 and against the audit CSV to 1e-16.
11. **`(Var_ZO/Var_PW) r_RMS^2` at the predicted constant?** Yes: `32/31` for the
    de-attenuated estimator, measured 1.0315–1.0363 at `d ≥ 8` in all three arms (≤ 0.4 %);
    `31/32` would be the plain-centred prediction and is off by 6.5 %.
12. **Cross-term issue: primary comparison or decomposition only?** Decomposition only.
    The error-channel comparison needs no cross term (exact by linearity). The
    registered cross-term rule was replaced post hoc; that changes the reported flag set
    for the *total-MSE* decomposition, nothing else.
13. **Coding errors forcing a rerun?** No. The one code bug (§7.2 `nan`) is in the
    report generator and its values are recoverable; the phase-convention error is in a
    comparison column, not in the sweep.
14. **Safe to use today:** §0 system table; §3 `c*` measured (all `d`), collapse sd,
    columns bracketed, `p = 0.4876` and the §6 both-level CI `[0.4869, 0.4883]`; §4 E1b
    `c*` and ratios, `p_E1b = 0.4359`; §7.1 nominal `p` 0.4876 / 0.4870 / 0.4871 and the
    per-`d` `c*` of every arm; §7.1b eps ratios; §7.3 cosines / ESS (with the `d=64`
    provenance caveat); §7.4 cosines (scripts now tracked); the `32/31` law and
    `ω_RMS = ω` from the audit artifacts; the corrected closed-form comparison in
    `lqr_audit_ruleA.csv` (≤ 1.8 SE on the registered `d`-set).
15. **Not safe until fixed:** §3 "c* analytic" and "rel. diff" columns and "closed form:
    0.4691" (wrong `θ`, wrong aggregate); §5's flag set presented as the registered
    outcome (it is post hoc — print both rules or neither); §7.1's "p (omega_inf)"
    column as a fit and "Rule B … CONFIRMED at fixed rank" (not a registered outcome,
    ladder incomplete, no CIs); §7.2's `nan` rows; the §6 single-level rows as "10 000
    resamples" (3 000).
