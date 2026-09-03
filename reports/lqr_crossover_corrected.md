# LQR estimator-crossover study: corrected canonical report

**Status.** Canonical analysis of the existing LQR crossover data. No sweep was rerun, no
`.npz` modified, no estimator or preregistration altered. Every number cites its
artifact; `.npz` files are identified by the 12-character sha256 prefixes in
`reports/artifacts/lqr_npz_manifest.csv`. The historical report
`reports/lqr_crossover.md` (`98da377`, `8e41e64`) is left in place; the entry-by-entry
diff is `reports/lqr_crossover_corrected_changes.md`; the paper-facing claim ledger is
`reports/artifacts/lqr_claim_ledger.csv`. Audit records: `reports/lqr_crossover_audit.md`,
`reports/lqr_code_correctness_audit.md` (`53e75d9`, `bbfbf59`).

**Regeneration.** `JAX_PLATFORMS=cpu python scripts/lqr_crossover/corrected_analysis.py
--bootstrap` and `... corrected_levelset.py` write every table below to
`reports/artifacts/lqr_corrected_*.{csv,json}` from the saved data.

---

## Verdicts

1. **ESTIMATOR IMPLEMENTATION: SUPPORTED.** `Q^pi` gradient to 4e-8 by finite
   differences; planted field to 3e-10; PW and centred ZO match independent
   implementations to 1e-14 and are linear in `Q` to 4e-14 on identical draws; the
   `M/(M-1)` factor is applied once and makes the mean exact; the stored error-channel
   statistic is `tr Cov` about the exact blurred mean; the sweep ran in float64
   (`lqr_code_correctness_checks.csv`, 132 checks).
2. **ERROR-CHANNEL MECHANISM: SUPPORTED WITH QUALIFICATION.** Across the planted
   sinusoidal error families considered here, the crossover of the two error-channel
   variances sits at `c* ≈ sqrt(d M/(M-1))`, collapses onto `sigma·omega`, and satisfies
   `(Var_e[ZO]/Var_e[PW]) r_RMS^2 = 32/31` to 0.4 % at `d ≥ 8` in all three rank arms.
   Qualifications: the rank-one exponent is algebraic (prereg Sec. 3); at the crossover
   the error channel carries ≤ 5 % (PW) and ≈ 0.1 % (ZO) of total estimator error in the
   policy-reachable region; nothing here concerns learned critics.
3. **UNIVERSAL SUP-NORM SQRT(d) CLAIM: REFUTED.** Under the registered sup-norm
   frequency `omega_inf = sup||grad e||_2 / ||e||_inf = omega/sqrt r`, the full-rank
   exponent is `0.4871 − 0.5 = −0.0129`. The nominal/RMS crossover itself keeps
   ≈ `sqrt(d)` scaling at every tested rank; what fails is the universal statement in
   the sup-norm coordinate, because the map from nominal/RMS to sup-norm frequency is
   rank dependent.
4. **RULE B: REFUTED** at full rank under the registered sup-norm frequency (prereg
   Sec. 5.3, third bullet). The CONFIRMED branch cannot be adjudicated as registered:
   the ladder has three of four rungs (`ceil(sqrt d)` not run). Descriptively, rank-1 and
   rank-2 nominal exponents are 0.4876 [0.4869, 0.4883] and 0.4870 [0.4854, 0.4887].
5. **ACTUAL MPO E-STEP CONNECTION: SUPPORTED WITH QUALIFICATION.** The first-order
   bridge `Delta mu = sigma·ubar + (sigma^2/eta)·g_ZO_plain + O(eta^-2)` is exact as an
   expansion (verified, rel. 7e-4 at `eta = 10^3`). It is a mathematical bridge only:
   at first order `ubar` dominates, and in trained checkpoints the first-order identity
   is adequate in 1 of 10 conditions (`reports/ubar_ratio.md` P1). No statement equating
   the E-step with centred ZO is made.
6. **UBAR DIMENSION-AMPLIFICATION HYPOTHESIS: REFUTED.** Preregistered
   (`docs/prereg_ubar_ratio.md` P4) and tested on WalkerRun `d = 6 → 22`: all-pairs
   `Rho` medians 1.035 (arm A) and 0.485 (arm B) against the registered 1.3
   (`reports/ubar_ratio.md` Sec. 9, `5de9b3e`). Unpaired in seed and `alpha`;
   estimator-level only.
7. **LARGE-M RETURN-RESCUE HYPOTHESIS: REFUTED** on the exploratory HumanoidRun
   evidence: weighted-MLE return 666.17 (M = 32, n = 8) → 137.16 (M = 128, n = 5) →
   10.98 (M = 512, n = 2) (`reports/m_sample_count_audit.md`; exploratory seeds 201+,
   GPU-confounded arms, M = 512 below the n = 5 floor). The LQR §7.4 prediction held at
   the estimator level and failed at the return level. **Walker confirmatory
   return-level test: PENDING** (`ledger/runs_m_sweep_confirmatory.jsonl`: 16 runs
   planned, none launched, no exports). Estimator quality ≠ policy quality.

---

## 0. Execution precision and the policy-reachable region

**Dtype of the existing sweeps, established from provenance, not from the current
environment.** Every cited sweep `.npz` embeds `git_sha = 33c7632`. At that commit,
`scripts/lqr_crossover/__init__.py` sets `JAX_PLATFORMS=cpu` and
`jax.config.update("jax_enable_x64", True)` and **asserts** at import that
`jnp.zeros(1).dtype == float64` and the platform is CPU (lines 29–41 of the file at
`33c7632`); `sweep.py` imports that package before `jax` (line 14 vs 16); the kernel's
inputs are `jnp.asarray` of float64 numpy arrays and its draws are `jax.random.normal`
under x64; no `float32` or `astype` occurs anywhere in the harness. Every stored
statistic (`s1_*, s2_*, s3_*, eps, sigmas, omegas`) in all 28 cited sweep files is
float64 — numpy preserves the JAX dtype, so float32 arithmetic would have left float32
files. The audit and correction scripts run under the same import and in numpy float64;
no analysis uses a different precision from the sweep.

**Could any discrepancy below be precision?** No. The most delicate cell,
`sigma = 0.01, omega = 0.1` (`c = 10^-3`), stores `s3_pw = 1.4e-14` and
`s3_zo = 9.1e-8` in units of `(eps/sigma)^2`; an independent recomputation reproduces
them to 4.4 % and 0.2 % (Monte-Carlo bound 8.7 %), and the saved column follows the
analytic power laws `s3_pw ∝ c^{4.09}`, `s3_zo ∝ c^{1.92}`, `|s2_pw| ∝ c^{2.05}`,
`|s2_zo| ∝ c^{0.96}` (TEST K). The only delicate operation is an O(1) − O(1)
subtraction losing ~4 digits.

**Policy-reachable region.** The real policy has an effective minimum standard deviation
of 0.1 (`min_std = 0.1`, `src/networks/jax_models.py:336`; `actor_min_std` is dead
config). The swept `sigma` grid is `0.01 · 300^{k/19}`, `k = 0..19`; its first **8 of 20
columns** (`sigma` = 0.0100, 0.0135, 0.0182, 0.0246, 0.0332, 0.0449, 0.0606, 0.0818) lie
below 0.1 — **272 of the 680 cells per file are outside the policy-reachable region.**
Their values are numerically real and are retained in every registered statistic; they
are marked wherever they enter one. Throughout, **REGISTERED GRID RESULT** denotes a
statistic over all 680 cells as registered, and **POLICY-REACHABLE INTERPRETATION** the
same statistic restricted to `sigma ≥ 0.1`.

---

## 1. Design as executed

Discounted LQR, `n = 2d`, `gamma = 0.99`, `W = I`, identity cost, `K = 1.15 K_DARE`;
normalised critic with `||alpha_Q H||_2 = 1`, `E_s||g*|| = 1`. Planted error
`e = (eps/sqrt r) sum_j sin(omega v_j^T a + phi_j)`, `V` orthonormal rows, `phi ~ U[0,2pi)`
per state, `eps = 0.05 · q_spread(s, sigma)` per (state, `sigma`). `M = 32`. Rank-one
primary: 64 states × 40 batches × 250 = `N = 10^4` per state; rank-2, full-rank and
`eps = 0.20` arms: 32 × 20 × 100 = 2 000. Grid 20 `sigma` × 34 `omega`, common log ratio
`300^{1/19}`, 53 distinct `c = sigma·omega`. Guards: `rho_closed` 0.45–0.69, `cond_H`
1.0–16.2, zero rejections at every `d` (`out/index.jsonl`).

**Estimators, as implemented.** PW: `(1/M) sum_i grad_a Q(mu + sigma u_i)`. Centred ZO:
`(M/(M-1)) (1/(sigma M)) sum_i (Q_i − Qbar) u_i`, `Qbar` the same batch's mean, centred
before multiplying by `u`, de-attenuated once. Both are linear in `Q`. **The centred ZO
estimator is unbiased, after `M/(M-1)`, for the Gaussian-blurred gradient
`E[grad e(mu + sigma u)]`** — by Stein's identity
`(1/sigma) E[e(mu + sigma u) u] = E[grad e(mu + sigma u)]` — and not in general for
`grad e(mu)`. The sweep subtracts exactly that blurred mean, so the stored `s3` is a
variance.

---

## 2. The error channel, and what is and is not decomposed

For identical randomness `xi`, linearity gives

```
g(Q^pi + e; xi) − g(Q^pi; xi) = g(e; xi)        (PW and centred ZO; verified to 4e-14)
```

so the critic-error-channel variance

```
V_e(g) = tr Cov[ g(Q^pi + e; xi) − g(Q^pi; xi) ] = tr Cov[ g(e; xi) ]
```

**drops no cross term.** The cross term belongs only to the total-variance decomposition

```
Var[g(Q^pi + e)] = Var[g(Q^pi)] + Var[g(e)] + 2 Cov(g(Q^pi), g(e)).
```

The primary result (Sec. 3) is a statement about `V_e`; the cross-term diagnostic
(Sec. 5) is a statement about the total-variance decomposition, and its failure — under
any reading — does not bear on the error-channel identity.

---

## 3. E1a: the error-channel crossover, and Rule A

`c*(d)` is the interior root, in `log c`, of the state-averaged log-ratio
`log Var_e[ZO] − log Var_e[PW]` collapsed onto the 53 `c` level sets (PCHIP + `brentq`).
Exactly one sign change exists at every `d`; 0 % of states lack a bracket; the collapse
sd across the 20 `sigma` columns is 0.0005–0.0078 (registered < 0.05).

| d | c* measured (root of state mean) | c*, median of per-state roots | comparator B: analytic, measurement-matched | comparator A: analytic, per-state roots (median) | old report's column (`theta = phi`) | bootstrap SE | \|dev\|/SE vs B | vs old |
|---|---|---|---|---|---|---|---|---|
| 1 † | 1.2014 | 1.2015 | 1.1903 | 1.1878 | 1.2098 | 0.0024 | **4.69** | 3.55 |
| 2 | 1.5182 | 1.5193 | 1.5132 | 1.5144 | 1.4897 | 0.0018 | 2.80 | **16.01** |
| 4 | 2.0495 | 2.0483 | 2.0480 | 2.0479 | 2.0414 | 0.0018 | 0.87 | **4.66** |
| 8 | 2.8746 | 2.8755 | 2.8738 | 2.8738 | 2.8741 | 0.0024 | 0.32 | 0.22 |
| 16 | 4.0603 | 4.0605 | 4.0640 | 4.0640 | 4.0640 | 0.0036 | 1.01 | 1.02 |
| 32 | 5.7465 | 5.7437 | 5.7474 | 5.7474 | 5.7474 | 0.0049 | 0.17 | 0.17 |
| 64 | 8.1306 | 8.1339 | 8.1280 | 8.1280 | 8.1280 | 0.0074 | 0.35 | 0.35 |

† `d = 1` is registered as secondary (prereg 5.4) and is excluded from the primary fit.
Source: `lqr_corrected_ruleA.csv`; SE from a 1 000-resample hierarchical bootstrap of
`c*`.

**Comparators.** *B (measurement-matched)*: the closed forms of `reference.py` evaluated
on every swept cell with the realised `theta_s(cell) = omega_j v^T mu_s + phi_s`,
averaged over states and over the cells of each `c` level set, then rooted with the same
PCHIP/`brentq` step — the analytic twin of the measured statistic. *A (preregistered
wording)*: the same per-state analytic curves rooted per state, then aggregated. *Old*:
`report.py` passed `phi[:, 0]` as `theta` (dropping `omega v^T mu`, median 13.5 rad at
`omega = 5`) and took a median of per-state roots on an unbounded `c` axis.

**Why A and B differ.** Root-of-mean and mean-of-roots differ by Jensen; the two agree to
≤ 0.3 % here (0.2 % at `d = 1`) because the per-state curves are nearly parallel in
`log c`. Both differ from the old column because that column used the wrong phase.

**Rule A, adjudicated.** Registered text: `p ∈ [0.35, 0.65]` **and** "the per-`d` `c*`
within 4 Monte Carlo standard errors of the Sec. 3 closed form" → VERIFIED. The prereg
does not say how the closed form is aggregated; the mathematically appropriate
comparator is B, because the measured `c*` is the root of the state-averaged ratio. Under
B every registered `d` is within 2.8 SE → **Rule A: VERIFIED.** Under the old column
`d = 2` (16 SE) and `d = 4` (4.7 SE) would have failed. `d = 1` fails under both (4.7 SE
vs B) — a structurally different regime, which is why the prereg excluded it. Note that
the PCHIP interpolation itself contributes 0.1–0.25 % at `d ≤ 4`, comparable to the SE;
B inherits the same step, which is the point of matching.

**Fitted exponent.** OLS of `ln c*` on `ln d`, unweighted, `d ∈ {2, 4, 8, 16, 32, 64}`:
**`p = 0.4876`**, 95 % CI **[0.4869, 0.4883]** (registered hierarchical bootstrap, 10 000
resamples, `out/bootstrap_p.json`). Closed form under B on the same `d`-set: 0.4920. With
`d = 1`: 0.4680 (secondary). **This is a verification of the rank-one algebra, not a test
of Claim 4**: `Var_e[PW]` is `d`-independent and `Var_e[ZO] = (d−1)·const + const`, so
`p → 1/2` by construction (prereg Sec. 3).

---

## 4. E1b: the total-error crossover, and the `eps` arm

| d | c* E1a | c* E1b (full MSE) | ratio | c* E1b at `eps = 0.20` | ratio 0.05/0.20 (4× predicts 4.00) |
|---|---|---|---|---|---|
| 1 | 1.2014 | 43.375 | 36.1 | 10.841 | 4.001 |
| 2 | 1.5182 | 52.130 | 34.3 | 13.051 | 3.994 |
| 4 | 2.0495 | 66.169 | 32.3 | 16.524 | 4.004 |
| 8 | 2.8746 | 87.948 | 30.6 | 22.006 | 3.997 |
| 16 | 4.0603 | 120.40 | 29.7 | 30.210 | 3.986 |
| 32 | 5.7465 | 166.73 | 29.0 | 41.914 | 3.978 |
| 64 | 8.1306 | 233.06 | 28.7 | 59.008 | 3.950 |

`p_E1b = 0.4359`. E1b crosses 29–36× later than E1a because the zeroth-order operator
pays the classical `(d+1)/(M−1)` factor on the smooth part; it scales as `1/eps`, so it is
a joint statement about error amplitude and frequency, not an intrinsic threshold. No
verdict is registered for E1b. E1a's `eps`-independence is structural (`eps` is factored
out analytically) and is an implementation check, not a finding.

---

## 5. The cross-term diagnostic

### 5.1 What was registered, and what the code did

Prereg Sec. 5.4: "If `|2 Cov| > 0.25 · Var_e` **anywhere in the swept region**, Claim 4's
variance decomposition is flagged as incomplete in the paper. It is reported either way."
`analyze.cross_share` at the prereg commit `33c7632` computed exactly that cell-wise
ratio; at the results commit `98da377`, after the data existed, it was replaced by an
at-crossover ratio and a total-MSE fraction, and `report.py` printed the former.
`|2 Cov|` is read per state (Claim 4 is a per-state statement): `mean_s |2Cov_s| /
mean_s Var_{e,s}`, which is also what `report.py` computed; the state-pooled
`|mean_s 2Cov_s|` is given for completeness and is 3–20× smaller because phases cancel
across states.

### 5.2 Registered outcome (REGISTERED GRID RESULT)

| d | A. max over the entire grid, PW | ZO | cell of the maximum | in reachable region? |
|---|---|---|---|---|
| 1 | 7.1e5 | 5.7e4 | `sigma = 0.01, omega = 0.1` | no |
| 2 | 3.8e5 | 3.5e4 | same | no |
| 4 | 3.8e5 | 2.0e4 | same | no |
| 8 | 3.0e5 | 1.4e4 | same | no |
| 16 | 4.3e5 | 9.3e3 | same | no |
| 32 | 3.6e5 | 9.2e3 | same | no |
| 64 | 3.3e5 | 5.9e3 | same | no |

**Registered verdict: flagged at every `d`, for both operators.** Every maximum lies at
the smallest-`c` corner, outside the policy-reachable region.

**Why it diverges (analytic).** In units of `(eps/sigma)^2`, rank one, as `c → 0`:
PW's error `(c/sqrt r)(mean_i cos(theta + c t_i) − E cos) ≈ −c^2 sin theta · tbar`, so
`Var_e ∝ c^4` while `Cov(g(Q^pi), g(e)) ∝ c^2` → ratio `∝ c^{−2}`; ZO's error
`∝ c cos theta · (1/M) sum_i (t_i − tbar) u_i`, so `Var_e ∝ c^2`, `Cov ∝ c` → ratio
`∝ c^{−1}`. Measured slopes on the `sigma = 0.01` column: PW −2.05, ZO −0.92. The
registered rule is therefore unbounded by construction in the corner where both terms are
negligible against the smooth part. **Its failure does not invalidate the primary
error-channel identity**, which is isolated by subtraction and operator linearity
(Sec. 2).

### 5.3 Post-hoc readings (POLICY-REACHABLE INTERPRETATION)

The report's at-crossover value is a single cell chosen by `np.argmin` — the first
level-set cell in row-major order, i.e. the **smallest `sigma`** on `c = c*`: `sigma` =
0.0135, 0.01, 0.01, 1.219, 0.0818, 0.0246, 0.149 for `d` = 1…64. Five of seven are
unreachable. Evaluated along the whole level set (`lqr_corrected_levelset.csv`, 20 cells
per `d`, 12 reachable):

| d | B. report's cell, PW / ZO | B'. reachable level set, median PW / ZO | max PW / ZO | C. reachable-grid max PW / ZO | D. `\|2Cov\|/MSE_total`, reachable level-set median PW / ZO | reachable-grid max PW / ZO |
|---|---|---|---|---|---|---|
| 1 | 0.509 / 6.639 | 19.26 / 41.41 | 33.7 / 69.3 | 6.5e4 / 5.1e3 | 0.051 / 0.019 | 0.210 / 0.026 |
| 2 | 0.165 / 14.70 | 9.53 / 20.00 | 18.5 / 29.7 | 3.4e4 / 3.2e3 | 0.032 / 0.012 | 0.109 / 0.022 |
| 4 | 0.069 / 3.791 | 3.75 / 7.60 | 6.08 / 11.2 | 3.4e4 / 1.8e3 | 0.019 / 0.006 | 0.071 / 0.014 |
| 8 | 0.095 / 0.233 | 0.069 / 0.222 | 0.114 / **0.255** | 2.7e4 / 1.2e3 | 0.001 / 0.0002 | 0.050 / 0.010 |
| 16 | 0.008 / 0.161 | 0.033 / 0.154 | 0.043 / 0.179 | 3.7e4 / 810 | 0.0006 / 0.0002 | 0.027 / 0.008 |
| 32 | 0.002 / 0.115 | 0.017 / 0.124 | 0.020 / 0.143 | 3.1e4 / 769 | 0.0005 / 0.0001 | 0.013 / 0.007 |
| 64 | 0.005 / 0.095 | 0.010 / 0.094 | 0.011 / 0.104 | 2.7e4 / 483 | 0.0004 / 0.0001 | 0.008 / 0.004 |

Under the at-crossover reading restricted to reachable cells, both operators exceed 0.25
at `d ∈ {1, 2, 4}` (PW by 3.7–19×, ZO by 7.6–41×), `d = 8` is marginal for ZO (max
0.255), and `d ≥ 16` is clear. Reading C (registered ratio, reachable grid maximum) still
flags every `d` — the divergence begins well above `sigma = 0.1`. Reading D never exceeds
0.21 (PW, `d = 1`) or 0.026 (ZO).

### 5.4 How much of total error the error channel is, at the crossover

| d | `Var_e/MSE_total` PW, reachable level-set median (max) | ZO, median (max) | `Var_e/MSE_total` at the report's cell, PW / ZO |
|---|---|---|---|
| 1 | 0.003 (0.048) | 0.0005 (0.0008) | 0.736 / 0.0007 (σ = 0.0135) |
| 2 | 0.003 (0.059) | 0.0006 (0.0008) | 0.836 / 0.0008 (σ = 0.01) |
| 4 | 0.005 (0.077) | 0.0008 (0.0010) | 0.918 / 0.0010 (σ = 0.01) |
| 8 | 0.015 (0.182) | 0.0010 (0.0011) | 0.009 / 0.0009 (σ = 1.22) |
| 16 | 0.018 (0.135) | 0.0011 (0.0012) | 0.215 / 0.0012 (σ = 0.082) |
| 32 | 0.028 (0.141) | 0.0011 (0.0012) | 0.741 / 0.0012 (σ = 0.025) |
| 64 | 0.046 (0.183) | 0.0012 (0.0012) | 0.124 / 0.0012 (σ = 0.149) |

At the point where Claim 4's two error sensitivities are equal, and at
`eps_frac = 0.05`, **the critic-error channel is ≤ 5 % (median; ≤ 18 % at any reachable
cell) of pathwise's total estimator error and ≈ 0.1 % of zeroth-order's.** The smooth
`Q^pi` part dominates both. The large PW values at the report's unreachable cells
(`sigma ≈ 0.01`) arise because the smooth pathwise error `∝ sigma^2` vanishes there.

---

## 6. Bootstrap

Registered procedure (`analyze.bootstrap_p`): 64 (or 32) states with replacement, then
40 (or 20) batches within each drawn state; `c*` re-rooted and `p` re-fit in every
resample; `np.random.default_rng(20260902)`; percentile 95 % CI; `d ∈ {2, 4, 8, 16, 32,
64}`.

| arm | resampling | resamples | estimate | SD | 95 % CI | provenance |
|---|---|---|---|---|---|---|
| rank 1 | states + batches | 10 000 | 0.4876 | 0.00034 | [0.4869, 0.4883] | `out/bootstrap_p.json` (`98da377`) |
| rank 1 | states only, batches frozen | **10 000** | 0.4876 | 0.00026 | [0.4871, 0.4881] | rerun at the registered count during post-hoc audit (original 3 000: same to 4 d.p.) |
| rank 1 | batches only, states frozen | **10 000** | 0.4876 | 0.00022 | [0.4872, 0.4880] | same |
| rank 2 | states + batches | 10 000 | 0.4871 | 0.00082 | [0.4854, 0.4887] | computed during post-hoc audit from preregistered saved experimental data |
| full rank | states + batches | 10 000 | 0.4871 | 0.00053 | [0.4861, 0.4882] | same |

Variance by level (rank 1, 10 000): total 1.16e-7; state level 6.9e-8 (60 %); batch
level 5.0e-8 (43 %) — both levels contribute, so the interval is not an artefact of `N`.
Source: `lqr_corrected_bootstrap.json`. (The rank-2 bootstrap mean 0.4871 differs from
the point fit 0.4870 in the fourth decimal; the CI is the statistic.)

---

## 7. The rank ladder, frequency conventions, and Rule B

### 7.1 Directly fitted quantity: `p_nominal`

| arm | rank | per-`d` c* (d = 1 … 64) | `p_nominal` (direct OLS fit) | 95 % CI |
|---|---|---|---|---|
| rank-one | 1 | 1.201, 1.518, 2.050, 2.875, 4.060, 5.747, 8.131 | **0.4876** | [0.4869, 0.4883] |
| r = 2 | 2 (`r = 1` at `d = 1`) | 1.201, 1.522, 2.049, 2.877, 4.069, 5.744, 8.124 | **0.4870** | [0.4854, 0.4887] |
| full rank | d | 1.197, 1.522, 2.051, 2.871, 4.065, 5.748, 8.131 | **0.4871** | [0.4861, 0.4882] |

The full-rank crossovers equal rank-one's to < 0.5 % at every `d`: **the nominal/RMS
crossover does not move with rank in the three arms tested.**

### 7.2 Transformed coordinates (algebraic, not fitted)

For a rank-`r` field with orthonormal rows: `||e||_inf = eps sqrt r`,
`sup_a ||grad e||_2 = eps omega`, `max_j sup_a |d_j e| = eps omega/sqrt r` (coordinate
basis, exact for the full arm's `V = I`). Hence under the registered primary convention
`omega_inf = omega/sqrt r`, and under the componentwise one `omega/r`. At fixed rank
these are constants and leave the exponent unchanged; at `r = d` they shift it by exactly
`−1/2` and `−1`:

| arm | `omega_inf/omega` (registered, L2) | `p_omega_inf = p_nominal − 1/2·[r = d]` | componentwise `p` |
|---|---|---|---|
| rank-one | 1 | 0.4876 | 0.4876 |
| r = 2 | `1/sqrt 2` (constant) | 0.4870 | 0.4870 |
| full rank | `1/sqrt d` | **−0.0129 = 0.4871 − 0.5** | −0.5129 |

**`−0.0129` is an algebraic relabelling of the measured nominal exponent 0.4871**
(`report.py` applies a hardcoded `−0.5`; refitting `ln(c*/sqrt d)` reproduces it to
2.7e-16). It is not a second experimental exponent, and the historical report's Sec. 7.2
`nan` rows (a dictionary-key bug) are these two values.

**Safe interpretation.** The nominal/RMS crossover retains approximately `sqrt(d)`
scaling at full rank. The universal `sqrt(d)` statement expressed in the registered
sup-norm frequency does not survive, because the mapping from nominal/RMS frequency to
sup-norm frequency is rank dependent.

### 7.3 Rule B, verbatim and adjudicated

> Ranks `r ∈ {1, 2, ceil(sqrt(d)), d}`, under the registered primary norm convention, at
> `M = 32`. … `p ∈ [0.35, 0.65]` at every rank, with 95 % bootstrap CIs excluding 0 and
> excluding 1: **CONFIRMED**. … `p ∈ [0.8, 1.2]`: linear … `p` outside both bands, or
> `c*(d)` not monotone in `d`, or `p` differing across ranks by more than the CIs allow:
> **Claim 4's dimensional prediction is wrong as stated and must be withdrawn from the
> abstract**, and the paper reports the rank dependence instead.

At full rank under the registered convention `p_omega_inf = −0.0129`, outside both bands
→ the third bullet. **RULE B: REFUTED at full rank under the registered sup-norm
frequency.** The CONFIRMED branch is not adjudicable as registered: the ladder has three
of the four registered rungs (the `ceil(sqrt d)` rung was never run), so "confirmed at
fixed rank" is not a registered verdict and is not reported as one. Descriptively, the
rank-1 and rank-2 nominal exponents are ≈ 0.49 with CIs excluding 0 and 1 (Sec. 6).

**The missing rung.** For `r ≈ sqrt d`, the nominal/RMS exponent is expected to remain
near 1/2 (the three existing arms are indistinguishable), and the registered sup-norm
coordinate would shift it by `−(1/2)·(1/2) = −1/4` algebraically. Running it completes
Rule B as registered; its transformed exponent would not independently establish a new
law.

---

## 8. RMS frequency and the finite-`M` variance law

With `theta_j = omega v_j^T mu + phi_j` and `a ~ N(mu, sigma^2 I)`,
`E[e^2]/eps^2 = (1/r)[sum_j (½ − ½ e^{−2c^2} cos 2theta_j) + e^{−c^2} sum_{i≠j} sin theta_i sin theta_j]`,
`E||grad e||^2/eps^2 = (omega^2/r) sum_j (½ + ½ e^{−2c^2} cos 2theta_j)`,
`omega_RMS^2 = E||grad e||^2/E[e^2]`, `r_RMS = sigma omega_RMS/sqrt d`. Verified against
Monte Carlo to 5e-3. At every `d`'s crossover cell `omega_RMS/omega` = 1.0000 (1.0034 at
`d = 2`), so **the nominal exponents of Sec. 7.1 are already RMS-convention exponents.**

For the de-attenuated estimator, `Var_e[PW] ≈ E||grad e||^2/M` and
`Var_e[ZO] ≈ d E[e^2]/((M−1) sigma^2)`, so `(Var_e[ZO]/Var_e[PW]) r_RMS^2 → M/(M−1) =
32/31 = 1.0323` and the crossover sits at `r_RMS = sqrt(32/31) = 1.016`
(`reference.py`'s `c* → sqrt(d M/(M−1))`). Measured medians over cells with
`r_nom ∈ [0.5, 3]`:

| arm | d = 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| rank-one | 1.438 | 1.142 | 1.037 | 1.0331 | 1.0320 | 1.0332 | 1.0323 |
| r = 2 | 1.446 | 1.108 | 1.037 | 1.0363 | 1.0337 | 1.0332 | 1.0325 |
| full rank | 1.440 | 1.123 | 1.041 | 1.0315 | 1.0321 | 1.0324 | 1.0328 |

**Across the planted sinusoidal error families considered here**, the product converges
to `32/31` (≤ 0.4 % at `d ≥ 8`, every arm); the small-`d` excess is the along-`v` term
of the closed form, whose share is `1/d`. This is a controlled-family empirical identity.
It is not a theorem for arbitrary learned critic errors, and it is distinct from the
sup-norm bound Claim 4 states: the bound is worst-case; the RMS relation is what the
realised variances follow. (The `sqrt(1 − 1/M)` constant quoted elsewhere applies to the
plain, un-de-attenuated centred estimator; the two differ by exactly `(M/(M−1))^2`.)

---

## 9. The shipped E-step

**Bridge.** With `w_i = softmax(Q_i/eta)` and whitened displacement `d_ES = sum_i w_i u_i`,
expanding `w_i = (1/M)(1 + (Q_i − Qbar)/eta) + O(eta^{−2})` gives, in action space,

```
Delta mu = sigma·d_ES = sigma·ubar + (sigma^2/eta)·g_ZO_plain + O(eta^{−2}),
g_ZO_plain = (1/(sigma M)) sum_i (Q_i − Qbar) u_i
```

(verified on the implementation: relative residual 7e-4 at `eta = 10^3`). At first order
the term that survives is `ubar`, not `g_ZO`: `cos(d_ES, g_ZO) = −0.09` at large `eta`,
rising to 0.62 at `eta = 1` and 0.87 at `eta = 0.1` (ESS/M 0.16). In the LQR arm the
E-step's cosine to the exact estimand tracks ZO's within 0.03–0.11 (Sec. 7.3 of the
historical report). **No statement that the E-step "implements" centred ZO is exact; the
correspondence is first-order in `1/eta` and `ubar`-dominated at that order.**

**The `ubar` mechanism has been tested and is not open.** `docs/prereg_ubar_ratio.md`
registered the first-order identity (P1) and the dimension-amplification hypothesis (P4)
on trained checkpoints. Outcomes (`reports/ubar_ratio.md`, `5de9b3e`): P1 holds in 1 of
10 task-arm conditions (hopper arm A; e.g. walker arm B `residual_linear = 5.07`), so the
expansion does not describe the shipped update quantitatively; P4, the only within-task
manipulation of estimator-visible `d` (WalkerRun 6 → 22, padded), gives all-pairs `Rho`
medians **1.035 (arm A)** and **0.485 (arm B)** against the registered threshold 1.3:
**DIMENSION AMPLIFICATION: REFUTED** (unpaired in seed and `alpha`; estimator-level;
says nothing about returns). The paper connection is therefore: the first-order E-step
contains a finite-sample `ubar` term; its proposed dimension amplification was
preregistered and refuted in the trained system; the expansion is a mathematical bridge,
not an accepted explanation of the real RL gap.

---

## 10. The `M` sweep: estimator alignment versus policy return

**LQR (estimator level, exploratory, `out/m_sweep_estep*.json`).** At
`sigma = 0.367, c = 8.15`, the cosine of the E-step and centred-ZO estimates to the exact
LQR estimand rises with `M` at every `d`, and the ZO-minus-E-step gap shrinks (d = 21:
0.1151 → 0.0162 from `M = 32` to 2 048). **This measures estimator/estimand alignment,
not policy performance.**

**Real system (return level, exploratory namespace).** HumanoidRun weighted-MLE, per-seed
means over evals 15–19 (`reports/m_sample_count_audit.md`): pathwise control 738.61;
`M = 32` **666.17** (n = 8); `M = 128` **137.16** (n = 5); `M = 512` **10.98** (n = 2).
Final-eval aggregates in `exports_manifest.csv`: 672.2 (n = 22, includes `dec` variants),
98.8 (n = 7), 10.5 (n = 4). Seeds 201+ are exploratory ("never confirmatory",
`ledger/README.md`); the `M = 128` and `M = 512` arms each ran on a single, different GPU;
`M = 512` is below the registered `n = 5` floor.

**The LQR prediction that increasing `M` improves finite-sample estimator alignment was
supported at the estimator level, but its implied return-level rescue was not supported
on HumanoidRun; policy performance deteriorated sharply as `M` increased.** The
historical report's "decisive test … rerun the d = 21 arm at M = 128 and M = 512" was
run; it did not rescue return. **Walker confirmatory return-level test: PENDING**
(`docs/prereg_m_sweep_confirmatory.md`, `4ab5169`; `ledger/runs_m_sweep_confirmatory.jsonl`
lists 16 runs at `M ∈ {128, 512}`, seeds 301–308, all `planned`, no SLURM job, no
exports). Its outcome is not inferred from the LQR. **Estimator quality ≠ policy
quality.**

---

## 11. Provenance and reporting record

| item | status |
|---|---|
| sweep `.npz` (34 registered arms) | `git_sha = prereg_sha = 33c7632`; git-ignored; sha256 in `lqr_npz_manifest.csv` |
| `estep_d64_rank1_M32.npz` | regenerated twice at `98da377`; §7.3 `d = 64` row comes from it; other six `estep` files `33c7632` |
| `d6_rank1_M32_unit_H_identity.npz` | **POST HOC** (addendum A1, `d2cb9f6`; run `ed42e10`); excluded from every registered fit; `c*(6) = 2.4924`; at-crossover cross term PW 0.330 / ZO 0.857 |
| `msweep_hi.py`, `msweep_hiw.py`, `probe2_full_estimator.py` | untracked when their results were produced; committed `c0a1ff1` |
| `analyze.cross_share` | rewritten at `98da377` after the data existed (Sec. 5.1) |
| `report.py` Sec. 7.2 | prints `nan` (key `"full"` vs `"full rank"`); values −0.0129 / −0.5129; not patched here so the historical report still regenerates byte-identically |
| norm conventions | two of four recorded (`2/inf`, `inf/inf`); `rms` conventions stored as `nan`; RMS supplied by Sec. 8 |
| bootstrap counts | single-level runs were 3 000; rerun at 10 000 (Sec. 6) |
| `ceil(sqrt d)` rank rung | **NOT RUN** |
| E2 (prereg 5.5) | **NOT RUN** |
| "CONFIRMED at fixed rank" | not a registered verdict; not reported as one |
| direct fits vs transformations | Sec. 7.1 vs 7.2 |
| `sigma < 0.1` cells | 272 of 680 per file; outside the policy-reachable region; marked wherever they enter a statistic |
| sweep dtype | float64, x64 asserted at `33c7632` (Sec. 0) |

---

## 12. Answers

**A. Does any finding require rerunning the existing LQR sweep?** No. Every correction
is an analysis, aggregation, labelling or provenance change computed from the saved
batch-resolved statistics.

**B. Which conclusions change only in wording?** Rule A (VERIFIED; the supporting
comparator changes); the Sec. 7.2 `nan` rows; "unbiased" → "unbiased for the blurred
gradient"; `−0.0129` labelled as `0.4871 − 0.5`; `d = 6` labelled post hoc;
`estep_d64` provenance; single-level bootstrap counts (values unchanged to 4 d.p.).

**C. Which conclusions change scientifically?** (i) The registered cross-term verdict:
flagged at every `d`, not `{1, 2, 4}`; the `{1, 2, 4}` reading is post hoc, and its
single-cell values sat in the unreachable region. (ii) Rule B: REFUTED at full rank,
with no CONFIRMED verdict at fixed rank. (iii) The primary error channel drops no cross
term — the historical report's "Claim 4's decomposition is incomplete" applies to the
total-variance decomposition only. (iv) New: at the crossover the error channel is ≤ 5 %
(PW) / ≈ 0.1 % (ZO) of total estimator error. (v) The LQR §7.4 return-level prediction
failed on HumanoidRun; the `ubar` amplification hypothesis was refuted.

**D. Which analyses can be completed from saved artifacts without new experiments?**
Everything in Secs. 3–8 and 11: both analytic comparators, the 4-SE clause, all four
cross-term readings and `Var_e/MSE_total` along the level set, the rank-2 and full-rank
CIs, the 10 000-resample single-level runs, the RMS product, the Sec. 7.2 values.

**E. What preregistered experiment remains genuinely unfinished?** The `ceil(sqrt d)`
rank rung of Rule B (and, separately registered as a follow-on, E2). Completing the rung
is expected to leave the nominal exponent near 1/2 and shift the sup-norm coordinate by
≈ −1/4; it closes Rule B, it does not discover a law.

**F. What is the status of the Walker confirmatory M test?** PENDING — 16 runs
registered (`4ab5169`), none launched, no exports, no report.

**G. Safest one-paragraph interpretation for the paper.** In a closed-form LQR with a
planted sinusoidal critic error, the two estimators' critic-error-channel variances —
isolated exactly by subtraction under shared randomness — cross at
`sigma·omega ≈ sqrt(d M/(M−1))`, collapsing onto `sigma·omega` and satisfying
`(Var_e[ZO]/Var_e[PW]) r_RMS^2 = M/(M−1)` to 0.4 % for `d ≥ 8` at every tested rank; for
a rank-one field this exponent is fixed at 1/2 by the algebra, so the sweep verifies the
estimator identities rather than testing them. The nominal crossover does not move with
the error field's rank, but the universal `sqrt(d)` statement in the registered sup-norm
frequency fails at full rank because that frequency scales as `omega/sqrt r`; Rule B is
refuted at full rank and its fixed-rank branch is unadjudicated. At the crossover the
error channel is at most a few percent of total estimator error, and the total-error
crossover lies 29–36× higher and scales as `1/eps`. The shipped MPO E-step is not the
centred estimator: it equals `sigma·ubar + (sigma^2/eta)·g_ZO` to first order, a bridge
whose `ubar` term was tested and whose dimension-amplification explanation was refuted in
the trained system; and the LQR-level finding that larger `M` improves estimator
alignment did not translate into return on HumanoidRun, where return collapsed with `M`,
with the Walker confirmatory test still pending. Nothing here is evidence about learned
critics or about DMC returns beyond those two stated results.
