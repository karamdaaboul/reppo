# Changes from `reports/lqr_crossover.md` to `reports/lqr_crossover_corrected.md`

Each entry: OLD CLAIM (verbatim or paraphrased from the historical report), CORRECTED
CLAIM, WHY, SCIENTIFIC CONSEQUENCE. The historical report is left in place. Artifacts
cited are under `reports/artifacts/` unless stated.

---

**1. The analytic comparator for Rule A**

- OLD: Sec. 3 table column "c* analytic" (1.2098, 1.4897, 2.0414, …) and "rel. diff";
  "closed form: 0.4691".
- CORRECTED: comparator B, the closed form evaluated at the realised
  `theta = omega v^T mu + phi` on every swept cell and aggregated exactly as the
  measurement (state mean of the c-collapsed log-ratio, then the PCHIP root):
  1.1903, 1.5132, 2.0480, 2.8738, 4.0640, 5.7474, 8.1280; and comparator A, the
  preregistered per-state analytic roots (median 1.1878, 1.5144, 2.0479, …).
- WHY: the old column passed `phi[:, 0]` as `theta` (dropping `omega v^T mu`, median
  13.5 rad) and took a median of per-state roots on an unbounded `c` axis, neither of
  which mirrors the measured statistic.
- CONSEQUENCE: Rule A's 4-SE clause, never computed before, passes on the registered
  `d`-set under B (≤ 2.8 SE) and would have failed at `d = 2` (16 SE) and `d = 4`
  (4.7 SE) under the old column. The verdict VERIFIED is unchanged; its support changes.

**2. What the cross term is a term of**

- OLD: "Claim 4's decomposition drops `2 Cov(g[Q^pi], dg)`"; "the dropped cross term
  is not negligible at d = [1, 2, 4] … Claim 4's variance decomposition is incomplete".
- CORRECTED: the error-channel variance `V_e = tr Cov[g(Q^pi+e;xi) − g(Q^pi;xi)]
  = tr Cov[g(e;xi)]` drops nothing; the cross term belongs only to the total-variance
  decomposition `Var[g(Q^pi+e)] = Var[g(Q^pi)] + Var[g(e)] + 2Cov`.
- WHY: both estimators are linear in `Q` (verified to 4e-14 on identical draws), so the
  primary quantity is isolated by subtraction exactly.
- CONSEQUENCE: no cross-term result bears on the primary crossover or on `p`; it bears on
  how much of total estimator error the error channel represents (entry 4).

**3. The registered cross-term verdict**

- OLD: evaluated "AT the crossover, not maximised over the grid"; flag set `{1, 2, 4}`.
- CORRECTED: REGISTERED GRID RESULT — the literal rule (maximum anywhere on the swept
  grid of `|2Cov|/Var_e`) flags **every `d`** (PW 3.0e5–7.1e5, ZO 5.9e3–5.7e4, every
  maximum at `sigma = 0.01`, policy-unreachable). POLICY-REACHABLE INTERPRETATION — the
  post-hoc at-crossover reading, evaluated along the reachable part of the crossover
  level set, flags `{1, 2, 4}` for both operators with `d = 8` marginal (ZO max 0.255).
- WHY: `analyze.cross_share` was rewritten at the results commit `98da377` from the
  registered rule to the at-crossover one; the registered outcome was never printed.
  Additionally the report's at-crossover values are single cells chosen by `np.argmin`,
  i.e. the smallest-`sigma` cell of the level set (`sigma` = 0.0135, 0.01, 0.01 for
  `d` = 1, 2, 4 — unreachable).
- CONSEQUENCE: a preregistered verdict changes (all `d` flagged as registered); the
  post-hoc reading is reported as post hoc, with both denominators and reachability.

**4. What the error channel is worth at the crossover (new)**

- OLD: not reported.
- CORRECTED: along the reachable crossover level set, `Var_e / MSE_total` has median
  0.3–4.6 % (PW) and ≈ 0.1 % (ZO) at every `d`; the smooth `Q^pi` part dominates total
  estimator error where Claim 4's two sensitivities are equal.
- WHY: computed from the saved `s1, s2, s3` at every level-set cell.
- CONSEQUENCE: the crossover is a statement about a channel that carries ≤ 5 % of
  total error at `eps_frac = 0.05`; E1b (total error) crosses 29–36x later.

**5. The full-rank exponent**

- OLD: Sec. 7.1 column "p (omega_inf, registered primary)" listing 0.4876, 0.4870,
  **−0.0129**; "Rule B: REFUTED at full rank (p = −0.0129 …)".
- CORRECTED: report `p_nominal` first (0.4876 / 0.4870 / 0.4871, direct fits); then
  the transformed coordinate `p_omega_inf = p_nominal − 1/2 = −0.0129` at `r = d`,
  labelled an algebraic relabelling.
- WHY: `report.py` adds a hardcoded `−0.5`; the refit of `ln(c*/sqrt d)` reproduces it
  to 2.7e-16; the full-rank per-`d` crossovers equal rank-one's to < 0.5 %.
- CONSEQUENCE: the nominal/RMS crossover retains ≈ `sqrt(d)` scaling at full rank; what
  fails is the universal statement in the registered sup-norm frequency, because
  `omega_inf/omega = 1/sqrt r` is rank dependent.

**6. Rule B's verdict**

- OLD: "Rule B is CONFIRMED at fixed rank (0.4876 at r=1, 0.4870 at r=2)".
- CORRECTED: "RULE B: REFUTED at full rank under the registered sup-norm frequency."
  Descriptively, rank-1 and rank-2 nominal/RMS exponents are ≈ 0.49 with 10 000-resample
  CIs [0.4869, 0.4883] and [0.4854, 0.4887].
- WHY: the registered CONFIRMED branch requires four rungs `{1, 2, ceil(sqrt d), d}` and
  per-rank CIs excluding 0 and 1; the `ceil(sqrt d)` rung was never run.
- CONSEQUENCE: no fixed-rank CONFIRMED verdict exists as registered.

**7. Section 7.2 table**

- OLD: both rows print `nan`.
- CORRECTED: −0.0129 (L2 reading, `1/sqrt d`) and −0.5129 (componentwise, `1/d`).
- WHY: `ladders.get("full")` against a key spelled `"full rank"` in `report.py`.
- CONSEQUENCE: cosmetic; values recoverable, `report.py` left untouched here so the
  historical report still regenerates byte-identically.

**8. Norm conventions**

- OLD: prereg 5.1 says all four conventions are recorded.
- CORRECTED: two recorded (`2/inf`, `inf/inf`); the two `rms` conventions are stored as
  `nan`. The RMS frequency is instead supplied by the audit's closed form
  (`omega_RMS/omega = 1.000` at every crossover cell).
- WHY: `sweep.py` writes `nan` for `val_norm = "rms"`.
- CONSEQUENCE: none for the registered primary; stated for completeness.

**9. Bootstrap counts and coverage**

- OLD: Sec. 6 rows "states only" and "batches only" at 3 000 resamples; no CI for the
  `r = 2` or full-rank arms.
- CORRECTED: all four rerun/computed at the registered 10 000 from the saved data
  (values in the corrected report Sec. 6), labelled "computed during post-hoc audit from
  preregistered saved experimental data".
- WHY: registered count is 10 000; the arms' CIs are recoverable from saved batches.
- CONSEQUENCE: Rule B's CI clause is now checkable for the rungs that exist; verdict
  unchanged (rungs still missing).

**10. "Unbiased"**

- OLD: "the centred estimator (unbiased after `M/(M-1)`)".
- CORRECTED: "unbiased, after `M/(M-1)`, for the Gaussian-**blurred** gradient
  `E[grad e(mu + sigma u)]` (Stein: `(1/sigma) E[e(mu+sigma u) u] = E[grad e]`); not
  for `grad e(mu)`."
- WHY: the estimand was unstated.
- CONSEQUENCE: none numerically; the sweep already subtracts the blurred mean.

**11. The E-step and centred ZO**

- OLD: "If `d_ESTEP` tracks `g_ZO`, the theory describes the code" (left open); Sec. 7.4
  "the deficit is a sampling artefact, not a property of the operator".
- CORRECTED: first-order bridge `Delta mu = sigma ubar + (sigma^2/eta) g_ZO_plain +
  O(eta^-2)` (verified, rel 7e-4 at `eta = 1e3`); at that order `ubar` dominates; in
  trained checkpoints the first-order identity holds in 1 of 10 conditions
  (`reports/ubar_ratio.md` P1); the `ubar` dimension-amplification hypothesis was
  preregistered and **REFUTED** in the matched WalkerRun `d = 6 → 22` probe
  (Rho 1.035 / 0.485 vs 1.3).
- WHY: the project already tested this; the LQR report predates that verdict.
- CONSEQUENCE: the expansion is a mathematical bridge, not an accepted explanation of the
  real RL gap; no statement equating the E-step with ZO is made.

**12. The M-sweep prediction**

- OLD: "the decisive test in the real system is to rerun the d = 21 arm at M = 128 and
  M = 512"; "a finite-sample deficit which closes to ~0.99 by M = 2048".
- CORRECTED: at the estimator level the LQR prediction held (alignment improves with M);
  at the return level it did not: HumanoidRun weighted-MLE return fell 666.17 (M = 32,
  n = 8) → 137.16 (M = 128, n = 5) → 10.98 (M = 512, n = 2), exploratory namespace;
  the Walker confirmatory return-level test is **PENDING** (16 runs planned, none
  launched).
- WHY: `reports/m_sample_count_audit.md`, `exports_manifest.csv`,
  `ledger/runs_m_sweep_confirmatory.jsonl`.
- CONSEQUENCE: "estimator quality ≠ policy quality"; the LQR §7.4 return prediction is
  not successful and must not be described as such.

**13. Provenance labels**

- OLD: single `git_sha 33c76325a5`; d = 6 absent; E2 unmentioned; `ceil(sqrt d)` rung
  unmentioned.
- CORRECTED: `estep_d64_rank1_M32.npz` from `98da377` (re-run twice); `d = 6` POST HOC
  (addendum A1) and excluded from every registered fit; E2 (prereg 5.5) NOT RUN;
  `ceil(sqrt d)` rung NOT RUN; `msweep_hi/hiw.py` and `probe2_full_estimator.py` were
  untracked when their results were produced (committed `c0a1ff1`); `.npz` ignored, hashes
  in `lqr_npz_manifest.csv`.
- CONSEQUENCE: none numerically; every number now carries its origin.

**14. Precision and reachability**

- OLD: "sigma < 0.1 is out of reach for the DMC arms" (limitations only); dtype unstated.
- CORRECTED: sweep dtype float64 established from the `__init__.py` at `33c7632`
  (`jax_enable_x64` set and asserted before any array), the stored dtypes of all 28
  cited files, and an independent recomputation of the `sigma = 0.01` corner; 8 of 20
  `sigma` columns (272 of 680 cells per file) lie below the effective `min_std = 0.1`
  and are marked outside the policy-reachable region wherever they enter a statistic.
- CONSEQUENCE: no small-`c` value is a precision artifact; every registered grid
  maximum lies in the unreachable region and is reported as such.
