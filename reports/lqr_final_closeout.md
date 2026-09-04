# LQR crossover study: final close-out

**Scope.** Closes the LQR study. Inputs: `docs/prereg_lqr_crossover.md` (`33c7632`,
addendum A1 `d2cb9f6`), the corrected canonical report `reports/lqr_crossover_corrected.md`
(`00091c4`), the two audits (`fc23826`, `53e75d9`/`bbfbf59`), and the one preregistered
item completed here, the Rule B rung `r = ceil(sqrt d)` (run note `5cb384d`, committed
before the run; artifacts `reports/artifacts/lqr_ruleB_fourrungs*.{csv,json}`,
`lqr_ruleB_fourrungs_rung_ceilsqrt.csv`). No settled analysis was reopened; no
exploratory experiment was added.

---

## Preregistration closure table

Read from `docs/prereg_lqr_crossover.md` verbatim; nothing inferred from the historical
report.

| item | exact preregistered requirement (quoted) | unconditional or triggered | trigger | already satisfied? | action required |
|---|---|---|---|---|---|
| E0 / E0a gates G1–G11 | Sec. 8 table; Sec. 1: gates "had run and passed" before commit | unconditional (precondition) | — | yes (Sec. 8; G6a/G6b re-verified in the correctness audit) | none |
| Rule A, rank-one E1a | 5.2: "`p in [0.35, 0.65]` AND the per-`d` `c*` within 4 Monte Carlo standard errors of the Sec. 3 closed form: VERIFIED" | unconditional | — | yes — VERIFIED (measurement-matched comparator ≤ 2.8 SE on the registered `d`-set; `lqr_corrected_ruleA.csv`) | none |
| E1b | Sec. 4 prediction; no decision rule in Sec. 5 | descriptive | — | reported (corrected report Sec. 4) | none |
| Rule B rung `r = 1` | 5.3: "Ranks `r in {1, 2, ceil(sqrt(d)), d}` … Fit `p` as above at each rank, and fit the joint exponent across the ladder" | unconditional | — | yes (`rank1`) | none |
| Rule B rung `r = 2` | same | unconditional | — | yes (`rank_r2`); CI computed at `00091c4` | none |
| **Rule B rung `r = ceil(sqrt(d))`** | same | unconditional | — | **no** at `00091c4` | **run — done here** (`5cb384d` note; files `*_csd.npz`) |
| Rule B rung `r = d` | same | unconditional | — | yes (`full`); CI computed at `00091c4` | none |
| Rule B joint adjudication | 5.3 three branches + CONTAMINATED | unconditional | — | partially at `00091c4` (three rungs) | **complete on four rungs — done here** |
| 5.4 statistics (root-find, no-bracket, collapse, bootstrap 10 000, cross term) | 5.4 verbatim | unconditional | — | yes; departures recorded in the corrected report (aggregation order, cross-term rule replaced post hoc, single-level counts rerun at 10 000) | none |
| eps-invariance arm | Sec. 7 guard: "the eps-invariance arm at `eps_frac in {0.05, 0.20}`" | unconditional guard | — | yes (`eps20` files) | none |
| **E2** | 5.5 title: "E2 (follow-on; registered now so it cannot be chosen later)". Body registers the metric `S`, the initial-state law, and implementation requirements, and states: "A coarse `gamma`-discounted-occupancy arm is run on a subgrid; if the return crossover moves between metrics, that is reported in one sentence." Sec. 1: "Committed before: … any E2 run." | **ambiguous** — the section fixes *how* a follow-on would be run and a one-sentence reporting rule; it names no trigger, no decision rule, no grid, no statistic, and no definition of the "return crossover" it refers to; nothing in Secs. 4–5 or 9 conditions any verdict on E2 | none stated | no | **E2 STATUS: REGISTERED BUT REQUIREMENT AMBIGUOUS** — not run, not reinterpreted |
| Sec. 9 "If Rule B is refuted" | "the dimensional prediction comes out of the abstract, and the paper reframes around `omega` alone — while stating … that `omega` itself is not well-posed for a full-rank error field until the norm in Claim 4 is fixed" | triggered | Rule B REFUTED | trigger satisfied (below) | paper-side reframing, not an experiment |

Every unconditional experimental obligation is now satisfied. The one registered item
that is not run is E2, whose requirement is ambiguous as written.

---

## A. The `r = ceil(sqrt d)` rung

**Design, as registered and as run** (`reports/lqr_rank_ceilsqrt_runnote.md`, `5cb384d`,
committed before the run; `git_sha` in every file = `5cb384d`, `prereg_sha = d2cb9f6`):
`kind = rank_r`, `r(d)` = 1, 2, 2, 3, 4, 6, 8 for `d` = 1, 2, 4, 8, 16, 32, 64; the
registered `d`-set, 20 × 34 `sigma` × `omega` grid, `M = 32`, `eps = 0.05 q_spread`,
`unit_H`, identity cost, registered seeds, 32 states × 20 × 100 (`N = 2000`, the ladder's
replicate count), registered crossover extraction and guards. Nothing was changed after
the existing results. Files tagged `_csd` (sha256 in `lqr_npz_manifest.csv`).

**Provenance gates.** Where `ceil(sqrt d)` coincides with an existing rung (`d = 1, 2, 4`)
the new files are **bit-identical** to the committed `rank_r2` files, as the registered
seeds require. Guards: `rho_closed` 0.452–0.690, `cond_H` 1.00–16.19, 0 retries at every
`d`, `eps/q_spread = 0.05` — not CONTAMINATED. Every `d` bracketed (0 % of states without
a bracket); collapse sd 0.0008–0.0101 (registered < 0.05).

| d | r | c* | omega_RMS/omega at crossover | c*·omega_RMS/omega | c*/sqrt r |
|---|---|---|---|---|---|
| 1 | 1 | 1.2011 | 0.9960 | 1.1963 | 1.2011 |
| 2 | 2 | 1.5215 | 0.9712 | 1.4777 | 1.0758 |
| 4 | 2 | 2.0489 | 0.9987 | 2.0462 | 1.4488 |
| 8 | 3 | 2.8693 | 1.0000 | 2.8693 | 1.6566 |
| 16 | 4 | 4.0597 | 1.0000 | 4.0597 | 2.0299 |
| 32 | 6 | 5.7433 | 1.0000 | 5.7433 | 2.3447 |
| 64 | 8 | 8.1360 | 1.0000 | 8.1360 | 2.8765 |

**Exponents (registered `d`-set {2, 4, 8, 16, 32, 64}).**

| quantity | value | kind | 95 % CI |
|---|---|---|---|
| `p_nominal` | **0.4873** | direct fit | [0.4860, 0.4886] (registered hierarchical bootstrap, 10 000, seed 20260902) |
| `p_RMS` | **0.4935** | direct fit of `c*·omega_RMS/omega` (`omega_RMS` from the exact realised field moments) | — |
| `p_omega_inf` | **0.2706** | algebraic relabelling: `p_nominal + slope of −½ ln ceil(sqrt d) on ln d = 0.4873 − 0.2167` | [0.2693, 0.2719] (rigid shift of the nominal CI) |

The run note's expectation was recorded before inspection and was not used in the
analysis: the nominal exponent stayed near 1/2 (0.4873); the sup-norm coordinate moved it
by the coordinate algebra of the step function `ceil(sqrt d)` (−0.2167, near the rough
−1/4); the resulting 0.27 is not a new scaling law.

### Rule B — complete adjudication on all four registered rungs

Registered wording (5.3): ranks `{1, 2, ceil(sqrt(d)), d}` "under the registered primary
norm convention" (`omega_inf = sup||grad e||_2 / ||e||_inf = omega/sqrt r`);
CONFIRMED requires "`p in [0.35, 0.65]` at every rank, with 95 % bootstrap CIs excluding
0 and excluding 1"; the third branch is "`p` outside both bands, or `c*(d)` not
monotone in `d`, or `p` differing across ranks by more than the CIs allow".

| rung | `p_nominal` (fit) | `p_omega_inf` (registered coordinate) | 95 % CI | in [0.35, 0.65] | CI excludes 0 and 1 | in [0.8, 1.2] | `c*(d)` monotone |
|---|---|---|---|---|---|---|---|
| r = 1 | 0.4876 | 0.4876 | [0.4869, 0.4883] | yes | yes | no | yes |
| r = 2 | 0.4870 | 0.4870 | [0.4854, 0.4887] | yes | yes | no | yes |
| r = ceil(sqrt d) | 0.4873 | **0.2706** | [0.2693, 0.2719] | **no** | yes | no | yes |
| r = d | 0.4871 | **−0.0129** | [−0.0139, −0.0118] | **no** | **no** | no | yes |

Two rungs lie outside both bands and the rungs' intervals are disjoint. Under the
registered joint rule:

```
RULE B: REFUTED
```

(third branch; not CONTAMINATED). This replaces the historical "REFUTED at full rank /
CONFIRMED at fixed rank" with the single verdict the registered rule returns. Per Sec. 9
the `sqrt(d)` dimensional prediction comes out of the abstract and the paper reports the
rank dependence — while stating that `omega` is not well-posed for a full-rank field until
Claim 4 fixes its norm. Source: `lqr_ruleB_fourrungs.csv/json`.

## B. E2

Quoted above. **E2 STATUS: REGISTERED BUT REQUIREMENT AMBIGUOUS.** Not run; not
reinterpreted; no evidence was created to fill it.

---

## The eight questions

**1. Did the production estimator implementation pass validation?** Yes. The harness
imports `src/jaxrl/estimators.py` and its kernel arithmetic was verified against it and
against independent numpy implementations to 1e-14; `Q^pi` gradient 4e-8 by finite
differences; planted field 3e-10; linearity in `Q` 4e-14; `M/(M-1)` applied once; float64
end to end (`lqr_code_correctness_checks.csv`, 132 checks).

**2. Did the critic-error-channel mechanism survive?** Yes, as a controlled-family
identity. `V_e = tr Cov[g(Q^pi+e;xi) − g(Q^pi;xi)]` drops no cross term; the two error
channels cross at `c* ≈ sqrt(d M/(M−1))` at every tested rank, and
`(Var_e[ZO]/Var_e[PW]) r_RMS^2 = 32/31` to 0.4 % at `d ≥ 8`. Qualification: at the
crossover the channel is ≤ 5 % (PW) / ≈ 0.1 % (ZO) of total estimator error, and the
rank-one exponent is fixed at 1/2 by construction.

**3. What does RMS frequency show?** `omega_RMS/omega = 1.000` at every crossover cell of
every rung (0.971–0.999 at `d ≤ 2`), so the nominal exponents are already
RMS-convention exponents: 0.4876 / 0.4870 / 0.4873 / 0.4871 across `r = 1, 2,
ceil(sqrt d), d` (`p_RMS` for the new rung 0.4935). The nominal/RMS crossover does not
move with the error field's rank. This is an empirical statement about the planted
sinusoidal families here, kept separate from the sup-norm bound.

**4. What happened to the universal sup-norm `sqrt(d)` claim?** Refuted. In the registered
sup-norm coordinate the exponent is 0.4876, 0.4870, **0.2706**, **−0.0129** across the
four rungs — each an algebraic relabelling of a ≈ 0.487 nominal exponent by
`omega_inf/omega = 1/sqrt r(d)`. The mapping from nominal/RMS to sup-norm frequency is
rank dependent, so no universal `sqrt(d)` statement survives in that coordinate.

**5. What is Rule B's final preregistered verdict?** **REFUTED**, on all four registered
rungs under the registered joint rule (table above).

**6. Was E2 required, and if so what happened?** Its requirement is ambiguous as
registered (no trigger, no decision rule, no design beyond the metric and state law); it
was not run. E2 STATUS: REGISTERED BUT REQUIREMENT AMBIGUOUS.

**7. What does LQR NOT establish?** Anything about learned neural critics (`omega` is
unmeasured there and `e` is planted, quadratic `Q^pi`); anything about DMC returns; that
the shipped E-step is the centred estimator (first-order bridge only; `ubar`
dimension-amplification refuted on trained checkpoints); that larger `M` improves the
policy (estimator alignment improved, HumanoidRun return collapsed 666 → 137 → 11; Walker
confirmatory test pending); that the registered cross-term diagnostic is satisfied (it
flags every `d` and diverges by construction in the policy-unreachable corner); a
`sqrt(d)` threshold in the sup-norm frequency.

**8. Are there any remaining LQR experiments required before submission?** No. Every
unconditional preregistered LQR item is complete; the only registered item not run (E2)
is ambiguous as written and is reported as such rather than reinterpreted.

---

## Paper-facing table

| paper claim | result | status | main text / appendix / omit |
|---|---|---|---|
| The production PW and centred-ZO estimators reproduce the closed-form error-channel identities in a controlled LQR | `c*` within 2.8 SE of the measurement-matched closed form at every registered `d`; `p = 0.4876` [0.4869, 0.4883] | VERIFIED | appendix (one paragraph in main text) |
| The critic-error-channel crossover collapses onto `sigma·omega` and scales as `sqrt(d M/(M−1))` for a rank-one planted field | collapse sd ≤ 0.008; `p` pinned to 1/2 by construction | VERIFIED | appendix |
| The nominal/RMS crossover is unchanged across error-field rank `r ∈ {1, 2, ceil(sqrt d), d}` | `p_nominal` 0.4876 / 0.4870 / 0.4873 / 0.4871; per-`d` `c*` agree to < 0.5 % | SUPPORTED | main text (one sentence) + appendix |
| `(Var_e[ZO]/Var_e[PW]) r_RMS^2 = M/(M−1)` for the de-attenuated estimator | 32/31 to 0.4 % at `d ≥ 8`, all four rungs | SUPPORTED (controlled family) | appendix |
| The zeroth-order operator has smaller critic-error variance when `sigma·omega_inf > sqrt(d)` universally (Claim 4 in the registered sup-norm frequency) | exponent 0.49 / 0.49 / 0.27 / −0.01 across rungs | REFUTED | main text: withdraw from abstract; state the norm ambiguity (prereg Sec. 9) |
| Rule B (four-rung joint rule) | third branch | REFUTED | appendix (verdict stated) |
| At the error-channel crossover the error channel is a few percent of total estimator error; the total-error crossover is 29–36× higher and scales as `1/eps` | `Var_e/MSE_total` ≤ 5 % (PW), ≈ 0.1 % (ZO); E1b ratios 28.7–36.1; ratio 4.00 for 4× `eps` | SUPPORTED | appendix |
| Registered cross-term diagnostic (`|2Cov| > 0.25 Var_e` anywhere on the grid) | flagged at every `d`; maxima at `sigma = 0.01` (policy-unreachable); `∝ c^{−2}` by construction | REFUTED (as registered) | appendix, with the policy-reachable reading beside it |
| At-crossover cross-term reading flags `d ∈ {1, 2, 4}` (`d = 8` marginal) | reachable level-set medians ZO 41 / 20 / 7.6; PW 19 / 9.5 / 3.7 | POST-HOC | appendix, labelled post hoc |
| `d = 6` (Walker's `d`) sits inside the at-crossover flag set | PW 0.330 / ZO 0.857 | POST-HOC | appendix, labelled post hoc |
| The shipped E-step equals `sigma·ubar + (sigma^2/eta) g_ZO` to first order | verified (rel. 7e-4 at `eta = 10^3`); `ubar`-dominated at that order; first-order identity holds in 1/10 trained conditions | SUPPORTED (as a bridge only) | appendix |
| `ubar` dimension amplification explains the real RL gap | Rho 1.035 / 0.485 vs 1.3 (WalkerRun 6 → 22) | REFUTED | appendix (one sentence) |
| Larger `M` improves estimator/estimand alignment in the LQR | gap 0.1151 → 0.0162 at `d = 21` | SUPPORTED (exploratory) | appendix |
| Larger `M` improves weighted-MLE policy return | HumanoidRun 666.17 → 137.16 → 10.98 (exploratory); Walker confirmatory PENDING | REFUTED (exploratory) | main text (one sentence), Walker status stated |
| E2 stationary-`S` / discounted-occupancy follow-on | not run; requirement ambiguous | NOT TESTED | omit (state in appendix that it is registered and unrun) |
| Any statement about learned critics or DMC returns derived from the LQR | — | NOT TESTED | omit |

---

```
LQR STATUS: FREEZE
```

**No further LQR experiments are recommended for this paper.**
