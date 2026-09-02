# Preregistration 2: MC `Q^pi` oracle precision replication on corrected WalkerRun

**Status: IMMUTABLE once committed.** Written and committed before any pilot-2 oracle
value is computed. Preregistration 1 (`docs/prereg_mc_oracle_walker_pilot.md`, commit
`63c2cd2`) §11 states that the pilot specification is not changed after seeing results
and rerun under a new definition, and that *a second pilot would require a new
preregistration*. This is that preregistration. Preregistration 1 is not edited.

---

## 0. Why there is a second pilot

Pilot 1 (`reports/mc_oracle_walker_pilot.md`, commit `fd57fda`) returned
**NOT YET PRECISE ENOUGH TO SCALE**. Its own diagnosis:

* **Criterion B was the binding failure on both checkpoints.** The debiased whitened
  gradient energy was negative for PW (`N_0.10 = -4.14`, CI `[-21.67, +12.19]`) and
  had an interval crossing zero for WML (`5997`, CI `[-1366, +15863]`).
* **97.8% (PW) and 91.2% (WML) of the naive squared gradient was Monte-Carlo noise.**
* Criterion D (horizon) passed with a twentyfold margin, so truncation was never the
  problem.
* Criterion E failed on WML with relative interval width 1.26 against a 0.60 threshold.

**Pilot 2 is a precision replication, not a redesign.** Exactly three constants
change — the RNG root, the state count and the rollout count. The estimand, the
estimators, the finite-difference steps, the horizons, the horizon-doubling subset,
the bootstrap scheme, the decision thresholds and the interpretation rules are all
carried over unchanged from preregistration 1, which is incorporated here by
reference and remains binding wherever this document does not explicitly say
otherwise.

## 1. What changes, and nothing else

| Constant | Pilot 1 | Pilot 2 |
|---|---|---|
| RNG root | 20260902 | **20260903** |
| States `S` | 64 (32 per arm) | **256 (128 per arm)** |
| Rollouts per group `R` | 8 | **96** |
| Rollouts per action point | 16 | **192** |

Unchanged: the two checkpoints; `K = 16` perturbations; two independent rollout groups
A and B; `H = 500` primary with the `H = 1000` doubling subset (first 8 states of each
stratum, first 8 perturbations, `c = 0.10`); `c in {0.10, 0.05}` full coordinate
finite differences; `burn_in = 50`; the `K/(K-1)` centering correction; the
cross-product noise debiasing; the paired stratified hierarchical bootstrap at 10,000
replicates with `np.random.default_rng(20260902)`; criteria A-E and their thresholds;
and every interpretation rule in preregistration 1 §14.

Cost: `S * R` rises by a factor of **48**, to roughly 9.8e9 environment steps per
checkpoint for the main run plus 3.2e8 for the doubling subset — about 6 hours on one
H100 per checkpoint, from the measured 483k steps/s of pilot 1.

**Pilot 2 uses a fresh state bank at a new root and is statistically independent of
pilot 1.** It is not a top-up of pilot 1's rollouts, and pilot-1 states are not reused.
This makes the WML `r_RMS` figure an out-of-sample replication rather than a
refinement, and removes any question of reusing states after seeing what they produced.

## 2. Checkpoints — unchanged, and deliberately so

Exactly the two prospectively fixed checkpoints of preregistration 1 §1, with the same
checksums:

| Role | Path | `actor.npz` sha256 (first 16) | `critic.npz` sha256 (first 16) |
|---|---|---|---|
| PW-trained | `exports/WalkerRun_pathwise_fa_s301_final` | `e0cbb41c2bbd92ac` | `b8f22a654a53e8f7` |
| WML-trained | `exports/WalkerRun_weighted_mle_s301_final` | `0e68d7ecc210cb99` | `c56c4d27a3265ef1` |

**Seed 301 is retained for both arms. The WML checkpoint is not replaced.** Pilot 1
found its policy to be extreme — training-logged entropy `-33.34` against PW's
`-2.62`, pre-squash `sigma` with median 1.97 but mean 16.0 and maximum 1575.9, and
87.3% of base actions carrying at least one saturated coordinate. Swapping it out on
those grounds would be exactly the post-hoc selection this programme forbids: the
checkpoint was chosen prospectively, and its saturation is a **property to be measured
and reported**, not a defect to be designed around. The consequence — that a large
part of the WML `r_RMS` reflects `tanh` saturation rather than critic roughness — is
restated in §7 and will be restated in the report.

## 3. Primary estimand — unchanged

The **primary estimand remains the true unclipped soft `Q^pi`**, exactly as traced in
`reports/mc_oracle_code_trace.md` and defined in preregistration 1 §2:

```
Q_soft^pi(s0, a0) = E[ r_0 + sum_{t>=1} gamma^t ( r_t - alpha log pi(a_t|s_t) ) ]
```

`gamma = 0.99`; `alpha = 0.014509915374219418` read programmatically as
`float(actor.temperature())`; the externally fixed `a_0` carries no entropy term; `pi`
is the exported online actor at `scale = 1.0`; the environment receives
`clip(a, -0.999, 0.999)` at every step including `t = 0`; **no `Q_phi` bootstrap at the
horizon**; `Q_phi` is queried at the unclipped `tanh(y)`, matching `reppo.py:741-744`.

Centred error, the `r_RMS` definition, and the naming rule (*the RMS analogue of the
Claim-4 error-frequency ratio*, never "the measured `omega_infinity`") are as in
preregistration 1 §2.

## 4. Secondary estimand — the critic-target support, now preregistered

Pilot 1 discovered, and reported as an error in its own preregistered estimand, that
`src/jaxrl/utils.py:45` clips the regression target into `[vmin, vmax] = [0, 150]`
before the two-hot encoding, and that the 151-bin HL-Gauss head cannot represent a
value outside that interval either. So `Q_phi` is fit to `E[clip(G, 0, 150)]` while the
primary estimand is `E[G]`. On PW this changed nothing (0.0% of rollouts affected); on
WML it was material (30.6% of rollouts, 34.5% of action points).

That calculation was post-hoc in pilot 1. **In pilot 2 it is preregistered as a
SECONDARY diagnostic, and it does not replace the primary.** Two variants, both
reported:

| Name | Oracle | Rationale |
|---|---|---|
| `clip_per_rollout` | `mean_r clip(G_r, 0, 150)` | closest to how training clips each sampled target before encoding it |
| `project_mean` | `clip(mean_r G_r, 0, 150)` | the oracle mean projected onto the representable interval |

Both **approximate** the training target rather than matching it: training clips a
`lambda`-return that bootstraps on `Q_phi` (itself `>= 0`), whereas these clip a
500-step unbootstrapped MC return, so `E[clip(G_lambda)] != E[clip(G_MC)]`. Criteria
A-E are evaluated on the **primary** estimand only. The secondary variants are
reported with the same statistics beside it, and the GO/NO-GO decision does not
consult them.

Also reported: the fraction of rollouts below `vmin`, above `vmax`, and the fraction of
action points whose oracle mean falls below `vmin`.

## 5. Rollout count — sized from a measured scaling law, not a guess

`scripts/analysis/mc_oracle_power.py` recomputes the pilot-1 estimators using only
`R' in {1, 2, 4, 8}` of the 8 rollouts per group, averaging over the disjoint
sub-blocks available at each `R'`, bootstraps the sampling standard error at each, and
fits

```
se^2(R, S) = ( v_state + v_noise / R ) * (64 / S)
```

`v_state` is the irreducible state-heterogeneity floor that no number of rollouts can
cross. Fitted constants (`reports/artifacts/mc_oracle_power.json`, computed on
completed and already-reported pilot-1 data before this document was written):

| | `v_state` | `v_noise` | `se` at `S=64, R=8` |
|---|---|---|---|
| PW `D` | **4.807** | 12.54 | 2.593 |
| PW `N_0.10` | 0 | 2232 | 8.765 |
| WML `D` | 4922 | 1.485e5 | 176.2 |
| WML `N_0.10` | 0 | 2.978e8 | 4587 |
| WML `r_0.10` | 0 | 14.72 | 0.481 |

**The decisive finding is that PW's criterion A is blocked by a state floor.** Its
`se(D)` tends to `sqrt(4.807) = 2.193` as `R -> infinity`, against a criterion-A target
of `D/1.96 = 1.601`. **More rollouts alone could never have fixed pilot 1**, which is
why `S` rises as well as `R`. `N` and `r_RMS` fit `v_state = 0` — pure Monte-Carlo
noise — so for those, rollouts are quadratically more cost-effective than states
(`se ~ 1/R` at cost `~ R`, versus `se ~ 1/sqrt(S)` at cost `~ S`).

Projected standard errors at the chosen `S = 256, R = 96`:

| | projected `se` | criterion target | margin |
|---|---|---|---|
| PW `D` | 1.111 | 1.601 (criterion A) | 44% |
| PW `N_0.10` | 2.411 | resolves `|N| >= 4.73` | see §6 |
| PW `N_0.05` | 8.177 | resolves `|N| >= 16.0` | see §6 |
| WML `D` | 40.2 | 213.2 (criterion A) | 5.3x |
| WML `N_0.10` | 881 | 3060 (criterion B) | 3.5x |
| WML `N_0.05` | 3732 | 5517 (criterion B) | 1.5x |
| WML `r_0.10` | 0.196 | 0.2367 (criterion E) | 21% |

The `r_0.10` scaling constant is the least trustworthy of these: the all-points fit is
dominated by the `R' = 1` and `R' = 2` sub-blocks, where `r` is frequently undefined
and its bootstrap standard error is inflated well beyond the `1/R` law. Anchoring
instead on the well-behaved `R' = 4 -> 8` range implies four to eight times more
headroom. **The conservative all-points fit is used for sizing**, so the design is not
resting on the optimistic reading.

## 6. What pilot 2 can and cannot resolve — stated before it runs

Sizing a study on a quantity that the previous study failed to resolve has an
unavoidable circularity, and it is better named than hidden.

* **PW `N_0.10` is the case at risk.** Pilot 1 gave `-4.14 +- 7.41` on 64 states and
  `+15.05` on the 16-state doubling subset — at most 2.6 standard errors apart, both
  consistent with a true value anywhere in roughly `[0, 15]`. Pilot 2 resolves it iff
  the true `|N_0.10| >= 4.73`. **If PW's true gradient energy is below about 4.7,
  criterion B will fail again.** That outcome would not be a repeat of pilot 1's
  non-result: it would be a quantitative upper bound, `N_0.10 < ~4.7` at 48 times the
  sampling effort, and it should be reported as such.
* **PW `D` is also at risk.** Pilot 1 gave `3.138 +- 2.593`. Pilot 2 resolves it iff
  the true `D >= ~2.2`. A true `D` at the low end of pilot 1's interval would fail
  criterion A again even at the reduced state floor.
* **PW criterion E cannot be evaluated unless B passes first**, since `r_RMS` is
  undefined when `N <= 0`. This is a structural feature of the estimator, not a
  choice, and it is recorded now so it is not presented later as a surprise.
* **`N_0.05` is intrinsically four times noisier than `N_0.10`** because `z ~ 1/c`
  gives noise variance `~ 1/c^2`; pilot 1 measured 3.8x (PW) and 5.1x (WML) against
  the predicted 4x. Criterion C therefore remains the hardest of the five, and a
  failure of C must again be reported as *finite-difference bias unresolved* rather
  than as bias demonstrated.

## 7. Carried-over cautions that pilot 2 does not fix

* **WML `tanh` saturation.** 87.3% of pilot-1 WML base actions had a saturated
  coordinate. Where the action saturates, `Q^pi` is genuinely flat in `y` while
  `Q_phi` is not, so a large part of the WML `r_RMS` measures saturation rather than
  critic roughness. More rollouts do not change this. The preregistered no-clip
  sensitivity retained only 12.4% of WML points in pilot 1 and was uninformative
  there; it is computed again and will again be reported with its retention fraction.
* **Material anisotropy.** Pilot-1 WML per-dimension `sigma` means ranged 6.3 to 46.2,
  anisotropy 51.7. Per preregistration 1 §12, **no scalar `sigma*omega` point is
  forced onto the isotropic `(sigma*omega, sqrt(d))` plane**; placement is `r_RMS`
  against 1.
* **`Q_phi` is structurally confined to `(0, 150)`** by its support.
* **The `lambda`-return truncation flag is off by one** (`reports/mc_oracle_code_trace.md`
  §3), affecting ~1/1000 of training transitions.
* **`H = 500` is truncation.** The term remains *finite-horizon Monte-Carlo estimate of
  soft `Q^pi`*. No finite per-step bound on the soft reward exists, so `gamma^H` is
  again not offered as a truncation-bias bound.

## 8. Validation

`scripts/analysis/test_mc_oracle.py` must pass in full before the pilot-2 run, as it
did 18/18 on CPU and 18/18 on the H100 for pilot 1. T1 and T1b remain the state
restoration gate. **No test is weakened after inspecting the pilot-2 outcome.**

The pilot registry in `scripts/analysis/mc_oracle_walker.py` (`PILOTS["p1"]`,
`PILOTS["p2"]`) defaults to `p1`, so the pilot-1 numbers remain exactly reproducible
from the same file; `set_pilot` refuses any tag that is not registered. A regression
check confirming that the analysis code reproduces the pilot-1 primary values
(`D = 3.1384` / `417.7896`, `N_0.10 = -4.1428` / `5997.2723`, verdict unchanged) was run
before this document was committed.

## 9. Decision rule — unchanged from preregistration 1 §11

For **each** checkpoint: **A** `D > 0` with bootstrap lower bound `> 0`; **B**
`N_0.10 > 0` and `N_0.05 > 0` with bootstrap lower bounds `> 0`; **C**
`r_0.05/r_0.10` and `N_0.05/N_0.10` both in `[0.8, 1.25]`; **D** centred
`RMS(Q_1000 - Q_500) < 0.25 * ` centred `RMS(e_500)` and the `H=1000/H=500`
gradient-energy ratio at `c = 0.10` in `[0.8, 1.25]`; **E**
`width(95% CI for r_0.10) <= 0.60 * point`.

Negative cross-products are reported as `UNRESOLVED - ORACLE PRECISION INSUFFICIENT`
and are **not clamped to zero**. Placement stays three-valued:
`BELOW BOUNDARY` / `NEAR BOUNDARY - INTERVAL CROSSES 1` / `ABOVE BOUNDARY`.

**PASS TO SCALE only if both checkpoints satisfy A-E.**

## 10. Stop rule

**This is the last pilot run under this preregistration.** If both checkpoints pass
A-E, the verdict is `ORACLE PILOT FEASIBLE - SCALE TO ALL 16 CORRECTED WALKER
CHECKPOINTS`, and **that scaling job is not launched in this task** — it requires
explicit approval. If either fails, the verdict is
`ORACLE PILOT NOT YET PRECISE ENOUGH TO SCALE`, the dominant source is named from the
list in preregistration 1 §11, and **no third pilot is run without a third
preregistration**.

## 11. Conditional follow-up: covariance-freeze ablation

Pilot 1's evidence of abnormal WML policy-width growth (entropy `-33.34` vs `-2.62`;
`sigma` median 1.97, mean 16.0, max 1575.9; 87.3% saturation) came from one checkpoint
on a 64-state bank. Pilot 2 re-measures all of it on an independent 256-state bank.

**If that evidence persists**, the next step is a *separate* preregistration for a
covariance-freeze ablation testing whether variance and saturation dynamics cause the
WalkerRun performance gap. It is **not** designed, preregistered or run in this task,
and nothing in pilot 2 licenses a causal claim about the return gap. The trigger
condition is recorded now so the decision is not made retrospectively:

> The ablation is preregistered only if, on the pilot-2 bank, the WML checkpoint's
> per-state RMS `sigma` has a p95 exceeding five times its median **and** the base
> action clip rate exceeds 50%.

Pilot 1's values were p95 `90.05` against median `4.26` (a ratio of 21.1) and a clip
rate of 87.3%, so on pilot-1 evidence the trigger would fire; pilot 2 decides it on
independent data.

## 12. Registered expectation, to be scored afterwards

Pilot 1's registered expectation was mostly wrong — I predicted an interval-width
failure and got a sign failure — and it was scored in the report. Recording another:

* I expect **WML to pass A, B and E** at this rollout count, and to pass **D**.
* I expect **criterion C to remain the binding failure**, on WML through the
  `N_0.05/N_0.10` ratio, because the `c = 0.05` arm retains four times the noise
  variance and pilot 1's ratio was 1.80 against a `[0.8, 1.25]` band.
* I expect **PW to remain the harder case**, and I put roughly even odds on PW's
  criterion B: pilot 1's two estimates of its `N_0.10` straddle the `4.73` resolution
  threshold.
* I expect **WML's `r_0.10` to land between 1.2 and 1.9**, replicating pilot 1's 1.547
  out of sample, and its interval still to cross 1.
* I expect the overall verdict to be **NOT YET PRECISE ENOUGH TO SCALE** again, driven
  by C rather than by B.

---

## Design lock

Everything above is fixed at the commit that adds this file. The RNG root, state
count, rollout count, checkpoints, estimands, estimators, finite-difference steps,
horizons, doubling subset, bootstrap scheme, decision thresholds, stop rule,
follow-up trigger and interpretation rules are all settled before any pilot-2 oracle
value exists.
