# Log-sigma decomposition — the displacement mechanism is NOT SUPPORTED

> **Verdict under the decision rule frozen in `docs/prereg_logsigma_decomp.md` (commit
> `5edaca3`): NOT SUPPORTED.** The pathwise arm shows a systematically positive curvature
> residual, which that rule names explicitly as a not-supported condition, and its magnitude
> exceeds the weighted-MLE displacement term with a bootstrap interval excluding zero.

Job `3823977`, commit `0948dfc`, node n23g0009, 00:01:16, 2026-09-07T18:50:38+02:00.
Script sha256 `255a180526d9e0abab5fc2c2a69f66e73c84f4fec98d1fdfc94bb16f59079d54`.
Bank hashes verified in-run; per-checkpoint actor/critic hashes in
`logsigma_theory_provenance.json`.

## 1. The verdict, and what carried it

The rule was: SUPPORTED iff P1 and P2 and P3; NOT SUPPORTED if `A` is indistinguishable
from zero gate-open **or if PW shows a comparable systematic positive residual**.

**PW's curvature residual is systematically positive**, not sign-varying as predicted:

| task | fraction of states with curvature residual > 0 |
|---|---|
| walker PW_ent | 0.748 |
| g1 PW_ent | 0.929 |
| leap PW_ent | 0.636 |

and the direct comparison runs the wrong way:

```
walker   median A_WML - median |curv|_PW  =  -0.0895   95% CI [-0.1016, -0.0777]   excludes 0
```

The pathwise curvature term is **larger** than the weighted-MLE displacement term. **P3
fails.** Under the frozen rule that alone settles the verdict.

## 2. Two defects in this measurement, disclosed

Both were caught by checks written into the preregistration before the run.

**The gate proxy does not reproduce the training gate.** The prereg predicted a gate-open
fraction of 0.45-0.53, matching the logged `fr_gate_operator`, and required any material
deviation be reported as a discrepancy. Measured:

```
gate-open fraction: walker 0.026 / 0.052,  g1 0.000,  leap 0.000   (M = 32)
```

The prereg defined the trust-region KL as `KL(pi_old || weighted-MLE fit)`. The repository's
gate uses `KL(pi_old || pi_theta)` — how far the **policy** has moved from `pi_old`, not how
far the **fitted target** lies from it. With finite `M` the fitted `sigma_w` is biased low,
so the KL is large almost everywhere and the gate reads closed. Consequently `A_open` is
undefined for G1 and LEAP and **P1 is untestable on two of three tasks**. Walker alone has
gate-open states, where `A_open = 0.0524`, 95% CI `[0.0445, 0.0606]`, excluding zero.

**P2 as computed is vacuous.** The script evaluated `spearman(A, sqrt(A))` rather than
`spearman(A, ||Dmu||)`. Since the first pair is monotone by construction the number carries
no information. **P2 is not tested.**

Neither defect changes the verdict: P3 is independent of both and fails.

## 3. The quantitative prediction also fails

`A_pred = sigma^2 ||g||^2 / eta^2`, from exact exponential tilting of a Gaussian under a
locally linear `Q`, with `g` the pre-tanh gradient `grad_a Q * sech^2(mu)`:

| task | arm | A measured | A predicted | ratio | Spearman |
|---|---|---|---|---|---|
| walker | PW_ent | 0.3436 | 0.2165 | 1.59 | 0.747 |
| walker | WML_noent | 0.2347 | 0.9503 | 0.25 | 0.383 |
| g1 | PW_ent | 1.6220 | 1.9846 | 0.82 | 0.740 |
| g1 | WML_noent | 1.5693 | 2.6000 | 0.60 | 0.726 |
| leap | PW_ent | 1.2268 | 2.2342 | 0.55 | 0.522 |
| leap | WML_noent | 0.7929 | **427.79** | **0.002** | 0.340 |

The regression slope of `A` on `A_pred` is approximately zero in every cell. The
linearisation is exact only for small `sigma`; at the `sigma ~ 3-7` of the wide
weighted-MLE cells, with tanh saturating, it collapses — which is precisely the regime the
mechanism was proposed to explain.

## 4. The seed-population prediction is refuted

`sigma* ~ 1/||g||` implies a policy should be narrow where the critic's action-gradient is
large, so own-seed states should carry **larger** `||g||`. Measured:

| task | arm | ‖g‖ own-seed | ‖g‖ sibling | ratio | sigma own | sigma sibling |
|---|---|---|---|---|---|---|
| walker | PW_ent | 0.0505 | 0.3171 | 0.16 | 0.529 | 0.509 |
| walker | WML_noent | 0.1440 | 0.0218 | **6.61** | 0.387 | 30.955 |
| g1 | PW_ent | 0.0169 | 0.0425 | 0.40 | 0.360 | 0.379 |
| g1 | WML_noent | 0.0173 | 0.0629 | 0.27 | 0.287 | 1.069 |
| leap | PW_ent | 0.0431 | 0.1003 | 0.43 | 0.285 | 0.974 |
| leap | WML_noent | 0.0393 | 0.0975 | 0.40 | 0.261 | 45.473 |

Only Walker `WML_noent` matches the prediction. The other five cells, including both other
weighted-MLE cells, go the opposite way. **Refuted.**

## 5. What survives

`Term A` is real, non-negative by construction, and non-zero: pooled medians 0.23 to 1.62
across cells. The exact algebraic split `dL/dlog_sigma = -(A + B)` holds and is a correct
statement about the M-step. What does **not** hold is the claim that this term is what
distinguishes the two operators: the pathwise arm's curvature term is comparably sized and,
on Walker, larger.

The descriptive results this project has established are untouched by this outcome —
matched-state widths across three tasks, the 48/48 paired seed ratios, the seed-population
dependence, and the entropy factorials. None of them depended on this mechanism.

## 6. Consequence for the eps_E arm

`docs/prereg_eps_e_theory_test.md` (commit `0948dfc`) states in advance: if the
decomposition finds `A` indistinguishable from zero under `5edaca3`'s rule, "this arm's
result is reported but carries no theoretical weight." The rule has failed for a different
reason — a comparably sized pathwise residual rather than a vanishing `A` — but the
conclusion is the same. **The `eps_E = 0.1` arm no longer tests the tilting theory.** It
remains a valid measurement of whether `eps_E` moves trained width, and is reported as
exploratory.

## 7. What would be needed to test a mechanism properly

Recorded so the next attempt does not repeat this one's errors.

* A gate proxy that reproduces the training gate, i.e. `KL(pi_old || pi_theta)` against the
  policy as it was during training, not against the fitted target. That requires
  `actor_target`, which is not exported, so it needs either a training-time probe or a
  calibrated surrogate whose gate fraction is validated against the logged
  `fr_gate_operator` before use.
* A setting where the linearisation is exact by construction rather than assumed. The LQR
  and planted-critic-error machinery already in this repository provides analytically known
  `Q`, which removes both the nonlinearity and the critic-error confound that defeated the
  prediction here.
* `P2` computed as specified.
