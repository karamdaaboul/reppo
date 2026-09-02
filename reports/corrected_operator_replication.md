# Corrected operator replication: faithful repair under the published REPPO gate

**1. Did the original g1 difference reproduce under corrected code?**
**Yes.** `PW-1 minus WML-32` on `G1JoystickFlatTerrain` (d=29) is **+9.51** (paired
median, final-window score), 95% CI **[+2.78, +15.28]**, **8/8** seeds positive, exact
two-sided p = **0.0078**. The interval excludes zero and the sign matches the original.
**A task-level operator difference persists under the corrected common trust-region
implementation.**

**2. Did the Walker control remain a non-detection?**
**No — and this is the most consequential result here.** `WalkerRun` (d=6) now shows
**+135.58** [**+108.97, +200.76**], **8/8** positive, p = **0.0078**. In the old
64-run study the same contrast was **+2.68 with 5/8** and was not detected. The control
task has become a detection, and its effect is an order of magnitude larger than g1's.

**3. Was exact HumanoidRun reproduction possible?**
**No.** `HUMANOIDRUN CORRECTED REPLICATION NOT REPRODUCIBLE FROM RETAINED ARTIFACTS`
(§4). It was excluded; no task was substituted.

**4. Did every construct-validity test pass?**
**Yes.** Legacy parity exact (36/36 arrays byte-identical, max abs diff `0.0`);
15/15 correctness tests; 14/14 smoke acceptance on both arms; 32/32 integrity.

---

## 5. Provenance

| | |
|---|---|
| Design lock | `796e66e` `docs/faithful_repair_design.md` |
| Source trace | `a3d2352` `reports/corrected_replication_code_trace.md` |
| **Correction code** | **`cfbd8dd`** |
| **Preregistration** | **`7edb8e8`** `docs/prereg_corrected_operator_replication.md` |
| Pre-launch ledger | `b3e3c39`, checksum `b9095160ba93b02b72bc633712a4aeb7a8e916babae17af06c265b639a08a41e` |
| Launch commit | `1c6259e` |
| Scheduler | `3444831` (c23g, 16), `3444832` (c25g, 16) |
| Cost | 19.5 GPU-h over 32 runs |

Analysis environment: Python 3.12.14, JAX 0.5.2, flax 0.10.6, H100 (c23g 94 GB /
c25g 80 GB).

## 6. Code corrections

Exactly **two** behaviour-affecting changes, enabled together
(`faithful_same_point`, `fresh_minibatch_key`; plus `log_faithful_diag` for
diagnostics). **This experiment cannot attribute any outcome to either individually.**

1. **Same-point likelihood.** The pre-squash latent `y_i` is materialised and both the
   old- and new-policy log probabilities are computed from it. No clip is applied, so
   the corrected-path clip rate is **zero by construction**; the tanh Jacobian cancels
   exactly (`2.8e-17`); `arctanh` is never called.
2. **Fresh minibatch innovations.** The actor key is split inside the minibatch scan.
   Only the actor stream is touched, so the environment realisation for a seed is
   unchanged.

**Preserved unchanged, as published REPPO design:** the piecewise policy/KL gate and
the unbounded exponential log-space multiplier. Git history places both — *and the
clip that separated the two log probabilities* — in upstream `c3b13de` (2025-08-04),
long before this project began. The repair therefore fixes an **upstream coding
inconsistency**, not one introduced here, and the arms differ from *released* REPPO in
the two ways above.

## 7. Parity evidence

With every flag off, against pristine `07319d4` on identical config and seed:
**36/36 arrays byte-identical, max abs difference `0.000000e+00`**, and
`eval_return_curve`, `kl_curve`, `entropy_curve`, `pi_sigma_curve`, `alpha_curve` all
identical. Job `3442080`.

## 8. Integrity audit

32/32 completed, no failures, no missing runs, no duplicate task–arm–seed, 32/32
exports with checksums, single horizon `52 297 728`, single `num_eval = 20`, one frozen
α per task identical across arms, no non-finite values, 21/21 diagnostic fields
non-zero in every run, counterbalancing exactly 4 per task×arm×architecture.

**Reload verification:** rolling each final checkpoint's own policy reproduces the
logged return with correlation **0.9989** across 32 checkpoints, median relative
difference **6.4%**, max **62.7%**. The max sits on a low-return g1 checkpoint where a
small absolute difference is a large relative one; the rank ordering is preserved. This
is a stochastic-policy rollout under a different evaluation seed, not a bit-level
reproduction, and is reported as such.

## 9. Seed-level results (all eight pairs, both tasks)

Primary score = mean of the final three of 21 evaluations. Contrast `PW-1 − WML-32`;
positive = pathwise higher. Unit = the seed pair.

### WalkerRun, d=6

| seed | PW-1 | WML-32 | difference |
|--:|--:|--:|--:|
| 301 | 909.79 | 784.55 | +125.24 |
| 302 | 898.47 | 777.20 | +121.27 |
| 303 | 907.87 | 798.90 | +108.97 |
| 304 | 910.98 | 710.22 | +200.76 |
| 305 | 908.44 | 762.51 | +145.93 |
| 306 | 909.28 | 717.01 | +192.27 |
| 307 | 912.19 | 887.08 | +25.11 |
| 308 | 915.03 | 646.53 | +268.50 |

IQM: PW-1 **909.62**, WML-32 **760.32**. Paired median **+135.58**, mean **+148.51**,
95% CI **[+108.97, +200.76]**, **8/8** positive, exact p **0.0078** → **DETECTED**.

### G1JoystickFlatTerrain, d=29

| seed | PW-1 | WML-32 | difference |
|--:|--:|--:|--:|
| 301 | 26.98 | 15.62 | +11.36 |
| 302 | 29.60 | 17.41 | +12.19 |
| 303 | 18.79 | 14.53 | +4.26 |
| 304 | 11.92 | 9.14 | +2.78 |
| 305 | 23.59 | 15.92 | +7.67 |
| 306 | 11.15 | 9.33 | +1.82 |
| 307 | 30.01 | 11.84 | +18.17 |
| 308 | 29.70 | 14.42 | +15.28 |

IQM: PW-1 **24.74**, WML-32 **14.10**. Paired median **+9.51**, mean **+9.19**,
95% CI **[+2.78, +15.28]**, **8/8** positive, exact p **0.0078** → **DETECTED**.

**Secondary score** (last evaluation only, the 64-run study's locked definition):
Walker **+129.30** [+98.98, +254.89], 8/8; g1 **+7.56** [+1.05, +14.71], **7/8**,
p = 0.0703. Both intervals still exclude zero; g1's sign count weakens to 7/8 under
this definition, which is reported rather than selected against.

## 10. Preregistered KL and gate diagnostics (diagnostic only)

Medians over 8 seeds. None of this entered training, the gate, the dual, or any
exclusion.

| task | arm | sampled KL | analytic KL | bias | RMSE | corr | operator branch | KL-only branch | λ | σ | ESS |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| walker | PW-1 | 0.0864 | 0.0868 | −0.0002 | 0.0015 | 0.986 | 0.549 | 0.451 | 0.026 | 0.531 | — |
| walker | WML-32 | 0.0874 | 0.0913 | −0.0037 | 0.0040 | 0.995 | 0.553 | 0.447 | 0.346 | 1.981 | 20.39 |
| g1 | PW-1 | 0.0934 | 0.0924 | +0.0011 | 0.0015 | 0.975 | 0.524 | 0.476 | 0.006 | 0.477 | — |
| g1 | WML-32 | 0.1161 | 0.1160 | +0.0002 | 0.0016 | 1.000 | 0.472 | 0.528 | 0.235 | 0.564 | 20.29 |

**The repair achieved its stated purpose.** The sampled KL now tracks the true KL:
bias `−0.004` to `+0.001`, RMSE `≤0.004`, correlation **0.975–1.000**. Before the
repair the two terms referred to different action points and the sampled quantity was
not a KL of anything.

**The gate does not suppress the arms unequally on Walker.** Operator-branch fraction
`0.549` (PW-1) vs `0.553` (WML-32), a difference of `−0.004`. Yet Walker shows the
largest difference of the two tasks. **On Walker the operator difference is not
explained by unequal gating.** On g1 there is a modest asymmetry — `0.524` vs `0.472`,
5.2 percentage points, with WML-32 carrying a 1.26× larger analytic KL — so the gate
suppresses the WML operator somewhat more often there. That asymmetry is reported as an
observation; it is not established as the cause of anything.

The registered expectation that the M=32 sampled estimate would be too noisy to
threshold reliably is **not** borne out after the repair: with the log probabilities at
a common point the estimator is accurate enough that sampled and analytic gate
decisions largely agree.

## 11. Frozen same-critic diagnostic (equal-query control)

At final checkpoints, identical states, common random numbers, whitened metric.
Medians over 8 seeds.

| task | arm | ‖PW-1‖ | ‖PW-32‖ | ‖ZO-32‖ | ‖c‖ | ‖v‖ | cos(PW-1,PW-32) | cos(PW-32,ZO-32) | cos(ZO-32,c) | ‖ūbar‖/‖c‖ |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| walker | PW-1 | 0.1076 | 0.0877 | 0.0932 | 0.0145 | 0.4167 | 0.755 | 0.887 | 1.000 | 12.70 |
| walker | WML-32 | 0.7033 | 0.4773 | 0.4904 | 1.2116 | 1.2919 | 0.536 | 0.800 | 0.837 | 0.30 |
| g1 | PW-1 | 0.0083 | 0.0069 | 0.0096 | 0.0170 | 0.9457 | 0.766 | 0.641 | 1.000 | 30.57 |
| g1 | WML-32 | 0.0255 | 0.0168 | 0.0241 | 1.8375 | 2.1472 | 0.766 | 0.659 | 0.832 | 0.39 |

What this separates, and it matters for reading §9:

* **Action-query budget is a large effect.** `cos(PW-1, PW-32) = 0.54–0.77`. The
  one-sample operator that `PW-1` actually trains with points appreciably away from the
  32-sample pathwise direction on the same critic. **`PW-1` is not `PW-32`**, and the
  training comparison in §9 is `PW-1` vs `WML-32`, not an equal-query contrast.
* **Operator difference at equal query is smaller than the budget effect on g1**
  (`cos(PW-32, ZO-32) = 0.64–0.66`) and smaller still on Walker (`0.80–0.89`).
* **The uniform term dominates `v` on pathwise-trained critics** (`‖ūbar‖/‖c‖` = 12.7
  and 30.6) but not on WML-trained ones (`0.30`, `0.39`), consistent with the earlier
  audit and unchanged by this experiment.

## 12. Comparison with the old runs — descriptive only, never pooled

Separate experiments, separate code, separate seed namespaces, separate ledgers. The
old runs are **not** reinterpreted and the two sets are **never** combined.

| task | old (legacy, seeds 101–108) | corrected (seeds 301–308) |
|---|---|---|
| WalkerRun `PW − WML` | median **+2.68**, 5/8, **not detected** | median **+135.58**, 8/8, **detected** |
| g1 `PW − WML` | median **+11.90**, 8/8, detected | median **+9.51**, 8/8, **detected** |
| Walker PW arm | 908.45 | 909.62 |
| Walker WML arm | 899.28 | **760.32** |
| g1 PW arm | 29.56 | 24.74 |
| g1 WML arm | 17.19 | 14.10 |

The g1 contrast is close in magnitude across the two implementations. The Walker
contrast is not: the pathwise arm is essentially unchanged (908 → 910) while the
weighted-MLE arm falls from 899 to 760. **We do not attribute that shift to either of
the two changes, or to any old defect.** Doing so requires a randomized ablation that
separates the same-point repair from the PRNG change, which this design deliberately
does not attempt.

## 13. Limitations

1. **Two changes, confounded by design.** Same-point likelihood and fresh minibatch
   innovations are always on together. No outcome here is attributable to either alone.
2. **The Walker control is no longer a control.** With both tasks detecting, this
   design contains no within-experiment non-detection against which to calibrate.
3. **`PW-1` vs `WML-32` is not an equal-query comparison**, and §11 shows the budget
   effect is large. The training result is an algorithmic operator comparison only.
4. **HumanoidRun absent**, so nothing here speaks to the manuscript's d=21 headline.
5. **No dimension claim is possible or made.** Two tasks, and the larger effect is at
   the *lower* dimension — which is one more reason no trend may be read from this.
6. **Reload verification is a stochastic rollout**, not a bit-level reproduction.
7. `omega` remains unmeasured on every learned critic.
8. Return-level shifts relative to the old study are observations about two different
   implementations, not a causal decomposition.

## 14. Allowed and forbidden wording

**Allowed.** "A task-level operator difference persists under the corrected common
trust-region implementation, on both tasks." "Under the corrected implementation the
Walker comparison, previously a non-detection, becomes a detection." "The repair makes
the sampled KL an accurate estimate of the true KL (correlation 0.975–1.000)." "The
gate suppresses the two arms almost equally on Walker and modestly unequally on g1."
"The one-sample pathwise operator differs appreciably from its 32-sample counterpart."

**Forbidden.** That Claim 4 caused any of this. That any individual old defect caused
the original result. Attribution to the same-point repair or the PRNG change
individually. Any action-dimension trend — note the larger effect is at d=6. Any claim
that learned-critic `omega` was measured. Pooling these runs with the old ones. Calling
either set "clean". Describing `PW-1` and `WML-32` as sample-budget matched.

## 15. Reproduction

```bash
cd ~/repos/reppo
# correctness and parity
JAX_PLATFORMS=cpu ./.venv/bin/python scripts/analysis/test_faithful_repair.py
sbatch --account=rwth2182 --array=0-1 slurm/fr_parity.sh
sbatch --account=rwth2182 --array=0-1 slurm/fr_smoke.sh      # seeds 402-403
# ledger, then the 32 confirmatory runs
./.venv/bin/python scripts/analysis/mk_fr_ledger.py
sbatch --account=rwth2182 --partition=c23g --export=ALL,FR_ARCH=c23g --array=0-15 slurm/fr_launch.sh
sbatch --account=rwth2182 --partition=c25g --export=ALL,FR_ARCH=c25g --array=0-15 slurm/fr_launch.sh
# analysis
./.venv/bin/python scripts/analysis/fr_analyse.py
./.venv/bin/python scripts/analysis/fr_diagnostics.py
sbatch --account=rwth2182 --array=0-31 slurm/fr_samecritic.sh
./.venv/bin/python scripts/analysis/fr_samecritic_aggregate.py
./.venv/bin/python scripts/analysis/fr_figures.py
```

Artifacts: `reports/artifacts/corrected_runs.csv` (32),
`corrected_paired_results.csv`, `corrected_diagnostics.csv`,
`corrected_samecritic.csv`, `fr_samecritic/*.npz` (32);
`reports/figures/fig_corrected_learning_curves.{pdf,png}`,
`fig_corrected_operator_diagnostics.{pdf,png}`.
Ledger: `ledger/runs_faithful_repair.jsonl` (pre-launch, checksummed) and
`ledger/runs.d.faithful_repair/*.json` (post-run). The 32 voided rows from the
launcher-bug attempt are retained under `ledger/runs.d.faithful_repair.void_*/`.
