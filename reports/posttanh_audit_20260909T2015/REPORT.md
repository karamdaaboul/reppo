# Post-tanh policy-geometry audit — REPORT

Offline checkpoint analysis. No training launched, no training code altered, no thresholds
tuned, no tasks added, **no returns inspected**.

## A. Completeness and validity audit

### A.1 Repository identity

```
path            /hpcwork/qzi10910/estep_wt
branch          estep-study
HEAD            3be8c1117154449e05f0a1279b4a82bff28e9dce
git status      tracked tree CLEAN (38 untracked analysis/figure artifacts, none discarded)
src/jaxrl/reppo.py   sha256 2c4ff079d2a8f357c6f096cb0e416334a85f1a3deeed085b03d37856a810366b
scripts/load_ckpt.py sha256 1bab1d9e9ae1bec6b5949e2d57f2009930e705e9d6c6df1854044318db572925
```

**Matches the previously audited reference exactly** (branch, HEAD and `reppo.py` hash all
identical). Nothing was reset or discarded.

Actor implementation vs the code that generated each arm: the deployed inference path is
`scripts/load_ckpt.py::load(...).policy_dist(s)`, which reconstructs the actor from the
stored graphdef and parameters. It is the same function the existing pre-tanh analyses use,
and validation gate V6 reproduces the three published Walker pre-tanh medians to 4 decimal
places (7.3096, 0.9062, 0.3375), confirming compatibility with every arm's checkpoints.

### A.2 Frozen protocol

| file | sha256 |
|---|---|
| `PROTOCOL.md` (authoritative) | `20d7ad43378d2966eba74470435a083efa27ee08312343f1f1ea9d0434f2b04d` |
| `PROTOCOL.md.orig` (first draft) | `ed79808a922f505d20277fada0fbdec1aaa27241204d6ae1d62e8925a0a296db` |
| `AMENDMENT_1.md` | `9a595d95b1362805d5846fd13067b88b297a2d8e63a434572ee1d6eda3683b9d` |

Frozen 2026-09-09T23:06:16+02:00 (`FROZEN_AT.txt`), **before any post-tanh value was
computed**. Two changes, both made before any outcome value existed and both preserved:

1. **Timestamp correction** — the first draft carried a placeholder freeze time. One line
   changed; original preserved.
2. **AMENDMENT 1, quadrature scheme** — unavoidable. Validation gate V3 failed and the
   failure was real: Gauss-Hermite does not converge for `tanh` moments at large `sigma`.
   At `mu = -4, sigma = 10` it wanders (`v` = 0.787 / 0.884 / 0.849 / 0.805 at orders
   32/64/96/128) and `numpy.hermgauss` returns non-finite weights above `n ~ 200`, so the
   order cannot simply be raised. **This is exactly the regime the data occupies** — the
   canonical WML arm on Walker has median pre-tanh sigma 7.31. Replaced by two-panel
   Gauss-Legendre split at the tanh transition, which converges to `1.3e-14` by `N = 128`
   per panel. Reported `N = 128`, sensitivity `N = 64`, reference `N = 512`.
   No change to run selection, populations, contrasts, metric definitions, aggregation,
   bootstrap, missing-data policy or interpretation gates.

**No equivalence margin was frozen. Conclusion 3 is therefore unavailable by construction**,
and an interval containing 1 is reported as "no detected difference", never as equivalence.

### A.3 Run manifest — 72/72 cells complete

All three tasks x three arms x eight seeds present with `actor.npz`, `critic.npz`,
`normalizer.npz`, `meta.json`. **Matched seed intersection = 8 for every task**
(301-308); no cell was dropped, no run substituted.

Effective configuration verified per cell, with fields absent from stored configs resolved to
the code defaults in `src/jaxrl/reppo.py` and reported as effective values:

| task | arm | algo | eff. eps_e | steps | iter | frac | evals | M | obs | act | hid | min_std | norm | norm_eps |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| walker | PW | pathwise | 0.5 (unused on this path) | 52297728 | 399 | 1.0 | 20 | 32 | 24 | 6 | 512 | 0.1 | True | 0.01 |
| walker | WML05 | weighted_mle | 0.5 | 52297728 | 399 | 1.0 | 20 | 32 | 24 | 6 | 512 | 0.1 | True | 0.01 |
| walker | WML01 | weighted_mle | **0.1** | 52297728 | 399 | 1.0 | 20 | 32 | 24 | 6 | 512 | 0.1 | True | 0.01 |
| g1 | PW / WML05 / WML01 | pathwise / weighted_mle | 0.5 / 0.5 / **0.1** | 52297728 | 399 | 1.0 | 20 | 32 | 103 | 29 | 512 | 0.1 | True | 0.01 |
| leap | PW / WML05 / WML01 | pathwise / weighted_mle | 0.5 / 0.5 / **0.1** | 52297728 | 399 | 1.0 | 20 | 32 | 32 | 16 | 512 | 0.1 | True | 0.01 |

No field varied across seeds within any task x arm cell. All checkpoints are `_final`
(`checkpoint_frac = 1.0`, `iteration = 399`) and all arms share an identical training budget.
`eps_e` is stored as 0.5 for PW but is never read on the pathwise path.

**Observation normalization**: each policy carries its own normalizer and all 8 seeds within
every cell have distinct normalizer hashes. The primary analysis uses each policy's **native**
normalizer, applied internally by `load_ckpt.py:129`, because that defines the deployed
policy. No common-normalizer variant was invented.

### A.4 Data integrity

| bank | sha256 (16) | matches registered |
|---|---|---|
| walker neutral | `8adfeb0bf70bddcd` | **yes** |
| g1 neutral | `cf6f7880b3a5b594` | **yes** |
| leap neutral | `0053b0f361e45b92` | **yes** |

`*_onpolicy_bank_newarms` and `*_onpolicy_bank_eps01` present for all three tasks.
**`BASELINES_UNCHANGED = YES`** — all 48 pre-`eps01` baseline hashes still match, so the PW
baselines and stored banks were not modified by the `eps_E` experiment.

### A.5 Validation gates — ALL PASS

| gate | result |
|---|---|
| V1 `sigma -> 0` gives `v -> 0` | PASS (max v 1e-6 / 1e-8 / 1e-10 at sigma 1e-3 / 1e-4 / 1e-5) |
| V2 `mu = 0` gives zero squashed mean | PASS (`<= 5.6e-17` at every sigma) |
| V3 N=128 vs N=512 reference | PASS (max abs `2.4e-14`; **0 cells flagged** under the frozen conjunctive criterion) |
| V4 analytic saturation vs independent Gaussian CDF | PASS (`2.2e-16`) |
| V5 decomposition reconstructs total | PASS (`0.0` for variance and saturation) |
| V6 actor output vs published pre-tanh sigma | PASS (7.3096 / 0.9062 / 0.3375, exact to 4 dp) |
| V7 byte-identical raw states across arms | PASS (`0030172b21357237`) |
| V8 finite, correct shape, both quadratures | PASS |

Full-run numerics: **0 clamped negative variances** across all 216 seed-level aggregates,
**0/216 flagged quadrature cells**, **max decomposition residual `1.1e-16`**.

No independent code-review agent was run for this audit; the failures that were found were
found by the validation gates themselves and are documented above.

---

## B. Numerical results

Primary population = neutral bank. Estimator = mean of per-seed log ratios, exponentiated.
Paired seed-level bootstrap, `rng(20260910)`, 100000 resamples, 95% percentile. n = 8 for
every cell.

### B.1 Post-tanh action variance, primary contrast and secondaries

| task | contrast | ratio | 95% CI | excludes 1 |
|---|---|---|---|---|
| **walker** | **WML0.5 / PW** | **15.003** | **[13.034, 17.390]** | **yes** |
| walker | WML0.1 / WML0.5 | 0.486 | [0.427, 0.562] | yes |
| walker | WML0.1 / PW | 7.291 | [5.971, 9.109] | yes |
| **g1** | **WML0.5 / PW** | **5.108** | **[4.406, 6.129]** | **yes** |
| g1 | WML0.1 / WML0.5 | 0.314 | [0.259, 0.387] | yes |
| g1 | WML0.1 / PW | 1.606 | [1.309, 1.961] | yes |
| **leap** | **WML0.5 / PW** | **2.713** | **[2.082, 3.723]** | **yes** |
| leap | WML0.1 / WML0.5 | 1.002 | [0.861, 1.141] | **no** |
| leap | WML0.1 / PW | 2.718 | [2.164, 3.833] | yes |

### B.2 All individual seed values (301-308), neutral bank

```
walker WML05/PW     13.354 21.944 13.442 17.122 16.842 10.822 16.195 12.895
walker WML01/WML05   0.435  0.532  0.519  0.717  0.427  0.559  0.407  0.372
walker WML01/PW      5.807 11.668  6.982 12.283  7.187  6.048  6.589  4.799
g1     WML05/PW      5.039  5.289  4.093  5.356  4.499  8.796  3.781  5.304
g1     WML01/WML05   0.508  0.369  0.273  0.261  0.457  0.199  0.268  0.294
g1     WML01/PW      2.558  1.949  1.116  1.399  2.056  1.750  1.013  1.557
leap   WML05/PW      1.979  2.393  1.635  3.106  3.973  2.127  2.215  6.516
leap   WML01/WML05   1.074  0.826  1.201  0.918  0.661  1.156  1.072  1.267
leap   WML01/PW      2.126  1.977  1.963  2.852  2.627  2.459  2.374  8.254
```

Every Walker and G1 seed for the primary contrast lies above 1; the smallest is 3.78 (g1,
seed 307). Every LEAP seed also lies above 1, smallest 1.635.

### B.3 Saturation, P(|a| > 0.95), paired differences

| task | contrast | difference | 95% CI | excludes 0 |
|---|---|---|---|---|
| walker | WML0.5 - PW | **+0.2256** | [+0.1877, +0.2598] | yes |
| walker | WML0.1 - WML0.5 | **-0.2012** | [-0.2300, -0.1695] | yes |
| walker | WML0.1 - PW | +0.0243 | [-0.0141, +0.0796] | no |
| g1 | WML0.5 - PW | -0.0249 | [-0.0561, +0.0114] | no |
| g1 | WML0.1 - WML0.5 | -0.0496 | [-0.1039, +0.0085] | no |
| g1 | WML0.1 - PW | -0.0745 | [-0.1269, -0.0173] | yes |
| leap | WML0.5 - PW | +0.0346 | [-0.0487, +0.1239] | no |
| leap | WML0.1 - WML0.5 | +0.0531 | [-0.0133, +0.1101] | no |
| leap | WML0.1 - PW | +0.0878 | [+0.0039, +0.1609] | yes |

### B.3b Does tightening eps_E close the behavioural WML-PW gap?

The decision-relevant framing. "Narrows" is the ratio-of-ratios `WML0.1/WML0.5`, which is
algebraically `(WML0.1/PW)/(WML0.5/PW)` and telescopes exactly (verified to `2.2e-16`).
"Closes" requires the `WML0.1/PW` interval to contain 1.

| task | gap at eps_E=0.5 | gap at eps_E=0.1 | ratio-of-ratios | narrows? | closes? |
|---|---|---|---|---|---|
| walker | 15.003 [13.034, 17.390] | **7.291 [5.971, 9.109]** | 0.486 [0.427, 0.562] | **yes** | **no** |
| g1 | 5.108 [4.406, 6.129] | **1.606 [1.309, 1.961]** | 0.314 [0.259, 0.387] | **yes** | **no** |
| leap | 2.713 [2.082, 3.723] | **2.718 [2.164, 3.833]** | 1.002 [0.861, 1.141] | **no** | **no** |

Per-seed `WML0.1/PW`, all eight above 1 on every task:

```
walker   5.807 11.668  6.982 12.283  7.187  6.048  6.589  4.799   min 4.799
g1       2.558  1.949  1.116  1.399  2.056  1.750  1.013  1.557   min 1.013
leap     2.126  1.977  1.963  2.852  2.627  2.459  2.374  8.254   min 1.963
```

**Tightening eps_E narrows the behavioural gap on Walker and G1, does not move it on LEAP,
and closes it on no task.** G1 comes closest: its lower bound is 1.309 and one seed sits at
1.013.

**This diverges from the pre-tanh picture on G1.** The earlier pre-tanh comparison found the
G1 operator gap closed at `eps_E = 0.1` (median pre-tanh sigma ratio 0.965, interval
containing 1), whereas the post-tanh action variance ratio is 1.606 with an interval
excluding 1. The two use different statistics - median of `sigma` versus mean of `v` - so
they are not one-to-one, but the qualitative conclusions differ: **squashing does not preserve
that closure.**

### B.4 Secondary populations — the primary contrast is stable across all three banks

| task | neutral | PW rollout | canonical-WML rollout |
|---|---|---|---|
| walker | 15.003 [13.034, 17.390] | 22.440 [18.197, 26.282] | 12.952 [11.475, 14.639] |
| g1 | 5.108 [4.406, 6.129] | 5.799 [5.178, 6.722] | 5.262 [4.460, 6.364] |
| leap | 2.713 [2.082, 3.723] | 3.254 [2.545, 4.323] | 2.415 [1.809, 3.385] |

All nine intervals exclude 1. The conclusion does not depend on which of the three existing
populations is used.

### B.5 Mean-versus-scale decomposition (output-level, common states)

Symmetric two-factor decomposition of the post-tanh **variance** difference, mean over seeds.
Identity `total = C_sigma + C_mu` holds to `1.1e-16` in every cell.

| task | pair | total | C_sigma | C_mu | scale share |
|---|---|---|---|---|---|
| walker | WML0.5 vs PW | +0.5723 | +0.5731 | -0.0008 | **100.1%** |
| walker | WML0.1 vs WML0.5 | -0.3099 | -0.3372 | +0.0273 | 108.8% |
| walker | WML0.1 vs PW | +0.2624 | +0.2460 | +0.0163 | 93.8% |
| g1 | WML0.5 vs PW | +0.1179 | +0.1079 | +0.0100 | **91.5%** |
| g1 | WML0.1 vs WML0.5 | -0.1003 | -0.1065 | +0.0062 | 106.2% |
| g1 | WML0.1 vs PW | +0.0176 | +0.0103 | +0.0073 | 58.3% |
| leap | WML0.5 vs PW | +0.3122 | +0.2580 | +0.0542 | **82.6%** |
| leap | WML0.1 vs WML0.5 | +0.0083 | +0.0177 | -0.0093 | 212.0% |
| leap | WML0.1 vs PW | +0.3206 | +0.2854 | +0.0352 | 89.0% |

For **saturation** the decomposition is far less stable — shares of -138%, +353%, +372%
appear, because the totals are small and `C_sigma` and `C_mu` partially cancel. Those shares
are reported in `decomposition.csv` but are not interpretable as a split; only the variance
decomposition supports a reading.

This is an **output-level decomposition on common states, not a causal training
decomposition.**

### B.6 Quadrature agreement and warnings

`N = 64` vs `N = 128` per panel across all 216 seed-level aggregates: **0 cells flagged**
under the frozen conjunctive criterion (rel > 1e-3 AND abs > 1e-8). **0 negative variances
clamped.** No missing data: 72/72 cells, n = 8 everywhere. No NaN, no abnormal termination.

**Supplementary check (post hoc, changes nothing):** `|mu/sigma| > 15` occurs in up to 13% of
elements, which collapses one integration panel. Verified benign - in that regime `tanh` is
saturated across the whole range so a single panel is exact; N=128 vs N=2048 agrees to
`1.6e-13`, and an independent 2e7-sample Monte Carlo at the most extreme real cell
(`mu/sigma = 131.7`) agrees to `6.6e-15`. Truncation mass at T=15 is `7.3e-51`. See
`NUMERICS_NOTE.md`.

---

## C. Conclusions, one per task

Under the frozen interpretation gates:

* **Walker — "The pre-tanh difference survives as a post-tanh action-distribution
  difference."** WML0.5/PW = 15.00x [13.03, 17.39], excludes 1, all 8 seeds above 1.
* **G1 — "The pre-tanh difference survives as a post-tanh action-distribution
  difference."** 5.11x [4.41, 6.13], excludes 1, all 8 seeds above 1.
* **LEAP — "The pre-tanh difference survives as a post-tanh action-distribution
  difference."** 2.71x [2.08, 3.72], excludes 1, all 8 seeds above 1.

Conclusion 3 was unavailable by construction: no equivalence margin was frozen.

## D. Required statements

**Does a behavioural-geometry claim survive?** Yes. The difference is present in the bounded
action distribution actually emitted after `a = tanh(z)`, on all three tasks, on all three
existing state populations, with every individual seed on the same side.

**Does only a latent pre-squash parameterization claim survive?** No — the claim is not
confined to the latent parameterization. The pre-tanh scale difference is not absorbed by the
squashing function.

**Does the `eps_E` intervention change the bounded action distribution?**
Walker **yes** (0.486x [0.427, 0.562]); G1 **yes** (0.314x [0.259, 0.387]); LEAP **no detected
difference** (1.002x [0.861, 1.141] — the interval contains 1, which is not evidence of
equivalence and no equivalence margin was frozen).

**Mean location, scale, or both?** Overwhelmingly **scale**. For the primary contrast the
scale factor accounts for 100.1% (walker), 91.5% (g1) and 82.6% (leap) of the post-tanh
variance difference, with the mean factor contributing little and, on walker, marginally
negative. The saturation decomposition does not support a stable split and is not read.

**Prohibited framings, explicitly not claimed.** These results are not evidence about
exploration, uncertainty, calibration, performance benefit or return causality. Returns were
not inspected. Three tasks are reported separately and no population of environments is
inferred from them.
