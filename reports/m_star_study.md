# What sets the E-step's sample requirement

Adjudicated against `docs/prereg_m_star.md` (Tracks A/B, commit `e4a5a0b`) and
`docs/prereg_m_sweep_dmc.md` (Track C, commit `7d1b729`), both committed before any run.

**Every number below is labelled DETERMINED (fixed before the run by prior analysis) or
MEASURED (could have come out otherwise).**

---

## 0. The question

Chatterjee & Diaconis (*Ann. Appl. Prob.* 28(2):1099-1135, 2018) prove that
self-normalised importance sampling needs `M ~ exp(D(target||proposal))`, necessary and
sufficient. In the E-step that KL is pinned at `eps_E` by the dual, giving
`M* ~ exp(eps_E)` -- about 1.6 at `eps_E = 0.5`, and **independent of d**. The LQR sweep
disagreed. Both cannot be right.

Candidate resolution, registered as a hypothesis: CD governs a **scalar** expectation;
`g_ZO` and `d_ESTEP` are **d-vectors**. The KL budget controls the weights and says
nothing about resolving `d` directions.

---

## 1. Determined in advance -- not findings

- **DETERMINED.** E0a measured `(M-1) MSE_ZO/||g*||^2 = d+1` to three digits over seven
  dimensions, so for centred ZO `M*-1 ∝ d+1` analytically. The measured ZO log-log slope
  is **1.14** at `tau = 0.95`; this confirms arithmetic, it does not discover anything.
- **DETERMINED (registered in prereg Sec. 2 before running).** Since
  `d_ESTEP ≈ ubar + a_hat_ZO/eta` and `ubar` carries energy `d/M`, `b ≈ 1` was expected
  for ESTEP too. The measured ESTEP slope is **1.19**. Also not a discovery.
- The open quantities were therefore the **ratio** `M*_ESTEP/M*_ZO` (Q1), the **`eps_E`
  dependence** (Q2), and **ESS blindness** (Q3).

---

## 2. Track A -- the M* surface

`M*(tau; d, eps_E)` = smallest `M` whose mean cosine to the exact estimand reaches `tau`,
interpolated on `log M`. Headline `tau = 0.95` (registered), all three reported.
32 states x 48 replicates per cell; one draw of `u` per cell serves all three arms and all
six `eps_E`, so the arms are **paired on identical samples**.

### 2.1 M* at tau = 0.95, low-frequency regime (c = 8.15)

| d | PW | ZO | eps=0.05 | 0.1 | **0.5** | 1.0 | 2.0 | 4.0 |
|---|---|---|---|---|---|---|---|---|
| 2 | 11.0 | 40.6 | 625.5 | 318.3 | **70.7** | 304.1 | >2048 | >2048 |
| 4 | 9.5 | 65.8 | 759.0 | 411.5 | **114.7** | 106.6 | 315.1 | >2048 |
| 8 | 14.6 | 156.3 | 1830.7 | 1006.9 | **313.1** | 385.2 | >2048 | >2048 |
| 16 | 21.0 | 350.5 | >2048 | 1819.4 | **628.5** | 630.8 | 1206.2 | >2048 |
| **21** | 20.0 | 444.9 | >2048 | >2048 | **954.7** | 1018.0 | >2048 | >2048 |
| 32 | 40.5 | 969.5 | >2048 | >2048 | **1946.8** | >2048 | >2048 | >2048 |
| 64 | 57.3 | >2048 | >2048 | >2048 | >2048 | >2048 | >2048 | >2048 |

22 of 42 ESTEP cells right-censored at `tau=0.95`; 14 of 42 at `tau=0.90`; 41 of 42 at
`tau=0.99`. Censored cells are excluded from the fit, as registered.

**MEASURED.** At `d = 21, eps_E = 0.5` the E-step needs `M* ≈ 955` to reach cosine 0.95.
Production runs `M = 32`.

### 2.2 The registered fit, and why it is void

    log M* = a*eps_E + b*log d + c        (unweighted OLS, censored excluded)
    low  regime: a = -0.4221,  b = +0.7672   (n=20, R2=0.537, resid sd 0.645)
    high regime: a = -0.4387,  b = +0.7793   (n=20, R2=0.552, resid sd 0.637)

**The registered non-monotonicity guard FIRED** (prereg Sec. 3): `M*(eps_E)` has an
**interior minimum at 4 of 7 dimensions**, at `eps_E = 0.5` or `1.0`, with Spearman
`rho(M*, eps_E)` of -0.80, -0.70, -0.80, -0.20 at d = 2, 4, 8, 16.

Per the prereg, `a` is therefore **reported but not interpretable**, adjudication is void,
and the descriptive branch is taken. This is exactly the cancellation failure the prereg
was written to catch: `a = -0.42` sits outside the CD null band, but a linear coefficient
fitted across a U-shaped curve does not mean what the decision rule assumed.

### 2.3 What the eps_E curve actually shows (Q2)

Using `tau = 0.90`, where censoring is lightest, at `d = 4`:

| eps_E | 0.05 | 0.1 | 0.5 | 1.0 | 2.0 | 4.0 |
|---|---|---|---|---|---|---|
| M* | 322.9 | 180.1 | 50.3 | **37.9** | 50.7 | 206.3 |

- **Falling branch (eps_E 0.05 -> 1.0): M\* FALLS 8.5x. CD predicts a RISE of
  `exp(0.95) = 2.6x`. CD is wrong in SIGN here.** MEASURED.
- Rising branch (1.0 -> 4.0): M* rises 5.4x; CD predicts `exp(3) = 20.1x`. Same direction,
  **CD overpredicts by ~4x**. MEASURED.
- The minimum sits at `eps_E ≈ 0.5-1.0`. Production uses 0.5. MEASURED, and it means the
  shipped `eps_E` is close to sample-optimal -- which was not anticipated.

**The registered mechanistic prediction was CONFIRMED**: as `eps_E -> 0`, `eta -> inf`,
the signal term `a_hat/eta` shrinks while `ubar` stays pure noise, so `M*` rises. This was
written down before the run and could have failed; had `M*` fallen monotonically with
`eps_E`, the mechanism would have been refuted.

**Verdict on CD: its prediction does not describe this estimator.** But see Sec. 5 --
whether that *contradicts the theorem* is a separate question, and the answer is no.

### 2.4 Q1 -- ESTEP versus centred ZO

`M*_ESTEP / M*_ZO` at `eps_E = 0.5`, `tau = 0.95`:

| d | 2 | 4 | 8 | 16 | 21 | 32 |
|---|---|---|---|---|---|---|
| ratio | 1.76 | 1.74 | 2.00 | 1.79 | 2.14 | 2.01 |

**MEASURED: the softmax E-step needs about 2x the samples of the centred estimator, and
that factor does NOT grow with d** (1.76 -> 2.01 across a 16x range of `d`). So the
E-step's extra cost over the estimator the theory is stated about is a constant factor,
not a dimensional penalty. Q1's "does the difference grow with d" is answered **no**.

### 2.5 The frequency axis

ZO and ESTEP `M*` are **identical** in the two regimes (e.g. ZO at `d=21`: 444.9 low vs
444.8 high; ESTEP at `eps=0.5`: 954.7 vs 954.0). Pathwise `M*` moves from 20.0 to
right-censored. MEASURED: **the zeroth-order arms' sample requirement is independent of the
error frequency; pathwise's is entirely driven by it.**

---

## 3. Track B -- ESS is blind

### 3.1 The concrete case

At `d = 21, M = 32, eps_E = 0.5` -- the committed HumanoidRun arm-B configuration:

| quantity | value |
|---|---|
| cos(ESTEP, g*) | **0.523** |
| cos(PW, g*) | 0.969 |
| ESS | 15.46 / 32 (**ESS/M = 0.483**) |
| KL(w \|\| uniform) | **0.4898** (budget `eps_E` = 0.5) |

**MEASURED. The dual is hitting its KL budget to two decimals and ESS/M sits at a
healthy-looking one half, while the estimator is at cosine 0.52 to the truth.** The
published cohort observed `ESS ∈ [18.8, 19.4]` at this cell (`meta.json:ess_final`), which
is the same picture. Nothing in the ESS family was capable of flagging this.

### 3.2 Correlations with the deficit (registered target: cos_PW - cos_ESTEP)

Low regime, 420 cells, deficit spanning a factor of 92:

| diagnostic | Pearson | Spearman | range in worst-deficit decile |
|---|---|---|---|
| ESS | -0.399 | -0.605 | [1.00, 233.8] |
| **ESS/M** | **+0.159** | **+0.114** | [0.014, 0.921] (64x) |
| ESS/d | -0.348 | -0.773 | [0.016, 3.65] |
| **KL(w\|\|uniform)** | **-0.056** | **-0.120** | [0.045, 3.996] |
| logit spread | +0.238 | -0.117 | [0.298, 6730] |

**MEASURED: `ESS/M` and `KL(w||uniform)` -- the two diagnostics the algorithm actually
logs -- are essentially uncorrelated with the deficit** (|Spearman| ≤ 0.12), and within the
worst-deficit decile `ESS/M` still ranges over a factor of 64. They are blind.

**This supports the reading that Section 7.6 could observe `ESS ∈ [18.8, 19.4]` at d=21 and
correctly conclude the arm was not misconfigured -- while the arm was nonetheless losing.**
Those are compatible, not contradictory: the configuration was fine and the sample count
was not.

### 3.3 The registered out-of-sample test, and a flaw in it

The registered test (fit on low regime, evaluate on high) **fails for every diagnostic**,
including the registered candidate `ESS/d` (out-of-sample R2 between -4.2 and -4.9).

**That test is ill-posed, and I am reporting it as registered before saying so.** The
target `cos_PW - cos_ESTEP` contains pathwise's frequency sensitivity, which flips the
deficit negative in the high regime (range [-0.604, +0.354]). No diagnostic of the
E-step's own health can predict a quantity that depends on the *other* operator's response
to `omega`. The registered test therefore cannot be passed by any E-step diagnostic.

Re-run against the E-step's own error `1 - cos_ESTEP`, with the axis that actually moves
the target held out (this is **not** the registered test, and is labelled as such):

| diagnostic | leave-one-d-out mean R2 | leave-one-eps-out mean R2 |
|---|---|---|
| ESS | -0.225 | +0.225 |
| ESS/M | -0.748 | -0.196 |
| **ESS/d** | **+0.343** | **+0.461** |
| KL | -0.711 | -0.110 |
| logit spread | -0.758 | -0.156 |

**MEASURED: `ESS/d` is the only diagnostic with positive out-of-sample R2 under either
hold-out.** It is not a strong predictor -- it explains a third to a half of the variance
and degrades at the edges of the `d` range (R2 = -0.09 at d=2, +0.27 at d=64) -- but it is
the only candidate that transfers at all, and it was registered before being tested.

**Honest summary: `ESS/d` failed the test I registered and passed two better-posed tests I
did not register.** Both are reported. A reader who wants only the pre-registered answer
should read: no diagnostic passed.

---

## 4. Track C -- the DMC arm

Pre-registered at `7d1b729` before launch. **Status: running.** 16 runs queued --
`M = 128` seeds 201-208 on GPU 1, `M = 512` seeds 211-218 on GPU 0. The control (arm A,
pathwise frozen-alpha, n=9, mean 738.614) and the `M = 32` arm (n=8, mean 666.174,
gap **+72.440**, pooled t = 2.816) are NOT re-run.

Seed blocks are **disjoint across M**. The export tag `HumanoidRun_weighted_mle_s{seed}`
does not encode `M`, so a shared seed would silently overwrite. This was registered as a
hazard in prereg Sec. 3.1 and then confirmed live: the two smoke runs both used seed 299
and overwrote one another. The cost of disjoint blocks is that the two M arms are not
seed-paired with each other; both are compared against the unchanged arm A, so the
primary contrast is unaffected.

### 4.1 Smoke test (passed)

One iteration at each M: both compile and run, **no OOM at `M = 512`** on the 32 GB card,
`estep_num_samples` correctly echoed into `meta.json`. ESS at initialisation is 121.7/128
and 484.7/512 (`ESS/M ≈ 0.95`) -- expected with an untrained critic and nearly flat `Q`,
and **not** representative of the converged value (~0.59 in the published cohort).

### 4.2 Compute asymmetry -- REGISTERED AS A REPORTED OUTCOME

Measured here, from 1- and 4-iteration runs (marginal cost = `(t4 - t1)/3`), against the
published `M = 32` anchor of 4400 s / 399 iterations:

| M | s / iteration | full 399-iter run | critic rows per minibatch | vs pathwise |
|---|---|---|---|---|
| 32 (committed) | 11.0 | 73 min | 65,536 | 32x |
| 128 | **19.4** | ~2.2 h / seed | 262,144 | 128x |
| 512 | **32.6** | ~3.6 h / seed | 1,048,576 | **512x** |

**MEASURED, and the useful part: wall-clock scales far SUB-LINEARLY in M.** A 16x increase
in `M` (32 -> 512) costs only **3.0x** wall-clock, because the E-step critic evaluation is
one term among environment stepping, critic training and the M-step. The 512x figure is
critic rows, not time.

That materially changes the price of the fix. If the d=21 deficit is finite-sample, buying
`M = 512` costs roughly 3x wall-clock per seed, not 16x. A win at `M = 512` is still
reported with its price -- but the price is smaller than the critic-query ratio suggests.

## 5. Does this contradict Chatterjee & Diaconis?

**No, and the report should not be read as claiming so.**

CD bound the sample size needed for a self-normalised IS estimate of a **scalar**
expectation to concentrate. `d_ESTEP` is a **d-vector**, and `M*` here is defined by a
**directional** criterion (cosine), which requires every one of `d` components to be
resolved simultaneously. A `d`-fold requirement is exactly what one expects when `d`
independent directions must each be estimated, and it does not violate a theorem about one
scalar functional.

**Does the data support that explanation, or is it merely consistent with it?** Honestly:
**consistent with, and only weakly supportive.** What the data shows is that `M*` scales
about linearly in `d` and that `eps_E` does not govern it. That is compatible with the
scalar-vs-vector reading, but a `b ≈ 1` measurement alone does not establish the mechanism
-- and `b ≈ 1` was substantially DETERMINED in advance (Sec. 1), so it is close to no
evidence at all about *why*.

A test that would actually discriminate, and which was **not run**: measure `M*` for a
scalar functional of the same E-step weights -- e.g. the weighted mean of a single fixed
linear projection `v^T u` -- at several `d`. If CD is right about the scalar case, that
`M*` should be `d`-independent and near `exp(eps_E)`, while the vector `M*` grows. That
comparison, on the same samples, would separate "vectors need more" from "the E-step is
just inefficient". It is the obvious next experiment.

---

## 6. What I did not test

- **The scalar control just described.** Without it, the scalar-vs-vector explanation is
  a plausible reading, not a demonstrated mechanism.
- **Whether cosine is the right criterion.** `M*` is defined by a directional threshold.
  A trust-region update that normalises the step may care about direction only, but that
  is an assumption, not something measured here.
- **Anything outside the LQR.** Planted rank-one error, quadratic `Q^pi`, a single
  `(sigma, omega)` pair per regime, one system per `d`. No system-to-system variability was
  sampled.
- **`tau = 0.99` at all.** 41 of 42 cells censor, so the surface is unmeasured there.
- **`d = 64` at `eps_E = 0.5`**, which censors at `tau = 0.95`; the `M*` there exceeds 2048
  and is unknown.
- **The KL confound in Track C** (`estep_num_samples` also sets the KL sample count), by
  registered choice.
- **Return.** Everything in Tracks A and B is cosine to an estimand. Cosine is not return,
  and only Track C speaks to return.
