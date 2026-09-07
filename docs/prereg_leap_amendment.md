# Amendment — LEAP entropy factorial

Amends `docs/prereg_g1_entropy_factorial.md` (commit `aa5533d`) for the LEAP arms.
Committed and pushed **before** any LEAP factorial run was launched. Append-only; the
amended document itself is not edited.

## 1. No directional prediction is preregistered

```
LEAP_PW_DIRECTIONAL_PREDICTION  = NONE
LEAP_WML_DIRECTIONAL_PREDICTION = NONE
```

**Why.** The entropy-force scalar `F_H = alpha * [1 - 2*sigma^2*E(sech^2 y)]` was measured
on existing checkpoints in `reports/entropy_force_measurement.md` (commit `56e6c8b`) and
put through a prespecified validation gate on four known outcomes. It scored **3/4** on the
measurement designated primary, missing Walker `WML_noent`.

The local entropy-force scalar was therefore **not validated as a predictor of final
trained matched-state scale**. Two alternative variants do reach 4/4 — neutral states with
the median, and own-arm states with the mean — but both would be selected after seeing
which one validates, and the four defensible variants give three mutually inconsistent
LEAP predictions, with the `WML_ent` direction flipping between state populations and the
`PW_noent` direction flipping between the median and the mean. Preregistering any of them
would be choosing a variant rather than measuring a mechanism.

No directional entropy prediction will be scored when LEAP completes, because none was
registered.

## 2. Terminology, fixed

**PRIMARY mechanistic outcome** — matched-state common-bank median pre-tanh sigma: both
policies evaluated on **identical frozen states** from the canonical LEAP bank. This is a
property of the learned state-conditional scale function over that state population. It is
**not** the experienced exploration scale of either policy.

**SECONDARY** — the two bank halves reported separately; `sat95`; `sat99`; and own-state
quantities **only where genuine own states exist**.

Genuine own states exist for `PW_ent` and `WML_noent`, whose rollouts built the bank. They
do **not** exist for the new cells `PW_noent` and `WML_ent`. Any figure for those two cells
computed on same-arm baseline states is labelled **PROXY** and is never called on-policy,
on-arm, or own-distribution.

## 3. No new state-generation pipeline

A LEAP 8x8 policy-seed by state-seed matrix for the **new** cells would require genuine
own final-policy seed-specific states, generated under a frozen deterministic rule. No such
rule exists today, and the existing bank generators are not deterministic: both key the
PRNG on `hash((arm, seed))`, which Python salts per process, so they cannot regenerate a
bank byte-identically.

The 8x8 matrices being computed under `docs/protocol_seed_matrix.md` therefore cover the
two **baseline** cells only, for which the bank already holds per-seed states. **No new
state-generation pipeline is built for completeness**, and no 8x8 is preregistered for the
new LEAP cells.

## 4. Expected return outcome, recorded in advance

LEAP's baseline return gap is `+4.30`, 95% CI `[-1.34, +10.53]`, an interval containing
zero. The return half of this factorial is therefore **expected to be uninformative**, and
the geometry half is the purpose. This is recorded now so that an inconclusive return
result is not later presented as a finding in either direction.

## 5. Caveat carried forward unchanged

Alpha is **PW-calibrated by construction**. LEAP's frozen `ent_start` is
`7.82382907e-04`, fixed from the pathwise arm's runs, and the weighted-MLE arm has never
carried an entropy term. `WML_ent` therefore tests matched entropy **geometry**, not that
operator's own optimal temperature.

## 6. Reproducibility caveat

Exact same-seed trajectory reproducibility is absent on contact-rich tasks; it was measured
on G1 and LEAP is of the same class. Seed pairing may therefore provide less
common-random-number variance reduction than under deterministic execution. Nothing frozen
is reinterpreted on that basis, and the end-to-end bitwise no-op test is replaced by the
fixed-batch test `T1b` accordingly.

## 7. Everything else unchanged

The frozen primary return statistic from commit `7edb8e8` — mean of the final three of 21
logged evaluations, contrast PW minus WML paired within seed, paired percentile bootstrap
of the paired median over the 8 seed differences, 10,000 resamples,
`np.random.default_rng(20260902)`, 95 percent interval — is not redefined. Seeds are
301-308. `BASELINES_RERUN = NO`.
