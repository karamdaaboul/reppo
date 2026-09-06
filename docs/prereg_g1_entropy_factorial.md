# Preregistration — G1 entropy factorial: stated prediction

Recorded before any G1 entropy-factorial run exists. Drafted 2026-09-06 while the CLAIX
login node was unreachable; committed at the first opportunity thereafter. No G1 run of
either new cell had been launched at drafting time or at commit time.

Append-only. Once committed this text is not edited.

## 1. Stated prediction, fixed before results

Quoted verbatim as given by the principal:

> We expect the entropy intervention to have a smaller effect on G1 than on
> Walker, but treat this as an empirical prediction rather than a theorem.
>
> The motivation is:
>
> (a) G1 uses frozen alpha ≈ 2.08e-4, approximately 70x smaller than Walker's
>     alpha ≈ 1.45e-2, so the absolute actor-entropy gradient is expected to be
>     substantially weaker, all else equal.
>
> (b) The previously frozen G1 analysis found a much smaller baseline
>     matched-state scale separation:
>         WML/PW ≈ 1.81x
>     compared with Walker ≈ 16.05x.
>
> Do NOT infer the sign of the entropy force from median sigma alone.
>
> For
>     F_H = alpha [1 - 2 sigma^2 E(sech^2(y))],
>
> sigma < 1/sqrt(2) is sufficient to guarantee widening pressure, but
> sigma > 1/sqrt(2) does not by itself guarantee contraction because the
> expectation depends on mu and the state distribution.
>
> Therefore the preregistered scientific prediction is only:
>
>     entropy will modify G1 less strongly than Walker in magnitude.
>
> Whether WML+H narrows, whether PW-H narrows, and whether the operator gap
> persists are empirical outcomes.

## 2. What is and is not predicted

**Predicted:** the magnitude of the entropy intervention's effect is smaller on G1 than
on Walker.

**Explicitly not predicted, and to be reported as empirical outcomes:** whether the
weighted_mle cell narrows when the entropy term is added; whether the pathwise cell
narrows when it is removed; whether the operator gap persists in either condition.

**Explicitly forbidden inference:** the sign of the entropy force must not be read off
median sigma alone. `sigma < 1/sqrt(2) ≈ 0.7071` is *sufficient* for widening pressure
because `E[sech^2] <= 1` bounds the bracket below by `1 - 2 sigma^2 > 0`, but
`sigma > 1/sqrt(2)` is *not* sufficient for contraction, since `E(sech^2(y))` depends on
mu and on the state distribution. Any sign claim must be measured on-arm, per the method
already used for Walker, not inferred from a width summary.

## 3. Cell naming

The Walker convention carries over unchanged, and `PW-H` / `WML-H` remain barred: the
same suffix would mean "minus entropy" in one arm and "with entropy" in the other.

| cell | arm | entropy | export tag |
|---|---|---|---|
| `PW_ent` | pathwise | ON | `pathwise_fa` (existing G1 baseline) |
| `PW_noent` | pathwise | OFF | `pathwise_fa_noent` |
| `WML_ent` | weighted_mle | ON | `weighted_mle_ent` |
| `WML_noent` | weighted_mle | OFF | `weighted_mle` (existing G1 baseline) |

`WML_ent` is a **hybrid ablation**: neither standard MPO nor standard REPPO.

## 4. Status of the two cited figures

Both are cited by the prediction, so both are recorded here with their verification state
at drafting time. The CLAIX login node was refusing connections, so neither could be
re-verified against artifacts at that moment.

**(b) baseline scale separation — consistent with the committed artifact.** The frozen
cross-task analysis in `reports/artifacts/fixed_bank_width_saturation.json`, reported in
`reports/eps_e_offline_analysis.md`, records `median_paired_ratio`: Walker **16.05x**,
G1 **1.81x**, LEAP 6.38x, each 8/8 seed pairs. These match the quoted values.

**(a) alpha ratio — VERIFIED.** Both values are confirmed from the executed run configs
and from all 32 baseline export metadata files, before this preregistration was committed.

| task | executed `ent_start` | `alpha_entropy` in all 16 exports | `update_entropy_lagrangian` |
|---|---|---|---|
| Walker | `0.014509912580251694` | `1.45099154e-02`, identical across all 16 | `false` (frozen) |
| G1 | `0.00020752247655764222` | `2.07522389e-04`, identical across all 16 | `false` (frozen) |

Ratio `0.014509912580251694 / 0.00020752247655764222 = 69.92`. The quoted `2.08e-4` and
"approximately 70x" are both correct, to three significant figures and to within 0.1x
respectively. No amendment is required.

Note the two arms within a task share the same frozen alpha exactly, so the alpha
difference is strictly between tasks, not between arms.

## 5. Two definitional notes, recorded so later reports cannot drift

**The ratio convention needs no restatement.** The fixed convention is the median over
seeds of per-seed log ratios, exponentiated. Because the median commutes with monotone
transforms, `exp(median(log r)) = median(r)` exactly, so the `median_paired_ratio` values
quoted above (16.05x, 1.81x) are already in the fixed convention and are not a
mean-of-ratios.

**The quoted baseline ratios compare the two existing baselines**, i.e. `WML_noent`
against `PW_ent`. That is not the same contrast as the factorial's "entropy off in both",
which is `WML_noent` against `PW_noent`. On Walker those differ materially: 16.05x for
the baseline pair versus 22.952x for entropy-off-in-both. Any G1 comparison against the
1.81x anchor must state which of the two contrasts it means.

## 6. Analysis, if and when the arms are run

Unchanged from the Walker factorial unless amended here: the frozen return scalar from
commit `7edb8e8`; the fixed-bank width and saturation on the canonical G1 bank
`g1_fixed_state_bank.npz`, sha256
`cf6f7880b3a5b59433727f7a245b5ce97749096981f03233fd434ab3371935e9`, hash verified before
use; width and saturation reported three ways, full bank and each arm-derived half;
paired percentile bootstrap of the paired median over the 8 seed differences, 10,000
resamples, `np.random.default_rng(20260902)`; interactions bootstrapped directly with
both deltas formed inside each replicate; no invented thresholds or p-values.

Comparing "smaller effect on G1 than on Walker" requires a stated scale. Effect
magnitudes will be reported on both the additive scale (return, in task units, which are
not comparable across tasks) and the multiplicative/log scale for width (which is
dimensionless and is the comparable one). The prediction is assessed on the width log
scale and on the within-task relative return change, with the incomparability of raw
return units stated rather than glossed.

## 7. Launch status at drafting

No G1 entropy-factorial run has been launched. This document states a prediction; it does
not itself authorise a launch.
