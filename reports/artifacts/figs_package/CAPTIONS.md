# Figure package — draft captions

All figures are built from frozen artifacts by `scripts/analysis/entropy_and_figures.py`.
No new training. Every panel states its reduction and its state source.

## FIG 1 — `fig1_corrected_returns.{pdf,png}`
Data `fig1_corrected_returns_data.csv`.

> **Corrected operator replication, PW-1 versus WML-32, paired seeds 301-308.**
> Per-seed score is `score_window3`, the mean of the final three of 21 logged
> evaluations, the preregistered primary definition
> (`docs/prereg_corrected_operator_replication.md` section 4, commit `7edb8e8`).
> Contrast is `PW-1 minus WML-32`, paired within seed; the independent unit is the
> seed pair. Grey verticals join the two arms of a pair; markers are not joined
> across seeds because seeds have no ordering. Reported interval is the paired
> percentile bootstrap of the **paired median** over the 8 seed-level differences,
> 10 000 resamples, `np.random.default_rng(20260902)`, 95 %. State source: final
> evaluation rollouts of each run. Walker `+135.58 [+108.97, +200.76]` 8/8;
> G1 `+9.51 [+2.78, +15.28]` 8/8; LEAP `+4.30 [-1.34, +10.53]` 5/8, interval
> contains zero, classified `Inconclusive`.

## FIG 2 — `fig2_fixed_bank_width.{pdf,png}`
Data `fig2_fig3_width_saturation_data.csv`.

> **Final policy width on a frozen state bank, paired seeds.** Per-arm median
> pre-tanh sigma over a frozen, arm-major state bank evaluated with each run's own
> final checkpoint; reduction is the median over states and action coordinates.
> Log y-axis. Grey verticals join paired seeds. State source: the frozen banks
> (`walker_fixed_state_bank.npz` sha256 `8adfeb0b…aa21`; `g1_fixed_state_bank.npz`
> `cf6f7880…35e9`; `leap_fixed_state_bank.npz` `0053b0f3…b158e`), 3072 states each,
> 1536 from each arm's rollouts, at fixed depths. WML-32 is wider in 8/8 seed pairs
> on all three tasks; median paired ratio Walker 16.05x, G1 1.81x, LEAP 6.38x.

## FIG 3 — `fig3_saturation.{pdf,png}`
Data `fig2_fig3_width_saturation_data.csv`.

> **Action saturation on the frozen state bank, paired seeds.** Probability that a
> sampled action exceeds `|a| > 0.95` and `|a| > 0.99`, computed exactly from the
> pre-tanh Gaussian via `P(|tanh Y| > t) = Phi((-c-mu)/s) + 1 - Phi((c-mu)/s)` with
> `c = atanh(t)`; reduction is the mean over bank states and action coordinates.
> Same banks and checkpoints as FIG 2. WML-32 exceeds PW-1 at `|a| > 0.95` in 8/8
> seed pairs on all three tasks. On Walker the WML-32 policy places 73-84 % of its
> action mass beyond `|a| > 0.95` against 21-26 % for PW-1.

## FIG 4 — `fig4_controlled_theory.{pdf,png}`
Data `fig4_controlled_theory_data.csv`, from `reports/artifacts/planted_sweep.csv`.

> **Controlled planted critic error: estimator-level sensitivity.** Each point is one
> cell of a controlled sweep in which a known error `e` is planted in an otherwise
> exact critic; the quantity plotted is the ratio of **error-induced gradient
> variances**, `Var[g_ZO]_e / Var[g_PW]_e`, against `sigma*omega/sqrt(d)` with
> `omega := ||grad e||_inf / ||e||_inf`. The dashed line marks the predicted boundary
> at 1. Reduction: one point per sweep cell, 240 cells, `d` in {2,4,8,16,32,64}.
> **This is a statement about estimator sensitivity to critic error under a planted
> perturbation, not about the implemented operator and not a claim about how
> performance scales with task action dimension.** The collapse across `d` is a
> property of the plotted normalisation, which divides by `sqrt(d)` by construction;
> it is not evidence of a universal task-dimension law.
