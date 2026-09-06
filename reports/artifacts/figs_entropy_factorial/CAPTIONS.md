# Walker entropy factorial — draft captions

Produced by `scripts/analysis/entropy_factorial_analysis.py`, job `3740916` at `4665b08`.
Source artifacts: `entropy_factorial_cells.csv` (cell x seed) and
`entropy_factorial_percoord.csv` (cell x seed x coordinate).

## `fig_ef_return.{pdf,png}`

> **Walker entropy factorial: return, paired seeds 301-308.** Per-seed scalar is
> `score_window3`, the mean of the final three of 21 logged evaluations, the
> preregistered primary definition (commit `7edb8e8`). Four cells: `PW_ent`,
> `PW_noent`, `WML_ent`, `WML_noent`, where the suffix denotes whether the actor
> entropy term is present. Reduction: one point per cell per seed, no aggregation
> across seeds. State source: each run's own final evaluation rollouts. Paired
> effects, paired percentile bootstrap of the paired median over the 8 seed
> differences, 10,000 resamples, `np.random.default_rng(20260902)`:
> `PW_ent - WML_ent = +22.939 [+3.696, +36.328]`, 7/8 positive;
> `PW_noent - WML_noent = +127.111 [+111.937, +166.093]`, 8/8 positive.

## `fig_ef_width.{pdf,png}`

> **Walker entropy factorial: policy width, paired seeds.** Median pre-tanh sigma per
> checkpoint, reduced as the median over bank states and action coordinates, on the
> frozen neutral Walker bank `walker_fixed_state_bank.npz`, sha256
> `8adfeb0b...aa21`, verified before use; the same 3072 states for all 32 checkpoints.
> Log y-axis. Ratio convention where quoted: the **median over seeds of per-seed log
> ratios**, exponentiated, not a mean of ratios. `WML/PW = 4.533x [3.255x, 5.987x]`
> with entropy on in both, `22.952x [13.170x, 33.400x]` with it off in both.

## `fig_ef_saturation.{pdf,png}`

> **Walker entropy factorial: action saturation, per coordinate, frozen bank.**
> `P(|a| > 0.95)` and `P(|a| > 0.99)` per **action coordinate**, computed exactly from
> the pre-tanh Gaussian via `P(|tanh Y| > t) = Phi((-c-mu)/s) + 1 - Phi((c-mu)/s)` with
> `c = atanh(t)`; the policy is not sampled. Reduction: mean over the 3072 bank states
> and the 6 action coordinates, one point per cell per seed. Same bank and hash as the
> width figure. Note that `PW_noent` has the narrowest median sigma of the four cells
> yet more than double `PW_ent`'s saturation, because its width distribution grows a
> heavy tail (mean 0.701, max 202.0, against 0.497 and 4.2).
