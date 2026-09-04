# Run note: Rule B rung r(d) = ceil(sqrt(d)) — written and committed BEFORE the run

**Obligation.** `docs/prereg_lqr_crossover.md` Sec. 5.3: "Ranks `r in {1, 2, ceil(sqrt(d)), d}`,
under the registered primary norm convention, at `M = 32`. Fit `p` as above at each
rank, and fit the joint exponent across the ladder." Three rungs exist (`rank1`,
`rank_r2`, `full`); this run supplies the fourth. It is unconditional (part of Rule B's
definition), not triggered.

**Design — the registered ladder arm, unchanged.** `kind = rank_r`, `rank = ceil(sqrt d)`
= 1, 2, 2, 3, 4, 6, 8 for `d` = 1, 2, 4, 8, 16, 32, 64; `M = 32`; `eps_frac = 0.05`
(`eps = 0.05 · q_spread(s, sigma)` per state and `sigma`); `normalize = unit_H`;
`cost = identity`; the registered 20 x 34 `sigma` x `omega` grid; state generation and RNG
by the registered conventions (`SEED_ROOT + d` system, `+2000 + d` states and field,
`+3000 + d` kernel key); replicate counts as the other two ladder arms:
`n_states = 32`, `n_batch = 20`, `r_batch = 100` (`N = 2000`); crossover by
`analyze.crossover_by_c`; contamination guards `rho(A_K) <= 0.99`, `cond(H) <= 50`,
`eps/q_spread` nominal. Files are tagged `_csd` so that the `d = 2, 4` cases (where
`ceil(sqrt d) = 2` coincides with the existing `rank_r2` arm and, by construction, must
reproduce it bit-for-bit) do not overwrite the manifest-hashed files. Nothing in the design
was chosen from existing results. Command:

    for d in 1 2 4 8 16 32 64; do
      JAX_PLATFORMS=cpu python scripts/lqr_crossover/sweep.py --d $d --kind rank_r \
          --rank $(python -c "import math;print(math.ceil(math.sqrt($d)))") \
          --n-states 32 --n-batch 20 --r-batch 100 --tag _csd
    done

**Expectation, recorded before any outcome is inspected.**

1. The nominal/RMS crossover exponent should remain near 1/2 if the already-observed RMS
   mechanism is rank robust (the three existing arms give 0.4876 / 0.4870 / 0.4871).
2. Because `omega_inf = omega / sqrt(r)` and `r ~ sqrt(d)`, the registered sup-norm
   coordinate introduces approximately a `d^(-1/4)` factor. With `r(d) = ceil(sqrt d)` a
   step function, the exact shift is the OLS slope of `-(1/2) ln r(d)` on `ln d` over the
   registered `d`-set, which is close to but not exactly `-1/4`.
3. Therefore an `omega_inf` exponent near 1/4 is expected mainly from coordinate algebra,
   not as an independent new scaling law.

This expectation is not used to modify the analysis or to exclude any data. The joint
Rule B verdict is adjudicated on all four rungs against the registered text after the run.
