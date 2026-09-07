# Preregistration — Walker WML eps_E = 0.1 as a signed test of the tilting theory

Committed and pushed **before** launch. Append-only. Supersedes nothing; the earlier
offline `eps_E` analysis (`reports/eps_e_offline_analysis.md`, commit `fb4ab3f`) recommended
this arm as exploratory. It is registered here instead as a **directional test** of the
mechanism set out in `docs/prereg_logsigma_decomp.md` (commit `5edaca3`).

## 1. The prediction and where it comes from

For locally linear `Q(y) ≈ g·y` on the pre-tanh variable, exponential tilting of a Gaussian
is exact:

```
w_i ∝ exp(g·y/eta)  ⟹  tilted = N(mu + sigma^2 g/eta, sigma^2)
Dmu = sigma^2 g / eta            A = ||Dmu||^2/sigma^2 = sigma^2 ||g||^2 / eta^2
```

`A` is the displacement term in the exact split `dL/dlog_sigma = -(A + B)` and is
non-negative identically. `B` is the finite-sample spread deficit, `B ≈ -d/ESS`. Setting
`A + B = 0` gives a stationary width

```
sigma* = (eta / ||g||) * sqrt(d / ESS)
```

**`eps_E` enters only through `eta`.** The MPO dual makes `eta` decrease as the KL budget
`eps_E` grows, and the already-measured offline counterfactual on identical candidates gives,
across the eight Walker seeds:

| eps_E | eta (mean over seeds) | ESS/M | w_max |
|---|---|---|---|
| 0.5 (executed) | 0.958 | 0.739 | 0.153 |
| 0.1 | 5.495 | 0.909 | 0.056 |

so `eta` rises by a factor of **5.73** and `ESS/M` from 0.739 to 0.909.

Both terms of `sigma*` move the same way. `A ∝ 1/eta^2` falls by about `5.73^2 ≈ 33`, and
`sqrt(d/ESS)` falls as ESS rises. The theory therefore predicts, unambiguously:

```
PREDICTION: Walker WML at eps_E = 0.1 has a SMALLER matched-state median pre-tanh sigma
            than Walker WML at eps_E = 0.5.
```

This is a **signed, quantitative** prediction, not a direction guessed from a width summary.
Order of magnitude, holding `||g||` and `d` fixed and taking ESS from the table,
`sigma*` scales by `(5.495/0.958) * sqrt(0.739/0.909) ≈ 5.2` in `eta/||g||` terms against
`sqrt(d/ESS)` falling by `0.90`, so the leading effect is a **substantial narrowing**. No
precise multiplier is registered, because `||g||` is itself a function of the policy that
will change during training; only the **direction** is registered.

## 2. What would refute it

```
REFUTED if the matched-state median pre-tanh sigma at eps_E = 0.1 is
         >= the eps_E = 0.5 value, on the frozen neutral Walker bank,
         under the frozen ratio convention, with the paired interval excluding
         a narrowing.
```

A null result (interval containing 1) is recorded as **inconclusive**, not as support.

Two ways the prediction could fail while the mechanism still holds, both recorded now so
they are not invoked post hoc as rescues:

* `||g||` may fall along with `eta` if the critic's action-gradient shrinks as the policy
  changes, partially cancelling the effect. Recorded as an anticipated confound; it would
  weaken the magnitude, not flip the sign.
* If `A` turns out to be indistinguishable from zero in the decomposition (step 1-3,
  job `3823977`), then the mechanism is already not supported under `5edaca3`'s decision
  rule and this arm tests nothing. **If that decomposition fails its own gate, this arm's
  result is reported but carries no theoretical weight.**

## 3. Design

Exactly one config field differs from the executed Walker WML-32 baseline.

| field | baseline | this arm |
|---|---|---|
| `eps_e` | 0.5 | **0.1** |

Everything else pinned to the recovered Walker baseline: task `WalkerRun`, `actor_update_mode
weighted_mle`, `estep_num_samples 32`, seeds 301-308, 128 minibatches x 4 epochs,
`total_time_steps 50000000` (52,297,728 executed), `num_eval 20`, `ent_start
0.014509912580251694` with `update_entropy_lagrangian false`, `kl_bound 0.1`, `reduce_kl
true`, `reverse_kl false`, `actor_kl_clip_mode clipped`, `mstep_decoupled false`,
`sqrt_rho 1.0`, `lr 3e-4`, `max_grad_norm 0.5`, `gamma 0.99`, `lmbda 0.95`, `vmin 0`,
`vmax 150`, 151 bins, `hl_gauss`, episode length 1000, `min_std` reaching the model 0.1.

**Export isolation, with no change to experimental source.** `eps_e` is not part of the
export tag, so these runs would otherwise write the baseline's own paths
`WalkerRun_weighted_mle_s*` and destroy them, the failure mode that already cost two Walker
checkpoints once in this project. Rather than edit `scripts/train_and_export.py`, which is
experimental source and would have to be changed on the workstation, the runs execute from a
**dedicated git worktree** at the same commit with its own real `exports/` directory.
`train_and_export.py:29` derives `REPO_ROOT` from `__file__`, so every export lands inside
that worktree and the canonical `exports/` tree is unreachable from the job. This is the
same isolation the `T1b` fixed-batch test uses.

Consequences recorded now: the tag inside the worktree is the baseline's tag, and the
**parent directory is what disambiguates**, so these checkpoints must never be copied into
the canonical tree under those names. The canonical Walker baselines are hashed before and
after the launch and required to be unchanged.

Partition: the Walker WML-32 baseline ran entirely on `c25g`, so this arm runs on `c25g`.

Cost: 8 seeds x about 22 min = roughly 2.9 GPU-hours.

## 4. Outcomes

**PRIMARY (mechanistic)** — matched-state median pre-tanh sigma on the canonical neutral
Walker bank, sha256 `8adfeb0bf70bddcdbd64a84b972b4dbd62617c64e1c948589a7adc30cf64aa21`,
hash verified before use, paired within seed against the `eps_E = 0.5` baseline, ratio
convention the median over seeds of per-seed log ratios exponentiated. Both bank halves
reported separately.

**SECONDARY** — `A`, `B` and `net` from the same decomposition as `5edaca3`, so the
predicted fall in `A` can be checked directly; logged `eta`, ESS, `w_max`, gate-fire
fraction; `sat95`, `sat99`.

**RETURN** — the frozen statistic from commit `7edb8e8`, reported but **not** the test.
Return is not predicted here in either direction.

## 5. What this does not test

It varies `eta` through `eps_E` and observes width. It does not manipulate `||g||` or `d`,
so it tests one factor of `sigma*` and not the formula as a whole. It cannot separate the
`1/eta^2` route through `A` from the ESS route through `B`, since `eps_E` moves both; the
decomposition's `A` and `B` values are what separate them, and they are reported alongside.
No causal claim about return is made.
