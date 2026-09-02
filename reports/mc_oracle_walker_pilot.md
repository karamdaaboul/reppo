1. **PILOT FEASIBILITY: NOT YET PRECISE ENOUGH TO SCALE**
2. **PW-trained checkpoint `r_RMS`: UNRESOLVED — ORACLE PRECISION INSUFFICIENT.** The debiased gradient energy is negative at both step sizes (`N_0.10 = -4.14`, 95% CI `[-21.67, +12.19]`), so no square root is taken and no interval is quoted; `r_RMS` is defined in only 31% of bootstrap replicates.
3. **WML-trained checkpoint `r_RMS` = 1.55, 95% CI [0.57, 2.52]** at `c = 0.10` (defined in 94% of replicates, so the interval is conditional on that). Placement: **NEAR BOUNDARY — INTERVAL CROSSES 1**.
4. **Finite-difference stability: FAIL** on both. Finite-difference bias is therefore reported as **unresolved**, and the failure is attributable to noise, not demonstrably to bias.
5. **Horizon stability: PASS** on both, by a wide margin (`N_1000/N_500` = 1.018 and 1.001; centred tail RMS 1.1% and 0.12% of the centred error RMS, against a 25% threshold).

---

**One preregistration-level defect was found in this pilot and is reported in §4 and §11: `Q_phi` is fit to a *clipped* return, and the preregistered estimand is the unclipped one.** That is my error, not a property of the data. It leaves the PW result untouched (0.0% of PW rollouts are affected) and materially distorts the WML result (30.6% of rollouts, 34.5% of action points). It is the first thing a scaled study must fix.

---

## 1. Provenance

| Item | Value |
|---|---|
| Branch | `estep-study` |
| Oracle code commit (unchanged across both runs) | `8fea5cf` |
| Preregistration commit | `63c2cd2` |
| State-bank commit | `2527a3a` |
| HEAD at analysis | `2f38177` |
| Working tree | clean at each run |
| Remote | `origin/estep-study` = `2f38177` |
| Corrected-replication report commit | `d731e63` |
| Corrected-replication preregistration | `7edb8e8` |

Both pilot jobs ran on the **same node**, `w25g0006` (partition `c25g`), on an **NVIDIA H100 80GB HBM3**. `scripts/analysis/mc_oracle_walker.py` was last modified at `8fea5cf`, before either job started, and the only later commit (`2f38177`) adds `scripts/analysis/mc_oracle_package.py` and touches nothing the oracle uses. The two arms are therefore byte-identical code on identical hardware.

| Job | Arm | Node | Elapsed | Start |
|---|---|---|---|---|
| `3463096` | WML pilot + horizon | `w25g0006` | 00:09:29 | 2026-09-02T13:37:09 |
| `3464732` | PW tests + pilot + horizon | `w25g0006` | 00:13:14 | 2026-09-02T13:56:27 |

**Software.** Python 3.12.14; jax 0.5.2; jaxlib 0.5.1; flax 0.10.6; distrax 0.1.5; optax 0.2.5; mujoco 3.10.0; mujoco-mjx 3.10.0; brax 0.14.2; mujoco_playground 0.0.5; numpy 2.5.1; scipy 1.18.0.

**Checkpoints (fixed in the preregistration before any oracle value existed).**

```
exports/WalkerRun_pathwise_fa_s301_final
  actor.npz       e0cbb41c2bbd92aca194bfd058f28d4188a6f7bdce5c4fac6103b8711d8812b4
  critic.npz      b8f22a654a53e8f7bbd587d4b66a51d7516607b658ea1da143013ecf53ba8ba9
  normalizer.npz  59586248211d5b165a13ac182da9c521bf9add2ad163deac62db32cda55280b0
  meta.json       6298c9111cb588cc2656a7d2fdbe4af1a776a970a0158fe9a7b2e0fa2bf5d5d9

exports/WalkerRun_weighted_mle_s301_final
  actor.npz       0e68d7ecc210cb9938384cd08294edf42f0d3f7ec24039ce59b22760151790dd
  critic.npz      c56c4d27a3265ef1f61eddef51ff89917da8b596d2bab190d86fb197aa2a20da
  normalizer.npz  8dc1ce8cbb835a0dd6bc982131aa869706740dec7405523dc1ce2624a9e64aae
  meta.json       10d2a50fab4d293c055873541d5d7885a451cb85795a8d675a38c5eead7ce8ba
```

Both: `WalkerRun`, seed 301, faithful-repair arm, final checkpoint, `d = 6`,
`time_steps = 52297728`, `iteration = 399`, `gamma = 0.99`, `lmbda = 0.95`,
`eps_e = 0.5`, `estep_num_samples = 32`, `hl_gauss = True`, `num_bins = 151`,
`vmin = 0`, `vmax = 150`, `max_episode_steps = 1000`, `reward_scaling = 1.0`,
`obs_dim = critic_obs_dim = 24`, `action_pad = 0`, `min_std = 0.1`.

## 2. Planted-error commit verification

`git show 7663d03 -- reports/planted_error_phase_diagram.md` returns a 13,042-byte
report present at that commit. It contains, verbatim: `240 cells`; *"Zero
misclassifications in 240 cells."*; *"**Median `r* = 1.0085`, full range [0.974,
1.128]** over 19 independent slices."*; and `scripts/planted/run_tests.py` —
**12/12 pass**.

The boundary was genuinely **pre-marked**, not fitted after the fact:
`scripts/planted/planted_sweep_config.json` was added at commit `7534b77`
(2026-09-01 21:14:40) — 17 minutes before the report commit `7663d03`
(21:31:43) — and states the prediction verbatim as *"the zeroth-order estimator has
the smaller ERROR-INDUCED variance when sigma*omega > sqrt(d)"*, i.e. `r = 1`.

**PLANTED-ERROR PHASE DIAGRAM ALREADY COMPLETE — NOT RERUN.** No compute was spent
repeating it.

## 3. Exact critic-target trace

Full line-level trace in `reports/mc_oracle_code_trace.md`. The three findings that
matter, none of them the textbook SAC target:

1. **`Q_phi` is the expectation of a 151-bin HL-Gauss categorical head** over
   `linspace(0, 150, 151)` (`jax_models.py:305-310`), at the **online** critic
   parameters. The variable is named `critic_target_model` but is built from
   `train_state.critic.params` (`reppo.py:729-732`); there is no critic target network
   in this update. Single head — no ensemble, no twin, no min reduction. The
   representation structurally confines `Q_phi` to `(0, 150)`.
2. **The target is a TD(`lambda = 0.95`) return** (`reppo.py:612-640`), not a one-step
   backup. The Retrace importance weight is present but **inert**: `config/reppo.yaml`
   sets `exploration_noise_min = exploration_noise_max = 1.0`, so `rho == 1`. The
   bootstrap `V_{t+1} = Q_phi(s_{t+1}, a_{t+1})` is a one-sample estimate at a sampled
   action and enters linearly, so the fixed point is unaffected by that sampling.
3. **The entropy term sits on `a_{t+1}`**: `soft_reward = reward - gamma *
   log_prob.sum(-1) * temperature()` (`reppo.py:561-564`), so
   `r_tilde_t = r_t - gamma * alpha * log pi(a_{t+1}|s_{t+1})` and unrolling gives
   `sum_k gamma^k r_k - alpha * sum_{j>=1} gamma^j log pi(a_j|s_j)`. **The externally
   fixed `a_0` carries no entropy term.** Verified, not assumed; the `gamma` inside
   `soft_reward` is exactly what shifts the entropy series to start at `j = 1`.

Also recorded, and not assumed away: **the truncation flag is off by one.**
`transition.truncated = next_env_state.truncated` flags `s_{t+1}`, but the reverse
scan's carry supplies `batch.truncated[t+1]`, which flags `s_{t+2}`. At a time-limit
boundary the `where` takes the wrong branch and `done_t = 1` then removes the
bootstrap entirely. This touches ~1/1000 of transitions, and those are separately
masked out of the critic loss by `reppo.py:713`.

`alpha` was read programmatically as `float(actor.temperature())` =
**0.014509915374219418**, identical for both arms, frozen
(`update_entropy_lagrangian=false`). It differs from the launch command's
`ent_start = 0.014509912580251694` in the eighth decimal — the float32 log/exp
round-trip. The value from the checkpoint was used.

## 4. The `Q^pi` actually used — and a defect in the preregistered estimand

The oracle computes

```
Q_soft^pi(s0, a0) = E[ r_0 + sum_{t>=1} gamma^t ( r_t - alpha log pi(a_t|s_t) ) ]
```

by finite-horizon Monte Carlo at `H = 500`, with `pi` the exported **online** actor at
`scale = 1.0` (`reppo.py:516`, `export_ckpt.py:160`), the environment receiving
`clip(a, -0.999, 0.999)` at every step including `t = 0`, and **no `Q_phi` bootstrap
at the horizon**. That last point is a deliberate departure from
`scripts/critic_fidelity/common.py:soft_return`, which adds `gamma^H Q_phi(s_H, a_H)`:
the oracle is differenced *against* `Q_phi`, so a `Q_phi` tail would make
`e = Q_phi - Q_oracle` partly a comparison of `Q_phi` with itself.

### The defect

`src/jaxrl/utils.py:45` is

```python
x = jnp.clip(inp, vmin, max=vmax).squeeze() / (1 - epsilon)
```

so the `lambda`-return is **clipped into `[0, 150]` before the two-hot encoding**.
`Q_phi` therefore estimates `E[clip(G, 0, 150)]`, not `E[G]`. I quoted that line in
the code trace but did not carry it into the estimand, and the preregistration
inherited the unclipped definition. The preregistration is immutable, so the primary
result in §11-§13 is reported exactly as registered; the consequence is quantified as
an explicitly post-hoc diagnostic in §11.

How much it bites is entirely arm-dependent:

| | fraction of rollouts with `Q_oracle < 0` | fraction of action points whose mean is `< 0` |
|---|---|---|
| PW | **0.0%** | **0.0%** |
| WML | **30.6%** | **34.5%** |

The PW result is untouched — its post-hoc clipped statistics are numerically
*identical* to the primary. The WML result is not.

The mechanism is the entropy term. WML's training-logged policy entropy is
**-33.34** (i.e. `E[log pi] = +33.3`) against PW's **-2.62**; `log pi` for a
tanh-Gaussian grows without bound through the `tanh` log-det Jacobian as the
pre-squash sample moves out, so `-gamma * alpha * log pi` drives the soft return
strongly negative. The WML oracle returns reach **-2394.91**. `Q_phi`, floored at ~0
by its support and by the target clip, cannot follow it there.

## 5. State-restoration validation (phase-2 gate)

**PASSED.** `MjxGymnaxWrapper.step` calls `self.env.step(state, action)` and **discards
the RNG key** (`jax_wrappers.py:103`), so WalkerRun dynamics and its auto-reset are a
pure function of `(state, action)`. This was verified empirically rather than assumed.

The complete state is the wrapped `LogEnvState` pytree: **197 leaves**, covering the
MJX pipeline state (`qpos`/`qvel` and the rest of `Data`), the observation, reward and
done fields, the auto-reset buffers, episode counters, the truncation flag, and the
`info` dict.

T1: one state cloned into two batch entries, identical first action, identical
continuation innovations, 100 steps. Result on GPU:

```
max|diff|  obs=0  rew=0  done=0  act=0  soft_r=0  acc=0
```

Exact zeros, not "close". T1b: same restored state, different first action, same
innovations — final observation gap 25.16 with an innovation gap of exactly 0, so the
branches diverge through the action alone while the common random numbers hold.

## 6. Preregistration commit

`docs/prereg_mc_oracle_walker_pilot.md`, commit **`63c2cd2`**, committed before any
oracle value was computed for either checkpoint. Everything computed before that
commit was the phase-2 gate and the phase-5 unit tests, on reset states, with no state
bank, no `e`, no `D`, no `N` and no `r_RMS`.

## 7. State bank

`reports/artifacts/mc_oracle_state_bank.npz`, sha256
**`64a96411621dcd6586a6280a81b5d6c25e65c40a83252c3c86a58174d5642a1e`**, matching the
value recorded in `mc_oracle_state_bank_manifest.json`.

64 states, 32 rolled under the corrected PW-301 policy and 32 under WML-301, 50-step
burn-in from reset with actions clipped to `+-0.999`, RNG root `20260902` with
purpose-separated blake2b fold-ins. All 64 are distinct. All 197 leaves are saved, not
the observation alone. Mean observation norm 27.49 on the PW-source stratum and 25.86
on the WML-source stratum.

`rho_mix = 0.5 rho_PW301 + 0.5 rho_WML301`. States from one arm are **not** called
neutral. Both checkpoints were evaluated on the same 64 raw environment states, each
applying its own normalizer, policy, `Sigma`, critic and continuation policy. The
standardized `u_k` were shared between the checkpoints and verified equal at analysis
time (`assert np.allclose(runs["PW"]["u"], runs["WML"]["u"])`).

## 8. Oracle validation tests

**18/18 pass on the H100** (and 18/18 on CPU beforehand). Full log in
`logs/mc-oracle_3464732.out`.

| Test | Result |
|---|---|
| T1 full-state clone/replay, 100 steps | PASS — all fields exactly 0 |
| T1b branching diverges, innovations common | PASS — gap 25.16, innovation gap 0 |
| T2 forced `a0` = policy sample == plain rollout | PASS — 7.93e-08 |
| T3 soft return == hand-unrolled `sum gamma^t r_tilde_t` | PASS — 8.94e-08 |
| T4 no entropy term on the fixed `a0` | PASS — an `a0` term would shift by >= 0.0105 |
| T5 discount indexing at every prefix `k=1..6` | PASS — 1.19e-07 |
| T6 normalizers reproduce saved statistics | PASS — 4.77e-07 |
| T7 `Q_phi` == existing `load_ckpt` evaluator | PASS — exactly 0 |
| T7b hand-written tanh-Normal density == distrax | PASS — 1.55e-06 (float32) |
| T8 / T8b common random numbers | PASS — branch spread exactly 0 / 1.39 |
| T9 groups A and B independent | PASS — corr = -0.0069 |
| T10 / T10b finite differences | PASS — 7.9e-14 on a quadratic; `c^2` ratio 4.00 |
| T11 cross-product debiasing | PASS — 2.0435 vs 2.0000; naive 26.92 |
| T12 `K/(K-1)` centering factor | PASS — 1.8760 -> 2.0010 |
| T13 shorter horizon is the exact prefix | PASS — reproducible to 0 |
| T14 state bank round-trips | PASS — 197 leaves, 0 mismatched |

One test failed on first run and the fault was in the test: T8 used `Harness.reset`,
which gives every batch entry its own initial state, so the four "branches" were four
different states and the test was measuring state heterogeneity rather than common
random numbers. `run_pilot` already tiled one state across its branches; the test now
does the same, and the spread is exactly 0. **No test was weakened after inspecting
the pilot outcome** — the pilot had not been run when this was fixed.

## 9. `Q^pi` estimates and Monte-Carlo variance

Per checkpoint, over `64 x 16 = 1024` base action points, 16 rollouts each (8 in group
A, 8 in group B), `H = 500`:

| | PW-trained | WML-trained |
|---|---|---|
| `Q_oracle` mean | 68.62 | **-4.08** |
| `Q_oracle` range | `[1.53, 86.12]` | `[-2394.91, 59.96]` |
| `Q_oracle` median (per rollout) | — | 44.64 |
| trajectory sd across the 16 rollouts (median) | 1.09 | 38.06 |
| per-group standard error (median) | 0.369 | 11.70 |
| per-group standard error (p95) | 3.42 | 51.49 |
| per-group standard error (max) | 9.62 | 307.78 |
| `Q_phi` mean | 72.92 | 32.90 |
| `Q_phi` range | `[43.88, 85.73]` | `[0.30, 55.31]` |
| mean gap `Q_phi - Q_oracle` | **+4.30** | **+36.98** |
| episodes ending within `H=500` | 0 | 0 |

The WML oracle's per-point standard error is **32x** the PW one. That single number
explains most of what follows.

`n_done = 0` everywhere at `H = 500`: bank states sit at step 50 and no rollout reaches
the 1000-step limit, exactly as the preregistration anticipated.

## 10. Centred versus uncentred critic error

Centring is what the estimator actually sees, and it removes a great deal:

| | uncentred error RMS | centred error RMS `sqrt(D)` | ratio |
|---|---|---|---|
| PW | 10.63 | **1.77** | 6.0x |
| WML | 70.91 | **20.44** | 3.5x |

Statewise mean critic error `mean_u e(s,y)`:

| | median | min | max | PW-source states | WML-source states |
|---|---|---|---|---|---|
| PW critic | 1.54 | -4.40 | 47.74 | 1.77 | 6.83 |
| WML critic | 13.36 | -18.84 | 318.69 | 33.48 | 40.48 |

So a per-state constant offset accounts for the large majority of the raw critic
error, and would have inflated a frequency estimate built on the uncentred quantity by
roughly 6x (PW) and 3.5x (WML) in the denominator. This is the concrete justification
for the centring requirement.

## 11. Noise-debiased denominator `D`

| | `D` | 95% CI | criterion A |
|---|---|---|---|
| PW | 3.138 | `[-0.021, 9.195]` | **FAIL** (lower bound below 0 by 0.021) |
| WML | 417.79 | `[122.50, 776.76]` | **PASS** |

PW fails A by a hair: the bootstrap lower bound is `-0.021`, and `D > 0` in 97.2% of
replicates. WML passes decisively (`D > 0` in 100% of replicates).

The debiasing is doing real work. The naive estimator — squaring one noisy MC estimate,
which the preregistration forbids — would have reported:

| | debiased `D` | naive `D` | share of the naive value that is MC noise |
|---|---|---|---|
| PW | 3.138 | 6.643 | 53% |
| WML | 417.79 | 915.12 | 54% |

### POST-HOC: the clipped-target variant

Not preregistered; added after seeing the WML output; an **approximation**, because
training clips a `lambda`-return that bootstraps on `Q_phi >= 0` whereas this clips a
500-step unbootstrapped MC return, so `E[clip(G_lambda)] != E[clip(G_MC)]`.

| | primary `D` | clipped-target `D` | primary `r_0.10` | clipped `r_0.10` | clipped `r_0.05` |
|---|---|---|---|---|---|
| PW | 3.138 | **3.138** (identical) | unresolved | unresolved | unresolved |
| WML | 417.79 | **28.78** | 1.547 | **1.097** | **0.992** |

Under the clipped-target oracle, **93% of the WML centred error power disappears**,
`r_RMS` moves to essentially 1, and the two step sizes agree to within 10%
(`0.992 / 1.097 = 0.904`, inside the `[0.8, 1.25]` band that criterion C failed on).
This is a diagnostic, not a result, and it is not substituted for the primary. It does
say clearly where the remedy lies.

## 12. Noise-debiased numerator `N`

| | `N_0.10` | 95% CI | `N_0.05` | 95% CI | criterion B |
|---|---|---|---|---|---|
| PW | **-4.14** | `[-21.67, +12.19]` | **-4.20** | `[-50.14, +39.96]` | **FAIL** |
| WML | 5997.3 | `[-1365.7, +15862.8]` | 10812.7 | `[-10297.3, +40131.4]` | **FAIL** |

Both PW estimates are **negative**. Per preregistration §8 this is reported as
**UNRESOLVED — ORACLE PRECISION INSUFFICIENT**; the value is not clamped to zero and
no square root is taken. Both WML intervals cross zero.

This is the binding failure of the pilot, and the reason is not subtle:

| | signal `N_0.10` | MC noise variance | SNR | naive `N_0.10` would be | share that is noise |
|---|---|---|---|---|---|
| PW | -4.14 | 188.4 | **-0.022** | 186.66 | **97.8%** |
| WML | 5997 | 67067 | **0.089** | 67783 | **91.2%** |

Squaring a single noisy estimate would have reported a gradient energy of 186.7 for
PW, where the debiased estimate is indistinguishable from zero — a **45x**
overstatement — and 67783 for WML against 5997, an **11x** overstatement. The
cross-product design is the only reason this failure is visible at all rather than
being reported as a confident positive number.

The `c = 0.05` arm is far worse, as it must be: `z ~ 1/c`, so the noise variance
scales as `1/c^2`. Measured inflation from `c = 0.10` to `c = 0.05` is **3.8x** (PW)
and **5.1x** (WML), against a predicted 4x.

## 13. `r_RMS` at both finite-difference step sizes

| | `r_0.10` | 95% CI | replicates defined | `r_0.05` | 95% CI | replicates defined |
|---|---|---|---|---|---|---|
| PW | **unresolved** (`N<0`) | — | 30.9% | **unresolved** (`N<0`) | — | 40.8% |
| WML | 1.547 | `[0.571, 2.523]` | 93.7% | 2.077 | `[0.618, 4.168]` | 80.9% |

**The quoted intervals are conditional on `N > 0` and `D > 0`**, which is why the
fraction of defined replicates is reported beside each. For PW that fraction is 31%,
which is why no interval is quoted at all rather than quoting `[0.098, 2.915]` as
though it were valid.

State-only bootstrap (perturbations kept attached to their state), as a robustness
check: PW `[0.094, 2.571]`, WML `[0.675, 1.959]` — same picture, slightly narrower for
WML.

**Placement.** WML: **NEAR BOUNDARY — INTERVAL CROSSES 1**. PW: not placed.

**Criterion C (step-size stability): FAIL on both.**

| | `r_0.05 / r_0.10` | `N_0.05 / N_0.10` | pass band |
|---|---|---|---|
| PW | undefined | 1.014 | `[0.8, 1.25]` |
| WML | 1.343 | 1.803 | `[0.8, 1.25]` |

Finite-difference bias is therefore **unresolved**. It is worth being precise about
what this does and does not show: the `c = 0.05` arm carries 4-5x the noise variance
of the `c = 0.10` arm and has an SNR of `-0.006` (PW) and `0.032` (WML), so the
comparison cannot presently discriminate step-size bias from step-size noise. The
observation that PW's `N` ratio is 1.014 — well inside the band — while its `r` ratio
is undefined illustrates the same point.

**Criterion E (interval precision): FAIL on both.** WML relative width
`(2.523 - 0.571) / 1.547 = 1.263`, against a threshold of 0.60. PW: not computable.

## 14. `H = 500` versus `H = 1000` sensitivity

**Criterion D: PASS on both, comfortably.** On the prospectively fixed subset (first 8
states of each stratum = 16 states, first 8 perturbations, `c = 0.10`), with the
`H = 500` estimate taken as the literal prefix of the same `H = 1000` rollout:

| | centred tail RMS (debiased) | centred tail RMS (naive) | centred `e_500` RMS | ratio (debiased / naive) | threshold |
|---|---|---|---|---|---|
| PW | 0.0378 | 0.0424 | 3.372 | **0.011 / 0.013** | < 0.25 |
| WML | 0.0344 | 0.2936 | 27.700 | **0.0012 / 0.0106** | < 0.25 |

Both the debiased and the conservative naive numerator pass, so the verdict does not
depend on which is used.

| | `N_500` | `N_1000` | ratio | pass band | `r_500` | `r_1000` |
|---|---|---|---|---|---|---|
| PW | 15.05 | 15.32 | **1.018** | `[0.8, 1.25]` | 0.470 | 0.469 |
| WML | 8007.6 | 8015.3 | **1.001** | `[0.8, 1.25]` | 1.319 | 1.320 |

The mean 500-to-1000 tail is **+0.516** (PW) and **+0.327** (WML) in value units — a
near-constant per-state offset, which is why the *centred* tail is two orders of
magnitude smaller. Scaling geometrically by `gamma^500 = 6.6e-3`, the residual beyond
`H = 1000` is of order 0.003; that is an **empirical extrapolation from the measured
tail, not a bound**. No rigorous `|bias_H| <= gamma^H B/(1-gamma)` is offered, because
no finite per-step `B` exists: the reward is bounded on `[0,1]` but
`-gamma * alpha * log pi` is unbounded above for a tanh-Gaussian policy. **It is not
claimed that centring eliminated truncation bias**; the horizon-doubling test passed
on its own terms.

One thing this subset shows that the main pilot does not: **`N_500 = 15.05` for PW is
positive** on these 16 states, where the 64-state pilot gives `-4.14 +- 7.41`. Their
difference of 19.19 is **at most 2.6 standard errors** (that bound uses only the
main-pilot standard error; the subset carries its own, which can only widen the
denominator), and both are consistent with a true `N` somewhere in roughly `[0, 15]`. The sign of PW's gradient energy is subsample-dependent, which is
the cleanest single demonstration that it is unresolved rather than small-and-known.

## 15. State-source-stratified results

Strata are reported separately, as registered; **no selection between them was made
after seeing results**, and the primary weights them 50/50.

| Checkpoint | Stratum | `D` | `N_0.10` | `r_0.10` |
|---|---|---|---|---|
| PW | PW-source states | 0.215 | -3.09 | unresolved |
| PW | WML-source states | 6.061 | -5.20 | unresolved |
| WML | PW-source states | 320.6 | 2510.3 | 1.142 |
| WML | WML-source states | 515.0 | 9484.2 | 1.752 |

The PW critic's centred error power is **28x larger on WML-source states than on its
own** (6.06 vs 0.215). Both `N` values remain negative, so no `r_RMS` follows. For the
WML critic, both strata place `r_RMS` above 1, at 1.14 and 1.75.

## 16. PW-versus-WML critic comparison

On the identical mixed state bank, with the same bootstrap indices:

* Centred error power: WML **417.8** against PW **3.14** — a factor of 133. §11 shows
  that roughly 93% of the WML figure is an artefact of the estimand mismatch; the
  clipped-target variant gives 28.8, a factor of 9.2.
* Paired `PW - WML` difference in `r_0.10`: **undefined**, because PW's `r_RMS` is
  undefined. The bootstrap difference interval `[-2.058, +1.502]` (20.2% positive) is
  conditional on both being defined in the same replicate and is **not** a valid
  interval for the contrast; it is recorded only to show that the contrast is nowhere
  near resolvable at this precision.

**No interpretation of this difference as explaining the corresponding return
difference is offered or implied.**

## 17. Feasibility decision

```
ORACLE PILOT NOT YET PRECISE ENOUGH TO SCALE
```

Criterion-by-criterion:

| | A `D>0` | B `N>0` | C step size | D horizon | E interval | all |
|---|---|---|---|---|---|---|
| PW | FAIL (by 0.021) | **FAIL** | FAIL | PASS | FAIL | FAIL |
| WML | PASS | **FAIL** | FAIL | PASS | FAIL | FAIL |

**The dominant source is Monte-Carlo rollout variance in the finite-difference
gradient.** 97.8% (PW) and 91.2% (WML) of the naive squared whitened gradient is MC
noise. It is not horizon truncation (criterion D passes with 20x margin), not state
heterogeneity as the primary driver, and not an implementation or restoration issue
(18/18 tests pass, state restoration is exact to zero).

A second, independent problem is **not** a precision problem at all: the estimand
mismatch of §4. On the WML arm the preregistered unclipped `Q^pi` is not the quantity
`Q_phi` was fit to, over 34.5% of the action points.

A third contributor, specific to WML: **tanh saturation**. 87.3% of WML base actions
have at least one saturated coordinate (45.7% of coordinates), against 3.8% and 0.64%
for PW. Where the action saturates, `Q^pi` is genuinely flat in `y` while `Q_phi` is
not, so the measured `r_RMS` on the WML arm is substantially a measure of saturation
rather than of critic roughness. The preregistered no-clip sensitivity retains 95.9%
of PW points (`D = 3.55`, `N_0.10 = +1.21`, `r = 0.239`, but `N_0.05 = -5.99`, still
unstable) and only **12.4%** of WML points (`D = 10.56`, `N_0.10 = -2009.6`,
`N_0.05 = +2664.4` — a heavily selected minority and unstable in both directions). The
sensitivity is therefore uninformative on the WML arm, which is itself worth knowing.

### Estimated remedy, not run

Because `E[z_A z_B]` is unbiased whatever the noise, and the SNR is well below 1, the
variance of the product is dominated by the noise-squared term, which scales as
`1/R^2` in the per-group mean; so `se(N) ~ 1/R` in rollouts and `~ 1/sqrt(S)` in
states.

| | measured `se(N_0.10)` | needed for B | factor | rollouts per group | or states |
|---|---|---|---|---|---|
| WML | 3726 | 3060 | 1.22 | 8 -> **~10** | 64 -> ~95 |
| PW | 7.41 | ~2.6 (if true `N ~ 5`) | ~2.9 | 8 -> **~24** | 64 -> ~540 |

* **More rollouts per oracle point is the right lever**, and it is cheap here: each
  pilot took 9-13 minutes of one H100, so tripling the rollout count is under an hour
  per checkpoint. This is the recommended remedy.
* **More states is the wrong lever**: it buys only `1/sqrt(S)` and would need ~540
  states on the PW arm.
* **More perturbations** does not help `N`, which is already averaged over `S*K`
  points; the binding variance is per-point MC noise, not perturbation heterogeneity.
* **A longer horizon is not needed.** Criterion D passes with a 20x margin.
* **A larger finite-difference step** would cut the gradient noise as `1/c^2` — `c =
  0.20` would give roughly a 4x reduction — but since criterion C already fails, this
  would have to be paired with an explicit bias check rather than adopted blind.

**None of this should be spent before the estimand is corrected to the clipped target
of §4.** Doing so requires a **new preregistration**; preregistration §11 states that
the pilot specification is not changed after seeing results and rerun under a new
definition, and that is not done here.

### Scoring the registered expectation

The preregistration §13 recorded a prediction so it could be scored. It was mostly
wrong, and in an instructive direction.

| Registered | Outcome |
|---|---|
| WML satisfies A comfortably | **correct** — `D` CI `[122, 777]` |
| WML satisfies B comfortably | **wrong** — B fails on WML |
| PW is the binding case | **correct** — PW fails A, B, C and E |
| E is the criterion that fails first | **wrong** — B fails first, and more fundamentally |
| C passes | **wrong** — C fails on both |
| D passes comfortably | **correct** — passes with a 20x margin |

I expected an interval-width problem and got a sign problem. I did not anticipate that
the whitened gradient energy would be swamped by MC noise at 8 rollouts per group, nor
that the `c = 0.05` arm would be effectively pure noise.

## 18. Limitations

1. **The preregistered estimand is not the quantity `Q_phi` was fit to** (§4). This is
   the most important limitation and it is my error. It leaves PW untouched and
   distorts WML over a third of its domain.
2. **The WML policy's scale has an extreme tail.** Median pre-squash `sigma` at the
   bank states is 1.97 against the training-logged 2.20 — close agreement — but the
   mean is 16.0 and the maximum 1575.9. Per-dimension means are
   `[6.31, 11.19, 46.17, 10.77, 7.18, 14.67]`, RMS 90.4, **anisotropy 51.7**. Per
   preregistration §12, **no scalar `sigma*omega` point is forced onto the isotropic
   `(sigma*omega, sqrt(d))` plane** for this checkpoint. PW is nearly isotropic
   (per-dimension means 0.51-0.59, RMS 0.561, anisotropy 1.75).
3. **`omega_RMS` is UNRESOLVED for both.** The unwhitened cross-product is negative
   (PW -15.23, WML -2.55e4), so the secondary scalar diagnostic is not reported as a
   number.
4. **The bank samples early-episode states only** (50-step burn-in from reset). The
   median `sigma` matches the training-logged value, so the states are not obviously
   off-distribution, but late-episode states are absent by construction.
5. **`Q_phi` is structurally confined to `(0, 150)`** by its 151-bin support, so
   measured error is bounded by the representation wherever the true value leaves that
   interval.
6. **The `lambda`-return truncation flag is off by one** (§3), affecting ~1/1000 of
   training transitions.
7. **Two checkpoints, one seed, one task.** Nothing here generalises across seeds,
   tasks or action dimensions, and no such generalisation is attempted.
8. **Bootstrap intervals on `r_RMS` are conditional** on `N > 0` and `D > 0` in the
   replicate; the conditioning fraction is reported beside every interval.
9. **Networks are evaluated in float32**, as in training. T7 reproduces the existing
   evaluator exactly, so this adds no discrepancy relative to the trainer.
10. **`H = 500` is truncation.** The correct term throughout is *finite-horizon
    Monte-Carlo estimate of soft `Q^pi`*, and no infinite-horizon bound is claimed.

## 19. Exact reproduction commands

```bash
cd ~/repos/reppo && git checkout 8fea5cf     # oracle code as run

# validation (phase-2 gate + phase-5); 18/18 expected
python scripts/analysis/test_mc_oracle.py \
    exports/WalkerRun_pathwise_fa_s301_final /hpcwork/$USER/mco/roundtrip.npz

# state bank (deterministic; sha256 64a96411621dcd65...)
python scripts/analysis/mc_oracle_walker.py states \
    exports/WalkerRun_pathwise_fa_s301_final \
    exports/WalkerRun_weighted_mle_s301_final /hpcwork/$USER/mco/bank.npz

# the pilot: 2 checkpoints x 64 states x 16 perturbations x 25 branches
#            x (8 + 8) rollouts, H = 500, c in {0.10, 0.05}
for CK in WalkerRun_pathwise_fa_s301_final WalkerRun_weighted_mle_s301_final; do
  python scripts/analysis/mc_oracle_walker.py pilot \
      exports/$CK /hpcwork/$USER/mco/bank.npz /hpcwork/$USER/mco/pilot_$CK.npz
  python scripts/analysis/mc_oracle_walker.py horizon \
      exports/$CK /hpcwork/$USER/mco/bank.npz /hpcwork/$USER/mco/horizon_$CK.npz
done

# analysis, precision decomposition, figure, packaging
D=/hpcwork/$USER/mco
python scripts/analysis/mc_oracle_analyse.py \
    $D/pilot_PW.npz $D/pilot_WML.npz $D/horizon_PW.npz $D/horizon_WML.npz \
    reports/artifacts
python scripts/analysis/mc_oracle_precision.py \
    $D/pilot_PW.npz $D/pilot_WML.npz reports/artifacts/mc_oracle_precision.json
python scripts/analysis/mc_oracle_figure.py \
    reports/artifacts/mc_oracle_results.json reports/artifacts/mc_oracle_boot.npz \
    reports/figures/fig_mc_oracle_pilot
python scripts/analysis/mc_oracle_package.py $D reports/artifacts/mc_oracle_pilot_raw.npz
```

SLURM, as run: `sbatch -p c25g -A rwth2182 --export=ALL,MCO_CMD="..." slurm/mc_oracle.sh`.

**Artifacts.** `reports/artifacts/mc_oracle_state_bank.npz` (+ `_manifest.json`),
`mc_oracle_pilot_raw.npz` (6.1 MB, sha256 `26a960b0d110cd94...`),
`mc_oracle_qpi_summary.csv`, `mc_oracle_error_summary.csv`,
`mc_oracle_gradient_summary.csv`, `mc_oracle_results.json`, `mc_oracle_boot.npz`,
`mc_oracle_precision.json`; figure `reports/figures/fig_mc_oracle_pilot.{pdf,png}`.

---

## What this pilot does and does not license

**Supported.** That the MC oracle estimates the *centred* learned-critic error with
sufficient precision on the WML-trained checkpoint (`D = 417.8`, CI `[122.5, 776.8]`)
and marginally insufficient precision on the PW-trained one (`D = 3.14`, CI
`[-0.021, 9.195]`). That it estimates the *whitened gradient* of that error with
insufficient precision on **both**. That the RMS error-frequency analogue for the
WML-trained critic lies near the boundary with an interval crossing 1, and is
unresolved for the PW-trained critic. That the two critics differ in their measured
centred error spectrum on the common WalkerRun state distribution. That horizon
truncation at `H = 500` is not a material source of error here.

**Not supported, and not claimed anywhere above.** That Claim 4 predicts which
training arm should win. That `r_RMS > 1` means WML should have higher return. That
the oracle explains the WalkerRun return gap or the g1 return gap. That this measures
`omega_infinity`. That centring eliminates truncation bias. That PW states are
neutral. That `Q^pi` is exact. Any action-dimension trend.

**STOP.** The all-16-checkpoint WalkerRun study is **not** launched, and per §17 it
should not be launched under the current estimand at all.
