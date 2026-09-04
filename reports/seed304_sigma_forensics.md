# Forensic audit: WalkerRun WML-32 seed 304 sigma spike

Read-only. No training, no source modification, no rerun.

## 1. Provenance

```
RUN_DIRECTORY        = /hpcwork/qzi10910/reppo_runs/outputs/faithful_repair/walker_WML32_s304
RUN_GIT_SHA          = 1c6259ef959400eba617e1d1392bfaf58248688c
RESOLVED_CONFIG_HASH = 420dfe1d473f9b21   (from the run-local .hydra/config.yaml)
SEED                 = 304
TASK                 = WalkerRun
ARM                  = WML-32-faithful-repair
TREE_CLEAN           = YES
```
SLURM `3444831_13`, NVIDIA H100, 1380 s, status `completed`, checkpoint sha256
`731e9e5688135d1b...`. Guards recorded in the run config: `min_std` absent from the
config (it is the network constructor default 0.1), `freeze_sigma` absent,
`sqrt_rho` absent -- none of those fields existed at `1c6259e`.

## 2. What the plotted scalar means

`train/pi_sigma_mean`, from `pi_sigma = pi.distribution.scale` (`src/jaxrl/reppo.py:785`),
logged at `:1267`.

| | |
|---|---|
| raw tensor shape | `(1024, 6)`, **numel 6144** -- measured, not assumed |
| sigma formula | `exp(log_std) + min_std`, an additive floor (`src/networks/jax_models.py:425`) |
| `min_std` | `0.1` (`jax_models.py:336`) |
| state reduction | mean over the 1024 minibatch states |
| action-coordinate reduction | mean over the 6 coordinates -- the same `.mean()` |
| minibatch reduction | mean over the 128 minibatches (`reppo.py:1378`) |
| epoch reduction | **last epoch only** (`:1389`, `x[-1]`) |
| iteration reduction | **last outer iteration only** (`:1416`, `x[-1]`) |
| logging frequency | 21 evaluations, every 19 outer iterations = 2,490,368 env steps |

So `3452.77` is a **mean over states and coordinates**, not a maximum and not a mean
of `exp(log_std)`, taken from the last epoch of the last outer iteration before that
evaluation.

## 3. The spike is in the raw log

All 21 points, seed 304:

| eval | env step | sigma mean | sigma min | sigma max | KL | ESS | eta | lag_eff | gate open | return |
|---|---|---|---|---|---|---|---|---|---|---|
| 11 | 27,394,048 | 2.63 | 0.1000 | 8,449 | 0.1021 | 20.41 | 0.0591 | 0.337 | 0.581 | 707.96 |
| 12 | 29,884,416 | 4.95 | 0.1000 | 38,578 | 0.1006 | 21.29 | 0.0477 | 0.340 | 0.577 | 699.44 |
| 13 | 32,374,784 | 17.81 | 0.1000 | 177,746 | 0.1052 | 24.14 | 0.0781 | 0.115 | 0.588 | 693.60 |
| 15 | 37,355,520 | 8.48 | 0.1000 | 72,311 | 0.1071 | 22.66 | 0.0786 | 0.168 | 0.566 | 706.39 |
| 16 | 39,845,888 | 44.78 | 0.1000 | 510,445 | 0.1050 | 21.56 | 0.0538 | 0.217 | 0.569 | 729.35 |
| 17 | 42,336,256 | 340.65 | 0.1000 | 3,693,942 | 0.1112 | 21.01 | 0.0821 | 0.220 | 0.541 | 747.59 |
| 18 | 44,826,624 | 297.50 | 0.1000 | 3,545,939 | 0.1014 | 21.83 | 0.0984 | 0.219 | 0.588 | 741.86 |
| **19** | **47,316,992** | **3452.77** | **0.1000** | **42,358,900** | **0.1087** | **21.89** | **0.0856** | **0.235** | **0.558** | **737.50** |
| 20 | 49,807,360 | 477.30 | 0.1000 | 5,618,557 | 0.1109 | 21.37 | 0.1190 | 0.232 | 0.561 | 737.71 |
| 21 | 52,297,728 | 1.22 | 0.1000 | 4,512 | 0.1076 | 19.35 | 0.0248 | 0.340 | 0.530 | 655.47 |

No duplicate steps, no non-finite values in sigma or KL, stored `float32 (21,1)`, a
single contiguous run. **The rise is not a single-point glitch**: `pi_sigma_max`
grows monotonically from eval 3 (35.6) through eval 19 (4.24e7) and then falls.

```
RAW_LOG_SPIKE_CONFIRMED = YES
```

## 4. An internal inconsistency in the logged pair, confined to this seed

For one tensor of `N = 6144` positive entries, `max/mean <= N` necessarily. Across
all 8 WML seeds x 21 evaluations (168 points) and all 8 PW seeds (168 points),
**exactly 8 points violate this, all of them seed 304**, evals 12, 13, 15, 16, 17,
18, 19, 20, with `max/mean` from 7,788 to **12,268**. Zero violations elsewhere;
the PW maximum ratio is 5.4.

Both scalars are reductions of the same `pi_sigma` bound once at `:785`, and both
pass through the identical reduction chain, so the pair cannot both be faithful.
The discrepancy is close to a factor of two. **This is not explained here.** It
means the *numerical magnitude* of `pi_sigma_mean` at those eight points should not
be taken at face value. It does not affect the qualitative finding below, which
rests on checkpoint weights rather than on the logged scalar.

## 5. Per-coordinate structure, from existing checkpoints

Checkpoints exist at 12.45M (`p25`), 24.90M (`p50`) and 52.30M (`final`) env steps.
The spike is at 47.32M, so:

```
SPIKE_TENSOR_STRUCTURE_NOT_RECOVERABLE   (no checkpoint at or near 47.3M)
```

The final checkpoint is **post-collapse** and is not substituted for the spike. The
three available checkpoints bracket it and show a monotone trend. Evaluated on 1024
states from `reports/artifacts/cd_bank_walker_corrected.npz`:

| checkpoint | mean | geo-mean | median | max | max/mean |
|---|---|---|---|---|---|
| p25, 12.45M | 8.05 | 4.71 | 4.64 | 319 | 39.7 |
| p50, 24.90M | 32.52 | 8.50 | 8.31 | 28,613 | 879.7 |
| final, 52.30M | 604.63 | 3.70 | **2.95** | 1,146,363 | 1896.0 |

Per-coordinate mean:

| checkpoint | c0 | c1 | c2 | c3 | c4 | c5 |
|---|---|---|---|---|---|---|
| p25 | **14.22** | 3.59 | 10.26 | 6.98 | 6.97 | 6.28 |
| p50 | **105.36** | 8.45 | 37.83 | 11.20 | 15.90 | 16.41 |
| final | **3559.40** | 3.90 | 38.21 | 7.22 | 8.95 | 10.07 |

**Coordinate 0 diverges progressively while coordinates 1-5 stay bounded.** At the
final checkpoint coordinate 0's maximum is 1,146,363 (`log_std = 13.95`) against
207-2306 for the others. Crucially the **median falls** from 8.31 to 2.95 while the
mean rises 32.5 -> 604.6: the bulk of the policy contracts while a narrow tail in
one coordinate explodes.

Note the state-distribution caveat: these numbers are on the frozen bank, whereas
the logged scalar uses the live training minibatch. They are not expected to match
in magnitude, and they do not (logged final mean 1.22 against 604.63 here). The
*structure* -- one coordinate dominating -- is what is being read.

```
DOMINANT_ACTION_COORDINATE(S) = coordinate 0
```

Figure: `reports/figures/fig_s304_sigma_percoord.png`.

## 6. Diagnostics at the spike, and against the other seven seeds

Means over evals 16-20 (39.8M-49.8M):

| quantity | seed 304 | other 7 median | ratio |
|---|---|---|---|
| sigma mean | 922.60 | 2.92 | **316** |
| sigma max | 11,145,556 | 2,656 | **4196** |
| KL | 0.1075 | 0.1038 | **1.04** |
| ESS | 21.53 | 20.55 | **1.05** |
| eta | 0.0878 | 0.0388 | 2.26 |
| lag_eff | 0.2245 | 0.3885 | 0.58 |
| gate open | 0.5633 | 0.5729 | **0.98** |

**The KL, the gate fraction and the ESS are indistinguishable from the other
seeds.** No gate saturation, no ESS concentration, no NaN or Inf, no actor-loss
explosion, and the evaluation return is *higher* than the other seeds through the
spike window (737-748). Only `eta` (2.3x higher) and `lag_eff` (0.58x) differ, and
those are duals responding to the policy, so no causal direction is claimed.

The trust region reports a normal step size of ~0.107 while one coordinate's sigma
reaches 4e7 -- consistent with the accumulation mechanism recorded in
`reports/mechanism_followup.md`: the reference resets each outer iteration, so
sustained near-budget drift compounds across iterations.

**All eight WML seeds show the same heavy tail** (`sigma_max` 1,508-5,249 against
means near 3). Seed 304 is an extreme instance of a pattern present in every WML
seed, not a unique event.

## 7. Parameterization

`std = exp(log_std) + min_std` with **no upper bound anywhere**: no `log_std`
clipping, no sigma ceiling, no regularizer, no numerical guard. The only guard is
the additive lower floor. `sigma = 4.24e7` needs `log_std = 17.56`, and
`sigma = 1.15e6` at the final checkpoint corresponds to the measured
`log_std = 13.95`. Both are far inside float32 range, so this is a **mathematically
reachable parameter value, not a numerical overflow**.

## 8. Classification

```
B. REAL, LOCALIZED TO ONE/FEW ACTION COORDINATES
```

Real (Section 3), localized to coordinate 0 (Section 5), not a broad covariance
explosion (median falls while mean rises), and not a plotting artifact -- though
the logged mean's magnitude at eight points is internally inconsistent (Section 4)
and the spike-instant tensor is not recoverable.

**No causal claim is made about the return.** Seed 304 (655.5) and seed 308
(653.7) have nearly identical final returns with maximum sigma differing by three
orders of magnitude, and 304's return was *above* the cohort during the spike.

```
SHOULD_SEED_304_BE_RERUN_FOR_DEBUGGING = NO
```

The trajectory is genuine and independently confirmed from checkpoint weights, and
a genuine extreme trajectory is not grounds for replacing a seed. The one
unexplained item is the logged mean/max inconsistency, which lives in the metric
reduction rather than the training path; the run is deterministic, so an identical
rerun would reproduce the same logs and diagnose nothing. Diagnosing that requires
reproducing the reduction offline on a synthetic tensor with the same dynamic
range, or a short instrumented run -- neither of which is a rerun of seed 304.
