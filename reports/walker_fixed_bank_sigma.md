# Walker policy width on one neutral fixed state bank

Read-only. No training, no rerun, no source modification. Rollouts collect states
only; no parameter is updated.

## 1. Provenance of the original 1024-state bank — and a flaw it exposed

The seed-304 per-coordinate figure in `reports/seed304_sigma_forensics.md` used
`reports/artifacts/cd_bank_walker_corrected.npz[:1024]`. That bank is written
arm-major: policies are appended `for arm in (PW, WML): for seed in 301..308`, 128
states each. Verified directly:

```
first 1024 states -> sources PW-s301 ... PW-s308      arms present: ['PW']
last  1024 states -> sources WML-s301 ... WML-s308    arms present: ['WML']
```

**The 1024 states were drawn entirely from PW policies.** A WML checkpoint was
evaluated on states visited only by PW policies. Task WalkerRun; arm PW (all eight
seeds); checkpoints the eight corrected PW finals; sampling: 128 states per policy
after `burn_in = 50` stochastic steps from `env.reset`, RNG root 20260904; episode
depth **uniformly 50** — a single early-episode depth, not a distribution.

That is exactly the confound this audit removes. The coordinate-0 conclusion in
that report survives (Section 4 below), but it was reached on an arm-biased,
single-depth bank and should be read through the fixed-bank numbers.

## 2. Logging reduction, and the `max/mean > 6144` inconsistency

Traced in the source **as it ran at `1c6259e`**, which is byte-identical to HEAD in
this block: `pi_sigma = pi.distribution.scale` (`:739`), `pi_sigma_mean/min/max` at
`:1167-1169`, `metrics.mean(0)` over minibatches (`:1248`), `x[-1]` for epochs
(`:1259`), `x[-1]` for outer iterations (`:1286`).

| stage | mean | max |
|---|---|---|
| states | mean over 1024 | max over 1024 |
| action coordinates | mean over 6 (same `.mean()`) | max over 6 (same `.max()`) |
| minibatches | mean over 128 | mean over 128 |
| epochs | last epoch only | last epoch only |
| outer iterations | last iteration only | last iteration only |

Measured, not assumed: `pi.distribution.scale` from the actual training call is
`(1024, 6)`, size **6144**, `float32`.

**This does not explain the violation.** For positive entries `max/mean <= numel`
holds per minibatch and is preserved by averaging, since both scalars pass through
the identical chain. Tested numerically: a `(6144,)` float32 array with one entry
at `4.24e7` gives `max/mean = 6143.9` under big-first, big-last and shuffled
reduction orders, and at `4.24e9` gives `6144.0`. No ordering violates it.

Decisive arithmetic on the eval-19 pair:

```
logged mean 3452.7715  ->  implied sum = 3452.7715 x 6144 = 2.1214e7
logged max                                              = 4.2359e7
a single positive entry cannot exceed the sum of all entries:  FALSE
```

```
LOG_REDUCTION_EXPLAINED    = NO
VIOLATING_POINT_REPRODUCED = NO
```

The underlying per-minibatch tensors were never stored, so the arithmetic cannot be
reproduced from the run's own values. What would settle it: a short instrumented
run logging both scalars per minibatch, or persisting the raw per-minibatch array.
Until then the **magnitude** of `pi_sigma_mean` at those eight seed-304 points is
not trustworthy. Nothing below depends on it.

## 3. The fixed neutral bank

```
STATE_BANK_PATH = reports/artifacts/walker_fixed_state_bank.npz
NUM_STATES      = 3072
PROVENANCE      = all 16 corrected Walker policies (PW and WML, seeds 301-308),
                  32 environments each, states snapshotted at 6 episode depths,
                  RNG root 20260905, one deterministic key per policy
ARM BALANCE     = PW 1536 states, WML 1536 states  (balanced by construction)
EPISODE_DEPTH_DISTRIBUTION = 50:512  150:512  300:512  500:512  700:512  900:512
SHA256          = 8adfeb0bf70bddcdbd64a84b972b4dbd62617c64e1c948589a7adc30cf64aa21
```

**All 16 checkpoints are evaluated on this identical bank.** No policy is evaluated
only on its own states. This is proposed as the canonical bank for later Walker
policy probes.

## 4. All 16 final checkpoints on the identical bank

| arm | seed | median | mean | p95 | max |
|---|---|---|---|---|---|
| PW | 301 | 0.4836 | 0.5196 | 0.9099 | 4.8 |
| PW | 302 | 0.4484 | 0.4752 | 0.8209 | 2.0 |
| PW | 303 | 0.4678 | 0.5166 | 0.8927 | 23.7 |
| PW | 304 | 0.4472 | 0.4724 | 0.8271 | 2.2 |
| PW | 305 | 0.4623 | 0.5386 | 0.8831 | 24.4 |
| PW | 306 | 0.4515 | 0.4771 | 0.8357 | 3.6 |
| PW | 307 | 0.4358 | 0.4656 | 0.8288 | 2.7 |
| PW | 308 | 0.4422 | 0.5165 | 0.9548 | 11.5 |
| WML | 301 | 5.8720 | 47.49 | 135.79 | 10,511 |
| WML | 302 | 9.7079 | 439.91 | 691.81 | 1,625,928 |
| WML | 303 | 5.9475 | 1229.55 | 444.66 | 4,270,590 |
| WML | 304 | 8.6717 | 8265.94 | 582.61 | 37,527,200 |
| WML | 305 | 10.9482 | 196.81 | 481.47 | 178,271 |
| WML | 306 | 4.4509 | 43.97 | 111.61 | 102,314 |
| WML | 307 | 3.8256 | 51.66 | 136.12 | 28,648 |
| WML | 308 | 20.0716 | 1079.33 | 2340.43 | 733,732 |

Per-coordinate statistics for all 16 checkpoints are in
`reports/artifacts/walker_fixed_bank_sigma.json`.

## 5. Paired differences

Whole-policy median sigma, `delta = median(WML) - median(PW)`:

| seed | PW median | WML median | delta | ratio |
|---|---|---|---|---|
| 301 | 0.4836 | 5.8720 | +5.3884 | 12.1x |
| 302 | 0.4484 | 9.7079 | +9.2595 | 21.7x |
| 303 | 0.4678 | 5.9475 | +5.4797 | 12.7x |
| 304 | 0.4472 | 8.6717 | +8.2245 | 19.4x |
| 305 | 0.4623 | 10.9482 | +10.4859 | 23.7x |
| 306 | 0.4515 | 4.4509 | +3.9994 | 9.9x |
| 307 | 0.4358 | 3.8256 | +3.3898 | 8.8x |
| 308 | 0.4422 | 20.0716 | +19.6293 | 45.4x |

Per action coordinate, paired median difference:

| coord | s301 | s302 | s303 | s304 | s305 | s306 | s307 | s308 | n_pos |
|---|---|---|---|---|---|---|---|---|---|
| 0 | +5.86 | +18.90 | +6.62 | +23.87 | +17.40 | +2.32 | +3.60 | +31.55 | 8/8 |
| 1 | +3.89 | +6.04 | +1.77 | +3.92 | +4.22 | +4.55 | +1.71 | +11.48 | 8/8 |
| 2 | +12.70 | +16.34 | +11.48 | +41.48 | +18.82 | +11.65 | +9.82 | +256.66 | 8/8 |
| 3 | +5.94 | +5.27 | +4.94 | +5.06 | +9.22 | +3.92 | +1.55 | +10.04 | 8/8 |
| 4 | +2.46 | +6.23 | +2.86 | +7.20 | +5.51 | +1.75 | +3.16 | +6.14 | 8/8 |
| 5 | +6.96 | +10.33 | +19.78 | +5.56 | +26.27 | +6.68 | +13.10 | +27.37 | 8/8 |

**All 48 paired differences are positive.**

## 6. The frozen rule

Fixed before the results were inspected:

1. whole-policy median larger for WML in at least 7 of 8 seed pairs -> **8/8, met**;
2. at least 5 of 6 coordinates positive in a majority of seed pairs -> **6/6, met**
   (each coordinate is positive in 8 of 8, not merely a majority).

```
VERDICT: BROAD ARM-LEVEL WML WIDENING SUPPORTED
```

The rule was not modified after seeing the results, and the raw paired table above
stands independently of the threshold.

## 7. Answers

**Is the typical pre-tanh width systematically larger for WML than PW on identical
states?** **Yes.** On the same 3072 states, WML's median sigma is 8.8x to 45.4x
PW's in every seed pair, and every coordinate is positive in every pair. This is a
statement about the **median**, so it is not driven by the seed-304 tail: excluding
seed 304 entirely leaves 7/7 pairs and 6/6 coordinates unchanged in sign.

**Does WML also produce heavier tails?** **Yes, separately and far more extremely.**
WML maxima span 10,511 to 37,527,200 against PW's 2.0 to 24.4 — six to seven orders
of magnitude. The two findings are independent: the median result would hold with
every tail truncated.

**Reconciliation with the seed-304 forensics.** On this neutral bank, coordinate 0
still owns the extreme tail for WML seed 304 (max 3.75e7 against 377 to 184,501 for
the others), so that report's localisation conclusion survives. But coordinate **2**
carries the largest typical shift (+41.5 median for s304, +256.7 for s308), which
the PW-only bank did not show. Tail-dominant and median-dominant coordinates are
not the same coordinate.

## 8. Restrictions observed

Seed 304 was not rerun. No training was started. The eta-sigma feedback hypothesis
was not tested. **No claim is made that widening causes the return gap** -- this
document reports width only.
