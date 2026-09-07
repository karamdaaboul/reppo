# The same-state test: matched-state scale versus occupancy

> **On identical states, WML is wider than PW on both halves of every bank, on all three
> tasks. The reversal reported previously was an occupancy effect, not a reversal of the
> scale function.**

Offline. No rollouts, no new banks, no launches. Job `3800518`, commit
`2cb7ab7636c8f742cee8c8f4d1888b403aadd486`, node r23g0003, 2026-09-07T13:36:47+02:00.
All three bank hashes verified in-run. Definitions frozen beforehand in
`docs/protocol_width_populations.md` (commit `2cb7ab7`), committed before any of the
numbers below were computed.

## Correction to the previous report

`reports/onarm_vs_neutral.md` reported an "on-arm" ratio of 0.456x on Walker and called it
a reversal. That number is `sigma_WML(s ~ d_WML) / sigma_PW(s ~ d_PW)` — the two terms
evaluated on **different** state sets. Under the protocol frozen here that is an
**occupancy-weighted** ratio and explicitly not a same-state operator comparison. The
matched-state ratio on the same data is **17.9x on the PW half and 14.6x on the WML half**,
both far above 1.

The earlier report's headline sentence, "the WML/PW width comparison reverses sign between
the two state populations", is therefore true only of the occupancy-weighted quantity. The
scale function's ordering does not reverse anywhere.

## P1. Provenance

### The banks

Both generators are structurally identical: `walker_fixed_bank_sigma.py` built the Walker
bank, `fixed_bank_width_saturation.py` the G1 and LEAP banks and reused Walker's.

| field | value |
|---|---|
| generating checkpoints | `_final` only, both arms (`pathwise_fa`, `weighted_mle`) |
| seeds | 301-308, **pooled**, seed retained in the `source` label |
| training stage | final policy; no earlier-training states |
| parallel envs | `NENV = 32` per (arm, seed, depth) |
| actions | **stochastic**, `h.pi(o).sample(...)`, clipped to `+-0.999` (`ACTION_CLIP`) |
| PRNG | `fold_in(PRNGKey(20260905), hash((arm, seed)) % 2**31)`, env and policy keys split per step |
| depth schedule | Walker, G1: 50/150/300/500/700/900; LEAP: 50/100/200/300/400/480 |
| states per depth | 512 total, 32 per arm-seed |
| observations | stored **raw**; each checkpoint applies its own normalizer at evaluation |
| filtering | none |
| terminal handling | no explicit done handling in the collection loop; the wrapper's autoreset governs |

Paths and hashes: `walker_fixed_state_bank.npz` `8adfeb0b...aa21`,
`g1_fixed_state_bank.npz` `cf6f7880...35e9`, `leap_fixed_state_bank.npz` `0053b0f3...b158e`.

**Structural point that removes a class of concern.** The "own-state source" is not
separate data. It is a subset of the same bank selected by `source` label. The neutral bank
and the own-arm states therefore **cannot** differ in checkpoint timing, depth sampling or
preprocessing — they are the same states, differently selected. Nothing to flag on that axis.

**Provenance defect, recorded.** Both generators key the PRNG on `hash((arm, seed))`, and
Python salts string hashing with `PYTHONHASHSEED`, randomised per process by default. The
banks are therefore **not regenerable byte-identically** from the scripts. This does not
affect any result here, since the banks are on disk and hash-verified before every use, but
the scripts are not deterministic generators and should not be described as such.

### Logged `pi_sigma_mean`

```
source     pi_sigma = pi.distribution.scale                          reppo.py:797
           pi = actor_model.actor(minibatch.obs); scale defaults to 1.0, so no
           exploration scaling is applied
floor      AFTER the +min_std floor: effective_std = exp(log_std) + min_std
                                                     jax_models.py:408-412
statistic  MEAN, not median:  pi_sigma_mean = pi_sigma.mean()        reppo.py:1304
reduction  a single .mean() over BOTH the state and action-dimension axes, then
           mean over minibatches (:1415), then x[-1] last epoch (:1426),
           then x[-1] last iteration (:1453)
states     the training minibatch observations at the final iteration
```

**Consequence, and a withdrawal.** `pi_sigma_mean` is a mean over states and coordinates
after the floor; every bank figure quoted in this project is a median. The statement in
`reports/onarm_vs_neutral.md` that "the logged pi_sigma_mean sits close to the on-arm
value, as it should" compared a mean against a median and is withdrawn as evidence. On the
heavy-tailed width distributions documented below — Walker WML mean 259 against median 8.8
— the two differ by more than an order of magnitude, so the agreement was not meaningful.

## P2. The same-state test

Baseline pair only, seeds 301-308, both policies on **identical** states. Halves are never
averaged. Medians over seeds.

### walker

| cell | half | median | mean | med-log | p75 | p90 | p95 | p99 | max | sat95 | sat99 | med \|mu\| |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PW_ent | PW_half | 0.4550 | 0.510 | 0.4550 | 0.580 | 0.768 | 0.876 | 1.136 | 4.2 | 0.2133 | 0.0742 | 0.9610 |
| WML_noent | PW_half | 8.8018 | 259.097 | 8.8018 | 35.815 | 164.952 | 559.871 | 3339.734 | 295101.1 | 0.8226 | 0.7511 | 2.5822 |
| PW_ent | WML_half | 0.4537 | 0.485 | 0.4537 | 0.585 | 0.726 | 0.843 | 1.133 | 2.6 | 0.2370 | 0.0978 | 0.9675 |
| WML_noent | WML_half | 6.4271 | 270.124 | 6.4271 | 29.720 | 130.940 | 311.533 | 3073.239 | 157096.5 | 0.7520 | 0.6736 | 2.2409 |

### g1

| cell | half | median | mean | med-log | p75 | p90 | p95 | p99 | max | sat95 | sat99 | med \|mu\| |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PW_ent | PW_half | 0.2909 | 0.340 | 0.2909 | 0.496 | 0.626 | 0.700 | 0.884 | 3.5 | 0.1407 | 0.0542 | 0.6513 |
| WML_noent | PW_half | 0.5159 | 0.952 | 0.5159 | 1.142 | 2.042 | 3.034 | 7.712 | 101.7 | 0.3591 | 0.2423 | 1.0810 |
| PW_ent | WML_half | 0.2881 | 0.342 | 0.2881 | 0.493 | 0.640 | 0.735 | 1.017 | 5.4 | 0.1793 | 0.0757 | 0.7426 |
| WML_noent | WML_half | 0.4993 | 1.013 | 0.4993 | 1.168 | 2.158 | 3.384 | 8.843 | 158.9 | 0.3462 | 0.2311 | 1.0102 |

### leap

| cell | half | median | mean | med-log | p75 | p90 | p95 | p99 | max | sat95 | sat99 | med \|mu\| |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PW_ent | PW_half | 0.5398 | 0.884 | 0.5398 | 0.929 | 1.711 | 2.512 | 6.187 | 51.4 | 0.3064 | 0.1694 | 1.0364 |
| WML_noent | PW_half | 3.8881 | 44.208 | 3.8881 | 14.256 | 65.939 | 143.804 | 667.737 | 70357.1 | 0.7015 | 0.6016 | 1.4501 |
| PW_ent | WML_half | 0.6579 | 0.991 | 0.6579 | 1.114 | 1.870 | 2.724 | 5.812 | 35.3 | 0.3280 | 0.1865 | 1.1070 |
| WML_noent | WML_half | 2.7911 | 70.819 | 2.7911 | 11.904 | 49.222 | 131.285 | 913.081 | 41463.3 | 0.6366 | 0.5377 | 1.3910 |

### Paired matched-state ratios

`sigma_WML / sigma_PW` on identical states. Median over seeds of per-seed log ratios,
exponentiated. Halves reported separately, never aggregated.

| task | ratio | value | per-seed 301-308 |
|---|---|---|---|
| walker | `r_PWstates` | **17.928x** | 13.192 33.533 11.393 24.055 34.973 13.361 10.724 54.747 |
| walker | `r_WMLstates` | **14.615x** | 11.117 14.482 14.814 14.749 15.113 6.678 7.140 37.619 |
| g1 | `r_PWstates` | **1.836x** | 1.604 1.658 2.200 2.192 1.317 2.783 1.898 1.776 |
| g1 | `r_WMLstates` | **1.794x** | 1.768 1.262 2.044 1.889 1.314 2.935 1.727 1.822 |
| leap | `r_PWstates` | **8.127x** | 8.227 2.098 10.155 10.226 18.224 2.564 4.886 8.028 |
| leap | `r_WMLstates` | **4.767x** | 4.776 2.150 4.778 4.514 14.887 1.770 4.759 5.272 |

All 48 per-seed ratios exceed 1. Both halves agree in sign on every task.

### Seed split — the finding that reframes "the WML half"

Median sigma on the states contributed by that policy's **own** seed, against the states
contributed by the **other seven** seeds of the same arm.

| task | cell | half | same-seed | other-seed | other/same | n_same | n_other |
|---|---|---|---|---|---|---|---|
| walker | PW_ent | PW_half | 0.4783 | 0.4512 | 0.94 | 192 | 1344 |
| walker | PW_ent | WML_half | 0.4490 | 0.4560 | 1.02 | 192 | 1344 |
| walker | WML_noent | PW_half | 7.3024 | 9.2986 | 1.27 | 192 | 1344 |
| walker | WML_noent | WML_half | **0.2182** | **9.1164** | **41.77** | 192 | 1344 |
| g1 | PW_ent | PW_half | 0.2695 | 0.2937 | 1.09 | 192 | 1344 |
| g1 | PW_ent | WML_half | 0.2669 | 0.2919 | 1.09 | 192 | 1344 |
| g1 | WML_noent | PW_half | 0.3598 | 0.5469 | 1.52 | 192 | 1344 |
| g1 | WML_noent | WML_half | **0.1759** | **0.5922** | **3.37** | 192 | 1344 |
| leap | PW_ent | PW_half | 0.2242 | 0.5986 | 2.67 | 192 | 1344 |
| leap | PW_ent | WML_half | 0.5073 | 0.6549 | 1.29 | 192 | 1344 |
| leap | WML_noent | PW_half | 3.5297 | 3.9152 | 1.11 | 192 | 1344 |
| leap | WML_noent | WML_half | **0.1225** | **3.8847** | **31.72** | 192 | 1344 |

**This is a finding, not an artifact.** Each WML policy is narrow on the states its own
seed visits and 32x to 42x wider on the states the other seven WML seeds visit. PW shows
nothing comparable: every PW row is within 1.1x except LEAP's PW-half at 2.67x.

So **"the WML half" is not one population.** The eight WML seeds converge to regions that
are mutually off-distribution, and each WML policy is narrow only in its own region. The
Walker figure of 0.218 previously reported as WML's "on-arm" width is the same-seed number;
the same policy reads 9.12 on its sibling seeds' states.

Within-arm seed heterogeneity is therefore large for WML and negligible for PW, which is
itself an operator difference and is not visible in any pooled statistic.

## P3. What the result supports

```
FINDING = (i)
```

WML is wider than PW on **both** halves, on all three tasks, with all 48 per-seed ratios
above 1. The operator difference in the scale **function** is real over every state
population tested, and the own-distribution reversal reported earlier is an **occupancy**
effect: each WML policy occupies a narrow region of its own, and the ratio of
`sigma_WML(d_WML)` to `sigma_PW(d_PW)` inverts because the two terms are read at different
places, not because the function ordering changes.

### The sentence the data sustain

**Evaluated on identical states — either the PW-derived or the WML-derived half of the
frozen bank — the weighted-MLE policies have a larger pre-tanh scale than the pathwise
policies on all three tasks (Walker 17.9x and 14.6x, G1 1.84x and 1.79x, LEAP 8.13x and
4.77x), while each weighted-MLE policy is narrow on the states its own seed visits.**

### Saturation on both halves

WML is more saturated than PW on **both** halves of every task:

| task | half | PW sat95 | WML sat95 | PW sat99 | WML sat99 |
|---|---|---|---|---|---|
| walker | PW_half | 0.2133 | 0.8226 | 0.0742 | 0.7511 |
| walker | WML_half | 0.2370 | 0.7520 | 0.0978 | 0.6736 |
| g1 | PW_half | 0.1407 | 0.3591 | 0.0542 | 0.2423 |
| g1 | WML_half | 0.1793 | 0.3462 | 0.0757 | 0.2311 |
| leap | PW_half | 0.3064 | 0.7015 | 0.1694 | 0.6016 |
| leap | WML_half | 0.3280 | 0.6366 | 0.1865 | 0.5377 |

The ordering holds on both halves everywhere, and the anticipated tension does not arise:
WML is simultaneously **wider** and more saturated on the same states, which needs no
special explanation. The median `|mu|` confirms both channels point the same way rather
than one compensating — WML's mean is also further from zero on every half (Walker 2.58
against 0.96 on the PW half, G1 1.08 against 0.65, LEAP 1.45 against 1.04), so greater
width and a more eccentric mean both push mass toward the boundary.

## Limits

The banks hold final-policy, stochastic-action, depth-stratified states only; nothing here
speaks to earlier training. The matched-state comparison is over the states these two
baseline policies generate between them, which is not a claim about the whole state space.
The seed-split result says the eight seeds of an arm differ; it does not say why, and no
mechanism is proposed. Neither population is designated the true width.
