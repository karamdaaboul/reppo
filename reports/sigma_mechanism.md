# Policy width in the corrected runs: what sigma is, and what it does

**Dated 2026-09-04T17:39:02+02:00.** Descriptive. Read-only on all artifacts; no training, no source
modification. Source SHA `b48a6ed` (`src/` identical at `918f82c`).

---

## 1. What `sigma` is, established from source before anything was plotted

| Question | Answer | Source |
|---|---|---|
| policy distribution | `distrax.Transformed(distrax.Normal(loc, std), distrax.Tanh())` | `src/networks/jax_models.py:436-438` |
| pre- or post-tanh | `sigma` is the scale of the **pre-tanh Normal**; the tanh is the bijector applied after | `:437`, and `pi.distribution.scale` at `src/jaxrl/reppo.py:785` |
| parameterisation | `sigma = exp(log_std) + min_std`, an **additive floor, not a clip** | `jax_models.py:425` (`effective_std`) |
| `min_std` | **0.1** | `jax_models.py:336` |
| state-dependent | **yes** -- `loc, log_std = split(actor_module(obs))` | `jax_models.py:434-435` |
| action-coordinate-dependent | **yes** -- `log_std` has one component per action dimension | `jax_models.py:435` |
| reduction of the logged scalar | `pi_sigma_mean = pi.distribution.scale.mean()`, a mean over **both** the minibatch and the action coordinates | `reppo.py:785, :1267` |

`freeze_sigma = null` and `sqrt_rho = 1.0` in every primary arm, so both are exact
no-ops and the learned path above is what ran.

**Consequence for reading the plots.** A single `pi_sigma_mean` value averages over
states and over coordinates, so it cannot distinguish "every coordinate widened a
little" from "one coordinate widened enormously". `pi_sigma_max` is logged
separately and is used below where that distinction matters.

## 2. Trajectories

Seeds were **discovered from the run directories**, not assumed:
all six groups are seeds **301-308**, n=8 each. Script
`scripts/analysis/sigma_trajectories.py`, artifact
`reports/artifacts/sigma_trajectories.json`, figures
`reports/figures/fig_sigma_{walker,g1,leap}.png`.

"initial" is the **first logged evaluation** (~iteration 19 of 399), not the
network initialisation.

| task | arm | initial | final | final/initial | max | final-20% mean | ratio range over seeds |
|---|---|---|---|---|---|---|---|
| walker | PW | 0.6447 | 0.5251 | **0.828** | 0.656 | 0.523 | [0.618, 0.980] |
| walker | WML | 0.6908 | 2.7357 | **4.375** | **434.64** | **116.76** | [1.099, 7.153] |
| g1 | PW | 1.1189 | 0.3163 | **0.283** | 1.119 | 0.324 | [0.266, 0.310] |
| g1 | WML | 1.2240 | 0.3922 | **0.321** | 1.224 | 0.399 | [0.285, 0.359] |
| leap | PW | 0.9650 | 0.2754 | **0.285** | 0.965 | 0.284 | [0.225, 0.390] |
| leap | WML | 0.9697 | 0.5136 | **0.529** | 0.970 | 0.450 | [0.417, 0.815] |

### Per task

**WalkerRun.** The only task with **absolute widening**. PW contracts to 0.83 of
its first-evaluation width; WML expands to 4.4x, with a maximum of **434.6** and a
final-20% mean of **116.8** against a final value of 2.74 -- so the late trajectory
is highly non-monotone, with excursions orders of magnitude above where it ends.
Every WML seed has ratio > 1 (range 1.10 to 7.15); no PW seed does.

**G1JoystickFlatTerrain.** **Both arms contract**, and by similar amounts: PW to
0.283, WML to 0.321. No widening in either arm; the maximum equals the initial
value for both, i.e. width falls monotonically from the first evaluation.

**LeapCubeRotateZAxis.** **Both arms contract.** PW to 0.285, WML to 0.529 -- WML
contracts about half as much. Maximum equals initial for both.

**Across-task pattern.** WML's final/initial ratio exceeds PW's on all three tasks
(4.375 vs 0.828; 0.321 vs 0.283; 0.529 vs 0.285), and the seed ranges do not
overlap on Walker or LEAP. But **only Walker shows widening in absolute terms.**

No threshold for "substantial widening" is defined here, and none is used.

## 3. Mechanism: three separate questions

### 3.1 Does WML widen more than PW?

**Yes, descriptively, on all three tasks** -- WML's final/initial width ratio is
larger in every case, with non-overlapping seed ranges on Walker and LEAP. **But
"more" means "contracts less" on G1 and LEAP**; absolute widening occurs only on
Walker. Reporting this as "WML widens the policy" without the task qualifier would
be wrong for two of the three tasks.

### 3.2 Does the WML weighting mechanism *cause* covariance expansion?

**UNTESTED for these tasks.** This requires an intervention, and none exists:

* The committed weight-shuffle specification is **Probe 5** of
  `docs/prospective_padding_error_field_analysis.md` (commit `e69532a`, sha256
  `34dd111af742750c...`). Its Section 1 locks coordinates as `y = (x, z)` with
  `x in R^6` **real** and `z in R^k` **padded**, and Probe 5's outcome is
  *"Per-coordinate covariance; residual beyond `1 - 1/ESS`"* with the prediction
  *"shuffling preserves generic **padded** shrinkage and removes extra **real**
  contraction."* Its primary outcome is defined **only** on a real-versus-padded
  coordinate split.
* Walker (d=6, unpadded), G1 (d=29) and LEAP (d=16) have **no padded z-block**, so
  the specified outcome is undefined for them.

```
DOES_THE_COMMITTED_SPEC_COVER_WALKER_G1_LEAP = NO
```

The shuffle was therefore **not executed**. Reinterpreting a padded-coordinate
experiment as a Walker/G1/LEAP experiment would be a scope violation, and the plan's
own Section 4 rules out related over-reach ("No claim that inert padding isolates
raw action-dimension scaling").

A covariance-freeze intervention exists in the code (`freeze_sigma`, verified
default-off and byte-identical when unset) and its Phase 1 is reported in
`reports/covariance_freeze_phase1.md`, which states that **no confirmatory
frozen-sigma run exists and no freeze value has been chosen.** So the causal
question has an available instrument but no executed experiment.

### 3.3 Does covariance expansion explain the return differences?

**UNTESTED, and the descriptive pattern does not settle it.** The only task with
absolute widening (Walker) is also the task with by far the largest PW-minus-WML
return gap (+135.58, `reports/corrected_operator_replication.md`). That is
suggestive. It is not evidence of causation, for three reasons:

1. n = 3 tasks, no intervention, no randomisation over the proposed mediator.
2. The ordering is not even monotone: G1's gap (+9.51) exceeds LEAP's (+4.30,
   `Inconclusive`), yet G1's WML ratio (0.321) is **lower** than LEAP's (0.529).
3. The crossed frozen-critic dispersion gate already returned
   `REFUTED-AS-CRITIC-SOURCE` (`reports/crossed_dispersion_walkerrun_gate.md`),
   attributing the operator difference on Walker to **policy width rather than
   critic source** -- which makes width a live candidate mediator but says nothing
   about the direction of causation between width and return.

**The causal link between the WML weighting mechanism, policy width, and return
remains untested for Walker, G1 and LEAP.**

## 4. Severity

No new S3 or S4. Section 1 is a definitional clarification the manuscript should
adopt whenever it reports a sigma number: the logged scalar is a pre-tanh scale
averaged over states **and** coordinates, with an additive 0.1 floor. Section 3.1
requires that any "WML widens the policy" claim be stated per task.
