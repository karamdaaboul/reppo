# Does the forward-KL gate term push pre-tanh sigma up?

Offline only. Checkpoints read-only, never mutated. No training started, no training
implementation modified.

Job `3698181` (smoke `3698180` on `afterok`), commit `fb4ab3f`, 00:03:24, node n23g0016.
Script `scripts/analysis/kl_expansion_audit.py`, artifact
`reports/artifacts/kl_expansion_audit.json` (sha256 `390556fbd6097a5c60fd58972ecaeff7368e65ebd4059d577e2fb6a488a7657b`), log `slurm/logs/klexp_3698181.out`.
Bank `reports/artifacts/walker_fixed_state_bank.npz`
(`8adfeb0bf70bddcdbd64a84b972b4dbd62617c64e1c948589a7adc30cf64aa21`), 3072 states,
arm-major, reported both as the neutral full bank and split to each arm's own 1536.

---

## 1. ANALYTIC

For one diagonal Gaussian coordinate,
`KL(old||new) = log(sg_n/sg_o) + (sg_o^2 + dmu^2)/(2 sg_n^2) - 1/2`, so

```
d KL(old||new) / d log sigma_new  =  1 - [sigma_old^2 + (mu_new-mu_old)^2] / sigma_new^2
```

| sg_o | dmu | sg_n | hand | autodiff | abs err | sigma* | pressure |
|---|---|---|---|---|---|---|---|
| 0.40 | 0.00 | 0.40 | +0.000000 | +0.000000 | 0 | 0.4000 | NONE |
| 0.40 | 0.30 | 0.50 | -1.000000 | -1.000000 | 7.9e-07 | 0.5000 | WIDEN |
| 0.40 | 0.30 | 0.80 | +0.609375 | +0.609375 | <1e-7 | 0.5000 | CONTRACT |
| 0.30 | 0.90 | 0.35 | -6.346939 | -6.346939 | <1e-7 | 0.9487 | WIDEN |
| 0.45 | 0.20 | 0.45 | -0.197531 | -0.197531 | <1e-7 | 0.4924 | WIDEN |

Checked additionally against the repo's own sampled estimator (`reppo.py:889-917`, M draws
from pi_old, `logp_old` at the unclipped sample and `logp_theta` at the action clipped to
+-(1-1e-4)), 400k draws per cell.

```
ANALYTIC_AUTODIFF_MATCH = PASS
  closed form   max|err| = 7.88e-07   (tol 1e-5)
  repo sampled  max|err| = 1.66e-03   over non-clip cells (tol 5e-3)
```

**Fixed point.** `sigma_n* = sqrt(sigma_o^2 + dmu^2) >= sigma_o`, with equality only at
`dmu = 0`. The KL-minimising scale is strictly wider than the old scale whenever the mean
has moved at all.

**Sign, explicitly.** Gradient DESCENT moves `log sigma` by `-grad`. When
`sigma_n^2 < sigma_o^2 + dmu^2` the gradient is **negative**, so the step is **positive**
and **sigma INCREASES**. Minimising forward KL by widening is cheaper than by moving the
mean back.

**Contrast, and it is the crux.** Reverse KL(new||old) has
`d/d log sigma_n = -1 + sigma_n^2/sigma_o^2`, fixed point `sigma_n* = sigma_o` exactly
(verified: `+9.54e-08` at `sg_o=sg_n=0.4, dmu=0.3`). The widening incentive is a property
of the **forward** orientation this code uses, not of KL penalties in general.

## Loss terms, traced from source rather than assumed

| arm | improvement | policy entropy | KL gate |
|---|---|---|---|
| `pathwise` | `-value` (pathwise Q), `reppo.py:787` | **yes**, `alpha*log_prob`, `:1049` | yes, `:1084` |
| `weighted_mle` | `-sum_i w_i logp_theta_i`, `:985` | **none** | yes, `:1084` |

`target_entropy_loss` (`:1097`) wraps `stop_gradient(target_entropy)`, so it moves only the
temperature parameter and exerts no force on the policy. In `weighted_mle` the only two
forces on log-sigma are the weighted-MLE fit and the KL gate.

`actor min_std = 0.1` at runtime, although the run config records `actor_min_std: 0.0` —
the config key does not reach the model. Flagged, not acted on.

## 2. DECOMPOSE

`pi_old` is **not** exported (`actor.npz` holds only the live policy), so it is a calibrated
surrogate: `sigma_old = sigma*exp(-drift + jitter)` with `drift` the run's own signed
per-iteration `d log sigma`, and a per-state mean displacement whose scale and across-state
spread are fitted so the bank reproduces that seed's logged per-state KL **distribution**
(median to `fr_kl_q50`, interquartile ratio to `fr_kl_q75/q25`). The gate-fire fraction is
then a prediction, not a fitted target.

Medians over eight seeds, on-arm bank. Descent convention: `grad < 0` => WIDEN.

| arm | frac | sigma | gate_cl (logged) | clip fr | g_imp | g_ent | g_kl | g_kl noclip | sigma/sigma* | frac<1 | closed frac g_kl<0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PW | p25 | 0.567 | 0.483 (0.550) | 0.003 | +0.0007 | **-0.0123** | **-0.0009** | -0.0010 | 0.9915 | 0.961 | **0.644** |
| PW | p50 | 0.469 | 0.508 (0.554) | 0.009 | +0.0010 | -0.0131 | -0.0006 | -0.0007 | 0.9939 | 0.855 | 0.622 |
| PW | final | 0.455 | 0.480 (0.562) | 0.006 | +0.0008 | -0.0131 | -0.0004 | -0.0005 | 0.9940 | 0.884 | 0.624 |
| WML | p25 | 1.484 | 0.853 (0.555) | 0.235 | +0.5079 | 0 | +0.0627 | **-0.0016** | 1.0004 | 0.494 | 0.276 |
| WML | p50 | 3.117 | 0.956 (0.556) | 0.413 | +0.6394 | 0 | +0.1364 | -0.0082 | 0.9953 | 0.702 | 0.221 |
| WML | final | 6.427 | 0.949 (0.563) | 0.529 | +0.7733 | 0 | +0.1905 | -0.0082 | 0.9876 | 0.883 | 0.199 |

Pressures: PW improvement CONTRACT (negligible), entropy WIDEN, **KL WIDEN**. WML
improvement CONTRACT (large), no entropy term, KL CONTRACT as measured with the clip but
**WIDEN without it**.

`sigma/sigma*` is below 1 nearly everywhere, and `frac<1` — the fraction of coordinates
sitting **below** their own KL fixed point, where the exact gradient is negative — is
0.49-0.96. The policy is essentially always on the widening side of the fixed point.

The neutral full bank gives the same picture (PW `frac_kl_up` 0.614-0.634, WML 0.189-0.302);
the on-arm and neutral columns differ by less than the seed spread.

**The clip caveat, and it cuts against my first reading.** The action clip caps pre-tanh
`|y|` at `atanh(1-1e-4) = 4.95`. On the bank WML's sigma reaches 6.4 and 53 % of E-step
samples clip, which flips the measured `g_kl` positive. But the runs log
`fr_kl_sampled_minus_analytic_mean ~ 0` (+-5e-4) throughout, and logged `pi_sigma_mean`
peaks near 2.2-5.2, not 6.4. During training the clip was therefore nearly inactive, and
`g_kl_noclip` — negative at every WML checkpoint — is the better proxy for the force that
actually acted. The positive `g_kl` column is an artifact of evaluating a wide policy on a
fixed bank, not a training-time contraction.

## 3. PARAMETER STEP

One optimizer step per component on temporary copies (`nnx.split`, Param-filtered);
no checkpoint touched. `adam = chain(clip_by_global_norm(0.5), adam(3e-4))`. Delta is the
change in median pre-tanh sigma on the same bank. Gated variants reproduce the real loss's
per-state switch; ungated variants exist because the gate closes on ~95 % of WML states,
which would otherwise measure "improvement" on a 5 % subset.

On-arm bank, medians over eight seeds:

| arm | frac | sigma0 | gated d_imp | d_ent | d_kl | d_full | ungated d_imp | ungated d_kl |
|---|---|---|---|---|---|---|---|---|
| PW | p25 | 0.567 | -3.77e-03 | +1.17e-02 | **+1.86e-02** | -2.60e-03 | -3.85e-03 | +1.83e-02 |
| PW | p50 | 0.469 | -3.85e-03 | +8.19e-03 | **+1.01e-02** | -3.23e-03 | -3.66e-03 | +1.01e-02 |
| PW | final | 0.455 | -3.36e-03 | +6.99e-03 | **+9.29e-03** | -2.09e-03 | -3.38e-03 | +9.32e-03 |
| WML | p25 | 1.484 | +7.55e-02 | 0 | -2.38e-01 | -1.27e-01 | -9.79e-02 | -2.39e-01 |
| WML | p50 | 3.117 | +5.89e-02 | 0 | -4.89e-01 | -4.41e-01 | -4.06e-01 | -4.89e-01 |
| WML | final | 6.427 | +2.41e-02 | 0 | -4.03e-01 | -3.86e-01 | -3.75e-01 | -4.03e-01 |

**In `pathwise` the KL term is the single largest sigma-increasing force in the actor loss** —
larger than the entropy term it is usually assumed to be dominated by (+0.0186 vs +0.0117 at
p25) — and the pathwise-Q term is what holds sigma down (-0.0038). The net is -0.0026:
PW sits in a balance, which is why its sigma stays near 0.5 for the whole run.

At WML's endpoint sigma every term contracts, consistent with a policy that has overshot
and is being pulled back (logged sigma for s301: 0.435 -> 3.162 peak -> 2.203 final).

## 2b. SIGMA SWEEP

The runs widen from sigma ~0.44 to ~3 **before** the first saved checkpoint, so no export
sits in the regime where the widening began. Rescaling the checkpoint's sigma while holding
the mean displacement fixed reconstructs it. Medians over eight seeds, on-arm bank:

| arm | sigma | clip fr | g_imp | g_ent | g_kl | g_kl noclip | frac<1 |
|---|---|---|---|---|---|---|---|
| PW | 0.057 | 0.001 | +0.0000 | -0.0145 | **-0.0285** | -0.0285 | 0.989 |
| PW | 0.142 | 0.001 | +0.0000 | -0.0144 | -0.0067 | -0.0067 | 0.971 |
| PW | 0.284 | 0.002 | +0.0002 | -0.0140 | -0.0026 | -0.0027 | 0.944 |
| PW | 0.567 | 0.003 | +0.0008 | -0.0122 | -0.0008 | -0.0009 | 0.894 |
| PW | 1.134 | 0.012 | +0.0016 | -0.0049 | +0.0003 | +0.0000 | 0.818 |
| PW | 2.268 | 0.082 | +0.0005 | +0.0192 | +0.0032 | +0.0003 | 0.722 |
| WML | 0.148 | 0.106 | **-1.3450** | 0 | **-0.3954** | -0.4322 | 0.991 |
| WML | 0.371 | 0.126 | -0.1244 | 0 | -0.0622 | -0.0916 | 0.976 |
| WML | 0.742 | 0.165 | +0.2273 | 0 | +0.0128 | -0.0312 | 0.956 |
| WML | 1.484 | 0.235 | +0.5027 | 0 | +0.0620 | -0.0067 | 0.910 |
| WML | 2.969 | 0.335 | +0.7277 | 0 | +0.1164 | +0.0026 | 0.832 |
| WML | 5.937 | 0.457 | +0.8755 | 0 | +0.1842 | +0.0053 | 0.723 |

At the sigma each arm actually started from (~0.44-0.9), the KL term widens in **both** arms.
It crosses zero near sigma ~0.7-1.1 and contracts above. So the KL term is a widening force
exactly in the regime where the widening happened, and a restoring force afterwards — a
self-limiting expansion, not an unbounded ratchet.

Decisively for attribution: at sigma = 0.148 the weighted-MLE term is **-1.345** against the
KL term's **-0.395**. In the regime that produced the widening, the improvement term pushes
sigma up 3.4x harder than the KL term does.

## 4. RATCHET CHECK

Consistency check, not a fit. If the step-1 fixed point were reached each iteration,
`sigma_{k+1}^2 = sigma_k^2 + dmu_k^2`, so growth per iteration is `sqrt(1 + (dmu/sigma)^2)`,
with `(dmu/sigma)^2 ~ 2*kl/d` from the logged mean KL over d = 6 dims, applied over
`400 * gate_fraction` firings.

| arm | median kl | (dmu/sg)^2 | gate | pred x/iter | pred cumulative | observed max/init | obs/pred |
|---|---|---|---|---|---|---|---|
| pathwise | 0.1058 | 0.0353 | 0.549 | 1.0175 | **4.47e+01** | **1.00** | 2.29e-02 |
| weighted_mle | 0.1066 | 0.0355 | 0.553 | 1.0176 | **4.80e+01** | **7.03** | 1.45e-01 |

The two arms have **the same** logged KL, the same implied displacement and the same gate
fraction, so the unconstrained ratchet predicts the same ~45-48x for both. PW realises 1.00x
and WML 7.03x. The ratchet is therefore not sufficient on its own: what differs between the
arms is not the KL term but what opposes it.

Seed 304 is the known anomaly (max/init 4033x, logged sigma peak 3452.77) and is excluded
from no statistic here; the table reports medians, which it does not drive.

## 5. CLASSIFICATION

```
KL_EXPANSION_PARTIAL
```

**Supported.** The mechanism is real and verified analytically and by autodiff on the repo's
own estimator. The forward-KL fixed point is strictly wider than `sigma_old`; the trained
policies sit below that fixed point in 49-99 % of coordinates at every checkpoint; the KL
term widens under a parameter step in `pathwise`, where it is the **largest** sigma-increasing
term in the loss, above the entropy term; and in the narrow regime both arms began from, the
KL term widens in both. Reverse KL would have no such incentive, so this is a consequence of
the forward orientation the implementation chose.

**Not supported.** KL expansion does not explain the PW/WML width gap, which was the reason
to look. Both arms carry the same KL term with the same logged KL, gate fraction and implied
displacement, and the ratchet predicts the same ~45x for each; PW realises 1.00x. In the
regime where WML widened, the weighted-MLE improvement term pushes sigma up 3.4x harder than
the KL term. The gap is therefore governed by what **opposes** the widening — the pathwise-Q
term in PW, which has no counterpart in WML — together with the absence of any policy-entropy
term in `weighted_mle`.

**Correction to my own earlier reading.** I first reported the KL term as contracting for WML
(`g_kl` +0.19). That measurement is confounded: it comes from evaluating a wide policy on a
fixed bank where 53 % of E-step samples hit the tanh clip, whereas the runs log
`fr_kl_sampled_minus_analytic_mean ~ 0` throughout, so the clip was nearly inactive during
training. The unclipped column, negative at every WML checkpoint, is the right proxy.

**Limits.** `pi_old` is a calibrated surrogate, not a reconstruction, because `actor_target`
is not exported; it reproduces each seed's logged per-state KL distribution, and its
predicted gate fraction matches the logged one for PW (0.48-0.51 vs 0.55-0.56) but
over-predicts for WML (0.85-0.96 vs 0.55-0.56), driven by the same clip confound. Part 3
compares one optimizer step per term; adam normalises gradient magnitude, so signs and
relative order are the informative content, not absolute deltas. Part 4 uses 21 logged eval
points, each aggregating 20 iterations, not per-iteration data.
