# Critic-reference convention for future critic-error analyses

**Dated 2026-09-04T11:22:22+00:00. Prospective.** This document fixes the reference quantity for
**future** critic-error mechanism analyses on new checkpoints or new tasks. It is
**not** a retroactive preregistration of the Walker MC-oracle pilot
(`docs/prereg_mc_oracle_walker_pilot.md`, `63c2cd2`), and it does **not** change the
interpretation of any already-run result.

## 1. Primary reference

```
e_true(s,a) = Q_phi(s,a) - Q_soft^pi(s,a)
```

with `Q_soft^pi` the **UNCLIPPED** environmental soft return,

```
Q_soft^pi(s0,a0) = E[ r_0 + sum_{t>=1} gamma^t ( r_t - alpha log pi(a_t|s_t) ) ]
```

evaluated by Monte Carlo under the frozen checkpoint policy with the checkpoint's
frozen `alpha`, first action forced to `a0`.

**Why primary.** The policy-improvement operator is ultimately intended to improve
the true soft objective. A representational floor or ceiling in `Q_phi` is a real
limitation of the learned critic, so it is legitimately part of that critic's
approximation error and must not be defined away.

## 2. Secondary decomposition

```
e_support(s,a) = Q_phi(s,a) - clip( Q_soft^pi(s,a), 0, 150 )
```

with `[0, 150] = [vmin, vmax]` from `config/env/mjx_dmc.yaml`, the same bounds
`src/jaxrl/utils.py:44-45` applies to the training target inside `hl_gauss`:
`x = jnp.clip(inp, vmin, max=vmax)`.

**What it isolates.** `e_support` is the error remaining **after** removing
critic-support mismatch. The difference `e_true - e_support` is exactly the part of
the error attributable to the oracle leaving the critic's representable range.

## 3. Rule

The secondary is **never** substituted for the primary after observing results.
Both are reported when they differ materially, and the primary carries any
verdict. A task whose oracle stays inside `[vmin, vmax]` has `e_true = e_support`
identically, and only one number is reported.

## 4. Scope

This convention binds future analyses. Existing results keep the definitions they
were run under, and any comparison across the boundary states which convention
each side used.
