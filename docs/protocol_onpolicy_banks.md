# Protocol — on-policy state banks for the two PROXY arms

**Frozen before any rollout is generated.** Append-only. This note fixes the sampling
specification, the RNG derivation and the reported statistics in advance, so that the
numbers cannot be chosen after seeing them.

## 1. What this closes

`reports/onarm_vs_neutral.md` reports an on-arm and a neutral width for four cells per
task, but only two of them have a real on-policy population. The three neutral banks
(`reports/artifacts/{walker,g1,leap}_fixed_state_bank.npz`) were built by rolling out the
two **baseline** checkpoints, so their `source` labels are `PW-sNNN` (from `PW_ent`) and
`WML-sNNN` (from `WML_noent`). For `PW_noent` and `WML_ent` no states from their own
rollouts exist, and `scripts/analysis/onarm_vs_neutral.py:8-11` substitutes the same-arm
*baseline* states and marks the row **PROXY**.

This protocol generates the missing two populations so that all four cells carry a real
on-policy width and the PROXY substitution is retired.

## 2. Sampling specification — mirrors the existing banks

Every value below is copied from the generator that produced the three neutral banks,
`scripts/analysis/fixed_bank_width_saturation.py::build`, and is not re-chosen here.

| quantity | value |
|---|---|
| envs rolled in parallel, `NENV` | 32 |
| depth stratification, Walker and G1 | 50, 150, 300, 500, 700, 900 |
| depth stratification, LEAP | 50, 100, 200, 300, 400, 480 |
| states per (arm, seed) | 32 envs x 6 depths = **192** |
| seeds | 301-308 |
| arms generated here | `PW_noent`, `WML_ent` |
| states per task bank | 2 arms x 8 seeds x 192 = **3072** |
| action during rollout | sampled from the policy, `pi(o).sample()`, then clipped to `+-ACTION_CLIP = 0.999` |
| observation normalizer | frozen at the checkpoint's saved statistics |
| episode length | read from each checkpoint's `meta.json` (`max_episode_steps`) |
| dtype stored | float32 |

The rollout policy for each block is that block's **own** final checkpoint:
`exports/<Task>_pathwise_fa_noent_s<seed>_final` for `PW_noent` and
`exports/<Task>_weighted_mle_ent_s<seed>_final` for `WML_ent`. This is what makes the
resulting states on-policy for that cell.

## 3. One deliberate deviation: the RNG keying is made reproducible

The existing generator derives its per-block key as

```python
key = jax.random.fold_in(jax.random.PRNGKey(ROOT), hash((task, arm, sd)) % (2**31))
```

`hash()` on `str` is salted by `PYTHONHASHSEED`, so that line is **not reproducible across
interpreter invocations** and the three existing banks are therefore not byte-regenerable.
That defect is already on record. It is not repaired retroactively — the existing banks are
frozen inputs to published results and are hash-pinned wherever they are used — but it is
**not repeated here**. This protocol derives the offset deterministically:

```python
off = int.from_bytes(hashlib.sha256(("%s|%s|%d" % (task, arm, seed)).encode()).digest()[:4],
                     "big") % (2**31)
key = jax.random.fold_in(jax.random.PRNGKey(ROOT_ONPOLICY), off)
```

with `ROOT_ONPOLICY = 20260907`, distinct from the existing banks' `ROOT = 20260905`. The
new banks are byte-regenerable from the committed script, and their key streams are
independent of the existing banks' by construction (different root, different derivation).
This satisfies the "fresh independent keys" requirement in two senses: independent of the
neutral banks, and independent across (task, arm, seed).

## 4. Outputs

One file per task, `reports/artifacts/<task>_onpolicy_bank_newarms.npz`, with the same key
schema as the existing banks plus one provenance field:

```
obs (3072, obs_dim) float32 | source (3072,) "<arm>-s<seed>" | depth (3072,)
depths (6,) | n_env () | root () | keying () = "sha256-deterministic"
```

`source` labels are `PW_noent-sNNN` and `WML_ent-sNNN` — deliberately **not** the `PW-`/
`WML-` labels used by the neutral banks, so the two families can never be silently mixed
by a substring match.

The sha256 of each new bank and of the generator script is printed and recorded in the
run log at generation time.

## 5. What is reported afterwards, fixed now

`scripts/analysis/onarm_vs_neutral_v2.py` reports, **per (task, arm, seed)**, for all four
arms and with no PROXY rows:

* `sigma_onpolicy_mean`, `sigma_onpolicy_median` — over that cell's own 192 states
* `sigma_neutral_mean`, `sigma_neutral_median` — over the full 3072-state frozen neutral
  bank for that task, hash-verified before use

All widths are **pre-tanh** `sigma = exp(log_std) + min_std`, taken over states and action
dimensions jointly, from `scripts/load_ckpt.py:152 policy_dist`. Mean and median are both
reported because they are known to disagree in this project: the logged
`train/pi_sigma_mean` is a mean and the width results are medians.

For the two baseline cells the on-policy population is their own-seed slice of the existing
neutral bank (`source == "PW-s<seed>"` / `"WML-s<seed>"`), which is genuinely on-policy for
them and is 192 states, matching the new blocks exactly. Nothing about the baseline cells
is regenerated.

## 6. Scope

Rollouts collect states only. No training, no gradient step, no checkpoint is written or
modified. This protocol does **not** decide which population is the correct one to quote;
`docs/protocol_width_populations.md` governs that question and is unchanged.
