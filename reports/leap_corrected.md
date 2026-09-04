# Corrected LeapCubeRotateZAxis operator replication — result

**Classification, by its exact preregistered name: `Inconclusive`.**

Preregistration `docs/prereg_leap_corrected.md` (`cbf204a`, Addendum L1 `8291680`).
The rule was fixed before any corrected LEAP outcome existed and is applied here
mechanically.

## 1. Provenance

| Item | Value |
|---|---|
| Training commit, all 16 runs | **`918f82c82a81935af905ef8f7d64fce1bfcf021e`** |
| Preregistration commit recorded in every run | `82916804351d150805fd42811a196cdbccc0c342` |
| Config hashes | `605efbf0…` (PW) x8, `7890bf72…` (WML) x8 |
| SLURM | array `3597493`, tasks 0-15, partition `c23g` |
| Status | **16/16 `COMPLETED`**, exit `0:0`, all exports present |
| Wall clock | min 1243 s, median 1474 s, max 1698 s |

Workstation and cluster both stood at `918f82c` with clean tracked trees, and both
rendered the identical LEAP config hash
`a1e015ced93d99962479a789881170c51f60fb305dfe4ebf931794c4e735397c`, before the
first job was submitted. Cluster-only history was preserved beforehand on
`preserve/cluster-pre-leap-20260904` (`7d590e6`), pushed, and re-verified reachable
after the fast-forward.

Configuration: `128/4`, minibatch 1024, 512 optimizer steps per iteration, 399
iterations, 52,297,728 env steps, `M = 32`, `eps_e 0.5`, `kl_bound 0.1`,
`alpha 0.000782382907345891` frozen, **`vmin -10`, `vmax 60`,
`max_episode_steps 500`** — the LEAP-specific values pinned by Addendum L1.

## 2. Per-seed result

Score = mean of the final three of 21 logged evaluations (`score_window3`).

| Seed | PW-1 | WML-32 | `Delta` | PW last | WML last |
|---|---|---|---|---|---|
| 301 | 35.3395 | 26.5971 | **+8.7424** | 36.9947 | 31.3261 |
| 302 | 29.2883 | 17.7655 | **+11.5228** | 27.2505 | 13.1472 |
| 303 | 16.0238 | 15.6912 | **+0.3326** | 17.6593 | 15.6463 |
| 304 | 30.0456 | 21.7869 | **+8.2587** | 34.5591 | 22.1573 |
| 305 | 35.0356 | 35.7677 | −0.7321 | 32.9972 | 34.2374 |
| 306 | 33.7935 | 23.2654 | **+10.5281** | 31.0586 | 20.3363 |
| 307 | 28.9198 | 36.0386 | −7.1188 | 26.4450 | 34.9588 |
| 308 | 32.5220 | 33.8608 | −1.3388 | 36.2749 | 33.7194 |

## 3. Primary statistic

```
Delta_LEAP = R_PW - R_WML, paired within seed

IQM        PW-1 31.4124    WML-32 26.3775
median     +4.2957
mean       +3.7744
95% CI     [-1.3388, +10.5281]
           paired percentile bootstrap, 10,000 resamples, np.random.default_rng(20260902)
n_pos      5/8      exact two-sided sign test p = 0.7266
```

## 4. The rule, applied

| Condition | |
|---|---|
| point estimate positive | **yes** |
| CI entirely above zero | no |
| CI contains zero | **yes** |
| CI entirely below zero (`CI_upper < 0`) | no |

`Inconclusive`: the confidence interval contains zero.

**Not `PW-supported`** — the interval does not exclude zero. **Not the strong
falsifier** — the interval is not entirely below zero.

## 5. What this does and does not establish

**Establishes.** Under the corrected faithful-repair implementation, on
LeapCubeRotateZAxis with eight paired seeds, the PW-minus-WML contrast is **not
resolved**. The point estimate favours PW (+4.30 median, +3.77 mean, 5 of 8 seeds
positive), but the paired interval spans zero and the sign test is far from
significant.

**Does not establish.** No PW advantage on LEAP. Also **no confirmation of the
legacy WML-favouring result**: the legacy sign was negative, and here the point
estimate is positive while the interval covers both. The honest summary is that
**the legacy LEAP direction does not reproduce as a detected effect under the
corrected implementation, and no replacement direction is detected either.**

This is the outcome the preregistration explicitly allowed for, and it is reported
as measured. Nothing was tuned after the returns were seen.

**Effect-size context, descriptive.** LEAP returns sit near 16-36, so the +4.30
median is roughly a sixth of the arm level. Two seeds (305, 308) are near-ties and
one (307) is strongly WML-favouring at −7.12, against four clearly PW-favouring
seeds. With n = 8 that spread is what leaves the interval open; this is a
statement about resolution, not evidence for a null.

## 6. Relation to the other corrected tasks

| Task | median `Delta` | 95% CI | classification |
|---|---|---|---|
| WalkerRun | +135.58 | [+108.97, +200.76] | detected |
| G1JoystickFlatTerrain | +9.51 | [+2.78, +15.28] | detected |
| **LeapCubeRotateZAxis** | **+4.30** | **[−1.34, +10.53]** | **`Inconclusive`** |

All three point estimates favour PW. LEAP is the one that does not resolve.

**Caveat that must travel with any cross-task reading.** Walker and G1 executed
under a preregistration/execution deviation documented in
`docs/execution_deviation_walker_g1.md` (`128/4` rather than their registered
`64/8` and `16/8`, and for G1 also `gamma 0.99` rather than `0.97` and
`critic_hidden_dim 512` rather than `1024`). LEAP was run at `128/4` deliberately,
prospectively, for cross-task comparability with what those runs actually did — not
with what they registered.

## 7. Artifacts

| Path | Contents |
|---|---|
| `docs/prereg_leap_corrected.md` | preregistration + Addendum L1 |
| `ledger/runs_leap_corrected.jsonl` | 16-row immutable pre-launch ledger |
| `ledger/runs.d.leap/*.json` | 16 completion records |
| `reports/artifacts/leap_canonical_config.json` | canonical resolved config + hashes |
| `reports/artifacts/leap_corrected_results.json` | per-seed scores, CI, classification |
| `scripts/analysis/leap_analyse.py` | this analysis |
| `slurm/leap_launch.sh` | launcher |
| `slurm/logs/leap_3597493_*.out` | 16 run logs |
