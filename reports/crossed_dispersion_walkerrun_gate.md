REFUTED-AS-CRITIC-SOURCE

# Crossed frozen-critic dispersion — WalkerRun reference-law gate

This is the preregistered reference-law control of
`docs/prereg_crossed_dispersion.md` (`d1de4e8`, addendum A1 `016c4a8`). It is
**not** the final four-task report. The legacy 64 and G1JoystickFlatTerrain were
not run.

## 1. The rule being applied, quoted before any numbers

Prereg section 11, verbatim:

> **P3 (kill condition).** If `sign(I)` differs between the A-law and B-law
> evaluations, the finding is attributable to policy width and **NOT** to critic
> source. The report's **first line** then records `REFUTED-AS-CRITIC-SOURCE`.

And the statistic, prereg section 4, verbatim:

> ```
> I = log[ D_PW(Q_PW) / D_PW(Q_WML) ]  +  log[ D_WML(Q_WML) / D_WML(Q_PW) ]
>       \_____________ bracket 1 _____/     \_____________ bracket 2 _____/
> ```

with `I_equal-query` (PRIMARY) using `D_{PW-32}` and `D_{WML-32}`, and
`I_operational` (SECONDARY) using `D_{PW-1}` and `D_{WML-32}`. ZO enters neither,
because it trained no critic.

## 2. Provenance

| Item | Value |
|---|---|
| Branch | `estep-study` |
| Analysis code | `scripts/analysis/crossed_dispersion.py` at **`a87e5d1`** |
| Gate analysis | `scripts/analysis/crossed_dispersion_gate.py` at `e74b297` |
| Launcher | `slurm/crossed_dispersion.sh` at `8f176d5` |
| Bank | `reports/artifacts/cd_bank_walker_corrected.npz` |
| Bank sha256 | `8b2cb3180fd307612c9bcd8c3298302e3fbd860ac54f8890cf1849e70711e36b` |
| Bank commit | `9f80614` |
| Tier | corrected only, seeds 301-308 (prereg sec 10; addendum A1) |

The `run` job refuses to start if `crossed_dispersion.py` differs from `HEAD`, and
verifies the bank hash before **and** after use. Both verifications passed.

Exact commands:

```
sbatch --time=00:30:00 -J cdsmoke slurm/crossed_dispersion.sh smoke   # job 3496373
sbatch --time=00:40:00 -J cdbank  slurm/crossed_dispersion.sh bank    # job 3496535
sbatch --time=03:00:00 -J cdrun   slurm/crossed_dispersion.sh run     # job 3496842
./.venv/bin/python scripts/analysis/crossed_dispersion_gate.py \
    reports/artifacts/cd_walker_corrected.csv reports/artifacts/cd_walker_gate.json
```

Job 3496535 ran 00:07:22 on `c23g`; job 3496842 ran 00:01:37 on `c23g`.

### Bank

2048 states = 16 policies x 128, both arms x seeds 301-308, burn-in **50**
stochastic-policy steps from `env.reset`, no post-hoc filtering and no
replacement. All 2048 states distinct. 128 evaluation states drawn once from RNG
root `20260904` and stored with the bank, so every cell sees the same states.
Per-policy checkpoint paths and `actor.npz` hashes are in
`reports/artifacts/cd_bank_walker_corrected_provenance.json`.

**Portability limitation.** MJX rollouts are not bit-portable across backends: the
same collection code produced different bank hashes on a CPU login node and on a
`c23g` GPU node. The bank was collected once and frozen, so no result here depends
on this, but the bank is byte-reproducible only on `c23g`.

### Checkpoint inventory

16/16 corrected WalkerRun checkpoints present:
`exports/WalkerRun_{pathwise_fa,weighted_mle}_s{301..308}_final`.

## 3. Validation, with tolerances

Run on scratch data only, in `/hpcwork/$USER/cd_scratch*`, never under
`reports/artifacts`, never reused. Code `1fde45b`. Tolerances are derived from the
dtype and the arithmetic *before* each check runs (`tol_arith`), not fitted to an
observed residual. **14/14 pass**, on both a CPU login node and the GPU node (job
3496373).

Unit stage, `JAX_ENABLE_X64=1`:

| Check | Result | Tolerance |
|---|---|---|
| C2a `PW-32` = mean of action gradients | 0 | 4.79e-13 |
| C2b `PW-1` = first sample of that cloud | 0 | exactly 0 |
| C2c `ZO-32` = centered closed form | 4.44e-16 | 5.06e-12 |
| C2d `ZO` centered: constant `F` annihilates | 7.47e-16 | 5.31e-12 |
| C2e `WML-32` = softmax(`F`/eta) on `(y-mu)` | 5.55e-16 | 9.14e-13 |
| C3 eta attains the preregistered dual minimum | 4.25e-16 rel | 1e-09, vs scipy bounded, 35 clouds |
| C4a unit whitened norm after normalisation | 2.22e-16 | 2.66e-14 |
| C4b `D` = mean squared whitened deviation | 2.22e-16 | 1.87e-13 |
| C4c `D` tracks an **anisotropic** `Sigma` change | 0 | 3.55e-14 |

C4c predicts `D = 0.1464466 -> 0.0527864` in closed form before measuring, and
measures `0.1464466 -> 0.0527864`.

Wiring stage, production float32:

| Check | Result |
|---|---|
| C0 scratch config self-consistent | `per_policy x n_policies == n_bank` asserted |
| C7 same seed reproduces the run | CSV byte-identical over two invocations |
| C1 innovations shared across laws and critic sources | one `u` per seed across all 4 law x critic cells; redraw bitwise identical |
| C5 operator x critic x law indexing | 32 rows, 32 unique cells, expected 32 |
| C6 law change moves `sigma` and `y`, nothing else | states, critics, innovations identical |

### Four defects found and fixed, all in the test, none in the analysis code

Reported under prereg section 12.6, which requires every failed check including
bugs in this analysis.

1. **C2d** asserted a constant `F` annihilates centered ZO to `1e-12`. In float32
   the mean of `M` identical values does not round-trip, giving `8.02e-07`. The
   bound is `|F| M eps max|u/sigma|`; in float64 the residual is `7.47e-16`.
2. **C3** compared the dual solution to scipy at an absolute `1e-6` and missed by
   `1.01e-6`. That was float32 noise in the objective, not solver failure: in
   float64 the relative excess is `4.25e-16`.
3. **C4c** perturbed `sigma` by a **uniform** factor. Under unit-whitened
   normalisation a uniform rescale cancels exactly, so `D` is invariant in that
   direction by construction and the check could never have passed. Replaced with
   an anisotropic perturbation on a non-axis-aligned toy.
4. The scratch config set `per_policy=4` with `n_bank=128` while `4 x 16 = 64`, so
   collection tripped its own size assertion.

The registered bank remains `2048 = 16 x 128`; the small numbers were scratch only.

## 4. Results, each reference law reported separately

`R = 200`, `S_eval = 128`, `M = 32` (`M = 1` for `PW-1`), verified constant across
all 128 rows. Eight seeds. Bootstrap: resample seeds, form `I` within each
replicate, 10,000 replicates, `np.random.default_rng(20260904)`.

### `D` by law, critic and operator (median over the eight seeds)

| Law | Operator | `D(Q_PW)` | `D(Q_WML)` |
|---|---|---|---|
| A (PW reference) | `PW-32` | 0.01669 | 0.01183 |
| A | `PW-1` | 0.28508 | 0.22788 |
| A | `WML-32` | 0.29619 | 0.31581 |
| A | `ZO-32` | 0.22684 | 0.22174 |
| B (WML reference) | `PW-32` | 0.12648 | 0.15462 |
| B | `PW-1` | 0.58972 | 0.66186 |
| B | `WML-32` | 0.37299 | 0.38736 |
| B | `ZO-32` | 0.60286 | 0.63811 |

### `I_equal-query` — PRIMARY

| Law | median `I` | 95% CI | excludes 0 | sign | `n_pos` `I` | `n_pos` b1 | `n_pos` b2 |
|---|---|---|---|---|---|---|---|
| **A** | **+0.4780** | [+0.3330, +0.5439] | yes | **+1** | 8/8 | 8/8 | 8/8 |
| **B** | **-0.3212** | [-0.3598, -0.0588] | yes | **-1** | 1/8 | 0/8 | 6/8 |

Brackets separately, as prereg section 4 requires:

| Law | median b1 | 95% CI b1 | median b2 | 95% CI b2 |
|---|---|---|---|---|
| A | +0.4048 | [+0.2678, +0.5337] | +0.0711 | [+0.0244, +0.1117] |
| B | -0.3455 | [-0.4356, -0.0997] | +0.0504 | [-0.0101, +0.0724] |

Per seed:

| Law | | 301 | 302 | 303 | 304 | 305 | 306 | 307 | 308 |
|---|---|---|---|---|---|---|---|---|---|
| A | b1 | +0.162 | +0.240 | +0.534 | +0.400 | +0.374 | +0.517 | +0.410 | +0.661 |
| A | b2 | +0.081 | +0.093 | +0.010 | +0.061 | +0.112 | +0.024 | +0.061 | +0.129 |
| A | `I` | +0.243 | +0.333 | +0.544 | +0.461 | +0.486 | +0.541 | +0.471 | +0.790 |
| B | b1 | -0.416 | -0.432 | -0.006 | -0.465 | -0.436 | -0.108 | -0.275 | -0.100 |
| B | b2 | +0.059 | +0.072 | +0.038 | +0.051 | +0.076 | +0.049 | -0.010 | -0.019 |
| B | `I` | -0.357 | -0.360 | +0.032 | -0.414 | -0.360 | -0.059 | -0.285 | -0.119 |

### `I_operational` — SECONDARY

| Law | median `I` | 95% CI | excludes 0 | sign | `n_pos` `I` | `n_pos` b1 | `n_pos` b2 |
|---|---|---|---|---|---|---|---|
| A | +0.3308 | [+0.2519, +0.4618] | yes | +1 | 8/8 | 8/8 | 8/8 |
| B | -0.0666 | [-0.0872, -0.0555] | yes | -1 | 0/8 | 0/8 | 6/8 |

### `PW-32` vs `ZO-32`, equal query — separate deliverable, not folded into `I`

| Law | Critic | `D_{PW-32}` | `D_{ZO-32}` |
|---|---|---|---|
| A | `Q_PW` | 0.01669 | 0.22684 |
| A | `Q_WML` | 0.01183 | 0.22174 |
| B | `Q_PW` | 0.12648 | 0.60286 |
| B | `Q_WML` | 0.15462 | 0.63811 |

## 5. Applying P3

`sign(I)` under the A-law is `+1`. `sign(I)` under the B-law is `-1`. They differ.
This holds for the primary `I_equal-query` and for the secondary
`I_operational`, and in both cases **both** laws' intervals exclude zero, so the
reversal is not an artefact of an interval straddling zero.

By the rule quoted in section 1, the classification is:

**`REFUTED-AS-CRITIC-SOURCE`**

The finding is attributable to policy width, not to critic source.

The reversal sits entirely in **bracket 1**, the pathwise operator: b1 goes
`+0.4048 -> -0.3455` between laws while b2 stays positive under both (`+0.0711`,
`+0.0504`), though b2's B-law interval includes zero. Per prereg section 4 this
asymmetry is reported rather than hidden in the sum: a sign flip driven by one
bracket is a different phenomenon than both contributing.

### Consequence, per the preregistered branch

Stop. The g1 breadth run is **not** prepared and **not** launched, and the legacy
64 are **not** run.

## 6. Integrity checks (prereg section 12)

**12.1 Common random numbers.** One `u` per `(task, seed)`, bitwise shared across
both reference laws and both critic sources; verified end-to-end by identical
`u_sha256` across all four law x critic cells of each seed, and distinct across
seeds. Redraw is bitwise identical (`np.array_equal`).

**12.2 ZO is centered, no softmax, no `ubar`.** Checked against a closed form
(C2c) and by annihilation of a constant `F` (C2d).

**12.3 The whitened metric uses the reference law's `Sigma`.** Asserted in code;
C4c confirms `D` tracks an anisotropic change in that `Sigma`. Within a cell the
same `Sigma` is used for sampling, for `Sigma^{-1}` in ZO, and for the norm.

**12.4 `sigma` distribution per arm, before any median is quoted.**

| Law (reference arm) | median | p95 | min | max | sd across states | sd across coords |
|---|---|---|---|---|---|---|
| A (PW) | 0.5205 | 0.8506 | 0.1092 | 5.134 | 0.1710 | 0.1200 |
| B (WML) | 4.2266 | 90.011 | 0.1133 | 740.60 | 43.638 | 7.1368 |

Per seed, `sigma` median / p95 / max:

| Seed | A median | A p95 | A max | B median | B p95 | B max |
|---|---|---|---|---|---|---|
| 301 | 0.5205 | 0.8506 | 5.13 | 4.2266 | 90.01 | 740.6 |
| 302 | 0.5167 | 0.8166 | 1.27 | 3.9897 | 110.90 | 1602.9 |
| 303 | 0.5385 | 0.8553 | 20.62 | 2.7907 | 76.84 | 1565.0 |
| 304 | 0.5000 | 0.7850 | 1.17 | 2.1136 | 36.25 | 287493.0 |
| 305 | 0.5237 | 0.8638 | 23.41 | 5.4147 | 96.33 | 71976.5 |
| 306 | 0.4940 | 0.7916 | 3.26 | 3.1452 | 67.91 | 2026.4 |
| 307 | 0.4991 | 0.7970 | 1.15 | 3.4724 | 72.39 | 1013.4 |
| 308 | 0.5120 | 0.8863 | 6.57 | 4.2132 | 162.03 | 26488.8 |

The two laws' widths differ by roughly an order of magnitude at the median and
two to five orders at the maximum. This is the width asymmetry the reference-law
control exists to detect, and it is what P3 has now detected.

**12.5 Saturation, fraction of `|tanh(y_i)| > 0.999`.**

| Law | fraction | range over seeds |
|---|---|---|
| A (PW reference) | 0.0063 | 0.0055 - 0.0176 |
| B (WML reference) | 0.5800 | 0.4019 - 0.5942 |

**12.6 Failures and anomalies, all reported.**

Beyond the four validation defects in section 3, the run produced two anomalies.
Both were diagnosed with dedicated read-only jobs rather than assumed.

*Anomaly A — positive E-step dual gap.* The run logs
`max(g(eta) - min(g(1.01 eta), g(0.99 eta)))`, which must be `<= 0` if `eta`
minimises. Observed values up to `+0.098`. Job 3497748
(`scripts/analysis/cd_eta_check.py`) splits the gap by whether `eta` is interior
to the preregistered clip `[1e-4, 10.0]`:

| Law | Critic | eta interior | eta clipped | max gap interior | max gap clipped |
|---|---|---|---|---|---|
| A | `Q_PW` | 1.000 | 0.000 | +2.29e-05 | n/a |
| A | `Q_WML` | 0.985 | 0.015 | +1.53e-05 | +0.0296 |
| B | `Q_PW` | 1.000 | 0.000 | +3.05e-05 | n/a |
| B | `Q_WML` | 0.934 | 0.066 | +1.53e-05 | +0.0706 |

Wherever `eta` is interior the gap is at float32 noise (`<= 3.05e-05`); it is
nonzero only where the **preregistered** clip binds, which for the WML critic is
1.5% of clouds under the A-law and 6.6% under the B-law. The solver is not at
fault; the constraint is. This is a property of the registered `eta` range, and
it is recorded here rather than changed.

*Anomaly B — zero-norm pathwise updates.* 9958 updates over the whole run had
zero whitened norm, all under the **B-law** only: `PW-1` 4886/204800 (2.386%) and
`PW-32` 93/204800 (0.045%). `WML-32` and `ZO-32` had none. Job 3498657
(`scripts/analysis/cd_zero_check.py`), seed 301 law B at full `R = 200`:

- per-coordinate saturation 0.5800; all-`d` saturated per sample 0.1206
- `PW-1` zero fraction 0.01902 (487 clouds), and
  **`P(all-d saturated | zero norm) = 1.0000`**
- the zero mask is **identical** across critic sources, for both `PW-1` and
  `PW-32`

So every zero-norm pathwise update is an action whose every coordinate is
`tanh`-saturated, where `dQ/dy` vanishes exactly in float32. It is a consequence
of the WML policy width in section 12.4, not a defect in the estimator, and it
is driven by the shared `y` rather than by either critic.

**Effect on the result.** `dispersion` maps a zero-norm update to the zero
vector, which shrinks `||gbar||` and therefore inflates `D`. This inflation
applies to the **same** clouds for both critic sources, so it partly cancels
inside a log ratio, but not exactly. It matters for `I_operational`, whose
bracket 1 uses `PW-1` (2.386% affected under the B-law). It is negligible for the
**primary** `I_equal-query`, whose bracket 1 uses `PW-32` (0.045% affected). The
primary classification therefore does not rest on the affected operator.

## 7. Scope

Corrected-tier WalkerRun only. Per prereg addendum A1, the rule is one bank per
task **x evidence tier**; a legacy-tier WalkerRun bank would be separate and does
not exist. No return-level claim, no `omega` claim, no Claim 4 claim, no dimension
claim is made or implied here. `D` is directional dispersion under a fixed
reference law and nothing else.

## 8. Artifacts

| Path | Contents |
|---|---|
| `reports/artifacts/cd_bank_walker_corrected.npz` | frozen 2048-state bank |
| `reports/artifacts/cd_bank_walker_corrected.sha256` | bank hash |
| `reports/artifacts/cd_bank_walker_corrected_provenance.json` | 16 collecting policies, hashes, command, git SHA |
| `reports/artifacts/cd_walker_corrected.csv` | 128 cells: seed x law x critic x operator |
| `reports/artifacts/cd_walker_corrected_diagnostics.csv` | eta, `sigma`, saturation per cell |
| `reports/artifacts/cd_walker_corrected_checks.json` | CRN and law-isolation checks, bank hash |
| `reports/artifacts/cd_walker_gate.json` | `I`, brackets, bootstrap CIs, classification |
| `slurm/logs/cd_cdsmoke_3496373.out` | validation, 14/14 |
| `slurm/logs/cd_cdbank_3496535.out` | bank collection |
| `slurm/logs/cd_cdrun_3496842.out` | crossed run, both laws |
| `slurm/logs/cd_etachk_3497748.out` | anomaly A |
| `slurm/logs/cd_zerochk_3498657.out` | anomaly B |
