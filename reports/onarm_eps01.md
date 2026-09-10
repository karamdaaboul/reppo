# The E-step concentration arm on its own occupancy — prereg §4.2 closed

Job `3884816` (`opb01`, COMPLETED 00:33:43, node r23g0002) built the own-rollout banks;
`scripts/analysis/onarm_eps01.py` applies `docs/prereg_estep_concentration.md` §4.2 to them.
Offline, no training. Bootstrap `np.random.default_rng(20260911)`, 10000 resamples — a fresh
key, distinct from §4.1's registered 20260910. All bank hashes verified in-run.

Banks: 1536 states per task (8 seeds × 32 envs × 6 depths), sha256-deterministic keying,
ROOT 20260909, generator sha256 `c6ac4d3214d50362`. LEAP on its shorter depth ladder
(50/100/200/300/400/480) as the protocol specifies.

## 1. The result

**The `eps_e = 0.1` width collapse is a matched-state phenomenon. It does not appear on the
states the policy actually visits.**

`eps01 / eps05` pre-tanh width ratio, paired within seed, computed **separately within each
population** and never across:

| task | OWN occupancy | MATCHED-STATE (neutral bank) |
|---|---|---|
| Walker | 0.730× [0.694, 1.144] | **0.148× [0.079, 0.223]** |
| G1 | 1.036× [0.950, 1.186] | **0.476× [0.326, 0.725]** |
| LEAP | 1.178× [0.976, 1.418] | 1.386× [0.602, 2.758] |

All three own-occupancy intervals contain 1. Two of the three matched-state intervals exclude
it. Tightening the E-step budget makes the policy dramatically narrower **where it does not
live**, and leaves it essentially unchanged **where it does**.

This does not weaken the §4.1 primary result — it is the same population dependence the
project's headline rests on, now shown to survive an intervention that moves matched-state
width by a factor of 6.8.

## 2. Operator gap, both populations

| task | arm | OWN occupancy | MATCHED-STATE |
|---|---|---|---|
| Walker | WML eps05 | 1.295× [0.972, 1.577] | 22.952× [13.170, 33.400] |
| Walker | WML eps01 | 0.996× [0.929, 1.207] | 2.433× [2.150, 4.575] |
| G1 | WML eps05 | 1.027× [0.912, 1.124] | 1.999× [1.639, 2.072] |
| G1 | WML eps01 | 0.988× [0.867, 1.332] | 0.965× [0.719, 1.475] |
| LEAP | WML eps05 | 0.874× [0.556, 1.173] | 2.611× [1.533, 4.455] |
| LEAP | WML eps01 | 1.000× [0.774, 1.263] | 2.953× [1.762, 7.654] |

**Own-occupancy parity is now 6/6** — every cell's interval contains 1, at both budgets, on
all three tasks. Matched-state separation holds in 5 of 6 (the exception is G1 at `eps01`,
which is the gap-closure already reported in §4.1).

## 3. Common-to-own amplification (registered §4.2)

Ratio of matched-state width to own-occupancy width, median over seeds:

| task | WML eps01 | WML eps05 | PW_noent |
|---|---|---|---|
| Walker | 5.233× [4.493, 7.261] | **27.572× [23.941, 48.454]** | 2.016× [1.719, 2.674] |
| G1 | 1.373× [1.130, 1.507] | 2.828× [2.496, 3.393] | 1.509× [1.283, 1.622] |
| LEAP | 22.373× [13.222, 67.303] | 26.669× [10.510, 31.626] | 6.633× [5.908, 10.351] |

> **Corrected 2026-09-09 after code review.** These were first published as a ratio of two
> across-seed medians (Walker `WML_eps05` 33.50×, LEAP `WML_eps01` 26.10×). That is not the
> frozen convention, which is the median over seeds of per-seed log ratios, exponentiated —
> and under the heavy skew these distributions carry the two diverge materially, by 18% on
> Walker `WML_eps05` and 14% on LEAP `WML_eps01`. The table above uses the frozen convention
> and now carries bootstrap intervals; every cell is backed by all 8 seeds. The qualitative
> reading is unchanged.

Both weighted-MLE budgets amplify far more than pathwise on Walker and LEAP. The tighter
budget cuts Walker's amplification about 5-fold; on LEAP it does not move it (the two
intervals overlap heavily).

Mean and median disagree sharply here, as this project has recorded elsewhere — Walker
`WML_eps05` matched-state mean is 318.36 against a median of 7.31, LEAP `WML_eps01` mean
431.65 against a median of 3.85. Medians are used throughout; means are in the CSV.

## 4. Saturation (prereg §4.3, exact under the policy)

| task | arm | own s95 | own s99 | neutral s95 | neutral s99 |
|---|---|---|---|---|---|
| Walker | WML eps01 | 0.3237 | 0.1520 | 0.5763 | 0.4355 |
| Walker | WML eps05 | 0.3978 | 0.2684 | 0.7895 | 0.7151 |
| Walker | PW_noent | 0.3549 | 0.1886 | 0.5697 | 0.4134 |
| G1 | WML eps01 | 0.2885 | 0.0848 | 0.3124 | 0.1088 |
| G1 | WML eps05 | 0.2040 | 0.1158 | 0.3510 | 0.2365 |
| G1 | PW_noent | 0.2801 | 0.1582 | 0.3846 | 0.2480 |
| LEAP | WML eps01 | 0.1951 | 0.0712 | 0.6889 | 0.5862 |
| LEAP | WML eps05 | 0.1773 | 0.0794 | 0.6810 | 0.5813 |
| LEAP | PW_noent | 0.2215 | 0.0990 | 0.5812 | 0.4395 |

On Walker the tighter budget cuts saturation on both populations (own s99 0.268 → 0.152;
matched-state s99 0.715 → 0.436). On LEAP it moves neither.

## 5. Cross-validation

`onarm_eps01.py` was written independently of `ec_all.py` and reproduces every shared
quantity. Matched-state width ratios 0.148 / 0.476 / 1.386 agree to three decimals; the
saturation values agree to five (Walker `WML_eps05` neutral s95 0.78950 both scripts, LEAP
`WML_eps01` 0.68890 both). The only differences are bootstrap interval endpoints, which use
different registered rng keys by design.

## 6. What this changes

* The population-dependence result is no longer a static description of two trained arms. It
  **survives an intervention** that moves matched-state width by 6.8× while leaving
  own-occupancy width statistically unchanged.
* The mechanism question in §5 of the notes sharpens: whatever distinguishes the operators
  acts on states off the policy's own occupancy, and the E-step KL budget controls how
  strongly. That is a narrower target than "the E-step is wider".
* Nothing here touches return. `eps01` still costs 20.9 return on G1 and gains 76.4 on
  Walker, and the λ equilibration failure is unaffected.

Artifacts: `reports/artifacts/onarm_eps01_rows.csv` (72 rows, per task/arm/seed),
`onarm_eps01_summary.csv` (27 rows, per statistic), full stdout at
`/hpcwork/qzi10910/gates/onarm_eps01.out`.
