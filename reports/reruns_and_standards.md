# The 48 instrumentation reruns, and the experiment against the deep-research standards

2026-09-09. Jobs `3884829`–`3884834`, all 48 `COMPLETED`, zero failures, zero TIMEOUT.
Walker WML 23 min, Walker PW 16 min, G1 WML 68 min, G1 PW 43 min, LEAP WML 29 min, LEAP PW
21 min. The 90-minute limit on G1 was necessary; 45 would have killed all sixteen G1 jobs.

## 1. Gates

**`BASELINES_UNCHANGED = YES`.** All 48 canonical baselines re-hashed against
`/hpcwork/qzi10910/logsplit/baselines_before.txt`: 48/48 match, 0 changed, 0 missing. The
export-tag isolation held; nothing was overwritten.

**`RERUNS_BITWISE_IDENTICAL = NO` — 12/48.** The pattern is completely structured:

| cell | identical |
|---|---|
| Walker PW | **8 / 8** |
| Walker WML | 4 / 8 |
| G1 WML, G1 PW, LEAP WML, LEAP PW | **0 / 8** |

**This gate should not have been set.** `reports/g1_nondeterminism.md` already recorded
`G1_SAME_CODE_REPRODUCIBLE = NO` — same commit, same config, same seed, two runs differing by
1.298e-01 on the actor and 6.654e-01 on the critic — and noted that the identical code is an
exact bitwise no-op on Walker (`0.000e+00`). Bitwise identity was therefore never attainable
on G1, and LEAP behaves the same way. The failure carries no information about whether the
instrumentation perturbed training.

What licenses the reruns instead is the earlier bit-check: at commit `0502b8f`, on Walker,
`TEST_BITWISE_IDENTICAL = YES` — the added logging left actor, critic and normalizer hashes
identical with `final_eval_return` matching to four decimals. The logging is a no-op on the
one task where that statement is testable.

**A new observation, not previously recorded.** On Walker, pathwise is 8/8 bitwise
reproducible while weighted-MLE is 4/8, on the same nodes — seeds 301, 305 and 308 all ran on
`n23g0013`, and 308 reproduced while 301 and 305 did not. Node assignment does not explain
it. The weighted-MLE path is measurably less reproducible than the pathwise path on a task
where the pathwise path is exactly deterministic. Plausibly the softmax over `M` sampled
actions and the dual solve introduce order-dependent reductions; **not tested, and it should
be, because it bears on every bitwise argument this project makes.**

**Consequence for interpretation.** The reruns are valid as *fresh samples of the same arm*
— same code, config and seeds — not as instrumentation of the specific canonical
checkpoints. Every statement below is about the arm, not about those checkpoints.

## 2. The payload — how each operator spends its KL budget

Split residual across all 48 runs: max `2.289e-06`, median `4.114e-07`. The decomposition is
exact algebra and the logging is correct.

| task | arm | mean part | width part | total | width share |
|---|---|---|---|---|---|
| Walker | WML | 0.08124 | 0.00392 | 0.08516 | **4.6%** |
| Walker | PW | 0.07359 | 0.00144 | 0.07503 | 1.9% |
| G1 | WML | 0.09343 | 0.00191 | 0.09534 | **2.0%** |
| G1 | PW | 0.08994 | 0.00096 | 0.09090 | 1.1% |
| LEAP | WML | 0.09390 | 0.00247 | 0.09637 | **2.6%** |
| LEAP | PW | 0.08846 | 0.00131 | 0.08977 | 1.5% |

Paired WML/PW ratio, median over seeds of per-seed log ratios, exponentiated:

| task | mean part | width part |
|---|---|---|
| Walker | 1.134× [1.047, 1.159] * | **2.463× [1.540, 5.628] *** |
| G1 | 1.035× [0.998, 1.222] | **1.975× [1.558, 2.360] *** |
| LEAP | 1.089× [1.007, 1.143] * | **1.892× [1.147, 2.764] *** |

`*` interval excludes 1.

**The two operators spend near-identical KL on moving the mean and roughly twice as much, in
the weighted-MLE case, on changing the width. 3/3 on the width part.** This converts the
descriptive matched-state width finding into a statement about *which part of the trust
region each operator consumes* — a mechanism-adjacent result that does not depend on the
refuted tilting story.

## 3. The return comparison at the aggregate standard

Instruments per Agarwal et al. 2021 and Patterson et al. JMLR 25(318). **The normaliser —
per-task min-max over both arms' 16 runs — is post-hoc and NOT registered. It must be
preregistered before this appears in the paper.**

**Aggregate IQM, stratified bootstrap (runs resampled within task):**

| arm | IQM | 95% CI |
|---|---|---|
| eps05 | 0.6440 | [0.5043, 0.7673] |
| eps01 | 0.3898 | [0.2716, 0.5234] |

**`eps01 − eps05 = −0.2542`, 95% CI `[−0.4213, −0.0530]`, excludes 0.**

**At the aggregate standard the deep research recommends, the E-step concentration
intervention is a net regression.** It cannot be presented as an improvement.

**Probability of improvement** — walker 0.844, g1 0.000, leap 0.484; average **0.443**,
95% CI [0.307, 0.568], **contains 0.5**.

The two aggregate instruments disagree, and the disagreement is the finding: probability of
improvement averages *over tasks* and washes the heterogeneity out, while IQM pools *runs*
and is dragged down by G1, where all eight eps01 runs sit at −5.8 to −7.0 against eps05's
+9.1 to +17.4. Reporting only one of the two would misrepresent the data.

**Per-task, reported alongside and never averaged in:**

| task | paired delta | 95% CI | sign test |
|---|---|---|---|
| Walker | +76.395 | [+36.257, +155.987] | 7/8 positive |
| G1 | −20.929 | [−21.996, −16.098] | **0/8 positive** |
| LEAP | −4.593 | [−7.236, +5.159] | 3/8 positive |

G1 is not noise: it is a uniform sign flip across every seed.

**Performance profile** — eps05 dominates eps01 at every threshold:

| τ | 0.00 | 0.12 | 0.25 | 0.38 | 0.50 | 0.62 | 0.75 | 0.88 | 1.00 |
|---|---|---|---|---|---|---|---|---|---|
| eps05 | 1.00 | 0.92 | 0.88 | 0.71 | 0.62 | 0.54 | 0.42 | 0.29 | 0.08 |
| eps01 | 1.00 | 0.58 | 0.58 | 0.54 | 0.54 | 0.38 | 0.25 | 0.12 | 0.04 |

All 48 individual runs are in `reports/artifacts/rliable_raw_runs.csv`, per Patterson et al.'s
requirement to publish every run.

## 4. What this means for the paper

* **The KL-split result is the strongest thing to come out of these runs.** 3/3 on the width
  part, near-parity on the mean part, exact decomposition. It is a mechanism statement that
  survives the refutation in the earlier log-sigma work.
* **The eps01 arm must be framed as a probe, not a proposal.** It collapses matched-state
  width, closes the operator gap on one task, and costs return in aggregate with an interval
  excluding zero. A mechanistic paper can carry that; a method paper cannot.
* **The seed question is now concrete.** Eight seeds carry the aggregate claim (the IQM
  interval excludes zero) and carry Walker and G1 per-task (both intervals exclude zero, G1 by
  a uniform 0/8 sign test). They do **not** carry LEAP, whose interval spans zero at 3/8.
  Either add seeds on LEAP or state it as unresolved.
* **Registration debt:** the normaliser, the IQM/stratified-bootstrap protocol, and the
  probability-of-improvement statistic are all post-hoc. They should be frozen in a
  preregistration before the manuscript uses them.

Artifacts: `reports/artifacts/klsplit_{rows,summary}.csv`,
`reports/artifacts/rliable_raw_runs.csv`, stdout at `/hpcwork/qzi10910/gates/klsplit.out`
and `/hpcwork/qzi10910/gates/rliable_std.out`.
