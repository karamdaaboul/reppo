# Blurring the Critic — research status, plan, and running experiments

**Written 2026-09-09 05:30 CEST.** Single source of truth for: what is established, what is
refuted, what the outside evidence says about venue and statistics, what the plan is, and
what is currently queued on the cluster.

Status labels used throughout, and they are load-bearing:

| label | meaning |
|---|---|
| **ESTABLISHED** | measured under a preregistered decision rule that was frozen before the run, with the rule's own gate passed |
| **MEASURED** | measured and reproducible, but not under a pre-registered signed prediction |
| **REFUTED** | a registered prediction failed its own frozen gate |
| **OPEN** | not yet measured, or measured and inconclusive |
| **UNVERIFIED** | outside literature claim that did not survive independent verification — treat as a lead, not a fact |

Standing terminology rule: **width claims are always qualified by population.** Nothing in
this project supports "the weighted-MLE operator is wider" without the qualifier
*matched-state*. On each policy's own visited states the two operators are near parity, and
that contrast is the result, not a caveat to it.

---

## 1. The story in one paragraph

Two actor-update operators — pathwise (SAC-style reparameterised, "PW") and the MPO-style
weighted-MLE E-step ("WML") — are compared inside one codebase (REPPO) where everything but
the actor loss is held fixed. **Part 1** is a controlled setting with a *synthetically
planted* critic-error field, so the ground-truth error is known: the linear pathwise and
centred zeroth-order estimators follow the predicted finite-`M` variance crossover at
`rho_RMS = 1`, but the complete nonlinear E-step does **not** inherit that boundary. **Part
2** is real REPPO training on three MJX tasks: on identical states the two operators'
pre-tanh Gaussian scales differ sharply and consistently, while on each policy's own visited
states they are near parity — a population-dependence result. The originally proposed
mechanism for that gap was tested and **refuted**. A KL-budget intervention
(`eps_e: 0.5 -> 0.1`) moves matched-state width hard in the predicted direction on one task,
closes the operator gap on another, and has inconsistent return effects.

---

## 2. Results ledger

### 2.1 Part 1 — controlled, planted critic-error field

| # | result | status |
|---|---|---|
| P1.1 | Pathwise vs centred zeroth-order estimators cross over at the predicted `rho_RMS = sigma*omega_RMS/sqrt(d) = 1`, replicated at `J = 1, 4, 8` superposed error waves | **ESTABLISHED** |
| P1.2 | The complete nonlinear E-step does **not** inherit that boundary | **ESTABLISHED** |
| P1.3 | Across 26 registered settings the E-step carries `0.13x` the signal-normalised critic-error variance yet `1.32x` the complete-update error | **ESTABLISHED** |
| P1.4 | Decomposition: pathwise loses ~92% of its update error to random critic-error variation; the E-step loses ~71% to a systematic mode-seeking response | **ESTABLISHED** |

Notes that must travel with these numbers:

* The "26/26" figure is **24/26** for the signal-normalised statistic. Do not write 26/26.
* The `4A_0, rho = 3` interval does **not** include zero at aggregate level.
* `rho_RMS^2 = sigma^2 * (g2/e2) / d` from the stored columns. The name-suggested
  `mean_g2 / mean_e2` form is wrong by 79% — this was checked, not assumed.
* `omega_eff = omega / sqrt(J)`.
* The counterfactual decomposition is additive: differences of medians telescope exactly;
  medians of per-cell differences do not. Use the former.

Artifacts: `reports/artifacts/fig_suite/` (six standalone PDFs, identical dimensions),
`fig_planted_error_shapes.pdf`, `fig_controlled_crossover.pdf`, `fig_main/`. Mac copy in
`~/Downloads/section6_1/`.

### 2.2 Part 2 — REPPO training, three MJX tasks, 8 seeds each

Tasks: WalkerRun (`d=6`), G1JoystickFlatTerrain (`d=29`), LeapCubeRotateZAxis (`d=16`).

| # | result | status |
|---|---|---|
| P2.1 | **Matched-state** median pre-tanh sigma is larger for WML than PW, on a neutral bank neither policy generated | **ESTABLISHED** |
| P2.2 | 18/18 split-half comparisons above 1 — the ordering holds on **both** halves of every bank, so it is not an artifact of bank construction (`scripts/analysis/split_half_gate.py`, commit `c709b41`) | **ESTABLISHED** |
| P2.3 | On each policy's **own visited states** the two operators are near parity — the effect is population-dependent, not a property of either policy alone | **ESTABLISHED** |
| P2.4 | The displacement / exponential-tilting mechanism proposed to explain P2.1 | **REFUTED** |

**P2.4 is a first-class result, not a failure to hide.** Under the rule frozen in
`docs/prereg_logsigma_decomp.md` (commit `5edaca3`), job `3823977` at `0948dfc`: the
pathwise curvature residual is systematically positive (0.748 / 0.929 / 0.636 of states on
walker / g1 / leap) and on Walker is *larger* than the WML displacement term —
`median A_WML - median |curv|_PW = -0.0895`, 95% CI `[-0.1016, -0.0777]`, excluding zero.
P3 fails; the verdict follows. Two defects disclosed, both caught by pre-registered checks:
the gate proxy did not reproduce the training gate (prereg used `KL(pi_old || WML fit)`, the
repo gates on `KL(pi_old || pi_theta)`), so P1 is untestable on two of three tasks; and P2
as computed evaluated `spearman(A, sqrt(A))`, monotone by construction, so P2 was not
tested. Neither changes the verdict — P3 is independent of both. The quantitative prediction
also fails (over-predicts 4x on walker WML, ~500x on leap WML), and the seed-population
prediction is refuted 5 cells to 1. What survives: `Term A` is real and non-zero, and the
exact split `dL/dlog_sigma = -(A + B)` is a correct statement about the M-step. What does
not survive is that this term *distinguishes the operators*.

Consequence recorded in advance (`docs/prereg_eps_e_theory_test.md`, commit `0948dfc`): the
`eps_e = 0.1` arm therefore **no longer tests the tilting theory** and is reported as
exploratory / a sensitivity check. That is why the prereg was renamed from
`prereg_budget_matched.md` to `docs/prereg_estep_concentration.md` (commit `4556b6d`) and
why the phrase "budget matching" is retired.

### 2.3 The `eps_e = 0.5 -> 0.1` E-step concentration arm — 24 runs, all completed

One config field differs from the executed WML baseline. `eps_e` was **not** part of the
export tag; that was caught before launch and fixed at commit `964b251` (the launch SHA),
which is the only reason these runs did not overwrite the baselines — the exact failure mode
that previously destroyed two Walker checkpoints. `BASELINES_UNCHANGED = YES`, zero TIMEOUT.

| task | matched-state width ratio (eps01 / eps05) | operator gap WML/PW: before -> after | P6.1 | return delta |
|---|---|---|---|---|
| Walker | **0.148x** [0.079, 0.223] | 22.95x -> 2.43x | HOLDS | **+76.4** [+36.3, +156.0] |
| G1 | **0.476x** [0.322, 0.725] | 2.00x -> **0.965x** (contains 1) | **FAILS** (eta fell 4x) | **-20.9** [-22.0, -16.1] |
| LEAP | 1.386x [0.602, 2.758] (contains 1) | 2.61x -> 2.95x (**not** closed) | HOLDS | -4.6 [-7.2, +5.2] |

**MEASURED.** Reading: tightening the E-step KL budget collapses matched-state width on
Walker by ~6.8x and closes the operator gap entirely on G1 — but the return effects have
**opposite signs across tasks** and LEAP is unresolved in both width and return.

Two problems that must be stated in the paper, not buried:

* **P6.1 fails on G1** because `eta` fell 4x — the arm did not hold the thing it was
  supposed to hold.
* **`lambda` equilibration fails in the new arm on all three tasks** (Walker
  `0.357 -> 3.9e-08`; G1 `0.374 -> 0.128`). **P6.2 is untestable as registered.** This is not
  a small caveat: the gate multiplier collapsing means the trust region is not doing what the
  registered analysis assumed. Note also that on LEAP the `eps_e = 0.5` **baseline** already
  fails equilibration, so LEAP was never going to test P6.2 in either arm.
* **The operator gap does not close on LEAP** (`2.61x -> 2.95x`). Of the three tasks the gap
  closes on exactly one (G1), collapses but persists on one (Walker), and does not move on
  one (LEAP). "The intervention closes the operator gap" is a G1 statement only.

The analysis output is regenerated to `/hpcwork/qzi10910/gates/ec_all_20260909.out`
(it had only ever been printed to a terminal — that gap is now closed). Script:
`/hpcwork/qzi10910/gates/ec_all.py`, applying `docs/prereg_estep_concentration.md` exactly,
bootstrap rng `20260910`, 10000 resamples, bank sha256s verified in-run.

**That output also shows directly why the 48 reruns in §5 are needed:** the `klmean` and
`klwidth` columns read `absent` for both `eps_e = 0.5` baselines and for `PW_noent` on all
three tasks. Only the eps01 arm was run at a commit carrying the KL-split instrumentation,
so at present there is **no baseline to compare its KL split against**. Until the reruns
land, the mean/width decomposition exists for one arm and cannot be used.

### 2.4 Frozen conventions — do not silently change these

* **Ratio convention:** median over seeds of per-seed log ratios, exponentiated.
* **`score_window3`:** mean of the final three logged evaluations (indices 18, 19, 20 of 21),
  frozen at commit `7edb8e8`.
* **Populations:** MATCHED-STATE SCALE / OCCUPANCY-WEIGHTED OWN SCALE / PROXY, defined in
  `docs/protocol_width_populations.md` (commit `2cb7ab7`).
* **On-policy banks:** 32 envs, 6 depths, 192 states per (arm, seed), sha256-keyed,
  `docs/protocol_onpolicy_banks.md` (commit `94d0607`).
* **KL split:** forward `KL(pi_old || pi_theta)` on pre-tanh Gaussians, mean/width split with
  `sigma_theta` in the denominator — **not** the reverse `decoupled_kls` form.
* **alpha is frozen** (`update_entropy_lagrangian: false`) in the no-entropy arms.

---

## 3. What the outside evidence says (deep research, 2026-09-09)

Two research passes ran. The first hit the monthly spend limit with 54/176 agents errored
including the synthesis pass; 12 claims were salvaged manually. The resume
(`wz4uoikaa`, 178 agents) completed but **81 agents failed on certificate errors**, so
only **one** claim survived 3-vote adversarial verification. Treat this section accordingly.

### 3.1 CONFIRMED (3-0, primary source fetched live)

**TMLR's single most important stated acceptance criterion is the claim-evidence link, not
novelty or benchmark performance.** From `jmlr.org/tmlr/acceptance-criteria.html`: "Are the
claims made in the submission supported by accurate and convincing evidence? This is the
most important criterion." The same page disclaims SOTA performance and states novelty "is
not a necessary criterion for acceptance," and its prescribed remedy for a claim-evidence
gap is *additional experiments **or** narrowing the claims*.

This is the best-evidenced venue route for this paper as it currently stands: a mechanistic
result, a refuted candidate mechanism, mixed-sign returns, and an unequilibrated multiplier.

### 3.2 REFUTED — do not lean on these

Three tempting extensions of the above were each voted **0-3**: that a diagnostic paper
"cannot be rejected on not-novel grounds"; that the Walker `+76` / G1 `-21` inconsistency is
"not disqualifying"; and that claim-scoping is a venue-sanctioned substitute for resolving
LEAP and the lambda failure. The policy text is real, the immunity is not. **The sign flip
in our own intervention still has to be explained or scoped, not waved at.**

### 3.3 UNVERIFIED but internally consistent — leads, not facts

The verifier agents for these errored out; the underlying sources are real and mutually
consistent, but nothing here has been independently confirmed.

* **Reporting package** (Agarwal et al. 2021 / rliable; Patterson et al., JMLR v25 2024;
  Jordan et al., ICML 2024): stratified-bootstrap CIs on aggregates, **IQM** rather than
  median or mean, performance profiles instead of per-task mean tables, probability of
  improvement, and publication of every individual run. Patterson et al. go further —
  "do not report standard errors," prefer percentile-bootstrap CIs, and distinguish
  confidence intervals (uncertainty in the mean) from tolerance intervals (per-seed spread).
* **Seeds:** the cited numbers *disagree sharply*, and the disagreement is itself worth
  stating. Agarwal et al. design for 3-10 runs/task and argue for better statistics rather
  than more seeds, but report per-task bootstrap CIs need ~20-30 runs for true 95% coverage
  vs 5-10 for stratified bootstrap on aggregates. Patterson et al. say 5 is almost never
  enough and 30 can be insufficient under skew. Jordan et al. conclude bootstrap CIs
  under-cover below 100 seeds per algorithm-environment pair. **Implication for us: 8 seeds
  x 3 tasks supports an aggregate claim; it does not support the per-task return claims the
  draft currently leans on.**
* **Mixed signs:** prescribed instruments are performance profiles over pooled runs and
  average probability of improvement — but there is a live disagreement about whether
  cross-task aggregation is legitimate at all, so report both aggregate and per-environment
  views and say why.
* **Mechanistic papers are a recognised contribution type** with their own standard —
  falsifiable hypotheses plus controlled confounds, not benchmark wins. Patterson et al.
  draw a categorical line between "scientific studies" and "demonstrations," summarised as
  "Design experiments to provide insights rather than state-of-the-art claims," and endorse
  small diagnostic settings where the experimenter knows the expected outcome (Baird's
  7-state counterexample as the model) as the correct *first* step, with benchmark runs as
  sanity checks. **This directly legitimises the Part 1 -> Part 2 structure.**
* **Elevated review risk:** critical/diagnostic work is reportedly harder to get through
  review, with two named failure modes (indifferent ACs; reviewers affiliated with the
  criticised work). rliable itself is the award-winning counterexample. The NeurIPS
  checklist requires abstract claims to match how far results generalise.
* **Other venues:** RLC/RLJ solicits evaluation methodology and meta-studies as named topics
  (the claim that its awards make negative results award-eligible drew a 1-2 split and is
  the weakest item here); NeurIPS 2026's MLRC track is reachable only *via* TMLR acceptance.
  **ICBINB's ICLR 2026 edition is explicitly not an option.**

### 3.4 NO EVIDENCE RETURNED — needs a fresh targeted search

Three of the five sub-questions came back empty in both passes:

* **Q3** which environment suites and which axes broaden generality for an actor-update-geometry claim (Brax, dm_control, Gymnasium MuJoCo, MetaWorld, MyoSuite, IsaacLab, ManiSkill, DMC-hard);
* **Q4** published protocols for measuring a **learned** critic's error/frequency content on real training data (value-error spectra, TD-error analysis, critic-gradient fidelity, Lipschitz estimation, estimating an effective `omega`);
* **Q5-adjacent** the MPO / weighted-MLE actor-update literature and the policy-entropy / action-scale-collapse literature.

The one applicable item is Patterson et al.'s rejection of "more environments is better" in
favour of deliberate, question-driven selection: **each added suite must answer a stated
question.** For an actor-update-geometry claim the obvious axes are action dimension,
reward sparsity, contact richness, and episode length.

---

## 4. The plan

Ordered by what most changes the paper's standing per unit of compute.

### Tier 1 — closes a hole a reviewer will certainly find

1. **Land the 48 instrumentation reruns** (running, §5). These give the mean/width KL split
   on the *baselines*, which is what turns "matched-state widths differ" into a statement
   about *which part of the KL budget* the two operators spend. Gate: `BASELINES_UNCHANGED`
   must be re-verified bitwise after they land.
2. **Land the eps01 on-policy banks** (running, §5) — prereg §4.2's secondary geometry
   endpoint, and the only way to say whether the eps01 arm's width collapse also shows up
   on its own occupancy.
3. **Re-do the statistics to the rliable standard**: IQM + stratified bootstrap on
   aggregates, performance profiles, probability of improvement, and an appendix table of
   every individual run. This is cheap — it is re-analysis of data already on disk — and it
   is the single most-cited standard in §3.3.
4. **Scope the return claims down.** 8 seeds does not carry per-task return claims under any
   bar cited in §3.3. Either report returns only in aggregate with per-task shown but
   explicitly not claimed, or add seeds. **Decision needed (§6).**

### Tier 2 — strengthens the mechanism story

5. **A mechanism that survives its own gate.** The tilting mechanism is refuted; the paper
   currently has a phenomenon without an explanation. The refutation report itself names what
   a proper test needs: a gate proxy *validated* against the logged `fr_gate_operator` rather
   than assumed; a setting where the linearisation is exact by construction (the LQR and
   planted-critic-error machinery already in this repo); and P2 computed as specified.
   This is the highest-value scientific work left.
6. **Explain or scope the G1 sign flip.** P6.1 fails on G1 because `eta` fell 4x. Either
   instrument `eta` directly across the arm, or state plainly that the intervention did not
   hold `eta` fixed and therefore does not isolate the KL budget.
7. **The lambda equilibration failure.** P6.2 is untestable as registered. Options: longer
   runs to let lambda equilibrate, a different equilibration criterion registered in advance,
   or reporting the failure as a finding about the trust region. **Decision needed (§6).**

### Tier 3 — breadth, only if Tier 1-2 land

8. **One additional environment suite, chosen to answer a stated question** — not "more is
   better." The axis this paper's claim actually depends on is **action dimension** (the
   `rho_RMS = sigma*omega_RMS/sqrt(d)` boundary has `d` in it explicitly), so a suite that
   extends the `d` range while holding the simulator family fixed is worth more than three
   suites that do not. Blocked on Q3 evidence (§3.4).
9. **Estimate an effective `omega` on a real critic** — this is the bridge from Part 1 to
   Part 2 and would let the planted-field prediction be checked against real runs. Blocked on
   Q4 evidence (§3.4); no protocol found yet.

### Explicitly NOT doing

* **MC-oracle scale-up.** Its own pilot concluded "NOT YET PRECISE ENOUGH TO SCALE" and it
  carries an unresolved prereg-level clipped/unclipped estimand defect. It needs an estimand
  redesign before any compute goes into it. Not launched.

---

## 5. Currently queued on the cluster

All submitted 2026-09-09, all currently `PENDING` (queue priority, not an error).
Working tree `/hpcwork/qzi10910/logsplit_wt`, detached at **`964b251`** (the launch SHA),
with its own real `exports/` directory so the canonical tree is unreachable from the jobs.

| job | name | what it is | array | limit |
|---|---|---|---|---|
| `3884816` | `opb01` | on-policy banks for the `eps_e = 0.1` arm — prereg §4.2 secondary geometry endpoint | 1 | — |
| `3884828` | `ls-smoke` | smoke test; the 48 reruns are **dependency-gated** on it reporting `SMOKE = PASS` | 1 | — |
| `3884829` | `ls-walker-WML` | baseline instrumentation rerun | `[1-8]` | 45 min |
| `3884830` | `ls-walker-PW` | baseline instrumentation rerun | `[1-8]` | 45 min |
| `3884831` | `ls-g1-WML` | baseline instrumentation rerun | `[1-8]` | **90 min** |
| `3884832` | `ls-g1-PW` | baseline instrumentation rerun | `[1-8]` | **90 min** |
| `3884833` | `ls-leap-WML` | baseline instrumentation rerun | `[1-8]` | 45 min |
| `3884834` | `ls-leap-PW` | baseline instrumentation rerun | `[1-8]` | 45 min |

**What the 48 reruns produce.** PW_noent and WML_noent at `eps_e = 0.5`, 3 tasks x 8 seeds,
re-run with the KL mean/width split instrumentation added at commit `0502b8f`
(`gaussian_kl_diag_split`, logging `fr_kl_mean_part_med`, `fr_kl_width_part_med`,
`fr_kl_split_resid_max`). Registered in `docs/prereg_estep_concentration_amendment1.md`
(commit `1ff064e`), committed **before** launch.

**Safeguards in place:**

* Preflight on `3884816` confirmed 24/24 checkpoints present and **no existing banks to
  clobber**.
* All 48 canonical baselines hashed to `/hpcwork/qzi10910/logsplit/baselines_before.txt`
  before launch. **These must be re-hashed and bitwise-compared when the reruns finish** —
  that check is not yet done and is the first thing to do on waking.
* G1 got 90 min, not 45. G1 runs take 65-70 min; a 45-min limit would have killed all 16 G1
  jobs. Vindicated by the earlier G1 batch whose first finisher took 48:47.
* Five unit tests are load-bearing: `tests/test_kl_split.py` (T1-T5) imports the **real**
  repo function, and `tests/test_export_tag.py` greps the tag rule out of source.

**On waking, in order:**

```
ssh rwth 'squeue -u qzi10910; sacct -u qzi10910 -S 2026-09-09 -X --format=JobID%13,JobName%14,State%11,Elapsed'
```

1. Confirm `SMOKE = PASS` on `3884828` (if it failed, the 48 never started — that is the
   dependency working, not a bug).
2. Check for `TIMEOUT` states, especially the 45-min tasks.
3. Re-hash the 48 baselines and diff against `baselines_before.txt`; require
   `BASELINES_UNCHANGED`.
4. Bitwise-compare the rerun exports against the canonical baselines — the instrumentation
   is read-only, so the trained weights must be **identical**. Any difference means the
   instrumentation perturbed training and the reruns are void.
5. Analyse the eps01 on-policy banks from `3884816`.

---

## 6. Decisions that are yours, not mine

1. **Returns:** scope the per-task return claims down to aggregate-only, or add seeds?
   (§4.4. The evidence in §3.3 says 8 seeds does not carry per-task claims, but that evidence
   is UNVERIFIED and the cited sources disagree with each other.)
2. **lambda equilibration:** longer runs, a re-registered criterion, or report the failure as
   a finding? (§4.7 — P6.2 is untestable as registered either way.)
3. **Venue:** the only confirmed evidence points at TMLR (§3.1), and the ICLR deadline is
   ~19 days out. This is a strategic call, not a technical one.
4. **New environments:** blocked on Q3 evidence, and the one applicable principle says the
   suite must be chosen to answer a stated question. My recommendation if you want one
   anyway: extend the **action-dimension** range, since `d` is explicit in the `rho_RMS`
   boundary.
5. **Re-run the three empty research questions (Q3, Q4, Q5-adjacent)?** Both passes returned
   nothing on them for infrastructure reasons, not because the literature is empty.

---

## 7. Housekeeping

* **Uncommitted** (held pending your approval, per the standing rule on non-prereg
  artifacts): all figure artifacts (`fig_suite/`, `fig_main/`, `fig_candidates/`,
  `fig_controlled_crossover*`, `fig_planted_error_shapes*`), `reports/controlled_figure_suite.md`,
  `reports/fig_*.md`, `scripts/planted/plot_style.py`, `make_controlled_suite.py`,
  `make_planted_error_shapes.py`, `make_main_figure.py`, `make_candidate_figures.py`,
  `scripts/analysis/build_onpolicy_banks_eps01.py`, `slurm/*.sh`.
* **Commits carry no `Co-Authored-By` or `Claude-Session` trailers** — the repo is anonymised
  for double-blind review.
* **The Overleaf git token was exposed in a session transcript. Rotate it.**

---

## 8. References

Tags record **verification status, not quality**. `CONFIRMED` survived three-vote adversarial
checking against the primary source. `UNVERIFIED` means the verifier agents errored before
reaching it — the source is real and was read, the claim drawn from it was not independently
checked. `SOURCE` is a primary document cited for what it is, with no claim attached.

### A — The algorithm under study

1. **Relative Entropy Pathwise Policy Optimization.** C. Voelcker, A. Brunnbauer, M. Hussing,
   M. Nauman, P. Abbeel, E. Eaton, R. Grosu, A. Farahmand, I. Gilitschenski, 2025.
   `SOURCE` — the codebase everything runs inside. https://arxiv.org/abs/2507.11019
2. **Soft Actor-Critic.** T. Haarnoja, A. Zhou, P. Abbeel, S. Levine, 2018.
   `SOURCE` — the pathwise operator. https://arxiv.org/abs/1801.01290
3. **Maximum a Posteriori Policy Optimisation.** A. Abdolmaleki, J. T. Springenberg, Y. Tassa,
   R. Munos, N. Heess, M. Riedmiller, 2018. `SOURCE` — MPO: the E-step, the temperature dual,
   the weighted-MLE M-step. https://arxiv.org/abs/1806.06920
4. **Relative Entropy Regularized Policy Iteration.** A. Abdolmaleki, J. T. Springenberg,
   J. Degrave, S. Bohez, Y. Tassa, D. Belov, N. Heess, M. Riedmiller, 2018. `UNVERIFIED` —
   the MPO follow-up. The repo's `mstep_decoupled` path (separate mean and covariance
   constraints) is attributed here, but **that attribution was not confirmed against the paper
   text** and should be checked before it goes in the manuscript.
   https://arxiv.org/abs/1812.02256

### B — Evidence standards and venue

5. **TMLR Acceptance Criteria.** `CONFIRMED (3-0)` — the only claim in §3 that survived
   verification; page fetched live 2026-09-09.
   https://jmlr.org/tmlr/acceptance-criteria.html
6. **TMLR Editorial Policies / Reviewer Guide.** `SOURCE`
   https://jmlr.org/tmlr/editorial-policies.html · https://jmlr.org/tmlr/reviewer-guide.html
7. **Deep Reinforcement Learning at the Edge of the Statistical Precipice.** R. Agarwal,
   M. Schwarzer, P. S. Castro, A. Courville, M. G. Bellemare. NeurIPS 2021, Outstanding Paper.
   `UNVERIFIED` — IQM, stratified bootstrap, performance profiles.
   https://arxiv.org/abs/2108.13264
8. **rliable.** Google Research. `SOURCE` — reference implementation of [7].
   https://github.com/google-research/rliable
9. **Empirical Design in Reinforcement Learning.** A. Patterson, S. Neumann, M. White,
   A. White. JMLR 25(318):1–63, 2024. `UNVERIFIED` — scientific studies vs demonstrations,
   diagnostic environments, seed counts.
   https://www.jmlr.org/papers/v25/23-0183.html · https://arxiv.org/abs/2304.01315
10. **Position: Benchmarking is Limited in Reinforcement Learning Research.** S. M. Jordan,
    A. White, B. Castro da Silva, M. White, P. S. Thomas. ICML 2024, PMLR v235. `UNVERIFIED` —
    bootstrap CI coverage below ~100 seeds.
    https://proceedings.mlr.press/v235/jordan24a.html
11. **AdaStop: adaptive statistical testing for sound comparisons of Deep RL agents.**
    T. Mathieu, R. Della Vecchia, A. Shilova, M. Medeiros Centa, H. Kohler, O.-A. Maillard,
    P. Preux. TMLR 2024. `UNVERIFIED` — **directly relevant to decision 1 in §6**: a sequential
    test for how many further seeds are actually needed, rather than guessing.
    https://arxiv.org/abs/2306.10882
12. **Position: ML Conferences Should Establish a "Refutations and Critiques" Track.**
    R. Schaeffer, J. Kazdan, Y. Denisov-Blanch, B. Miranda, M. Gerstgrasser, S. Zhang,
    A. Haupt, I. Gupta, E. Obbad, J. Dodge, J. Z. Forde, F. Orabona, S. Koyejo, D. Donoho,
    2025. `UNVERIFIED` — a proposal, **not an existing track**. Relevant because §2.2's P2.4 is
    a refutation of our own prior claim. https://arxiv.org/abs/2506.19882
13. **Venue routing.** `UNVERIFIED` — the RLC award-eligibility claim drew a 1–2 split and is
    the weakest item in the set.
    https://neurips.cc/public/guides/PaperChecklist ·
    https://rl-conference.cc/callforpapers.html ·
    https://neurips.cc/Conferences/2026/CallForReproducibility ·
    https://sites.google.com/view/icbinb-2026/reviewer-guidelines

### C — Internal provenance

Every empirical number in this document is backed by a preregistration frozen before the run
and a commit, not by a citation.

| Backs | Document | Commit / job |
|---|---|---|
| §2.4 populations | `docs/protocol_width_populations.md` | `2cb7ab7` |
| §2.4 state banks | `docs/protocol_onpolicy_banks.md` | `94d0607` |
| §2.2 split-half, 18/18 | `scripts/analysis/split_half_gate.py` | `c709b41` |
| §2.2 refutation rule | `docs/prereg_logsigma_decomp.md` | `5edaca3` |
| §2.2 consequence for eps_e | `docs/prereg_eps_e_theory_test.md` | `0948dfc` |
| §2.2 measurement | log-sigma decomposition run | job `3823977` |
| §2.3 the arm | `docs/prereg_estep_concentration.md` | `4556b6d` |
| §2.3 return statistic | `score_window3`, frozen | `7edb8e8` |
| §2.3 launch, export tag | `scripts/train_and_export.py` | `964b251` |
| §5 instrumentation | `gaussian_kl_diag_split` | `0502b8f` |
| §5 rerun registration | `docs/prereg_estep_concentration_amendment1.md` | `1ff064e` |
| §5 no-op proof | bit-check, `BITCHECK = PASS` | jobs `3837225` / `3837227` |

**The bit-check is what licenses the §5 reruns.** At commit `0502b8f` the added logging left
actor, critic and normalizer hashes bitwise identical on Walker seeds 301–302, with
`final_eval_return` matching to four decimals (`908.2679`, `891.2316`). The control arm also
reproduced the canonical baseline bitwise across two different c23g nodes, so node-to-node
variation is not a threat either. Residual open question: the reruns are queued on `c23g` while
timestamp matching suggests most canonical baselines came from `c25g` — left as-is by decision.
