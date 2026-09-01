# Claim-supersession map: integrating the uniform empirical-mean audit

Built **before** any manuscript edit. Evidence of record is the frozen-checkpoint
audit at repository HEAD `39c5171`; where an older synthesis conflicts with it, the
audit supersedes.

## Provenance

| | |
|---|---|
| Repo | `~/repos/reppo`, branch `estep-study`, HEAD **`39c5171743e0bea7e238f5618a241b7e6ad17697`** — matches the expected HEAD |
| `git status --short` | *empty*; 0 modified, 0 untracked |
| Repo remote | `origin/estep-study` = `39c5171` (identical) |
| **Manuscript** | **not in the repo** — Overleaf project `6a9023229856313e57f13f30` ("Main Paper"), branch `main`, HEAD `482d96c`, clean, up to date with `origin/main` |
| Manuscript files | `blurring_the_critic.tex` (1990 lines, 105 KB) and `main.tex` (13 lines, an unrelated stub). **These are the only two files in the project.** |
| Baseline compile | `pdflatex`+`bibtex` on CLAIX: 20 pages, 0 errors, 0 undefined references, 0 undefined citations, 0 multiply-defined labels |

### Git history of the evidence reports

| report | introduced | last touched |
|---|---|---|
| `docs/prereg_ubar_ratio.md` | `5912170` | `5912170` (**immutable — not modified**) |
| `reports/ubar_code_trace.md` | `5912170` | `5912170` |
| `reports/ubar_ratio.md` | `0fd0a30` | `0fd0a30` |
| `reports/artifacts/ubar_per_checkpoint.csv` | `0fd0a30` | `0fd0a30` |
| `reports/artifacts/ubar_batch_gradient.csv` | `0fd0a30` | `0fd0a30` |
| `reports/mechanism_evidence_synthesis.md` | `b92c89f` | `b92c89f` (**predates the ubar audit — updated by this task**) |
| `reports/g1_kl_readonly_audit.md` | `7663d03` | `7663d03` |
| `reports/probe4_padding_error_field_results.md` | `b92c89f` | `b92c89f` |
| `reports/planted_error_phase_diagram.md` | `7663d03` | `7663d03` |

### Two structural facts that shape every entry below

1. **The manuscript contains none of the newer evidence.** Keyword sweep of
   `blurring_the_critic.tex`: `ubar` 0, `\bar u` 0, `empirical mean` 0, `WML` 0,
   `g1`/`G1Joystick` 0, `Leap` 0, `d=29` 0, `64-run` 0, `KL gate` 0,
   `critic quality` 0. The paper rests entirely on the older two-task comparison
   (`WalkerRun` $d=6$, `HumanoidRun` $d=21$) plus the contaminated $k=16$ padding
   appendix. Integration is therefore mostly *addition and scope-narrowing*, not
   deletion of existing `ubar` text.
2. **`HumanoidRun` $d=21$ and `G1JoystickFlatTerrain` $d=29$ are different
   experiments.** The paper's headline gap is at $d=21$ ($n=9$ vs $8$); the 64-run
   ladder that the audits analyse uses $d=29$ with $n=8$ per arm. **No $d=21$
   checkpoint exists in the repository**, so no audit speaks to the paper's headline
   runs directly. Every entry keeps them distinct.

---

## Supersession entries

Line numbers are in `blurring_the_critic.tex` at Overleaf `482d96c`.

### S1 — Abstract, crossover as an operator-selection rule

* **Lines** 141–144.
* **Current** "neither method is safer in general. Scoring is the better estimator
  when $\sigma\omega > \sqrt{d}$, where $\sigma$ is the policy's standard deviation,
  $\omega$ is the slope of $e$ divided by its height, and $d$ is the action dimension."
* **Evidence** `reports/planted_error_phase_diagram.md` §3–4.
* **Valid?** **No.** Claim 4 orders only the *critic-error-induced variance of the
  centered value estimator*. The planted sweep confirms that crossover sharply
  (0/240 misclassified, per-slice $r^\*\in[0.974,1.128]$) but shows the operational
  update-MSE ordering does **not** follow: WML beats PW in only 34/76 cells above
  $r=1$, with its own crossover at $r^\*\approx1.67$.
* **Corrected** state the crossover for the error-induced variance channel of the
  centered estimator, and say in the same breath that it is not a general preference
  rule.
* **Reason** forbidden wording under Step 4.3; the sweep separates the two questions.

### S2 — Abstract, empirical result advertised as mechanism confirmation

* **Lines** 147–150.
* **Current** "We detect no difference between the operators at $d=6$ and find a gap
  at $d=21$, in the direction the condition predicts."
* **Evidence** `reports/g1_kl_readonly_audit.md` §7; `reports/mechanism_evidence_synthesis.md` §1.
* **Valid?** **Partly.** The two returns are as stated, but "in the direction the
  condition predicts" reads as mechanism confirmation, and the later 64-run ladder
  finds **no monotone dimension trend** across $d\in\{4,6,16,29\}$ and leaves the one
  detected difference confounded.
* **Corrected** report the two-task outcome as a direction, explicitly not as
  confirmation of the condition, and state that a wider four-task replication found
  no monotone trend.
* **Reason** Step 5 (abstract must not advertise an action-dimension return result).

### S3 — Introduction, crossover statement

* **Lines** 236–249.
* **Current** "Section \ref{sec:crossover} shows that scoring is the better estimator
  when $\sigma\omega>\sqrt d$ … the threshold grows with $\sqrt d$, so the balance
  tips toward differentiating as the action space grows."
* **Evidence** as S1.
* **Valid?** **Scope error.** True of the error-induced variance channel; not of the
  update or the return.
* **Corrected** scope to the error channel; retain the intuition; add that the
  controlled test confirms the channel and that the operational advantage does not
  transfer one-for-one.

### S4 — Introduction, empirical summary

* **Lines** 267–271.
* **Current** "We detect no operator difference at $d=6$ and find a gap at $d=21$, in
  the direction the condition predicts."
* Same treatment as S2.

### S5 — Contribution 2

* **Lines** 290–293.
* **Current** "A crossover condition, $\sigma\omega>\sqrt d$, written in properties of
  the critic's error…"
* **Corrected** add "…for the error-induced variance of the centered value estimator",
  and add that it is now verified in a controlled planted-error setting.

### S6 — Contribution 3

* **Lines** 294–297.
* **Current** "no detected difference at $d=6$, a gap at $d=21$, and a
  scale-normalised diagnostic…"
* **Corrected** add the four-task replication with no monotone trend, and that the one
  detected difference is confounded by a KL-gate asymmetry.

### S7 — Claim 4 statement and surrounding prose

* **Lines** 563–612 (`claim:crossover`, `eq:crossover`, the "counts how many
  oscillations" paragraph).
* **Current** the claim is already scoped to `Var[.]_e` inside the display, but the
  prose around it ("scoring wins", "the balance tips") generalises it.
* **Valid?** Claim statement valid; surrounding prose over-generalises.
* **Corrected** keep the claim verbatim; add an explicit scope paragraph naming the
  five things it does **not** order (total variance, exact $c$, full $v$, the neural
  actor gradient, trust-region update MSE, return).

### S8 — `\gzo` treated as the implemented E-step

* **Lines** 333–338 (`eq:estimators`), 555–562, 1546–1615 (Lemma `lem:zocov`), and
  every prose use of "zeroth-order operator" to mean the thing the code runs (32
  occurrences of `zeroth-order`).
* **Evidence** `reports/ubar_code_trace.md` §0.1–0.3; `reports/ubar_ratio.md` §2, §5.
* **Valid?** **No, as a description of the implementation.** The implemented mean
  score is exactly $v=\bar u+c$; $c$ reduces to $\hat m/\eta$ only in the small-logit
  regime, which fails its preregistered adequacy criteria in **9 of 10** audited
  conditions (residuals to $5.364$).
* **Corrected** keep `\gzo` as the *analysed estimator*; add an explicit statement
  that the implemented operator is $v=\bar u+c$ and that the arm labelled
  "zeroth-order" in the experiments is the implemented weighted-MLE E-step, not
  `\gzo`.

### S9 — Population first-order equivalence

* **Lines** 401–457 (`sec:same`, `prop:estep`, `cor:identical`), 1616–1691.
* **Valid?** **Yes, unchanged.** The audit is a finite-sample and implementation
  result and does not touch the population identity. Explicitly preserved.

### S10 — No statement of $\bar u_{\rm raw}$ vs $\bar u_{\rm fit}$

* **Location** absent; belongs in `sec:same`/`sec:crossover`.
* **Evidence** `reports/ubar_ratio.md` §6, §12.
* **Corrected** add: $\E\|\bar u_{\rm raw}\|^2=d/M$ (an RMS-squared identity, matched
  to within 0.53% over all 30 checks); the tanh transform is $\mu$-independent and
  cancels to $1.1\times10^{-15}$; the hard clip binds on up to **20.7%** of samples,
  raises RMS $\|\bar u\|$ by up to $3.0\times$ and induces a systematic mean vector up
  to $\approx18\times$ the raw one. $d/M$ must not be applied to $\bar u_{\rm fit}$.

### S11 — Padding appendix: pending analyses

* **Lines** 1922–1928 (`\expt{Run and report the committed offline analyses…}`),
  1018–1022 (`neither may be described as completed`).
* **Evidence** `reports/probe4_padding_error_field_results.md`.
* **Valid?** **Stale.** The restricted-$z$ oracle (Probe 1) and the crossed
  same-critic table (Probe 4) are complete.
* **Corrected** report Probe 4's **mixed** outcome in full: paired median
  $D_{\rm WML}-D_{\rm PW}=-0.0019$ on A-trained critics (**0/5** positive) and
  $+0.0137$ on B-trained (**5/5**), identically under both reference laws; all 10
  padded checkpoints trip the registered width gate; no return conclusion.

### S12 — Padding appendix: width ratios

* **Lines** 1885–1893 (table: width ratio PW $1.80$, ZO $4.25$).
* **Evidence** `reports/probe4_padding_error_field_results.md` §5.
* **Valid?** **Yes, and independently corroborated.** The regenerated checkpoints give
  1.74–1.93 (arm A) and 3.64–5.58 (arm B), 10/10 tripping the rule. Add the
  corroboration; change no number.

### S13 — "clean" at $d=21$

* **Line** 933 ("The zeroth-order arm is otherwise clean at $d=21$").
* **Valid?** **Narrowly.** It is a statement about $\sigma$, $\eta$, ESS and
  excursions on those runs, not about construct validity.
* **Corrected** replace "clean" with a statement of exactly which diagnostics were in
  range, and note that the KL gate was not among them.

### S14 — Limitations: two-point scaling

* **Lines** 1335–1350.
* **Corrected** add the four-task ladder with no monotone trend; the absence of any
  $d=21$ checkpoint; the unpaired seeds and differing $\alpha$ in the only matched
  $d$ contrast; missing optimizer state; KL-gating scope; no learned-critic $\omega$.

### S15 — $\omega$ and critic quality

* **Lines** 1352–1359, 989–995.
* **Valid?** **Yes** — the paper already says $\omega_\infty$ is not measured. Must
  **not** be weakened into "critic quality was eliminated". Add that the frozen
  same-critic probe measures variance conditional on $Q_\phi$, not error relative to
  $Q^\pi$.

### S16 — LaTeX defect, nested `abstract`

* **Lines** 124–125 and 154–155: `\begin{abstract}` and `\end{abstract}` each appear
  twice. Pre-existing; fixed as part of this pass.

### S17 — Missing bibliography

* **Line** 1399 `\bibliography{references}`; the Overleaf project has **no
  `references.bib`**, and lines 1362–1394 instruct edits to that non-existent file.
* **Not fixed here.** Fabricating 46 bibliography entries would be worse than the
  known gap. Reported as an open defect.

---

## Statements searched for and **not** found in the manuscript

None of the following appear, so nothing had to be deleted for them:
`ubar is the likely explanation of the high-dimensional gap`; `irreducible
sqrt(d/M)`; `the implemented E-step is g_ZO plus ubar`; `critic quality was
eliminated`; `the 64 runs are clean`; `the KL defect explains g1`; `dimension
amplification is observed`; `removing ubar should improve return`. The forbidden
claims that *do* exist are S1/S3 (`the E-step wins when sigma omega > sqrt d`) and
S2/S4/S6 (`the high-dimensional result supports Claim 4`, in the softer form "in the
direction the condition predicts").
