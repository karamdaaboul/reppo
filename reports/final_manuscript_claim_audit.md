# Final manuscript claim audit

Reconciliation of "Blurring the Critic" with the frozen-checkpoint evidence at
repository HEAD `39c5171`. No experiment or training job was launched.

## 1. Provenance

| | |
|---|---|
| Evidence repo | `~/repos/reppo`, branch `estep-study`, HEAD **`39c5171743e0bea7e238f5618a241b7e6ad17697`** — matches the expected HEAD; tree clean, 0 modified, 0 untracked at the start of this task |
| Manuscript | **not in the repo** — Overleaf project `6a9023229856313e57f13f30`, branch `main`, parent commit `482d96c`, clean and level with `origin/main` |
| Manuscript files | `blurring_the_critic.tex` (1990 lines) and `main.tex` (an unrelated 13-line stub). **The project contains no other files.** |
| Preregistration | `docs/prereg_ubar_ratio.md` at `5912170` — **verified unchanged**, `git diff --quiet` clean |

Evidence report histories are tabulated in `reports/manuscript_claim_supersession.md`
and were each verified with `git log -1 -- <file>`.

## 2. Files changed

| File | Change |
|---|---|
| `blurring_the_critic.tex` (Overleaf) | +329 / −45 lines over 2 commits (`3059cda`, `ab60455`); 17 anchored edits, every one asserted unique before application |
| `reports/mechanism_evidence_synthesis.md` | new §5 (ubar strand), §4 scoped and cross-referenced, decision table replaced with 14 rows, sections renumbered |
| `reports/ubar_ratio.md` | five numerical passages corrected (see §6) |
| `reports/manuscript_claim_supersession.md` | **new** — Step 1 deliverable |
| `reports/final_manuscript_claim_audit.md` | **new** — this file |
| `docs/prereg_ubar_ratio.md` | **untouched** |

## 3. Section-by-section scientific changes

**Abstract.** Removed "Scoring is the better estimator when `sigma*omega > sqrt(d)`";
replaced with the error-induced-variance scoping plus an explicit statement that the
crossover is not an operator-selection rule. Removed "in the direction the condition
predicts" as a mechanism claim; added the four-task replication with no monotone trend
and the confounding. Added one sentence introducing the implemented-operator audit.
Also repaired a pre-existing LaTeX defect: `abstract` was opened and closed twice.

**Introduction.** Scoped the crossover display to the error channel; added a forward
pointer to the controlled test and the finding that the update error does not reorder
at the same threshold. Rewrote the empirical summary. Revised contribution 2 (scope +
planted result), contribution 3 (four-task replication, confounding), and added a new
contribution for the implemented-operator audit.

**Theory.** Added a scope paragraph to Claim 4 naming the six things it does not order.
Added a new section `sec:implemented`, "What the implemented E-step actually computes",
which sets out the four-object hierarchy: the population identity (preserved
unchanged); the centred estimator `m_hat` (eq. `eq:mhat`); the exact decomposition
`v = ubar + c` (eq. `eq:vdecomp`) with `c ~= m_hat/eta` labelled a local expansion; and
the neural actor update described through `L_full = L_uniform + L_centered` (eq.
`eq:lossdecomp`), explicitly scoped to states where the objective is active. Added the
`ubar_raw` / `ubar_fit` distinction, the RMS-not-median framing of `d/M`, the exact
cancellation of the tanh Jacobian, and the clipping-induced bias.

**Experiments.** Three new subsections: `sec:ladder` (four-task replication, gate
readout, confounding), `sec:planted` (planted-error crossover and its operational
non-transfer), `sec:ubar` (the preregistered audit: P1 failure, orthogonality,
refuted dimension amplification, batch-gradient materiality). Softened the `d=21`
direction claim in `sec:operator` and replaced the unqualified "clean".

**Appendix.** The padding appendix's "run these analyses" note is replaced by the
completed crossed same-critic result, reported as **mixed** in full, with the
byte-identical regeneration provenance and the reaffirmed contamination.

**Limitations.** Added: no monotone trend across four tasks and an unexplained
confounded difference; unmatched seeds and differing `alpha` in the only matched `d`
contrast; no `d=21` checkpoint; absent optimizer state; the gating scope of the
decomposition; and that the frozen probes measure variance conditional on `Q_phi` and
so do not eliminate critic quality.

## 4. Old versus new claim table

The 14-row table lives in `reports/mechanism_evidence_synthesis.md` §7, each row
carrying allowed wording, forbidden wording and an exact evidence source. Summary of
status changes:

| Claim | Old | New |
|---|---|---|
| Same blurred-critic estimand | supported | **supported** (unchanged) |
| Population first-order identity | supported | **supported** (unchanged) |
| First-order description of implemented WML | implicit, treated as exact | **unsupported generally at operating logits** |
| Exact `v = ubar + c` | absent | **supported** (new row) |
| `ubar` dimension-amplified | conjectured in the earlier synthesis | **refuted in the matched probe** |
| `ubar` dominant at `d=29` | conjectured | **not supported**; centred component larger |
| Clipping-induced fitted-score bias | absent | **supported** (new row) |
| Claim 4 error-variance crossover | supported | **supported** (unchanged) |
| Claim 4 as an operational rule | asserted in abstract/intro | **unsupported** |
| Probe 4 | mixed | **mixed** (unchanged) |
| Dimension return trend | unidentified | **unidentified** (unchanged) |
| Cause of the g1 gap | unidentified | **unidentified and confounded** (unchanged) |

## 5. Stale claims removed

| Claim | Where | Disposition |
|---|---|---|
| "Scoring is the better estimator when `sigma*omega > sqrt(d)`" | abstract, intro | rewritten to the error-variance channel |
| "in the direction the condition predicts" | abstract, intro | removed as a mechanism claim |
| "in the direction `\eqref{eq:crossover}` predicts" | `sec:operator` | rewritten; direction reported, inference declined |
| "the zeroth-order arm is otherwise **clean** at `d=21`" | `sec:operator` | replaced by the specific diagnostics, with the gate named as not covered |
| "the balance tips toward differentiating as the action space grows" | intro | scoped to the error channel |
| "irreducible `O(sqrt(d/M))`" | synthesis §4 | "irreducible" withdrawn; extrapolation superseded by §5 |

Searched for and **absent from the manuscript**, so nothing needed removing:
"`ubar` is the likely explanation of the high-dimensional gap"; "the implemented E-step
is `g_ZO` plus `ubar`"; "critic quality was eliminated"; "the 64 runs are clean"; "the
KL defect explains g1"; "dimension amplification is observed"; "removing `ubar` should
improve return"; "antithetic sampling repairs g1".

## 6. Errors found in the evidence reports during validation

Cross-checking every manuscript number against the source CSVs surfaced a real
transcription error in `reports/ubar_ratio.md`, now corrected:

* The walker arm-B clip rate was written as **20.7%**, which is the *batch-level*
  figure from `ubar_batch_action.csv`. The Step-2 condition median is **22.1%**, and
  the per-checkpoint maximum anywhere is **43.3%** (leap arm B, seed 106). Corrected in
  three places plus the report's headline.
* "RMS inflation up to 3.0x" and "worst `|ratio-1|` 0.53% over 30 checks" were
  condition-median statements presented without that qualifier. Both now state the
  scope, and the per-checkpoint figures (15.1x; 2.13% over 222 checks) are given
  alongside.
* The mean-vector inflation "18x" is confirmed exactly (18.4x).

All other 33 checked values matched their sources within tolerance.

## 7. Remaining unresolved claims and defects

1. **The Overleaf project has no `references.bib`.** `\bibliography{references}` cannot
   resolve, and lines ~1360–1394 instruct edits to that non-existent file. Not fixed
   here: fabricating 46 bibliography entries would be worse than the known gap. **This
   must be supplied before submission.**
2. `HumanoidRun` (`d=21`) has no retained checkpoint, so no audit speaks directly to
   the headline table's runs. The manuscript now says so.
3. The `\verify{}`/`\action{}`/`\expt{}` notes unrelated to this reconciliation are
   left in place; they are the authors' own queue.
4. The cause of the `d=29` difference is unresolved and requires corrected reruns.
5. `omega` is unidentified on every learned critic.
6. The `k=6` (`d=12`) padding level is still described as pending.

## 8. Compilation

```bash
# on CLAIX; texstub/todonotes.sty and a stub references.bib are BUILD-ONLY
cd ~/tex_baseline
export TEXINPUTS=./texstub:
pdflatex -interaction=nonstopmode blurring_the_critic
bibtex   blurring_the_critic
pdflatex -interaction=nonstopmode blurring_the_critic
pdflatex -interaction=nonstopmode blurring_the_critic
```

| | before edits | after edits |
|---|---|---|
| PDF produced | yes | yes |
| pages | 20 | **24** |
| hard errors | 0 | **0** |
| undefined references | 0 | **0** |
| undefined citations | 0 (with stub bib) | **0** (with stub bib) |
| multiply-defined labels | 0 | **0** |

All seven new labels (`sec:implemented`, `sec:ladder`, `sec:planted`, `sec:ubar`,
`eq:mhat`, `eq:vdecomp`, `eq:lossdecomp`) resolve. A final automated sweep confirms
zero occurrences of each forbidden formulation, a balanced `abstract` environment, the
`ubar_raw`/`ubar_fit` distinction present, `eta` provenance stated, and the explicit
statement that states are never pooled across seeds. Without the build stubs the
document does not compile on this machine at all, because `todonotes` is unreadable in
the system tree and `references.bib` does not exist; that is true of the unmodified
document as well.

## 9. Diff summary

```
# Overleaf, 482d96c..ab60455
blurring_the_critic.tex | 377 +++++++++++++++++++++++++++++++++-----------
1 file changed, 329 insertions(+), 45 deletions(-)
```

Overleaf commits: `3059cda` (main reconciliation), `ab60455` (eta provenance).
Repository commit: `5de9b3e` (report reconciliation), on top of `39c5171`.

Repository: `reports/mechanism_evidence_synthesis.md` and `reports/ubar_ratio.md`
modified; `reports/manuscript_claim_supersession.md` and this file added;
`docs/prereg_ubar_ratio.md` verified byte-identical.
