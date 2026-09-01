# Probe 4 — crossed same-critic operator table against the padded reference

**Protocol.** `docs/prospective_padding_error_field_analysis.md`, sha256
`34dd111af742750c3f96258b15f460ddd867dc42510dceaa73db7125f93e409b`, executed as
committed. The committed row, verbatim:

> | 4 | Crossed same-critic table vs $\bar Q_\phi$ | Primary $D$: squared
> $\Sigma_x^{-1}$-norm of real-update change; secondary $L$, cosine, weight KL, ESS |
> Prediction: WML is affected more. Falsified if paired median
> $D_{\rm WML}-D_{\rm PW}\le0$. Evaluate each critic under both operators. |

**Provenance.** Analysis commit `6da5ad5` (probe) / `7534b77` (regeneration), branch
`estep-study`. JAX 0.5.2, Python 3.12.14, NVIDIA H100 94 GB, SLURM `c23g`,
account `rwth2182`. Probe jobs `3431304`, `3431369`. 111 s per (seed, law) cell.

**Ordering caveat.** The committed plan orders Probes 0–7 and Probe 4 is executed
here with **Probes 2 and 3 not yet run**. Probe 0 (Amendment A) and Probe 1 are on
record. This is a departure from the committed execution order and is reported as
such; it does not alter Probe 4's own definition, prediction or falsification rule.

---

## 1. Checkpoints

The 10 required checkpoints **did not exist on CLAIX**. Amendment A records that the
originals were produced at `3b96deb` in `~/workspaces/reppo_original` on a different
machine; they are not on this cluster and carry no ledger entry. They were
regenerated here.

**The regeneration preserves the original training process.** `slurm/pad16_parity.sh`
ran the identical config and seed under `3b96deb` and under the analysis commit and
compared every exported array:

```
arrays compared          : 35
BYTE-IDENTICAL arrays    : 35/35
max abs param difference : 0.000000e+00
final_eval_return        : 401.646728515625 both
eval_return_curve, kl_curve : identical
```

The only `src/` difference between the two commits is additive and default-off
(`log_estimator_diag`, `log_eval_iqm`), so **all three known defects are preserved**,
which is required: Probe 4 analyses critics produced by the original process, not
corrected training. No defect was fixed.

| task | k | seed | arm | expected path | found | integrity | d | pad | min_std | alpha | steps | return | checksum |
|---|--:|--:|---|---|---|---|--:|--:|--:|--:|--:|--:|---|
| WalkerRun | 16 | 0 | A | `exports/WalkerRun_pathwise_fa_pad16_s0_final` | FOUND | PASS | 22 | 16 | 0.1 | 0.01528 | 52297728 | 903.57 | `e23de5b89d19b9b9` |
| WalkerRun | 16 | 1 | A | `…_pathwise_fa_pad16_s1_final` | FOUND | PASS | 22 | 16 | 0.1 | 0.01528 | 52297728 | 910.33 | `6c00ed4d1afb89c1` |
| WalkerRun | 16 | 2 | A | `…_pathwise_fa_pad16_s2_final` | FOUND | PASS | 22 | 16 | 0.1 | 0.01528 | 52297728 | 909.58 | `25aae9ab2aa926d9` |
| WalkerRun | 16 | 3 | A | `…_pathwise_fa_pad16_s3_final` | FOUND | PASS | 22 | 16 | 0.1 | 0.01528 | 52297728 | 907.12 | `cfa1ebf9508cc86c` |
| WalkerRun | 16 | 4 | A | `…_pathwise_fa_pad16_s4_final` | FOUND | PASS | 22 | 16 | 0.1 | 0.01528 | 52297728 | 902.13 | `e958c5fb00ea3f89` |
| WalkerRun | 16 | 0 | B | `…_weighted_mle_pad16_s0_final` | FOUND | PASS | 22 | 16 | 0.1 | 0.01528 | 52297728 | 571.42 | `e55f63927d827a51` |
| WalkerRun | 16 | 1 | B | `…_weighted_mle_pad16_s1_final` | FOUND | PASS | 22 | 16 | 0.1 | 0.01528 | 52297728 | 630.11 | `722bdeec61b51171` |
| WalkerRun | 16 | 2 | B | `…_weighted_mle_pad16_s2_final` | FOUND | PASS | 22 | 16 | 0.1 | 0.01528 | 52297728 | 651.54 | `56647d9a9c4a0f6e` |
| WalkerRun | 16 | 3 | B | `…_weighted_mle_pad16_s3_final` | FOUND | PASS | 22 | 16 | 0.1 | 0.01528 | 52297728 | 733.09 | `df8dbc0501443506` |
| WalkerRun | 16 | 4 | B | `…_weighted_mle_pad16_s4_final` | FOUND | PASS | 22 | 16 | 0.1 | 0.01528 | 52297728 | 456.61 | `49d0f8e8e9fa81f6` |

Git SHA `7534b774aa44` for all ten. **FOUND 10/10, INTEGRITY PASS 10/10.**

Every launch parameter is recovered from the *preregistration*
(`docs/prereg_action_padding.md`: alpha frozen at 0.01528, 5 seeds per arm per level,
arms A-frozen pathwise and B-frozen weighted_mle with eps_e=0.5, M=32,
mstep_decoupled=false), not inferred. Only the literal shell command is
unrecoverable. **These are regenerated instances, not the originals**: Probe 1's
published numbers were computed on the originals, so Probe 1 and Probe 4 currently
sit on different checkpoint instances. Re-running Probe 1 here would put them on a
common basis and has not been done.

---

## 2. What `D` is, written out

Work in the checkpoint's whitened pre-tanh metric, where `Sigma = diag(sigma_c^2)`.
The committed trust-region step is `Delta_mu = sqrt(2 eps) Sigma ghat / ||ghat||_Sigma`
(`wasted_step_fraction_proposition.md` Sec. 1, resolved against `blurring_v6`
`prop:estep`). Whitening the displacement gives `Delta_mu_w = sqrt(2 eps) hhat/||hhat||_2`
with `hhat = Sigma^{1/2} ghat`, and `||v||^2_{Sigma^-1} = ||v_w||^2_2` exactly. So

```
D = 2 eps * || (uhat/||uhat||)_x  -  (ubar/||ubar||)_x ||^2_2
```

`uhat` the operator's whitened direction from `Q_phi`, `ubar` the same operator's
direction from `Qbar_phi = E_z[Q_phi(s,x,z)]`. `2 eps` multiplies both operators
identically, cancels from the sign of `D_WML - D_PW` and from their ratio, and is set
to 1. The identifying invariance is `Q^pi(s,x,z) = Q^pi(s,x)` — verified structurally,
not assumed (Sec. 5 check 1) — so centred variation of `Q_phi` in `z` is critic error.

Operators follow Amendment A: PW differentiates `Q_phi` at **unclipped** post-tanh
actions; WML uses the **raw self-normalised softmax** `w_i = softmax(Q_i/eta)` with no
centring or baseline, on **clipped** actions (`src/jaxrl/reppo.py:669`), with `eta`
read verbatim from the checkpoint.

**Recorded ambiguities** (plan Sec. 5: record, do not silently resolve):

* **A1 — clip under a crossed table.** Amendment A fixes the clip per *arm*; Probe 4
  varies the *operator* inside a fixed critic. Most literal reading: the clip belongs
  to the WML operator, since that is the code path it is a fact about. Applied that way.
* **A2 — no Probe-4 z budget is registered** (only Probe 1's oracle stream is). Set
  `N_Z = 1024` with an independent 256-draw pre-pass for centring, and the MC standard
  error is reported so convergence is visible rather than asserted.
* **A3 — no `eta` exists on the pathwise checkpoints.** They have no `eta_param` at
  all. Where it exists it is read verbatim (arm B: 0.0411–0.0556); where it does not,
  it is constructed from the operator's own registered definition — the MPO dual at
  the registered `eps_E = 0.5`, solved as a single batch-shared scalar per answer 4
  (arm A: 0.362–0.514). Source and value recorded per cell in
  `reports/artifacts/probe4_eta.csv`.

**Pairing.** Matched checkpoints; a common state set per seed built from equal
complete lanes under both arms' policies (1024 + 1024 = 2048 states), each critic
applying its own normalizer to identical raw observations; one common set of `M = 32`
pre-tanh action draws per state, half from the A law and half from the B law with
source labels retained, shared by all four cells; common random numbers throughout.

**Budget.** `M = 32` critic evaluations per state for every operator. PW additionally
requires `M` backward passes; WML requires none. Per (seed, law) cell:
`2048 x 32 = 65,536` critic forwards for `Q_phi`, `2048 x 32 x (1024 + 256) = 83.9M`
forwards for `Qbar`, and `2048 x 32 x 128 = 8.4M` backward passes for the PW
reference gradient. The backward-pass asymmetry favours WML and is **reported, not
corrected for**.

---

## 3. Primary result

Statistical unit = the training seed/checkpoint (n=5), never the individual state.
Aggregation within checkpoint first, then seed-level paired contrasts. Seed median is
the committed summary; mean secondary.

### Checkpoint law (primary)

| seed | `A_PW` | `A_WML` | `B_PW` | `B_WML` | **B: WML−PW** | **A: WML−PW** |
|--:|--:|--:|--:|--:|--:|--:|
| 0 | 0.0021271 | 0.00022619 | 0.017539 | 0.033055 | **+0.015517** | **−0.0019009** |
| 1 | 0.0024526 | 0.00019771 | 0.0095256 | 0.027721 | **+0.018195** | **−0.0022548** |
| 2 | 0.0021230 | 0.00032013 | 0.017054 | 0.026591 | **+0.0095367** | **−0.0018029** |
| 3 | 0.0034923 | 0.00020410 | 0.0066511 | 0.017566 | **+0.010915** | **−0.0032882** |
| 4 | 0.0013365 | 0.00013734 | 0.0055182 | 0.019216 | **+0.013698** | **−0.0011991** |

| critic | paired median | mean | sign | exact two-sided p | hierarchical bootstrap 95% CI | committed verdict |
|---|--:|--:|--:|--:|---|---|
| **A-trained** | **−0.0019009** | −0.0020892 | **0/5** positive | 0.0625 | [−0.0031955, −0.0012872] | **FALSIFIES** (median ≤ 0) |
| **B-trained** | **+0.013698** | +0.0135724 | **5/5** positive | 0.0625 | [+0.0095977, +0.0178959] | **SUPPORTS** |

### Common standardized law `z ~ N(0, I_k)` (mandatory sensitivity)

| critic | paired median | sign | 95% CI | verdict |
|---|--:|--:|---|---|
| **A-trained** | **−0.0018309** | **0/5** positive | [−0.0034031, −0.0012961] | **FALSIFIES** |
| **B-trained** | **+0.0222636** | **5/5** positive | [+0.0166551, +0.0288264] | **SUPPORTS** |

Both laws give the same qualitative split, so it is **not** an artefact of the
padded-coordinate width contamination (Sec. 5 check 9).

Bootstrap: 10,000 hierarchical replicates, `np.random.default_rng(20260831)`,
resampling seeds → whole lanes within seed → samples within lane, percentile
intervals — exactly as committed. `p = 0.0625` is the **smallest** two-sided value an
exact sign test can return at n=5; with five seeds neither direction can be
established at conventional significance, and this is not presented as though it
could be.

### Classification: **B — MIXED**

The committed prediction "WML is affected more" holds decisively on the B-trained
critics (5/5 seeds, both laws) and is **reversed** decisively on the A-trained critics
(0/5 seeds, both laws). The committed falsification rule is a single inequality and
does not nominate a primary critic; evaluating each critic under both operators, as
the same row requires, produces opposite verdicts. **The ordering depends on critic
source, which is category B.** This is reported as a mixed result, not resolved in
favour of the supporting half.

The dominant effect in the table is not the operator at all. `D` moves by roughly
**4–8x** between the A-trained and B-trained critic at fixed operator, and by
**~10x downward** (A) or **~2.8x upward** (B) between operators at fixed critic. The
critic that arm B's own training produced is far more contaminated in its inert
coordinates than arm A's, under *either* operator.

---

## 4. Secondary and exploratory quantities

Committed secondaries (`L`, cosine, weight KL, ESS), seed medians, checkpoint law:

| cell | `D` | `L` | cosine | weight KL | ESS | `V_e<=0` | `Qbar` MC s.e. |
|---|--:|--:|--:|--:|--:|--:|--:|
| A_PW | 0.0021271 | 0.0506 | 0.97319 | — | — | 0 | 0.00436 |
| A_WML | 0.00020410 | 0.4586 | 0.99947 | 0.00217 | 21.8 | 0 | 0.00436 |
| B_PW | 0.0095256 | 0.1036 | 0.94184 | — | — | 0 | 0.00397 |
| B_WML | 0.026591 | 0.6785 | 0.92828 | 0.1468 | 5.65 | 0 | 0.00397 |

**`L` is the largest effect in the whole probe.** `L` is the fraction of the
trust-region step spent in the `k=16` coordinates the simulator provably discards. PW
spends 5% (A) and 10% (B); **WML spends 46% (A) and 68% (B)**. An isotropic step would
be `k/d = 16/22 = 0.727`. So the WML operator's step is close to isotropic — it
allocates its trust region across padded and real coordinates almost as if the padded
ones carried signal — while PW concentrates on the real block. This is stable across
seeds and identical under both reference laws (A_WML 0.459/0.469, B_WML 0.679/0.677).

**This is exploratory, not preregistered as a directional prediction.** `L` is listed
as a committed *secondary*, but the plan attaches no prediction or threshold to it,
and Probe 1's row explicitly warns that `L` is degenerate in restricted-z settings.
It is reported as a descriptive quantity. It is also the quantity most obviously
connected to Sec. 5 of `reports/planted_error_phase_diagram.md` — the live `ubar`
term, which makes the raw-softmax E-step displace the mean isotropically — but that
connection is a hypothesis, generated after seeing this table, and is **not** tested
here.

ESS is 21.8 (A) vs 5.65 (B) out of `M=32`: the B-trained weights are far more
concentrated, consistent with the 8–13x smaller `eta` read from its checkpoints.

---

## 5. Integrity and falsification checks

| # | check | result |
|--:|---|---|
| 1 | padded coordinates do not enter dynamics or reward | **PASS** — `scripts/verify_action_pad.py 16`: a recording shim below the pad shows the simulator receives dim **6**, want 6; actor/critic/E-step all use 22; critic rejects a 6-dim action |
| 2 | action preprocessing does not leak z into x | **PASS** — `ActionPad` slices the first 6 coordinates; the same shim confirms it |
| 3 | actor covariance handled as committed | **PASS** — diagonal state-dependent `sigma`; effective `min_std = 0.1` on all 10 checkpoints, confirming Amendment A.1 item 1 (the configured 0.0 is never plumbed through) |
| 4 | gradients through z not accidentally stopped | **PASS** — `\|grad_z Q\|` is 0.051 against `\|grad_x Q\|` 0.351 (≈14%) on a coordinate whose true gradient is exactly zero; a stopped gradient would read 0 |
| 5 | `Qbar` converges under the chosen z budget | **PASS** — MC s.e. of `Qbar` is 3.12% of the padded `Q` spread in every cell |
| 6 | `V_e <= 0` tripwire (Amendment A.1 item 2) | **PASS** — **0** of 2048x32x4x5x2 state-action-cells across both laws; moments centred on an independent pre-pass and reduced in float64 |
| 7 | PW and WML start from the identical base policy | **PASS** — same checkpoint, same `mu_c`, `sigma_c`, same common action draws |
| 8 | update norms / trust regions comparable | **PASS** — both normalised to the same trust region by construction; zero-step frequency 0.0000 (A_PW, A_WML, B_PW) and 0.0073 (B_WML) |
| 9 | no NaN; clipping saturation not dominant | **PASS** — no NaN in any eval; the σ-width check below is reported separately |
| 10 | checkpoint loading reproduces logged evaluations | **PASS with caveat** — rolled returns are 2–8% below logged for 9/10 and 17% below for one (B s2), **all negative**, consistent with a fixed-horizon single-reset rollout versus the trainer's completed-episode average. It rules out a gross loading error; it is not a bit-level match |
| 11 | both A-trained and B-trained critics analysed | **PASS** — all four cells, both laws |

**Reported failure of a committed check.** `docs/prereg_action_padding.md` registers:
*"CONTAMINATED (report separately, claim neither) if: at the final checkpoint, median
sigma over the padded coordinates exceeds 1.5x median sigma over the real 6
coordinates."*

| arm | seed | `sigma_real` | `sigma_pad` | ratio | |
|---|--:|--:|--:|--:|---|
| A | 0–4 | 0.449–0.500 | 0.867–0.872 | 1.74–1.93 | CONTAMINATED |
| B | 0–4 | 0.163–0.248 | 0.818–0.966 | 3.64–5.58 | CONTAMINATED |

**10/10 checkpoints trip the rule.** The frozen entropy bonus inflates policy width in
the coordinates where `Q` is flat, and it does so ~2.5x more in arm B. This is the
contamination the padding plan anticipated. Consequences, stated plainly:

* No return-level conclusion may be drawn from these runs. None is drawn here — the
  arm A ≈ 905 vs arm B ≈ 457–733 gap is reported as a checkpoint property only, and
  the plan already places return-level rehabilitation out of scope.
* The checkpoint reference law inherits the inflated `Sigma_z`. This is why the
  standardized law is mandatory, and it gives the **same** qualitative split, which
  bounds the sensitivity of the primary result to the contamination.
* `L` is measured in the metric of a contaminated `Sigma_z`, so its absolute level
  should not be over-read; its A-vs-B *ordering* survives the law change.

---

## 6. Interpretation, within the committed bounds

**What Probe 4 establishes.** Under the committed primary outcome, evaluated as
committed, on regenerated checkpoints whose training is byte-verified identical to
the original process: the prediction that the weighted-MLE operator is more affected
by padded-coordinate critic error **holds on B-trained critics and is reversed on
A-trained critics**, identically under both reference laws. Result class: **mixed**.

**What it does not establish.** Nothing about returns, and no rehabilitation of the
contaminated padding experiment — both are explicitly out of scope in the plan, and
the σ-width rule fires on every checkpoint. Nothing about full `e = Q_phi - Q^pi` or
about the manuscript's `omega`: this probe identifies only the centred `z`-varying
error field, and says nothing about constant-in-`z` bias. No causal A-vs-B
critic-quality claim, which the plan also puts out of scope: the two arms' critics
were trained under their own state distributions.

**On the asymmetry.** The natural reading of "the operator that trained the critic is
the operator that is most hurt by it" is *not* what the table shows: A-trained critics
are hurt more by PW than by WML, and B-trained critics more by WML than by PW — i.e.
each critic is hurt more by **its own training operator**. That is a coherent pattern
and a hypothesis worth a designed test. It is a post-hoc observation on n=5 and is
labelled as such.

## 7. Reproduction

```bash
cd ~/repos/reppo
# 1. parity: training at the analysis commit == training at 3b96deb
sbatch --account=rwth2182 --array=0-1 slurm/pad16_parity.sh
# 2. regenerate the 10 checkpoints (ledger written before launch)
./.venv/bin/python scripts/analysis/mk_pad16_ledger.py
sbatch --account=rwth2182 --array=0-9 slurm/pad16_regen.sh
# 3. integrity
./.venv/bin/python scripts/probe4_integrity.py
JAX_PLATFORMS=cpu ./.venv/bin/python scripts/verify_action_pad.py 16
sbatch --account=rwth2182 slurm/p4_evalrepro.sh
sbatch --account=rwth2182 slurm/p4_sigma.sh
# 4. Probe 4 itself: seeds 0-4 x {checkpoint law, standardized law}
sbatch --account=rwth2182 --array=0-9 slurm/probe4.sh
# 5. committed statistics and figures
./.venv/bin/python scripts/probe4_report_crossed.py reports/artifacts ckpt
./.venv/bin/python scripts/probe4_report_crossed.py reports/artifacts std
./.venv/bin/python scripts/probe4_figures.py
```

Artifacts: `reports/artifacts/probe4_s{0..4}_{ckpt,std}.npz`,
`probe4_result_{ckpt,std}.json`, `probe4_checkpoints.csv`, `probe4_eta.csv`,
`probe4_sigma_contamination.csv`, `probe4_eval_reproduction.csv`, `fig_probe4.png`.
Ledger: `ledger/runs_pad16_regen.jsonl` (written before launch) and
`ledger/runs.d.pad16/*.json` (written after, with wall clock, GPU, SLURM id and
checkpoint checksum).
