# Appendix robustness: J = 4 independent replication, and J = 8

## The eight short answers

**1. Did the new J = 4 seed reproduce the old result?** **YES.** All 24 cells agree
within ±2.7%, all 24 bootstrap intervals overlap, and the qualitative pattern is
identical.

**2. Old J = 4 `B` vs new J = 4 `B`** (median across the 24 cells, full `d` grid):
**old 0.1717, new 0.1714 — a difference of −0.1%.** Registered band was
`[0.138, 0.206]`; the new value is inside it.

**3. J = 8 `B`** (on its own grid, `d in {16, 64}`): **0.0845**, 95% CI
`[0.0834, 0.0857]`.

**4. Trend**, all on the common grid `d in {16, 64}`:

| | `B` |
|---|---|
| `J = 1` | **0.2995** |
| `J = 4` | **0.1358** |
| `J = 8` | **0.0845** |

It decreases, monotonically. Most of that decrease is the weaker RMS field the
normalisation imposes, not mode cancellation — see Sec. 4.

**5. For J = 8, what mainly hurts pathwise?** **NOISE.** systematic/noise = **0.01**.

**6. For J = 8, what mainly hurts the E-step?** **SYSTEMATIC ERROR.**
systematic/noise = **3.07**.

**7. Does J = 8 support the same story as J = 4?** **YES.** The operator asymmetry is
unchanged: pathwise noise-dominated, E-step systematic-dominated, at every `J` tested.

**8. Is there any sign that the previous result was just lucky?** **NO.** The
replication is near-exact, the three independent `J = 1` estimates agree to 0.4%, and
every registered correctness check passed.

---

**Scope.** Appendix robustness checks only. **The main-paper result is unchanged**
(`reports/planted_error_mechanism.md`, `fig_planted_mechanism.pdf`), and nothing here
revises Claim 4.

**Provenance.** Pre-registration `docs/prereg_planted_j4_replication_j8.md`, committed
`e0fdc06` **before either run**. Prior: `44c86a9` (multimode prereg), `d800a5d`
(multimode results). Scripts `scripts/planted/multimode_sweep.py` (parameterised by
seed / `J` list / `d` list; the default path was verified to reproduce the committed run
**bit-identically**, max abs difference 0.0 over all numeric columns),
`analyse_j4rep_j8.py`, `make_j1_j4_j8_figure.py`. CPU only, float64.

```bash
# Experiment A and B (driver calls multimode_sweep.main with explicit seed/js/ds)
./.venv/bin/python scripts/planted/analyse_j4rep_j8.py
./.venv/bin/python scripts/planted/make_j1_j4_j8_figure.py
```

---

## 1. Correctness checks (prereg Sec. 6) — all pass

| check | replication run (`J`=1,4) | `J = 8` run |
|---|---|---|
| 1 orthonormality `max\|VV^T − I\|` | 1.11e-15 | 1.33e-15 |
| 2 `\|\|e\|\|_inf` numeric/analytic | **1.000000** | **1.000000** |
| 3 `\|\|grad e\|\|_inf` numeric/analytic | **1.000000** | **1.000000** |
| 4 non-finite values | 0 | 0 |
| 5 `eta` bracket hits | 0 | 0 |
| 6 min median ESS (floor 2) | 13.93 (min 10.48) | 13.95 (min 10.51) |
| 7 clean `Q^pi` independent of `J` (CRN checksum) | **0.00e+00** | **0.00e+00** |

Analytic `A_eff = 1.0` in both runs, as the normalisation requires. For `J = 8` the
analytic values are `||e||_inf = 1` and `||grad e||_inf = omega/sqrt(8)`, and the
20-start L-BFGS plus 200,000-point probe reaches both to six decimal places at every
cell. **Nothing blocks interpretation.**

---

## 2. Experiment A — J = 4 independent replication

Seed `20260905` against the committed `20260904`; everything else identical. Directions,
phases and action samples are drawn afresh — nothing is reused.

| `d` | `r_eff` | `B` old | `B` new | % diff | CIs overlap |
|---|---|---|---|---|---|
| 4 | 0.50 | 0.3708 | 0.3698 | −0.3% | yes |
| 4 | 1.00 | 0.2739 | 0.2743 | +0.2% | yes |
| 4 | 3.00 | 0.2739 | 0.2736 | −0.1% | yes |
| 16 | 0.50 | 0.1714 | 0.1706 | −0.4% | yes |
| 16 | 1.50 | 0.1726 | 0.1707 | −1.1% | yes |
| 16 | 3.00 | 0.1710 | 0.1729 | +1.1% | yes |
| 64 | 0.50 | 0.1024 | 0.0996 | −2.7% | yes |
| 64 | 3.00 | 0.0994 | 0.1018 | +2.4% | yes |

(Eight of 24 rows shown; the full table is in
`reports/artifacts/planted_j4_replication.csv` and the analysis log. Across all 24 cells
the difference ranges **−2.7% to +2.4%**, and **24 / 24** bootstrap intervals overlap.)

**Headline:** old **0.1717**, new **0.1714**, difference **−0.1%**, inside the
registered band `[0.138, 0.206]`.

**Decomposition** (medians over cells, full `d` grid):

| run | PW floor | +sys | +noise | **s/n** | E-step floor | +sys | +noise | **s/n** |
|---|---|---|---|---|---|---|---|---|
| old `J=4` | 0.0143 | +0.0013 | +0.1717 | **0.01** | 0.1655 | +0.0289 | +0.0089 | **3.23** |
| new `J=4` | 0.0143 | +0.0009 | +0.1708 | **0.01** | 0.1654 | +0.0288 | +0.0091 | **3.18** |

Registered qualitative test: pathwise `s/n < 1` ✓, E-step `s/n > 1` ✓, magnitude in band
✓. **J = 4 is independently replicated.**

---

## 3. Experiment B — J = 8

`c_j = 1/8`, eight orthonormal directions, `||e||_inf = 1`,
`||grad e||_inf = omega/sqrt(8)`, `omega = r_eff sqrt(8 d)/sigma`. `d in {16, 64}` only,
because eight orthonormal directions do not exist in `d = 4`; no replacement `d = 4` arm
was invented.

`B_J8` pooled over all blocks: **0.0845**, 95% CI `[0.0834, 0.0857]`, median 0.0863.

| by `d` | `B` | 95% CI |
|---|---|---|
| 16 | 0.1028 | [0.1022, 0.1033] |
| 64 | 0.0663 | [0.0658, 0.0668] |

By `r_eff` it is essentially flat — 0.0834 to 0.0857 across the whole grid from 0.5 to
3.0, every interval overlapping. As at `J = 1` and `J = 4`, the systematic displacement
is a property of the error amplitude, not its frequency.

---

## 4. The trend, and how much of it is just a weaker field

All on the common grid `d in {16, 64}`:

| arm | `B` | 95% CI |
|---|---|---|
| `J = 1` (old run) | 0.2999 | [0.2945, 0.3053] |
| `J = 1` (replication run) | 0.2999 | [0.2946, 0.3051] |
| `J = 1` (`J=8` run) | 0.2987 | [0.2936, 0.3038] |
| `J = 4` (old run) | 0.1357 | [0.1334, 0.1379] |
| `J = 4` (replication run) | 0.1360 | [0.1337, 0.1381] |
| **`J = 8`** | **0.0845** | [0.0834, 0.0857] |

The three independent `J = 1` estimates, from three different seeds, agree to within
**0.4%**. That is a further, unplanned replication check and it passes.

```
TREND:  J=1  0.2995   ->   J=4  0.1358   ->   J=8  0.0845
        B_J4/B_J1 = 0.4535    B_J8/B_J1 = 0.2823    B_J8/B_J4 = 0.6224
```

**`B` decreases as `J` increases. This must not be read as mode cancellation.** Under
`c_j = A_0/J` the sup-norm is held fixed but the RMS field strength falls as
`1/sqrt(J)`:

```
E_phi E_a[e^2] = sum_j c_j^2 / 2 = A_0^2/(2J)
RMS_4/RMS_1 = 0.5000    RMS_8/RMS_1 = 0.3536    RMS_8/RMS_4 = 0.7071
```

Calibrating with the **already-committed** amplitude experiment (no new sweep):
`B ~ A^p` with `p = 0.8985` on `d in {16, 64}`.

| | expected from weaker field alone | measured | residual |
|---|---|---|---|
| `J = 4` | 0.5365 | 0.4535 | **0.846** |
| `J = 8` | 0.3929 | 0.2823 | **0.719** |

So the systematic effect becomes smaller as `J` increases, and **most of that reduction
is expected simply because the typical error field is weaker** — a residual of 0.85 at
`J = 4` and 0.72 at `J = 8`, i.e. an extra 1.18x and 1.39x attributable to mode
structure. The mode-structure contribution grows with `J`, as one would expect, but it
remains the smaller of the two effects at both values.

---

## 5. Pathwise vs E-step at J = 8

Medians over cells, restricted to `d in {16, 64}` so all arms are comparable:

| arm | operator | floor | +systematic | +noise | = total | **s/n** | verdict |
|---|---|---|---|---|---|---|---|
| `J=1` | PW | 0.0328 | +0.0177 | +0.1295 | 0.1800 | 0.14 | NOISE |
| | ZO | 0.2148 | +0.0071 | +0.0373 | 0.2593 | 0.19 | NOISE |
| | **E-step** | 0.2438 | **+0.0426** | +0.0062 | 0.2926 | **6.89** | **SYSTEMATIC** |
| `J=4` old | PW | 0.0328 | +0.0024 | +0.1938 | 0.2291 | 0.01 | NOISE |
| | ZO | 0.2148 | +0.0002 | +0.0135 | 0.2285 | 0.02 | NOISE |
| | **E-step** | 0.2438 | **+0.0170** | +0.0050 | 0.2658 | **3.41** | **SYSTEMATIC** |
| `J=4` rep | PW | 0.0329 | +0.0025 | +0.1932 | 0.2286 | 0.01 | NOISE |
| | ZO | 0.2150 | +0.0000 | +0.0135 | 0.2285 | 0.00 | NOISE |
| | **E-step** | 0.2439 | **+0.0169** | +0.0049 | 0.2657 | **3.48** | **SYSTEMATIC** |
| **`J=8`** | PW | 0.0328 | +0.0026 | +0.2029 | 0.2384 | **0.01** | **NOISE** |
| | ZO | 0.2149 | +0.0001 | +0.0072 | 0.2222 | 0.01 | NOISE |
| | **E-step** | 0.2443 | **+0.0097** | +0.0031 | 0.2571 | **3.07** | **SYSTEMATIC** |

**The distinction survives.** Pathwise is noise-dominated at every `J` (0.14 → 0.01 →
0.01, i.e. increasingly so). The E-step is systematic-dominated at every `J`
(6.89 → 3.41 → 3.07). The ratio between the two operators spans more than two orders of
magnitude at `J = 8`, as it did at `J = 1` and `J = 4`.

---

## 6. Update quality

`G = Err[E-step] − Err[PW]` against `r_eff`; `J = 4` values restricted to
`d in {16, 64}`. **No crossover is fitted.**

| `r_eff` | `G` (`J=4` old) | `G` (`J=4` rep) | `G` (`J=8`) | better at `J=8` |
|---|---|---|---|---|
| 0.50 | 0.1557 | 0.1563 | 0.1436 | pathwise |
| 0.75 | 0.1109 | 0.1111 | 0.0958 | pathwise |
| 1.00 | 0.0753 | 0.0757 | 0.0590 | pathwise |
| 1.25 | 0.0465 | 0.0473 | 0.0293 | pathwise |
| 1.50 | 0.0250 | 0.0256 | 0.0072 | pathwise |
| 1.75 | 0.0074 | 0.0073 | **−0.0105** | **E-step** |
| 2.00 | −0.0069 | −0.0066 | **−0.0239** | **E-step** |
| 3.00 | −0.0444 | −0.0444 | **−0.0617** | **E-step** |

Pathwise is better at low `r_eff`, the E-step at high `r_eff`, at both `J`. The gap
shrinks from `J = 4` to `J = 8` at every `r_eff`, and the sign change moves slightly
earlier (between 1.5 and 1.75 at `J = 8`; between 1.75 and 2.0 at `J = 4`). The `J = 4`
old and replication columns agree to within 0.001 everywhere — the replication again.

---

## 7. What the systematic effect does

`cos(delta_err, v_signal)` per block, on `d in {16, 64}`:

| arm | median | mean | q05 | q95 | fraction < −0.9 |
|---|---|---|---|---|---|
| `J = 4` old | −0.8873 | −0.8122 | −0.9893 | −0.5667 | 0.500 |
| `J = 4` rep | −0.8794 | −0.8119 | −0.9887 | −0.5589 | 0.500 |
| **`J = 8`** | **−0.8140** | −0.7427 | −0.9813 | −0.4175 | 0.500 |

**Critic error still mainly SHRINKS the useful E-step signal** at `J = 8`. The
alignment weakens somewhat — median −0.89 → −0.81 — so the transverse, genuinely
misdirecting component grows a little with `J`, but the dominant effect remains a
reduction of the correct step rather than a rotation of it.

(These medians differ from the −0.96 quoted in `reports/planted_multimode.md` because
that figure pooled all three dimensions including `d = 4`, where the alignment is
near-perfect; here everything is restricted to `d in {16, 64}` so the arms are
comparable to `J = 8`.)

---

## 8. Centred ZO/PW theory check — appendix only

`R_var = Var_e[ZO]/Var_e[PW]` fitted against **both** conventions, as registered:

| run | `J` | x-axis | `beta` | `R^2` | crossing |
|---|---|---|---|---|---|
| old | 1 | `r_eff` = `r_nom` | −2.0268 | 0.9990 | 0.9931 |
| rep | 1 | `r_eff` = `r_nom` | −2.0176 | 0.9997 | 0.9896 |
| j8 | 1 | `r_eff` = `r_nom` | −1.9978 | 1.0000 | 0.9818 |
| old | 4 | `r_eff` | −2.0000 | 1.0000 | 0.4924 |
| old | 4 | **`r_nom`** | −2.0000 | 1.0000 | **0.9848** |
| rep | 4 | `r_eff` | −2.0022 | 1.0000 | 0.4931 |
| rep | 4 | **`r_nom`** | −2.0022 | 1.0000 | **0.9862** |
| j8 | 8 | `r_eff` | −1.9981 | 1.0000 | **0.3477** |
| j8 | 8 | **`r_nom`** | −1.9981 | 1.0000 | **0.9835** |

`r_nom` gives the cleaner collapse, and `J = 8` confirms the `1/sqrt(J)` displacement at
a second value: predicted `1/sqrt(8) = 0.3536`, measured **0.3477**. Under `r_nom` the
crossing is 0.9848 / 0.9862 / 0.9835 across the three multi-mode arms.

The observed scaling is **consistent with the characteristic ratio of the two bounds
Claim 4 supplies**. Claim 4 supplies bounds, so this does not establish that the measured
ratio must equal `r^-2`, and **Claim 4 is not revised on this evidence** — it is one
appendix observation on a three-point `J` ladder.

---

## 9. Figure

`reports/artifacts/fig_planted_j1_j4_j8.{pdf,png}`, source data
`fig_planted_j1_j4_j8_data.csv`. All panels restricted to `d in {16, 64}`.

- **Panel A** — `B` for `J = 1`, `J = 4` (both seeds) and `J = 8`. The `J = 4`
  replication line lies on top of the original, which is the visual form of answer 1.
- **Panel B** — total update error for pathwise and the E-step at `J = 4` and `J = 8`.
  Pathwise rises steeply with `r_eff` while the E-step is flat; they cross.
- **Panel C** — systematic vs noise contribution per operator at `J = 1, 4, 8`. The
  orange/blue reversal between PW and E-step is the whole story in one picture.

**Appendix only.** This does not replace `fig_planted_mechanism.pdf`.

---

## 10. Limitations

1. `J = 8` uses `d in {16, 64}` only; there is no `d = 4` arm and none was invented.
2. All modes share one frequency. A frequency spread would move `L_eff` inseparably from
   the mode count and remains untested.
3. Matching `||e||_inf` un-matches RMS strength; Sec. 4 separates the two with an
   amplitude calibration that assumes `B ~ A^p` locally, with `p` fitted over
   `A in {0.25, 1, 4}`.
4. `sigma = 0.4` and `A_0 = 1` only.
5. The `J` ladder has three points. The trend is descriptive, not a fitted law.

---

## 11. Stop

Both registered experiments are complete, every correctness check passed, and no
implementation defect surfaced. **No further planted-error experiment is run or
proposed** — no `J = 16`, no frequency spread, no additional amplitudes, dimensions or
`sigma` levels. The stop rule in `docs/prereg_planted_j4_replication_j8.md` Sec. 8
applies.
