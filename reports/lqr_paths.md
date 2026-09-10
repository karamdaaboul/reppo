# LQR optimisation paths at `d = 2`: pathwise against zeroth order

**All four panels confirm the registered prediction. The zeroth-order path jitters more
when the critic error wiggles slowly, the pathwise path jitters more when it wiggles
fast, and both follow the same average route in every panel.**

**Read that verdict with this beside it: three registered gate comparators failed, and
their tolerances were replaced by amendment A2 before any path ran.** The three are
`G0a` for the E-step arm, `G0b`, and `G0c` at one panel. In each case the measurement
shows the comparison was limited by floating-point cancellation rather than by the code
under test, and the evidence for that is in the Gates section below and in
`docs/prereg_lqr_paths.md` Sec. 13. The two gates that test the estimators and the
reference path, `G1` and `G2`, pass at every panel as registered and were not amended.
Every registered verdict is still printed and stored in `results/lqr_paths/gates*.json`.
A reader who rejects A2 should treat the four panel verdicts as resting on `G1`, `G2` and
`G3` alone.

Registered in `docs/prereg_lqr_paths.md`. Symbols, defined once. `d` is the action
dimension, fixed at 2. `M` is the number of action samples per estimator call, fixed at
32. `sigma` is the policy standard deviation. `omega` is the spatial frequency of the
planted critic error and `eps` is its amplitude. `PW` is the pathwise estimator, `ZO` the
centred zeroth-order estimator with the `M/(M-1)` de-attenuation, `ESTEP` the shipped
weighted maximum likelihood displacement. `g*` is the exact gradient of the
Gaussian-blurred critic. `J` is the mean distance of a path from the noise-free reference
path, in units of `sigma`. `rho` is the ratio of the median `J` over 100 path seeds,
`ZO` over `PW`. `r_tot` is the measured one-step total-error crossover, in units of
`sigma*omega`.

---

## Panels

**study_slow: CONFIRMED, `rho = 2.225` [2.096, 2.374], predicted above 1.**
The critic error is nearly invisible at this frequency, so the ordering is set by the
smooth part of the critic, where the zeroth-order operator pays the classical dimension
factor. Both arms track the same route, with median cosine to `g*` of 0.959 and 0.925.
The zeroth-order path ends 0.85 `sigma` from the reference endpoint against pathwise's
0.26.

**study_fast: CONFIRMED, `rho = 0.204` [0.192, 0.216], predicted below 1.**
At `sigma*omega = 159.9` the blur factor `exp(-c^2/2)` underflows to zero, so the error
contributes nothing to the target and everything to the pathwise samples. Pathwise
stalls near the start, with median cosine 0.584 and an endpoint 6.83 `sigma` from the
reference, while zeroth order ends 1.06 `sigma` away. This is the panel that carries the
figure's second point.

**big_slow: CONFIRMED, `rho = 1.636` [1.573, 1.784], predicted above 1.**
With the error thirty times larger the blurred landscape is itself wrinkled, and both
arms follow those wrinkles, so both jitters are small in absolute terms, 0.135 and 0.222.
The cosines fall to 0.446 and 0.581 because the estimand now turns along the path. This
panel needed the registered single doubling of the path length to `N = 160` for its
reference path to settle.

**big_fast: CONFIRMED, `rho = 0.596` [0.560, 0.614], predicted below 1.**
Neither arm reaches the target: the endpoints sit 7.36 and 5.10 `sigma` from the
reference. Pathwise is the worse of the two, as predicted, but the honest reading of this
panel is a regime where both operators fail rather than one winning. The error amplitude
here is 150 per cent of the within-state value spread, which is an illustration size and
not a measured one.

---

## Design as executed

State index 0 of the 32 the study drew at `d = 2`, accepted at the first index.
`||a* - mu_0|| = 1.3945` where `a*` is the exact maximiser of `Q^pi(s, .)` and `mu_0` is
the behaviour policy mean. `sigma = 0.13945`, which is above the effective `min_std` of
0.1, so the panels sit inside the band a real policy can reach. Error field: the study's
full-rank arm, rank 2, coordinate basis, phases from the study's own stream.

| panel | `eps` | `sigma` | `omega` | `sigma*omega` | `r_tot` | `N` | `rho` (ZO/PW) | verdict | `rho` (ESTEP/PW) |
|---|---|---|---|---|---|---|---|---|---|
| study_slow | 0.0120419 | 0.139454 | 2.5758 | 0.3592 | 39.985 | 80 | **2.225** [2.096, 2.374] | confirmed | 0.828 [0.761, 0.910] |
| study_fast | 0.0120419 | 0.139454 | 1146.909 | 159.941 | 39.985 | 80 | **0.204** [0.192, 0.216] | confirmed | 0.096 [0.087, 0.105] |
| big_slow | 0.361256 | 0.139454 | 2.5758 | 0.3592 | 1.9345 | 160 | **1.636** [1.573, 1.784] | confirmed | 2.201 [2.076, 2.317] |
| big_fast | 0.361256 | 0.139454 | 55.4886 | 7.7381 | 1.9345 | 80 | **0.596** [0.560, 0.614] | confirmed | 0.679 [0.653, 0.719] |

The slow panels sit at one quarter of the error-only tie `sqrt(d M/(M-1)) = 1.4368`. The
fast panels sit at four times their own `r_tot`. All intervals are paired percentile
bootstraps over the 100 path seeds, 10 000 resamples, `numpy.random.default_rng(20260902)`.

**Secondary measures.** Median per-step cosine to `g*`, and the median endpoint distance
from the reference path in units of `sigma`.

| panel | cos PW | cos ZO | cos ESTEP | end PW | end ZO | end ESTEP |
|---|---|---|---|---|---|---|
| study_slow | 0.959 | 0.925 | 0.951 | 0.262 | 0.854 | 0.219 |
| study_fast | 0.584 | 0.898 | 0.931 | 6.831 | 1.058 | 0.359 |
| big_slow | 0.446 | 0.581 | 0.193 | 0.176 | 0.224 | 0.252 |
| big_fast | 0.535 | 0.746 | 0.688 | 7.364 | 5.100 | 5.771 |

The `ESTEP` arm is reported here and is not drawn in the figures. Its magnitude is set by
the dual variable `eta` and not by the gradient, so only its direction is comparable, and
the crossover study already records that it is not the estimator Claim 4 is about. It
tracks `ZO` at study_fast and is the worst of the three at big_slow.

---

## Calibration

The one-step total-error crossover, from `10^4` shared replicates at `mu_0` on a 41 point
grid, with a percentile bootstrap over replicates.

| `eps` | value | `r_tot` | 95 % interval | `r_tot / 1.4368` |
|---|---|---|---|---|
| `eps_study` | 0.0120419 | 39.985 | [39.216, 40.564] | 27.8 |
| 10 x | 0.120419 | 4.1721 | [4.1117, 4.2306] | 2.90 |
| 30 x | 0.361256 | 1.9345 | [1.9133, 1.9570] | 1.35 |
| 100 x | 1.20419 | 1.5509 | [1.5335, 1.5695] | 1.08 |

`r_tot` falls as `1/eps` over the first three rows to within 4 per cent, which is the
scaling the crossover study measured (`reports/lqr_crossover_corrected.md` Sec. 4). The
registered rule then selects `eps_big = 30 x eps_study`, the smallest candidate whose
crossover has moved inside 1.5 times the error-only tie. At this state the crossover sits
27.8 times the tie, against the 34.3 the study reports at `d = 2`; the two differ because
this is one state at one width, and because this measure is an angular error rather than
a mean squared error.

**One assumption that does not hold at the slow panels.** The root-mean-square frequency
`omega_RMS` equals the nominal `omega` to four digits at both fast panels and is 2.28
times `omega` at the slow ones. The two agree only once `sigma*omega` is large. The dial
is the nominal `omega` throughout, as registered.

---

## Gates

`G1` and `G2` are the gates that test the estimators and the reference path. Both pass at
every panel, as registered. `G1`: the means of `PW` and de-attenuated `ZO` over `10^4`
replicates match `g*` at `mu_0` with a maximum absolute z of 1.73 against a bound of 4.
`G2`: every reference path settles inside a ball of radius `sigma` over its last 20
steps.

`G3` is the provenance gate. `G3b` reran the study's published full-rank `d = 2` arm
through `sweep.run_d` and returned `c* = 1.5218` against the published 1.522, a relative
deviation of `1.1e-4`. This environment reproduces the study.

**Three registered comparators failed and were amended, with the failures kept in the
record** (`docs/prereg_lqr_paths.md` Sec. 13, A2, and `results/lqr_paths/gates*.json`,
which store both verdicts). In each case the measurement shows the comparison is limited
by floating-point cancellation rather than by the code under test.

- `G0a`. `PW` and `ZO` reproduce the harness composition to `2.4e-15` and `2.0e-15`. The
  `ESTEP` arm reproduces it to `1.17e-12` against a registered bound of `1e-14`. Measured
  amplification of a relative perturbation of the critic values into the returned
  direction: `PW` exactly 0, because the pathwise operator never reads a critic value;
  `ZO` 1.41; `ESTEP` 48.2 at `eta = 0.196`. The bound was below the conditioning of the
  operator. Amended to `1e-11` for `ESTEP` alone.
- `G0b`. `lqr.q_pi` carries the additive constant `gamma v/(1 - gamma)`, 2096 out of a
  typical value of 3573, so differencing two of its values loses eight digits. The
  absolute deviation is flat at the machine floor while the differenced quantity varies
  over four orders of magnitude, and the relative deviation falls as `1/|delta|`. An
  algebra error would do the opposite. Amended to the cancellation floor of the
  comparison, which the measured `4.2e-14` sits well inside.
- `G0c`. The blurred error gradient carries `exp(-c^2/2)`, which is `1e-13` at big_fast
  and underflows to exactly zero at study_fast. A Gauss-Hermite comparator cannot resolve
  that: at study_fast it returns 8.302 for a quantity whose true value is zero. Amended
  to require agreement to the comparator's own demonstrated resolution, and to retire the
  comparator where the blur underflows. The crossover study met the same wall and
  switched to an absolute bound at `c = 6`.

---

## Figures

`reports/figures/lqr_paths_{study,big}_{slow,fast}.pdf`, seed 0 only, `PW` in navy
`#1F4E79` and `ZO` in brick `#C0504D`, a black diamond at the start, a thin circle of
radius `sigma` at the start, 15 gray contour levels behind.

The registered background rule asks for at least 8 grid points per error period and falls
back to contouring `Q^pi` when that would need more than 600 points per axis. It fired at
exactly one panel. study_fast needs 2766 points per axis and its background is therefore
`Q^pi`, which is also what the estimand is there, since the blurred error gradient
underflows to zero. The other three panels contour the full critic `Q_phi`: 7 points
needed at study_slow, 8 at big_slow, 152 at big_fast.

---

## Limitations

1. `d = 2` only. Nothing here bears on the `sqrt(d)` scaling of the crossover study, and
   a single dimension cannot carry an exponent.
2. One state, one phase draw, one system. These are panels at a point, not a population.
3. `sigma` is fixed and there is no covariance update, so this is the mean-only dynamics
   of Proposition 2 and not the algorithm.
4. `eps_big` is an illustration size, 150 per cent of the within-state value spread, and
   not the study's size of 5 per cent.
5. `Q^pi` is quadratic and globally smooth and the error is planted. A neural critic is
   neither, and nothing here concerns learned critics.
6. `J` is a divergence from each panel's own reference path, and those differ between
   panels, so only the within-panel ratio `rho` is interpreted.

---

## Provenance

Preregistration `4606164`. Calibration outcome (A1), gate amendment (A2) and the code
`a874a8b`, committed before any path run. This report, the four figures and the
`results/lqr_paths/` records are uncommitted at the time of writing, because CLAUDE.md
asks for explicit approval before figures, analysis scripts and reports are committed.
Amendment A3 corrects one statement of fact in Section 9 of the preregistration: the
figures are not git-ignored, since `.gitignore:226` re-includes `reports/figures/`.

Run on the FZI workstation, host `karam`, `Linux-6.8.0-138-generic-x86_64`, Python
3.12.3, JAX 0.5.2, NumPy 2.5.1, SciPy 1.18.0, matplotlib 3.11.1, device `TFRT_CPU_0`,
float64 enabled. The harness pins the process to CPU at import, so the workstation's two
GPUs are recorded and not used. Wall time: gates 82.5 s, calibrate 10.6 s, panel gates
20.0 s, paths 6.7 s, report 1.8 s, figures 1.9 s.

`results/lqr_paths/` holds `calibration.json`, `calibration.csv`, `gates.json`,
`gates_panels.json`, `summary.json`, `figures.json`, `env.json`, and one `.npz` per panel
with the per-seed paths. The `.npz` files are git-ignored repository-wide;
`npz_manifest.csv` carries their sha256, following the convention of
`reports/artifacts/lqr_npz_manifest.csv`. The script's own sha256 is recorded in
`env.json` for every stage, because the gates and the calibration ran before the script
was committed.
