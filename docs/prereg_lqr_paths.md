# Pre-registration: LQR optimisation paths, pathwise against zeroth order at `d = 2`

**Status.** Committed before any calibration run and before any path run. Nothing below
was chosen after seeing a result of this experiment. Sections 1 to 12 are append-only.
The calibration outcome goes below the amendment line in Section 13, and is committed
before any path run.

**Revision history.** v1.0 (2026-09-10): as committed.

**Companions.** `docs/prereg_lqr_crossover.md` (`d2cb9f6`) registers the harness this
experiment imports. `reports/lqr_crossover_corrected.md` (`00091c4`) is the canonical
report of that study and is the source of every published number quoted here.
`reports/lqr_final_closeout.md` closes it. Style model for a small preregistered run:
`reports/lqr_rank_ceilsqrt_runnote.md` (`5cb384d`).

---

## 1. What this is, and what it is not

The figure is an intuition figure in the style of Carvalho et al. (2021,
arXiv:2107.09359, Fig. 1). It draws the path of a Gaussian policy's mean over the
critic landscape, one path per gradient estimator, in a linear quadratic regulator
(LQR) where every quantity is closed form.

Symbols, defined once. `d` is the action dimension, fixed at 2. `n` is the state
dimension, fixed at `2d = 4`. `s` is one state. `a` is an action. `mu` is the policy
mean, a point in action space. `sigma` is the policy standard deviation, one scalar for
both coordinates. `M` is the number of action samples per estimator call, fixed at 32.
`Q^pi` is the exact action-value function of the behaviour policy. `e` is the planted
critic error. `Q_phi = Q^pi + e` is the critic the estimators see. `g*` is the exact
Gaussian-blurred gradient of `Q_phi` with respect to the mean. `eps` is the amplitude
of `e`. `omega` is the spatial frequency of `e`. `PW` is the pathwise estimator. `ZO`
is the centred zeroth-order estimator with the `M/(M-1)` de-attenuation of the study.
`ESTEP` is the shipped weighted maximum likelihood displacement of
`actor_update_mode="weighted_mle"`. `N` is the number of update steps. `R` is the
number of path seeds.

**What the figure is meant to show.** First, that both estimators follow the same
average route, and that route is the gradient of the blurred critic. Second, that which
path jitters more depends on how fast the critic error wiggles relative to `sigma`.

**What this is not.** It is not new evidence for the `sqrt(d)` scaling of the crossover
study, because `d` is fixed at 2 and a single dimension cannot carry an exponent. It is
not evidence about learned critics, because the error field is planted. It is not
evidence about return, because no environment is stepped and no policy is trained.

**Why the regimes are placed against the total-error crossover and not against the
error-only tie.** The crossover study measured two different crossings. The error-only
tie is where the two estimators' critic-error-channel variances are equal. It sits at
`sigma*omega = sqrt(d M/(M-1))`, which is 1.4368 at `d = 2` and `M = 32`
(`reports/lqr_crossover_corrected.md` Sec. 3 and Sec. 8). The total-error crossover is
where the two estimators' full errors are equal. It sits 28.7 to 36.1 times further out
in `sigma*omega`, is 34.3 times further out at `d = 2`, and moves as `1/eps`
(same report, Sec. 4). The reason is that at the error-only tie the critic error carries
at most about 5 per cent of `PW`'s total estimator error and about 0.1 per cent of
`ZO`'s (same report, Sec. 5.4). Path jitter is driven by total error, not by the error
channel alone. Placing a panel at the error-only tie and calling it a crossing would
therefore predict an ordering the data does not have. Every regime below is placed
relative to a total-error crossover measured here.

---

## 2. What is imported, and from where

Nothing in the list below is reimplemented. Every entry is imported from the file named,
at the commit named. Commits are the last commit that touched the file.

| object | file | commit |
|---|---|---|
| package import hook, `SEED_ROOT = 20260902`, float64 and CPU asserts | `scripts/lqr_crossover/__init__.py` | `33c7632` |
| `build_system`, `sample_states`, `q_coeffs`, `q_of_u_factory`, `grad_a_q_pi`, `q_pi`, `q_spread_closed_form` | `scripts/lqr_crossover/lqr.py` | `33c7632` |
| `draw_error`, `PlantedError`, `e_value`, `e_grad`, `blurred_e_grad`, `theta`, `omega_inf` | `scripts/lqr_crossover/error_field.py` | `33c7632` |
| `both_from_shared_u` (`PW` and `ZO` on shared draws) | `scripts/lqr_crossover/estimators.py` | `33c7632` |
| `whitened_pathwise`, `centred_zo`, `pathwise_mean`, `deattenuation_factor`, `softmax_displacement` | `src/jaxrl/estimators.py` | `33c7632` |
| `solve_eta`, `ETA_GRID`, `EPS_E = 0.5` (the E-step dual, as the study solves it) | `scripts/lqr_crossover/estep_arm.py` | `33c7632` |
| `make_kernel`, `run_d`, `SIGMAS`, `OMEGAS` (error-only paired-difference sweep) | `scripts/lqr_crossover/sweep.py` | `98da377` |
| `crossover_by_c`, `solve_crossover`, `err_only`, `c_grid` (crossover extraction) | `scripts/lqr_crossover/analyze.py` | `98da377` |
| `c_star_asymptote`, `vsin`, `vcos` (closed forms) | `scripts/lqr_crossover/reference.py` | `33c7632` |

Two consequences of that import list are registered here so they cannot be treated as
choices made later.

1. `scripts/lqr_crossover/__init__.py` sets `JAX_PLATFORMS=cpu` and asserts at import
   that the first device is a CPU and that `jax_enable_x64` took effect. Any script that
   imports the harness therefore runs on CPU on every machine, including a machine with
   a GPU. This experiment does not modify that file. The device list actually used is
   recorded in `results/lqr_paths/env.json`.
2. The E-step arm is composed exactly as `scripts/lqr_crossover/estep_arm.py` composes
   it: solve the dual for `eta` on the study's grid, form
   `w = softmax(Q/eta)` over the sample axis, then take
   `softmax_displacement(w, u, axis=0)`. The unused wrapper
   `scripts/lqr_crossover/estimators.py:estep_displacement` is not called. All three
   arms use a non negative sample axis, which is the axis convention of the production
   call site `src/jaxrl/reppo.py:1247` and of every E-step call site in the study.

---

## 3. System, state, start point and width

**System.** `lqr.build_system(d=2, seed=SEED_ROOT + 2)` with the study's shipped
defaults: `gamma = 0.99`, `n = 2d = 4`, `W = I`, identity cost, `k_perturb = 1.15`,
`normalize="unit_H"`, `sigma_ref = 0.1`. This is the same system object the crossover
study used at `d = 2`. Its guard diagnostics are recorded with the results:
`rho_closed = 0.5347`, `cond_H = 16.19` as published.

**State.** One random number stream, `np.random.default_rng(SEED_ROOT + 2000 + d)`, is
consumed in the study's order: first `lqr.sample_states(sys, rng, 32)`, then
`error_field.draw_error(rng, 32, d, kind="full", omega=1.0)`. `n_states = 32` is the
count the study's full-rank arm used, so the state and the phases below are that arm's
first state and its phases and not a fresh draw. The state used is `states[i]` for the
first index `i` in `0, 1, 2, ...` that satisfies the width condition of the next
paragraph.

**Start point.** `mu_0 = -K s` in the study's normalised coordinates, that is, the third
return value of `lqr.q_coeffs(sys, states)` at the chosen index. This is the mean of the
behaviour policy at `s`.

**Target.** `Q^pi(s, .)` is concave quadratic, so the maximiser is closed form. With
`Hn` and `g_mu` the first two return values of `lqr.q_coeffs`,
`a* = mu_0 + 0.5 * Hn^{-1} g_mu`. `Hn` is positive definite by construction, so `a*` is
the maximum and not a saddle.

**Width.** `sigma = ||a* - mu_0|| / 10`, fixed for the whole experiment. There is no
covariance update anywhere in this design. If `||a* - mu_0|| < 1e-3` the state is
rejected and the next index in the declared list is taken.

**Reachability note, registered in advance.** The real policy has an effective minimum
standard deviation of 0.1 (`min_std = 0.1`, `src/networks/jax_models.py:336`). The
`sigma` produced by the rule above is whatever the system gives. Its value and whether
it lies above or below 0.1 are recorded in the amendment and repeated in the report's
limitations. The rule is not adjusted to land in the reachable band.

---

## 4. Critic error

**Field.** The study's full-rank arm at `d = 2`, that is `kind="full"` with rank
`k = d = 2`:

    e(a) = (eps / sqrt(2)) * [ sin(omega v_1^T a + phi_1) + sin(omega v_2^T a + phi_2) ]

`v_1` and `v_2` are the orthonormal rows the harness supplies for that arm, which are
the coordinate basis, and `phi_1`, `phi_2` are the phases drawn by the stream of
Section 3. The choice of `kind="full"` over `kind="rank_r"` is registered here: it is
the study's own full-rank rung, and at `d = 2` it fixes `V = I`, so the drawn error
pattern is axis aligned in the figure. The rank-2 rung of the study uses a random
orthonormal basis instead, which is a rotation of action space; the two differ only by
that rotation, and the figure inherits whichever is chosen. The same `v_j` and `phi_j`
are used in every panel. Only `eps` and `omega` change between panels.

**Amplitude.** `eps_study = 0.05 * lqr.q_spread_closed_form(sys, s, sigma)`, the study's
definition: 5 per cent of the closed-form within-state spread of `Q^pi` under the
behaviour policy at that state and that width. It is computed once, at `s` and at the
`sigma` of Section 3, and held fixed along every path. Holding it fixed matters: if
`eps` were recomputed as `mu` moves, the error amplitude would drift along the path and
the panel would no longer have one error size.

**Second amplitude.** `eps_big` is the smallest multiple of `eps_study` in
`{10, 30, 100}` whose calibrated total-error crossover `r_tot` of Section 6 satisfies
`r_tot <= 1.5 * 1.4368`, that is, whose total-error crossover has moved in to within one
and a half times the error-only tie. If no multiple qualifies, `eps_big = 100 *
eps_study` and the report says so.

**Frequency.** `omega` is the dial. Every regime below is stated as a value of the
product `sigma*omega`, and `omega` is set to that product divided by the `sigma` of
Section 3. The realised root-mean-square frequency `omega_RMS` of the study's Sec. 8 is
not identical to `omega`; it approaches `omega` as `sigma*omega` grows and was measured
at 1.0034 times `omega` at the `d = 2` crossover cell of the rank-one arm and 0.9712
times `omega` at the `d = 2` cell of the rank-2 arm. `omega_RMS` is computed from the
study's closed form and reported for every panel, so the gap is visible rather than
assumed away.

---

## 5. Update rule and arms

**Update, identical for every arm.**

    mu_{t+1} = mu_t + 0.2 * sigma * g_hat_t / ||g_hat_t||,   t = 0 .. N-1,   N = 80

Every arm takes the same step length. Only the direction comes from the estimator. With
a covariance `Sigma = sigma^2 I` this is the normalised natural-gradient step of
Proposition 2 and Corollary 3 of the paper. The path length is `16 sigma` and the
distance from `mu_0` to `a*` is `10 sigma` by construction, so a path that goes straight
to the target arrives at step 50 and has 30 steps left to sit there.

**Arms.**

1. `PW`: `both_from_shared_u(..., axis=0)["g_pw"]`.
2. `ZO`: `both_from_shared_u(..., axis=0)["g_zo"]`, which is the centred estimator with
   the `M/(M-1)` de-attenuation applied once.
3. `ESTEP`: the composition of Section 2, item 2, with the study's `eps_e = 0.5`.
4. Reference: the same update driven by the exact `g*`, that is
   `grad_a Q^pi(s, mu_t)` plus `error_field.blurred_e_grad(pe, eps, mu_t, sigma)`. The
   first term is exact because a Gaussian blur shifts a quadratic by a constant. The
   second is the harness's closed form. The reference path is noise free and there is one
   of it per panel.

`M = 32` in every arm, the study's primary value.

**Common random numbers.** At step `t` of seed `r`, all three estimator arms use the
same standard normal draws `u_{t,1..M}`, obtained from
`jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(SEED_ROOT + 8000 + d), r), t)`.
The arms do not share `mu_t`, because each arm walks its own path, so the critic closure
differs between arms at the same step. Sharing `u` is what makes the arm comparison
paired.

---

## 6. Regimes, and the calibration rule that places them

The calibration below is run before any path run and its outcome is appended in
Section 13.

**One-step angular error.** At the fixed `(s, mu_0, sigma)`, for each `eps` in
`{eps_study, eps_big}` and each grid value of `sigma*omega`, draw `10^4` replicates of
`u` of shape `(M, d)` and compute the mean over replicates of `1 - cos(g_hat, g*)` for
`PW` and for `ZO`. `g*` is recomputed at every grid value, because the blurred error
gradient depends on `omega`. One single block of `10^4` replicate draws is used at every
grid point and for both `eps` and both arms, so replicate `r` is the same random draw
everywhere and the whole calibration is paired. The block comes from
`jax.random.PRNGKey(SEED_ROOT + 7000 + d)`.

**Grid.** `sigma*omega` on `numpy.logspace(-1, 3, 41)`.

**Definition of the crossover.** Let `D(x)` be `ZO`'s mean angular error minus `PW`'s at
`sigma*omega = x`. `r_tot(eps)` is the first grid interval in which `D` changes sign from
positive to negative, located inside that interval by linear interpolation of `D` against
`log(sigma*omega)`. If `D` does not change sign anywhere on the grid for `eps_study`,
that is reported and only the `eps_big` panels are run. A percentile bootstrap interval
for `r_tot` is computed by resampling the `10^4` replicate indices with replacement,
1000 resamples, `numpy.random.default_rng(20260902)`, recomputing the entire curve in
each resample. The inferential unit of this bootstrap is the replicate. That is a
statement about Monte Carlo precision at one state, not an inferential claim about
seeds, and the report says so.

**Panels.** Four panels, `{eps_study, eps_big} x {slow, fast}`:

- slow: `sigma*omega = 1.4368 / 4 = 0.3592`, the same value for both `eps`.
- fast: `sigma*omega = 4 * r_tot(eps)`, one value per `eps`.

**Reported alongside.** The error-only tie at `mu_0` from the harness's paired-difference
method, as a check against 1.4368. See gate G3.

---

## 7. Gates

All gates run before any path run. Every gate prints what it measured. A blocking gate
that fails stops the experiment and is reported as a failure.

**Staging.** G0a, G0b, G0c, G3, and G1 at `eps = 0` do not depend on the calibration and
run first. G1 at the four panel frequencies and G2 do depend on it. Until the calibration
exists they print DEFER, never PASS, following the repository's convention that a check
which cannot run must not look green. They are then run, and must pass, before any path
run. The path stage refuses to start unless the gate record shows every gate passing for
the panels it is about to run.

**G0a, arm reproduction.** On 100 fixed draws of `u` of shape `(M, d)` from
`jax.random.PRNGKey(SEED_ROOT + 9000 + d)`, in float64, the per-step function used by
the path code, evaluated at `mu = mu_0`, must reproduce the harness composition
`lqr.q_of_u_factory` plus `error_field.e_value` fed to `both_from_shared_u`, and the
study's E-step composition, to a maximum relative deviation of `1e-14`. The count of
bitwise identical outputs is reported. Blocking.

**G0b, the moving-mean closure.** The path code evaluates the critic at a mean that
moves, which the study's `q_of_u_factory` does not do, since it fixes the mean at
`-K s`. The path code uses the identity in the header of `scripts/lqr_crossover/lqr.py`:
for any mean `m`, `Q^pi(s, m + sigma u) = const(s, sigma, m) - sigma^2 u^T H u + sigma
u^T grad_a Q^pi(s, m)`, with `grad_a Q^pi` taken from `lqr.grad_a_q_pi`. The gate checks
that identity against the harness's independent exact form `lqr.q_pi`, which the study
uses only in its gates. For 100 random pairs `(u, u')` and 10 random means `m` along the
straight line from `mu_0` to `a*`, the difference `q(u) - q(u')` produced by the path
closure must equal the same difference computed from `lqr.q_pi` plus
`error_field.e_value`, to a relative deviation of `1e-12`. Blocking.

**G0c, the exact target.** `g*(m)` used by the reference path must match a central
finite difference of `Q_phi` blurred by quadrature at 10 random means, to a relative
deviation of `1e-7`. The smooth part is checked by central differences of `lqr.q_pi`
directly. Blocking.

**G1, unbiasedness at the start.** At `mu_0`, over `10^4` replicates, the mean of `PW`
and the mean of de-attenuated `ZO` must each match `g*` within four standard errors per
coordinate. Checked at the two panel frequencies of each `eps` and at `eps = 0`.
Blocking.

**G2, the reference path has settled.** In every panel, the last 20 points of the
reference path must lie inside a ball of radius `sigma` centred at their mean. If not,
`N` is doubled once, to 160, and the check is repeated. If it still fails, no path run
happens for that panel, and the failure is reported with the diagnostic quantities
`||g*(mu_t)||` and the per-step displacement. This gate uses only the noise-free
reference path, so it is allowed before any seed run. Blocking, per panel.

**G3, the error-only tie.** Two parts, both computed with the harness's own
paired-difference kernel `sweep.make_kernel` and its own root finder
`analyze.crossover_by_c`.

- G3a, single state. At the state and phases of Section 3, with `kind="full"`, the
  error-only tie `c*` must satisfy `|log(c* / 1.4368)| <= log(1.5)`. The wide tolerance
  is registered because one state carries one phase draw, and the closed forms of
  `reference.py` that pin the tie tightly are rank-one only, so no closed form exists for
  this field. Blocking.
- G3b, reproduction of a published number. The study's full-rank `d = 2` arm is rerun
  through `sweep.run_d(d=2, kind="full", n_states=32, n_batch=20, r_batch=100)` with a
  distinct output tag, and its `c*` must fall within 2 per cent of the published 1.522
  (`reports/lqr_crossover_corrected.md` Sec. 7.1). This is the provenance check that the
  environment running this experiment reproduces the environment that produced the
  study. Blocking.

Side effects of G3b, declared: it writes one new `.npz` under
`scripts/lqr_crossover/out/` under a tag that exists nowhere on disk, which the script
verifies before running, and it appends one row to `scripts/lqr_crossover/out/index.jsonl`.
It overwrites nothing.

---

## 8. Metrics, predictions and the decision rule

**Jitter.** For one seed and one arm,

    J = (1/N) * sum_{t=1..N} ||mu_t - mu_t^ref|| / sigma

with `mu_t^ref` the reference path of the same panel at the same step index. `J` is
therefore a divergence from the noise-free route, in units of the policy width.

**Ratio.** Per panel, `rho = median_r(J_ZO) / median_r(J_PW)` over the `R = 100` seeds,
`r = 0 .. 99`.

**Interval.** A paired percentile bootstrap over seeds: resample the 100 seed indices
with replacement, apply the same index set to both arms, recompute `rho`, 10 000
resamples, `numpy.random.default_rng(20260902)`, 95 per cent percentile interval. The
inferential unit is the path seed. No bootstrap is taken over steps, over coordinates,
or over states.

**Predictions.** `rho > 1` in both slow panels. `rho < 1` in both fast panels.

**Verdict per panel.**

- confirmed: the 95 per cent interval excludes 1 in the predicted direction.
- refuted: the interval excludes 1 in the other direction.
- inconclusive: otherwise.

**Secondary measures, reported and not adjudicated.**

- the median over steps and seeds of `cos(g_hat_t, g*(mu_t))` for each arm, with `g*`
  evaluated at that arm's own current mean;
- the final distance to the reference endpoint, `||mu_N - mu_N^ref|| / sigma`, median
  over seeds;
- `rho_ESTEP = median_r(J_ESTEP) / median_r(J_PW)`, with the same bootstrap, reported in
  the table only and not drawn in the figures;
- `eta` and the effective sample size of the E-step weights, so that an `eta` sitting at
  the edge of the study's grid is visible.

**What this check tests.** Whether the one-step ordering measured in the calibration
carries over to an accumulated path. It does not test the crossover location itself,
which the calibration measures directly, and it does not test any dimensional scaling.

**No retuning.** No panel, seed count, step size, `N`, `sigma`, state, or error field is
changed after any path result is seen. A failed prediction is reported as failed.

**Order of operations.** No comparison, ratio or figure of path results is computed until
every seed of the panel it belongs to has finished.

---

## 9. Figures

Four single-panel figures, matplotlib, PDF, at
`reports/figures/lqr_paths_{study,big}_{slow,fast}.pdf`. That is where this study's
existing figure `fig_lqr_crossover_contours.pdf` lives. The pattern `figures/` at
`.gitignore:207` matches that directory, so committing them needs `git add -f`, exactly
as the existing figure needed. Size 3.4 by 3.0 inches, serif font (STIX), equal aspect.

- Background: light gray contours of `Q_phi(s, .)`, 15 levels, on a grid with at least 8
  points per error period, where the error period is `2*pi/omega`. The registered
  fallback: if 8 points per period would need more than 600 grid points per axis, the
  background contours `Q^pi` instead and the caption says so. The grid is never coarser
  than 400 points per axis.
- Paths, seed 0 only, fixed here: `PW` in navy `#1F4E79`, `ZO` in brick `#C0504D`. The
  reference path is not drawn.
- Markers: a black diamond at the start, and a dot in each arm's colour at its end.
- One thin gray circle of radius `sigma` at the start, because it shows how many wiggles
  of the error field fit inside the blur.
- Legend with two entries, `PW` and `ZO`. Axis labels `a_1` and `a_2`. No title and no
  other annotation.
- The slow and fast panels of the same `eps` share axis limits, taken as the union of the
  drawn content of both panels padded by 10 per cent.

---

## 10. Compute and reproducibility

The work is small: 4 panels times 3 arms times 100 seeds times 80 steps times 32 samples
at `d = 2`. It is expected to take minutes.

`scripts/lqr_paths.py` has five stages, run as
`python scripts/lqr_paths.py --stage <stage> --out results/lqr_paths` with `<stage>` in
`gates`, `calibrate`, `paths`, `figures`, `report`. The harness is JAX, so the script
inherits `jax_enable_x64` and the CPU pin from the harness import, maps over seeds with
`jax.vmap`, and compiles the per-step update with `jax.jit`. It detects and records the
device list rather than choosing one, because the harness fixes the platform at import
(Section 2, item 1).

`results/lqr_paths/env.json` records hostname, device list, JAX version, NumPy version,
git hash, and wall time per stage. One `.npz` per panel holds the per-seed paths, with
the seed as the leading axis; `.npz` files are git-ignored repository-wide, so a manifest
with sha256 per file is written beside them, following
`reports/artifacts/lqr_npz_manifest.csv`.

**Where it runs.** The runs are executed on the FZI workstation (`gpu`,
`ispe-als-gpu-01.fzi.de` port 1222, repository `/home/human/repos/reppo`), at the same
commit as the Mac working tree. `nvidia-smi` is logged once, as a record of the machine.
The process is on CPU regardless, because of the harness pin of Section 2, and
`env.json` records the device list that JAX actually reported.

`scripts/lqr_paths.sbatch` is provided for the RWTH CLAIX-2023 cluster and requests one
GPU and 30 minutes. Its account, partition and environment lines are copied from the
repository's existing CLAIX-2023 job scripts (`slurm/gates.sh`, `slurm/ubar.sh`):
account `rwth2182`, partition `c23g`, `--gres=gpu:1`, and the repository `.venv`. The
GPU is requested because this preregistration was asked for one; the harness pins the
process to CPU at import, so the GPU is not used, and `env.json` records that. No other
cluster is targeted. The two machines of this project are the two in `CLAUDE.md`.

---

## 11. Limitations, registered in advance

1. `d = 2` only. Nothing here bears on any dimensional scaling.
2. One state, one phase draw, one system. The panels are an illustration at a point, not
   a population statement.
3. `sigma` is fixed and there is no covariance update, so the figure shows the mean-only
   dynamics of Proposition 2 and not the algorithm.
4. `eps_big` is an illustration size chosen so the critic error dominates. It is not the
   study's error size and no claim is made that a learned critic has it.
5. `Q^pi` is quadratic and globally smooth and the error is planted. A neural critic is
   neither.
6. The reference path is a stationary point of the blurred critic, not of `Q^pi`. At
   large `eps` those differ, and the figure shows the blurred critic's answer.

---

## 12. What would make this experiment misleading

Registered so the guards cannot be quietly dropped.

- Reading the slow and fast panels as evidence about `sqrt(d)`. They are not, and
  Section 1 says so.
- Comparing `J` across panels. `J` is a divergence from that panel's own reference path,
  and the reference paths differ between panels. Only the within-panel ratio `rho` is
  interpreted.
- Letting `eps` drift along a path, which would change the panel's error size mid-figure.
  Section 4 fixes `eps` once.
- Drawing the fast panels with a contour grid too coarse to resolve the error period,
  which would show a smooth landscape and hide the mechanism. Section 9 registers the
  resolution rule and the fallback to `Q^pi` with a caption.
- Quoting the ESTEP row as an operator comparison. Its magnitude is set by `eta`, its
  direction is normalised here, and the study already records that it is not the
  estimator Claim 4 is about.

---

## 13. Amendment line

**Everything above this line is append-only and was committed before any run of this
experiment. Everything below is appended after the run it reports, and never edits
anything above.**

### A1. Calibration outcome (2026-09-10, before any path run)

Run on the FZI workstation, `scripts/lqr_paths.py` sha256 `9334cfca...`, prereg
`4606164`, CPU, float64. Stage wall times: gates 78 s, calibrate 12 s.

**Setup as executed.** State index 0 of the 32 drawn, accepted on the first try.
`||a* - mu_0|| = 1.394540`. `sigma = 0.1394540`, which is above the effective
`min_std` of 0.1, so the panels sit inside the band the real policy can reach.
`eps_study = 0.01204190`. System guards as published: `rho_closed = 0.5347`,
`cond_H = 16.19`, zero rejections.

**Calibrated total-error crossover.** Grid `sigma*omega` in
`logspace(-1, 3, 41)`, `10^4` shared replicates, percentile bootstrap over replicates
with 1000 resamples.

| eps | value | `r_tot` | 95 % interval | `r_tot / 1.4368` |
|---|---|---|---|---|
| `eps_study` | 0.0120419 | 39.985 | [39.216, 40.564] | 27.8 |
| 10 x | 0.120419 | 4.1721 | [4.1117, 4.2306] | 2.90 |
| 30 x | 0.361256 | 1.9345 | [1.9133, 1.9570] | 1.35 |
| 100 x | 1.20419 | 1.5509 | [1.5335, 1.5695] | 1.08 |

`r_tot` falls as `1/eps` across the first three rows to within 4 per cent, which is the
scaling the crossover study measured (`reports/lqr_crossover_corrected.md` Sec. 4). It
saturates at 100 x, where the error already dominates and the crossover approaches the
error-only tie from above.

**`eps_big` by the registered rule.** The threshold is `1.5 x 1.4368 = 2.1553`. The 10 x
candidate gives 4.1721 and does not qualify. The 30 x candidate gives 1.9345 and does.
**`eps_big = 30 x eps_study = 0.361256`.**

**Panels.**

| panel | `eps` | `sigma*omega` | `omega` | `omega_RMS/omega` | `exp(-c^2/2)` | `N` |
|---|---|---|---|---|---|---|
| study_slow | 0.0120419 | 0.35921 | 2.5758 | 2.2776 | 9.375e-01 | 80 |
| study_fast | 0.0120419 | 159.941 | 1146.909 | 1.0000 | 0 (underflow) | 80 |
| big_slow | 0.361256 | 0.35921 | 2.5758 | 2.2776 | 9.375e-01 | 160 |
| big_fast | 0.361256 | 7.73810 | 55.4886 | 1.0000 | 9.943e-14 | 80 |

`N = 160` at big_slow is the registered single doubling of gate G2, which the panel
needed and passed.

**Two facts about these panels, recorded because they were assumed and are not true in
general.** First, `omega_RMS` equals `omega` to four digits at both fast panels and is
2.2776 times `omega` at the slow ones. The root-mean-square frequency approaches the
nominal one only as `sigma*omega` grows, so at the slow panels the two conventions
differ by a factor of 2.3. The dial remains the nominal `omega`, as registered. Second,
at both fast panels the Gaussian blur factor `exp(-c^2/2)` is `1e-13` or smaller, and at
study_fast it underflows to exactly zero. The blurred error gradient is therefore
negligible or exactly absent from the estimand `g*` at the fast panels: the error field
is invisible to the target and visible only to the estimators. That is the mechanism the
figure is about, and it is stated here rather than discovered later.

**Gate G3.** G3a, this state and these phases, error-only tie `c* = 1.5387` against the
asymptote 1.4368, `|log ratio| = 0.0685` against the registered bound 0.4055: PASS.
G3b, the study's full-rank `d = 2` arm rerun through `sweep.run_d` with tag
`_paths_g3b`, `c* = 1.5218` against the published 1.522, relative deviation `1.1e-4`
against the bound 0.02: PASS. The environment reproduces the study.

---

### A2. Three registered comparators that measure floating point, not the code

**All four estimator-facing gates pass. Three comparator tolerances do not, and the
reason in every case is cancellation inside the comparison rather than an error in the
quantity under test.** The registered verdicts stand in the record and are reported as
failures. The replacement criteria below are appended here, before any path run, with
the measurement that motivates each. Nothing above the amendment line is altered.

| gate | as registered | verdict as registered |
|---|---|---|
| G0a PW | max rel 2.370e-15, bound 1e-14 | PASS |
| G0a ZO | max rel 2.017e-15, bound 1e-14 | PASS |
| G0a ESTEP | max rel 1.166e-12, bound 1e-14 | **FAIL** |
| G0b | max rel 1.415e-09 as registered, 5.481e-09 at `eps = 0`, bound 1e-12 | **FAIL** |
| G0c smooth | max rel 7.025e-08, bound 1e-07 | PASS |
| G0c error part, slow panels | rel 3.48e-15 and 4.24e-15, bound 1e-07 | PASS |
| G0c error part, study_fast | blur underflows, closed form exactly zero | PASS |
| G0c error part, big_fast | rel 1.53, bound 1e-07 | **FAIL** |
| G1, all four panels and `eps = 0` | max abs z 1.73, bound 4 | PASS |
| G2, all four panels | tail radius 0.100 to 0.936 sigma, bound 1 | PASS |

**A2.1 G0a, the E-step arm.** The two routes feed the same closures the same draws and
their critic values differ by at most `1.33e-15` in absolute value, which is float64
rounding of two operation orders. That difference is amplified into the direction by the
E-step and by nothing else. Measured amplification of a relative perturbation of the
critic values into the returned direction, over 60 draws: pathwise **0.000**, which is
exact because the pathwise operator never reads a critic value, only its gradient;
zeroth order **1.41**; E-step **48.2**. At this state `eta = 0.1955` and
`sd(q)/eta = 1.164`, so the softmax exponent multiplies any perturbation by about five
before the weights are formed. The registered `1e-14` is below the conditioning of the
operator and cannot be met by any implementation of it.

Replacement, per arm: **`1e-14` for PW and ZO, `1e-11` for ESTEP.** The E-step bound is
one order above the measured worst case of `1.166e-12` over 100 draws, which is itself
consistent with 48 times a few times `1e-15`. Consequence, for scale: a relative
direction error of `1.166e-12` moves a step of `0.2 sigma = 2.789e-02` by `3.3e-14` in
action units.

**A2.2 G0b, the moving-mean closure.** `lqr.q_pi` returns the full action value
including the additive constant `gamma v/(1 - gamma)`, which is 2096.6 here out of a
typical `|q_pi|` of 3573. The gate differences two such numbers, so it loses about eight
digits before the comparison starts. The signature is decisive: the ABSOLUTE deviation
is flat while the differenced quantity varies over four orders of magnitude, and it
tracks the machine floor, while the RELATIVE deviation falls as `1/|delta|`.

| draw scale | max abs deviation | max rel deviation | min `\|delta\|` | max `\|q_pi\|` | floor `eps_mach x alpha_Q x \|q_pi\|` |
|---|---|---|---|---|---|
| 0.01 | 3.698e-14 | 1.271e-08 | 4.279e-07 | 3575 | 3.034e-14 |
| 0.1 | 4.735e-14 | 6.882e-10 | 2.931e-05 | 3576 | 3.036e-14 |
| 1 | 3.823e-14 | 2.869e-11 | 4.130e-04 | 3593 | 3.050e-14 |
| 10 | 4.441e-14 | 5.170e-11 | 3.283e-04 | 4514 | 3.831e-14 |
| 100 | 9.095e-13 | 4.116e-13 | 3.790e-02 | 6.046e+04 | 5.132e-13 |

An error in the closure's algebra would scale with `|delta|` and leave the relative
deviation flat. The opposite is observed at every scale.

Replacement: **the absolute deviation must sit at the cancellation floor of the
comparison, `|delta_closure - delta_exact| <= 10 x eps_mach x alpha_Q x max|q_pi|`.**
As executed over the gate's own draws: maximum absolute deviation `4.241e-14`, criterion
`3.070e-13` with `max|q_pi| = 3617.35`, so the deviation is 1.38 times the bare floor
`eps_mach x alpha_Q x max|q_pi| = 3.070e-14` and well inside the criterion. The table
above is the separate diagnostic that varies the draw scale; its worst case is selected
differently and its floor is quoted without the factor of ten.

**A2.3 G0c, the blurred error gradient at high frequency.** The closed form carries the
factor `exp(-c^2/2)`, so at the fast panels the quantity being checked is `1e-13` of the
integrand it is computed from. A Gauss-Hermite estimate in float64 cannot resolve that.

| panel | `c` | `exp(-c^2/2)` | `\|closed\|` | `\|quad_400\|` | `\|quad_400 - closed\|` | quadrature's own resolution |
|---|---|---|---|---|---|---|
| study_slow | 0.359 | 9.375e-01 | 1.494e-02 | 1.494e-02 | 5.20e-17 | 5.03e-17 |
| big_slow | 0.359 | 9.375e-01 | 4.482e-01 | 4.482e-01 | 1.33e-15 | 1.67e-15 |
| big_fast | 7.738 | 9.943e-14 | 1.364e-12 | 1.475e-12 | 1.11e-13 | 1.01e-13 |
| study_fast | 159.9 | 0 | 0 | 8.302e+00 | 8.302e+00 | 8.302e+00 |

The last column is the change in the quadrature between 200 and 400 nodes per axis, that
is, the comparator's own uncertainty. At big_fast the closed form sits inside it. At
study_fast the comparator returns 8.302 for a quantity whose true value is zero, so it
carries no information at all.

The phrase "relative deviation" admits two readings and the gate records both. Per
component, which is the stricter and is what the registered verdict is adjudicated on:
`3.48e-15` at study_slow, `4.24e-15` at big_slow, `1.53` at big_fast. Against the vector
magnitude: `3.48e-15`, `2.97e-15`, `8.10e-02`. Both readings fail at big_fast and both
pass at the slow panels, so the verdict does not turn on the choice.

Replacement: **the closed form must agree with the quadrature to the quadrature's own
demonstrated resolution**, that is
`|quad_400 - closed| <= max(1e-7 x |closed|, 2 x |quad_400 - quad_200|)`, and where the
blur factor underflows to zero the comparator is retired and the closed form is required
to be exactly zero. This is the same move the crossover study made: its gate G5a checks
this closed form relatively up to `c = 3` and switches to an absolute bound at `c = 6`
(`docs/prereg_lqr_crossover.md` Sec. 8). The closed form's only dependence on `c` is the
scalar `exp(-c^2/2)`, and it is verified here to `3.5e-15` and `4.2e-15` relative at the
two slow panels.

**What is not amended.** G1 and G2 are the gates that test the estimators and the
reference path rather than a comparator. They pass at every panel as registered, and no
tolerance of theirs is touched. The three replacements above change no panel, no seed,
no metric and no prediction.

---

### A3. One correction of fact (2026-09-10)

Section 9 states that committing the figures needs `git add -f` because the pattern
`figures/` at `.gitignore:207` matches `reports/figures/`. That is wrong.
`.gitignore:226` and `:227` carry `!reports/figures/` and `!reports/figures/**`, which
re-include the directory and its contents, added precisely because published figures had
been force-added one at a time and two were silently lost. The four figures are ordinary
untracked files and are committed normally. Nothing else in Section 9 is affected.
