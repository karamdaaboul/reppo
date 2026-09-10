# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

Two things share one tree:

1. **Upstream REPPO** — the official code release for *Relative Entropy Pathwise Policy
   Optimization* (arXiv 2507.11019). JAX (`src/jaxrl/`) and Torch (`src/torchrl/`)
   implementations, hydra configs, and the paper's raw results (`results/`).
2. **A controlled study on top of it** (branch `estep-study`) comparing two actor-update
   operators *inside the same trainer*: `pathwise` (PW, the published loss) and
   `weighted_mle` (WML, an MPO E-step). Most of the tree — `docs/`, `ledger/`, `reports/`,
   `scripts/analysis/`, `slurm/` — is the apparatus for that study, not algorithm code.

Practical consequence: a change to `src/jaxrl/reppo.py` is a change to a *measurement
instrument*. Read "Bit-identity discipline" below before touching it.

## Commands

```bash
uv sync                       # local; creates .venv
uv sync --frozen              # cluster / anywhere reproducing a recorded run
```

Everything is invoked through the venv interpreter explicitly (`./.venv/bin/python ...`) —
that is what the slurm scripts and every docstring assume.

**Training (upstream entrypoint, throws the trained state away):**

```bash
./.venv/bin/python src/jaxrl/reppo.py env=mjx_dmc env.name=WalkerRun
```

**Training + checkpoint export (what the study actually uses):**

```bash
CUDA_VISIBLE_DEVICES=1 ./.venv/bin/python scripts/train_and_export.py \
    env=mjx_dmc env.name=WalkerRun experiment_overrides=mjx_dmc_large_data \
    seed=0 num_trials=1 num_seeds=1 wandb.mode=disabled
```

`num_trials` defaults to 10 upstream (sequential re-runs each overwriting `metrics.npz`) —
pass `num_trials=1`. Task ids go verbatim to the mujoco_playground registry, so CamelCase:
`WalkerRun`, `HumanoidRun`, `G1JoystickFlatTerrain`, `LeapCubeRotateZAxis`. The driver exits
`SystemExit(2)` and writes no export if a NaN reaches `eval/episode_return`.

**Tests** are standalone scripts, not pytest (there is no pytest in `uv.lock`). Each prints a
PASS/FAIL table and exits non-zero on failure; run one at a time:

```bash
./.venv/bin/python tests/test_kl_split.py
./.venv/bin/python tests/test_entropy_factorial.py
./.venv/bin/python tests/test_export_tag.py
./.venv/bin/python scripts/verify_estep.py      # E-step arm verification, needs GPU
```

Checks that need the canonical `exports/` tree print **DEFER**, never silently pass, so a run
on the wrong machine cannot look green. Some gates are end-to-end and only run under slurm
(`slurm/t1_bitwise.sh`, `slurm/fr_parity.sh`, `slurm/pad16_parity.sh`).

**Offline checkpoint analysis is CPU-only** — this venv's JAX wants its CUDA backend at
import, so set `JAX_PLATFORMS=cpu` (`CUDA_VISIBLE_DEVICES=""` alone raises).

**Lint/format:** `pyproject.toml` configures ruff (line-length 88, black-compatible quoting,
`E4,E7,E9,F`, `E731` ignored for jax lambdas). Neither ruff nor pytest is a project
dependency; use `uvx ruff` if you need it.

**Cluster:** `slurm/README.md` is authoritative — bootstrap, the confirmatory ladder's array
index decode, ledger collection, resubmission. Both launchers refuse a dirty working tree,
because the ledger records `git_sha` per run.

## Machines

Two remote machines, both reached by ssh alias from the Mac. Neither is a fallback for the
other — they do different jobs.

| alias | host | role |
|---|---|---|
| `rwth` (also `aachen`) | `login25-1.hpc.itc.rwth-aachen.de`, user `qzi10910` | RWTH CLAIX-2023/2025, SLURM. **All training and all confirmatory runs.** |
| `gpu` | `ispe-als-gpu-01.fzi.de:1222`, user `human` | FZI workstation, repo at `/home/human/repos/reppo`. Interactive development; no scheduler. |

`~/.ssh/config` sets `ControlMaster auto` with an 8-hour `ControlPersist` for `rwth`.
**Do not pass `-o ControlPath=none`** — it forces a fresh authentication on every call and
looks exactly like a rejected key, which has cost hours of misdiagnosis. Ride the existing
control socket.

### RWTH cluster

Account `rwth2182`; scratch is `$HPCWORK` = `/hpcwork/qzi10910`. `slurm/README.md` is
authoritative for launching — bootstrap, the ladder's array-index decode, ledger collection,
monitoring, resubmission. What that file does not cover:

**GPU partitions.** `c23g` (50 nodes, NVIDIA H100 94 GB) and `c25g` (29 nodes, H100 80 GB
HBM3), both `MaxTime=30-00:00:00`. `c25g` is newer and usually has shorter waits. `*_low`
variants exist for low-priority work. **Which partition a run lands on is recorded per seed in
the ledger** (`gpu_architecture`, consumed by `slurm/fr_launch.sh`), and the canonical
baselines deliberately alternate between the two — so any comparison that needs matched
hardware must pin the partition explicitly and check the ledger first.

**Worktrees on `$HPCWORK`** — the analysis reads whichever tree you are standing in, so
running in the wrong one silently produces wrong answers:

| path | state | purpose |
|---|---|---|
| `/hpcwork/qzi10910/estep_wt` | `estep-study` @ `3be8c11` | current work; analysis, banks, reports |
| `/hpcwork/qzi10910/logsplit_wt` | detached @ `964b251` | **pinned on purpose** — produced the 48 instrumented reruns; moving it destroys their provenance |
| `/hpcwork/qzi10910/reppo_runs/outputs/` | — | hydra run dirs (`metrics.npz`, `.hydra/`) |

Each campaign worktree has its own real `exports/` directory. That isolation is what stops a
new arm from overwriting canonical checkpoints — see "Exports and their names".

**Wall-clock, 50 M steps, one seed** (use these to set `--time`, and add headroom):
Walker ~23 min, LEAP ~29 min, G1 **~68 min**. A 45-minute limit kills every G1 job; G1 needs
90 minutes. Queue waits have run from minutes to a full day.

**Offline analysis runs on the login node** — it is CPU-only work over exported checkpoints
(`JAX_PLATFORMS=cpu`), needs no job, and takes seconds to a few minutes. Do not submit it.

### FZI workstation

Verified: alias `gpu`, `ispe-als-gpu-01.fzi.de` port **1222**, user `human`, repo at
`/home/human/repos/reppo`, tracking the same `origin`
(`git@github.com:karamdaaboul/reppo.git`) and currently on `estep-study` @ `3be8c11`.

**Unverified — fill in when the host is reachable:** GPU model and count, driver, whether a
scheduler is present, and whether any run in the ledger was produced here. It was unreachable
(connection timeout) at the last check, so nothing beyond the above is documented rather than
guessed. No confirmatory run should be attributed to this machine until its hardware is
recorded the way the cluster's is.

**Both machines must sit at the same commit before a campaign.** The two diverged once
(`kl-split` held the training instrumentation, `estep-study` held the analysis apparatus, and
no single commit had both) which blocked an experiment outright. They were merged at
`3be8c11`; verify with `git rev-parse HEAD` on each plus a content check
(`find src -name '*.py' | sort | xargs sha256sum | sha256sum`) — commit lineage alone is not
evidence.

## Architecture

**Config is hydra, layered** `config/reppo.yaml` ← `env/` ← `platform/` ←
`experiment_overrides/`. Override on the command line (`PARAM=VALUE`). `config/reppo.yaml`
carries the study's knobs with their rationale inline; treat its comments as normative.

**One trainer, two operators.** `src/jaxrl/reppo.py` (~1750 lines) holds `ReppoConfig` (the
frozen dataclass of every knob, heavily commented — read it before adding a flag),
`make_train_fn` with the rollout/learn closures, and the operator switch
`actor_update_mode: "pathwise" | "weighted_mle"`. The E-step primitives (`estep_weights`,
`eta_dual_loss`), the KL forms (`gaussian_kl_diag`, `gaussian_kl_diag_split`) and
`effective_sample_size` are module-level pure functions so tests can import them without the
training stack. `tests/reppo_upstream_snapshot.py` is a pristine copy of upstream's trainer,
kept only so `verify_estep.py` can assert the pathwise arm is bit-identical to published
code — never import it from training code.

**Bit-identity discipline (the load-bearing invariant).** Every research flag defaults to an
*exact* no-op, branched in Python so no XLA op is emitted, and the defaults must stay
byte-identical to the published implementation. Diagnostics that add critic forwards or an
extra autodiff pass (`log_q_spread`, `log_estimator_diag`, `log_cov_diag`, `log_eval_iqm`,
`log_faithful_diag`) perturb XLA fusion and hence float32 rounding — they are **off by
default and must stay off in confirmatory runs**; measure those quantities offline from an
exported checkpoint instead (`scripts/q_spread_from_ckpt.py`). When adding a flag, add its
no-op proof to the same commit.

**Exports and their names.** `scripts/train_and_export.py` builds the directory tag as
`<Env>_<mode><variant>_s<seed>_{p25,p50,final}`, where `variant` accumulates suffixes for
every dimension that could otherwise collide (`_fa` frozen alpha, `_dec_bs*`, `_pad16`,
`_m<M>`, `_rho*`, `_ent`/`_noent`, `_eps01`). **A suffix is appended only when the parameter
differs from its shipped default**, so tags already on disk stay byte-stable. Silently
overwriting a baseline export is a failure mode that has destroyed checkpoints in this
project twice — any new arm needs a tag suffix *and* a test that the new tags collide with
nothing (`tests/test_export_tag.py` reads the rule out of the source rather than
reimplementing it).

**Checkpoints are analysed standalone.** `scripts/load_ckpt.py:load(dir)` returns
`q_scalar`, `q_grad_a`, `policy_dist`, `policy_sample`, all taking **raw unnormalized**
observations and applying the saved running statistics internally. `mu`/`sigma` are the
**pre-squash** Gaussian parameters; the behaviour policy is `Tanh(Normal(mu, sigma))`, the
deterministic action is `tanh(mu)`.

**Where the study's state lives.**

| path | what it is |
|---|---|
| `docs/prereg_*.md`, `docs/protocol_*.md` | preregistrations and frozen protocols, append-only |
| `ledger/runs.jsonl` (+ `runs.d.*/`) | append-only run ledger; see `ledger/README.md` |
| `reports/*.md`, `reports/artifacts/` | analysis writeups and their data artifacts |
| `scripts/analysis/`, `scripts/{planted,lqr_crossover,critic_fidelity,msweep_audit}/` | analysis and offline-probe harnesses |
| `slurm/` | one job script per campaign; `ladder_matrix.sh` is the shared frozen matrix |
| `/Users/daaboul/workspaces/paper_status/` | current paper status; start at `research_status_20260909.md`, then `reruns_and_standards.md`, `posttanh_audit/REPORT.md` |

`exports/`, `outputs/`, `logs/` and `wandb/` are git-ignored and are symlinks into `$HPCWORK`
on the cluster.

## Research conventions — do not silently change these

- **Preregister before running.** Prereg docs and their amendments are committed *before* the
  runs they register; they are append-only. Analysis choices frozen in a protocol document
  are not re-chosen after seeing results.
- **Seed namespaces are contractual** (`ledger/README.md`): 0–4 pilot, 901 calibration,
  101–108 RESERVED confirmatory, 201+ permanently exploratory, 20260902 offline analysis.
  A development run must never be relabelled as confirmatory evidence. Failed and abandoned
  runs are recorded with their status, never deleted.
- **The inferential unit is the training seed.** Never bootstrap over states, dimensions or
  bank halves.
- **Frozen statistics:** ratio = median over seeds of per-seed log ratios, exponentiated;
  `score_window3` = mean of the final three logged evals (indices 18,19,20 of 21), frozen at
  `7edb8e8`; forward KL is `KL(pi_old || pi_theta)` on pre-tanh Gaussians with `sigma_theta`
  in the denominator — *not* the reverse `decoupled_kls` form.
- **Width claims always name their population.** Never write "WML is wider" unqualified:
  matched-state and own-occupancy give different answers, and the finding is the population
  dependence itself. Definitions: `docs/protocol_width_populations.md` (`2cb7ab7`).
- **Known non-reproducibility:** G1 and LEAP are not run-to-run reproducible
  (`G1_SAME_CODE_REPRODUCIBLE = NO`), so bitwise gates are untestable there — establish no-ops
  on fixed data instead. Canonical baselines are split across the `c23g`/`c25g` partitions by
  preregistered design, which blocks comparisons requiring matched hardware.

## Commits

The repository is anonymised for double-blind review: **commit as the user only, with no
`Co-Authored-By` or `Claude-Session` trailers** (the last 25 commits carry none; older ones
predate this rule). Messages follow the existing style — a specific, result-bearing subject
line under ~88 characters, then a prose body explaining what changed and *why it was
necessary*, including the hazard it avoids and the verification performed. Do not commit
non-prereg artifacts (figures, analysis scripts, reports) without explicit approval;
preregistrations and their amendments are the exception.
