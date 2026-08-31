# Running on RWTH CLAIX-2023 (WestAI)

Two job scripts, both calling the same entrypoint as a local run
(`scripts/train_and_export.py`) with the same hydra overrides:

| script | array index means | use it for |
|---|---|---|
| **`ladder.sh`** | one `(task, arm, seed)` triple | the registered confirmatory ladder |
| `train.sh`      | one seed, one shared config     | ad-hoc sweeps over seeds |

`ladder_matrix.sh` holds the frozen task/alpha/env definitions and is sourced by
both `ladder.sh` **and** the local driver `scripts/run_confirmatory_ladder.sh`, so
the cluster and the workstation can never launch different commands.

## Hardware and billing

`c23g` is the CLAIX-2023 ML segment: 50 nodes, 2× Xeon 8468 (96 cores), 512 GB host
RAM, **4× NVIDIA H100 94 GB**. One GPU is capped at 24 cores / 122 GB and billed as
24 core-hours per GPU-hour, which is exactly what the job headers request
(`--gres=gpu:1 --cpus-per-task=24 --mem-per-cpu=5000M`).

**There is no GPU test partition.** `devel` is free and needs no `--account`, but its
nodes have no accelerators, so it cannot be used to smoke-test these jobs. Use a
short `c23g` job instead (see below) — it costs ~0.1 GPU-hours.

## 1. Bootstrap — once, on a **login** node

```bash
git clone https://github.com/karamdaaboul/reppo.git && cd reppo
slurm/bootstrap.sh
```

This creates `$HPCWORK/reppo_runs/{exports,outputs,logs}` and symlinks `exports/`,
`outputs/` and `logs/` in the repo at them (`$HOME` has a small quota; `$HPCWORK` is
Lustre with **no backup**, which is fine for regenerable checkpoints), runs
`uv sync --frozen`, and prints the resolved package versions to record in the ledger.

`uv.lock` is tracked, so `--frozen` reproduces the exact dependency set used for
every other run in the study. Compute nodes have no internet, so the venv must exist
before any job starts — the job scripts only check for it and refuse to run otherwise.

There is deliberately **no `module load CUDA`**: `pyproject.toml` pins
`jax[cuda12]==0.5.2`, which ships the CUDA runtime, cuBLAS and cuDNN as pip wheels
inside `.venv`. JAX loads those in preference to any system install, so a module
would only risk a version mismatch.

## 2. Smoke test — 2 short runs, ~0.1 GPU-hours

```bash
ACCOUNT=<project> SMOKE=1 slurm/submit_ladder.sh
```

Runs array indices 0 and 32 (one `mjx_humanoid` G1 run and one `mjx_dmc` Hopper run)
at `TOTAL_STEPS=2621440`. `SMOKE=1` remaps the seeds into the **201+ exploratory**
namespace and writes to `outputs/smoke/` and `ledger/runs.d.smoke/`, so no reserved
confirmatory seed (101–108) is spent and the real ledger is untouched.

Check afterwards: both tasks `COMPLETED` in `sacct`, three `exports/*_s20*_{p25,p50,final}/`
directories per run each holding `actor.npz critic.npz normalizer.npz meta.json`, two
**distinct** `outputs/smoke/<tag>/` directories, and `seff <jobid>` showing GPU use.

## 3. The confirmatory ladder

```bash
ACCOUNT=<project> slurm/submit_ladder.sh                                # 48 runs
ACCOUNT=<project> TASKS="g1 leap hopper walker" slurm/submit_ladder.sh  # 64 runs
ACCOUNT=<project> THROTTLE=8 slurm/submit_ladder.sh                     # 8 at a time
```

**Always submit from the repo root** — Slurm's working directory is wherever `sbatch`
is called from, and every path in the scripts is relative to it.

3 tasks × 8 seeds (101–108) × 2 arms = 48 runs, ~34 GPU-hours (≈820 core-hours).
The array index decodes as

```
arm  = [A, B][ idx % 2 ]                 A = pathwise, B = weighted_mle (M=32)
seed = [101..108][ (idx / 2) % 8 ]
task = TASKS[ idx / 16 ]                 frozen priority: g1 -> leap -> hopper -> walker
```

so low indices — scheduled first under the `%N` throttle — walk the registered task
priority. `THROTTLE` defaults to 16, giving three waves and ~3 h wall-clock.

Both launchers refuse to start from a **dirty working tree**: the ledger records
`git_sha` for every confirmatory run, and a SHA that does not describe the code is
worthless. Commit first, or set `ALLOW_DIRTY=1` for a throwaway run.

Every run gets `hydra.run.dir=outputs/conf/<tag>`. This is not cosmetic: 48 tasks
starting in the same second would otherwise all resolve to Hydra's default
`outputs/<date>/<time>/` and overwrite each other's `metrics.npz`.

Dry-run the whole matrix without submitting anything:

```bash
for i in $(seq 0 47); do DRY_RUN=1 SLURM_ARRAY_TASK_ID=$i bash slurm/ladder.sh; done
```

## 4. Collect the ledger

Each array task writes `ledger/runs.d/conf-<tag>.json` — 48 concurrent appends to a
single JSONL would interleave and corrupt lines. Merge them afterwards:

```bash
python slurm/collect_ledger.py                              # -> ledger/runs.jsonl
python slurm/collect_ledger.py --dry-run                    # preview
python slurm/collect_ledger.py --dir ledger/runs.d.smoke --dry-run
```

It is idempotent (skips `run_id`s already present), so it is safe to run after a
partial array and again after the re-submissions. It exits non-zero and names any
run whose status is not `completed`.

## 5. Monitoring and cancelling

```bash
squeue --me                                      # RWTH asks: poll at most every 2 min
squeue --me -o "%.14i %.9P %.12j %.2t %.10M %R"
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS
seff <jobid>_<task>                              # per-task efficiency
tail -f logs/reppo-ladder_<jobid>_<task>.out

scancel <jobid>            # the whole array
scancel <jobid>_<task>     # one run
```

Each `.out` opens with hostname, date, job/array ids, **git commit**, the decoded
task/arm/seed/alpha, and `nvidia-smi`, so every result is traceable to the code that
produced it.

## 6. Re-submitting failures

`scripts/train_and_export.py` raises `SystemExit(2)` if any NaN reaches
`eval/episode_return`, and writes no export. Re-submit exactly the failed indices:

```bash
sbatch -A <project> -p c23g --array=7,19,42 --export=ALL,TASKS="g1 leap hopper" slurm/ladder.sh
```

The index decode is deterministic, so index 7 is always the same `(task, arm, seed)`.

## 7. Ad-hoc seed sweeps — `train.sh`

Unchanged from before, minus the placeholders. Config comes from environment
variables; unset ones keep the `config/reppo.yaml` defaults.

| variable | default | meaning |
|---|---|---|
| `ENVNAME` | `WalkerRun` | mujoco_playground task id (CamelCase) |
| `TASK` | `mjx_dmc` | hydra env group: `mjx_dmc`, `mjx_humanoid`, `brax` |
| `ACTOR_UPDATE_MODE` | `pathwise` | `pathwise` (arm A) or `weighted_mle` (arm B) |
| `ALPHA` | *(learned)* | set to **freeze** the entropy dual at this value |
| `EPS_E` | `0.5` | E-step KL budget for the eta dual |
| `MSTEP_DECOUPLED` | `false` | `true` = MPO decoupled M-step |
| `EPS_MU` / `EPS_SIGMA` | `0.1` / `5.0e-5` | decoupled M-step bounds |
| `BETA_SIGMA_FIXED` | *(learned)* | hold beta_sigma constant |
| `ESTEP_NUM_SAMPLES` | `32` | E-step samples M per state |
| `TOTAL_STEPS` | `50000000` | environment steps |
| `OVERRIDES` | `mjx_dmc_large_data` | hydra `experiment_overrides` group |

```bash
sbatch -A <project> --array=0-8 --export=ALL,ENVNAME=HumanoidRun slurm/train.sh
```

Note `train.sh` exposes no per-task `env.vmin/vmax/max_episode_steps` or
`env.asymmetric_obs`, so it cannot launch the LEAP or G1 configurations — use
`ladder.sh` for those.

## Outputs

Checkpoints land in `exports/<env>_<arm><variant>_s<seed>_{p25,p50,final}/` and are
standalone-loadable with `scripts/load_ckpt.py`. Frozen-alpha pathwise runs get the
`_fa` suffix, so arm A of the ladder appears as e.g.
`HopperHop_pathwise_fa_s101_final`. `exports/`, `outputs/` and `logs/` are all
git-ignored and live under `$HPCWORK`.
