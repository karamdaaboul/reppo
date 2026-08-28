# Running on RWTH CLAIX-2023

`slurm/train.sh` is a Slurm **job array**: one array task per seed, all sharing
one config. It calls the same entrypoint as a local run
(`scripts/train_and_export.py`) with the same hydra overrides, so anything that
works locally works here unchanged.

## Before the first submission

Two placeholders in the header must be filled in:

```
#SBATCH --time=__TIME_LIMIT__      # check the c23g limit:  sinfo -p c23g -o "%P %l"
#SBATCH --account=__PROJECT_ID__   # your project id, e.g. rwth1234
```

**Test on the `devel` partition first.** It is free, needs no `--account`, and
is capped at 1 hour -- long enough for a short run to prove the environment,
the GPU, and the config resolve before any quota is spent:

```bash
sbatch --partition=devel --time=00:30:00 --array=0 \
       --export=ALL,TOTAL_STEPS=2621440 slurm/train.sh
```

Do not run real experiments on `devel`.

## Submitting

Always submit **from the repo root** -- Slurm's working directory is wherever
`sbatch` is called from, and every path in the script is relative to it.

```bash
sbatch --array=0-8 slurm/train.sh                  # 9 seeds -- the usual case
sbatch slurm/train.sh                              # default array is only 0-2
```

**The header's `--array=0-2` is a small default, not the protocol.** Every
result in the study uses 9 seeds, and a 3-seed run cannot resolve the effects
being measured. Always pass `--array=0-8` (or more) for real experiments.

## Choosing the arm

The config is read from environment variables; unset ones keep the
`config/reppo.yaml` defaults.

| variable            | default              | meaning                                        |
|---------------------|----------------------|------------------------------------------------|
| `ENVNAME`           | `WalkerRun`          | mujoco_playground task id (CamelCase)          |
| `TASK`              | `mjx_dmc`            | hydra env group: `mjx_dmc`, `mjx_humanoid`, `brax` |
| `ACTOR_UPDATE_MODE` | `pathwise`           | `pathwise` (arm A) or `weighted_mle` (arm B)   |
| `ALPHA`             | *(learned)*          | set to **freeze** the entropy dual at this value |
| `EPS_E`             | `0.5`                | E-step KL budget for the eta dual              |
| `MSTEP_DECOUPLED`   | `false`              | `true` = MPO decoupled M-step                  |
| `EPS_MU`            | `0.1`                | decoupled M-step mean bound                    |
| `EPS_SIGMA`         | `5.0e-5`             | decoupled M-step scale bound                   |
| `BETA_SIGMA_FIXED`  | *(learned)*          | set to hold beta_sigma constant                |
| `ESTEP_NUM_SAMPLES` | `32`                 | E-step samples M per state                     |
| `TOTAL_STEPS`       | `50000000`           | environment steps                              |
| `OVERRIDES`         | `mjx_dmc_large_data` | hydra `experiment_overrides` group             |

Pass them with `--export`, which forwards your environment plus the listed
variables:

```bash
# arm A, HumanoidRun, 9 seeds
sbatch --array=0-8 --export=ALL,ENVNAME=HumanoidRun slurm/train.sh

# arm B, frozen alpha, eta dual, single KL clip
sbatch --array=0-8 --export=ALL,ENVNAME=HumanoidRun,ACTOR_UPDATE_MODE=weighted_mle,ALPHA=0.00329 \
       slurm/train.sh

# arm B with the decoupled M-step and a fixed beta_sigma
sbatch --array=0-2 --export=ALL,ENVNAME=HumanoidRun,ACTOR_UPDATE_MODE=weighted_mle,ALPHA=0.00329,MSTEP_DECOUPLED=true,BETA_SIGMA_FIXED=23 \
       slurm/train.sh
```

## Monitoring

```bash
squeue --me                          # your queued and running jobs
squeue --me -o "%.10i %.9P %.8j %.2t %.10M %.6D %R"
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS
tail -f logs/reppo_<jobid>_<task>.out
```

Each `.out` starts with the hostname, `nvidia-smi`, the resolved config, the
seed, and the **git commit hash**, so every result is traceable to the code
that produced it.

## Cancelling

```bash
scancel <jobid>          # the whole array
scancel <jobid>_<task>   # one seed
scancel --me             # everything you own
```

## Outputs

Checkpoints go to `exports/<env>_<arm>_s<seed>_{p25,p50,final}/` and are
standalone-loadable with `scripts/load_ckpt.py`. Slurm logs go to `logs/`.
Both directories are git-ignored.

## Environment -- build the venv ONCE on a login node

Compute nodes have **no internet**, so the script never installs anything. It
checks that `.venv` exists and exits with an error otherwise. Build it once,
from the repo root, on a **login node**:

```bash
uv sync --frozen
```

`pyproject.toml` pins `jax[cuda12]==0.5.2`, which ships the CUDA runtime,
cuBLAS and cuDNN as pip wheels inside `.venv`. JAX loads those in preference
to any system install, so there is **no** `module load CUDA` -- one would only
risk a version mismatch. Set `VENV=/path/to/.venv` to point at a venv kept
elsewhere.

## Storage -- outputs go to `$HPCWORK`

`$HOME` has a small quota. The `exports/` directory is already ~1 GB locally
and nine humanoid seeds produce several GB, so the script writes checkpoints
to `$HPCWORK/reppo_runs/exports` and symlinks `exports/` in the repo to it,
keeping every relative path (and `scripts/load_ckpt.py`) unchanged. Override
with `OUT_ROOT=...`.

Slurm's own `--output`/`--error` paths are fixed in the header and resolve
relative to the submit directory, so before the **first** submission point
`logs/` at `$HPCWORK` too:

```bash
mkdir -p "$HPCWORK/reppo_runs/logs" && ln -sfn "$HPCWORK/reppo_runs/logs" logs
```

## Billing

One GPU on `c23g` is limited to 24 cores and 122 GB, and is charged as
24 core-hours per GPU-hour. The header requests exactly that share
(`--cpus-per-task=24`, `--mem-per-cpu=5000M`). A full 50M-step run takes
roughly 20-40 minutes on an H100-class card, so a 9-seed array is on the
order of 4-6 GPU-hours.
