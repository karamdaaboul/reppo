# Frozen definitions for the registered confirmatory dimension ladder (prereg L.1).
#
# Sourced by BOTH launchers so the two can never drift:
#   scripts/run_confirmatory_ladder.sh   local, two GPUs, bash workers
#   slurm/ladder.sh                      SLURM job array, one GPU per array task
#
# Nothing here may be changed on outcomes. The alphas are the seed-901 calibration
# values frozen in docs/prereg_dimension_ladder.md (L.1.10 / L.1.16 / L.1.31); the env args
# are the launch commands registered in L.1.7.

# Arms. A is the pathwise operator, B is weighted MLE with M = 32. The mode also
# sets the KL Monte-Carlo sample count (16 vs 32) -- a known, registered coupling
# (L.1.14 / L.1.17), deliberately left intact.
ARMS="A B"

# Confirmatory seed namespace. Reserved; see ledger/README.md.
SEEDS="101 102 103 104 105 106 107 108"

# Fixed task priority, set before outcomes: G1 -> LEAP -> Hopper -> Walker.
# Walker is excluded by default pending the L.1.16 cohort decision.
TASKS_DEFAULT="g1 leap hopper"

env_args () {
  case "$1" in
    g1)     echo "env=mjx_humanoid env.name=G1JoystickFlatTerrain env.asymmetric_obs=false experiment_overrides=mjx_humanoid_large_data" ;;
    leap)   echo "env=mjx_dmc env.name=LeapCubeRotateZAxis env.asymmetric_obs=false experiment_overrides=mjx_dmc_large_data env.vmin=-10 env.vmax=60 env.max_episode_steps=500 hyperparameters.max_episode_steps=500" ;;
    hopper) echo "env=mjx_dmc env.name=HopperHop experiment_overrides=mjx_dmc_large_data" ;;
    walker) echo "env=mjx_dmc env.name=WalkerRun experiment_overrides=mjx_dmc_large_data" ;;
    *)      echo "env_args: unknown task '$1'" >&2; return 1 ;;
  esac
}

alpha_of () {
  case "$1" in
    g1)     echo "0.00020752247655764222" ;;
    leap)   echo "0.000782382907345891" ;;
    hopper) echo "0.00037288447492755949" ;;
    walker) echo "0.014509912580251694" ;;   # seed-901 calibration (L.1.31); 0.01528 retired
    *)      echo "alpha_of: unknown task '$1'" >&2; return 1 ;;
  esac
}

taskname () {
  case "$1" in
    g1)     echo G1JoystickFlatTerrain ;;
    leap)   echo LeapCubeRotateZAxis ;;
    hopper) echo HopperHop ;;
    walker) echo WalkerRun ;;
    *)      echo "taskname: unknown task '$1'" >&2; return 1 ;;
  esac
}

mode_of () {  # mode_of <arm>
  [ "$1" = A ] && echo pathwise || echo weighted_mle
}

algorithm_version_of () {  # algorithm_version_of <arm>
  [ "$1" = A ] && echo "A-pathwise" || echo "B-weighted-mle"
}

# Decode a flat array index into (task, arm, seed).
#
#   arm  = ARMS[  idx % 2 ]
#   seed = SEEDS[ (idx / 2) % 8 ]
#   task = TASKS[  idx / 16 ]
#
# Index order therefore walks the frozen task priority, which matters under an
# --array=...%N throttle because low indices are scheduled first. Sets the shell
# variables LADDER_TASK, LADDER_ARM and LADDER_SEED.
ladder_decode () {  # ladder_decode <idx> [tasks]
  local idx="$1" tasks="${2:-$TASKS_DEFAULT}"
  local -a t s a
  # shellcheck disable=SC2206
  t=($tasks); s=($SEEDS); a=($ARMS)
  local n_arms=${#a[@]} n_seeds=${#s[@]}
  local per_task=$(( n_arms * n_seeds ))
  local total=$(( per_task * ${#t[@]} ))
  if [ "$idx" -lt 0 ] || [ "$idx" -ge "$total" ]; then
    echo "ladder_decode: index $idx out of range 0..$((total-1))" >&2
    return 1
  fi
  LADDER_ARM=${a[$(( idx % n_arms ))]}
  LADDER_SEED=${s[$(( (idx / n_arms) % n_seeds ))]}
  LADDER_TASK=${t[$(( idx / per_task ))]}
}

ladder_size () {  # ladder_size [tasks]
  local tasks="${1:-$TASKS_DEFAULT}"
  local -a t a s
  # shellcheck disable=SC2206
  t=($tasks); a=($ARMS); s=($SEEDS)
  echo $(( ${#t[@]} * ${#a[@]} * ${#s[@]} ))
}
