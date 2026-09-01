#!/usr/bin/env bash
# Submit the confirmatory ladder as a SLURM job array.
#
#   ACCOUNT=rwth1234 slurm/submit_ladder.sh                              # 48 runs on c23g
#   ACCOUNT=rwth1234 TASKS="g1 leap hopper walker" slurm/submit_ladder.sh # 64 runs
#   ACCOUNT=rwth1234 THROTTLE=8 slurm/submit_ladder.sh                   # 8 concurrent
#   ACCOUNT=rwth1234 SMOKE=1 slurm/submit_ladder.sh                    # 2 short test runs
#
# Run from the repo root. The array size is derived from TASKS, so the header in
# slurm/ladder.sh carries no --array and no --account.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck source=slurm/ladder_matrix.sh
source slurm/ladder_matrix.sh

TASKS=${TASKS:-$TASKS_DEFAULT}
THROTTLE=${THROTTLE:-16}
SMOKE=${SMOKE:-0}
N=$(ladder_size "$TASKS")

# ACCOUNT is required for every real partition. The `devel` partition is free and
# needs no account, but its nodes have NO GPUs, so it cannot be used to smoke-test
# this job -- a smoke run is a short, cheap c23g job instead (~0.1 GPU-hours).
if [ -z "${ACCOUNT:-}" ]; then
  echo "ERROR: ACCOUNT is not set. Pass your RWTH/WestAI project id, e.g." >&2
  echo "         ACCOUNT=rwth1234 slurm/submit_ladder.sh" >&2
  echo "       List your projects with:  r_wlm_usage -q" >&2
  exit 1
fi

for d in exports outputs logs .venv; do
  [ -e "$d" ] || { echo "ERROR: $d missing -- run slurm/bootstrap.sh first." >&2; exit 1; }
done

if [ "$SMOKE" = 1 ]; then
  # Two short runs exercising both env groups AND both arms: index 0 is G1
  # (mjx_humanoid, arm A pathwise) and index 33 is Hopper (mjx_dmc, arm B
  # weighted MLE). Arm is idx % 2, so an even/even pair such as 0,32 would
  # leave the weighted-MLE E-step, dual and ESS code paths untested.
  # slurm/ladder.sh remaps the seeds into the 201+ exploratory namespace, so
  # no reserved confirmatory seed is spent.
  : "${TOTAL_STEPS:=2621440}"
  echo "SMOKE: 2 short runs on ${PARTITION:-c23g}, TOTAL_STEPS=$TOTAL_STEPS (~0.1 GPU-hours)"
  set -x
  sbatch --account="$ACCOUNT" --partition="${PARTITION:-c23g}" \
         --time=00:30:00 --array=0,33 --job-name=reppo-smoke \
         --export=ALL,TASKS="$TASKS",SMOKE=1,TOTAL_STEPS="$TOTAL_STEPS" \
         slurm/ladder.sh
  exit $?
fi

DIRTY=$(git status --porcelain | wc -l)
if [ "$DIRTY" -ne 0 ] && [ "${ALLOW_DIRTY:-0}" != 1 ]; then
  echo "ERROR: working tree is dirty ($DIRTY files). The ledger records git_sha for" >&2
  echo "       every confirmatory run; commit first or set ALLOW_DIRTY=1." >&2
  exit 1
fi

echo "tasks    : $TASKS"
echo "runs     : $N   (array 0-$((N-1))%$THROTTLE)"
echo "account  : $ACCOUNT"
echo "commit   : $(git rev-parse HEAD)"
set -x
sbatch --account="$ACCOUNT" --partition="${PARTITION:-c23g}" \
       --array="0-$((N-1))%$THROTTLE" \
       --export=ALL,TASKS="$TASKS" \
       slurm/ladder.sh
