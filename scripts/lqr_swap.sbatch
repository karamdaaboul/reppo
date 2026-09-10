#!/usr/bin/env bash
# docs/prereg_lqr_swap.md, Sec. 8. RWTH CLAIX-2023.
#
# Header copied from scripts/lqr_paths.sbatch, which took it from slurm/gates.sh and
# slurm/ubar.sh. Account rwth2182, a c23g partition, the repository .venv.
#
# THE GPU IS REQUESTED AND NOT USED, for the same reason as the path study:
# scripts/lqr_crossover/__init__.py sets JAX_PLATFORMS=cpu and asserts at import that the
# first device is a CPU, so anything importing the harness runs on CPU.
# results/lqr_swap/env.json records the device list JAX actually reported. CLAUDE.md's
# rule for offline analysis is that it belongs on a login node rather than in the queue,
# and that remains the cheaper way to run this.
#
#   sbatch scripts/lqr_swap.sbatch
#
#SBATCH --account=rwth2182
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=lqr_swap
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/hpcwork/qzi10910/estep_wt}" || exit 1

echo "host $(hostname)  commit $(git rev-parse HEAD)  $(date -Is)"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=index,name,driver_version \
    --format=csv,noheader

OUT=results/lqr_swap
mkdir -p "$OUT" logs

# The order is fixed by the preregistration. `components` produces the blind
# predictions, which are committed below the amendment line before `measure` runs, so
# this script stops there unless BLIND_COMMITTED is set to the commit that carries them.
for STAGE in gates components; do
    echo "--- stage $STAGE  $(date -Is)"
    ./.venv/bin/python scripts/lqr_swap.py --stage "$STAGE" --out "$OUT" || {
        echo "stage $STAGE failed with rc=$?"; exit 1; }
done

if [ -z "${BLIND_COMMITTED:-}" ]; then
    echo "stopping before measure: set BLIND_COMMITTED=<commit of the A1 predictions>"
    exit 0
fi
echo "blind predictions committed at ${BLIND_COMMITTED}"
for STAGE in measure figures report; do
    echo "--- stage $STAGE  $(date -Is)"
    ./.venv/bin/python scripts/lqr_swap.py --stage "$STAGE" --out "$OUT" || {
        echo "stage $STAGE failed with rc=$?"; exit 1; }
done
echo "done $(date -Is)"
