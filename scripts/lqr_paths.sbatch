#!/usr/bin/env bash
# docs/prereg_lqr_paths.md, Sec. 10. RWTH CLAIX-2023.
#
# Account, partition and environment lines are copied from the repository's existing
# CLAIX-2023 job scripts (slurm/gates.sh, slurm/ubar.sh). There is no sbatch script for
# the crossover study to copy from: that study is CPU-only offline analysis and ran on a
# login node.
#
# THE GPU IS REQUESTED AND NOT USED. scripts/lqr_crossover/__init__.py sets
# JAX_PLATFORMS=cpu and asserts at import that the first device is a CPU, so anything
# that imports the harness runs on CPU. The request is here because the preregistration
# asks for it; results/lqr_paths/env.json records the device list JAX actually reported.
# The whole job is minutes. CLAUDE.md's rule for offline analysis is that it belongs on
# the login node, not in the queue, and that remains the cheaper way to run this.
#
#   sbatch scripts/lqr_paths.sbatch
#
#SBATCH --account=rwth2182
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=lqr_paths
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/hpcwork/qzi10910/estep_wt}" || exit 1

echo "host $(hostname)  commit $(git rev-parse HEAD)  $(date -Is)"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=index,name,driver_version \
    --format=csv,noheader

OUT=results/lqr_paths
mkdir -p "$OUT" logs

# The order is fixed by the preregistration: the panel-dependent gates cannot run before
# the calibration, and the path stage refuses to start unless every blocking gate passes
# for the panels it is about to run.
for STAGE in gates calibrate gates paths figures report; do
    echo "--- stage $STAGE  $(date -Is)"
    ./.venv/bin/python scripts/lqr_paths.py --stage "$STAGE" --out "$OUT" || {
        echo "stage $STAGE failed with rc=$?"; exit 1; }
done
echo "done $(date -Is)"
