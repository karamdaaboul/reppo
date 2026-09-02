#!/usr/bin/env bash
#SBATCH --job-name=fr-sc
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=5000M
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?}" || exit 1
source .venv/bin/activate
export CUDA_DEVICE_ORDER=PCI_BUS_ID JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p reports/artifacts/fr_samecritic
mapfile -t CK < <(./.venv/bin/python -c "
import json
for r in (json.loads(l) for l in open('ledger/runs_faithful_repair.jsonl')):
    print(r['expected_export_final'])" | sort)
C="${CK[${SLURM_ARRAY_TASK_ID:?}]}"
python scripts/analysis/fr_samecritic.py "$C" "reports/artifacts/fr_samecritic/$(basename "$C").npz"
