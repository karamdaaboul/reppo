#!/usr/bin/env bash
#SBATCH --account=rwth2182
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --job-name=gates
#SBATCH --output=/hpcwork/qzi10910/estep_wt/slurm/logs/gates_%j.out
set -uo pipefail
cd /hpcwork/qzi10910/estep_wt
echo "host $(hostname)  commit $(git rev-parse HEAD)  $(date -Is)"
echo "--- T1: KL split unit tests (kl-split worktree, 0502b8f)"
( cd /hpcwork/qzi10910/bitchk_split && JAX_PLATFORMS=cpu ./.venv/bin/python tests/test_kl_split.py >/dev/null 2>&1; echo "    T1 rc=$?" )
echo
echo "--- T2 (eta dual direction) + OF (one-field resolved config)"
./.venv/bin/python /hpcwork/qzi10910/gates/test_estep_concentration.py
echo "gates rc=$?  $(date -Is)"
