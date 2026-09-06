#!/usr/bin/env bash
#SBATCH --account=rwth2182
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=slurm/logs/t2t5_g1_%j.out
set -uo pipefail
cd "$HOME/repos/reppo"
echo "host $(hostname)  commit $(git rev-parse HEAD)  $(date -Is)"
echo "########## Walker d=6 (regression: must reproduce the validated result)"
./.venv/bin/python tests/test_entropy_factorial.py
echo "rc_walker=$?"
echo
echo "########## G1 d=29"
EF_TEST_CKPT=exports/G1JoystickFlatTerrain_weighted_mle_s301_final \
EF_TEST_CKPT_PW=exports/G1JoystickFlatTerrain_pathwise_fa_s301_final \
EF_TEST_BANK=reports/artifacts/g1_fixed_state_bank.npz \
EF_TEST_LABEL="G1 d=29" \
./.venv/bin/python tests/test_entropy_factorial.py
echo "rc_g1=$?  $(date -Is)"
