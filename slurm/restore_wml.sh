#!/usr/bin/env bash
#SBATCH --account=rwth2182
#SBATCH --partition=c25g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/restore_wml_%j.out
bash "$HOME/repos/reppo/slurm/restore_s301.sh" weighted_mle
