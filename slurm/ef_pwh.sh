#!/usr/bin/env bash
#SBATCH --account=rwth2182
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --array=1-8
#SBATCH --job-name=ef-PWH
#SBATCH --output=slurm/logs/ef_pwh_%A_%a.out
EF_ARM=PW-H bash "$HOME/repos/reppo/slurm/ef_launch.sh"
