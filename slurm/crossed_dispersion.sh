#!/usr/bin/env bash
# Crossed frozen-critic dispersion, WalkerRun reference-law gate.
# Modes:  smoke | bank | run
#   smoke  the scratch validation (tiny config, scratch dir, nothing published)
#   bank   collect the preregistered 2048-state corrected-tier WalkerRun bank
#   run    the crossed design under BOTH reference laws on the frozen bank
#SBATCH --account=rwth2182
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm/logs/cd_%x_%j.out
set -euo pipefail
cd "$HOME/repos/reppo"
mkdir -p slurm/logs reports/artifacts
MODE="${1:?mode: smoke|bank|run}"
PY=./.venv/bin/python
echo "host $(hostname)  mode $MODE  git $(git rev-parse --short HEAD)  $(date -Is)"
git diff --quiet HEAD -- scripts/analysis/crossed_dispersion.py \
  || { echo "REFUSING: crossed_dispersion.py is dirty relative to HEAD"; exit 3; }

case "$MODE" in
  smoke)
    S=/hpcwork/$USER/cd_scratch_slurm
    rm -rf "$S"; mkdir -p "$S"
    JAX_ENABLE_X64=1 $PY scripts/analysis/test_crossed_dispersion.py "$S" unit
    $PY scripts/analysis/test_crossed_dispersion.py "$S" pipe
    $PY scripts/analysis/test_crossed_dispersion.py "$S" report
    ;;
  bank)
    OUT=reports/artifacts/cd_bank_walker_corrected.npz
    [ -e "$OUT" ] && { echo "REFUSING: $OUT exists; the bank is read-only"; exit 4; }
    $PY scripts/analysis/crossed_dispersion.py bank walker "$OUT"
    sha256sum "$OUT" | tee reports/artifacts/cd_bank_walker_corrected.sha256
    ;;
  run)
    BANK=reports/artifacts/cd_bank_walker_corrected.npz
    sha256sum -c reports/artifacts/cd_bank_walker_corrected.sha256
    $PY scripts/analysis/crossed_dispersion.py run walker "$BANK" \
        reports/artifacts/cd_walker_corrected.csv
    sha256sum -c reports/artifacts/cd_bank_walker_corrected.sha256
    ;;
  *) echo "unknown mode $MODE"; exit 2;;
esac
echo "done $(date -Is)"
