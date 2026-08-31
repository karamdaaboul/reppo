#!/usr/bin/env bash
# One-time setup on a CLAIX LOGIN node. Run once from the repo root:
#
#     slurm/bootstrap.sh
#
# Compute nodes have no internet, so the venv must exist on shared storage before
# any job starts. $HOME has a small quota, so run artifacts live in $HPCWORK and the
# repo directories are symlinks into it -- created HERE rather than inside the job,
# because 48 array tasks racing on `ln -sfn` is a bug waiting to happen.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [ -n "${SLURM_JOB_ID:-}" ]; then
  echo "ERROR: run this on a login node, not inside a job (no internet on compute nodes)." >&2
  exit 1
fi

OUT_ROOT=${OUT_ROOT:-${HPCWORK:-$PWD}/reppo_runs}
if [ -z "${HPCWORK:-}" ]; then
  echo "WARNING: \$HPCWORK is unset; falling back to $OUT_ROOT inside \$HOME." >&2
  echo "         $HOME has a small quota -- check with r_quota." >&2
fi

echo "== 1. run-artifact directories under $OUT_ROOT"
mkdir -p "$OUT_ROOT/exports" "$OUT_ROOT/outputs" "$OUT_ROOT/logs"
for d in exports outputs logs; do
  if [ -e "$d" ] && [ ! -L "$d" ]; then
    echo "  $d/ is a real directory, leaving it alone (move it aside to use \$HPCWORK)"
  else
    ln -sfn "$OUT_ROOT/$d" "$d"
    echo "  $d -> $(readlink -f "$d")"
  fi
done

echo "== 2. python environment"
command -v uv >/dev/null || { echo "ERROR: uv not on PATH." >&2; exit 1; }
# uv.lock is tracked, so --frozen resolves the exact set used for every other run.
uv sync --frozen
echo "  venv: $PWD/.venv"

echo "== 3. resolved versions (record these in the ledger)"
./.venv/bin/python - <<'PY'
import importlib.metadata as md
for p in ("jax", "jaxlib", "jax-cuda12-plugin", "flax", "optax", "distrax",
          "mujoco", "mujoco-mjx", "brax", "hydra-core", "torch"):
    try:
        print(f"  {p:20s} {md.version(p)}")
    except md.PackageNotFoundError:
        print(f"  {p:20s} -")
PY

echo "== 4. cluster"
sinfo -p c23g -o "%P %l %D %G %c %m" 2>/dev/null || echo "  (sinfo unavailable -- not on the cluster?)"
r_quota 2>/dev/null || echo "  (r_quota unavailable)"

echo
echo "Bootstrap complete. Next:"
echo "  ACCOUNT=<project> SMOKE=1 slurm/submit_ladder.sh   # 2 short runs, ~0.1 GPU-h"
echo "  ACCOUNT=<project>          slurm/submit_ladder.sh   # the full ladder"
