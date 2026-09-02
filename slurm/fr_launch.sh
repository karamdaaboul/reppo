#!/usr/bin/env bash
# Confirmatory faithful-repair replication, driven from the immutable pre-launch
# ledger ledger/runs_faithful_repair.jsonl. One array task per ledger row of the
# requested architecture. Nothing here chooses anything: every field is read.
#SBATCH --job-name=fr-conf
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=5000M
#SBATCH --time=05:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?}" || exit 1
ARCH=${FR_ARCH:?set FR_ARCH to c23g or c25g}
IDX=${SLURM_ARRAY_TASK_ID:?}

read -r RUN_ID CMD EXPECT < <(./.venv/bin/python - "$ARCH" "$IDX" <<'PY'
import json, sys
arch, idx = sys.argv[1], int(sys.argv[2])
rows = [json.loads(l) for l in open("ledger/runs_faithful_repair.jsonl")]
rows = [r for r in rows if r["gpu_architecture"] == arch]
rows.sort(key=lambda r: (r["task_key"], r["arm"], r["seed"]))
r = rows[idx]
print(r["run_id"], r["command"], r["expected_export_final"])
PY
)
source .venv/bin/activate
export CUDA_DEVICE_ORDER=PCI_BUS_ID JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=true WANDB_MODE=disabled
SHA=$(git rev-parse HEAD)
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo "=========== faithful-repair confirmatory ==========="
echo "run_id $RUN_ID | arch $ARCH | job ${SLURM_ARRAY_JOB_ID:-}_${IDX} | $(hostname) | gpu ${GPU:-unknown}"
echo "commit $SHA"
echo "cmd    $CMD"
echo "==================================================="
t0=$(date +%s); start=$(date -Is)
if $CMD; then status=completed; else status=failed; fi
t1=$(date +%s)
echo "DONE $RUN_ID status=$status wall_s=$((t1-t0))"
mkdir -p ledger/runs.d.faithful_repair
./.venv/bin/python - "$RUN_ID" "$status" "$((t1-t0))" "$start" \
    "${SLURM_ARRAY_JOB_ID:-}_${IDX}" "${GPU:-unknown}" "$SHA" "$EXPECT" <<'PY'
import json, sys, os, hashlib, glob
rid, status, wall, start, slurm, gpu, sha, exp = sys.argv[1:9]
csum = None
if os.path.isdir(exp):
    h = hashlib.sha256()
    for f in sorted(glob.glob(exp + "/*")):
        h.update(os.path.basename(f).encode()); h.update(open(f, "rb").read())
    csum = h.hexdigest()
json.dump(dict(run_id=rid, status=status, wall_clock_s=int(wall),
               gpu_hours=round(int(wall)/3600, 4), start=start, slurm_job=slurm,
               gpu=gpu, git_sha=sha, export_path=exp,
               export_present=os.path.isdir(exp), checkpoint_sha256=csum),
          open("ledger/runs.d.faithful_repair/%s.json" % rid, "w"), indent=1)
PY
[ "$status" = completed ] || exit 1
