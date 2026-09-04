#!/usr/bin/env bash
# Corrected LEAP confirmatory replication, driven from the immutable pre-launch
# ledger ledger/runs_leap_corrected.jsonl. One array task per ledger row.
# Nothing here chooses anything: every field is read from the ledger, and the
# task refuses to run if the row's registered partition is not the one it landed
# on, or if the command does not look like the registered trainer.
#SBATCH --account=rwth2182
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=5000M
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/leap_%A_%a.out
set -uo pipefail
cd "$HOME/repos/reppo"
IDX=${SLURM_ARRAY_TASK_ID:?}
LEDGER=ledger/runs_leap_corrected.jsonl
PREREG=docs/prereg_leap_corrected.md
PREREG_COMMIT=$(git log -1 --format=%H -- "$PREREG")
GIT_COMMIT=$(git rev-parse HEAD)
mkdir -p slurm/logs

mapfile -t F < <(./.venv/bin/python - "$IDX" <<'PY'
import json, sys
rows = [json.loads(l) for l in open("ledger/runs_leap_corrected.jsonl")]
r = rows[int(sys.argv[1])]
print(r["run_id"]); print(r["command"]); print(r["expected_export_final"])
print(r["partition"]); print(r["config_hash"]); print(r["seed"]); print(r["arm"])
PY
)
RUN_ID="${F[0]}"; CMD="${F[1]}"; EXPECT="${F[2]}"
PART="${F[3]}"; CFGHASH="${F[4]}"; SEED="${F[5]}"; ARM="${F[6]}"

echo "=========== corrected LEAP confirmatory ==========="
echo "run_id      $RUN_ID"
echo "arm         $ARM   seed $SEED"
echo "partition   registered=$PART  actual=${SLURM_JOB_PARTITION}"
echo "host        $(hostname)"
echo "git commit  $GIT_COMMIT"
echo "prereg      $PREREG_COMMIT  ($PREREG)"
echo "config hash $CFGHASH"
echo "dirty       $(git status --porcelain | wc -l) modified/untracked files"
echo "slurm       ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if [ "$PART" != "${SLURM_JOB_PARTITION}" ]; then
  echo "REFUSING: registered partition $PART != actual ${SLURM_JOB_PARTITION}"; exit 3
fi
case "$CMD" in "python scripts/train_and_export.py "*) : ;;
  *) echo "REFUSING: unexpected command in ledger"; exit 4 ;; esac

VENV=$PWD/.venv
[ -x "$VENV/bin/python" ] || { echo "ERROR: no venv at $VENV"; exit 1; }
OUT_ROOT=${HPCWORK:-$PWD}/reppo_runs
mkdir -p "$OUT_ROOT/exports" "$OUT_ROOT/logs"
[ -L exports ] || [ -e exports ] || ln -sfn "$OUT_ROOT/exports" exports
export XLA_PYTHON_CLIENT_PREALLOCATE=true WANDB_MODE=disabled
export CUDA_DEVICE_ORDER=PCI_BUS_ID

t0=$(date +%s); start=$(date -Is)
./.venv/bin/${CMD} 2>&1 | tail -25
rc=${PIPESTATUS[0]}
t1=$(date +%s)
[ $rc -eq 0 ] && status=completed || status=failed
echo "status $status  rc $rc  wall $((t1-t0))s"

mkdir -p ledger/runs.d.leap
./.venv/bin/python - "$RUN_ID" "$status" "$((t1-t0))" "$start" "$GIT_COMMIT" \
    "$PREREG_COMMIT" "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" "$EXPECT" "$CFGHASH" <<'PY'
import hashlib, json, os, subprocess, sys
rid, status, wall, start, git, prereg, slurm, exp, cfg = sys.argv[1:10]
gpu = subprocess.run(["nvidia-smi","--query-gpu=name","--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip().splitlines()
d = dict(run_id=rid, status=status, wall_clock_s=int(wall), start=start,
         git_sha=git, prereg_commit=prereg, config_hash=cfg, tier="confirmatory",
         slurm_job=slurm, partition=os.environ.get("SLURM_JOB_PARTITION"),
         gpu=(gpu[0] if gpu else None), export_path=exp,
         export_present=os.path.isdir(exp))
if d["export_present"]:
    h = hashlib.sha256()
    for fn in sorted(os.listdir(exp)):
        h.update(open(os.path.join(exp, fn), "rb").read())
    d["checkpoint_sha256"] = h.hexdigest()
json.dump(d, open("ledger/runs.d.leap/%s.json" % rid, "w"), indent=1)
with open("ledger/runs.jsonl", "a") as f:
    f.write(json.dumps(d, sort_keys=True) + "\n")
print("ledger updated:", rid, status)
PY
echo "done $(date -Is)"
exit $rc
