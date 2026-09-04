#!/usr/bin/env bash
# Confirmatory WalkerRun M-sweep, driven from the immutable pre-launch ledger
# ledger/runs_m_sweep_confirmatory.jsonl. One array task per ledger row index.
# The task REFUSES to run if the row's registered partition is not the partition
# it landed on, so the Sec. 2 seed-to-partition pairing cannot be broken by a
# mis-typed --partition.
#SBATCH --account=rwth2182
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/msweep_%A_%a.out
set -uo pipefail
cd "$HOME/repos/reppo"
IDX=${SLURM_ARRAY_TASK_ID:?}
PREREG=docs/prereg_m_sweep_confirmatory.md
PREREG_COMMIT=$(git log -1 --format=%H -- "$PREREG")
GIT_COMMIT=$(git rev-parse HEAD)

read -r RUN_ID M SEED PART CMD OUTP EXPORT <<< "$(
  ./.venv/bin/python - "$IDX" <<'PY'
import json, sys
i = int(sys.argv[1])
rows = [json.loads(l) for l in open("ledger/runs_m_sweep_confirmatory.jsonl")]
r = rows[i]
print(r["run_id"], r["M"], r["seed"], r["partition"],
      "@@" + r["command"] + "@@", r["output_path"], r["expected_export_final"])
PY
)"
CMD=$(./.venv/bin/python - "$IDX" <<'PY'
import json, sys
rows = [json.loads(l) for l in open("ledger/runs_m_sweep_confirmatory.jsonl")]
print(rows[int(sys.argv[1])]["command"])
PY
)

echo "=========== confirmatory M-sweep ==========="
echo "run_id      $RUN_ID"
echo "M           $M   seed $SEED"
echo "partition   registered=$PART  actual=${SLURM_JOB_PARTITION}"
echo "host        $(hostname)"
echo "git commit  $GIT_COMMIT"
echo "prereg      $PREREG_COMMIT   ($PREREG)"
echo "dirty       $(git status --porcelain | wc -l) files"
echo "slurm       ${SLURM_JOB_ID}  array ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if [ "$PART" != "${SLURM_JOB_PARTITION}" ]; then
  echo "REFUSING: registered partition $PART != actual ${SLURM_JOB_PARTITION}"; exit 3
fi
case "$CMD" in "python scripts/train_and_export.py "*) : ;;
  *) echo "REFUSING: unexpected command in ledger"; exit 4 ;; esac

t0=$(date +%s); start=$(date -Is)
CUDA_DEVICE_ORDER=PCI_BUS_ID XLA_PYTHON_CLIENT_PREALLOCATE=false \
  ./.venv/bin/${CMD} 2>&1 | tail -25
rc=${PIPESTATUS[0]}
t1=$(date +%s)
[ $rc -eq 0 ] && status=completed || status=failed
echo "status $status  rc $rc  wall $((t1-t0))s"

mkdir -p ledger/runs.d.m_sweep
./.venv/bin/python - "$RUN_ID" "$status" "$((t1-t0))" "$start" "$GIT_COMMIT" \
    "$PREREG_COMMIT" "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" "$EXPORT" <<'PY'
import hashlib, json, os, subprocess, sys
rid, status, wall, start, git, prereg, slurm, exp = sys.argv[1:9]
gpu = subprocess.run(["nvidia-smi","--query-gpu=name","--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip().splitlines()
d = dict(run_id=rid, status=status, wall_clock_s=int(wall), start=start,
         git_sha=git, prereg_commit=prereg, tier="confirmatory",
         slurm_job=slurm, partition=os.environ.get("SLURM_JOB_PARTITION"),
         gpu=(gpu[0] if gpu else None), export_path=exp,
         export_present=os.path.isdir(exp))
if d["export_present"]:
    h = hashlib.sha256()
    for fn in sorted(os.listdir(exp)):
        h.update(open(os.path.join(exp, fn), "rb").read())
    d["checkpoint_sha256"] = h.hexdigest()
json.dump(d, open("ledger/runs.d.m_sweep/%s.json" % rid, "w"), indent=1)
with open("ledger/runs.jsonl", "a") as f:
    f.write(json.dumps(d, sort_keys=True) + "\n")
print("ledger updated:", rid, status)
PY
echo "done $(date -Is)"
exit $rc
