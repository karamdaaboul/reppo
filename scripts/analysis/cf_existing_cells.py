"""Inventory the 16 existing WalkerRun learned-sigma cells (phase 1.1).

These are the corrected-replication runs that will serve as the LEARNED half of the
2x2 covariance-freeze design. This table is what fixes the hardware pairing: every
new frozen run must execute on the same GPU architecture as its (operator, seed)
learned counterpart, and that mapping is read from here rather than assumed.

Deliberately contains NO return values. The frozen arms have not run; recording the
learned returns here would put outcome data one join away from an analysis that is
supposed to stay blinded until the integrity audit passes.

Usage: cf_existing_cells.py <out.csv>
"""

from __future__ import annotations

import csv
import glob
import hashlib
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LEDGER = os.path.join(REPO, "ledger", "runs_faithful_repair.jsonl")
RUNSD = os.path.join(REPO, "ledger", "runs.d.faithful_repair")
# array id -> partition, confirmed from sacct
ARRAY_PARTITION = {"3444831": "c23g", "3444832": "c25g"}
ARM_DIR = {"PW-1-faithful-repair": "PW1", "WML-32-faithful-repair": "WML32"}


def dir_sha256(d):
    """Exactly the definition used by slurm/fr_launch.sh:63-67."""
    if not os.path.isdir(d):
        return None
    h = hashlib.sha256()
    for f in sorted(glob.glob(d + "/*")):
        h.update(os.path.basename(f).encode())
        h.update(open(f, "rb").read())
    return h.hexdigest()


def file_sha256(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.isfile(p) else None


def main(out):
    rows = [json.loads(l) for l in open(LEDGER)]
    walker = [r for r in rows if r["task_key"] == "walker" and not r.get("voided")]
    walker.sort(key=lambda r: (r["arm"], r["seed"]))
    assert len(walker) == 16, len(walker)

    recs, problems = [], []
    for r in walker:
        rid = r["run_id"]
        post_path = os.path.join(RUNSD, rid + ".json")
        post = json.load(open(post_path)) if os.path.isfile(post_path) else {}
        arm_dir = ARM_DIR[r["arm"]]
        cfg = os.path.join(REPO, "outputs", "faithful_repair",
                           "walker_%s_s%d" % (arm_dir, r["seed"]), ".hydra", "config.yaml")
        exp = os.path.join(REPO, r["expected_export_final"])
        slurm = str(post.get("slurm_job", ""))
        array = slurm.split("_")[0] if "_" in slurm else ""
        part = ARRAY_PARTITION.get(array, "UNKNOWN")
        now = dir_sha256(exp)
        led = post.get("checkpoint_sha256")

        if part != r["gpu_architecture"]:
            problems.append("%s: ledger arch %s but ran on %s"
                            % (rid, r["gpu_architecture"], part))
        if led and now and led != now:
            problems.append("%s: checkpoint checksum changed since launch" % rid)
        if not os.path.isfile(cfg):
            problems.append("%s: resolved config missing" % rid)
        if post.get("status") != "completed":
            problems.append("%s: status=%s" % (rid, post.get("status")))

        recs.append(dict(
            run_id=rid, operator=("PW-1" if "PW" in r["arm"] else "WML-32"),
            arm=r["arm"], task=r["task"], seed=r["seed"],
            actor_update_mode=r["actor_update_mode"],
            actor_sample_count=r["actor_sample_count"], M=r["M"],
            gpu_architecture_ledger=r["gpu_architecture"],
            partition_actual=part, gpu_model=post.get("gpu", ""),
            slurm_job=slurm, status=post.get("status", ""),
            wall_clock_s=post.get("wall_clock_s", ""),
            config_hash=r["config_hash"],
            resolved_config_path=os.path.relpath(cfg, REPO),
            resolved_config_sha256=file_sha256(cfg),
            alpha_entropy=r["alpha_entropy_frozen"],
            horizon_total_time_steps=r["total_time_steps"],
            num_eval=r["num_eval"], eps_e=r["eps_e"], kl_bound=r["kl_bound"],
            faithful_same_point=r["faithful_same_point"],
            fresh_minibatch_key=r["fresh_minibatch_key"],
            log_faithful_diag=r["log_faithful_diag"],
            output_path=r["output_path"],
            export_final=r["expected_export_final"],
            export_p25=r["expected_export_p25"], export_p50=r["expected_export_p50"],
            export_present=os.path.isdir(exp),
            checkpoint_sha256_ledger=led, checkpoint_sha256_recomputed=now,
            checksum_match=(led == now),
            git_sha=post.get("git_sha", r["git_sha"]),
            prereg_commit=r["prereg_commit"], correction_commit=r["correction_commit"],
            command=r["command"],
        ))

    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
        w.writeheader()
        w.writerows(recs)

    print("wrote %s  (%d cells, no return values by design)" % (out, len(recs)))
    print("\nhardware pairing that the frozen runs must reproduce:")
    for op in ("PW-1", "WML-32"):
        for part in ("c23g", "c25g"):
            ss = sorted(r["seed"] for r in recs
                        if r["operator"] == op and r["partition_actual"] == part)
            print("  %-7s %s : seeds %s" % (op, part, ss))
    n23 = sum(1 for r in recs if r["partition_actual"] == "c23g")
    print("\ntotals: c23g %d, c25g %d" % (n23, len(recs) - n23))
    print("checksums verified: %d/%d" % (sum(1 for r in recs if r["checksum_match"]), len(recs)))
    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print("  " + p)
        raise SystemExit(1)
    print("\nno problems: all 16 completed, checksums stable, configs present")


if __name__ == "__main__":
    main(sys.argv[1])
