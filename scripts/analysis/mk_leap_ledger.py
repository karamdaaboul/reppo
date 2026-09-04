"""Immutable pre-launch ledger for the corrected LEAP replication.

Written BEFORE any job is submitted. Every command is built from the SAME
`PINNED` list that `leap_config.py` hashes, so the registered configuration, the
config hash and the executed command cannot drift apart.
"""
from __future__ import annotations
import hashlib, json, os, subprocess, sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO); sys.path.insert(0, REPO)
from scripts.analysis.leap_config import PINNED, ARMS, SEEDS, ALPHA, render  # noqa: E402

PREREG = "docs/prereg_leap_corrected.md"
PREREG_COMMIT = subprocess.check_output(
    ["git", "log", "-1", "--format=%H", "--", PREREG]).decode().strip()
PREREG_SHA256 = hashlib.sha256(open(PREREG, "rb").read()).hexdigest()
PARTITION = "c23g"          # both arms on one partition: the comparison is symmetric
OUT = "ledger/runs_leap_corrected.jsonl"

rows = []
for arm, mode in ARMS.items():
    for seed in SEEDS:
        cmd = ("python scripts/train_and_export.py " + " ".join(PINNED) +
               " seed=%d" % seed +
               " hyperparameters.actor_update_mode=%s" % mode +
               " hydra.run.dir=outputs/leap_corrected/leap_%s_s%d" % (arm, seed))
        _, _, sci = render(arm, seed)
        rows.append(dict(
            run_id="leap-%s-s%d" % (arm, seed),
            namespace="leap_corrected", tier="confirmatory",
            task="LeapCubeRotateZAxis", task_key="leap", d=16,
            arm="%s-faithful-repair" % ("PW-1" if arm == "PW" else "WML-32"),
            seed=seed, M=32,
            prereg=PREREG, prereg_commit=PREREG_COMMIT, prereg_sha256=PREREG_SHA256,
            config_hash=hashlib.sha256(
                json.dumps(sci, sort_keys=True).encode()).hexdigest(),
            partition=PARTITION,
            command=cmd,
            output_path="outputs/leap_corrected/leap_%s_s%d" % (arm, seed),
            expected_export_final="exports/LeapCubeRotateZAxis_%s%s_s%d_final" % (
                mode, "_fa" if mode == "pathwise" else "", seed),
            alpha=float(ALPHA), gamma=sci["gamma"], lmbda=sci["lmbda"],
            vmin=sci["vmin"], vmax=sci["vmax"],
            max_episode_steps=sci["max_episode_steps"],
            num_mini_batches=sci["num_mini_batches"], num_epochs=sci["num_epochs"],
            num_envs=sci["num_envs"], num_steps=sci["num_steps"],
            eps_e=sci["eps_e"], kl_bound=sci["kl_bound"], num_eval=sci["num_eval"],
            status="planned", slurm_job=None,
        ))

with open(OUT, "w") as f:
    for r in rows:
        f.write(json.dumps(r, sort_keys=True) + "\n")
print("wrote %s: %d rows" % (OUT, len(rows)))
print("prereg commit %s" % PREREG_COMMIT)
print("prereg sha256 %s" % PREREG_SHA256)
print("ledger sha256 %s" % hashlib.sha256(open(OUT, "rb").read()).hexdigest())
import collections
c = collections.Counter(r["config_hash"] for r in rows)
print("distinct config hashes: %d (expect 2, one per arm)" % len(c))
for h, n in c.items():
    print("   %s  x%d" % (h, n))
print("expected export tags:")
for r in rows[:2] + rows[-1:]:
    print("   %-22s %s" % (r["run_id"], r["expected_export_final"]))
