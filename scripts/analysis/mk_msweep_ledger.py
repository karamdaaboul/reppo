"""Immutable pre-launch ledger for the confirmatory WalkerRun M-sweep.

One row per registered run, written BEFORE any job is submitted, exactly as
ledger/runs_faithful_repair.jsonl was. Every row carries the preregistration
commit so a run can never be traced to the wrong protocol.
"""
from __future__ import annotations
import hashlib, json, os, subprocess, sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO)
sys.path.insert(0, REPO)
from scripts.analysis.m_sweep_assert import (            # noqa: E402
    ALPHA, SEEDS, partition_for, tag, resolve,
)

PREREG_PATH = "docs/prereg_m_sweep_confirmatory.md"
PREREG_COMMIT = subprocess.check_output(
    ["git", "log", "-1", "--format=%H", "--", PREREG_PATH]).decode().strip()
PREREG_SHA256 = hashlib.sha256(open(PREREG_PATH, "rb").read()).hexdigest()
OUT = "ledger/runs_m_sweep_confirmatory.jsonl"

CMD = ("python scripts/train_and_export.py env=mjx_dmc env.name=WalkerRun "
       "experiment_overrides=mjx_dmc_large_data seed={seed} num_trials=1 "
       "num_seeds=1 wandb.mode=disabled "
       "hyperparameters.actor_update_mode=weighted_mle "
       "hyperparameters.update_entropy_lagrangian=false "
       "hyperparameters.ent_start=" + ALPHA + " "
       "hyperparameters.faithful_same_point=true "
       "hyperparameters.fresh_minibatch_key=true "
       "hyperparameters.log_faithful_diag=true "
       "hyperparameters.estep_num_samples={M} "
       "hydra.run.dir=outputs/m_sweep_confirmatory/walker_M{M}_s{seed}")

rows = []
for M in (128, 512):
    for seed in SEEDS:
        r = resolve(M, seed)
        p = partition_for(M, seed)
        ctrl = "c25g" if seed % 2 == 1 else "c23g"
        rows.append(dict(
            run_id="ms-walker-M%d-s%d" % (M, seed),
            namespace="m_sweep_confirmatory",
            tier="confirmatory",
            task="WalkerRun", task_key="walker", d=6,
            arm="WML-%d-confirmatory" % M, M=M, seed=seed,
            prereg=PREREG_PATH, prereg_commit=PREREG_COMMIT,
            prereg_sha256=PREREG_SHA256,
            partition=p, control_partition=ctrl, hardware_paired=(p == ctrl),
            hardware_note=("paired to this seed's WML-32 control" if p == ctrl else
                           "DISCLOSED MISMATCH: M=512 is on c23g for all seeds "
                           "(prereg Sec. 2); condition (iii) is not within-hardware "
                           "on this seed"),
            command=CMD.format(M=M, seed=seed),
            output_path="outputs/m_sweep_confirmatory/walker_M%d_s%d" % (M, seed),
            expected_export_final="exports/%s_final" % tag(M, seed),
            ent_start=float(ALPHA), update_entropy_lagrangian=False,
            faithful_same_point=True, fresh_minibatch_key=True,
            log_faithful_diag=True, log_cov_diag=False,
            sqrt_rho=r["sqrt_rho"], freeze_sigma=r["freeze_sigma"],
            eps_e=r["eps_e"], kl_bound=r["kl_bound"],
            states_per_iteration=r["states_per_iteration"],
            states_per_minibatch=r["states_per_minibatch"],
            updates_per_iteration=r["updates_per_iteration"],
            iterations=r["iterations"], total_updates=r["total_updates"],
            env_steps=r["env_steps"], num_eval=r["num_eval"],
            control_arm="WalkerRun_weighted_mle_s%d_final" % seed,
            status="planned", slurm_job=None,
        ))

with open(OUT, "w") as f:
    for r in rows:
        f.write(json.dumps(r, sort_keys=True) + "\n")
print("wrote %s: %d rows" % (OUT, len(rows)))
print("prereg commit %s" % PREREG_COMMIT)
print("prereg sha256 %s" % PREREG_SHA256)
print("sha256 of ledger %s" % hashlib.sha256(open(OUT, "rb").read()).hexdigest())
for M in (128, 512):
    ps = {r["partition"] for r in rows if r["M"] == M}
    n = sum(1 for r in rows if r["M"] == M)
    nm = sum(1 for r in rows if r["M"] == M and not r["hardware_paired"])
    print("  M=%-4d %d runs, partitions %s, disclosed mismatches %d"
          % (M, n, sorted(ps), nm))
