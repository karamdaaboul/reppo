"""Append the four A2 placement rows to the pre-launch M-sweep ledger.

Append-only: the sixteen rows registered at 6284d92 are untouched. The new rows
carry the A2 commit and the A2-era sha256 of the preregistration, and are marked
hardware-mismatched with the reason.
"""
from __future__ import annotations
import hashlib, json, os, subprocess, sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO); sys.path.insert(0, REPO)
from scripts.analysis.m_sweep_assert import ALPHA, resolve, tag   # noqa: E402

PREREG = "docs/prereg_m_sweep_confirmatory.md"
A2_COMMIT = subprocess.check_output(["git", "log", "-1", "--format=%H", "--", PREREG]).decode().strip()
A2_SHA = hashlib.sha256(open(PREREG, "rb").read()).hexdigest()
LEDGER = "ledger/runs_m_sweep_confirmatory.jsonl"
SEEDS = (301, 303, 305, 307)
M = 128

existing = [json.loads(l) for l in open(LEDGER)]
n0 = len(existing)
assert n0 == 16, n0
base = {r["seed"]: r for r in existing if r["M"] == M}

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
       "hydra.run.dir=outputs/m_sweep_confirmatory/walker_M{M}_s{seed}_a2")

new = []
for seed in SEEDS:
    r = resolve(M, seed)
    orig = base[seed]
    new.append(dict(
        run_id="ms-walker-M%d-s%d-a2" % (M, seed),
        supersedes=orig["run_id"],
        namespace="m_sweep_confirmatory", tier="confirmatory",
        task="WalkerRun", task_key="walker", d=6,
        arm="WML-%d-confirmatory" % M, M=M, seed=seed,
        prereg=PREREG, prereg_commit=orig["prereg_commit"],
        addendum="A2", addendum_commit=A2_COMMIT, prereg_sha256_at_a2=A2_SHA,
        partition="c23g", control_partition="c25g", hardware_paired=False,
        hardware_note=("DISCLOSED MISMATCH (Addendum A2): the M=32 control for this "
                       "seed ran on c25g, this M=128 run is on c23g. Delta_128 on "
                       "this seed contains a machine difference as well as an M "
                       "difference. Primary adjudication uses this run."),
        placement_reason=("c25g array 3519010 and its gating smoke 3519008 had not "
                          "started since submission; author decision per A2.3"),
        collision_hazard=("the queued c25g row %s writes the SAME export tag %s; if "
                          "3519010 ever starts it will overwrite this run's export"
                          % (orig["run_id"], orig["expected_export_final"])),
        command=CMD.format(M=M, seed=seed),
        output_path="outputs/m_sweep_confirmatory/walker_M%d_s%d_a2" % (M, seed),
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

with open(LEDGER, "a") as f:
    for r in new:
        f.write(json.dumps(r, sort_keys=True) + "\n")
print("appended %d rows at indices %d-%d" % (len(new), n0, n0 + len(new) - 1))
print("A2 commit  %s" % A2_COMMIT)
print("A2 sha256  %s" % A2_SHA)
for i, r in enumerate(new):
    print("  idx %-3d %-26s seed %d  %s  supersedes %s"
          % (n0 + i, r["run_id"], r["seed"], r["partition"], r["supersedes"]))
print("\nledger now %d rows; first 16 unchanged: %s"
      % (n0 + len(new),
         [json.loads(l) for l in open(LEDGER)][:16] == existing))
