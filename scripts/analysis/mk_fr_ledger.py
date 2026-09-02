"""Immutable pre-launch ledger for the 32-run faithful-repair replication.
Written BEFORE submission. One row per run. Arms counterbalanced across GPU
architecture within task and seed."""
import json, hashlib, subprocess, datetime, os

SHA = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
PREREG = subprocess.check_output(
    ["git", "rev-list", "-1", "HEAD", "--", "docs/prereg_corrected_operator_replication.md"],
    text=True).strip()
CORR = "cfbd8dd"
NOW = datetime.datetime.now().astimezone().isoformat()
TASKS = {
    "walker": dict(env="WalkerRun", d=6, alpha="0.014509912580251694",
                   env_args="env=mjx_dmc env.name=WalkerRun experiment_overrides=mjx_dmc_large_data"),
    "g1": dict(env="G1JoystickFlatTerrain", d=29, alpha="0.00020752247655764222",
               env_args=("env=mjx_humanoid env.name=G1JoystickFlatTerrain "
                         "env.asymmetric_obs=false experiment_overrides=mjx_humanoid_large_data")),
}
ARMS = {"PW-1-faithful-repair": "pathwise", "WML-32-faithful-repair": "weighted_mle"}
FLAGS = ("hyperparameters.faithful_same_point=true "
         "hyperparameters.fresh_minibatch_key=true "
         "hyperparameters.log_faithful_diag=true")
HORIZON, NUM_EVAL = 52297728, 20

rows = []
for tk, t in TASKS.items():
    for seed in range(301, 309):
        parity = (seed - 301) % 2                    # counterbalance across architecture
        for arm, mode in ARMS.items():
            is_pw = arm.startswith("PW")
            # parity 0: PW->c23g, WML->c25g ; parity 1: PW->c25g, WML->c23g
            gpu = ("c23g" if is_pw else "c25g") if parity == 0 else ("c25g" if is_pw else "c23g")
            run_id = "fr-%s-%s-s%d" % (tk, "PW1" if is_pw else "WML32", seed)
            outdir = "outputs/faithful_repair/%s_%s_s%d" % (tk, "PW1" if is_pw else "WML32", seed)
            variant = "_fa"
            exp = "exports/%s_%s%s_s%d" % (t["env"], mode, variant if is_pw else "", seed)
            cmd = ("python scripts/train_and_export.py %s "
                   "seed=%d num_trials=1 num_seeds=1 wandb.mode=disabled "
                   "hyperparameters.actor_update_mode=%s "
                   "hyperparameters.update_entropy_lagrangian=false "
                   "hyperparameters.ent_start=%s %s "
                   "hydra.run.dir=%s") % (t["env_args"], seed, mode, t["alpha"], FLAGS, outdir)
            cfg_hash = hashlib.sha256(cmd.encode()).hexdigest()[:16]
            rows.append(dict(
                run_id=run_id, namespace="faithful_repair_confirmatory",
                task=t["env"], task_key=tk, d=t["d"], arm=arm, actor_update_mode=mode,
                seed=seed, git_sha=SHA, correction_commit=CORR, prereg_commit=PREREG,
                config_hash=cfg_hash, alpha_entropy_frozen=t["alpha"],
                actor_sample_count=(1 if is_pw else 32), M=32, eps_e=0.5, kl_bound=0.1,
                faithful_same_point=True, fresh_minibatch_key=True, log_faithful_diag=True,
                published_gate="unchanged (actor_kl_clip_mode=clipped)",
                published_multiplier="unchanged (exp(lagrangian_log_param), unbounded)",
                total_time_steps=HORIZON, num_eval=NUM_EVAL,
                gpu_architecture=gpu, slurm_job=None, status="planned",
                registered_at=NOW, output_path=outdir, expected_export_final=exp + "_final",
                expected_export_p25=exp + "_p25", expected_export_p50=exp + "_p50",
                command=cmd))

os.makedirs("ledger", exist_ok=True)
path = "ledger/runs_faithful_repair.jsonl"
with open(path, "w") as f:
    for r in rows:
        f.write(json.dumps(r, sort_keys=True) + "\n")
chk = hashlib.sha256(open(path, "rb").read()).hexdigest()
open("ledger/runs_faithful_repair.sha256", "w").write(chk + "  " + os.path.basename(path) + "\n")
print("rows: %d" % len(rows))
print("ledger checksum (sha256): %s" % chk)
print("\ncounterbalance check - runs per (arm, architecture):")
import collections
c = collections.Counter((r["arm"], r["gpu_architecture"]) for r in rows)
for k in sorted(c): print("  %-24s %-5s %d" % (k[0], k[1], c[k]))
print("\nper (task, arm, architecture):")
c2 = collections.Counter((r["task_key"], r["arm"], r["gpu_architecture"]) for r in rows)
for k in sorted(c2): print("  %-7s %-24s %-5s %d" % (k[0], k[1], k[2], c2[k]))
assert len({(r["task"], r["arm"], r["seed"]) for r in rows}) == 32, "duplicate task-arm-seed"
assert len({r["output_path"] for r in rows}) == 32, "output path collision"
print("\nno duplicate task-arm-seed; 32 unique output paths")
