import json, subprocess, datetime, os
sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
now = datetime.datetime.now().astimezone().isoformat()
gap = ("Original k=16 padded runs were produced on a DIFFERENT machine "
       "(~/workspaces/reppo_original at 3b96deb, Amendment A) and carry NO ledger entry; "
       "the literal launch command is not recoverable. Reconstructed fields and evidence: "
       "seeds/arms/k/d from reports/probe1_restricted_z.md:9; alpha=0.01528 and the "
       "52.3M-step/21-eval budget from reports/probe_k6_report.md:32 (consistent with "
       "config/experiment_overrides/mjx_dmc_large_data.yaml plus the config default); "
       "export tag construction verified against scripts/train_and_export.py. "
       "Training behaviour at 07319d4 was verified BYTE-IDENTICAL to 3b96deb by "
       "slurm/pad16_parity.sh (35/35 arrays, max abs diff 0.0), so the regeneration "
       "preserves all three known defects. These checkpoints are REGENERATED, not the "
       "originals: Probe 1's published numbers were computed on the originals, so Probe 1 "
       "and Probe 4 sit on different checkpoint instances unless Probe 1 is re-run here.")
recs = []
for i in range(10):
    arm = "A" if i < 5 else "B"
    mode = "pathwise" if i < 5 else "weighted_mle"
    seed = i if i < 5 else i - 5
    variant = "_fa_pad16" if mode == "pathwise" else "_pad16"
    recs.append(dict(
        run_id="pad16-regen-%s-s%d" % (arm, seed), namespace="probe4_regeneration",
        label=None, task="WalkerRun", arm=arm, actor_update_mode=mode, seed=seed,
        action_pad=16, action_dim=22, real_dim=6,
        git_sha=sha, parity_verified_against="3b96debe5a8865cfa51f6ab4aaef294f3a36433d",
        alpha_entropy_frozen=0.01528, update_entropy_lagrangian=False,
        kl_bound=0.1, actor_kl_clip_mode="clipped", estep_num_samples=32, eps_e=0.5,
        expected_export="exports/WalkerRun_%s%s_s%d_final" % (mode, variant, seed),
        status="planned", registered_at=now, gpu=None, slurm_job=None,
        wall_clock_s=None, gpu_hours=None, checkpoint_sha256=None,
        return_metrics=None, estimator_diag=None,
        reason=("Regenerate the 10 k=16 padded checkpoints required by Probe 4 of "
                "docs/prospective_padding_error_field_analysis.md (sha256 "
                "34dd111af742750c3f96258b15f460ddd867dc42510dceaa73db7125f93e409b). "
                "Originals are absent from CLAIX."),
        provenance_gap=gap))
os.makedirs("ledger", exist_ok=True)
with open("ledger/runs_pad16_regen.jsonl", "w") as f:
    for r in recs:
        f.write(json.dumps(r) + "\n")
print("wrote ledger/runs_pad16_regen.jsonl with %d entries at sha %s" % (len(recs), sha))
