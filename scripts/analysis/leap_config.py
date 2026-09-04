"""Render and hash the canonical corrected-LEAP configuration.

Run on BOTH machines; the printed hash must be identical before any run.
Every scientifically relevant value is pinned on the command line, so the Hydra
`_self_` composition order (which discards experiment_overrides for these keys)
cannot change what executes. That ordering is NOT repaired here: repairing it
would alter how historical reproduction commands resolve, and is a separate
decision.

Usage: leap_config.py [out.json]
"""
from __future__ import annotations
import hashlib, json, os, subprocess, sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO)
from hydra import compose, initialize_config_dir  # noqa: E402

ALPHA = "0.000782382907345891"      # alpha_901(LEAP), prereg_dimension_ladder.md:1637
SEEDS = tuple(range(301, 309))
ARMS = {"PW": "pathwise", "WML": "weighted_mle"}

PINNED = [
    "env=mjx_dmc", "env.name=LeapCubeRotateZAxis",
    "experiment_overrides=mjx_dmc_large_data",
    # --- LEAP-SPECIFIC pins, recovered from the single legacy LEAP launch
    # command in ledger/runs.jsonl. Without these the run silently takes the
    # mjx_dmc group defaults (vmin 0, vmax 150, 1000 steps), which are
    # Walker's values and NOT what every legacy LEAP run used. Addendum L1.
    "env.asymmetric_obs=false",
    "env.vmin=-10", "env.vmax=60",
    "env.max_episode_steps=500",
    "hyperparameters.max_episode_steps=500",
    "num_trials=1", "num_seeds=1", "wandb.mode=disabled",
    # --- optimization protocol, pinned to the EXECUTED corrected Walker/G1 values
    "hyperparameters.num_envs=1024",
    "hyperparameters.num_steps=128",
    "hyperparameters.num_mini_batches=128",
    "hyperparameters.num_epochs=4",
    "hyperparameters.total_time_steps=50000000",
    "hyperparameters.num_eval=20",
    # --- operator / E-step
    "hyperparameters.estep_num_samples=32",
    "hyperparameters.eps_e=0.5",
    "hyperparameters.kl_bound=0.1",
    # --- faithful-repair arm definition
    "hyperparameters.update_entropy_lagrangian=false",
    "hyperparameters.ent_start=" + ALPHA,
    "hyperparameters.faithful_same_point=true",
    "hyperparameters.fresh_minibatch_key=true",
    "hyperparameters.log_faithful_diag=true",
    # --- explicit no-ops, pinned so they cannot drift
    "hyperparameters.sqrt_rho=1.0",
    "hyperparameters.freeze_sigma=null",
    "hyperparameters.log_cov_diag=false",
]
SCI = ["num_envs","num_steps","num_mini_batches","num_epochs","lr","anneal_lr",
       "max_grad_norm","polyak","gamma","lmbda","total_time_steps","num_eval",
       "actor_update_mode","estep_num_samples","eps_e","kl_bound","ent_start",
       "update_entropy_lagrangian","faithful_same_point","fresh_minibatch_key",
       "log_faithful_diag","sqrt_rho","freeze_sigma","log_cov_diag",
       "critic_hidden_dim","actor_hidden_dim","exploration_noise_max",
       "exploration_noise_min","normalize_env","max_episode_steps","min_std",
       "vmin","vmax"]

def render(arm, seed):
    ov = PINNED + ["seed=%d" % seed,
                   "hyperparameters.actor_update_mode=%s" % ARMS[arm]]
    with initialize_config_dir(config_dir=os.path.join(REPO,"config"), version_base=None):
        c = compose(config_name="reppo", overrides=ov)
    h = c.hyperparameters
    sci = {}
    for k in SCI:
        v = h.get(k, c.env.get(k, None))
        sci[k] = None if v is None else (bool(v) if isinstance(v,bool) else
                 (float(v) if isinstance(v,(int,float)) else str(v)))
    return c, h, sci

def main(out=None):
    ne, ns = 1024, 128
    nmb, nep = 128, 4
    tr = 50000000//ns//ne; ei = tr//20
    iters = (tr//ei + int(tr % ei > 0))*ei
    print("canonical corrected-LEAP configuration")
    print("  task LeapCubeRotateZAxis   env=mjx_dmc   d=16   obs=32")
    print("  num_envs %d  num_steps %d  num_mini_batches %d  num_epochs %d" % (ne,ns,nmb,nep))
    print("  minibatch size %d   optimizer steps/iteration %d" % (ne*ns//nmb, nep*nmb))
    print("  iterations %d   env steps %d" % (iters, iters*ne*ns))
    print("  M 32  eps_e 0.5  kl_bound 0.1  num_eval 20")
    print("  alpha %s (frozen, update_entropy_lagrangian=false)" % ALPHA)
    hashes, rows = {}, []
    for arm in ARMS:
        for seed in SEEDS:
            c, h, sci = render(arm, seed)
            hh = hashlib.sha256(json.dumps(sci, sort_keys=True).encode()).hexdigest()
            hashes.setdefault(arm, set()).add(hh)
            rows.append(dict(arm=arm, seed=seed, hash=hh, sci=sci))
    print()
    for arm in ARMS:
        assert len(hashes[arm]) == 1, (arm, hashes[arm])
        print("  LEAP_CONFIG_HASH[%-3s] = %s   (identical across all 8 seeds)"
              % (arm, sorted(hashes[arm])[0]))
    combined = hashlib.sha256(
        "|".join(sorted(h for s in hashes.values() for h in s)).encode()).hexdigest()
    print("  LEAP_CONFIG_HASH[combined] = %s" % combined)
    print("\n  git HEAD = %s" % subprocess.check_output(["git","rev-parse","HEAD"]).decode().strip())
    if out:
        json.dump(dict(combined=combined,
                       per_arm={a: sorted(v)[0] for a,v in hashes.items()},
                       pinned=PINNED, alpha=ALPHA, rows=rows),
                  open(out,"w"), indent=1)
        print("  wrote %s" % out)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv)>1 else None)
