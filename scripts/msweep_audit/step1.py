"""STEP 1: effective post-merge config for every M=32/128/512 run."""
import json, os, glob, sys
from omegaconf import OmegaConf

ROOT="/home/human/workspaces/reppo_original"

def effective(run_dir):
    """Reproduce reppo.py:1370  cfg.hp = merge(cfg.hp, cfg.experiment_overrides.hp)."""
    c = OmegaConf.load(os.path.join(run_dir, ".hydra", "config.yaml"))
    hp = OmegaConf.merge(c.hyperparameters, c.experiment_overrides.hyperparameters)
    return c, hp

# baseline M=32 arm: hydra_run_dir recorded in each export's meta.json
runs=[]
for tag in [f"HumanoidRun_weighted_mle_s{s}_final" for s in (0,1,2,3,5,6,7,8)]:
    p=os.path.join(ROOT,"exports",tag,"meta.json")
    if os.path.exists(p):
        m=json.load(open(p)); runs.append(("M32", m["seed"], m["hydra_run_dir"], m))
for d in sorted(glob.glob(os.path.join(ROOT,"outputs/msweep/M*_s2*/"))):
    tag=os.path.basename(d.rstrip("/")); M,seed=tag.split("_s")
    ex=os.path.join(ROOT,"exports",f"HumanoidRun_weighted_mle_s{seed}_final","meta.json")
    m=json.load(open(ex)) if os.path.exists(ex) else None
    runs.append((M, int(seed), d.rstrip("/"), m))

rows=[]
for arm, seed, rd, meta in runs:
    if not os.path.exists(os.path.join(rd,".hydra","config.yaml")):
        rows.append(dict(arm=arm,seed=seed,err="no .hydra",rd=rd)); continue
    c,hp = effective(rd)
    ne, ns, nmb, nep = hp.num_envs, hp.num_steps, hp.num_mini_batches, hp.num_epochs
    states = ne*ns                       # states collected per learn step (rollout)
    mb_states = states//nmb              # minibatch size in STATES
    grad_steps_per_iter = nmb*nep        # optimizer updates per learn iteration
    rows.append(dict(
        arm=arm, seed=seed, rd=os.path.relpath(rd,ROOT),
        M=hp.estep_num_samples, mode=hp.actor_update_mode,
        num_envs=ne, num_steps=ns, states_per_iter=states,
        num_mini_batches=nmb, mb_states=mb_states, num_epochs=nep,
        grad_steps_per_iter=grad_steps_per_iter,
        total_time_steps=hp.total_time_steps, lr=hp.lr, opt="adam(max_grad_norm=%s)"%hp.max_grad_norm,
        eps_e=hp.eps_e, kl_bound=hp.kl_bound, ent_start=hp.ent_start,
        actor_min_std=hp.actor_min_std, kl_clip=hp.actor_kl_clip_mode,
        update_kl_lag=hp.update_kl_lagrangian, update_ent_lag=hp.update_entropy_lagrangian,
        mstep_dec=hp.mstep_decoupled, kl_num_samples=hp.get("kl_num_samples","<absent>"),
        polyak=hp.polyak, gamma=hp.gamma, lmbda=hp.lmbda, num_bins=hp.num_bins,
        seed_cfg=c.seed,
        iters=(meta or {}).get("iteration"), env_steps=(meta or {}).get("time_steps"),
        train_s=(meta or {}).get("train_seconds"), final=(meta or {}).get("final_eval_return"),
        nan=(meta or {}).get("nan_in_eval"), exported=meta is not None))
json.dump(rows, open("/tmp/claude-1001/-home-human-workspaces-reppo-original/52e91891-229f-40e5-b38b-e15e5143a5bd/scratchpad/step1.json","w"), indent=1)

# --- the two questions ---
keys_budget=["num_envs","num_steps","states_per_iter","num_mini_batches","mb_states",
             "num_epochs","grad_steps_per_iter","total_time_steps","lr","opt","polyak",
             "gamma","lmbda","num_bins","eps_e","kl_bound","ent_start","actor_min_std",
             "kl_clip","update_kl_lag","update_ent_lag","mstep_dec","kl_num_samples"]
print("=== effective config per arm (unique value sets) ===")
for k in keys_budget:
    byarm={}
    for r in rows:
        if "err" in r: continue
        byarm.setdefault(r["arm"],set()).add(str(r.get(k)))
    vals={a:sorted(v) for a,v in byarm.items()}
    same = len({tuple(v) for v in vals.values()})==1
    print(f"  {k:22s} {'SAME' if same else 'DIFFERS':8s} " +
          "  ".join(f"{a}={'/'.join(v)}" for a,v in sorted(vals.items())))
print()
print("=== horizon actually executed ===")
for arm in ("M32","M128","M512"):
    rs=[r for r in rows if r.get("arm")==arm and r.get("iters")]
    if rs:
        print(f"  {arm}: iters={sorted({r['iters'] for r in rs})} "
              f"env_steps={sorted({r['env_steps'] for r in rs})} "
              f"n_exported={len(rs)} train_s={[round(r['train_s']) for r in rs]}")
