"""Neutral fixed-bank policy width + exact tanh-saturation for Walker, G1, LEAP.

One bank per task, balanced across arms by construction; the SAME bank is used by
all 16 checkpoints of that task. Walker REUSES the already-saved canonical bank.
Each checkpoint applies its own observation normalizer (scripts/load_ckpt.py:127),
which is correct: the policy is a function of raw observations.

Saturation is exact, not Monte Carlo:
  P(|tanh Y| > t) = Phi((-c-mu)/s) + 1 - Phi((c-mu)/s),  c = atanh(t),  Y~N(mu,s^2)

Read-only on checkpoints; rollouts collect states only. No training.
"""
from __future__ import annotations
import hashlib, json, os, sys
import numpy as np
from scipy.special import erf

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO); sys.path.insert(0, REPO)
import jax, jax.numpy as jnp                                          # noqa: E402
from scripts.load_ckpt import load                                    # noqa: E402
from scripts.critic_fidelity.common import Harness, ACTION_CLIP       # noqa: E402

SEEDS = list(range(301, 309))
TASKS = {
 "walker": dict(pw="exports/WalkerRun_pathwise_fa_s%d_final",
                wml="exports/WalkerRun_weighted_mle_s%d_final",
                depths=(50,150,300,500,700,900),
                bank="reports/artifacts/walker_fixed_state_bank.npz"),
 "g1":     dict(pw="exports/G1JoystickFlatTerrain_pathwise_fa_s%d_final",
                wml="exports/G1JoystickFlatTerrain_weighted_mle_s%d_final",
                depths=(50,150,300,500,700,900),
                bank="reports/artifacts/g1_fixed_state_bank.npz"),
 "leap":   dict(pw="exports/LeapCubeRotateZAxis_pathwise_fa_s%d_final",
                wml="exports/LeapCubeRotateZAxis_weighted_mle_s%d_final",
                depths=(50,100,200,300,400,480),
                bank="reports/artifacts/leap_fixed_state_bank.npz"),
}
NENV, ROOT = 32, 20260905
T95, T99 = np.arctanh(0.95), np.arctanh(0.99)

def Phi(z): return 0.5*(1.0+erf(z/np.sqrt(2.0)))
def sat(mu, sg, c): return Phi((-c-mu)/sg) + 1.0 - Phi((c-mu)/sg)

def build(task, cfg):
    obs_all, src_all, dep_all = [], [], []
    for arm in ("PW","WML"):
        pat = cfg["pw"] if arm=="PW" else cfg["wml"]
        for sd in SEEDS:
            h = Harness(pat % sd, NENV)
            key = jax.random.fold_in(jax.random.PRNGKey(ROOT), hash((task,arm,sd)) % (2**31))
            k1, key = jax.random.split(key)
            o,_,st = h.reset(k1)
            for t in range(1, max(cfg["depths"])+1):
                ka,kb,key = jax.random.split(key,3)
                a = jnp.clip(h.pi(o).sample(seed=ka), -ACTION_CLIP, ACTION_CLIP)
                o,_,st,_,_,_ = h.env.step(jax.random.split(kb,NENV), st, a)
                if t in cfg["depths"]:
                    obs_all.append(np.asarray(o)); src_all += ["%s-s%d"%(arm,sd)]*NENV
                    dep_all += [t]*NENV
            print("    %s %s s%d" % (task, arm, sd), flush=True)
    obs = np.concatenate(obs_all,0).astype(np.float32)
    np.savez(cfg["bank"], obs=obs, source=np.array(src_all), depth=np.array(dep_all),
             depths=np.array(cfg["depths"]), n_env=np.array(NENV), root=np.array(ROOT))
    return obs, np.array(src_all), np.array(dep_all)

def stats(x): return dict(median=float(np.median(x)), mean=float(x.mean()),
                          p95=float(np.percentile(x,95)), maximum=float(x.max()))

def main():
    OUT = {}
    for task, cfg in TASKS.items():
        print("\n########## %s" % task)
        if os.path.exists(cfg["bank"]):
            z = np.load(cfg["bank"], allow_pickle=True)
            obs, src, dep = np.asarray(z["obs"],np.float32), np.array(z["source"]), np.array(z["depth"])
            print("  reusing bank", cfg["bank"])
        else:
            obs, src, dep = build(task, cfg)
        sha = hashlib.sha256(open(cfg["bank"],"rb").read()).hexdigest()
        npw = int(np.char.startswith(src,"PW").sum()); nwml = len(src)-npw
        print("  BANK_PATH=%s\n  BANK_SHA256=%s\n  NUM_STATES=%d  PW=%d  WML=%d"
              % (cfg["bank"], sha, obs.shape[0], npw, nwml))
        print("  DEPTH_DISTRIBUTION=" + " ".join("%d:%d"%(d,int((dep==d).sum())) for d in sorted(set(dep.tolist()))))
        ob = jnp.asarray(obs)
        res, satr = {}, {}
        for arm in ("PW","WML"):
            pat = cfg["pw"] if arm=="PW" else cfg["wml"]
            for sd in SEEDS:
                c = load(pat % sd)
                mu, sg = c.policy_dist(ob)
                mu, sg = np.asarray(mu,np.float64), np.asarray(sg,np.float64)
                res[(arm,sd)] = sg
                satr[(arm,sd)] = (sat(mu,sg,T95), sat(mu,sg,T99))
        d = res[("PW",301)].shape[-1]
        e = dict(bank=cfg["bank"], bank_sha256=sha, n_states=int(obs.shape[0]),
                 pw_states=npw, wml_states=nwml, depths=list(cfg["depths"]),
                 n_env=NENV, rng_root=ROOT, d=d, whole={}, percoord={}, paired={}, sat={})
        print("\n  %-4s %-5s %10s %12s %12s %13s | %10s %10s"
              % ("arm","seed","median","mean","p95","max","P|a|>.95","P|a|>.99"))
        for arm in ("PW","WML"):
            for sd in SEEDS:
                s = stats(res[(arm,sd)]); e["whole"]["%s_s%d"%(arm,sd)] = s
                s95, s99 = float(satr[(arm,sd)][0].mean()), float(satr[(arm,sd)][1].mean())
                e["sat"]["%s_s%d"%(arm,sd)] = dict(p95=s95, p99=s99,
                    percoord95=[float(satr[(arm,sd)][0][:,j].mean()) for j in range(d)],
                    percoord99=[float(satr[(arm,sd)][1][:,j].mean()) for j in range(d)])
                e["percoord"]["%s_s%d"%(arm,sd)] = [dict(coord=j, **stats(res[(arm,sd)][:,j])) for j in range(d)]
                print("  %-4s %-5d %10.4f %12.4f %12.4f %13.1f | %10.5f %10.5f"
                      % (arm,sd,s["median"],s["mean"],s["p95"],s["maximum"],s95,s99))
        ratios, deltas = [], []
        print("\n  paired: seed | PW med | WML med | ratio | dSat95 | dSat99")
        for sd in SEEDS:
            a=float(np.median(res[("PW",sd)])); b=float(np.median(res[("WML",sd)]))
            r=b/a; ratios.append(r); deltas.append(b-a)
            d95=e["sat"]["WML_s%d"%sd]["p95"]-e["sat"]["PW_s%d"%sd]["p95"]
            d99=e["sat"]["WML_s%d"%sd]["p99"]-e["sat"]["PW_s%d"%sd]["p99"]
            e["paired"]["s%d"%sd]=dict(pw=a,wml=b,ratio=r,delta=b-a,dsat95=d95,dsat99=d99)
            print("    %d | %8.4f | %9.4f | %6.1fx | %+.5f | %+.5f"%(sd,a,b,r,d95,d99))
        npos = int(sum(x>0 for x in deltas))
        cpos = []
        for j in range(d):
            row=[float(np.median(res[("WML",s)][:,j])-np.median(res[("PW",s)][:,j])) for s in SEEDS]
            cpos.append(int(sum(x>0 for x in row)))
        e["paired_percoord_npos"]=cpos
        e["summary"]=dict(wml_gt_pw_seed_pairs=npos,
                          coords_positive_8of8=int(sum(c==8 for c in cpos)),
                          coords_positive_5plus=int(sum(c>=5 for c in cpos)),
                          median_paired_ratio=float(np.median(ratios)), d=d)
        print("\n  WML_GT_PW_SEED_PAIRS = %d/8" % npos)
        print("  coordinates positive in 8/8 = %d of %d" % (e["summary"]["coords_positive_8of8"], d))
        print("  coordinates positive in >=5/8 = %d of %d" % (e["summary"]["coords_positive_5plus"], d))
        print("  median paired width ratio = %.2fx" % e["summary"]["median_paired_ratio"])
        nsat = int(sum(e["paired"]["s%d"%s]["dsat95"]>0 for s in SEEDS))
        e["summary"]["sat95_wml_gt_pw_pairs"]=nsat
        print("  saturation P(|a|>0.95): WML > PW in %d/8 pairs" % nsat)
        OUT[task]=e
    json.dump(OUT, open("reports/artifacts/fixed_bank_width_saturation.json","w"), indent=1)
    print("\nwrote reports/artifacts/fixed_bank_width_saturation.json")

if __name__ == "__main__":
    main()
