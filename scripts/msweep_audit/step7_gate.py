"""STEP 7 (exact): elementwise KL and gate branch, recovered from lagrangian_loss.

  lagrangian_loss_i = -lambda * (kl_i - kl_bound)      (reppo.py:893-895)
  => kl_i = kl_bound - lagrangian_loss_i / lambda
  => gate fires (state gets ONLY the KL penalty, no WML signal)  <=>  kl_i >= kl_bound
                                                                <=>  lagrangian_loss_i <= 0
"""
import numpy as np, glob, os, json, collections
ROOT="/home/human/workspaces/reppo_original"; KLB=0.1
ARMS=collections.OrderedDict()
for s in (0,1,2,3,5,6,7,8):
    m=json.load(open(f"{ROOT}/exports/HumanoidRun_weighted_mle_s{s}_final/meta.json"))
    ARMS.setdefault("M32",[]).append((s,m["hydra_run_dir"]))
for d in sorted(glob.glob(f"{ROOT}/outputs/msweep/M*_s2*/")):
    t=os.path.basename(d.rstrip("/")); M,sd=t.split("_s")
    ARMS.setdefault(M,[]).append((int(sd),d.rstrip("/")))
MSZ={"M32":32,"M128":128,"M512":512}
LATE=slice(-6,-1)
rows=[]
for arm,runs in ARMS.items():
    for seed,rd in runs:
        p=os.path.join(rd,"metrics.npz")
        if not os.path.exists(p): continue
        z=np.load(p)
        ret=np.asarray(z['eval/episode_return']).reshape(len(z['time_step']),-1)[:,0]
        if (~np.isnan(ret)).sum()<16: continue
        ll=np.asarray(z['train/lagrangian_loss']); ll=ll.reshape(ll.shape[0],-1)   # (T,2048)
        lam=np.asarray(z['train/lagrangian']).reshape(ll.shape[0],-1)[:,0]          # (T,)
        kl_i = KLB - ll/np.maximum(lam,1e-12)[:,None]        # exact elementwise KL
        gated = (ll<=0.0)                                     # kl_i >= kl_bound
        L=np.arange(ll.shape[0])[LATE]
        rows.append(dict(arm=arm,M=MSZ[arm],seed=seed,ret=float(np.nanmean(ret[LATE])),
            lam=float(lam[L].mean()), gate=float(gated[L].mean()),
            kl_mean=float(kl_i[L].mean()), kl_med=float(np.median(kl_i[L])),
            kl_sd=float(kl_i[L].std()),
            kl_iqr=float(np.percentile(kl_i[L],75)-np.percentile(kl_i[L],25)),
            kl_p05=float(np.percentile(kl_i[L],5)), kl_p95=float(np.percentile(kl_i[L],95)),
            kl_neg=float((kl_i[L]<0).mean()),   # KL estimate < 0 -> pure estimator noise
            ))
json.dump(rows,open("/tmp/claude-1001/-home-human-workspaces-reppo-original/52e91891-229f-40e5-b38b-e15e5143a5bd/scratchpad/step7_gate.json","w"),indent=1)
print(f"{'':14s}"+"".join(f"{a:>24s}" for a in ("M=32 (n=8)","M=128 (n=5)","M=512 (n=2)")))
print("-"*88)
for k,lab,dp in [("ret","return",1),("gate","GATE-FIRED frac",4),("lam","lambda_KL",3),
                 ("kl_mean","kl_i mean",4),("kl_med","kl_i median",4),("kl_sd","kl_i SD (across states)",4),
                 ("kl_iqr","kl_i IQR",4),("kl_p05","kl_i p05",4),("kl_p95","kl_i p95",4),
                 ("kl_neg","frac kl_i < 0",4)]:
    c=[]
    for arm in ("M32","M128","M512"):
        v=[r[k] for r in rows if r["arm"]==arm]
        c.append(f"{np.mean(v):.{dp}f} +/- {np.std(v,ddof=1) if len(v)>1 else 0:.{dp}f}" if v else "--")
    print(f"{lab:14s}"+"".join(f"{x:>24s}" for x in c))
print("\nper-seed gate fraction:")
for arm in ("M32","M128","M512"):
    print(f"  {arm:5s} "+"  ".join(f"s{r['seed']}={r['gate']:.3f}(ret {r['ret']:.0f})"
          for r in rows if r["arm"]==arm))
