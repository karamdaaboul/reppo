"""STEP 6: trajectories + temporal ordering. Descriptive only; no causal claim."""
import numpy as np, glob, os, json, collections
ROOT="/home/human/workspaces/reppo_original"
ARMS=collections.OrderedDict()
for s in (0,1,2,3,5,6,7,8):
    m=json.load(open(f"{ROOT}/exports/HumanoidRun_weighted_mle_s{s}_final/meta.json"))
    ARMS.setdefault("M32",[]).append((s,m["hydra_run_dir"]))
for d in sorted(glob.glob(f"{ROOT}/outputs/msweep/M*_s2*/")):
    t=os.path.basename(d.rstrip("/")); M,sd=t.split("_s")
    ARMS.setdefault(M,[]).append((int(sd),d.rstrip("/")))
MSZ={"M32":32,"M128":128,"M512":512}
def series(rd):
    p=os.path.join(rd,"metrics.npz")
    if not os.path.exists(p): return None
    z=np.load(p); T=len(z['time_step'])
    g=lambda k: np.asarray(z[k]).reshape(T,-1).mean(1) if k in z.files else np.full(T,np.nan)
    ret=np.asarray(z['eval/episode_return']).reshape(T,-1)[:,0]
    if (~np.isnan(ret)).sum()<16: return None
    return dict(ret=ret, sig=g('train/pi_sigma_mean'), ent=g('train/entropy'),
                ess=g('train/ess'), q=g('train/q'), absa=g('train/abs_batch_action'),
                eta=g('train/eta'), lam=g('train/lagrangian'))
print("Per-arm MEAN trajectory over 21 evals (0 -> 52.3M steps)\n")
for key,lab,dp in [("ret","eval return",0),("sig","pi sigma mean",3),
                   ("ent","entropy",1),("q","Q",1),("absa","|action|",3),("eta","eta",4)]:
    print(f"--- {lab} ---")
    for arm in ("M32","M128","M512"):
        ss=[series(rd) for _,rd in ARMS[arm]]; ss=[x for x in ss if x]
        if not ss: continue
        A=np.nanmean(np.stack([x[key] for x in ss]),0)
        print(f"  {arm:5s} "+" ".join(f"{v:{6}.{dp}f}" for v in A))
    print()
# temporal ordering: index of first eval where sigma has fallen 25% from its own max,
# vs index where return first exceeds 25% of its own final-arm max
print("Temporal ordering per seed  (eval index, 0-20; 'sigma25' = first eval where")
print("pi_sigma has dropped 25% below its own running max; 'ret25' = first eval where")
print("return reaches 25% of that seed's own maximum)\n")
for arm in ("M32","M128","M512"):
    for sd,rd in ARMS[arm]:
        s=series(rd)
        if not s: continue
        sig,ret=s["sig"],s["ret"]
        rm=np.fmax.accumulate(np.nan_to_num(sig))
        i_sig=next((i for i in range(len(sig)) if sig[i]<0.75*rm[i]), None)
        rmax=np.nanmax(ret)
        i_ret=next((i for i in range(len(ret)) if ret[i]>=0.25*rmax), None)
        print(f"  {arm:5s} s{sd:<4d} sigma25={str(i_sig):>4s}  ret25={str(i_ret):>4s}  "
              f"final_ret={ret[-1]:7.1f}  sigma_final={sig[-1]:.3f}  sigma_max={np.nanmax(sig):.3f}")
