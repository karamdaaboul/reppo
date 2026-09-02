"""STEPS 4,6,7,8 from stored training metrics. No training, no GPU."""
import numpy as np, glob, os, json, collections

ROOT="/home/human/workspaces/reppo_original"
ARMS=collections.OrderedDict()
# M=32 baseline: run dirs recorded in each export's meta.json
for s in (0,1,2,3,5,6,7,8):
    m=json.load(open(f"{ROOT}/exports/HumanoidRun_weighted_mle_s{s}_final/meta.json"))
    ARMS.setdefault("M32",[]).append((s, m["hydra_run_dir"]))
for d in sorted(glob.glob(f"{ROOT}/outputs/msweep/M*_s2*/")):
    tag=os.path.basename(d.rstrip("/")); M,seed=tag.split("_s")
    ARMS.setdefault(M[0]+M[1:],[]).append((int(seed), d.rstrip("/")))

MSZ={"M32":32,"M128":128,"M512":512}
def load(rd):
    p=os.path.join(rd,"metrics.npz")
    return np.load(p) if os.path.exists(p) else None

def col(z,k):
    if k not in z.files: return None
    a=np.asarray(z[k])
    return a.reshape(a.shape[0],-1)          # (n_eval, rest)

print("="*100)
print("STEP 4/7/8 -- per-arm aggregates over the LAST 5 evals of each completed run")
print("="*100)
LATE=slice(-6,-1)   # evals 15..19 (0-based), avoiding the final export eval
rows=[]
for arm,runs in ARMS.items():
    M=MSZ[arm]
    for seed,rd in runs:
        z=load(rd)
        if z is None: continue
        ret=col(z,'eval/episode_return')[:,0]
        finite=~np.isnan(ret)
        if finite.sum()<16: continue          # NaN'd or in-flight: excluded from aggregates
        NANS=np.full((len(ret),1),np.nan)
        def g(k):
            v=col(z,k)
            return NANS if v is None else v
        ess=g('train/ess')[:,0]; wmax=g('train/w_max')[:,0]; eta=g('train/eta')[:,0]
        qsp=g('train/q_spread')[:,0]; kl=g('train/kl')[:,0]
        lag=g('train/lagrangian')[:,0]
        al=col(z,'train/actor_loss')          # (n_eval, 2048) elementwise
        # gate branch: clipped mode picks kl*lagrangian (~0.1*lam, O(0.1)) when gated,
        # else the WML objective (-sum w logp, O(10)). The two are orders apart.
        gated = (np.abs(al) < 1.0).mean(axis=1)
        rows.append(dict(arm=arm,M=M,seed=seed,
            ret=np.nanmean(ret[LATE]),
            ess=ess[LATE].mean(), ess_frac=(ess[LATE]/M).mean(),
            wmax=wmax[LATE].mean(), wmax_x_M=(wmax[LATE]*M).mean(),
            eta=eta[LATE].mean(), qspread=qsp[LATE].mean(),
            kl=kl[LATE].mean(), lag=lag[LATE].mean(), gated=gated[LATE].mean(),
            ent=g('train/entropy')[LATE].mean(),
            sig=g('train/pi_sigma_mean')[LATE].mean(),
            sig_max=g('train/pi_sigma_max')[LATE].mean(),
            sig_min=g('train/pi_sigma_min')[LATE].mean(),
            absa=g('train/abs_batch_action')[LATE].mean(),
            absp=g('train/abs_pred_action')[LATE].mean(),
            q=g('train/q')[LATE].mean(), tgt=g('train/target_values')[LATE].mean(),
            vloss=g('train/value_loss')[LATE].mean(),
            closs=g('train/critic_update_loss')[LATE].mean(),
            gna=g('train/grad_norm_actor')[LATE].mean(),
            gnc=g('train/grad_norm_critic')[LATE].mean(),
            lt4=g('train/ess_frac_lt4')[LATE].mean(),
            essp5=g('train/ess_p5')[LATE].mean(),
        ))
json.dump(rows, open("/tmp/claude-1001/-home-human-workspaces-reppo-original/52e91891-229f-40e5-b38b-e15e5143a5bd/scratchpad/steps4678.json","w"), indent=1, default=float)

FIELDS=[("ret","return",1),("ess","ESS",1),("ess_frac","ESS/M",3),("wmax","w_max",4),
        ("wmax_x_M","w_max*M",2),("eta","eta",5),("qspread","logit spread",3),
        ("kl","KL(pi_old||pi)",4),("lag","lambda_KL",3),("gated","gate-fired frac",3),
        ("ent","entropy",2),("sig","sigma mean",3),("sig_max","sigma max",3),
        ("absa","|a| batch",3),("absp","|a| pred",3),("q","Q",1),("tgt","target",1),
        ("vloss","value loss",4),("gna","|g| actor",3),("gnc","|g| critic",3),
        ("lt4","frac ESS<4",4),("essp5","ESS p5",2)]
print(f"\n{'metric':16s}" + "".join(f"{a:>26s}" for a in ("M=32 (n=8)","M=128 (n=5)","M=512 (n=2)")))
print("-"*100)
for k,lab,dp in FIELDS:
    cells=[]
    for arm in ("M32","M128","M512"):
        v=[r[k] for r in rows if r["arm"]==arm]
        cells.append(f"{np.nanmean(v):.{dp}f} +/- {np.nanstd(v,ddof=1) if len(v)>1 else 0:.{dp}f}" if v else "--")
    print(f"{lab:16s}" + "".join(f"{c:>26s}" for c in cells))
