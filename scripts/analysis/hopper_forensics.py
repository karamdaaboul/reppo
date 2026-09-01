#!/usr/bin/env python
"""Hopper collapsed-vs-successful diagnostic. Read-only; writes only to --outdir."""
import argparse, os, re, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

KV = re.compile(r"([A-Za-z_]\w*)=(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")
IDX = {("A",101):32,("A",102):34,("A",103):36,("A",104):38,("A",105):40,
       ("A",106):42,("A",107):44,("A",108):46,
       ("B",101):33,("B",102):35,("B",103):37,("B",104):39,("B",105):41,
       ("B",106):43,("B",107):45,("B",108):47}

def curve(logdir, jobid, idx):
    rows=[]
    p=f"{logdir}/reppo-ladder_{jobid}_{idx}.out"
    for line in open(p, errors="replace"):
        if " step=" not in line: continue
        d=dict(KV.findall(line))
        if "step" not in d: continue
        rows.append({k: float(d.get(k,"nan")) for k in ("step","ret","sigma","ent","kl","ess")})
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--logdir", default="/hpcwork/qzi10910/reppo_runs/logs")
    ap.add_argument("--jobid", default="3397984"); ap.add_argument("--outdir", required=True)
    a=ap.parse_args(); os.makedirs(a.outdir, exist_ok=True)
    sel=[("A",103,"A s103  success (639.6)","#1b6ca8","-"),
         ("A",102,"A s102  collapsed (1.34)","#c0392b","-"),
         ("B",106,"B s106  success (227.2)","#2e8b57","--"),
         ("B",102,"B s102  collapsed (0.64)","#e67e22","--")]
    fig,ax=plt.subplots(1,3,figsize=(15,4.2))
    for arm,seed,lab,c,ls in sel:
        cv=curve(a.logdir,a.jobid,IDX[(arm,seed)]); x=cv["step"]/1e6
        ax[0].plot(x,cv["ret"],color=c,ls=ls,label=lab,lw=1.8)
        ax[1].plot(x,cv["sigma"],color=c,ls=ls,lw=1.8)
        ax[2].plot(x,cv["ent"],color=c,ls=ls,lw=1.8)
    for i,(t,yl) in enumerate([("HopperHop return","eval episode return"),
                               ("Policy width","train/pi_sigma_mean"),
                               ("Policy entropy","train/entropy")]):
        ax[i].set_title(t); ax[i].set_xlabel("env steps (M)"); ax[i].set_ylabel(yl)
        ax[i].grid(alpha=.3)
    ax[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Hopper: collapsed vs successful seeds (all 50M steps, no runs modified)", y=1.02)
    fig.tight_layout(); fig.savefig(f"{a.outdir}/fig4_hopper_diagnostic.png",dpi=160,bbox_inches="tight")
    print("wrote", f"{a.outdir}/fig4_hopper_diagnostic.png")
    # divergence: first eval where success and collapsed separate by >5% of final success
    for (arm,) in [("A",),("B",)]:
        s_seed, c_seed = (103,102) if arm=="A" else (106,102)
        s=curve(a.logdir,a.jobid,IDX[(arm,s_seed)]); c=curve(a.logdir,a.jobid,IDX[(arm,c_seed)])
        thr=0.05*s["ret"][-1]; n=min(len(s["ret"]),len(c["ret"]))
        di=next((i for i in range(n) if s["ret"][i]-c["ret"][i]>thr), None)
        print(f"arm {arm}: success s{s_seed} final={s['ret'][-1]:.1f} | collapsed s{c_seed} final={c['ret'][-1]:.2f}")
        if di is not None:
            print(f"   diverge at eval {di+1}/{n}, step={s['step'][di]/1e6:.1f}M "
                  f"(success {s['ret'][di]:.1f} vs collapsed {c['ret'][di]:.2f})")
        print(f"   final sigma: success={s['sigma'][-1]:.4f} collapsed={c['sigma'][-1]:.4f}")
        print(f"   final ent  : success={s['ent'][-1]:.3f} collapsed={c['ent'][-1]:.3f}")

if __name__=="__main__": main()
