"""C-section report for one padding level k: per-seed Humanoid-format tables, paired gap with
percentile bootstrap (10,000 resamples, numpy default_rng(20260830)), sigma real/pad at
p25/p50/final and within-sd all/real-only from probe JSONs if present."""
import re, os, sys, json, glob, numpy as np
S = "/tmp/claude-1001/-home-human-workspaces-safe-rl/2f6f000e-ea36-4609-bb5e-182397ba52ad/scratchpad"
k = int(sys.argv[1]); seeds = [0, 1, 2, 3, 4]
P = re.compile(r"step=(\d+) ret=([-\d.na]+) .*?ent=([-\d.]+) sigma=([\d.]+).*?temp=([\d.]+) kl=([\d.]+) \| ess=([\d.na]+) w_max=[\d.na]+ qspread=[-\d.na]+ eta=([\d.na]+).*?essP=([\d.na]+)/([\d.na]+)/([\d.na]+)/([\d.na]+) lt4=([\d.na]+)")
def load(f):
    r = [P.search(l).groups() for l in open(f) if P.search(l)]; return np.array([[float(x) for x in g] for g in r])
def evs(d): return np.where((np.diff(d[:, 2]) > 5) | (np.diff(d[:, 3]) / d[:-1, 3] > 0.5))[0] + 1
fin = {}
for arm, tag in [("A-frozen", "A"), ("B-frozen", "B")]:
    print(f"\n## WalkerRun pad{k} (d={6+k})  {arm}  alpha=0.01528  final = last eval (52.3M steps, 21 evals)")
    print(f"{'seed':>4}{'final':>8}{'ent_mn':>8}{'sig_mn':>8}{'eta rng':>14}{'ESS mn':>8}{'p5/p25/med/p75':>22}{'ESS<4':>7}{'worst':>8}{'up-ev':>9}{'NaN':>5}")
    for s in seeds:
        f = f"{S}/pad{k}_{tag}_s{s}.log"
        if not os.path.exists(f) or not any("_final | return" in l for l in open(f)): print(f"{s:>4}  (not finished)"); continue
        d = load(f); n = len(d); nan = bool(np.isnan(d[:, 1]).any()); fin[(tag, s)] = d[-1, 1]; e = evs(d)
        B_ = tag == "B"
        print(f"{s:>4}{d[-1,1]:>8.1f}{d[:,2].mean():>8.2f}{d[:,3].mean():>8.3f}{(f'{d[:,7].min():.3f}-{d[:,7].max():.3f}' if B_ else '-'):>14}{(f'{d[:,6].mean():.2f}' if B_ else '-'):>8}"
              f"{(f'{d[:,8].mean():.1f}/{d[:,9].mean():.1f}/{d[:,10].mean():.1f}/{d[:,11].mean():.1f}' if B_ else '-'):>22}{(f'{d[:,12].mean():.3f}' if B_ else '-'):>7}{np.diff(d[:,1]).min():>8.1f}{len(e):>5}/{n-1:<3}{'Y' if nan else 'N':>4}")
common = [s for s in seeds if ("A", s) in fin and ("B", s) in fin]
if common:
    A = np.array([fin[("A", s)] for s in common]); B = np.array([fin[("B", s)] for s in common]); D = A - B
    rng = np.random.default_rng(20260830); bs = np.array([D[rng.integers(0, len(D), len(D))].mean() for _ in range(10000)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    t = D.mean() / (D.std(ddof=1) / np.sqrt(len(D))) if len(D) > 1 else float("nan")
    tw = (A.mean() - B.mean()) / np.sqrt(A.var(ddof=1) / len(A) + B.var(ddof=1) / len(B)) if len(D) > 1 else float("nan")
    print(f"\n## Gap pad{k}: seeds {common}  A {A.mean():.1f}+-{A.std(ddof=1):.1f}  B {B.mean():.1f}+-{B.std(ddof=1):.1f}")
    print(f"   paired Delta = A-B per seed: {np.round(D,1).tolist()}  mean {D.mean():+.1f}")
    print(f"   paired percentile bootstrap 95% CI [{lo:+.1f}, {hi:+.1f}]  (10000 resamples, default_rng(20260830))  paired t={t:+.2f}  Welch t={tw:+.2f}")
    print(f"   CI excludes zero: {lo > 0 or hi < 0}")
# sigma real/pad and within-sd from probe jsons
print(f"\n## sigma real vs padded (median over 2048 visited states x coords), per arm/seed/checkpoint")
for tag, name in [("pathwise_fa", "A-frozen"), ("weighted_mle", "B-frozen")]:
    for s in seeds:
        row = []
        for c in ("p25", "p50", "final"):
            f = f"{S}/probes/WalkerRun_{tag}_pad{k}_s{s}_{c}.sigma.json"
            if os.path.exists(f):
                j = json.load(open(f)); row.append(f"{c}[{(j['frac'] or 1):.2f}]: real {j['sigma_real_median']:.3f} pad {j['sigma_pad_median']:.3f} R={j['ratio_pad_over_real_median']:.2f}")
        if row: print(f"  {name} s{s}: " + " | ".join(row))
    fs = sorted(glob.glob(f"{S}/probes/WalkerRun_{tag}_pad{k}_s*_final.sigma.json"))
    if fs:
        R = [json.load(open(f))["ratio_pad_over_real_median"] for f in fs]; print(f"  {name} final R per seed {np.round(R,2).tolist()}  median {np.median(R):.2f}")
print(f"\n## within-state sd_i(Q) at final: all {6+k} dims vs real-6-only perturbations (padded held at tanh(mu)); 2048 states x 32 actions, PRNGKey(0)")
for tag, name in [("pathwise_fa", "A-frozen"), ("weighted_mle", "B-frozen")]:
    for s in seeds:
        f = f"{S}/probes/WalkerRun_{tag}_pad{k}_s{s}_final.json"
        if os.path.exists(f):
            j = json.load(open(f)); print(f"  {name} s{s}: all-dims {j['within_sd_mean']:.4f}  real-only {j['within_sd_real_only']:.4f}  pad-only {j['within_sd_pad_only']:.4f}  | Q p50 {j['q_p1_p50_p99'][1]:.1f}  sat {j['action_sat']:.3f}")
