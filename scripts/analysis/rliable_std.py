#!/usr/bin/env python3
"""The eps01 vs eps05 return comparison, re-done to the aggregate standard.

Instruments from Agarwal et al. 2021 (arXiv:2108.13264) and Patterson et al. JMLR 25(318):
IQM over runs pooled across tasks, stratified bootstrap (resample runs WITHIN task),
probability of improvement, and a performance profile. Per-task views are reported
alongside, never averaged with the aggregate.

NOT REGISTERED. The normalizer below is a post-hoc choice and must be preregistered before
this goes in the paper. Stated explicitly so it cannot be mistaken for a frozen convention.
"""
import csv, os, sys
import numpy as np

os.chdir("/hpcwork/qzi10910/estep_wt"); sys.path.insert(0, os.getcwd())
R = "/hpcwork/qzi10910/reppo_runs/outputs"
SEEDS, NB, BOOT = list(range(301, 309)), 10000, 20260913
TASKS = {
 "walker": dict(eps05=R + "/faithful_repair/walker_WML32_s%d",
                eps01=R + "/estep_concentration/walker_WML_eps01_s%d"),
 "g1":     dict(eps05=R + "/faithful_repair/g1_WML32_s%d",
                eps01=R + "/estep_concentration/g1_WML_eps01_s%d"),
 "leap":   dict(eps05=R + "/leap_corrected/leap_WML_s%d",
                eps01=R + "/estep_concentration/leap_WML_eps01_s%d"),
}


def sw3(d):
    p = os.path.join(d, "metrics.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p)
    k = "eval/episode_return"
    if k not in z.files:
        return None
    v = np.asarray(z[k]).reshape(-1)
    # score_window3 is frozen as "the final three logged evaluations (18,19,20 of 21)".
    # Those clauses coincide ONLY at exactly 21 evals; a longer schedule would silently
    # yield a mid-training window. Refuse rather than guess.
    return float(np.mean(v[18:21])) if v.size == 21 else None


raw = {}
for t, C in TASKS.items():
    for arm in ("eps05", "eps01"):
        raw[(t, arm)] = [sw3(C[arm] % s) for s in SEEDS]

print("=" * 86)
print("RAW score_window3, every individual run  (Patterson et al.: publish every run)")
print("=" * 86)
for t in TASKS:
    for arm in ("eps05", "eps01"):
        v = raw[(t, arm)]
        print("  %-7s %-6s " % (t, arm) + " ".join("%8.2f" % x if x is not None else "    n/a" for x in v))
    print("  " + "-" * 82)

# per-task min-max over BOTH arms' runs -> [0,1]; stated, post-hoc, not registered
norm = {}
for t in TASKS:
    allv = [x for arm in ("eps05", "eps01") for x in raw[(t, arm)] if x is not None]
    lo, hi = min(allv), max(allv)
    for arm in ("eps05", "eps01"):
        norm[(t, arm)] = np.array([(x - lo) / (hi - lo) for x in raw[(t, arm)] if x is not None])

rng = np.random.default_rng(BOOT)


def iqm(x):
    x = np.sort(np.asarray(x, float))
    n = x.size
    k = int(np.floor(n * 0.25))
    return float(x[k:n - k].mean()) if n - 2 * k > 0 else float(x.mean())


def strat_boot(per_task, stat, nb=NB):
    """Resample runs with replacement independently WITHIN each task."""
    out = np.empty(nb)
    for b in range(nb):
        pooled = []
        for v in per_task:
            i = rng.integers(0, v.size, v.size)
            pooled.append(v[i])
        out[b] = stat(np.concatenate(pooled))
    return out


print("\n" + "=" * 86)
print("AGGREGATE  --  IQM of per-task min-max normalised score, stratified bootstrap")
print("  normaliser: per-task min-max over both arms' 16 runs.  POST-HOC, NOT REGISTERED.")
print("=" * 86)
for arm in ("eps05", "eps01"):
    pt = [norm[(t, arm)] for t in TASKS]
    point = iqm(np.concatenate(pt))
    bs = strat_boot(pt, iqm)
    print("  %-6s IQM %.4f   95%% CI [%.4f, %.4f]" % (arm, point, *np.percentile(bs, [2.5, 97.5])))

pt5 = [norm[(t, "eps05")] for t in TASKS]
pt1 = [norm[(t, "eps01")] for t in TASKS]
d = np.empty(NB)
for b in range(NB):
    a = np.concatenate([v[rng.integers(0, v.size, v.size)] for v in pt1])
    c = np.concatenate([v[rng.integers(0, v.size, v.size)] for v in pt5])
    d[b] = iqm(a) - iqm(c)
pt = iqm(np.concatenate(pt1)) - iqm(np.concatenate(pt5))
lo, hi = np.percentile(d, [2.5, 97.5])
print("  eps01 - eps05  IQM difference %+.4f   95%% CI [%+.4f, %+.4f]   %s"
      % (pt, lo, hi, "EXCLUDES 0" if lo * hi > 0 else "contains 0"))

print("\n" + "=" * 86)
print("PROBABILITY OF IMPROVEMENT  --  P(eps01 run > eps05 run), averaged over tasks")
print("=" * 86)
def poi(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return float((np.greater.outer(a, b) + 0.5 * np.equal.outer(a, b)).mean())
per = []
for t in TASKS:
    p = poi([x for x in raw[(t, "eps01")] if x is not None],
            [x for x in raw[(t, "eps05")] if x is not None])
    per.append(p)
    print("  %-7s P = %.3f" % (t, p))
bs = np.empty(NB)
for b in range(NB):
    ps = []
    for t in TASKS:
        a = np.array([x for x in raw[(t, "eps01")] if x is not None])
        c = np.array([x for x in raw[(t, "eps05")] if x is not None])
        ps.append(poi(a[rng.integers(0, a.size, a.size)], c[rng.integers(0, c.size, c.size)]))
    bs[b] = np.mean(ps)
lo, hi = np.percentile(bs, [2.5, 97.5])
print("  average P = %.3f   95%% CI [%.3f, %.3f]   %s"
      % (np.mean(per), lo, hi, "EXCLUDES 0.5" if (lo - .5) * (hi - .5) > 0 else "contains 0.5"))

print("\n" + "=" * 86)
print("PER-TASK  --  reported alongside, never averaged into the aggregate")
print("=" * 86)
for t in TASKS:
    # Pair ONLY on seeds present in both arms. Filtering each arm independently would
    # mispair every seed after a gap while raising no error.
    both = [(x, y) for x, y in zip(raw[(t, "eps01")], raw[(t, "eps05")])
            if x is not None and y is not None]
    a = np.array([x for x, _ in both]); c = np.array([y for _, y in both])
    per_seed = a - c
    bs = np.array([np.median(per_seed[rng.integers(0, per_seed.size, per_seed.size)]) for _ in range(NB)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    npos = int((per_seed > 0).sum())
    print("  %-7s paired delta median %+9.3f  95%% CI [%+8.3f, %+8.3f]  %s   sign %d/%d positive"
          % (t, np.median(per_seed), lo, hi, "EXCL 0" if lo * hi > 0 else "cont 0",
             npos, per_seed.size))

print("\n" + "=" * 86)
print("PERFORMANCE PROFILE  --  fraction of runs above threshold tau (n=%d per arm)"
      % sum(norm[(t, "eps05")].size for t in TASKS))
print("=" * 86)
allp = {arm: np.concatenate([norm[(t, arm)] for t in TASKS]) for arm in ("eps05", "eps01")}
print("  %-6s " % "tau" + "".join("%7.2f" % x for x in np.arange(0, 1.01, 0.125)))
for arm in ("eps05", "eps01"):
    print("  %-6s " % arm + "".join("%7.2f" % (allp[arm] >= x).mean() for x in np.arange(0, 1.01, 0.125)))

with open("reports/artifacts/rliable_raw_runs.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["task", "arm", "seed", "score_window3"])
    for t in TASKS:
        for arm in ("eps05", "eps01"):
            for s, v in zip(SEEDS, raw[(t, arm)]):
                w.writerow([t, arm, s, v])
print("\n  wrote reports/artifacts/rliable_raw_runs.csv  (all 48 individual runs)")
