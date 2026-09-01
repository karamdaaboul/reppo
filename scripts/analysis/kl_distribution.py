#!/usr/bin/env python
"""Distribution of the logged actor KL (train/kl) per evaluation. Read-only.

train/kl is `kl.mean()` (src/jaxrl/reppo.py:1002), the forward KL
E[log pi_old - log pi_theta]. reverse_kl=false in config, so BOTH arms take the
same `else` branch and log the same field against the same cfg.kl_bound=0.1.

Caveat encoded here: that branch estimates the SAME estimand with a
DIFFERENT number of Monte-Carlo samples per arm --
  n_estep = cfg.estep_num_samples (32) if weighted_mle else 16
so arm A's KL is a 16-sample estimate and arm B's a 32-sample estimate.
"""
import argparse, re, numpy as np
from collections import defaultdict

KV = re.compile(r"([A-Za-z_]\w*)=(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")
TASKS = ["hopper", "walker", "leap", "g1"]
ARMS = ["A", "B"]
SEEDS = list(range(101, 109))
BOUND = 0.1


def decode(i):
    return TASKS[i // 16] if False else ["g1", "leap", "hopper", "walker"][i // 16], ARMS[i % 2], SEEDS[(i // 2) % 8]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", default="/hpcwork/qzi10910/reppo_runs/logs")
    ap.add_argument("--jobid", default="3397984")
    a = ap.parse_args()
    K = defaultdict(list)
    for i in range(64):
        task, arm, seed = decode(i)
        for line in open(f"{a.logdir}/reppo-ladder_{a.jobid}_{i}.out", errors="replace"):
            if " step=" not in line:
                continue
            d = dict(KV.findall(line))
            if "kl" in d:
                K[(task, arm)].append(float(d["kl"]))
    print(f"kl_bound = {BOUND};  'within 10%' = |kl-0.1| <= 0.01  i.e. kl in [0.09, 0.11]\n")
    print(f"{'task':7} {'arm':4} {'n':>5} {'median':>9} {'q25':>9} {'q75':>9} {'IQR':>9} "
          f"{'in[.09,.11]':>12} {'>=0.09':>8} {'>0.10':>8} {'max':>9}")
    for t in TASKS:
        for arm in ARMS:
            v = np.array(K[(t, arm)], float)
            q25, med, q75 = np.percentile(v, [25, 50, 75])
            w = float(np.mean(np.abs(v - BOUND) <= 0.1 * BOUND))
            ge = float(np.mean(v >= 0.09)); gt = float(np.mean(v > BOUND))
            print(f"{t:7} {arm:4} {len(v):5d} {med:9.4f} {q25:9.4f} {q75:9.4f} {q75-q25:9.4f} "
                  f"{w:12.3f} {ge:8.3f} {gt:8.3f} {v.max():9.4f}")


if __name__ == "__main__":
    main()
