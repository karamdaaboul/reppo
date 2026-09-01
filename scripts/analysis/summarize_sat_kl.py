#!/usr/bin/env python
"""Aggregate audit_sat_kl.py JSONs per task and arm. Read-only."""
import argparse, glob, json, re, numpy as np
from collections import defaultdict

TASK = {"G1JoystickFlatTerrain": "g1", "LeapCubeRotateZAxis": "leap",
        "HopperHop": "hopper", "WalkerRun": "walker"}
ORDER = ["hopper", "walker", "leap", "g1"]
D = {"hopper": 4, "walker": 6, "leap": 16, "g1": 29}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); a = ap.parse_args()
    G = defaultdict(list)
    for p in sorted(glob.glob(f"{a.dir}/*.json")):
        d = json.load(open(p))
        m = re.match(r"([A-Za-z0-9]+)_(pathwise_fa|weighted_mle)_s(\d+)_final", d["ckpt"])
        if not m or m.group(1) not in TASK:
            continue
        if not (101 <= int(m.group(3)) <= 108):
            continue
        G[(TASK[m.group(1)], "A" if m.group(2) == "pathwise_fa" else "B")].append(d)
    print(f"tau = 1 - 1e-4 = 0.9999 (policy log-prob clamp, reppo.py:712); "
          f"env ClipAction bound is 0.999 and is NOT used here\n")
    print(f"{'task':7} {'d':>3} {'arm':4} {'n':>3} {'M':>3} {'frac coords':>12} {'frac vec any':>13} "
          f"{'frac vec all':>13}")
    for t in ORDER:
        for arm in "AB":
            v = G[(t, arm)]
            if not v: continue
            fc = np.array([x["frac_coords_saturated"] for x in v])
            fa = np.array([x["frac_vectors_any_saturated"] for x in v])
            fl = np.array([x["frac_vectors_all_saturated"] for x in v])
            print(f"{t:7} {D[t]:3d} {arm:4} {len(v):3d} {v[0]['M']:3d} {fc.mean():12.4f} "
                  f"{fa.mean():13.4f} {fl.mean():13.4f}")
    print(f"\n{'task':7} {'arm':4} {'KL (a) code':>12} {'KL (b) fixed':>13} {'difference':>11} "
          f"{'diff p90':>10} {'diff max':>10} {'logged kl med':>14} {'artifact %':>11}")
    LOGGED = {("hopper","A"):0.0969,("hopper","B"):0.1017,("walker","A"):0.1054,("walker","B"):0.1038,
              ("leap","A"):0.1089,("leap","B"):0.1081,("g1","A"):0.1000,("g1","B"):0.1047}
    for t in ORDER:
        for arm in "AB":
            v = G[(t, arm)]
            if not v: continue
            ka = np.array([x["kl_code_path"] for x in v]).mean()
            kb = np.array([x["kl_fixed_path"] for x in v]).mean()
            p90 = np.array([x["kl_diff_p90"] for x in v]).mean()
            mx = np.array([x["kl_diff_max"] for x in v]).max()
            lg = LOGGED[(t, arm)]
            print(f"{t:7} {arm:4} {ka:12.6f} {kb:13.6f} {ka-kb:11.6f} {p90:10.6f} {mx:10.4f} "
                  f"{lg:14.4f} {100*(ka-kb)/lg:10.1f}%")


if __name__ == "__main__":
    main()
