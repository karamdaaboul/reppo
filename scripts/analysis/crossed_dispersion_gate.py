"""P1-P3 for the crossed-dispersion WalkerRun reference-law gate.

Reads the cell CSV written by crossed_dispersion.py and applies the
preregistered rules mechanically. Nothing here chooses a convention: which
operators enter I is fixed by prereg sec 4, the bootstrap by P1.

    I = log[ D_PW(Q_PW) / D_PW(Q_WML) ] + log[ D_WML(Q_WML) / D_WML(Q_PW) ]
          bracket 1                        bracket 2

  I_equal-query (PRIMARY)  uses D_{PW-32} and D_{WML-32}
  I_operational (SECONDARY) uses D_{PW-1}  and D_{WML-32}

Usage: crossed_dispersion_gate.py <cells.csv> <out.json>
"""
from __future__ import annotations

import csv, json, sys
import numpy as np

BOOT, RNG_SEED = 10000, 20260904          # P1 / sec 13
VARIANTS = {"equal_query": ("PW-32", "WML-32"), "operational": ("PW-1", "WML-32")}


def load(path):
    D = {}
    for r in csv.DictReader(open(path)):
        D[(r["law"], int(r["seed"]), r["critic"], r["operator"])] = float(r["D"])
    return D


def brackets(D, law, seed, op_pw, op_wml):
    """(bracket 1, bracket 2). Each operator on its own critic over the other."""
    b1 = np.log(D[(law, seed, "PW", op_pw)] / D[(law, seed, "WML", op_pw)])
    b2 = np.log(D[(law, seed, "WML", op_wml)] / D[(law, seed, "PW", op_wml)])
    return b1, b2


def boot_median(vals, rng):
    n = len(vals)
    idx = rng.integers(0, n, size=(BOOT, n))
    return np.median(np.asarray(vals)[idx], axis=1)


def main(cells, out):
    D = load(cells)
    seeds = sorted({k[1] for k in D})
    laws = sorted({k[0] for k in D})
    res = {"seeds": seeds, "laws": laws, "n_boot": BOOT, "rng_seed": RNG_SEED,
           "variants": {}}

    for vname, (op_pw, op_wml) in VARIANTS.items():
        rng = np.random.default_rng(RNG_SEED)          # P1: fixed, one stream
        V = {"operators": {"bracket1": op_pw, "bracket2": op_wml}, "by_law": {}}
        for law in laws:
            b1 = np.array([brackets(D, law, s, op_pw, op_wml)[0] for s in seeds])
            b2 = np.array([brackets(D, law, s, op_pw, op_wml)[1] for s in seeds])
            I = b1 + b2
            bm = boot_median(I, rng)
            lo, hi = np.percentile(bm, [2.5, 97.5])
            V["by_law"][law] = {
                "per_seed": {"I": I.tolist(), "bracket1": b1.tolist(),
                             "bracket2": b2.tolist()},
                "median_I": float(np.median(I)),
                "ci95_median_I": [float(lo), float(hi)],
                "ci_excludes_zero": bool(lo > 0 or hi < 0),
                "sign_I": int(np.sign(np.median(I))),
                "n_pos": {"I": int((I > 0).sum()), "bracket1": int((b1 > 0).sum()),
                          "bracket2": int((b2 > 0).sum())},
                "median_bracket1": float(np.median(b1)),
                "median_bracket2": float(np.median(b2)),
                "ci95_median_b1": [float(x) for x in
                                   np.percentile(boot_median(b1, rng), [2.5, 97.5])],
                "ci95_median_b2": [float(x) for x in
                                   np.percentile(boot_median(b2, rng), [2.5, 97.5])],
            }
        s = {law: V["by_law"][law]["sign_I"] for law in laws}
        V["signs_agree"] = len(set(s.values())) == 1
        V["p1_pass"] = all(V["by_law"][l]["median_I"] > 0
                           and V["by_law"][l]["ci_excludes_zero"] for l in laws)
        V["p3_refuted"] = not V["signs_agree"]
        res["variants"][vname] = V

    # separate deliverable, NOT folded into I (prereg sec 4)
    res["pw32_vs_zo32"] = {
        law: {c: {"PW-32": float(np.median([D[(law, s, c, "PW-32")] for s in seeds])),
                  "ZO-32": float(np.median([D[(law, s, c, "ZO-32")] for s in seeds]))}
              for c in ("PW", "WML")} for law in laws}

    prim = res["variants"]["equal_query"]
    if prim["p3_refuted"]:
        res["classification"] = "REFUTED-AS-CRITIC-SOURCE"
    elif prim["p1_pass"]:
        res["classification"] = "PASS"
    else:
        res["classification"] = "NOT-PASS (signs agree; P1 CI does not exclude zero)"
    json.dump(res, open(out, "w"), indent=1)

    print("classification:", res["classification"])
    for vname, V in res["variants"].items():
        print("\n== %s  (bracket1 %s, bracket2 %s)"
              % (vname, V["operators"]["bracket1"], V["operators"]["bracket2"]))
        for law in laws:
            e = V["by_law"][law]
            print("  law %s  median I %+.4f  CI [%+.4f, %+.4f]  excl0 %s  "
                  "sign %+d  n_pos I %d/%d b1 %d/%d b2 %d/%d"
                  % (law, e["median_I"], *e["ci95_median_I"], e["ci_excludes_zero"],
                     e["sign_I"], e["n_pos"]["I"], len(seeds),
                     e["n_pos"]["bracket1"], len(seeds),
                     e["n_pos"]["bracket2"], len(seeds)))
            print("        bracket1 med %+.4f CI [%+.4f, %+.4f] | "
                  "bracket2 med %+.4f CI [%+.4f, %+.4f]"
                  % (e["median_bracket1"], *e["ci95_median_b1"],
                     e["median_bracket2"], *e["ci95_median_b2"]))
        print("  signs agree across laws:", V["signs_agree"], " P1 pass:", V["p1_pass"])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
