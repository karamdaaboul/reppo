#!/usr/bin/env python3
"""Post-tanh policy-geometry audit. Phases 3, 4, 6. READ-ONLY, offline checkpoints only."""
import csv, hashlib, json, os, sys, time
import numpy as np

REPO = "/hpcwork/qzi10910/estep_wt"
OUT = sys.argv[1]
os.chdir(REPO); sys.path.insert(0, REPO); sys.path.insert(0, OUT)
import jax.numpy as jnp                                        # noqa: E402
from scripts.load_ckpt import load                             # noqa: E402
from posttanh_lib import V, S, posttanh_var, sat_prob, decompose, paired_bootstrap  # noqa: E402

SEEDS = list(range(301, 309))
N_HI, N_LO = 128, 64
TASKS = {"walker": "WalkerRun", "g1": "G1JoystickFlatTerrain", "leap": "LeapCubeRotateZAxis"}
ARMS = {"PW": "pathwise_fa_noent", "WML05": "weighted_mle", "WML01": "weighted_mle_eps01"}
PAIRS = [("WML05", "PW"), ("WML01", "WML05"), ("WML01", "PW")]
RNG_SEED, NBOOT, LEVEL = 20260910, 100000, 95.0

t0 = time.time()
per_seed, quad_rows, decomp_rows, bank_info = [], [], [], []

for tk, env in TASKS.items():
    zb = np.load("reports/artifacts/%s_fixed_state_bank.npz" % tk)
    src_n = np.array([str(x) for x in zb["source"]])
    zn = np.load("reports/artifacts/%s_onpolicy_bank_newarms.npz" % tk)
    src_w = np.array([str(x) for x in zn["source"]])
    banks = {
        "neutral": np.asarray(zb["obs"], np.float64),
        "pw_rollout": np.asarray(zn["obs"], np.float64)[np.char.startswith(src_w, "PW_noent-")],
        "wml_rollout": np.asarray(zb["obs"], np.float64)[np.char.startswith(src_n, "WML-")],
    }
    for bn, ob in banks.items():
        # F6: an empty mask would propagate NaN to the CSV with only a RuntimeWarning
        assert ob.shape[0] > 0, "empty bank %s/%s -- source dtype or prefix mismatch" % (tk, bn)
        assert np.all(np.isfinite(ob)), "non-finite obs in %s/%s" % (tk, bn)
        bank_info.append(dict(task=tk, bank=bn, n_states=int(ob.shape[0]),
                              sha256=hashlib.sha256(np.ascontiguousarray(ob).tobytes()).hexdigest()))
        print("  %-7s %-12s n=%d" % (tk, bn, ob.shape[0]), flush=True)

    params = {}   # (arm, bank, seed) -> (mu, sigma)
    for arm, tag in ARMS.items():
        for s in SEEDS:
            ck = load("exports/%s_%s_s%d_final" % (env, tag, s))
            for bn, ob in banks.items():
                mu, sg = ck.policy_dist(jnp.asarray(ob))
                mu = np.asarray(mu, np.float64); sg = np.asarray(sg, np.float64)
                params[(arm, bn, s)] = (mu, sg)
                v_hi, ncl, mxcl = V(mu, sg, N_HI)
                v_lo, _, _ = V(mu, sg, N_LO)
                sat = S(mu, sg, 0.95)
                assert np.isfinite(v_hi) and np.isfinite(v_lo) and np.isfinite(sat), \
                    "non-finite metric at %s/%s/%s/s%d" % (tk, arm, bn, s)
                d = abs(v_hi - v_lo); rel = d / max(abs(v_hi), 1e-300)
                per_seed.append(dict(task=tk, arm=arm, bank=bn, seed=s,
                                     V=v_hi, V_lo=v_lo, sat95=sat,
                                     n_clamped=ncl, max_clamp=mxcl,
                                     pre_tanh_sigma_med=float(np.median(sg)),
                                     pre_tanh_mu_absmed=float(np.median(np.abs(mu)))))
                quad_rows.append(dict(task=tk, arm=arm, bank=bn, seed=s, V_N128=v_hi,
                                      V_N64=v_lo, abs_diff=d, rel_diff=rel,
                                      flagged=bool(np.isfinite(rel) and np.isfinite(d)
                                                   and rel > 1e-3 and d > 1e-8)))
        print("    %-7s %-6s done  (%.0fs)" % (tk, arm, time.time() - t0), flush=True)

    for A, B in PAIRS:                     # decomposition on the PRIMARY bank
        for s in SEEDS:
            muA, sgA = params[(A, "neutral", s)]
            muB, sgB = params[(B, "neutral", s)]
            for mname, F in (("variance", lambda m, g: V(m, g, N_HI)[0]),
                             ("sat95", lambda m, g: S(m, g, 0.95))):
                r = decompose(muA, sgA, muB, sgB, F)
                decomp_rows.append(dict(task=tk, pair="%s_vs_%s" % (A, B), metric=mname,
                                        seed=s, **{k: float(v) for k, v in r.items()}))
    print("    %-7s decomposition done (%.0fs)" % (tk, time.time() - t0), flush=True)
    del params

# ---------------- Phase 6 statistics ----------------
summary = []
def row_rng(*key):
    """F8: deterministic per-row stream so intervals do not depend on loop order."""
    h = int(hashlib.sha256("|".join(map(str, key)).encode()).hexdigest()[:12], 16)
    return np.random.default_rng([RNG_SEED, h])
for tk in TASKS:
    for bn in ("neutral", "pw_rollout", "wml_rollout"):
        for A, B in PAIRS:
            for metric in ("V", "sat95"):
                g = lambda arm, s: [r for r in per_seed if r["task"] == tk and r["arm"] == arm
                                    and r["bank"] == bn and r["seed"] == s][0][metric]
                seeds_ok = [s for s in SEEDS
                            if any(r["task"] == tk and r["arm"] == A and r["bank"] == bn and r["seed"] == s for r in per_seed)
                            and any(r["task"] == tk and r["arm"] == B and r["bank"] == bn and r["seed"] == s for r in per_seed)]
                a = np.array([g(A, s) for s in seeds_ok]); b = np.array([g(B, s) for s in seeds_ok])
                if metric == "V":
                    lr = np.log(a / b)
                    m, lo, hi = paired_bootstrap(lr, row_rng(tk, bn, A, B, "mean"), NBOOT, LEVEL, np.mean)
                    md, mdlo, mdhi = paired_bootstrap(lr, row_rng(tk, bn, A, B, "median"), NBOOT, LEVEL, np.median)
                    summary.append(dict(task=tk, bank=bn, contrast="%s/%s" % (A, B), metric="V_ratio",
                                        n=len(seeds_ok), est=np.exp(m), lo=np.exp(lo), hi=np.exp(hi),
                                        excludes_null=bool((np.exp(lo)-1)*(np.exp(hi)-1) > 0),
                                        est_median_conv=np.exp(md), lo_median_conv=np.exp(mdlo),
                                        hi_median_conv=np.exp(mdhi),
                                        seed_values=";".join("%.6g" % x for x in (a / b))))
                else:
                    d = a - b
                    m, lo, hi = paired_bootstrap(d, row_rng(tk, bn, A, B, "sat"), NBOOT, LEVEL, np.mean)
                    summary.append(dict(task=tk, bank=bn, contrast="%s-%s" % (A, B), metric="sat95_diff",
                                        n=len(seeds_ok), est=m, lo=lo, hi=hi,
                                        excludes_null=bool(lo * hi > 0),
                                        est_median_conv="", lo_median_conv="", hi_median_conv="",
                                        seed_values=";".join("%.6g" % x for x in d))) 

def dump(name, rows):
    p = os.path.join(OUT, name)
    with open(p, "w", newline="") as fh:
        k = sorted(set().union(*[set(r) for r in rows]))
        w = csv.DictWriter(fh, fieldnames=k); w.writeheader(); w.writerows(rows)
    print("  wrote %s (%d rows)" % (p, len(rows)))

dump("per_seed_results.csv", per_seed)
dump("task_summary.csv", summary)
dump("quadrature_convergence.csv", quad_rows)
dump("decomposition.csv", decomp_rows)
json.dump(dict(rng_seed=RNG_SEED, nboot=NBOOT, level=LEVEL, N_hi=N_HI, N_lo=N_LO,
               seeds=SEEDS, tasks=list(TASKS), arms=ARMS, pairs=PAIRS, banks=bank_info),
          open(os.path.join(OUT, "analysis_config.json"), "w"), indent=1)
print("  total clamped: %d  max clamp magnitude: %.3e"
      % (sum(r["n_clamped"] for r in per_seed), max(r["max_clamp"] for r in per_seed)))
print("  flagged quadrature cells: %d/%d" % (sum(r["flagged"] for r in quad_rows), len(quad_rows)))
print("  max |decomp resid|: %.3e" % max(abs(r["resid"]) for r in decomp_rows))
print("  elapsed %.0fs" % (time.time() - t0))
