#!/usr/bin/env python3
"""Walker entropy factorial: read-only analysis of all four cells.

ANALYSIS_ONLY. Loads exported checkpoints and logged metrics. Starts no training,
modifies no experimental source.

Cells (these names are used everywhere; "PW-H"/"WML-H" are deliberately avoided
because the same suffix would mean opposite things in the two arms):
    PW_ent     corrected Walker PW-1        entropy ON   tag pathwise_fa
    PW_noent   new                          entropy OFF  tag pathwise_fa_noent
    WML_ent    new                          entropy ON   tag weighted_mle_ent
    WML_noent  corrected Walker WML-32      entropy OFF  tag weighted_mle
"""
import csv, hashlib, json, os, sys
import numpy as np
import jax, jax.numpy as jnp
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.getcwd())
from scripts.load_ckpt import load

ART   = "reports/artifacts"
TASK  = os.environ.get("EF_TASK", "walker")
_T = {
    "walker": dict(env="WalkerRun", d=6, figd="figs_entropy_factorial",
                   bank="walker_fixed_state_bank.npz",
                   sha="8adfeb0bf70bddcdbd64a84b972b4dbd62617c64e1c948589a7adc30cf64aa21",
                   pw_run="/walker_PW1_s%d", wml_run="/walker_WML32_s%d",
                   pwn_run="/walker_PW-H_s%d", wmle_run="/walker_WML+H_s%d",
                   ef="/hpcwork/qzi10910/reppo_runs/outputs/entropy_factorial"),
    "g1": dict(env="G1JoystickFlatTerrain", d=29, figd="figs_g1_entropy_factorial",
               bank="g1_fixed_state_bank.npz",
               sha="cf6f7880b3a5b59433727f7a245b5ce97749096981f03233fd434ab3371935e9",
               pw_run="/g1_PW1_s%d", wml_run="/g1_WML32_s%d",
               pwn_run="/g1_PW_noent_s%d", wmle_run="/g1_WML_ent_s%d",
               ef="/hpcwork/qzi10910/reppo_runs/outputs/g1_entropy_factorial"),
}[TASK]
FIGD  = os.path.join(ART, _T["figd"])
BANK  = os.path.join(ART, _T["bank"])
BANK_SHA = _T["sha"]
SEEDS = list(range(301, 309))
BOOT_N, RNG_SEED = 10000, 20260902
FR = "/rwthfs/rz/cluster/hpcwork/qzi10910/reppo_runs/outputs/faithful_repair"
EF = "/hpcwork/qzi10910/reppo_runs/outputs/entropy_factorial"
EF = _T["ef"]
ENVN = _T["env"]
CELLS = {
    "PW_ent":    dict(tag="pathwise_fa",       run=FR + _T["pw_run"],   arm="PW",  ent=True),
    "PW_noent":  dict(tag="pathwise_fa_noent", run=EF + _T["pwn_run"],  arm="PW",  ent=False),
    "WML_ent":   dict(tag="weighted_mle_ent",  run=EF + _T["wmle_run"], arm="WML", ent=True),
    "WML_noent": dict(tag="weighted_mle",      run=FR + _T["wml_run"],  arm="WML", ent=False),
}
ORDER = ["PW_ent", "PW_noent", "WML_ent", "WML_noent"]
T95, T99 = float(np.arctanh(0.95)), float(np.arctanh(0.99))
os.makedirs(FIGD, exist_ok=True)
out = {}

def sat(mu, sg, c):
    """Exact P(|tanh Y| > t) for Y ~ N(mu, sg), c = atanh(t). Per COORDINATE."""
    return norm.cdf((-c - mu) / sg) + 1.0 - norm.cdf((c - mu) / sg)

def boot_ci(vals, stat, n=BOOT_N, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    v = np.asarray(vals, float); k = len(v)
    bs = [stat(v[rng.integers(0, k, k)]) for _ in range(n)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

def boot_interaction(d_ent, d_noent, n=BOOT_N, seed=RNG_SEED):
    """Resample seeds once per replicate; form both deltas and their difference inside."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(d_ent, float), np.asarray(d_noent, float)
    k = len(a); bs = []
    for _ in range(n):
        i = rng.integers(0, k, k)
        bs.append(np.median(a[i]) - np.median(b[i]))
    return float(np.median(bs)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

def logged(cell, seed):
    p = (CELLS[cell]["run"] % seed) + "/metrics.npz"
    if not os.path.exists(p): return {}
    z = np.load(p)
    g = lambda k: float(np.asarray(z[k]).reshape(-1)[-1]) if k in z.files else float("nan")
    return dict(gate=g("train/fr_gate_operator"), kl=g("train/kl"),
                pi_sigma_mean=g("train/pi_sigma_mean"), eta=g("train/eta"),
                ess=g("train/ess"), entropy=g("train/entropy"), temp=g("train/temp"))

def main():
    h = hashlib.sha256(open(BANK, "rb").read()).hexdigest()
    print("bank sha256 %s  %s" % (h, "VERIFIED" if h == BANK_SHA else "MISMATCH"))
    assert h == BANK_SHA, "bank hash mismatch, refusing to proceed"
    z = np.load(BANK)
    bank = jnp.asarray(z["obs" if "obs" in z.files else ("states" if "states" in z.files else z.files[0])])
    half = bank.shape[0] // 2
    SUBSETS = {"full": slice(None), "pw_half": slice(0, half), "wml_half": slice(half, None)}
    print("bank %d states; halves %d PW-derived / %d WML-derived\n" % (bank.shape[0], half, half))

    rows_coord, rows_cell = [], []
    ret, width, diag = {}, {}, {}
    for cell in ORDER:
        c = CELLS[cell]
        ret[cell], width[cell], diag[cell] = {}, {}, {}
        for sd in SEEDS:
            d = "exports/" + ENVN + "_%s_s%d_final" % (c["tag"], sd)
            ck = load(d)
            meta = ck.meta
            curve = np.asarray(meta["eval_return_curve"], float)
            assert curve.size == 21, "%s curve has %d entries" % (d, curve.size)
            ret[cell][sd] = float(curve[[18, 19, 20]].mean())     # frozen scalar
            per = {}
            for sub, sl in SUBSETS.items():
                mu, sg = ck.policy_dist(bank[sl])
                mu_n, sg_n = np.asarray(mu), np.asarray(sg)
                s95 = sat(mu_n, sg_n, T95); s99 = sat(mu_n, sg_n, T99)
                per[sub] = dict(median=float(np.median(sg_n)), mean=float(sg_n.mean()),
                                p95=float(np.percentile(sg_n, 95)), max=float(sg_n.max()),
                                sat95=float(s95.mean()), sat99=float(s99.mean()))
                if sub == "full":
                    for j in range(sg_n.shape[1]):
                        rows_coord.append(dict(cell=cell, seed=sd, coord=j,
                            median_sigma=float(np.median(sg_n[:, j])),
                            mean_sigma=float(sg_n[:, j].mean()),
                            p95_sigma=float(np.percentile(sg_n[:, j], 95)),
                            max_sigma=float(sg_n[:, j].max()),
                            sat95=float(s95[:, j].mean()), sat99=float(s99[:, j].mean())))
            width[cell][sd] = per
            lg = logged(cell, sd)
            diag[cell][sd] = dict(alpha_kl=float(meta.get("alpha_kl", float("nan"))),
                                  alpha_ent=float(meta.get("alpha_entropy", float("nan"))),
                                  ess_final=float(meta.get("ess_final", float("nan"))), **lg)
            rows_cell.append(dict(cell=cell, seed=sd, score_window3=ret[cell][sd],
                **{("%s_%s" % (sub, k)): v for sub, dd in per.items() for k, v in dd.items()},
                **diag[cell][sd]))

    # ---------------- 1 + 3: returns
    print("=" * 78 + "\n1. FROZEN RETURN SCALAR  score_window3 = mean of evals 18,19,20 of 21\n" + "=" * 78)
    print("  seed |    PW_ent   PW_noent    WML_ent  WML_noent")
    for sd in SEEDS:
        print("   %d | %9.3f %10.3f %10.3f %10.3f" % (sd, *[ret[c][sd] for c in ORDER]))
    print("  mean | %9.3f %10.3f %10.3f %10.3f" % tuple(np.mean([ret[c][s] for s in SEEDS]) for c in ORDER))

    d_ent = [ret["PW_ent"][s] - ret["WML_ent"][s] for s in SEEDS]
    d_noent = [ret["PW_noent"][s] - ret["WML_noent"][s] for s in SEEDS]
    print("\n" + "=" * 78 + "\n3. PAIRED RETURN EFFECTS  (PW minus WML, positive = pathwise higher)\n" + "=" * 78)
    print("  seed |  Delta_ent  Delta_noent")
    for i, sd in enumerate(SEEDS):
        print("   %d | %10.3f %12.3f" % (sd, d_ent[i], d_noent[i]))
    for nm, dv in (("Delta_ent", d_ent), ("Delta_noent", d_noent)):
        lo, hi = boot_ci(dv, np.median)
        print("  %-12s paired median %+9.3f  95%% CI [%+9.3f, %+9.3f]  %d/8 positive"
              % (nm, np.median(dv), lo, hi, sum(1 for x in dv if x > 0)))

    # ---------------- 2 + 4: width
    print("\n" + "=" * 78 + "\n2. FIXED-BANK WIDTH AND SATURATION\n" + "=" * 78)
    print("  Saturation: P(|a| > t) per ACTION COORDINATE, computed exactly from the")
    print("  pre-tanh Gaussian (not sampled), averaged over bank states and coordinates.")
    print("  State distribution: the frozen neutral bank. Thresholds t = 0.95, 0.99.\n")
    print("  cell        subset   | med sigma    mean      p95       max |  sat95   sat99")
    for cell in ORDER:
        for sub in ("full", "pw_half", "wml_half"):
            v = lambda k: np.median([width[cell][s][sub][k] for s in SEEDS])
            print("  %-11s %-8s | %9.4f %9.3f %8.3f %9.1f | %.4f  %.4f"
                  % (cell, sub, v("median"), v("mean"), v("p95"), v("max"), v("sat95"), v("sat99")))
        print()

    print("=" * 78 + "\n4. PAIRED WIDTH EFFECTS\n" + "=" * 78)
    print("  RATIO CONVENTION: median over seeds of per-seed log ratios, exponentiated.")
    print("  (Not a mean of ratios.)\n")
    lr = {}
    for sub in ("full", "pw_half", "wml_half"):
        lr_ent = [np.log(width["WML_ent"][s][sub]["median"]) - np.log(width["PW_ent"][s][sub]["median"]) for s in SEEDS]
        lr_no = [np.log(width["WML_noent"][s][sub]["median"]) - np.log(width["PW_noent"][s][sub]["median"]) for s in SEEDS]
        lr[sub] = (lr_ent, lr_no)
        for nm, v in (("log_ratio_ent", lr_ent), ("log_ratio_noent", lr_no)):
            lo, hi = boot_ci(v, np.median)
            print("  %-8s %-16s median log %+7.4f -> %8.3fx   95%% CI [%.3fx, %.3fx]"
                  % (sub, nm, np.median(v), np.exp(np.median(v)), np.exp(lo), np.exp(hi)))
        print()

    # ---------------- 5: interactions
    print("=" * 78 + "\n5. FACTORIAL INTERACTIONS  (descriptive; no test was preregistered)\n" + "=" * 78)
    m, lo, hi = boot_interaction(d_ent, d_noent)
    print("  I_return = Delta_ent - Delta_noent")
    print("    point %+9.3f   95%% CI [%+9.3f, %+9.3f]" % (np.median(d_ent) - np.median(d_noent), lo, hi))
    for sub in ("full", "pw_half", "wml_half"):
        a, b = lr[sub]
        m2, lo2, hi2 = boot_interaction(a, b)
        print("  I_width (%s) = log_ratio_ent - log_ratio_noent" % sub)
        print("    point %+7.4f -> %6.3fx   95%% CI [%.3fx, %.3fx]"
              % (np.median(a) - np.median(b), np.exp(np.median(a) - np.median(b)), np.exp(lo2), np.exp(hi2)))
    print("\n  Four cell means, read the 2x2 directly:")
    print("  outcome            |    PW_ent   PW_noent    WML_ent  WML_noent")
    print("  score_window3      | %9.3f %10.3f %10.3f %10.3f"
          % tuple(np.mean([ret[c][s] for s in SEEDS]) for c in ORDER))
    print("  bank median sigma  | %9.4f %10.4f %10.4f %10.4f"
          % tuple(np.mean([width[c][s]["full"]["median"] for s in SEEDS]) for c in ORDER))
    print("  sat95              | %9.4f %10.4f %10.4f %10.4f"
          % tuple(np.mean([width[c][s]["full"]["sat95"] for s in SEEDS]) for c in ORDER))

    # ---------------- 6: diagnostics
    print("\n" + "=" * 78 + "\n6. DIAGNOSTICS\n" + "=" * 78)
    print("  alpha_kl is actor.lagrangian() (scripts/export_ckpt.py:204). The audit's")
    print("  lambda_eff is that same KL multiplier, so they are the SAME field.\n")
    print("  cell        | alpha_kl(final)      | gate fire | logged sigma | bank median | ratio | eta   ESS")
    for cell in ORDER:
        ak = [diag[cell][s]["alpha_kl"] for s in SEEDS]
        gt = [diag[cell][s]["gate"] for s in SEEDS]
        ls = [diag[cell][s]["pi_sigma_mean"] for s in SEEDS]
        bm = [width[cell][s]["full"]["median"] for s in SEEDS]
        et = [diag[cell][s]["eta"] for s in SEEDS]
        es = [diag[cell][s]["ess"] for s in SEEDS]
        print("  %-11s | %.4f  [%.4f,%.4f] |   %.3f   | %11.4f | %11.4f | %5.2f | %5.3f %5.2f"
              % (cell, np.median(ak), min(ak), max(ak), np.median(gt), np.median(ls),
                 np.median(bm), np.median(bm) / np.median(ls), np.median(et), np.median(es)))

    # ---------------- 7: integrity
    print("\n" + "=" * 78 + "\n7. INTEGRITY\n" + "=" * 78)
    man = {}
    with open(os.path.join(ART, "exports_manifest.csv")) as f:
        for r in csv.DictReader(f): man[r["dir"]] = r
    def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.exists(p) else None
    nb = 0; okb = 0
    for cell in ("PW_ent", "WML_noent"):
        for sd in SEEDS:
            d = "exports/" + ENVN + "_%s_s%d_final" % (CELLS[cell]["tag"], sd)
            if d in man:
                nb += 1
                okb += int(sha(os.path.join(d, "actor.npz")) == man[d]["sha256_actor"])
    print("  BASELINE_MANIFEST_MATCH   = %s  (%d/%d)" % ("PASS" if okb == nb == 16 else "FAIL", okb, nb))
    nn = 0
    for cell in ("PW_noent", "WML_ent"):
        for sd in SEEDS:
            d = "exports/" + ENVN + "_%s_s%d_final" % (CELLS[cell]["tag"], sd)
            nn += int(all(os.path.exists(os.path.join(d, f)) for f in
                          ("actor.npz", "critic.npz", "meta.json", "normalizer.npz")))
    print("  NEW_EXPORTS_COMPLETE      = %s  (%d/16)" % ("PASS" if nn == 16 else "FAIL", nn))
    tags = set(CELLS[c]["tag"] for c in ORDER)
    print("  NEW_TAGS_NON_COLLIDING    = %s  (%d distinct tags)"
          % ("PASS" if len(tags) == 4 else "FAIL", len(tags)))

    # ---------------- CSVs
    with open(os.path.join(ART, "%s_entropy_factorial_percoord.csv" % TASK), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_coord[0])); w.writeheader(); w.writerows(rows_coord)
    with open(os.path.join(ART, "%s_entropy_factorial_cells.csv" % TASK), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_cell[0])); w.writeheader(); w.writerows(rows_cell)
    out.update(returns=ret, width={c: {s: width[c][s] for s in SEEDS} for c in ORDER},
               diag=diag, d_ent=d_ent, d_noent=d_noent,
               I_return=dict(point=np.median(d_ent) - np.median(d_noent), lo=lo, hi=hi))
    with open(os.path.join(ART, "%s_entropy_factorial.json" % TASK), "w") as f:
        json.dump(out, f, indent=1, default=float)

    # ---------------- figures
    cols = {"PW_ent": "#1f77b4", "PW_noent": "#7fb3d5", "WML_ent": "#d62728", "WML_noent": "#f1948a"}
    x = np.arange(len(SEEDS))
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for c in ORDER:
        ax.plot(x, [ret[c][s] for s in SEEDS], "o", color=cols[c], label=c, ms=7)
    ax.set_xticks(x); ax.set_xticklabels(SEEDS, fontsize=8)
    ax.set_xlabel("seed"); ax.set_ylabel("score_window3\n(mean of final 3 of 21 evals)", fontsize=9)
    ax.set_title("%s entropy factorial: return, paired seeds" % ENVN, fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout()
    for e in ("pdf", "png"): fig.savefig(os.path.join(FIGD, "fig_ef_return." + e), dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for c in ORDER:
        ax.plot(x, [width[c][s]["full"]["median"] for s in SEEDS], "o", color=cols[c], label=c, ms=7)
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(SEEDS, fontsize=8)
    ax.set_xlabel("seed"); ax.set_ylabel("median pre-tanh sigma\n(frozen neutral bank, full)", fontsize=9)
    ax.set_title("%s entropy factorial: policy width, paired seeds" % ENVN, fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=.3, which="both")
    fig.tight_layout()
    for e in ("pdf", "png"): fig.savefig(os.path.join(FIGD, "fig_ef_width." + e), dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for ax_, key, lab in ((axes[0], "sat95", "P(|a| > 0.95)"), (axes[1], "sat99", "P(|a| > 0.99)")):
        for c in ORDER:
            ax_.plot(x, [width[c][s]["full"][key] for s in SEEDS], "o", color=cols[c], label=c, ms=7)
        ax_.set_xticks(x); ax_.set_xticklabels(SEEDS, fontsize=8)
        ax_.set_ylim(0, 1); ax_.set_xlabel("seed"); ax_.set_ylabel(lab, fontsize=9)
        ax_.grid(alpha=.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("%s entropy factorial: action saturation, per coordinate, frozen bank" % ENVN, fontsize=10)
    fig.tight_layout()
    for e in ("pdf", "png"): fig.savefig(os.path.join(FIGD, "fig_ef_saturation." + e), dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("\n  wrote CSVs, JSON and figures under %s" % FIGD)

if __name__ == "__main__":
    main()
