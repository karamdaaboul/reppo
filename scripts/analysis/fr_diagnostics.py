"""Preregistered sampled-vs-analytic KL and gate diagnostics (prereg section 5).
Diagnostic only: none of this entered training, the gate, the dual, or any exclusion."""
import json, numpy as np, csv, collections
led = [json.loads(l) for l in open("ledger/runs_faithful_repair.jsonl")]
ROOT = "/hpcwork/qzi10910/reppo_runs"
rows = []
for r in led:
    z = np.load("%s/%s/metrics.npz" % (ROOT, r["output_path"]))
    g = lambda k: np.asarray(z[k]).reshape(np.asarray(z[k]).shape[0], -1)[:, 0]
    ks, ka = g("train/fr_kl_q50"), g("train/fr_kl_analytic_med")
    d = ks - ka
    rows.append(dict(task=r["task_key"], arm="PW-1" if "PW" in r["arm"] else "WML-32",
                     seed=r["seed"],
                     kl_sampled=np.nanmean(ks), kl_analytic=np.nanmean(ka),
                     bias=np.nanmean(d), rmse=np.sqrt(np.nanmean(d**2)),
                     corr=np.corrcoef(ks[np.isfinite(ks) & np.isfinite(ka)],
                                      ka[np.isfinite(ks) & np.isfinite(ka)])[0, 1],
                     gate_op=np.nanmean(g("train/fr_gate_operator")),
                     gate_kl=np.nanmean(g("train/fr_gate_kl_only")),
                     kl_q90=np.nanmean(g("train/fr_kl_q90")),
                     lag=np.nanmean(g("train/fr_lag_eff")),
                     sigma=np.nanmean(g("train/fr_sigma_mean")),
                     sat=np.nanmean(g("train/fr_action_sat")),
                     ess=np.nanmean(g("train/ess"))))
with open("reports/artifacts/corrected_diagnostics.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
    for r in rows: w.writerow(r)

print("=== sampled-vs-analytic KL and GATE, medians over 8 seeds ===")
print("%-7s %-7s %9s %9s %8s %8s %6s | %9s %9s | %7s %7s %6s"
      % ("task","arm","kl_samp","kl_anal","bias","rmse","corr",
         "gate_op","gate_klonly","lag","sigma","ESS"))
agg = {}
for (t, a), grp in collections.OrderedDict(
        (k, [r for r in rows if (r["task"], r["arm"]) == k])
        for k in dict.fromkeys((r["task"], r["arm"]) for r in rows)).items():
    m = lambda k: float(np.median([r[k] for r in grp]))
    agg[(t, a)] = {k: m(k) for k in ("kl_sampled","kl_analytic","bias","rmse","corr",
                                     "gate_op","gate_kl","lag","sigma","ess")}
    print("%-7s %-7s %9.4f %9.4f %+8.4f %8.4f %6.3f | %9.4f %9.4f | %7.4f %7.4f %6.2f"
          % (t, a, m("kl_sampled"), m("kl_analytic"), m("bias"), m("rmse"), m("corr"),
             m("gate_op"), m("gate_kl"), m("lag"), m("sigma"), m("ess")))

print("\n=== THE CONSTRUCT-VALIDITY QUESTION: does the gate suppress the arms unequally? ===")
for t in ("walker", "g1"):
    p, w_ = agg[(t, "PW-1")], agg[(t, "WML-32")]
    print("  %-7s operator-branch fraction:  PW-1 %.4f   WML-32 %.4f   difference %+.4f"
          % (t, p["gate_op"], w_["gate_op"], p["gate_op"] - w_["gate_op"]))
    print("  %-7s KL-only fraction:          PW-1 %.4f   WML-32 %.4f   ratio %.2fx"
          % (t, p["gate_kl"], w_["gate_kl"], w_["gate_kl"] / max(p["gate_kl"], 1e-9)))
    print("  %-7s analytic KL median:        PW-1 %.4f   WML-32 %.4f   ratio %.2fx"
          % (t, p["kl_analytic"], w_["kl_analytic"],
             w_["kl_analytic"] / max(p["kl_analytic"], 1e-9)))
