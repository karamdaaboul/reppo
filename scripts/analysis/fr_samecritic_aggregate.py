import numpy as np, glob, collections, csv
rows = []
for f in sorted(glob.glob("reports/artifacts/fr_samecritic/*.npz")):
    z = np.load(f, allow_pickle=True)
    g = lambda k: z[k].item() if z[k].shape == () else z[k]
    rows.append({k: g(k) for k in z.files})
for r in rows:
    r["task"] = "walker" if r["env"] == "WalkerRun" else "g1"
    r["arm"] = "PW-1" if r["mode"] == "pathwise" else "WML-32"
with open("reports/artifacts/corrected_samecritic.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
    for r in rows: w.writerow(r)

print("=== RELOAD VERIFICATION (final eval reproduces from checkpoint) ===")
lg = np.array([float(r["logged_return"]) for r in rows])
rl = np.array([float(r["reload_return"]) for r in rows])
rel = np.abs(rl - lg) / np.maximum(np.abs(lg), 1e-9)
print("  n=%d   median |rel diff| %.4f   max %.4f   corr %.4f"
      % (len(rows), np.median(rel), rel.max(), np.corrcoef(lg, rl)[0, 1]))

print("\n=== FROZEN SAME-CRITIC DIAGNOSTIC (identical states, common random numbers) ===")
print("%-7s %-7s | %8s %8s %8s %8s %8s | %11s %12s %10s"
      % ("task","arm","|PW-1|","|PW-32|","|ZO-32|","|c|","|v|",
         "cos(PW1,PW32)","cos(PW32,ZO32)","cos(ZO32,c)"))
agg = {}
for k in dict.fromkeys((r["task"], r["arm"]) for r in rows):
    g = [r for r in rows if (r["task"], r["arm"]) == k]
    m = lambda c: float(np.median([float(r[c]) for r in g]))
    agg[k] = {c: m(c) for c in ("n_pw1","n_pw32","n_zo32","n_c","n_v","n_ubar",
                                "cos_pw1_pw32","cos_pw32_zo32","cos_zo32_c",
                                "cos_pw32_v","cos_pw1_v","R2_ubar_c")}
    a = agg[k]
    print("%-7s %-7s | %8.5f %8.5f %8.5f %8.5f %8.5f | %11.3f %12.3f %10.3f"
          % (k[0], k[1], a["n_pw1"], a["n_pw32"], a["n_zo32"], a["n_c"], a["n_v"],
             a["cos_pw1_pw32"], a["cos_pw32_zo32"], a["cos_zo32_c"]))

print("\n=== what the diagnostic separates ===")
for k in sorted(agg):
    a = agg[k]
    print("  %-7s %-7s  query budget: cos(PW-1, PW-32) = %.3f  (1 sample vs 32, same critic)"
          % (k[0], k[1], a["cos_pw1_pw32"]))
    print("  %-7s %-7s  operator    : cos(PW-32, ZO-32) = %.3f  (equal query, different operator)"
          % ("", "", a["cos_pw32_zo32"]))
    print("  %-7s %-7s  softmax     : cos(ZO-32, c) = %.3f   uniform term: |ubar|/|c| = %.2f"
          % ("", "", a["cos_zo32_c"], a["R2_ubar_c"]))
    print("  %-7s %-7s  full score  : cos(PW-32, v) = %.3f" % ("", "", a["cos_pw32_v"]))
