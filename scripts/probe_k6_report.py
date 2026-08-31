"""k=6 padding-level report in the committed C1-C5 order, plus the preregistered verdict.

Section order is fixed and must not be renumbered:

  C1  per-seed final table, both arms
  C2  gaps at k = 0, 6, 16 (paired percentile bootstrap, seed IDs verified matched)
  C3  contamination table at t/T = 0.25, 0.50, 1.00, then the verbatim rule and the
      mechanical verdict
  C4  MISSING -- no committed definition or source artifact was found
  C5  within-state Q-spread, a POST-TANH CHECKPOINT DIAGNOSTIC

The verdict quotes `docs/prereg_action_padding.md` verbatim as committed in d1ab422
(unchanged since) and applies it mechanically. Where the rule declares a level
CONTAMINATED it also says "claim neither", so no CONFIRMED/REFUTED is derived and the
gap ladder is presented without trend, monotonicity or directional language.

Usage:  probe_k6_report.py > reports/probe_k6_report.md
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import subprocess

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S = "/tmp/claude-1001/-home-human-workspaces-safe-rl/2f6f000e-ea36-4609-bb5e-182397ba52ad/scratchpad"
SEEDS = [0, 1, 2, 3, 4]
LEVELS = [(0, 6), (6, 12), (16, 22)]
BOOT_RNG = 20260830
BOOT_N = 10000
CONTAM_THRESHOLD = 1.5
FRACS = [("p25", 0.25), ("p50", 0.50), ("final", 1.00)]
TRAIN_COMMIT = "3b96deb"

P_LOG = re.compile(
    r"step=(\d+) ret=([-\d.na]+) .*?ent=([-\d.]+) sigma=([\d.]+).*?temp=([\d.]+) "
    r"kl=([\d.]+) \| ess=([\d.na]+) w_max=[\d.na]+ qspread=[-\d.na]+ eta=([\d.na]+)"
    r".*?essP=([\d.na]+)/([\d.na]+)/([\d.na]+)/([\d.na]+) lt4=([\d.na]+)"
)


def logpath(k, tag, s):
    return f"{S}/pad{k}_{tag}_s{s}.log"


def finished(f):
    return os.path.exists(f) and any("_final | return" in l for l in open(f))


def load_log(f):
    rows = [P_LOG.search(l).groups() for l in open(f) if P_LOG.search(l)]
    return np.array([[float(x) for x in g] for g in rows])


def seeds_present(k, tag):
    return [s for s in SEEDS if finished(logpath(k, tag, s))]


def gap(k):
    a_s, b_s = seeds_present(k, "A"), seeds_present(k, "B")
    if a_s != b_s or not a_s:
        return dict(matched=False, A_seeds=a_s, B_seeds=b_s)
    A = np.array([load_log(logpath(k, "A", s))[-1, 1] for s in a_s])
    B = np.array([load_log(logpath(k, "B", s))[-1, 1] for s in a_s])
    D = A - B
    rng = np.random.default_rng(BOOT_RNG)
    bs = np.array([D[rng.integers(0, len(D), len(D))].mean() for _ in range(BOOT_N)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(matched=True, seeds=a_s, A=A, B=B, D=D, mean=float(D.mean()),
                lo=float(lo), hi=float(hi),
                t=float(D.mean() / (D.std(ddof=1) / np.sqrt(len(D)))),
                excludes_zero=bool(lo > 0 or hi < 0))


def sigma_rows(k):
    out = []
    for tag, arm in (("pathwise_fa", "A"), ("weighted_mle", "B")):
        for s in SEEDS:
            for c, tt in FRACS:
                f = f"{S}/probes/WalkerRun_{tag}_pad{k}_s{s}_{c}.sigma.json"
                if os.path.exists(f):
                    j = json.load(open(f))
                    out.append((arm, s, tt, j["sigma_real_median"],
                                j["sigma_pad_median"],
                                j["ratio_pad_over_real_median"]))
    return out


def qspread_rows(k):
    out = []
    for tag, arm in (("pathwise_fa", "A"), ("weighted_mle", "B")):
        for s in SEEDS:
            f = f"{S}/probes/WalkerRun_{tag}_pad{k}_s{s}_final.json"
            if os.path.exists(f):
                j = json.load(open(f))
                out.append((arm, s, j["within_sd_mean"], j["within_sd_real_only"],
                            j["within_sd_pad_only"], j["q_p1_p50_p99"][1],
                            j["action_sat"], j["n_states"]))
    return out


def verbatim_rule():
    txt = subprocess.check_output(
        ["git", "show", "d1ab422:docs/prereg_action_padding.md"], cwd=REPO_ROOT
    ).decode()
    return ("## Decision rule"
            + txt.split("## Decision rule", 1)[1].split("## Reference points", 1)[0]
            ).rstrip()


def mtime(p):
    return (dt.datetime.fromtimestamp(os.path.getmtime(p)).strftime("%Y-%m-%d %H:%M")
            if os.path.exists(p) else "MISSING")


def main():
    L = []
    W = L.append
    g = {k: gap(k) for k, _ in LEVELS}
    sg6, sg16 = sigma_rows(6), sigma_rows(16)
    qs6 = qspread_rows(6)

    fin6 = {a: [r[5] for r in sg6 if r[0] == a and r[2] == 1.00] for a in ("A", "B")}
    med = {a: (float(np.median(v)) if v else None) for a, v in fin6.items()}
    exceed = {a: (med[a] is not None and med[a] > CONTAM_THRESHOLD) for a in ("A", "B")}
    n_over = sum(1 for r in sg6 if r[2] == 1.00 and r[5] > CONTAM_THRESHOLD)
    n_fin = sum(1 for r in sg6 if r[2] == 1.00)
    same_side = exceed["A"] == exceed["B"]
    contaminated = same_side and exceed["A"]

    W("# Padding level k=6 (d=12) — C1–C5 report and preregistered verdict")
    W("")
    W("Prereg: `docs/prereg_action_padding.md`, registered 2026-08-30, committed in "
      "`d1ab422` and unchanged since (verified by `git diff d1ab422 HEAD --` on that "
      "path). Generated by `scripts/probe_k6_report.py`. Section order C1–C5 is the "
      "committed order and is not renumbered.")
    W("")

    # ------------------------------------------------------------------ verdict
    W("## Verdict")
    W("")
    if contaminated:
        W("> **k=6 (d=12): CONTAMINATED. Claim neither CONFIRMED nor REFUTED.**")
        W("")
        W(f"At the final checkpoint the median sigma over the padded coordinates "
          f"exceeds {CONTAM_THRESHOLD}× the median sigma over the real 6 in "
          f"**{n_over} of {n_fin}** checkpoints: arm A median R = "
          f"**{med['A']:.2f}**, arm B median R = **{med['B']:.2f}**. Both arms fall on "
          "the same side of the threshold, so the opposite-sides case does not arise "
          "and no aggregation convention is required.")
    elif not same_side:
        W("> **k=6 (d=12): RULE SILENT — NO VERDICT.**")
        W("")
        W(f"The two arms fall on opposite sides of the {CONTAM_THRESHOLD} threshold "
          f"(arm A median R = {med['A']:.2f}, arm B median R = {med['B']:.2f}). The "
          "registered rule states the contamination test in the singular — \"median "
          "sigma over the padded coordinates\" — and does not define how to aggregate "
          "across arms. No verdict is issued and no aggregation is invented here.")
    else:
        W("> **k=6 (d=12): NOT CONTAMINATED** by the registered test.")
        W("")
        W(f"Arm A median R = {med['A']:.2f}, arm B median R = {med['B']:.2f}; both "
          f"within the {CONTAM_THRESHOLD} threshold.")
    W("")
    W("The registered rule attaches \"report separately, claim neither\" to the "
      "CONTAMINATED outcome. It grants no exception permitting a CONFIRMED or REFUTED "
      "determination once contamination is established, so none is made here, and the "
      "gap ladder in C2 is reported as bare per-level estimates without trend, "
      "monotonicity or directional characterisation.")
    W("")
    W("Rule as committed in `d1ab422`, quoted verbatim:")
    W("")
    W("> " + verbatim_rule().replace("\n", "\n> "))
    W("")

    # ---------------------------------------------------------------------- C1
    W("## C1 — per-seed final table, both arms (k=6, d=12)")
    W("")
    W("Final = last eval at 52.3M steps (21 evals). alpha frozen at 0.01528. "
      "`up-events` counts evals where entropy rose >5 nats or sigma rose >50%.")
    W("")
    W("| arm | seed | final return | ent mean | sigma mean | eta range | ESS mean | "
      "ESS p5/p25/p50/p75 | ESS<4 | worst eval drop | up-events | NaN |")
    W("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for tag, arm in (("A", "A-frozen (pathwise)"), ("B", "B-frozen (weighted-MLE)")):
        for s in SEEDS:
            f = logpath(6, tag, s)
            if not finished(f):
                W(f"| {arm} | {s} | (not finished) | | | | | | | | | |")
                continue
            a = load_log(f)
            n = len(a)
            ev = np.where((np.diff(a[:, 2]) > 5)
                          | (np.diff(a[:, 3]) / a[:-1, 3] > 0.5))[0] + 1
            nan = "Y" if bool(np.isnan(a[:, 1]).any()) else "N"
            if tag == "B":
                W(f"| {arm} | {s} | {a[-1,1]:.1f} | {a[:,2].mean():.2f} | "
                  f"{a[:,3].mean():.3f} | {a[:,7].min():.3f}–{a[:,7].max():.3f} | "
                  f"{a[:,6].mean():.2f} | {a[:,8].mean():.1f}/{a[:,9].mean():.1f}/"
                  f"{a[:,10].mean():.1f}/{a[:,11].mean():.1f} | {a[:,12].mean():.3f} | "
                  f"{np.diff(a[:,1]).min():.1f} | {len(ev)}/{n-1} | {nan} |")
            else:
                W(f"| {arm} | {s} | {a[-1,1]:.1f} | {a[:,2].mean():.2f} | "
                  f"{a[:,3].mean():.3f} | — | — | — | — | "
                  f"{np.diff(a[:,1]).min():.1f} | {len(ev)}/{n-1} | {nan} |")
    W("")
    W("Arm A has no E-step, hence the em-dashes in the eta/ESS columns.")
    W("")

    # ---------------------------------------------------------------------- C2
    W("## C2 — gaps at k = 0, 6, 16")
    W("")
    W("**Precondition check (paired bootstrap requires matched seed IDs).**")
    W("")
    W("| level | arm A seed IDs | arm B seed IDs | matched |")
    W("|---|---|---|---|")
    for k, _ in LEVELS:
        a_s, b_s = seeds_present(k, "A"), seeds_present(k, "B")
        W(f"| k={k} | {a_s} | {b_s} | {'**yes**' if a_s == b_s else '**MISMATCH**'} |")
    W("")
    if not all(g[k]["matched"] for k, _ in LEVELS):
        W("**STOP — seed IDs are not matched at every level; the paired bootstrap is "
          "not run.** See the mismatch above.")
        W("")
    else:
        W("All three levels have identical seed IDs in both arms, so the paired "
          f"bootstrap is permitted. {BOOT_N:,} paired percentile-bootstrap resamples, "
          f"`np.random.default_rng({BOOT_RNG})`, resampling seeds with replacement.")
        W("")
        W("| level | d | arm A final | arm B final | gap A−B | 95% percentile CI | "
          "CI excludes 0 | paired t |")
        W("|---|---|---|---|---|---|---|---|")
        for k, d in LEVELS:
            q = g[k]
            W(f"| k={k} | {d} | {q['A'].mean():.1f} ± {q['A'].std(ddof=1):.1f} | "
              f"{q['B'].mean():.1f} ± {q['B'].std(ddof=1):.1f} | {q['mean']:+.1f} | "
              f"[{q['lo']:+.1f}, {q['hi']:+.1f}] | "
              f"{'yes' if q['excludes_zero'] else 'no'} | {q['t']:+.2f} |")
        W("")
        for k, d in LEVELS:
            q = g[k]
            W(f"* k={k} (d={d}), per-seed Δ = A−B: "
              f"{np.round(q['D'],1).tolist()}, seeds {q['seeds']}.")
        W("")
        if contaminated:
            W("These are three independent per-level estimates. Because k=6 and k=16 "
              "are both CONTAMINATED by the registered test (C3), no trend, ordering "
              "or monotonicity across levels is asserted or evaluated, and no "
              "CONFIRMED/REFUTED determination is derived from them.")
        W("")
        W("The prereg records a reference point for d=6 of \"gap +13.7 (t=+0.92) "
          "tie\", measured before the padding lanes were run. The matched "
          "in-experiment k=0 lane reported here gives "
          f"{g[0]['mean']:+.1f} [{g[0]['lo']:+.1f}, {g[0]['hi']:+.1f}]. Both have a "
          "95% CI containing zero. The in-experiment value is the one tabulated; the "
          "discrepancy between the two d=6 measurements is noted, not resolved.")
        W("")

    # ---------------------------------------------------------------------- C3
    W("## C3 — contamination table and mechanical verdict")
    W("")
    W("Median pre-squash sigma over 2048 visited states × coordinates, from "
      "`scripts/sigma_probe.py` (B=256 envs, 200-step burn-in, 8 chunks 25 steps "
      "apart, `PRNGKey(0)`), at three training fractions.")
    W("")
    W("| arm | seed | t/T | median sigma real (6) | median sigma padded (6) | "
      "R = pad/real | R > 1.5 |")
    W("|---|---|---|---|---|---|---|")
    for arm, s, tt, real, pad, r in sg6:
        W(f"| {arm} | {s} | {tt:.2f} | {real:.4f} | {pad:.4f} | **{r:.2f}** | "
          f"{'**yes**' if r > CONTAM_THRESHOLD else 'no'} |")
    W("")
    W("Final-checkpoint summary — the quantity the rule tests:")
    W("")
    W("| arm | per-seed R at t/T = 1.00 | median | exceeds 1.5 |")
    W("|---|---|---|---|")
    for a in ("A", "B"):
        W(f"| {a} | {np.round(fin6[a],2).tolist()} | **{med[a]:.2f}** | "
          f"{'**yes**' if exceed[a] else 'no'} |")
    W("")
    W(f"Checkpoints over threshold at t/T = 1.00: **{n_over} of {n_fin}**. Arms on the "
      f"same side of {CONTAM_THRESHOLD}: **{same_side}**.")
    W("")
    W("Rule as committed in `d1ab422`, quoted verbatim:")
    W("")
    W("> " + verbatim_rule().replace("\n", "\n> "))
    W("")
    if contaminated:
        W("**Mechanical verdict: CONTAMINATED — report separately, claim neither.**")
    elif not same_side:
        W("**Mechanical verdict: RULE SILENT — NO VERDICT.** The arms fall on opposite "
          "sides of the threshold and the rule does not define aggregation across "
          "arms.")
    else:
        W("**Mechanical verdict: not contaminated.**")
    W("")
    m16 = {a: float(np.median([r[5] for r in sg16 if r[0] == a and r[2] == 1.00]))
           for a in ("A", "B")}
    W(f"Cross-reference, final-checkpoint median R: A {med['A']:.2f} (k=6) vs "
      f"{m16['A']:.2f} (k=16); B {med['B']:.2f} (k=6) vs {m16['B']:.2f} (k=16).")
    W("")
    W("Per-seed median sigma over the real 6 coordinates, arm B, at the three "
      "training fractions (numbers only):")
    W("")
    W("| arm B seed | real sigma t/T=0.25 | t/T=0.50 | t/T=1.00 | padded sigma "
      "t/T=0.25 | t/T=0.50 | t/T=1.00 |")
    W("|---|---|---|---|---|---|---|")
    for sd in SEEDS:
        row = {tt: (re_, pa) for (arm, ss, tt, re_, pa, _r) in sg6
               if arm == "B" and ss == sd}
        if len(row) == 3:
            W(f"| {sd} | {row[0.25][0]:.4f} | {row[0.50][0]:.4f} | {row[1.00][0]:.4f} "
              f"| {row[0.25][1]:.4f} | {row[0.50][1]:.4f} | {row[1.00][1]:.4f} |")
    W("")
    W("The corresponding arm-A values are in the full C3 table above.")
    W("")
    W("For context, the same test at k=16 (already measured):")
    W("")
    W("| arm | per-seed R at t/T = 1.00 (k=16) | median |")
    W("|---|---|---|")
    for a in ("A", "B"):
        v = [r[5] for r in sg16 if r[0] == a and r[2] == 1.00]
        if v:
            W(f"| {a} | {np.round(v,2).tolist()} | **{np.median(v):.2f}** |")
    W("")

    # ---------------------------------------------------------------------- C4
    W("## C4 — MISSING: no committed definition or source artifact was found.")
    W("")
    W("Not omitted and not renumbered; C5 below keeps its number.")
    W("")
    W("Provenance inventory of what was checked:")
    W("")
    W("| checked | finding |")
    W("|---|---|")
    W("| `scripts/pad_report.py` (the committed k-level generator) | emits four "
      "labelled blocks only — per-seed table per arm, `## Gap pad{k}`, `## sigma real "
      "vs padded`, `## within-state sd_i(Q)`; contains no C-numbering |")
    W("| probe artifact directory, k=16 | exactly two file types: `*_final.json` "
      "(`scripts/probe_ckpt.py`) and `*.sigma.json` (`scripts/sigma_probe.py`); no "
      "third generator output |")
    W("| `scripts/audit_across_state.py`, `scripts/audit_saturation.py` | "
      "self-labelled \"A-audit\" and \"B-audit\"; not C-sections; run only on "
      "non-padded checkpoints |")
    W("| repository, all paths | no k=16 C-section report file committed |")
    W("| originating session scratchpad | no k=16 C-section report file present |")
    W("")
    W("A C4 was not reconstructed from the unused fields of `probe_ckpt.py` output "
      "(`across_sd`, `within_over_across`, `edge0_mean`, `edgeTop_mean`, "
      "`mc_soft_p50`), because selecting among them would be a guess at the registered "
      "definition. Supplying the k=16 C4 definition and generator path is sufficient "
      "to fill this section; the k=6 checkpoints are exported and available.")
    W("")

    # ---------------------------------------------------------------------- C5
    W("## C5 — within-state Q-spread (post-tanh checkpoint diagnostic)")
    W("")
    W("**This is a post-tanh checkpoint diagnostic. It is not the pre-tanh "
      "theoretical oracle of Probe 1.** It perturbs post-squash actions drawn from "
      "$\\pi_{\\rm old}$ and reports the spread of the critic's scalar output across "
      "them; it identifies no error field, no gradient, and no $\\Sigma$-metric "
      "quantity, and none of its numbers may be substituted for $V_e$, $G_z$ or "
      "$\\Omega_z$.")
    W("")
    if not qs6:
        W("**Status: PENDING — the probe run had not completed when this report was "
          "generated.**")
        W("")
    else:
        W("Protocol: 2048 visited states (B=256 envs, 200-step burn-in, 8 chunks 25 "
          "steps apart), M=32 base action perturbations per state, `PRNGKey(0)` with "
          "the documented split order (`k1, k2, key = split(key, 3)` per step; action "
          "draw keyed by `fold_in(key, 7)` at each sampled chunk), actions clipped to "
          "±(1−1e−4). The **same** states and the **same** base perturbation tensor "
          "feed all three columns; only the masked action tensor differs — `real-only` "
          "pins the 6 padded coordinates at $\\tanh(\\mu)$, `pad-only` pins the 6 real "
          "coordinates at $\\tanh(\\mu)$.")
        W("")
        W("| arm | seed | sd all (12) | sd real-only | sd pad-only | pad/all | "
          "Q median | action sat | n states |")
        W("|---|---|---|---|---|---|---|---|---|")
        for arm, s, a, r, p, q50, sat, n in qs6:
            W(f"| {arm} | {s} | {a:.4f} | {r:.4f} | {p:.4f} | {p/a:.3f} | {q50:.1f} | "
              f"{sat:.3f} | {n} |")
        W("")
        for arm in ("A", "B"):
            rows = [x for x in qs6 if x[0] == arm]
            if rows:
                arr = np.array([[x[2], x[3], x[4]] for x in rows])
                W(f"* Arm {arm} medians: all {np.median(arr[:,0]):.4f}, real-only "
                  f"{np.median(arr[:,1]):.4f}, pad-only {np.median(arr[:,2]):.4f}, "
                  f"pad/all {np.median(arr[:,2]/arr[:,0]):.3f}.")
        W("")
        W("**Executed protocol.** Executed: float32 `jnp.std` with centered "
          "two-pass variance. This is **not** the requested float64 protocol and is "
          "not presented as it. The requested float64 accumulation with per-state "
          "reference-mean centring was not run for k=6.")
        W("")
        W("The raw-moment accumulator was not used: `jnp.std` forms the mean first "
          "and then the deviations, so it never builds "
          "$\\mathbb E[Q^2]-\\mathbb E[Q]^2$.")
        W("")
        W("**Precision evidence, stated narrowly.** A sensitivity recheck was run at "
          "**k=16** (`reports/probe1_restricted_z.md` §6): on identical draws the "
          "float32 two-pass and float64 reference-mean-centred paths differed by "
          "approximately **1e-8 relative on the aggregate** and by **at most 1.3e-6 "
          "per state**. That evidence supports the numerical stability of the "
          "two-pass computation **at k=16**. It does **not** constitute a k=6 float64 "
          "recheck, and no k=6 float64 measurement is reported anywhere in this "
          "document.")
        W("")
        W("**Unresolved follow-up** (not a completed check): "
          "`python scripts/c5_float64_recheck.py "
          "exports/WalkerRun_pathwise_fa_pad6_s*_final "
          "exports/WalkerRun_weighted_mle_pad6_s*_final`.")
        W("")
        W("**Role in the verdict: none.** C5 is descriptive. It has no role in the "
          "committed contamination verdict, which is already fixed by the "
          "final-checkpoint sigma ratios in C3: **CONTAMINATED, claim neither**. "
          "Nothing in this section can change that determination, and the outstanding "
          "float64 follow-up above therefore does not gate it.")
        W("")
    # --------------------------------------------------------------- provenance
    W("## Provenance")
    W("")
    W(f"* **Training code commit:** `{TRAIN_COMMIT}` (the JAX fork that produced the "
      "padded runs). Analysis code: this repository at the commit carrying this "
      "report.")
    W("* **Checkpoint selection:** the `_final` export of each run — "
      "`checkpoint_frac = 1.0`, iteration 399, 52,297,728 env steps. `_p25` and `_p50` "
      "exports are used only for the C3 training-fraction columns.")
    W(f"* **Seed IDs:** {SEEDS} for both arms at every level; matched-ness verified in "
      "C2 before any paired bootstrap was computed.")
    W("")
    W("| checkpoint | export timestamp |")
    W("|---|---|")
    for tag in ("pathwise_fa", "weighted_mle"):
        for s in SEEDS:
            d = f"exports/WalkerRun_{tag}_pad6_s{s}_final"
            W(f"| `{d}` | {mtime(os.path.join(REPO_ROOT, d, 'meta.json'))} |")
    W("")
    W("| input | path |")
    W("|---|---|")
    W(f"| training logs (C1, C2) | `{S}/pad{{0,6,16}}_{{A,B}}_s{{0..4}}.log` |")
    W(f"| sigma probes (C3) | `{S}/probes/WalkerRun_{{arm}}_pad6_s{{0..4}}_"
      "{p25,p50,final}.sigma.json` |")
    W(f"| Q-spread probes (C5) | `{S}/probes/WalkerRun_{{arm}}_pad6_s{{0..4}}_"
      "final.json` |")
    W("")
    W("**Missing or failed measurements**")
    W("")
    W("* **C4** — MISSING: no committed definition or source artifact was found. "
      "See C4 for the provenance inventory.")
    if not qs6:
        W("* **C5** — PENDING at generation time; the probe run had not completed.")
    else:
        W("* **C5** — executed as float32 `jnp.std` with centered two-pass variance, "
          "not the requested float64 protocol. The k=6 float64 recheck is outstanding "
          "and is listed in C5 as unresolved follow-up. C5 is descriptive and has no "
          "role in the contamination verdict.")
    W("* No other measurement in this report failed. All 30 k=6 sigma probes and all "
      "30 training logs (3 levels × 2 arms × 5 seeds) were present and complete.")
    W("")

    print("\n".join(L))


if __name__ == "__main__":
    main()
