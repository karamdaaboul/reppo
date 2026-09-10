#!/usr/bin/env python3
"""Rerun-to-canonical validity audit. READ-ONLY.

Closes the 2-of-6 gap left by scripts/analysis/klsplit.py, which resolved canonical runs
through reports/artifacts/exports_manifest.csv. That manifest contains ZERO rows tagged
pathwise_fa_noent, so four of six cells silently resolved to nothing and printed
"canonical metrics not found" while the other two printed numbers. This script resolves
canonical runs from exports/*/meta.json by metadata key instead, and asserts a unique match.

Launches nothing. Writes nothing outside its own new report/CSV. Modifies no existing
artifact, seed, config or result.

Hard rule: an empty comparison can never return PASS. Every count is asserted.
"""
from __future__ import annotations
import csv, hashlib, json, os, subprocess, sys, time

CANON_ROOT = "/hpcwork/qzi10910/estep_wt"
RERUN_ROOT = "/hpcwork/qzi10910/logsplit_wt"
SEEDS = list(range(301, 309))
W3 = (18, 19, 20)          # frozen score_window3 indices, of 21
SERIES = "eval_return_curve"

# Fields expected to differ for documented storage/export reasons ONLY.
WHITELIST = {
    "hydra_run_dir",        # different git worktree by design (export isolation)
    "train_seconds",        # wall-clock, never reproducible
}
# Curve fields are compared separately / are outputs, not configuration.
CURVE_SUFFIX = "_curve"
# Identity of a training configuration. Used to resolve canonical <- rerun uniquely.
# METADATA GAP, documented and load-bearing:
#   meta.json does NOT record update_entropy_lagrangian anywhere (verified: no such key at
#   top level or in actor_kwargs; alpha_entropy, ent_start and alpha_curve are IDENTICAL
#   between the _ent and _noent exports of every task). Metadata alone therefore resolves
#   each rerun to TWO canonical runs. The export tag is config-derived -- scripts/
#   train_and_export.py builds it from the resolved config -- so it is provenance, not a
#   guessed directory name, and it is the only carrier of that config bit. It is included
#   in the key for that reason and that reason only.
KEY_FIELDS = [
    "env_name", "actor_update_mode", "seed", "eps_e", "estep_num_samples",
    "time_steps", "num_eval", "obs_dim", "action_dim", "critic_obs_dim",
    "gamma", "lmbda", "max_episode_steps", "hl_gauss", "normalize_env",
    "reward_scaling", "checkpoint_frac", "iteration", "action_pad", "normalizer_eps",
]

fails: list[str] = []


def check(cond, msg):
    if not cond:
        fails.append(msg)
    return cond


def load_meta(d):
    p = os.path.join(d, "meta.json")
    if not os.path.exists(p):
        return None
    try:
        return json.load(open(p))
    except Exception as e:
        fails.append("unreadable meta.json at %s: %s" % (d, e))
        return None


def alpha_frozen(m):
    c = m.get("alpha_curve") or []
    return len(set(round(float(x), 12) for x in c)) <= 1 if c else None


def ent_start(m):
    return (m.get("actor_kwargs") or {}).get("ent_start")


def tag_of(name, m):
    """Config-derived export tag: <ENV>_<tag>_s<seed>_final -> <tag>."""
    env, seed = m.get("env_name"), m.get("seed")
    pre, suf = "%s_" % env, "_s%s_final" % seed
    if name.startswith(pre) and name.endswith(suf):
        return name[len(pre):-len(suf)]
    return None


def keyof(m, tag):
    k = tuple(m.get(f) for f in KEY_FIELDS)
    return k + (alpha_frozen(m),
                round(float(ent_start(m)), 12) if ent_start(m) is not None else None,
                tag)


def sha256_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def enumerate_exports(root):
    base = os.path.join(root, "exports")
    out = {}
    if not os.path.isdir(base):
        fails.append("no exports dir at %s" % base)
        return out
    for name in sorted(os.listdir(base)):
        if not name.endswith("_final"):
            continue
        d = os.path.join(base, name)
        m = load_meta(d)
        if m is not None:
            out[name] = (d, m)
    return out


def main():
    t0 = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    print("=" * 100)
    print("RERUN-TO-CANONICAL VALIDITY AUDIT   %s" % t0)
    print("=" * 100)

    rerun = enumerate_exports(RERUN_ROOT)
    canon = enumerate_exports(CANON_ROOT)
    print("  rerun export dirs (_final):     %d" % len(rerun))
    print("  canonical export dirs (_final): %d" % len(canon))

    # ---- derive cells from rerun metadata, do not assume ----
    cells = {}
    for name, (d, m) in rerun.items():
        cid = (m.get("env_name"), m.get("actor_update_mode"), alpha_frozen(m))
        cells.setdefault(cid, []).append((name, d, m))

    print("\n" + "-" * 100)
    print("EXPECTED CELLS, derived from rerun metadata (env_name, actor_update_mode, alpha_frozen)")
    print("-" * 100)
    for cid in sorted(cells, key=lambda x: (str(x[0]), str(x[1]))):
        seeds = sorted(m.get("seed") for _, _, m in cells[cid])
        print("  %-24s %-14s alpha_frozen=%-5s  n=%d  seeds=%s"
              % (cid[0], cid[1], cid[2], len(seeds), seeds))
    check(len(cells) == 6, "expected 6 cells, derived %d" % len(cells))

    # ---- index canonical by metadata key ----
    canon_by_key = {}
    for name, (d, m) in canon.items():
        canon_by_key.setdefault(keyof(m, tag_of(name, m)), []).append((name, d, m))

    rows, cellrows = [], []
    expected_pairs = 6 * len(SEEDS)
    compared = unmatched = duplicate = 0

    for cid in sorted(cells, key=lambda x: (str(x[0]), str(x[1]))):
        env, arm, froz = cid
        entries = {m.get("seed"): (name, d, m) for name, d, m in cells[cid]}
        found = sorted(entries)
        missing = [s for s in SEEDS if s not in entries]
        dup = [s for s in found if sum(1 for _, _, m in cells[cid] if m.get("seed") == s) > 1]
        matched = comp = 0
        mx_abs = mx_rel = mx_w3 = 0.0
        worst_seed = worst_idx = None
        discrep: dict[str, int] = {}
        rerun_path = canon_path = ""

        for s in SEEDS:
            if s not in entries:
                continue
            rname, rdir, rm = entries[s]
            rerun_path = rerun_path or os.path.dirname(rdir)
            cands = canon_by_key.get(keyof(rm, tag_of(rname, rm)), [])
            if len(cands) != 1:
                if len(cands) == 0:
                    unmatched += 1
                    fails.append("%s %s s%d: ZERO canonical matches" % (env, arm, s))
                else:
                    duplicate += 1
                    fails.append("%s %s s%d: %d canonical matches (%s)"
                                 % (env, arm, s, len(cands), [c[0] for c in cands]))
                continue
            cname, cdir, cm = cands[0]
            canon_path = canon_path or os.path.dirname(cdir)
            matched += 1

            a = rm.get(SERIES); b = cm.get(SERIES)
            if not check(isinstance(a, list) and len(a) == 21,
                         "%s %s s%d rerun %s length %s != 21" % (env, arm, s, SERIES, a and len(a))):
                continue
            if not check(isinstance(b, list) and len(b) == 21,
                         "%s %s s%d canon %s length %s != 21" % (env, arm, s, SERIES, b and len(b))):
                continue
            af = [float(x) for x in a]; bf = [float(x) for x in b]
            if not check(all(x == x and abs(x) != float("inf") for x in af + bf),
                         "%s %s s%d: non-finite value in return series" % (env, arm, s)):
                continue

            for i, (x, y) in enumerate(zip(af, bf)):
                da = abs(x - y)
                dr = da / abs(y) if y != 0 else (0.0 if da == 0 else float("inf"))
                if da > mx_abs:
                    mx_abs, worst_seed, worst_idx = da, s, i
                mx_rel = max(mx_rel, dr)
            w3r = sum(af[i] for i in W3) / 3.0
            w3c = sum(bf[i] for i in W3) / 3.0
            mx_w3 = max(mx_w3, abs(w3r - w3c))

            for k in sorted(set(rm) | set(cm)):
                if k in WHITELIST or k.endswith(CURVE_SUFFIX) or k.endswith("_leaf_paths"):
                    continue
                if rm.get(k) != cm.get(k):
                    discrep[k] = discrep.get(k, 0) + 1

            comp += 1
            compared += 1
            rows.append(dict(env=env, arm=arm, tag=tag_of(rname, rm), seed=s,
                             rerun=rname, canonical=cname,
                             rerun_git=rm.get("git_commit"), canon_git=cm.get("git_commit"),
                             max_abs_this_seed=max(abs(x - y) for x, y in zip(af, bf)),
                             w3_rerun=w3r, w3_canon=w3c, w3_diff=abs(w3r - w3c)))

        check(found == SEEDS, "%s %s: seed set %s != expected %s" % (env, arm, found, SEEDS))
        check(not dup, "%s %s: duplicate seeds %s" % (env, arm, dup))
        check(comp == len(SEEDS), "%s %s: compared %d/%d seed pairs" % (env, arm, comp, len(SEEDS)))
        cellrows.append(dict(env=env, arm=arm, alpha_frozen=froz,
                             tag=(tag_of(*[(n, m) for n, d, m in cells[cid]][0]) if cells[cid] else None), rerun_path=rerun_path,
                             canon_path=canon_path, expected=len(SEEDS), found=len(found),
                             matched=matched, compared=comp, missing=missing, duplicates=dup,
                             len_ok="21/21", max_abs=mx_abs, max_rel=mx_rel, max_w3=mx_w3,
                             worst_seed=worst_seed, worst_idx=worst_idx,
                             discrepancies=discrep,
                             verdict="PASS" if (comp == len(SEEDS) and not missing and not dup) else "FAIL"))

    print("\n" + "=" * 100)
    print("PER-CELL AUDIT")
    print("=" * 100)
    for c in cellrows:
        print("\n  %s / %s   (alpha_frozen=%s)" % (c["env"], c["arm"], c["alpha_frozen"]))
        print("    rerun     %s" % c["rerun_path"])
        print("    canonical %s" % c["canon_path"])
        print("    seeds expected/found/matched/compared: %d/%d/%d/%d   missing=%s  dup=%s"
              % (c["expected"], c["found"], c["matched"], c["compared"], c["missing"] or "-", c["duplicates"] or "-"))
        print("    series length: %s" % c["len_ok"])
        print("    max |diff| = %.6g   max rel = %.6g   max |score_window3 diff| = %.6g"
              % (c["max_abs"], c["max_rel"], c["max_w3"]))
        print("    worst: seed %s, eval index %s" % (c["worst_seed"], c["worst_idx"]))
        print("    export tag: %s" % c.get("tag"))
        print("    config discrepancies (non-whitelisted): %s"
              % (", ".join("%s x%d" % (k, v) for k, v in sorted(c["discrepancies"].items())) or "none"))
        print("    -> %s" % c["verdict"])

    check(len(cellrows) == 6, "located %d cells, expected 6" % len(cellrows))
    check(compared == expected_pairs, "compared %d pairs, expected %d" % (compared, expected_pairs))
    check(unmatched == 0, "%d unmatched" % unmatched)
    check(duplicate == 0, "%d duplicate matches" % duplicate)

    print("\n" + "=" * 100)
    print("TOTALS")
    print("=" * 100)
    print("  EXPECTED CELLS:     6")
    print("  COMPARED CELLS:     %d" % len(cellrows))
    print("  EXPECTED RUN PAIRS: %d" % expected_pairs)
    print("  COMPARED RUN PAIRS: %d" % compared)
    print("  UNMATCHED:          %d" % unmatched)
    print("  DUPLICATE MATCHES:  %d" % duplicate)
    print("  FAILED ASSERTIONS:  %d" % len(fails))
    for f in fails:
        print("     - %s" % f)
    ok = (len(fails) == 0 and compared == expected_pairs and compared > 0 and len(cellrows) == 6)
    print("  OVERALL: %s" % ("PASS" if ok else "FAIL"))

    out = os.path.join(CANON_ROOT, "reports/artifacts/audit_rerun_validity.csv")
    if rows:
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print("\n  per-seed rows -> %s  (%d rows)" % (out, len(rows)))
    else:
        print("\n  NO ROWS COMPARED -- nothing written. This is a FAIL, never a PASS.")

    print("\n  PROVENANCE")
    print("    timestamp:   %s" % t0)
    print("    script:      %s" % os.path.abspath(__file__))
    print("    script sha256: %s" % sha256_file(os.path.abspath(__file__)))
    for label, root in (("canonical", CANON_ROOT), ("rerun", RERUN_ROOT)):
        try:
            sha = subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip()
            st = subprocess.check_output(["git", "-C", root, "status", "--porcelain", "--untracked-files=no"], text=True).strip()
            print("    %s tree: %s  HEAD %s  tracked-clean %s" % (label, root, sha, "YES" if not st else "NO"))
        except Exception as e:
            print("    %s tree: git query failed: %s" % (label, e))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
