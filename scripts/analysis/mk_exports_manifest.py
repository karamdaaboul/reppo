"""Inventory every exports/*_final checkpoint directory.

READ-ONLY on the checkpoints: opens each file to hash it and reads meta.json;
nothing under exports/ is written, renamed or removed.

Usage: mk_exports_manifest.py <out.csv>
"""
from __future__ import annotations
import csv, datetime, glob, hashlib, json, os, subprocess, sys

ROOT = "/rwthfs/rz/cluster/home/qzi10910/repos/reppo"
os.chdir(ROOT)
FILES = ("actor.npz", "critic.npz", "meta.json", "normalizer.npz")
COLS = ["dir", "env_name", "actor_update_mode", "seed", "seed_index", "action_dim",
        "action_pad", "obs_dim", "critic_obs_dim", "estep_num_samples", "eps_e",
        "alpha_entropy", "alpha_kl", "gamma", "lmbda", "time_steps", "iteration",
        "checkpoint_frac", "final_eval_return", "ess_final", "ess_degenerate",
        "nan_in_eval", "normalize_env", "train_seconds", "hydra_run_dir",
        "files_present", "n_missing", "bytes_total", "mtime_utc",
        "sha256_actor", "sha256_critic", "sha256_meta", "sha256_normalizer"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main(out):
    dirs = sorted(glob.glob("exports/*_final"))
    rows = []
    for d in dirs:
        r = {c: "" for c in COLS}
        r["dir"] = d
        present, total, newest = [], 0, 0.0
        for fn in FILES:
            p = os.path.join(d, fn)
            if os.path.isfile(p):
                present.append(fn)
                st = os.stat(p)
                total += st.st_size
                newest = max(newest, st.st_mtime)
                r["sha256_" + fn.split(".")[0]] = sha(p)
        r["files_present"] = "|".join(present)
        r["n_missing"] = len(FILES) - len(present)
        r["bytes_total"] = total
        r["mtime_utc"] = (datetime.datetime.utcfromtimestamp(newest)
                          .strftime("%Y-%m-%dT%H:%M:%SZ") if newest else "")
        mp = os.path.join(d, "meta.json")
        if os.path.isfile(mp):
            try:
                m = json.load(open(mp))
            except Exception as e:                       # reported, not swallowed
                r["files_present"] += "|META_UNREADABLE:%s" % type(e).__name__
                m = {}
            for k in COLS:
                if k in m and not isinstance(m[k], (list, dict)):
                    r[k] = m[k]
        rows.append(r)
        print("  %-58s %d/%d files" % (d, len(present), len(FILES)), flush=True)

    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader(); w.writerows(rows)

    sha_git = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    n_inc = sum(1 for r in rows if r["n_missing"])
    print("\n%d directories, %d incomplete, manifest %s" % (len(rows), n_inc, out))
    print("git %s  manifest sha256 %s" % (sha_git[:12], sha(out)[:16]))
    # never modified anything under exports/
    print("exports/ writes: 0 (this script only reads)")


if __name__ == "__main__":
    main(sys.argv[1])
