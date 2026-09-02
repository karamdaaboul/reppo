"""Byte-level parity comparison for the covariance-freeze flag-off path (phase 1.8).

Compares the pre-freeze source against the freeze source with freeze_sigma=null,
for both operators. Requires exact equality: every serialized array, every metric
curve, max absolute difference 0.0.

Usage: cf_parity_compare.py <parity_dir> <out.json>
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

PAIRS = [("PW-1", "A_PW", "B_PW"), ("WML-32", "A_WML", "B_WML")]
# expected, documented differences: paths and timings, plus the new key that
# records the flag itself
META_SKIP = {"hydra_run_dir", "train_seconds"}


def npz_cmp(pa, pb, report):
    za, zb = np.load(pa), np.load(pb)
    if sorted(za.files) != sorted(zb.files):
        report.append(("KEYS", os.path.basename(pa),
                       "differ: %s vs %s" % (sorted(za.files), sorted(zb.files))))
        return 0, 1
    ok = bad = 0
    for k in sorted(za.files):
        a, b = za[k], zb[k]
        same = a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b)
        if same:
            ok += 1
        else:
            bad += 1
            d = (float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
                 if a.shape == b.shape else float("nan"))
            report.append(("ARRAY", "%s::%s" % (os.path.basename(pa), k),
                           "max|diff| %.6g" % d))
    return ok, bad


def meta_cmp(pa, pb, report):
    ma, mb = json.load(open(pa)), json.load(open(pb))
    keys = (set(ma) | set(mb)) - META_SKIP
    ok = bad = 0
    for k in sorted(keys):
        va, vb = ma.get(k, "<missing>"), mb.get(k, "<missing>")
        if k == "actor_kwargs":
            va = {kk: vv for kk, vv in va.items() if kk != "freeze_sigma"}
            vb = {kk: vv for kk, vv in vb.items() if kk != "freeze_sigma"}
            if mb.get("actor_kwargs", {}).get("freeze_sigma", "ABSENT") not in (None,):
                report.append(("META", "actor_kwargs.freeze_sigma",
                               "expected null, got %r"
                               % mb["actor_kwargs"].get("freeze_sigma")))
                bad += 1
        if va == vb:
            ok += 1
        else:
            bad += 1
            extra = ""
            if isinstance(va, list) and isinstance(vb, list) and len(va) == len(vb):
                extra = " max|diff| %.6g" % float(
                    np.max(np.abs(np.asarray(va, float) - np.asarray(vb, float))))
            report.append(("META", k, "differ%s" % extra))
    return ok, bad


def main(root, out):
    summary = {}
    total_bad = 0
    for label, A, B in PAIRS:
        report, ok, bad = [], 0, 0
        da, db = os.path.join(root, A), os.path.join(root, B)
        if not (os.path.isdir(da) and os.path.isdir(db)):
            summary[label] = {"status": "MISSING", "dirs": [da, db]}
            total_bad += 1
            continue
        subs = sorted(set(os.listdir(da)) | set(os.listdir(db)))
        for sub in subs:
            sa, sb = os.path.join(da, sub), os.path.join(db, sub)
            if not (os.path.isdir(sa) and os.path.isdir(sb)):
                report.append(("EXPORT", sub, "present in only one arm"))
                bad += 1
                continue
            for f in ("actor.npz", "critic.npz", "normalizer.npz"):
                o, b_ = npz_cmp(os.path.join(sa, f), os.path.join(sb, f), report)
                ok += o
                bad += b_
            o, b_ = meta_cmp(os.path.join(sa, "meta.json"),
                             os.path.join(sb, "meta.json"), report)
            ok += o
            bad += b_
        # the training metric curves, independent of the exports
        ma = os.path.join(root, "run_%s" % A, "metrics.npz")
        mb = os.path.join(root, "run_%s" % B, "metrics.npz")
        if os.path.isfile(ma) and os.path.isfile(mb):
            o, b_ = npz_cmp(ma, mb, report)
            ok += o
            bad += b_
        else:
            report.append(("METRICS", "metrics.npz", "missing in one or both arms"))
            bad += 1
        summary[label] = {"status": "IDENTICAL" if bad == 0 else "MISMATCH",
                          "compared": ok + bad, "identical": ok, "mismatched": bad,
                          "exports": subs, "detail": report[:40]}
        total_bad += bad

    with open(out, "w") as f:
        json.dump(summary, f, indent=1)
    for label, s in summary.items():
        print("%-8s %-10s  %s/%s identical"
              % (label, s.get("status"), s.get("identical", 0), s.get("compared", 0)))
        for row in s.get("detail", []):
            print("     %s %s %s" % row)
    print("\nPARITY %s" % ("PASS - max absolute difference 0.0 everywhere"
                           if total_bad == 0 else "FAIL"))
    raise SystemExit(0 if total_bad == 0 else 1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
