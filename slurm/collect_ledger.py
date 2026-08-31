#!/usr/bin/env python
"""Merge the per-array-task ledger records into ledger/runs.jsonl.

Each array task writes one ``ledger/runs.d/<run_id>.json``; 48 concurrent appends to
a single JSONL would interleave and corrupt lines. This merges them in deterministic
run_id order, skipping run_ids already present, so it is idempotent and safe to
re-run after a partially completed array.

    python slurm/collect_ledger.py                    # merge ledger/runs.d
    python slurm/collect_ledger.py --dir ledger/runs.d.smoke --dry-run
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="ledger/runs.d", help="directory of per-run json files")
    ap.add_argument("--ledger", default="ledger/runs.jsonl", help="target jsonl")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src = REPO_ROOT / args.dir
    dst = REPO_ROOT / args.ledger
    if not src.is_dir():
        print(f"nothing to merge: {src} does not exist", file=sys.stderr)
        return 0

    seen = set()
    if dst.exists():
        for line in dst.read_text().splitlines():
            if line.strip():
                seen.add(json.loads(line).get("run_id"))

    new, skipped, bad = [], 0, 0
    for f in sorted(src.glob("*.json")):
        try:
            rec = json.loads(f.read_text())
        except json.JSONDecodeError as e:
            print(f"SKIP (malformed) {f.name}: {e}", file=sys.stderr)
            bad += 1
            continue
        if rec.get("run_id") in seen:
            skipped += 1
            continue
        seen.add(rec["run_id"])
        new.append(rec)

    new.sort(key=lambda r: r["run_id"])
    failed = [r["run_id"] for r in new if r.get("status") != "completed"]

    print(f"{len(new)} new, {skipped} already present, {bad} malformed -> {dst}")
    if failed:
        print(f"WARNING: {len(failed)} not completed: {', '.join(failed)}", file=sys.stderr)
    if args.dry_run:
        for r in new:
            print(f"  {r['run_id']:28s} {r.get('status'):10s} {r.get('wall_clock_s')}s")
        return 0

    if new:
        with dst.open("a") as fh:
            for r in new:
                fh.write(json.dumps(r) + "\n")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
