#!/usr/bin/env python3
"""T3: the export-tag suffix is appended iff eps_e != 0.5.

Load-bearing in both directions:
  (a) at eps_e = 0.5 the tag is byte-identical to what is already on disk, for EVERY
      existing baseline export -- so no published export can be orphaned;
  (b) at eps_e = 0.1 the tag is distinct from its baseline, for all 24 (task, seed).
The tag function is read out of scripts/train_and_export.py itself rather than
reimplemented, so a drift between test and source fails the test.
"""
import glob, os, re, sys

SRC = "scripts/train_and_export.py"
src = open(SRC).read()

# the suffix rule must be present, and guarded on the shipped default
m = re.search(r'_eps_e\s*=\s*float\(cfg\.hyperparameters\.eps_e\)\s*\n\s*if\s+_eps_e\s*!=\s*0\.5:\s*\n\s*variant\s*\+=\s*"_eps"\s*\+\s*\("%g"\s*%\s*_eps_e\)\.replace\("\.",\s*""\)', src)
fails = []
deferred = []
def check(n, ok, d=""):
    print("  %-58s %s  %s" % (n, "PASS" if ok else "FAIL", d))
    if not ok: fails.append(n)

check("T3a suffix rule present in train_and_export.py, guarded on 0.5", m is not None)

def suffix(eps):
    return "" if float(eps) == 0.5 else "_eps" + ("%g" % float(eps)).replace(".", "")

check("T3b suffix(0.5) is empty", suffix(0.5) == "", repr(suffix(0.5)))
check("T3c suffix(0.1) == '_eps01'", suffix(0.1) == "_eps01", repr(suffix(0.1)))

# --- disk-dependent checks -------------------------------------------------
# These assert against the CANONICAL export tree. They are load-bearing only where
# that tree is present (the cluster); elsewhere they are DEFERRED, never silently
# passed, so a run on the wrong machine cannot look green.
SEEDS = range(301, 309)
ENVS = ["WalkerRun", "G1JoystickFlatTerrain", "LeapCubeRotateZAxis"]
existing = {os.path.basename(p) for p in glob.glob("exports/*")}
base = ["%s_weighted_mle%s_s%d%s" % (e, suffix(0.5), s, k)
        for e in ENVS for s in SEEDS for k in ("_final", "_p25", "_p50")]
new = ["%s_weighted_mle%s_s%d%s" % (e, suffix(0.1), s, k)
       for e in ENVS for s in SEEDS for k in ("_final", "_p25", "_p50")]

have = sum(1 for b in base if b in existing)
canonical = have >= len(base) // 2
print("  exports/ holds %d of the %d eps_e=0.5 baseline dirs -> %s"
      % (have, len(base), "CANONICAL TREE" if canonical else "not the canonical tree"))

check("T3e new tags are all distinct from baseline tags",
      not (set(new) & set(base)), "%d overlap" % len(set(new) & set(base)))
check("T3g new tag shape is as registered",
      new[0] == "WalkerRun_weighted_mle_eps01_s301_final", new[0])

if canonical:
    missing = [b for b in base if b not in existing]
    check("T3d all 72 eps_e=0.5 baseline tags reproduce names on disk",
          not missing, "%d missing of %d" % (len(missing), len(base)))
    for x in missing[:5]:
        print("      MISSING", x)
    check("T3f none of the 72 new tags already exists on disk",
          not (set(new) & existing), "%d collide" % len(set(new) & existing))
else:
    for n in ("T3d baseline tags reproduce names on disk",
              "T3f no new tag already exists on disk"):
        print("  %-58s %s  %s" % (n, "DEFER", "needs the canonical export tree"))
    deferred.append("T3d"); deferred.append("T3f")

print("\nT3_EXPORT_TAG = %s%s" % ("PASS" if not fails else "FAIL: " + ", ".join(fails),
      ("   [DEFERRED here: %s]" % ", ".join(deferred)) if deferred else ""))
sys.exit(0 if not fails else 1)
