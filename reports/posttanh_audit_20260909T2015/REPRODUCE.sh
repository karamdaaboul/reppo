#!/usr/bin/env bash
# Exact reproduction. Read-only; launches nothing.
set -euo pipefail
cd /hpcwork/qzi10910/estep_wt
OUT=reports/posttanh_audit_20260909T2015
./.venv/bin/python $OUT/manifest.py      $PWD/$OUT   # Phase 2 run manifest
./.venv/bin/python $OUT/test_posttanh.py              # Phase 5 validation gates
./.venv/bin/python $OUT/analysis.py      $PWD/$OUT   # Phases 3,4,6
./.venv/bin/python $OUT/make_figure.py   $PWD/$OUT   # figure
