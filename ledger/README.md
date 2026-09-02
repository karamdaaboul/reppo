# Experiment ledger

`runs.jsonl` is the machine-readable ledger. One JSON object per line, append-only.
Failed and abandoned runs are NEVER deleted; they are recorded with their status.

Namespaces (see docs/prereg_dimension_ladder.md, Amendments L.0 / L.0b):

| seed range | namespace | may be inspected? | may become confirmatory evidence? |
|---|---|---|---|
| 0--4   | retrospective / pilot        | yes (already outcome-seen) | no |
| 901    | prospective calibration      | yes, calibration fields only | no (calibration is excluded from comparisons) |
| 101--108 | RESERVED confirmatory      | not until the campaign runs | yes |
| 201+   | EXPLORATORY / ALGORITHM DEVELOPMENT | yes, freely | **never** |
| 20260902 | OFFLINE ANALYSIS (`analysis`)     | yes, freely | no (not a training run) |

Runs in the 201+ development namespace are permanently exploratory. They must
never be relabelled as confirmatory evidence.

The `analysis` namespace (`SEED_ROOT`, a date-derived integer, currently 20260902) is
for CPU-only offline analysis that trains nothing: the LQR crossover harness
(`scripts/lqr_crossover/`, see `docs/prereg_lqr_crossover.md`). These records share
`runs.jsonl` for provenance but are not training runs and carry no `return_metrics`.
The namespace exists so such work can never consume a confirmatory seed: 101--108 are
reserved, and an offline probe must not silently occupy one.

Field schema per record:

    run_id            unique string
    namespace         one of: pilot | calibration | confirmatory | development
    label             EXPLORATORY / ALGORITHM DEVELOPMENT  (development namespace only)
    task              env name
    seed              int
    git_sha           exact SHA the run launched at
    command           full command line
    config            resolved, post-merge hyperparameters that differ from base
    algorithm_version identifier for the estimator variant (A / B / C-<name>)
    changed_params    list of parameters changed vs the previous version
    reason            why the change was made
    status            launched | running | completed | failed | diverged | abandoned
    return_metrics    filled after completion
    estimator_diag    filled after completion
    wall_clock_s      seconds
    gpu               GPU model and index
    gpu_hours         wall_clock_s / 3600
