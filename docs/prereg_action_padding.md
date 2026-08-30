# Pre-registration: redundant-action padding on WalkerRun

Registered: 2026-08-30 (before any k=16 seed has finished; k=0 seeds 0-1 complete at registration time).

## Setup
mjx_dmc WalkerRun. The policy acts in R^(6+k); the simulator receives the first 6
coordinates (`src/env_utils/action_pad.py`). The critic consumes the full 6+k action,
the E-step samples all 6+k coordinates, the entropy bonus covers all 6+k dims.
Levels k = 0, 6, 16 (d = 6, 12, 22; d=22 matches HumanoidRun's d=21).
Arms: A-frozen (pathwise) and B-frozen (weighted_mle, eps_e=0.5, M=32, mstep_decoupled=false).
alpha frozen at 0.01528 (WalkerRun's own learned-alpha median) for ALL levels.
5 seeds per arm per level. Order of execution: k=0, then k=16, then k=6.

## Prediction
The operator gap (A-frozen minus B-frozen, final eval return) is ~0 at d=6 and
becomes increasingly negative as d grows.

## Decision rule
CONFIRMED if: gap at d=22 is negative AND its 95% bootstrap CI excludes zero AND
the gap is monotone non-increasing across d = 6, 12, 22.

REFUTED if: the CI at d=22 contains zero, OR the ordering is not monotone.

CONTAMINATED (report separately, claim neither) if: at the final checkpoint, median
sigma over the padded coordinates exceeds 1.5x median sigma over the real 6
coordinates. In that case part of any gap is entropy-driven width drift in
dimensions where Q is flat, not dimension per se.

If REFUTED: the sqrt(d) term is not doing the work, the HumanoidRun result is a
task effect, and the paper reframes around omega. This gets reported in full
alongside the other three refuted predictions.

## Reference points already measured
WalkerRun   d=6   gap +13.7  (t=+0.92)  tie
HumanoidRun d=21  gap -72.4  (t=-2.82)  pathwise ahead
