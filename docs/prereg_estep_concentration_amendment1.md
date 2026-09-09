# Amendment 1 to docs/prereg_estep_concentration.md — baseline instrumentation reruns

Appended before launch. Does not alter any registered prediction, endpoint,
decision rule or interpretation clause in the parent document (commit `4556b6d`).

## What is added

`docs/prereg_estep_concentration.md` Sec. 2 registered a LEAN scope of 24 runs and
stated that **no baseline reruns are registered**. This amendment adds 48 runs:
`PW_noent` and `WML_noent` at `eps_e = 0.5`, on Walker, G1 and LEAP, seeds 301-308,
executed on the KL-split source at commit `964b251`.

## Why

The parent document's own Sec. 2 gives the reason and predicted the need:

> no existing run carries the KL split, and verdict branch (c) names the KL width
> part as the candidate residual asymmetry, so branch (c) is untestable without it.

That is now the live question. The `eps_e = 0.1` arm logs
`fr_kl_mean_part_med` and `fr_kl_width_part_med`; every `eps_e = 0.5` baseline logs
neither, so the mean-versus-width decomposition of the trust-region KL cannot be
compared between budgets or between operators. Measured in the new arm, the mean
part exceeds the width part by 25 to 33 times; whether that ratio differs between
the two operators is exactly the asymmetry the refuted displacement mechanism was
meant to explain, and it cannot be looked at with the data on hand.

## What these runs are, and are not

They are **instrumentation only**. Training is bit-identical to the executed
baselines by the bit-check already passed and recorded in the parent document
Sec. 1.3: at fixed `eps_e` the KL split draws no randomness, enters no loss, and its
inputs are `stop_gradient`ed. No scientific configuration field changes. They
produce no new policy, no new return, and no new width; the only new quantity is the
logged KL decomposition of runs that already exist.

**Export isolation.** At `eps_e = 0.5` the registered `_eps` tag suffix is empty, so
these runs would otherwise write the canonical baselines' own paths. They therefore
execute from a dedicated git worktree with a real `exports/` directory, so the
canonical tree is unreachable from the jobs. Their exports are **never** copied into
the canonical tree. The canonical baselines are hashed before and after and required
unchanged.

**Bitwise correspondence is measured, not assumed.** All 48 run on `c23g`, whereas
some of the baselines they mirror ran on `c25g`. Rather than assume the logged
trajectories correspond, each rerun's exported actor, critic and normaliser are
compared bitwise against its canonical baseline after the fact. Seeds that match
bitwise carry logged diagnostics that provably belong to the published run; seeds
that do not are reported separately and their diagnostics are treated as a
same-configuration replicate rather than as the published run's own trace.

## What is NOT changed

No prediction is added or altered. P6.1 and P6.2 stand as written. The
interpretation clauses of Sec. 7 stand as written. Nothing here licenses a claim
about the displacement-versus-curvature mechanism, which remains NOT SUPPORTED under
`5edaca3`'s rule. The KL split is a description of where the trust-region budget is
spent, not a mechanism test, and any mechanism claim would need its own
preregistration.

## Cost

48 runs. Expected wall clock per run from the executed baselines: Walker about 15
minutes pathwise and 22 weighted-MLE, LEAP about 21 and 29, G1 about 42 and 69.
Roughly 26 GPU-hours in total.
