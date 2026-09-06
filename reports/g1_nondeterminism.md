# G1 is not run-to-run deterministic, and what that costs

Read-only finding, produced while gating the G1 entropy factorial. No G1 factorial run
has been launched.

## The measurement

`slurm/g1_determinism_probe.sh`, job `3763864`, commit `aa5533d`, node n23g0004. Two runs
of the **same commit** with the **same config and same seed**, 2,621,440 steps,
`actor_update_mode=pathwise`:

| file | verdict | max abs diff |
|---|---|---|
| `actor.npz` | DIFFERS | 1.298e-01 |
| `critic.npz` | DIFFERS | 6.654e-01 |

Final returns: **-2.236** and **-2.275**.

```
G1_SAME_CODE_REPRODUCIBLE = NO
```

## Why it matters for the entropy factorial gate

The end-to-end T1 no-op test at d=29 (`slurm/logs/t1_g1_3761668.out`, job `3761668`)
reported:

| comparison | actor max abs diff | critic max abs diff |
|---|---|---|
| T1: pre-change vs new code, both flags OFF | 1.414e-01 | 5.194e-01 |
| control: same code, run twice | 1.298e-01 | **6.654e-01** |

The flags-off difference is the same size as pure run-to-run noise and **smaller on the
critic**. T1's premise -- that a bitwise difference indicates a code change -- does not
hold on this task, so its FAIL carries no information about the intervention. This is a
property of G1, not of the flags: the identical code is an exact bitwise no-op on Walker
(T1 `0.000e+00`), and the Walker seed-301 restoration reproduced bit-exactly across four
days and a different node.

The no-op was therefore established by a test that is deterministic by construction.

## T1b, the fixed-batch no-op test

`scripts/analysis/t1b_fixed_batch.py` with `slurm/t1b_run.sh`, job `3764762`.

One batch is collected once and dumped. Both code versions then load **that same batch**
and the same PRNG keys, rebuild the same initial train state, and run one real
`learn_step` -- reppo's own actor and critic loss, reached through the closure rather
than reimplemented. No environment stepping enters the comparison, so G1's
nondeterminism is excluded by construction.

```
initial actor params    IDENTICAL over 12 leaves   max|diff| = 0.000e+00
initial critic params   IDENTICAL over 19 leaves   max|diff| = 0.000e+00
post-step actor params  IDENTICAL over 12 leaves   max|diff| = 0.000e+00
post-step critic params IDENTICAL over 19 leaves   max|diff| = 0.000e+00

INITIAL_STATE_IDENTICAL = PASS
T1b_FIXED_BATCH_NOOP    = PASS
```

The initial-state check matters: without it a pass could mean the two sides never
differed in the first place. The trees were confirmed distinct independently, the harness
printing `flags in this tree: False` for the pre-change worktree and `True` for the new one.

With T2-T5 also passing at d=29 (`3761370`: 2.151e-16, 7.081e-15, 0.000e+00, 2.302e-04,
alongside a Walker regression reproducing its validated values exactly), the no-op is
established for G1 on fixed data at three independent levels: loss algebra, gradient
routing, and a full parameter update.

## The wider consequence, which is not about this factorial

**G1's frozen confirmatory result assumes a within-seed reproducibility that G1 does not
have.** `reports/corrected_operator_replication.md` reports G1 `PW-1 minus WML-32` as
`+9.51`, 95% CI `[+2.78, +15.28]`, 8/8 positive, from **one run per seed**. If rerunning
the same seed with the same code lands somewhere else, then part of what the paired design
attributes to the operator is within-seed noise, and the interval is narrower than the
true uncertainty.

What is and is not known:

* **Known:** at 2,621,440 steps, two identical-code runs differ by 0.039 in return on a
  scale of about -2.25, roughly 1.7% relative.
* **Not known:** the size of this at the confirmatory budget of 52,297,728 steps, which is
  20x longer and where the divergence has far more opportunity to compound. It could be
  smaller in relative terms if training contracts toward an attractor, or much larger.
* **Not known:** whether Walker's determinism is a property of the task or of its specific
  configuration. Walker was bit-reproducible across days and nodes, so the pipeline is
  capable of determinism; G1 is a contact-rich MJX humanoid and Walker is not.

Quantifying it needs replicate runs at a fixed seed under distinct export tags at the full
budget. That is a separate experiment and is not proposed here.

Until it is measured, G1 paired-seed intervals should be read as **lower bounds on
uncertainty**, and any G1 claim that rests on a narrow interval or on an 8/8 sign count
should say so. The Walker results are unaffected.
