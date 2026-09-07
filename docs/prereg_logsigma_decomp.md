# Preregistration — per-term log-sigma gradient decomposition

Written and committed **before** any number below was computed. Append-only.

Question: does the weighted-MLE arm's pre-tanh width excess come from a Q-weighted
**displacement** term that the pathwise arm structurally lacks?

Offline throughout. Frozen checkpoints and the existing neutral fixed-state bank only. No
training, no new state generation, no launches.

## Convention

`F := -dL/dlog_sigma`, so **F > 0 widens** and **F < 0 contracts**. This matches
`docs/protocol_width_populations.md` and every earlier report in this project. All
quantities are in **pre-tanh space**; `tanh` enters only as the critic's argument when
forming the weights, never inside the MLE gradient.

## The algebra being tested, fixed here

Per state, with `pi_old = N(mu_old, sigma_old^2 I)` in pre-tanh space, `d` the action
dimension, and `M` samples:

```
a_i    = mu_old + sigma_old * eps_i,        eps_i ~ N(0, I)
w_i   propto exp( Q(s, tanh(a_i)) / eta ),  normalised to sum to 1
mu_w   = sum_i w_i a_i
Dmu    = mu_w - mu_old
```

The weighted negative log-likelihood the M-step minimises, evaluated at
`(mu_old, sigma_old)`, has

```
dL/dlog_sigma = d - (1/sigma_old^2) * sum_i w_i ||a_i - mu_old||^2
```

and since `sum_i w_i (a_i - mu_w) = 0` the cross term vanishes, giving the exact split

```
sum_i w_i ||a_i - mu_old||^2 = sum_i w_i ||a_i - mu_w||^2 + ||Dmu||^2
```

so, with

```
Term A (displacement)     = ||Dmu||^2 / sigma_old^2
Term B (weighted spread)  = (1/sigma_old^2) * sum_i w_i ||a_i - mu_w||^2  -  d
```

we have `dL/dlog_sigma = -(A + B)` and therefore

```
Net WML log-sigma gradient (descent, widening-positive) = A + B
```

**`Term A >= 0` identically**, being a squared norm over a positive scale. It can only
push toward wider. `Term B` is negative under uniform weights, since the weighted spread
of `M` samples about their own weighted mean is biased below `d * sigma_old^2`. The two
compete, which is why the **net** sign is measured and not predicted.

Pathwise comparators, same convention, same states:

```
entropy ON  common term    = alpha * dH/dlog_sigma, the term both arms would carry
entropy OFF curvature residual = -E_eps[ sigma_old * grad_a Q . eps ]
```

The curvature residual is the pathwise arm's only route to a log-sigma gradient once the
entropy term is removed. By Stein's lemma its expectation is a curvature quantity, so it
is expected to be small and sign-varying rather than systematically positive.

## Fixed settings

* **States.** The canonical **neutral** fixed-state bank per task, confirmed to be the
  balanced neutral bank (equal states per arm, all 8 seeds, depth-stratified) and **not**
  training states. Bank sha256 verified in-run before use.
* **Checkpoints.** Per task and arm, the **final** critic and actor checkpoints, seeds
  301-308. All hashes recorded.
* **pi_old.** Per state, the pre-tanh `(mu_old, sigma_old)` of that arm's own final policy.
* **eta.** From the actual MPO dual at `eps_E = 0.5`, solved per state set by the same
  golden-section routine on `log eta` used in
  `scripts/analysis/walker_eps_counterfactual.py`, clipped to `[1e-4, 10]`.
* **M.** Both `M = 16` and `M = 32`, reported separately. The M-dependence of A and B is
  reported; A is expected to fall with M as the weighted mean concentrates, B to approach
  `-d/M`.
* **ESS.** `1 / sum_i w_i^2` per state.
* **RNG.** Fixed and documented in the script; the same `eps_i` draws are reused across
  arms within a state so the comparison is paired.

## Gate definition, fixed here

`trust-region KL` is the **forward** KL from `pi_old` to the proposed weighted-MLE fit,
computed analytically in pre-tanh space and summed over action coordinates:

```
pi_new = N(mu_w, sigma_w^2), sigma_w^2 = sum_i w_i (a_i - mu_w)^2   per coordinate
KL(pi_old || pi_new) summed over coordinates
```

```
gate-open   = trust-region KL <  0.10   (the improvement term is applied)
gate-closed = otherwise
```

`0.10` is the executed `kl_bound` for all three tasks. The gate-open fraction is expected
to land near **0.45-0.53**, matching the logged `fr_gate_operator` in the training runs; a
materially different fraction is reported as a discrepancy rather than quietly accepted.

Every quantity is reported for **gate-open**, **gate-closed**, and **pooled**.

## Reported quantities

Per task (Walker, G1, LEAP) and arm:

* median `A`, median `B`, median net WML gradient; gate-open and pooled
* pathwise entropy-ON common term and entropy-OFF curvature residual
* `Spearman(A, ||Dmu||)` per task
* bootstrap CI by resampling **states**, 10,000 resamples, on (i) the gate-open median `A`,
  and (ii) `median A_WML - median |curvature residual|_PW`, reporting whether each excludes 0
* M-dependence of `A` and `B` across `M in {16, 32}`

## Preregistered prediction

* **P1** — gate-open median `Term A > 0` with a bootstrap CI excluding 0, in **at least 2
  of 3** tasks.
* **P2** — `Spearman(A, ||Dmu||) > 0` in **at least 2 of 3** tasks.
* **P3** — WML `Term A` systematically exceeds the PW curvature residual in magnitude, and
  the PW residual is **not** systematically positive.

**Measured, not predicted:** the sign of the net WML gradient, since `B` competes with `A`;
and whether `A` ranks across tasks in the same order as the training width gap.

## Decision rule, fixed before results

```
Mechanism SUPPORTED   iff P1 and P2 and P3 all hold.
Mechanism NOT SUPPORTED if Term A is indistinguishable from 0 gate-open,
                        or if PW shows a comparable systematic positive residual.
```

The **sign of the net gradient does not enter the verdict**. A negative net with a clearly
positive `A` still supports the mechanism, because the claim is about the existence and
size of a displacement term the pathwise arm lacks, not about which term wins.

## What this design cannot show

It evaluates gradients at frozen final checkpoints on a fixed state bank. It is a statement
about the decomposition of one update's log-sigma gradient at those points, not a causal
account of how the trained width arose over 400 iterations, and no such causal claim is
made. Term A being positive does not by itself explain the observed width gap.
