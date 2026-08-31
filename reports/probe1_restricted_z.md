# Probe 1 — restricted-$z$ oracle

Plan: `docs/prospective_padding_error_field_analysis.md` (v1.1), Probe 1. Equations and metric: `docs/wasted_step_fraction_proposition.md` (v1.1). Conventions gated by **Amendment A** (audited commit `3b96deb`).

Produced by `scripts/probe1_restricted_z.py` (measurement) and `scripts/probe1_report.py` (aggregation). This report records outcomes only; no preregistration document was edited.

## 0. Design as executed

* **Checkpoints.** The 10 exported k=16 finals at code `3b96deb`: `WalkerRun_pathwise_fa_pad16_s{0..4}_final` (arm **A**, `actor_update_mode: pathwise`, frozen $\alpha$) and `WalkerRun_weighted_mle_pad16_s{0..4}_final` (arm **B**, `actor_update_mode: weighted_mle`). $d=22$, $k=16$, $r=6$.
* **States.** 2048 visited states per checkpoint (B=256 envs, 200-step burn-in, 8 temporal chunks 25 steps apart) rolled under that checkpoint's own $\pi_{\rm old}$, with the normalizer frozen at the checkpoint statistics.
* **Restriction.** $(s,x)$ held fixed; $x$ is one fresh pre-tanh draw per state from the checkpoint law's real block, reused across every $z$-batch and repeat.
* **Reference laws.** (i) checkpoint law $z\sim\mathcal N(\mu_{z,c},\Sigma_{z,c})$, state-dependent diagonal $\Sigma_{z,c}$ read from the actor head; (ii) common standardized law $z\sim\mathcal N(0,I_k)$ pre-tanh.
* **$F(u)$.** $F(u_z)=Q_\phi(s,\tanh([x,\ \mu_z+\Sigma_z^{1/2}u_z]))$, with the $\pm(1-10^{-4})$ clip applied before the critic call **for arm B only** (Amendment A answer 1; `src/jaxrl/reppo.py:669`). $Q$ is the **live** critic's HL-Gauss categorical mean (`src/networks/jax_models.py:305-310`); no target critic exists.
* **Gradients.** Differentiation is w.r.t. the whitened $u_z$, so autodiff returns $H_z=\Sigma_z^{1/2}\nabla_zQ_\phi$ directly, tanh Jacobian included (and, in arm B, the clip's zero Jacobian where it is active).
* **Sampling.** $M=32$, $R=2048$ repeated batches per state, plus an **independent** oracle stream of $N_{\rm oracle}=32768$ draws per state for $a_z,\nu_z,D_z,h_z,C_{zz}$, so the eq-(13)/(14) comparison is not self-fulfilling. All keys are folded off a probe-only root (blake2b of purpose|checkpoint on `PRNGKey(20260831)`); no training key is touched.
* **Not computed.** $L$ — a $z$-only step has $L=1$ by construction.

## 1. Gate — eqs (13)–(14) against repeated batches

**Eq (13)** $\mathbb E[\hat a_M]=(1-1/M)a$: per state and component, $z=(\overline{\hat a_M}-(1-1/M)a)/\mathrm{SE}$ with $\mathrm{SE}^2=\mathrm{Cov}(\hat a_M)_{jj}/R+(1-1/M)^2(D_{jj}-a_j^2)/N_{\rm oracle}$ — both MC sources. A correct match gives mean 0, sd 1.

**Eq (14)** $\mathrm{Cov}(\hat a_M)=V_M$: per state, $\rho=\|\widehat{\mathrm{Cov}}-V_M\|_F/\|V_M\|_F$ against a two-sided MC floor from independent split-halves of *both* streams, $\rho_{\rm null}=\sqrt{\rho_b^2+\rho_o^2}$. A correct match gives $\rho/\rho_{\rm null}\approx1$.

Falsification columns: `mean z, no (1-1/M)` drops the eq-(13) mean factor and `ratio, drop aa^T` drops the $-(M-2)aa^\top$ term of eq (14). Both must be worse than the retained forms for the test to have discriminated.

**Stein (19)** is an independent cross-check of the coordinate convention, not part of the gate: median $\|a_z-h_z\|/\|h_z\|$, where $a_z$ comes from the ZO moments of the oracle stream and $h_z$ from autodiff through tanh. `Stein floor` is the oracle split-half MC scale of $a_z$, $\|a_z^{h_0}-a_z^{h_1}\|/2\|h_z\|$. `$V_e\le0$` counts states with a non-positive measured error-field variance (a numerical-conditioning tripwire; must be 0).

| arm | seed | law | mean $z$ | sd $z$ | med \|$z$\| | frac \|$z$\|>1.96 | mean $z$, no $(1{-}1/M)$ | med $\rho/\rho_{\rm null}$ | med $\rho$ | med $\rho_{\rm null}$ | ratio, drop $aa^\top$ | Stein (19) | Stein floor | $V_e\le0$ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 0 | ckpt | 0.00542 | 1.01 | 0.68 | 0.0514 | -0.0211 | 0.999 | 0.0994 | 0.0992 | 2.36 | 0.0231 | 0.0224 | 0 |
| A | 1 | ckpt | -0.00436 | 0.991 | 0.67 | 0.0478 | 0.0881 | 0.997 | 0.0991 | 0.0993 | 2.37 | 0.023 | 0.0224 | 0 |
| A | 2 | ckpt | -0.00845 | 1 | 0.674 | 0.0503 | -0.0642 | 1 | 0.0995 | 0.0994 | 2.36 | 0.0231 | 0.0224 | 0 |
| A | 3 | ckpt | 0.00198 | 1 | 0.675 | 0.0502 | 0.204 | 0.999 | 0.0992 | 0.0996 | 2.36 | 0.0233 | 0.0226 | 0 |
| A | 4 | ckpt | -0.000574 | 0.996 | 0.666 | 0.0506 | 0.0292 | 0.999 | 0.0994 | 0.0995 | 2.35 | 0.0231 | 0.0224 | 0 |
| B | 0 | ckpt | 0.00572 | 0.998 | 0.672 | 0.0499 | -0.0648 | 0.997 | 0.102 | 0.102 | 2.08 | 0.0242 | 0.0241 | 0 |
| B | 1 | ckpt | 0.00456 | 0.994 | 0.671 | 0.048 | -0.00996 | 1 | 0.1 | 0.101 | 2.17 | 0.0241 | 0.0236 | 0 |
| B | 2 | ckpt | 0.00554 | 1 | 0.67 | 0.0496 | -0.0269 | 0.999 | 0.102 | 0.103 | 2.07 | 0.0244 | 0.0241 | 0 |
| B | 3 | ckpt | 0.00142 | 0.999 | 0.669 | 0.0499 | 0.00562 | 1 | 0.103 | 0.103 | 2.04 | 0.0251 | 0.0246 | 0 |
| B | 4 | ckpt | -0.00145 | 0.991 | 0.669 | 0.0482 | 0.024 | 0.997 | 0.102 | 0.102 | 2.08 | 0.0243 | 0.0241 | 0 |
| A | 0 | std | 0.0116 | 0.997 | 0.673 | 0.0483 | -0.0162 | 1 | 0.0992 | 0.0988 | 2.35 | 0.0232 | 0.0225 | 0 |
| A | 1 | std | 0.00295 | 0.998 | 0.677 | 0.0495 | 0.0939 | 1 | 0.0991 | 0.0989 | 2.36 | 0.023 | 0.0226 | 0 |
| A | 2 | std | 0.00776 | 0.996 | 0.669 | 0.0497 | -0.0463 | 1 | 0.099 | 0.0991 | 2.35 | 0.0231 | 0.0224 | 0 |
| A | 3 | std | -0.00211 | 0.993 | 0.672 | 0.0488 | 0.201 | 0.999 | 0.0992 | 0.099 | 2.34 | 0.0232 | 0.0224 | 0 |
| A | 4 | std | 0.00283 | 0.997 | 0.674 | 0.0497 | 0.0336 | 1 | 0.0994 | 0.0992 | 2.34 | 0.0233 | 0.0226 | 0 |
| B | 0 | std | -0.0076 | 1 | 0.679 | 0.0508 | -0.0938 | 0.998 | 0.0991 | 0.0993 | 2.34 | 0.0234 | 0.0226 | 0 |
| B | 1 | std | -0.0112 | 1 | 0.678 | 0.051 | -0.0206 | 0.999 | 0.0991 | 0.0991 | 2.33 | 0.0236 | 0.0227 | 0 |
| B | 2 | std | 0.0107 | 1 | 0.683 | 0.048 | -0.028 | 1 | 0.0995 | 0.0994 | 2.32 | 0.0235 | 0.0225 | 0 |
| B | 3 | std | -0.00436 | 1 | 0.678 | 0.0498 | -0.00567 | 1 | 0.0993 | 0.0994 | 2.32 | 0.0236 | 0.0224 | 0 |
| B | 4 | std | 0.00758 | 1.01 | 0.682 | 0.0516 | 0.0381 | 0.997 | 0.0993 | 0.0993 | 2.33 | 0.0235 | 0.0227 | 0 |

### 1b. Sharp scalar forms

The signed mean $z$ above cannot see the $(1-1/M)$ factor: it shifts each component by $-a_j/M$, and $a_j$ changes sign across states and components, so the shift averages away. The factor is therefore estimated directly, by regressing the repeated-batch mean on the oracle $a$ across all states and components. The oracle $a$ carries MC error, which attenuates a naive slope, so the denominator is de-attenuated by the known $\mathrm{Var}(a_j)=(D_{jj}-a_j^2)/N_{\rm oracle}$; the numerator needs no correction because the two streams are independent.

Predictions: $\hat c = 1-1/M = 0.96875$ (the variant dropping the factor predicts $\hat c = 1$); $\hat\lambda=\sum\mathrm{tr}\widehat{\mathrm{Cov}}/\sum\mathrm{tr}V_M = 1$ (the variant dropping $-(M-2)aa^\top$ predicts $\hat\lambda\ne1$). Intervals are 2,000-replicate lane bootstraps within the checkpoint.

| arm | seed | law | $\hat c$ [95% CI] | $1-1/M$ in CI? | $1$ in CI? | $\hat\lambda$ [95% CI] | $1$ in CI? | $\hat\lambda$ dropping $aa^\top$ |
|---|---|---|---|---|---|---|---|---|
| A | 0 | ckpt | 0.97036 [0.96846, 0.97217] | yes | **no** | 1.0005 [0.9984, 1.0029] | yes | 0.9492 [0.9471, 0.9514] |
| A | 1 | ckpt | 0.96906 [0.96832, 0.96993] | yes | **no** | 0.9991 [0.9974, 1.0006] | yes | 0.9475 [0.9459, 0.9489] |
| A | 2 | ckpt | 0.97480 [0.96734, 0.97687] | yes | **no** | 1.0014 [0.9988, 1.0046] | yes | 0.9499 [0.9475, 0.9534] |
| A | 3 | ckpt | 0.97161 [0.96058, 0.98379] | yes | **no** | 1.0010 [0.9818, 1.0175] | yes | 0.9558 [0.9342, 0.9762] |
| A | 4 | ckpt | 0.96818 [0.96625, 0.97115] | yes | **no** | 0.9983 [0.9955, 1.0025] | yes | 0.9473 [0.9445, 0.9514] |
| B | 0 | ckpt | 0.96888 [0.96782, 0.97008] | yes | **no** | 1.0003 [0.9990, 1.0018] | yes | 0.9547 [0.9535, 0.9559] |
| B | 1 | ckpt | 0.96876 [0.96790, 0.96976] | yes | **no** | 0.9998 [0.9980, 1.0021] | yes | 0.9529 [0.9512, 0.9552] |
| B | 2 | ckpt | 0.96977 [0.96771, 0.97165] | yes | **no** | 1.0000 [0.9974, 1.0025] | yes | 0.9545 [0.9518, 0.9570] |
| B | 3 | ckpt | 0.96894 [0.96737, 0.97058] | yes | **no** | 1.0006 [0.9987, 1.0028] | yes | 0.9564 [0.9543, 0.9587] |
| B | 4 | ckpt | 0.96913 [0.96843, 0.96981] | yes | **no** | 0.9995 [0.9984, 1.0004] | yes | 0.9535 [0.9524, 0.9544] |
| A | 0 | std | 0.96942 [0.96754, 0.97150] | yes | **no** | 1.0002 [0.9977, 1.0028] | yes | 0.9492 [0.9466, 0.9521] |
| A | 1 | std | 0.96864 [0.96769, 0.96948] | yes | **no** | 0.9995 [0.9982, 1.0006] | yes | 0.9482 [0.9470, 0.9493] |
| A | 2 | std | 0.95436 [0.94987, 0.96839] | **no** | **no** | 0.9798 [0.9736, 0.9993] | **no** | 0.9296 [0.9236, 0.9485] |
| A | 3 | std | 0.97617 [0.96816, 0.98092] | yes | **no** | 1.0221 [0.9995, 1.0372] | yes | 0.9769 [0.9487, 0.9953] |
| A | 4 | std | 0.97228 [0.96738, 0.97644] | yes | **no** | 1.0055 [0.9994, 1.0098] | yes | 0.9544 [0.9486, 0.9583] |
| B | 0 | std | 0.96895 [0.96810, 0.96989] | yes | **no** | 0.9999 [0.9986, 1.0013] | yes | 0.9487 [0.9474, 0.9501] |
| B | 1 | std | 0.96785 [0.96620, 0.96983] | yes | **no** | 1.0004 [0.9981, 1.0042] | yes | 0.9493 [0.9470, 0.9529] |
| B | 2 | std | 0.96849 [0.96709, 0.96994] | yes | **no** | 0.9976 [0.9956, 0.9999] | **no** | 0.9475 [0.9456, 0.9495] |
| B | 3 | std | 0.96899 [0.96781, 0.97013] | yes | **no** | 0.9984 [0.9966, 1.0001] | yes | 0.9480 [0.9462, 0.9497] |
| B | 4 | std | 0.96832 [0.96765, 0.96896] | yes | **no** | 0.9992 [0.9977, 1.0005] | yes | 0.9484 [0.9470, 0.9496] |

## 2. Error field: $V_e$, $G_z$, $\Omega_z$

Per-state values first; the arm row is the seed median of per-seed state medians, with a 10,000-replicate hierarchical bootstrap 95% interval (`default_rng(20260831)`; seeds, then whole lanes within seed, temporal chunks kept together). $V_e=\mathbb E[\tilde e_z^2]$, $G_z^2=\mathbb E\|\Sigma_z^{1/2}\nabla_zQ_\phi\|^2$, $\Omega_z=G_z/\sqrt{V_e}$ (eqs 8–9). $\Omega_z$ is the centered padded-subspace frequency, not the manuscript's full $\omega$.

| law | arm | seed | med $V_e$ | med $G_z$ | med $\Omega_z$ | med $\bar\sigma_z$ |
|---|---|---|---|---|---|---|
| checkpoint law | A | 0 | 0.0001608 | 0.01355 | 1.07 | 0.8703 |
| checkpoint law | A | 1 | 0.0001987 | 0.01506 | 1.069 | 0.8711 |
| checkpoint law | A | 2 | 0.0001976 | 0.01503 | 1.07 | 0.872 |
| checkpoint law | A | 3 | 0.0002241 | 0.01604 | 1.071 | 0.874 |
| checkpoint law | A | 4 | 0.0001751 | 0.01417 | 1.072 | 0.871 |
| checkpoint law | B | 0 | 0.0005056 | 0.02511 | 1.112 | 0.8101 |
| checkpoint law | B | 1 | 0.0003327 | 0.0203 | 1.118 | 0.909 |
| checkpoint law | B | 2 | 0.0005873 | 0.02705 | 1.119 | 0.7597 |
| checkpoint law | B | 3 | 0.001379 | 0.04211 | 1.143 | 0.8509 |
| checkpoint law | B | 4 | 0.0006744 | 0.02879 | 1.104 | 0.6916 |
| N(0, I_k) | A | 0 | 0.0001866 | 0.01489 | 1.09 | 1 |
| N(0, I_k) | A | 1 | 0.0002294 | 0.0165 | 1.09 | 1 |
| N(0, I_k) | A | 2 | 0.0002284 | 0.01645 | 1.09 | 1 |
| N(0, I_k) | A | 3 | 0.0002594 | 0.01759 | 1.091 | 1 |
| N(0, I_k) | A | 4 | 0.0002029 | 0.01556 | 1.093 | 1 |
| N(0, I_k) | B | 0 | 0.001617 | 0.04383 | 1.091 | 1 |
| N(0, I_k) | B | 1 | 0.0006021 | 0.02687 | 1.091 | 1 |
| N(0, I_k) | B | 2 | 0.002114 | 0.05043 | 1.097 | 1 |
| N(0, I_k) | B | 3 | 0.005811 | 0.08375 | 1.098 | 1 |
| N(0, I_k) | B | 4 | 0.00277 | 0.05769 | 1.095 | 1 |

| law | arm | $V_e$ seed-median [95% CI] | $G_z$ seed-median [95% CI] | $\Omega_z$ seed-median [95% CI] |
|---|---|---|---|---|
| checkpoint law | A | 0.0001976 [0.000162, 0.000221] | 0.01503 [0.0136, 0.0159] | 1.07 [1.07, 1.07] |
| checkpoint law | B | 0.0005873 [0.000339, 0.00136] | 0.02705 [0.0204, 0.0416] | 1.118 [1.1, 1.14] |
| N(0, I_k) | A | 0.0002284 [0.000189, 0.000256] | 0.01645 [0.015, 0.0174] | 1.09 [1.09, 1.09] |
| N(0, I_k) | B | 0.002114 [0.000625, 0.00571] | 0.05043 [0.0273, 0.083] | 1.095 [1.09, 1.1] |

**The Poincaré bound and what the numbers do and do not show.** By the Gaussian Poincaré inequality, $\mathrm{Var}(\tilde e_z)\le\mathbb E\|\Sigma_z^{1/2}\nabla_zQ_\phi\|^2$ for any function under a Gaussian reference law, with equality iff the function is affine in $z$. $\Omega_z\ge1$ is therefore forced analytically. The finite-sample per-state check produced a minimum estimated $\Omega_z$ of **1.023** across 40,960 evaluated state-checkpoint/law cells, with zero estimates below one. That is a consistency check on the estimator, not a proof of the inequality: the inequality is a theorem, and a finite-sample estimate landing below one would have indicated Monte-Carlo error or a coding fault, not a counterexample.

**Aggregate $\Omega_z$ and its definition.** $\Omega_z=G_z/\sqrt{V_e}$ with $G_z^2=\mathbb E\|\Sigma_z^{1/2}\nabla_zQ_\phi\|^2$ and $V_e=\mathrm{Var}_z(Q_\phi)$, both taken under the stated reference law for $z$ and both in the $\Sigma_z$-whitened metric, so $\Omega_z$ is dimensionless and law-specific. Per-seed state medians span **1.07-1.14**, decomposing by law as: 1.069-1.072 (arm A, checkpoint law, $\Sigma_z=\Sigma_{z,c}$), 1.090-1.093 (arm A, standardized law, $\Sigma_z=I_k$), 1.104-1.143 (arm B, checkpoint law) and 1.091-1.098 (arm B, standardized law). Each figure is a within-law quantity; the checkpoint-law and standardized-law columns use different metrics and are not interchangeable.

Via the Hermite decomposition ($\Omega^2=\sum_k k\,a_k$, $a_k$ the variance fraction at polynomial order $k$), $\Omega_z$ bounds the linear-in-$z$ energy fraction from below by $a_1\ge2-\Omega^2$. Evaluated at each arm x law median: 85% (A, checkpoint), 81% (A, standardized), 70-78% (B, checkpoint), 80% (B, standardized). Under every law measured here the padded error field is a dominantly linear tilt at the sampling scale rather than a high-frequency wiggle.

**Aggregation discrepancy check (proposition self-check 1).** $\Omega_z^2$ is a ratio, so it is formed per state and aggregated after. The pooled energy-weighted proxy $\sqrt{\overline{G_z^2}/\overline{V_e}}$ — which the proposition forbids as a substitute — is reported beside it.

| law | arm | seed | per-state median $\Omega_z$ | pooled proxy | proxy/median |
|---|---|---|---|---|---|
| checkpoint law | A | 0 | 1.07 | 1.072 | 1 |
| checkpoint law | A | 1 | 1.069 | 1.069 | 1 |
| checkpoint law | A | 2 | 1.07 | 1.078 | 1.01 |
| checkpoint law | A | 3 | 1.071 | 1.128 | 1.05 |
| checkpoint law | A | 4 | 1.072 | 1.074 | 1 |
| checkpoint law | B | 0 | 1.112 | 1.122 | 1.01 |
| checkpoint law | B | 1 | 1.118 | 1.124 | 1.01 |
| checkpoint law | B | 2 | 1.119 | 1.145 | 1.02 |
| checkpoint law | B | 3 | 1.143 | 1.165 | 1.02 |
| checkpoint law | B | 4 | 1.104 | 1.115 | 1.01 |
| N(0, I_k) | A | 0 | 1.09 | 1.092 | 1 |
| N(0, I_k) | A | 1 | 1.09 | 1.089 | 1 |
| N(0, I_k) | A | 2 | 1.09 | 1.092 | 1 |
| N(0, I_k) | A | 3 | 1.091 | 1.168 | 1.07 |
| N(0, I_k) | A | 4 | 1.093 | 1.094 | 1 |
| N(0, I_k) | B | 0 | 1.091 | 1.091 | 1 |
| N(0, I_k) | B | 1 | 1.091 | 1.092 | 1 |
| N(0, I_k) | B | 2 | 1.097 | 1.1 | 1 |
| N(0, I_k) | B | 3 | 1.098 | 1.099 | 1 |
| N(0, I_k) | B | 4 | 1.095 | 1.094 | 1 |

## 3. Padded-block error energies

The true padded gradient is zero ($\nabla_zQ^\pi=0$), so **all** of this energy is error. Energies are in the whitened metric ($\|g\|_\Sigma^2$), $Z=\|R_z\|^2$. `pred` columns are the analytic moments: eq (7) $\mathbb E[Z_{\rm PW}]=\|h_z\|^2+\mathrm{tr}C_{zz}/M$ and eq (16) $\mathbb E[Z_{\rm ZO}]=(1-1/M)^2\|a_z\|^2+\mathrm{tr}V_M$.

| law | arm | seed | med $Z_{\rm PW}$ | pred (7) | med $Z_{\rm ZO}$ | pred (16) | med $Z_{\rm ZO}/Z_{\rm PW}$ | $\|h_z\|^2$ | $\mathrm{tr}C_{zz}/M$ | $\|a_z\|^2$ | $\mathrm{tr}V_M$ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| checkpoint law | A | 0 | 0.0001517 | 0.0001517 | 0.0002206 | 0.0002198 | 1.45 | 0.0001506 | 1.04e-06 | 0.0001503 | 7.857e-05 |
| checkpoint law | A | 1 | 0.0001871 | 0.0001871 | 0.000271 | 0.0002714 | 1.45 | 0.0001858 | 1.268e-06 | 0.0001857 | 9.699e-05 |
| checkpoint law | A | 2 | 0.0001858 | 0.000186 | 0.0002704 | 0.0002693 | 1.45 | 0.0001847 | 1.279e-06 | 0.0001841 | 9.643e-05 |
| checkpoint law | A | 3 | 0.0002117 | 0.0002112 | 0.0003062 | 0.0003052 | 1.45 | 0.0002097 | 1.489e-06 | 0.0002092 | 0.0001092 |
| checkpoint law | A | 4 | 0.0001645 | 0.0001638 | 0.0002383 | 0.0002391 | 1.45 | 0.0001627 | 1.185e-06 | 0.0001638 | 8.561e-05 |
| checkpoint law | B | 0 | 0.0004331 | 0.0004329 | 0.0006589 | 0.0006582 | 1.51 | 0.0004261 | 6.011e-06 | 0.0004259 | 0.0002582 |
| checkpoint law | B | 1 | 0.0002894 | 0.0002895 | 0.000434 | 0.0004332 | 1.49 | 0.0002853 | 3.902e-06 | 0.0002869 | 0.0001668 |
| checkpoint law | B | 2 | 0.0005087 | 0.0005084 | 0.0007686 | 0.0007689 | 1.51 | 0.0005016 | 7.039e-06 | 0.0005001 | 0.0002977 |
| checkpoint law | B | 3 | 0.001167 | 0.001166 | 0.001781 | 0.00178 | 1.52 | 0.001142 | 1.945e-05 | 0.001139 | 0.0007001 |
| checkpoint law | B | 4 | 0.0005863 | 0.0005865 | 0.0008823 | 0.0008865 | 1.51 | 0.0005786 | 7.468e-06 | 0.0005796 | 0.0003439 |
| N(0, I_k) | A | 0 | 0.000174 | 0.0001739 | 0.000252 | 0.0002534 | 1.45 | 0.0001723 | 1.544e-06 | 0.0001729 | 9.073e-05 |
| N(0, I_k) | A | 1 | 0.000214 | 0.0002142 | 0.0003117 | 0.0003101 | 1.45 | 0.0002124 | 1.869e-06 | 0.0002122 | 0.0001112 |
| N(0, I_k) | A | 2 | 0.0002126 | 0.0002123 | 0.0003085 | 0.0003094 | 1.45 | 0.0002104 | 1.873e-06 | 0.0002117 | 0.0001109 |
| N(0, I_k) | A | 3 | 0.0002408 | 0.000241 | 0.0003506 | 0.0003502 | 1.45 | 0.0002389 | 2.188e-06 | 0.0002388 | 0.0001258 |
| N(0, I_k) | A | 4 | 0.0001877 | 0.0001877 | 0.000273 | 0.0002739 | 1.45 | 0.0001859 | 1.763e-06 | 0.0001867 | 9.858e-05 |
| N(0, I_k) | B | 0 | 0.001506 | 0.001508 | 0.00218 | 0.002185 | 1.45 | 0.001494 | 1.358e-05 | 0.001494 | 0.0007848 |
| N(0, I_k) | B | 1 | 0.0005589 | 0.0005602 | 0.0008153 | 0.0008128 | 1.46 | 0.000555 | 5.384e-06 | 0.0005554 | 0.0002928 |
| N(0, I_k) | B | 2 | 0.001956 | 0.001955 | 0.002837 | 0.002842 | 1.46 | 0.001937 | 1.914e-05 | 0.00193 | 0.001027 |
| N(0, I_k) | B | 3 | 0.005334 | 0.00534 | 0.007776 | 0.007804 | 1.46 | 0.005283 | 5.326e-05 | 0.005281 | 0.002825 |
| N(0, I_k) | B | 4 | 0.002565 | 0.002565 | 0.00372 | 0.003707 | 1.45 | 0.002541 | 2.458e-05 | 0.002518 | 0.00134 |

### Per-component error energies

Median over states of $\mathbb E[R_{{\rm PW},z,j}^2]$ and $\mathbb E[\hat a_{M,j}^2]$, pooled over the 5 seeds of each arm (component $j$ indexes the padded block, $z_0\ldots z_{15}$).

**checkpoint law, arm A**

| $j$ | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PW | 4.14e-06 | 6.79e-06 | 7.7e-06 | 4.47e-06 | 5.58e-06 | 5.22e-06 | 6.17e-06 | 5.92e-06 | 8.54e-06 | 5.05e-06 | 6e-06 | 5.55e-06 | 4.09e-06 | 3.78e-06 | 4.93e-06 | 4.7e-06 |
| ZO | 1.07e-05 | 1.3e-05 | 1.33e-05 | 1.09e-05 | 1.27e-05 | 1.19e-05 | 1.3e-05 | 1.28e-05 | 1.47e-05 | 1.18e-05 | 1.24e-05 | 1.14e-05 | 1.22e-05 | 1.17e-05 | 1.22e-05 | 1.14e-05 |

**checkpoint law, arm B**

| $j$ | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PW | 8.01e-06 | 6.65e-06 | 7.01e-06 | 7.71e-06 | 7.52e-06 | 8.53e-06 | 6.83e-06 | 8.34e-06 | 7.95e-06 | 6.11e-06 | 1.04e-05 | 9.48e-06 | 6.55e-06 | 8.89e-06 | 8.21e-06 | 9.23e-06 |
| ZO | 2.96e-05 | 2.93e-05 | 3.15e-05 | 3.13e-05 | 3.51e-05 | 3.37e-05 | 3.16e-05 | 3.11e-05 | 3.24e-05 | 3.3e-05 | 3.81e-05 | 3.25e-05 | 3.11e-05 | 3.52e-05 | 3.26e-05 | 3.12e-05 |

**N(0, I_k), arm A**

| $j$ | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PW | 4.76e-06 | 7.7e-06 | 8.75e-06 | 5.17e-06 | 6.33e-06 | 5.96e-06 | 7.09e-06 | 6.83e-06 | 9.7e-06 | 5.81e-06 | 6.72e-06 | 6.32e-06 | 4.74e-06 | 4.39e-06 | 5.61e-06 | 5.38e-06 |
| ZO | 1.22e-05 | 1.5e-05 | 1.52e-05 | 1.26e-05 | 1.46e-05 | 1.36e-05 | 1.5e-05 | 1.48e-05 | 1.7e-05 | 1.37e-05 | 1.43e-05 | 1.31e-05 | 1.43e-05 | 1.35e-05 | 1.39e-05 | 1.29e-05 |

**N(0, I_k), arm B**

| $j$ | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PW | 5.93e-05 | 5.28e-05 | 6.98e-05 | 5.16e-05 | 6.57e-05 | 7.8e-05 | 6.84e-05 | 6.03e-05 | 9.32e-05 | 8.44e-05 | 6.28e-05 | 5.76e-05 | 5.75e-05 | 6.75e-05 | 6.01e-05 | 7.07e-05 |
| ZO | 0.000127 | 0.000132 | 0.000141 | 0.000123 | 0.000132 | 0.000133 | 0.000144 | 0.000141 | 0.000173 | 0.000162 | 0.000147 | 0.000131 | 0.000141 | 0.000143 | 0.000137 | 0.000144 |

### Interpretation - where this probe sits relative to Claim 4

A dominantly linear $\tilde e_z$ is the low-$\omega$ regime of Claim 4: smooth error, where the pathwise estimator inherits the error robustly and the zeroth-order estimator's smoothing advantage is nil. Separately from that linearity finding - and **not** derived from $\Omega_z$, which is the centered padded-error frequency and carries no information about estimator variance - the ZO estimator also pays a large **ZO-to-pathwise noise-energy ratio** in the padded block.

That ratio is $\mathrm{tr}(V_M)\,/\,[\mathrm{tr}(C_{zz})/M]$. The numerator is the trace of the finite-$M$ covariance of the canonical ZO estimator $\hat a_M$, i.e. eq (14); the denominator is the pathwise estimator's sampling-noise energy at the same $M$, i.e. the $\mathrm{tr}(C_{zz})/M$ term of eq (7).

$C_{zz}$ is defined **in the whitened metric**: with $H(u)=\Sigma^{1/2}\nabla_yQ_\phi(s,\tanh(\mu+\Sigma^{1/2}u))$ and $C=\operatorname{Cov}(H)$ (proposition Sec. 2), $C_{zz}$ is the padded-block diagonal sub-block of that whitened covariance. $V_M$ is likewise already a whitened-coordinate quantity, since $R_{\rm ZO}=\Sigma^{1/2}\hat g_{\rm ZO}=\hat a_M$. Numerator and denominator are therefore in the same metric and the ratio is dimensionless.

**Scope of the ratio.** It compares **covariance (sampling-noise) energy only**. It excludes the squared estimator bias of each channel — $\|h_z\|^2$ for pathwise and $(1-1/M)^2\|a_z\|^2$ for ZO, the terms that sit alongside the noise terms in eqs (7) and (16) — and it is therefore **not** a total-MSE ratio between the two estimators. Against the true padded gradient $\nabla_zQ^\pi=0$ those bias terms are themselves error and dominate both channels here, so the noise ratio below characterises one component of the comparison, not the whole of it.

Seed medians, reported per checkpoint arm and per reference law:

| arm | reference law | ZO-to-pathwise noise-energy ratio |
|---|---|---|
| A (pathwise) | checkpoint | **75.5** (range 73.0-76.4) |
| A (pathwise) | standardized $\mathcal N(0,I_k)$ | **58.8** (range 57.1-59.4) |
| B (weighted-MLE) | checkpoint | **41.6** (range 33.5-44.9) |
| B (weighted-MLE) | standardized $\mathcal N(0,I_k)$ | **55.6** (range 53.2-60.0) |

Only with those four values on the table is a joint summary meaningful: across arms and laws the ratio spans roughly **42-76x**. No single value should be quoted as if it applied to both arms - arm A under its own law is ~75x while arm B under its own law is ~42x, and the two are measured under different $\Sigma_z$.

Taken together, this probe lands in the corner of the phase diagram where both terms favor pathwise; it does not exercise the high-$\omega$ corner where the crossover would reverse. That is a statement about these trained critics, not a validation of the full crossover claim.

## 4. Saturation covariate

Per batch, the fraction of the $M\times k$ padded post-tanh coordinates with $|\tanh z|\ge 1-10^{-4}$ (the arm-B clip boundary) and with $|\tanh z|>0.99$. `corr` columns are the within-state correlation across the $R$ batches between that fraction and the batch's padded error energy.

| law | arm | seed | med sat (clip) | mean sat (clip) | med sat (>0.99) | med corr(sat, $Z_{\rm ZO}$) | med corr(sat, $Z_{\rm PW}$) | zero-step frac ZO | zero-step frac PW |
|---|---|---|---|---|---|---|---|---|---|
| checkpoint law | A | 0 | 0 | 1.4e-09 | 0.00248 | 0.00221 | -0.00288 | 0 | 0 |
| checkpoint law | A | 1 | 0 | 2.79e-09 | 0.00251 | 0.0195 | -0.0192 | 0 | 0 |
| checkpoint law | A | 2 | 0 | 1.4e-09 | 0.00253 | -0.0181 | 0.0263 | 0 | 0 |
| checkpoint law | A | 3 | 0 | 1.4e-09 | 0.00259 | -0.00903 | 0.00651 | 0 | 0 |
| checkpoint law | A | 4 | 0 | 1.86e-09 | 0.0025 | -0.00801 | 0.00899 | 0 | 0 |
| checkpoint law | B | 0 | 0.000213 | 0.000717 | 0.0861 | 0.00773 | -0.00765 | 0 | 0 |
| checkpoint law | B | 1 | 0.000168 | 0.000569 | 0.0465 | 0.00899 | -0.00918 | 0 | 0 |
| checkpoint law | B | 2 | 0.000194 | 0.000969 | 0.091 | 0.0091 | -0.00644 | 0 | 0 |
| checkpoint law | B | 3 | 0.000711 | 0.00203 | 0.137 | 0.0119 | -0.00902 | 0 | 0 |
| checkpoint law | B | 4 | 0.000231 | 0.00104 | 0.114 | 0.00807 | -0.00804 | 0 | 0 |
| N(0, I_k) | A | 0 | 9.54e-07 | 6.98e-07 | 0.00813 | -0.00189 | -0.00177 | 0 | 0 |
| N(0, I_k) | A | 1 | 9.54e-07 | 7.45e-07 | 0.00813 | -0.00144 | -0.000682 | 0 | 0 |
| N(0, I_k) | A | 2 | 9.54e-07 | 7.03e-07 | 0.00813 | -0.00137 | -0.00141 | 0 | 0 |
| N(0, I_k) | A | 3 | 9.54e-07 | 7.26e-07 | 0.00813 | -0.00167 | -0.0026 | 0 | 0 |
| N(0, I_k) | A | 4 | 9.54e-07 | 7.14e-07 | 0.00813 | -0.000681 | -0.0014 | 0 | 0 |
| N(0, I_k) | B | 0 | 9.54e-07 | 7.05e-07 | 0.00813 | -0.00196 | -0.00159 | 0 | 0 |
| N(0, I_k) | B | 1 | 9.54e-07 | 7.17e-07 | 0.00813 | -0.00036 | -0.0015 | 0 | 0 |
| N(0, I_k) | B | 2 | 9.54e-07 | 7.15e-07 | 0.00813 | -0.000919 | -0.00125 | 0 | 0 |
| N(0, I_k) | B | 3 | 9.54e-07 | 7.23e-07 | 0.00813 | -0.0016 | -0.00143 | 0 | 0 |
| N(0, I_k) | B | 4 | 9.54e-07 | 7.39e-07 | 0.00813 | -0.00228 | -0.00151 | 0 | 0 |

## 5. Notes, deviations, and threats to validity

**(a) Correction to an Amendment A recordable — effective `min_std` is 0.1, not 0.0.** Amendment A records "$\sigma=e^{\log\sigma}+\text{min\_std}$ with `actor_min_std: 0.0` in the padded runs". The runs' Hydra configs do set `actor_min_std: 0.0`, but that knob is never plumbed through: `make_init` constructs `SACActorNetworks` without passing `min_std` (`src/jaxrl/reppo.py:281-293`), so the class default `min_std = 0.1` (`src/networks/jax_models.py:336,466`) is what actually trained — as `scripts/export_ckpt.py:31-35` already documents, and as the exported `meta.json` records (`actor_kwargs.min_std = 0.1`). This probe uses the effective 0.1. It matters here because $\Sigma_z$ is the checkpoint reference law and carries the additive floor into every $\Sigma$-metric quantity. This is a factual convention, not an analytic choice, and it is recorded here rather than by editing the preregistration.

**(b) Choice of $x$.** The plan fixes $(s,x)$ but does not say how $x$ is chosen. Used here: one fresh pre-tanh draw per state from the checkpoint law's real block, held fixed across every $z$-batch and every repeat of that state. The alternative ($x=\mu_x$) would condition on a measure-zero point the operator never samples.

**(c) Numerical conditioning.** The padded error field is $O(10^{-2})$ on a critic whose output is $O(50)$, so the one-pass form $\nu=\mathbb E[Q^2]-\mathbb E[Q]^2$ cancels ~7 significant digits and returns **negative** variances in float32 — it did so for 808/2048 states on a first pass. All Q-moments are therefore accumulated around a per-state reference mean from an independent pre-pass, and reduced in float64 on the host. The `$V_e\le0$` column of the gate table is the standing tripwire. Any later probe reusing these quantities must centre the same way.

**(d) Independence regime (Amendment A answer 3).** Amendment A restricts eq (14)'s repeated-batch use to across-epoch or offline re-drawn batches, because in training the E-step key is a closure constant across the minibatch scan. This probe draws every batch offline from fresh keys, so it sits squarely in the regime where (14) applies; nothing here tests the within-epoch reuse.

**(e) Bootstrap.** Two levels (seeds, then whole lanes within seed with temporal chunks kept together). The plan's third level — batches/$z$-samples within lane — is omitted because each per-state value already averages 2048 batches or 32768 oracle draws, whose residual MC scale is visible in the gate table's split-half floors and is orders of magnitude below the across-state spread.

**(f) Scope.** Five seeds per arm are not population evidence. This probe identifies the centered $z$-varying error field only: it says nothing about constant-in-$z$ critic bias, about full $e=Q_\phi-Q^\pi$, or about returns. $\Omega_z$ is the centered padded-subspace frequency, not the manuscript's $\omega$, and no RMS reading of the sup-norm Claim 4 is implied.

## 6. C5 recheck under the safe numerical path

Because the conditioning failure in §5(c) would have been invisible in aggregate, the two C5 within-state Q-spread cells that the padding thread leans on were recomputed under the safe path: per-state reference-mean centring from an independent pre-pass, accumulation in float64. Protocol otherwise identical to `scripts/probe_ckpt.py` — B=256, 200-step burn-in, 8 chunks 25 steps apart (2048 states), M=32 actions per state, `PRNGKey(0)`, same key-split order and the same `fold_in(key, 7)` action draw. Script: `scripts/c5_float64_recheck.py`.

Both paths consume the **identical** Q draws, so the old-vs-new difference carries no Monte-Carlo component whatsoever: it is pure float32 error.

| cell | quantity | original C5 | recheck (float64) | rel. diff | max per-state rel. diff |
|---|---|---|---|---|---|
| A-frozen s0 | $sd_{\rm all}$ | ≈0.044 | 0.0439397 | -4.4e-08 | 1.3e-07 |
| A-frozen s0 | $sd_{\rm real}$ | ≈0.041 | 0.0412751 | +3.0e-08 | 1.5e-07 |
| A-frozen s0 | $sd_{\rm pad}$ | ≈0.014 | 0.0135142 | +7.4e-09 | 1.3e-06 |
| B-frozen s1 | $sd_{\rm all}$ | ≈0.078 | 0.0782136 | -6.6e-08 | 1.3e-07 |
| B-frozen s1 | $sd_{\rm real}$ | ≈0.074 | 0.0734642 | +3.9e-08 | 3.1e-07 |
| B-frozen s1 | $sd_{\rm pad}$ | ≈0.017 | 0.0170027 | -6.4e-08 | 2.6e-07 |

Zero non-finite values and zero negative variances in either cell. **The C5 amplitudes are unaffected and no downstream number in the padding thread needs revisiting.** The reason C5 was safe while the first Probe 1 pass was not: `jnp.std` is a two-pass algorithm (mean first, then deviations) and never forms $\mathbb E[Q^2]-\mathbb E[Q]^2$. The corruption was specific to the one-pass raw-moment accumulator over 32768 samples, which C5 does not use.

