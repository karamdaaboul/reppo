# Preregistration: crossed frozen-critic dispersion analysis

**Status: IMMUTABLE once committed.** Written and committed before any dispersion
number exists. Corrections before launch go in a separate
`docs/amendment_crossed_dispersion_XX.md`, never by editing this file.

Repository `~/repos/reppo`, branch `estep-study`, HEAD at writing
`0af252de87b1cb58c2b04036da82e66729c83618`, working tree clean.

---

## 0. What this can and cannot show

**This measures sensitivity to variation in `Q_phi`, not in the critic error `e`.**
`Q^pi` is not observed. Each training operator shapes both the visited state-action
distribution (hence `Q^pi`) and the error field (hence `e`), and this design cannot
attribute a difference to either. A positive result supports **operator-critic
co-adaptation**. It does **not** establish an omega effect and does **not** test
Claim 4. Any sentence in the report implying otherwise is a defect.

This sits at the **frozen same-critic** level of the scope hierarchy: it does not
measure critic quality.

Forbidden under every outcome: return-level claims, omega claims, Claim-4 claims,
action-dimension claims, and any pooling of the corrected and legacy tiers.

## 1. Definitions and source facts

All algebra is in the **pre-squash** variable `y`.

**Confirmed from source: the checkpoint `sigma` is PRE-squash.**
`jax_models.py:431-438` builds `distrax.Transformed(distrax.Normal(loc, std),
distrax.Tanh())`, so `std` parameterises the base Normal; `gaussian()`
(`jax_models.py:465-469`) is documented *"Pre-squash Normal parameters (loc, scale) --
what MPO's KLs are defined on"*; and `reppo.py:752-754` logs `pi.distribution.scale`
with the comment *"pre-squash Gaussian std"*.

* `d` action dimension; `M` action samples per state per estimator evaluation.
* `mu, Sigma` pre-squash Gaussian mean and (diagonal) covariance of the reference law.
* `y_i = mu + Sigma^{1/2} u_i`, `u_i ~ N(0, I_d)`; `T` is `tanh`; `F_i = Q_phi(s, T(y_i))`.
* **PW-k**: `(1/k) sum_i grad_y Q_phi(s, T(y_i))`, i.e. `k` critic action-gradients
  pulled back through `T`.
* **ZO-k**: the **centered value-only** estimator
  `(1/k) sum_i (F_i - Fbar) Sigma^{-1} (y_i - mu)`, `Fbar = (1/k) sum_i F_i`.
  **No softmax. No `ubar` term.**
* **WML-k**: the **actual implemented** weighted-MLE mean update,
  `sum_i w_i (y_i - mu)` with `w_i = softmax(F_i / eta)` (`reppo.py:estep_weights`).
* `Q_PW`, `Q_WML`: critics from the pathwise and weighted-MLE arms.

## 2. `D` definition — fixed before running

**The implementation performs no trust-region normalisation of the mean update.**
Verified: a search for any normalisation of the actor update in `src/jaxrl/reppo.py`
returns only unrelated comments. The implementation's constraint is the *KL gate*
`jnp.where(kl < kl_bound, objective, kl * sg(lagrangian) * reduce_kl)` — a switch, not
a projection — and `optax.clip_by_global_norm` on the whole parameter tree, which is
not a per-state projection of `Delta mu`. So "normalise exactly as the implementation
does" **is not identifiable from the implementation**, and the normalisation below is
a definition adopted for this analysis, labelled as such.

For state `s`, reference law `(mu(s), Sigma(s))`, critic `Q`, operator `U`, and cloud
`r` of `M` innovations:

```
raw_{s,r}  = U's mean update in y-space
g_{s,r}    = raw_{s,r} / ||raw_{s,r}||_{Sigma^{-1}}         (unit whitened norm)
gbar_s     = (1/R) sum_r g_{s,r}
D_U(Q; s)  = (1/R) sum_r || g_{s,r} - gbar_s ||^2_{Sigma^{-1}}   =  1 - ||gbar_s||^2
D_U(Q)     = mean over the evaluation states of D_U(Q; s)
```

with `||v||^2_{Sigma^{-1}} = v^T Sigma^{-1} v`. Because every `g` has unit whitened
norm, `D in [0, 1]` and is purely **directional** dispersion.

**Why the normalisation constant does not matter.** Rescaling `D_U` by any constant
`c_U` that depends on the operator but not the critic cancels inside each bracket of
`I` (section 4). The manuscript's `sqrt(2 eps)` trust-region convention differs from
unit norm by exactly such a constant, so `I` is unchanged by the choice.

**Reported in the whitened metric only.** Raw-space norms are not comparable across
arms of different width.

`R = 200` independent clouds. Monte-Carlo uncertainty on every `D` is reported as the
standard error across the evaluation states of `D_U(Q; s)`, and separately as a
cloud-level jackknife over `R`.

## 3. The whitened metric and the reference law

**`Sigma` in the whitened metric is always the REFERENCE LAW's `Sigma`, never the
critic checkpoint's.** This is asserted in code and reported as an integrity check.
Within a cell the same `Sigma` is used for sampling, for `Sigma^{-1}` in ZO, and for
the norm, so the metric is consistent across critic sources by construction.

## 4. Primary statistic

```
I = log[ D_PW(Q_PW) / D_PW(Q_WML) ]  +  log[ D_WML(Q_WML) / D_WML(Q_PW) ]
      \_____________ bracket 1 _____/     \_____________ bracket 2 _____/
```

`I > 0` means each operator is more dispersed on the critic its own arm trained.

**Summing two log ratios is required, not cosmetic.** PW and WML updates are not
commensurable in absolute terms; the difference-of-differences on logs cancels
per-operator scale offsets, including the normalisation constant of section 2.

**Both brackets are reported SEPARATELY as well as summed, always.** Probe 4's signal
was asymmetric (0/5 one direction, 5/5 the other). If `I > 0` is driven by one bracket
alone that is a different phenomenon than both contributing, and the sum hides it.

### Which operators enter `I` — an ambiguity resolved explicitly

The scientific question is *"does each operator show different dispersion on critics
trained by itself"*. Only two operators trained a critic: the pathwise arm and the
weighted-MLE arm. **ZO trained no critic**, so "its own critic" is undefined for ZO
and ZO cannot enter `I`. Therefore:

* **`I_equal-query` (PRIMARY interaction)** uses `D_{PW-32}` and `D_{WML-32}`.
  Equal `M = 32` removes the sample-count confound while both operators remain ones
  that trained a critic.
* **`I_operational` (SECONDARY interaction)** uses `D_{PW-1}` and `D_{WML-32}`, the
  operators exactly as implemented.
* The task's **equal-query estimator comparison, PW-32 vs ZO-32**, is reported as a
  *separate* deliverable on both critics under both laws. It is not folded into `I`,
  and its table is separate from the operational table. The difference between the
  two is reported explicitly, as what the implementation adds beyond the estimator.

This resolution is recorded rather than settled silently.

## 5. Reference-law control — primary, run first

Dispersion depends directly on the sampling covariance, and the two arms' policies
differ substantially in width. **The entire crossed design is run TWICE:**

* **A-law**: reference `(mu, Sigma)` from the **PW** checkpoint of that seed and task.
* **B-law**: reference `(mu, Sigma)` from the **WML** checkpoint of that seed and task.

The reference law supplies the **common `mu`** required by the design; under a given
law both critic sources are evaluated at the same `mu` and `Sigma`, so critic source is
the only thing varying. **This is run before the tier breadth work, not after.**

**SECONDARY, reported separately:** each checkpoint evaluated under its **own**
`(mu, Sigma)`. This reintroduces the width confound by construction, which is why it
is secondary.

## 6. Common random numbers

Within a cell, the innovations `u_i` (shape `(R, S, M, d)`) are drawn once and
**shared bitwise across critic sources**. Asserted with `np.array_equal` and reported.
Critic source is then the only varying input.

## 7. State bank

**No suitable common bank exists.** The two banks in the repository
(`reports/artifacts/mc_oracle_state_bank.npz`, 64 states;
`mc_oracle_state_bank_p2.npz`, 256 states) are WalkerRun-only and both below the 2048
floor. Banks must therefore be collected. **Collecting states is environment
interaction, not training, and no checkpoint is written.**

Preregistered collection, one bank per task, before any bank exists:

* **Collecting policy:** a balanced mixture over **both arms and all eight seeds** of
  that task's tier — 128 states from each of the 16 `(arm, seed)` policies. No single
  arm or seed defines the bank, so critic source cannot be confounded with the state
  distribution.
* **Number of states:** `2048` per task (16 x 128).
* **Burn-in:** `50` stochastic-policy steps from `env.reset`, actions clipped to
  `+-0.999`, matching **`scripts/q_spread_from_ckpt.py:33-42`** exactly (the q-spread
  probe). That script collects **per checkpoint**; this design collects **one common
  bank per task**, because per-checkpoint states would confound critic source with
  state distribution and the design would fail.
* **Seed:** RNG root `20260904`, purpose-separated by blake2b fold-in on a string tag.
* Banks are saved with a sha256 and cited by path and hash in the report.

**Evaluation subset:** a fixed, randomly chosen `S_eval = 128` states drawn once from
the 2048-state bank per task under the same root, and used identically in **every**
cell of that task. States are repeated measures; **the seed is the independent unit.**

## 8. `eta` convention for WML

**`eta` is not identifiable from a PW checkpoint.** Verified: `eta_param` is present in
55/55 weighted-MLE checkpoints and absent from 55/55 pathwise checkpoints, because
`with_eta = (actor_update_mode == "weighted_mle")` (`reppo.py:374`).

Resolved by the most literal reading of the definition in the task, *"softmax weights
with eta from the dual"*:

* **PRIMARY:** `eta` is obtained per cloud by **solving the E-step dual**
  `g(eta) = eta * eps_e + eta * log((1/M) sum_i exp(F_i / eta))` numerically, with
  `eps_e` read from that checkpoint's `meta.json`, and clipped to the implementation's
  `[eta_min, eta_max] = [1e-4, 10.0]` (`jax_models.py:eta()`). This is uniform across
  every cell and imports nothing foreign.
* **SECONDARY sensitivity:** the checkpoint's **saved learned** `eta` where it exists
  (weighted-MLE checkpoints only). The implementation uses a *learned* `eta` trained
  toward the dual optimum, so the two differ; the gap is reported as a diagnostic.
* A saved-`eta` convention **cannot** be primary, because it is undefined for half the
  cells.

## 9. Clip convention

The source contains three distinct conventions:

| Convention | Site |
|---|---|
| **no clip** on the critic query | `reppo.py:741-744` (PW `pred_action`), `reppo.py:797` (`old_pi_action = tanh(y_i)`, faithful same-point) |
| `+-(1 - 1e-4)` | `reppo.py:767, 824, 890` (legacy E-step, reverse-KL, decoupled M-step) |
| `+-0.999` | `jax_wrappers.py:255-263` (environment `ClipAction`), `reppo.py:562` (stored transition action) |

* **PRIMARY: no clip.** `F_i = Q_phi(s, tanh(y_i))` with `tanh` unclipped. This is the
  operative convention on the corrected checkpoints, which is the primary tier, and it
  matches both the PW critic query and the faithful same-point E-step.
* **Reported sensitivity:** the same quantities under `+-(1 - 1e-4)`, which is what the
  **legacy** E-step used. Divergence between them is reported, not hidden.

## 10. Checkpoint tiers — never pooled

* **PRIMARY:** the **32 corrected checkpoints** — WalkerRun and G1JoystickFlatTerrain,
  arms `pathwise_fa` and `weighted_mle`, seeds 301-308. Verified present: 32/32.
* **SECONDARY / breadth, labelled RETROSPECTIVE:** the **legacy 64** — HopperHop,
  WalkerRun, LeapCubeRotateZAxis, G1JoystickFlatTerrain, both arms, seeds 101-108.
  Verified present: 64/64. These carry the audited construct-validity concerns
  (clipping between the two log probabilities, the KL threshold, the KL multiplier
  parameterisation). **They may replicate the interaction; they cannot establish it.**

Separate tables, separate CSVs, no pooled estimate.

## 11. Hypotheses

**P1 (primary).** Under **BOTH** reference laws, on the corrected checkpoints, the
paired median `I` is `> 0` with a 95% hierarchical bootstrap CI excluding zero.
Bootstrap **resamples seeds**, forms `I` **within each replicate**, 10,000 replicates,
`np.random.default_rng(20260904)`. Separately bootstrapped intervals are **not**
divided.

**P2 (sign consistency).** The per-seed sign count is reported for `I` and for **each
bracket separately**, under **each** law.

**P3 (kill condition).** If `sign(I)` differs between the A-law and B-law evaluations,
the finding is attributable to policy width and **NOT** to critic source. The report's
**first line** then records `REFUTED-AS-CRITIC-SOURCE`.

**P4 (breadth).** The legacy 64 are reported separately with the same statistic,
labelled retrospective. Agreement is supportive; disagreement is not fatal to P1 but
is reported prominently.

## 12. Integrity checks, all reported including failures

1. Common random numbers are bitwise identical across critic sources (`np.array_equal`).
2. ZO is centered, contains no softmax and no `ubar` term — asserted in code and by a
   unit test against a closed form.
3. The whitened metric uses the **reference law's** `Sigma`, consistently across cells.
4. The per-seed `sigma` distribution for each arm, reported **across seeds, across
   states and across coordinates separately**, **before** any median is quoted.
5. Whether any checkpoint fails a saturation sanity check under the reference law:
   reported as the fraction of `|tanh(y_i)| > 0.999` per arm and law.
6. Every failed check, including bugs found in this analysis, is reported.

## 13. Fixed constants

`M = 32` (and `M = 1` for the PW-1 operational arm); `R = 200`; `S_eval = 128`;
bank `2048` states per task; RNG root `20260904`; bootstrap
`np.random.default_rng(20260904)`, 10,000 replicates; `eps_e` and `alpha` read from
each checkpoint's `meta.json`; `eta` clip `[1e-4, 10.0]`.

## 14. Deliverables

`reports/crossed_dispersion.md` (verdict in the first three lines, then the section-0
scope disclaimer, then provenance, then tables);
`reports/artifacts/crossed_dispersion_primary.csv`;
`reports/artifacts/crossed_dispersion_legacy.csv`;
one figure of `I` per seed split by bracket, faceted by reference law and tier;
`scripts/analysis/crossed_dispersion.py`; exact reproduction commands.

---

## Design lock

Everything above is fixed at the commit that adds this file: the `D` definition and
its normalisation, the statistic `I` and which operators enter it, the reference-law
control, the tier separation, the state-bank rule and collection protocol, `R`,
`S_eval`, `M`, the RNG seeds, the checkpoint lists, the `eta` convention, the clip
convention, P1-P4, and the interpretation constraints.

---

## Addendum A1 — 2026-09-02 — bank scope per evidence tier

Append-only. No text above this line is altered.

Section 7 specifies **one bank per task**; section 10 separates the corrected
tier (seeds `301-308`) from the legacy 64 (seeds `101-108`) and forbids pooling.
WalkerRun is the one task that appears in **both** tiers, so those two clauses
under-determine which sixteen `(arm, seed)` policies collect the WalkerRun bank.

Resolved, prospectively, before any bank exists:

* The rule is **one bank per task x evidence tier**, not one bank per task.
* The reference-law gate reported in
  `reports/crossed_dispersion_walkerrun_gate.md` uses the **corrected-tier**
  WalkerRun bank: 16 policies = 2 arms x seeds `301-308`, 128 states each,
  2048 states total. This is what P1 already restricts the primary analysis to,
  so the gate's scope is unchanged by this addendum.
* A legacy-tier WalkerRun bank, if one is ever needed for P4, is a **separate**
  bank with its own collection, its own hash and its own tables. The two are
  never merged and never compared state-for-state.

Nothing else is amended. Section 7 already fixes the collection protocol used
here: burn-in `50` stochastic-policy steps from `env.reset` (line 167), `2048`
states per task as 16 x 128 (line 166), no post-hoc filtering or replacement.
