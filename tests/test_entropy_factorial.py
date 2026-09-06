#!/usr/bin/env python3
"""T2-T5 for the Walker entropy factorial. T1 is end-to-end, see slurm/t1_bitwise.sh.

T2-T5 operate on a reconstruction of the two objectives built line-by-line from
src/jaxrl/reppo.py:
    pathwise      objective = log_prob * sg(temperature) - value        (:1041 region)
    PW-H          objective = -value
    weighted_mle  objective = -sum_i w_i * logp_theta_i                 (:985)
    WML+H         objective = that + log_prob * sg(temperature)
`log_prob` is the summed log-probability of ONE fresh reparameterised sample from the
CURRENT policy, drawn before the arm branch at :786-790. Both arms reuse that same
tensor, which is what makes the entropy term "matched".

Run: .venv/bin/python tests/test_entropy_factorial.py
"""
import os, sys
import numpy as np
import jax, jax.numpy as jnp
# The identity under test is loss ALGEBRA. In float32 the check itself
# forms (ent - value) + value, and with |value| ~ 5e2 that cancellation
# costs ~6e-5, which is a property of the test's arithmetic and not of the
# intervention. x64 removes it so the algebra is tested, not the rounding.
jax.config.update("jax_enable_x64", True)

sys.path.insert(0, os.getcwd())
from scripts.load_ckpt import load

CKPT = "exports/WalkerRun_weighted_mle_s301_final"
CKPT_PW = "exports/WalkerRun_pathwise_fa_s301_final"
BANK = "reports/artifacts/walker_fixed_state_bank.npz"
M, KL_BOUND, REDUCE_KL, CLIP_A = 32, 0.1, 1.0, 1.0 - 1e-4
TOL_EXACT, TOL_GRAD = 1e-12, 0.0
fails = []

def logp_tanh(y, mu, sg):
    lp = -0.5 * ((y - mu) / sg) ** 2 - jnp.log(sg) - 0.5 * jnp.log(2 * jnp.pi)
    return lp - (2.0 * (jnp.log(2.0) - y - jax.nn.softplus(-2.0 * y)))

def logp_tanh_at_a(a, mu, sg):
    return logp_tanh(jnp.arctanh(jnp.clip(a, -CLIP_A, CLIP_A)), mu, sg)

def build(ck, bank, key):
    """Everything the arm branch sees, exactly as reppo.py computes it."""
    mu, sg = ck.policy_dist(bank)
    alpha = float(jnp.squeeze(ck.actor.temperature()))
    eta = max(float(jnp.squeeze(ck.actor.eta())), 1e-4) \
        if any("eta_param" in p for p in ck.meta["actor_leaf_paths"]) else 1.0
    kp, ko = jax.random.split(key)
    # the single fresh reparameterised sample from the CURRENT policy (:786-790)
    u = jax.random.normal(kp, mu.shape)
    y_theta = mu + sg * u
    a_theta = jnp.clip(jnp.tanh(y_theta), -CLIP_A, CLIP_A)
    log_prob = logp_tanh(y_theta, mu, sg).sum(-1)          # (B,)
    value = ck.q_scalar(bank, a_theta)                      # (B,)
    # the M E-step candidates from pi_old (here pi_old := current policy, sufficient
    # for a loss-algebra test; the identity under test does not involve pi_old)
    eps = jax.random.normal(ko, (M, *mu.shape))
    y_i = mu[None] + sg[None] * eps
    a_i = jnp.clip(jnp.tanh(y_i), -CLIP_A, CLIP_A)
    logp_i = logp_tanh_at_a(a_i, mu[None], sg[None]).sum(-1)
    q_i = jax.vmap(lambda aa: ck.q_scalar(bank, aa))(a_i)
    w_i = jax.nn.softmax(q_i / eta, axis=0)
    kl = jnp.zeros(mu.shape[0])                             # placeholder, set per test
    f64 = lambda x: jnp.asarray(x, jnp.float64)
    return dict(mu=f64(mu), sg=f64(sg), alpha=float(alpha), log_prob=f64(log_prob),
                value=f64(value), w_i=f64(w_i), logp_i=f64(logp_i), kl=f64(kl))

def objectives(B):
    ent = B["log_prob"] * B["alpha"]
    pw_base = ent - B["value"]
    pw_h = -B["value"]
    wml_base = -jnp.sum(B["w_i"] * B["logp_i"], axis=0)
    wml_h = wml_base + ent
    return ent, pw_base, pw_h, wml_base, wml_h

def gated(objective, kl, lagr=1.0):
    return jnp.where(kl < KL_BOUND, objective, kl * lagr * REDUCE_KL)

def check(name, ok, detail):
    print("  %-28s %s   %s" % (name, "PASS" if ok else "FAIL", detail))
    if not ok:
        fails.append(name)

def main():
    z = np.load(BANK)
    bank = jnp.asarray(z["states" if "states" in z.files else z.files[0]])[:512]
    ck = load(CKPT)
    B = build(ck, bank, jax.random.PRNGKey(0))
    ent, pw_base, pw_h, wml_base, wml_h = objectives(B)

    # a KL vector that puts roughly half the states on each side of the bound
    kl = jnp.linspace(0.0, 0.2, bank.shape[0])
    open_ = kl < KL_BOUND
    print("  gate-open states %d / %d" % (int(open_.sum()), open_.size))

    # T2 -----------------------------------------------------------------
    d_open = (gated(pw_base, kl) - gated(pw_h, kl))[open_]
    d_closed = (gated(pw_base, kl) - gated(pw_h, kl))[~open_]
    e2 = float(jnp.max(jnp.abs(d_open - ent[open_])))
    c2 = float(jnp.max(jnp.abs(d_closed)))
    check("T2_PW_MINUS_H", e2 <= TOL_EXACT and c2 == 0.0,
          "float64 gate-open max|diff - entropy| = %.3e (tol %.0e); gate-closed max|diff| = %.3e"
          % (e2, TOL_EXACT, c2))

    # T3 -----------------------------------------------------------------
    d_open = (gated(wml_h, kl) - gated(wml_base, kl))[open_]
    d_closed = (gated(wml_h, kl) - gated(wml_base, kl))[~open_]
    e3 = float(jnp.max(jnp.abs(d_open - ent[open_])))
    c3 = float(jnp.max(jnp.abs(d_closed)))
    check("T3_WML_PLUS_H", e3 <= TOL_EXACT and c3 == 0.0,
          "float64 gate-open max|diff - entropy| = %.3e (tol %.0e); gate-closed max|diff| = %.3e"
          % (e3, TOL_EXACT, c3))

    # T4: the term PW removes and the term WML+H adds, same state/params/RNG ------
    term_pw = pw_base - pw_h
    term_wml = wml_h - wml_base
    e4 = float(jnp.max(jnp.abs(term_pw - term_wml)))
    check("T4_MATCHED_TERM_EQUAL", e4 == 0.0,
          "max|PW term - WML+H term| = %.3e (exact: both reuse the same log_prob tensor)"
          % e4)

    # T5: gradient reaches log_std through the added term, and not without it -----
    def wml_loss(dls, add):
        sg = B["sg"] * jnp.exp(dls)
        lp_i = logp_tanh_at_a(
            jnp.clip(jnp.tanh(B["mu"][None] + B["sg"][None]
                              * jax.random.normal(jax.random.PRNGKey(3), (M, *B["mu"].shape))),
                     -CLIP_A, CLIP_A), B["mu"][None], sg[None]).sum(-1)
        obj = -jnp.sum(jax.lax.stop_gradient(B["w_i"]) * lp_i, axis=0)
        if add:
            u = jax.random.normal(jax.random.PRNGKey(0), B["mu"].shape)
            obj = obj + logp_tanh(B["mu"] + sg * u, B["mu"], sg).sum(-1) * B["alpha"]
        return jnp.mean(obj)
    z0 = jnp.zeros(B["sg"].shape)
    g_off = jax.grad(lambda d: wml_loss(d, False))(z0)
    g_on = jax.grad(lambda d: wml_loss(d, True))(z0)
    delta = float(jnp.max(jnp.abs(g_on - g_off)))
    check("T5_LOGSTD_GRADIENT", delta > 0.0,
          "max|grad_on - grad_off| w.r.t. log_std = %.3e (must be > 0)" % delta)

    print("\n  RESULT: %s" % ("ALL PASS" if not fails else "FAILED: " + ", ".join(fails)))
    return 1 if fails else 0

if __name__ == "__main__":
    sys.exit(main())
