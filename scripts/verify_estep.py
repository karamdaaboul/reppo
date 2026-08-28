"""Part 3 verification for the weighted-MLE (MPO E-step) actor arm.

(a) actor_update_mode="pathwise" must reproduce the ORIGINAL implementation exactly.
    Asserted end-to-end: a short training run under the patched module and under a
    pristine pre-patch snapshot must produce bit-identical parameters. That is
    strictly stronger than comparing one loss value, and needs no surgery to reach
    the closure `actor_loss` lives in.

(b) With weights forced uniform (w_i = 1/M) the weighted MLE must collapse onto the
    KL estimator already in the code. The exact identity is

        -sum_i (1/M) logp_theta_i  =  -mean_i logp_theta_i
                                   =  kl  -  mean_i logp_old_i

    since kl = mean_i logp_old_i - mean_i logp_theta_i. So the uniform-weight loss
    equals **+kl** plus a theta-independent constant, not -kl. Both terms are
    checked against the shipped `estep_weights`.

    The weight is the ANCHORED MPO form w propto exp(Q/eta): the E-step target
    q* propto pi_old * exp(Q/eta) has pi_old inside it, which cancels against the
    pi_old proposal. There is no 1/pi_old importance factor.
"""

from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import distrax  # noqa: E402
import numpy as np  # noqa: E402

import tests.reppo_upstream_snapshot as base  # noqa: E402
import src.jaxrl.reppo as new  # noqa: E402
from scripts.load_ckpt import load  # noqa: E402
from src.env_utils.jax_wrappers import MjxGymnaxWrapper  # noqa: E402

TINY = dict(
    lr=3e-4, gamma=0.99, total_time_steps=4096, num_steps=16, lmbda=0.95,
    lmbda_min=0.5, num_mini_batches=4, num_envs=64, num_epochs=1,
    max_grad_norm=0.5, normalize_env=True, polyak=1.0,
    exploration_noise_min=1.0, exploration_noise_max=1.0, exploration_base_envs=0,
    ent_start=0.01, ent_target_mult=0.5, kl_start=0.01, eval_interval=2, num_eval=2,
    max_episode_steps=1000, critic_hidden_dim=64, actor_hidden_dim=64,
    vmin=0, vmax=150, num_bins=51, hl_gauss=True, kl_bound=0.1, aux_loss_mult=1.0,
    update_kl_lagrangian=True, update_entropy_lagrangian=True, use_critic_norm=True,
    num_critic_encoder_layers=1, num_critic_head_layers=1, num_critic_pred_layers=1,
    use_simplical_embedding=False, use_critic_skip=False, use_actor_norm=True,
    num_actor_layers=2, actor_min_std=0.0, use_actor_skip=False, reduce_kl=True,
    reverse_kl=False, anneal_lr=False, actor_kl_clip_mode="clipped",
)


def _env():
    return MjxGymnaxWrapper("WalkerRun", episode_length=1000, reward_scale=1.0)


def check_a():
    print("=" * 72)
    print("(a) pathwise arm is bit-identical to the pre-patch implementation")
    print("=" * 72)
    hp_new = dict(TINY, actor_update_mode="pathwise", estep_num_samples=32,
                  log_q_spread=False, mstep_decoupled=False)
    cfg_new = new.ReppoConfig(**hp_new)
    cfg_base = base.ReppoConfig(**TINY)

    noop = lambda *a, **k: None  # noqa: E731
    fn_new = new.make_train_fn(cfg=cfg_new, env=_env(), log_callback=noop, num_seeds=1)
    fn_base = base.make_train_fn(cfg=cfg_base, env=_env(), log_callback=noop, num_seeds=1)

    key = jax.random.PRNGKey(0)
    s_new, m_new = jax.jit(fn_new, static_argnums=(1,))(key, cfg_new)
    s_base, m_base = jax.jit(fn_base, static_argnums=(1,))(key, cfg_base)

    worst = 0.0
    for label, a, b in [
        ("actor", s_new.actor.params, s_base.actor.params),
        ("critic", s_new.critic.params, s_base.critic.params),
    ]:
        d = jax.tree.map(lambda x, y: jnp.abs(x - y).max(), a, b)
        m = float(max(float(v) for v in jax.tree_util.tree_leaves(d)))
        worst = max(worst, m)
        print(f"  {label:<8} max |new - base| over all params = {m:.3e}")

    r_new = float(np.asarray(m_new["eval/episode_return"]).ravel()[-1])
    r_base = float(np.asarray(m_base["eval/episode_return"]).ravel()[-1])
    print(f"  eval return: new={r_new:.6f} base={r_base:.6f} diff={r_new - r_base:.3e}")
    ok = worst == 0.0 and r_new == r_base
    print(f"  -> {'PASS (exactly identical)' if ok else 'FAIL'}")
    return ok


def check_b():
    print()
    print("=" * 72)
    print("(b) uniform weights collapse the weighted MLE onto the KL estimator")
    print("=" * 72)
    ck = load(os.path.join(REPO_ROOT, "exports/WalkerRun_s0"))
    M, B = 32, 128
    rng = np.random.default_rng(0)
    obs = jnp.asarray(rng.normal(size=(B, ck.meta["obs_dim"])), dtype=jnp.float32)

    # exactly the arrays reppo.py builds in the forward-KL branch
    old_pi = ck.actor.actor(obs)
    a_i, logp_old_full = old_pi.sample_and_log_prob(
        sample_shape=(M,), seed=jax.random.PRNGKey(1)
    )
    a_i = jnp.clip(a_i, -1 + 1e-4, 1 - 1e-4)
    logp_old_i = logp_old_full.sum(-1)                       # (M, B)
    logp_theta_i = ck.actor.actor(obs).log_prob(a_i).sum(-1)  # (M, B)
    kl = logp_old_i.mean(0) - logp_theta_i.mean(0)

    w_uniform = jnp.ones((M, B)) / M
    loss_uniform = -jnp.sum(w_uniform * logp_theta_i, axis=0)
    identity = kl - logp_old_i.mean(0)
    err = float(jnp.abs(loss_uniform - identity).max())
    rel = err / float(jnp.abs(identity).max())
    print(f"  max |(-sum_i w_i logp_theta_i) - (kl - mean_i logp_old_i)| = {err:.3e}"
          f"  (rel {rel:.2e})")
    ok1 = rel < 1e-5

    # estep_weights must itself become uniform when the exponent is flat across i
    flat = jnp.zeros((M, B))
    w_flat = new.estep_weights(flat, jnp.float32(1.0))
    err2 = float(jnp.abs(w_flat - 1.0 / M).max())
    ess_flat = float(new.effective_sample_size(w_flat, axis=0).mean())
    print(f"  estep_weights on a flat exponent: max|w - 1/M| = {err2:.3e}, "
          f"ESS = {ess_flat:.3f} (want {M})")
    ok2 = err2 < 1e-6 and abs(ess_flat - M) < 1e-3

    # With the anchored weight w propto exp(Q/alpha), alpha -> inf now gives UNIFORM
    # weights: pi_old cancels against the proposal, so nothing but Q remains and a
    # large temperature flattens it. (The previous un-anchored form went to
    # w propto 1/pi_old instead -- that was the bug.)
    q_i = jnp.asarray(rng.normal(size=(M, B)), dtype=jnp.float32)
    w_big = new.estep_weights(q_i, jnp.float32(1e6))
    err3 = float(jnp.abs(w_big - 1.0 / M).max())
    loss_big = -jnp.sum(w_big * logp_theta_i, axis=0)
    err3b = float(jnp.abs(loss_big - identity).max())
    print(f"  alpha->inf now gives UNIFORM weights: max|w - 1/M| = {err3:.3e}, "
          f"and recovers the KL identity to {err3b:.3e}")
    ok3 = err3 < 1e-6 and err3b < 1e-3

    # (c) the eta dual must solve to KL(w || uniform) = eps_e. That stationarity
    # condition is what makes eta a real dual rather than a tuned constant.
    from scripts.critic_fidelity.common import ACTION_CLIP, Harness

    h = Harness(os.path.join(REPO_ROOT, "exports/WalkerRun_pathwise_s0_final"), 256)
    k = jax.random.PRNGKey(0)
    k, rk = jax.random.split(k)
    o, _, stt = h.reset(rk)
    for _ in range(50):
        k1, k2, k = jax.random.split(k, 3)
        aa = jnp.clip(h.pi(o).sample(seed=k1), -ACTION_CLIP, ACTION_CLIP)
        o, _, stt, _, _, _ = h.env.step(jax.random.split(k2, 256), stt, aa)

    pio = h.ck.actor.actor(h.na(o))
    ar = jnp.clip(pio.sample(sample_shape=(M,), seed=jax.random.PRNGKey(5)),
                  -1 + 1e-4, 1 - 1e-4)
    cobs = jnp.broadcast_to(h.nc(o), (M, *h.nc(o).shape))
    qr = h.ck.critic.critic(cobs, ar)

    eps_e = 0.5
    grid = jnp.exp(jnp.linspace(jnp.log(1e-3), jnp.log(10.0), 400))
    vals = jnp.array([new.eta_dual_loss(qr, e, eps_e) for e in grid])
    eta_star = float(grid[int(jnp.argmin(vals))])
    w = new.estep_weights(qr, jnp.float32(eta_star))
    kl_w = float(jnp.mean(jnp.sum(w * jnp.log(jnp.clip(w * M, 1e-12)), axis=0)))
    ess_star = float(new.effective_sample_size(w, axis=0).mean())
    qsp = float((qr / eta_star).std(axis=0).mean())
    print()
    print(f"  [dual] grid-minimised eta on a real checkpoint, eps_e={eps_e}, M={M}:")
    print(f"     eta*            = {eta_star:.4f}   (alpha was {float(ck.meta['alpha_entropy']):.5f})")
    print(f"     KL(w||uniform)  = {kl_w:.4f}   (dual stationarity wants eps_e={eps_e})")
    print(f"     q_spread        = {qsp:.2f}")
    print(f"     ESS             = {ess_star:.2f} of {M}")
    ok4 = abs(kl_w - eps_e) < 0.05
    print(f"     -> dual stationarity {'HOLDS' if ok4 else 'VIOLATED'}")

    ok = ok1 and ok2 and ok3 and ok4
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_c():
    """With beta_mu = beta_sigma = 0, the decoupled MLE term must reduce to the
    existing weighted MLE.

    The two objectives differ in VALUE (the decoupled sum evaluates both halves at the
    old parameters, and the coupled one carries the tanh Jacobian), but at theta = theta_old
    their GRADIENTS coincide exactly: mu takes its gradient only from
    log N(u; mu_theta, sg_old) and sigma only from log N(u; mu_old, sg_theta), and each
    equals the corresponding partial of the coupled log-density. Gradient equality is
    the substantive claim, so that is what is asserted.
    """
    print()
    print("=" * 72)
    print("(c) beta=0: decoupled MLE reduces to the existing weighted MLE")
    print("=" * 72)
    ck = load(os.path.join(REPO_ROOT, "exports/WalkerRun_weighted_mle_s0_final"))
    M, B = 32, 64
    rng = np.random.default_rng(0)
    obs = jnp.asarray(rng.normal(size=(B, ck.meta["obs_dim"])), dtype=jnp.float32)

    mu_old, sg_old = ck.actor.gaussian(obs)
    mu_old, sg_old = jax.lax.stop_gradient(mu_old), jax.lax.stop_gradient(sg_old)
    u_i = mu_old + sg_old * jax.random.normal(jax.random.PRNGKey(1), (M, *mu_old.shape))
    a_i = jnp.clip(jnp.tanh(u_i), -1 + 1e-4, 1 - 1e-4)
    w = jnp.asarray(rng.dirichlet(np.ones(M), size=B).T, dtype=jnp.float32)  # (M,B)

    def decoupled(params):
        mu, sg = params
        lp = new.gaussian_logp(u_i, mu[None], sg_old[None]) + new.gaussian_logp(
            u_i, mu_old[None], sg[None]
        )
        return jnp.sum(-jnp.sum(w * lp, axis=0))

    def coupled_base(params):
        # the weighted MLE the decoupled form is meant to reduce to, evaluated on the
        # pre-squash samples directly
        mu, sg = params
        return jnp.sum(-jnp.sum(w * new.gaussian_logp(u_i, mu[None], sg[None]), axis=0))

    def coupled_via_tanh(params):
        # what the shipped arm actually computes: log_prob of the CLIPPED squashed
        # action, which takes atanh internally
        mu, sg = params
        pi = distrax.Transformed(distrax.Normal(loc=mu, scale=sg), distrax.Tanh())
        return jnp.sum(-jnp.sum(w * pi.log_prob(a_i).sum(-1), axis=0))

    p0 = (mu_old, sg_old)  # evaluate at theta = theta_old
    gd = jax.grad(decoupled)(p0)
    gb = jax.grad(coupled_base)(p0)
    gt = jax.grad(coupled_via_tanh)(p0)
    for nm, i in [("d/d mu", 0), ("d/d sigma", 1)]:
        err = float(jnp.abs(gd[i] - gb[i]).max())
        rel = err / float(jnp.abs(gb[i]).max())
        print(f"  {nm:<10} max|decoupled - weighted MLE| = {err:.3e}   rel = {rel:.2e}")
    ok = all(
        float(jnp.abs(gd[i] - gb[i]).max()) / float(jnp.abs(gb[i]).max()) < 1e-4
        for i in (0, 1)
    )
    print(f"  -> gradients {'MATCH exactly' if ok else 'DIFFER'} at theta = theta_old")

    clipped = float((jnp.abs(jnp.tanh(u_i)) > 1 - 1e-4).mean())
    print(f"\n  [artifact] the shipped arm evaluates log_prob on the CLIPPED squashed")
    print(f"  action, so atanh cannot recover the pre-squash value for saturated samples.")
    print(f"     samples at the tanh clip           : {clipped:.2%}")
    print(f"     max |atanh(clip(tanh(u))) - u|     : "
          f"{float(jnp.abs(jnp.arctanh(a_i) - u_i).max()):.2e}")
    print(f"     |grad| via clipped tanh vs on u    : "
          f"mu {float(jnp.abs(gt[0]).max()):.2f} vs {float(jnp.abs(gb[0]).max()):.2f}, "
          f"sigma {float(jnp.abs(gt[1]).max()):.2f} vs {float(jnp.abs(gb[1]).max()):.2f}")
    print(f"  The decoupled M-step works on u directly and avoids this entirely.")
    return ok


def main() -> int:
    a = check_a()
    b = check_b()
    c = check_c()
    print()
    if a and b and c:
        print("VERIFICATION (a)+(b)+(c) PASSED")
        return 0
    print("VERIFICATION FAILED: " + ", ".join(
        n for n, v in [("(a)", a), ("(b)", b), ("(c)", c)] if not v))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
