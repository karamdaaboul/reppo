"""Freeze-specific unit tests (phase 1.9) and gradient semantics (phase 1.5).

Run: ./.venv/bin/python scripts/analysis/test_covariance_freeze.py
"""
from __future__ import annotations

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
from flax import nnx  # noqa: E402

from src.jaxrl.reppo import gaussian_kl_diag  # noqa: E402
from src.networks.jax_models import SACActorNetworks  # noqa: E402

D = 6
KW = dict(obs_dim=24, action_dim=D, hidden_dim=512, ent_start=0.0145,
          kl_start=0.01, use_norm=True, layers=3, use_skip=False,
          with_eta=False, with_betas=False)
SIG_A = 1.1
VEC = [0.30, 0.45, 0.60, 0.75, 0.90, 1.05]
R = []


def chk(name, ok, detail=""):
    R.append((name, bool(ok)))
    print("%-6s %-58s %s" % ("PASS" if ok else "FAIL", name, detail), flush=True)


def mk(freeze=None, seed=0):
    return SACActorNetworks(**KW, freeze_sigma=freeze, rngs=nnx.Rngs(seed))


def obs(n=64, seed=1):
    return jax.random.normal(jax.random.PRNGKey(seed), (n, KW["obs_dim"])) * 2.0


def scale_slice(state):
    """Path and slice of the log_std-specific half of the shared output layer."""
    paths = [(p, v) for p, v in jax.tree_util.tree_leaves_with_path(state)]
    hits = [(jax.tree_util.keystr(p), v) for p, v in paths
            if "output_layer" in jax.tree_util.keystr(p)]
    return hits


def main():
    o = obs()

    # T1 scalar broadcast
    a = mk(SIG_A)
    _, sg = a.gaussian(o)
    sg = np.asarray(sg)
    chk("T1 scalar freeze broadcasts to all %d coordinates" % D,
        sg.shape == (o.shape[0], D) and np.all(sg == np.float32(SIG_A)),
        "shape %s, unique values %s" % (sg.shape, np.unique(sg).tolist()))

    # T2 vector preserved per coordinate
    av = mk(VEC)
    _, sv = av.gaussian(o)
    sv = np.asarray(sv, np.float64)
    want = np.array(VEC, np.float32).astype(np.float64)
    chk("T2 vector freeze preserves each coordinate",
        np.all(sv == want[None, :]),
        "max|diff| %.3g" % float(np.max(np.abs(sv - want[None, :]))))

    # T3 arbitrary observations, including extreme ones
    wild = jnp.concatenate([o, o * 1e3, jnp.zeros_like(o), -o * 47.0], 0)
    _, sw = a.gaussian(wild)
    chk("T3 effective sigma is the requested value for arbitrary obs",
        bool(jnp.all(sw == jnp.float32(SIG_A))),
        "n=%d observations, all coords" % wild.shape[0])

    # T4 online vs target actor (different init seeds -> still identical sigma)
    on, tg = mk(SIG_A, seed=0), mk(SIG_A, seed=7)
    _, s_on = on.gaussian(o)
    _, s_tg = tg.gaussian(o)
    chk("T4 online and target actors use identical sigma",
        bool(jnp.all(s_on == s_tg)),
        "max|diff| %.3g (independent inits)"
        % float(jnp.max(jnp.abs(s_on - s_tg))))

    # T5 every distribution constructor agrees
    d_actor = a.actor(o).distribution.scale
    d_gauss = a.gaussian(o)[1]
    d_eff = a.effective_std(jnp.zeros((o.shape[0], D)))
    same = bool(jnp.all(d_actor == d_gauss) and jnp.all(d_actor == d_eff))
    det_l = np.asarray(mk(None, 0).det_action(o))
    det_f = np.asarray(mk(SIG_A, 0).det_action(o))
    chk("T5 actor()/gaussian()/effective_std() agree; det_action sigma-free",
        same and np.array_equal(det_l, det_f),
        "evaluation uses det_action = tanh(mu), which has no sigma at all")

    # ---- T6/T7/T8 gradient semantics -----------------------------------------
    def loss_of(model, params):
        m = nnx.merge(nnx.graphdef(model), params)
        mu, sg = m.gaussian(o)
        return jnp.sum(mu ** 2) + jnp.sum(sg ** 2)

    for tag, model in (("frozen", mk(SIG_A)), ("learned", mk(None))):
        params = nnx.state(model)
        g = jax.grad(lambda p: loss_of(model, p))(params)
        leaves = scale_slice(g)
        out = {k: np.asarray(v) for k, v in leaves}
        kern = [v for k, v in out.items() if "kernel" in k][0]
        bias = [v for k, v in out.items() if "bias" in k][0]
        gm, gs = kern[:, :D], kern[:, D:]          # mean half | scale half
        bm, bs = bias[:D], bias[D:]
        trunk = [np.asarray(v) for k, v in
                 [(jax.tree_util.keystr(p), v)
                  for p, v in jax.tree_util.tree_leaves_with_path(g)]
                 if "input_layer" in k and "kernel" in k]
        if tag == "frozen":
            chk("T6 scale-output-head gradient is exactly zero when frozen",
                np.all(gs == 0.0) and np.all(bs == 0.0),
                "max|grad| kernel %.3g bias %.3g"
                % (np.abs(gs).max(), np.abs(bs).max()))
            chk("T8 mean head and shared trunk keep nonzero gradient",
                np.abs(gm).max() > 0 and np.abs(trunk[0]).max() > 0,
                "mean-head %.4g, trunk %.4g"
                % (np.abs(gm).max(), np.abs(trunk[0]).max()))
        else:
            chk("T8b learned run does drive the scale head (control)",
                np.abs(gs).max() > 0,
                "max|grad| on scale half %.4g" % np.abs(gs).max())

    # T7 optimizer step leaves the scale half bit-identical
    model = mk(SIG_A)
    params = nnx.state(model)
    tx = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(3e-4))
    opt = tx.init(params)
    before = None
    for step in range(3):
        g = jax.grad(lambda p: loss_of(model, p))(params)
        upd, opt = tx.update(g, opt, params)
        params = optax.apply_updates(params, upd)
        k = [np.asarray(v) for kk, v in scale_slice(params) if "kernel" in kk][0]
        if before is None:
            before0 = k.copy()
        before = k
    k0 = [np.asarray(v) for kk, v in scale_slice(nnx.state(mk(SIG_A))) if "kernel" in kk][0]
    chk("T7 scale-output parameters unchanged after 3 optimizer steps",
        np.array_equal(k0[:, D:], before[:, D:]) and not np.array_equal(k0[:, :D], before[:, :D]),
        "scale half max|delta| %.3g ; mean half moved %.4g"
        % (np.abs(k0[:, D:] - before[:, D:]).max(),
           np.abs(k0[:, :D] - before[:, :D]).max()))

    # T9 analytic Gaussian KL loses its covariance contribution exactly
    mu0 = jax.random.normal(jax.random.PRNGKey(3), (128, D))
    mu1 = mu0 + 0.37
    s = jnp.full((128, D), SIG_A)
    kl = gaussian_kl_diag(mu0, s, mu1, s)
    mean_term = jnp.sum((mu0 - mu1) ** 2 / (2 * s ** 2), axis=-1)
    sl = jnp.full((128, D), 0.4)
    kl_learned = gaussian_kl_diag(mu0, s, mu1, sl)
    chk("T9 analytic KL reduces to its mean-displacement term when sigma is fixed",
        bool(jnp.max(jnp.abs(kl - mean_term)) < 1e-6)
        and bool(jnp.max(jnp.abs(kl_learned - mean_term)) > 1e-3),
        "max|KL - mean term| %.3g (differing-sigma control: %.4g)"
        % (float(jnp.max(jnp.abs(kl - mean_term))),
           float(jnp.max(jnp.abs(kl_learned - mean_term)))))

    # T10 flag off is exactly the learned path
    lr = mk(None)
    loc = lr.actor_module(o)
    _, ls = jnp.split(loc, 2, axis=-1)
    ref = jnp.exp(ls) + lr.min_std
    chk("T10 freeze_sigma=None reproduces exp(log_std)+min_std exactly",
        bool(jnp.all(lr.gaussian(o)[1] == ref))
        and bool(jnp.all(lr.actor(o).distribution.scale == ref)),
        "max|diff| %.3g" % float(jnp.max(jnp.abs(lr.gaussian(o)[1] - ref))))

    # guards
    bad = 0
    for v in ([0.0], [-1.0], [1.0] * 5, float("nan")):
        try:
            mk(v)
        except ValueError:
            bad += 1
    chk("T11 invalid freeze values are rejected", bad == 4,
        "%d/4 rejected (zero, negative, wrong length, NaN)" % bad)

    n = sum(1 for _, ok in R if ok)
    print("\n%d/%d tests pass" % (n, len(R)))
    raise SystemExit(0 if n == len(R) else 1)


if __name__ == "__main__":
    main()
