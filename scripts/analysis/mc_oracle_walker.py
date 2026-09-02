"""Monte-Carlo Q^pi oracle for corrected WalkerRun checkpoints.

Estimates the *finite-horizon Monte-Carlo* soft return that the critic was actually
regressed onto (traced line by line in ``reports/mc_oracle_code_trace.md``):

    Q_soft^pi(s0, a0) = E[ r_0 + sum_{t>=1} gamma^t ( r_t - alpha log pi(a_t|s_t) ) ]

with the externally fixed ``a0`` carrying NO entropy term, and **no Q_phi bootstrap
at the horizon** -- the oracle is differenced against Q_phi, so a Q_phi tail would
compare Q_phi with itself.

Three modes::

    states  <pw_ckpt> <wml_ckpt> <bank.npz>          build the common 64-state bank
    pilot   <ckpt> <bank.npz> <out.npz>              S=64 x K=16, H=500, c in {.10,.05}
    horizon <ckpt> <bank.npz> <out.npz>              fixed 16-state x 8-pert H=1000 subset

Design points that are load-bearing and are preregistered in
``docs/prereg_mc_oracle_walker_pilot.md``:

* The continuation policy is sampled **manually** rather than through distrax, so that
  the standard-normal innovations can be shared across the 25 finite-difference
  branches of one base point (common random numbers, prereg 4.4) while staying
  independent across perturbations, states, replicates and the A/B groups.
* Two independent rollout groups A and B per action point, 8 replicates each, so that
  squared quantities can be noise-debiased by cross-product (prereg 4.7).
* The observation normalizer is frozen at the checkpoint statistics; a drifting input
  transform would make pi non-stationary and Q^pi undefined.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from functools import partial

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from scripts.critic_fidelity.common import ACTION_CLIP, Harness  # noqa: E402

# ---------------------------------------------------------------- fixed design
# Two preregistered pilots. Only the RNG root, the state count and the rollout count
# differ; every other constant is shared, so pilot 2 is a precision replication of
# pilot 1 rather than a redesign. `p1` is the default so that the pilot-1 numbers in
# reports/mc_oracle_walker_pilot.md stay exactly reproducible from this file.
#
#   p1  docs/prereg_mc_oracle_walker_pilot.md          (commit 63c2cd2)
#   p2  docs/prereg_mc_oracle_walker_pilot_2.md        sized by the measured
#       variance-scaling law in reports/artifacts/mc_oracle_power.json
PILOTS = {
    "p1": {"root": 20260902, "s_per_arm": 32, "n_rep": 8},
    "p2": {"root": 20260903, "s_per_arm": 128, "n_rep": 96},
}

PILOT = "p1"
ROOT = 20260902          # prereg 4.2 state-generation RNG root
BURN = 50                # prereg 4.2, matching scripts/analysis/ubar_ratio.py:27
S_PER_ARM = 32           # prereg 4.2
S_TOTAL = 2 * S_PER_ARM
K_PERT = 16              # prereg 4.3
N_REP = 8                # prereg 4.4, per group
C_STEPS = (0.10, 0.05)   # prereg 4.6
H_MAIN = 500             # prereg 4.5
H_LONG = 1000            # prereg 4.5
SUB_STATES = 8           # prereg 4.5: first 8 of each stratum
SUB_PERTS = 8            # prereg 4.5
SUB_C = 0.10             # prereg 4.5


def set_pilot(tag: str):
    """Select a preregistered constant set. Refuses anything not registered."""
    global PILOT, ROOT, S_PER_ARM, S_TOTAL, N_REP
    if tag not in PILOTS:
        raise SystemExit("unknown pilot %r; registered: %s" % (tag, sorted(PILOTS)))
    cfg = PILOTS[tag]
    PILOT = tag
    ROOT = cfg["root"]
    S_PER_ARM = cfg["s_per_arm"]
    S_TOTAL = 2 * S_PER_ARM
    N_REP = cfg["n_rep"]
    print("pilot %s: root=%d S=%d (%d per arm) rollouts=%d per group"
          % (tag, ROOT, S_TOTAL, S_PER_ARM, N_REP), flush=True)


def fold(tag: str, *parts) -> jax.Array:
    """Purpose-separated fold-in off the single fixed root."""
    s = "|".join([tag] + [str(p) for p in parts])
    d = hashlib.blake2b(s.encode(), digest_size=4).digest()
    return jax.random.fold_in(jax.random.PRNGKey(ROOT), int.from_bytes(d, "big") % (2**31))


# ------------------------------------------------------------- policy, by hand
def gaussian_params(h: Harness, raw_obs):
    """Pre-squash (mu, sigma) at raw observations. Mirrors load_ckpt.policy_dist."""
    loc = h.ck.actor.actor_module(h.na(jnp.asarray(raw_obs)))
    mu, log_std = jnp.split(loc, 2, axis=-1)
    return mu, jnp.exp(log_std) + h.ck.actor.min_std


def tanh_normal_logprob(y, mu, sg):
    """log pi(tanh(y)) summed over action dims, for pi = Tanh(Normal(mu, sg)).

    log(1 - tanh(y)^2) = 2 (log 2 - y - softplus(-2y)), which is the numerically
    stable form; validated against distrax in test_mc_oracle.py (T7b).
    """
    z = (y - mu) / sg
    log_normal = -0.5 * z**2 - jnp.log(sg) - 0.5 * jnp.log(2.0 * jnp.pi)
    log_det = 2.0 * (jnp.log(2.0) - y - jax.nn.softplus(-2.0 * y))
    return jnp.sum(log_normal - log_det, axis=-1)


# ------------------------------------------------------------------- the oracle
class Oracle:
    """Truncated soft-return oracle over a batch of (n_base x n_branch) points."""

    def __init__(self, ckpt_dir: str, n_base: int, n_branch: int):
        self.n_base, self.n_branch = n_base, n_branch
        self.B = n_base * n_branch
        self.h = Harness(ckpt_dir, self.B)
        self.gamma = self.h.gamma
        # alpha read programmatically from the checkpoint, not from prose/meta
        self.alpha = float(np.asarray(self.h.ck.actor.temperature()).ravel()[0])
        self.d = self.h.action_dim

    @partial(jax.jit, static_argnums=(0, 5))
    def run(self, state, raw_obs, first_action, key, horizon: int):
        """Discounted truncated soft return of a0 then pi, plus the H_MAIN prefix.

        ``state`` carries a leading axis B; ``raw_obs`` is (B, obs_dim) unnormalized;
        ``first_action`` is (B, d), the unclipped tanh(y) -- ClipAction applies the
        training-faithful +-0.999 clip inside env.step.

        Returns ``(acc_full, acc_prefix, n_done)`` where ``acc_prefix`` is the
        accumulator frozen at step H_MAIN (equal to ``acc_full`` when
        ``horizon == H_MAIN``, which is T13 by construction).
        """
        nb, nbr, d = self.n_base, self.n_branch, self.d
        gamma, alpha = self.gamma, self.alpha

        # Innovations: (H, n_base, 1, d), broadcast over the n_branch axis so the
        # finite-difference branches of one base point share continuation randomness.
        # Always drawn at the FULL H_LONG length and then sliced: jax.random.normal
        # returns different values for different shapes, so drawing (horizon, ...)
        # directly would make the H=500 run something other than the prefix of the
        # H=1000 run, and T13 would be untestable rather than merely failing.
        eps_a = jax.random.normal(jax.random.fold_in(key, 1), (H_LONG, nb, 1, d))[:horizon]
        eps_e = jax.random.normal(jax.random.fold_in(key, 2), (H_LONG, nb, 1, d))[:horizon]

        # MjxGymnaxWrapper.step discards the key (jax_wrappers.py:103); MJX WalkerRun
        # dynamics are a pure function of (state, action). One constant is enough.
        step_keys = jax.random.split(jax.random.PRNGKey(0), self.B)

        def step(carry, xs):
            st, ob, acc, acc_pre, disc, alive, ndone = carry
            t, ea, ee = xs

            # executed action: a0 at t=0, else a fresh draw from pi(.|s_t)
            mu, sg = gaussian_params(self.h, ob)
            y = mu + sg * jnp.broadcast_to(ea, (nb, nbr, d)).reshape(self.B, d)
            sampled = jnp.tanh(y)
            act = jnp.where(t == 0, first_action, sampled)

            nobs, _, nst, rew, done, _ = self.h.env.step(step_keys, st, act)

            # entropy term at s_{t+1} under an independent draw, as collect_rollout does
            mu_n, sg_n = gaussian_params(self.h, nobs)
            y_n = mu_n + sg_n * jnp.broadcast_to(ee, (nb, nbr, d)).reshape(self.B, d)
            logp = tanh_normal_logprob(y_n, mu_n, sg_n)
            soft_r = rew - gamma * alpha * logp

            acc = acc + alive * disc * soft_r
            acc_pre = jnp.where(t == H_MAIN - 1, acc, acc_pre)
            dn = done.astype(jnp.float32)
            ndone = ndone + alive * dn
            alive = alive * (1.0 - dn)
            disc = disc * gamma
            return (nst, nobs, acc, acc_pre, disc, alive, ndone), None

        z = jnp.zeros((self.B,))
        init = (state, jnp.asarray(raw_obs), z, z, jnp.ones((self.B,)),
                jnp.ones((self.B,)), z)
        xs = (jnp.arange(horizon), eps_a, eps_e)
        (_, _, acc, acc_pre, _, _, ndone), _ = jax.lax.scan(step, init, xs)
        if horizon <= H_MAIN:
            acc_pre = acc
        return acc, acc_pre, ndone


# --------------------------------------------------------------- the state bank
def tile_states(state, reps):
    """Repeat each entry of a batched state ``reps`` times, keeping states adjacent."""
    return jax.tree.map(lambda x: jnp.repeat(x, reps, axis=0), state)


def gather_states(state, idx):
    return jax.tree.map(lambda x: jnp.asarray(x)[idx], state)


def flatten_state(state):
    """Flatten a state pytree to (ordered key list, list of arrays)."""
    leaves_with_path, treedef = jax.tree_util.tree_flatten_with_path(state)
    keys = [jax.tree_util.keystr(p) for p, _ in leaves_with_path]
    arrs = [np.asarray(v) for _, v in leaves_with_path]
    return keys, arrs, treedef


def build_bank(pw_ckpt: str, wml_ckpt: str, out: str):
    """32 states from each corrected policy, burn_in=50, saved in full."""
    parts, obs_parts, src = [], [], []
    for arm, ck in (("PW", pw_ckpt), ("WML", wml_ckpt)):
        h = Harness(ck, S_PER_ARM)
        key = fold("bank", arm)
        key, rk = jax.random.split(key)
        obs, _, st = h.reset(rk)
        for i in range(BURN):
            k1, k2, key = jax.random.split(key, 3)
            a = jnp.clip(h.pi(obs).sample(seed=k1), -ACTION_CLIP, ACTION_CLIP)
            obs, _, st, _, _, _ = h.env.step(jax.random.split(k2, S_PER_ARM), st, a)
        parts.append(st)
        obs_parts.append(np.asarray(obs))
        src += [arm] * S_PER_ARM

    state = jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *parts)
    raw_obs = np.concatenate(obs_parts, axis=0)
    keys, arrs, _ = flatten_state(state)

    payload = {"__obs__": raw_obs, "__source__": np.array(src)}
    for i, (k, a) in enumerate(zip(keys, arrs)):
        payload["%04d|%s" % (i, k)] = a
    np.savez(out, **payload)

    manifest = {
        "pilot": PILOT,
        "n_states": S_TOTAL,
        "per_arm": S_PER_ARM,
        "burn_in": BURN,
        "rng_root": ROOT,
        "sources": src,
        "pw_ckpt": pw_ckpt,
        "wml_ckpt": wml_ckpt,
        "leaf_keys": keys,
        "leaf_shapes": [list(a.shape) for a in arrs],
        "leaf_dtypes": [str(a.dtype) for a in arrs],
        "sha256": hashlib.sha256(open(out, "rb").read()).hexdigest(),
    }
    with open(out.replace(".npz", "_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print("bank:", out, S_TOTAL, "states,", len(keys), "leaves, sha256",
          manifest["sha256"][:16])
    return manifest


def load_bank(path: str, ref_state):
    """Restore the bank into the treedef of a freshly built ``ref_state``."""
    z = np.load(path, allow_pickle=True)
    raw_obs = z["__obs__"]
    src = [str(s) for s in z["__source__"]]
    keys, _, treedef = flatten_state(ref_state)
    names = sorted([k for k in z.files if "|" in k], key=lambda k: int(k.split("|", 1)[0]))
    saved_keys = [k.split("|", 1)[1] for k in names]
    if saved_keys != keys:
        raise ValueError("state bank leaf layout does not match the rebuilt env state")
    state = jax.tree_util.tree_unflatten(treedef, [jnp.asarray(z[n]) for n in names])
    return state, raw_obs, src


# ---------------------------------------------------------------- action points
def branch_offsets(d):
    """The 25 branches of a base point: base, then +-c*sigma_j*e_j for each c and j.

    Returned as (n_branch, d) multipliers of sigma: branch b adds ``off[b] * sigma``.
    Branch 0 is the base point. Order is fixed here and never re-derived.
    """
    off = [np.zeros(d)]
    meta = [("base", 0.0, -1, 0)]
    for c in C_STEPS:
        for j in range(d):
            for s in (+1, -1):
                v = np.zeros(d)
                v[j] = s * c
                off.append(v)
                meta.append(("fd", c, j, s))
    return np.stack(off), meta


def run_pilot(ckpt: str, bank: str, out: str, subset: bool = False):
    d_probe = Harness(ckpt, 1)
    d = d_probe.action_dim
    offs, bmeta = branch_offsets(d)
    if subset:
        offs = np.stack([offs[0]] + [o for o, m in zip(offs[1:], bmeta[1:])
                                     if m[1] == SUB_C])
        bmeta = [bmeta[0]] + [m for m in bmeta[1:] if m[1] == SUB_C]
    n_branch = len(offs)
    horizon = H_LONG if subset else H_MAIN

    # ---- states, policy, perturbations -----------------------------------
    ref = Harness(ckpt, S_TOTAL)
    _, ref_obs, ref_st = ref.reset(fold("ref"))
    state_all, raw_obs_all, src_all = load_bank(bank, ref_st)

    if subset:
        idx = list(range(SUB_STATES)) + list(range(S_PER_ARM, S_PER_ARM + SUB_STATES))
        k_take = SUB_PERTS
    else:
        idx = list(range(S_TOTAL))
        k_take = K_PERT
    idx = np.asarray(idx)
    n_state = len(idx)
    state_sel = gather_states(state_all, idx)
    raw_obs = raw_obs_all[idx]
    src = [src_all[i] for i in idx]

    mu, sg = gaussian_params(ref, jnp.asarray(raw_obs_all))
    mu = np.asarray(mu, np.float64)[idx]
    sg = np.asarray(sg, np.float64)[idx]

    # u is shared between the two checkpoints: it is drawn off the state index only
    u_full = np.asarray(jax.random.normal(fold("u"), (S_TOTAL, K_PERT, d),
                                          dtype=jnp.float32), np.float64)
    u = u_full[idx][:, :k_take]                                   # (n_state, k, d)

    y0 = mu[:, None, :] + sg[:, None, :] * u                      # (n_state, k, d)
    y = y0[:, :, None, :] + sg[:, None, None, :] * offs[None, None, :, :]
    a = np.tanh(y)                                                # (n_state,k,nbr,d)

    n_base = n_state * k_take
    B = n_base * n_branch
    a_flat = a.reshape(B, d).astype(np.float32)
    st_idx = np.repeat(np.arange(n_state), k_take * n_branch)

    # ---- deterministic Q_phi at every action point -----------------------
    cobs = np.asarray(ref.nc(jnp.asarray(raw_obs)), np.float32)
    q_phi = np.asarray(
        ref.ck.critic.critic(jnp.asarray(cobs[st_idx]), jnp.asarray(a_flat)), np.float64
    )

    # ---- rollouts --------------------------------------------------------
    chunk_base = int(os.environ.get("MCO_CHUNK", 256 if not subset else 128))
    chunk_base = min(chunk_base, n_base)
    # Oracle fixes its batch at chunk_base * n_branch, so a ragged final chunk would
    # silently feed the wrong shape. Require an exact division rather than pad.
    while n_base % chunk_base:
        chunk_base -= 1
    orc = Oracle(ckpt, chunk_base, n_branch)
    st_tiled_src = gather_states(state_sel, np.repeat(np.arange(n_state), k_take))

    tag = os.path.basename(ckpt)
    qh = np.zeros((2, N_REP, B))
    nd = np.zeros((2, N_REP, B))
    qh_pre = np.zeros((2, N_REP, B))
    for gi, grp in enumerate(("A", "B")):
        for r in range(N_REP):
            key = fold("roll", tag, grp, r, "sub" if subset else "main")
            for b0 in range(0, n_base, chunk_base):
                sl = slice(b0, b0 + chunk_base)
                stc = tile_states(gather_states(st_tiled_src, np.arange(n_base)[sl]),
                                  n_branch)
                obc = np.repeat(raw_obs[np.repeat(np.arange(n_state), k_take)[sl]],
                                n_branch, axis=0)
                fac = a_flat[b0 * n_branch:(b0 + chunk_base) * n_branch]
                acc, pre, dn = orc.run(stc, jnp.asarray(obc), jnp.asarray(fac),
                                       jax.random.fold_in(key, b0), horizon)
                s2 = slice(b0 * n_branch, (b0 + chunk_base) * n_branch)
                qh[gi, r, s2] = np.asarray(acc, np.float64)
                qh_pre[gi, r, s2] = np.asarray(pre, np.float64)
                nd[gi, r, s2] = np.asarray(dn, np.float64)
            print("  %s rep %d done" % (grp, r), flush=True)

    shape = (n_state, k_take, n_branch)
    np.savez(
        out,
        ckpt=np.array(ckpt), tag=np.array(tag), horizon=np.array(horizon),
        pilot=np.array(PILOT),
        subset=np.array(bool(subset)),
        alpha=np.array(orc.alpha), gamma=np.array(orc.gamma),
        state_index=idx, source=np.array(src),
        mu=mu, sigma=sg, u=u, y=y, action=a,
        offsets=offs, branch_kind=np.array([m[0] for m in bmeta]),
        branch_c=np.array([m[1] for m in bmeta]),
        branch_j=np.array([m[2] for m in bmeta]),
        branch_sign=np.array([m[3] for m in bmeta]),
        q_phi=q_phi.reshape(shape),
        q_oracle=qh.reshape(2, N_REP, *shape),
        q_oracle_prefix=qh_pre.reshape(2, N_REP, *shape),
        n_done=nd.reshape(2, N_REP, *shape),
        n_rep=np.array(N_REP), n_state=np.array(n_state), k=np.array(k_take),
    )
    print("wrote", out, "shape", shape)


if __name__ == "__main__":
    mode = sys.argv[1]
    args = sys.argv[2:]
    if args and args[-1] in PILOTS:          # optional trailing pilot tag
        set_pilot(args[-1])
        args = args[:-1]
    if mode == "states":
        build_bank(args[0], args[1], args[2])
    elif mode == "pilot":
        run_pilot(args[0], args[1], args[2], subset=False)
    elif mode == "horizon":
        run_pilot(args[0], args[1], args[2], subset=True)
    else:
        raise SystemExit("modes: states | pilot | horizon  [p1|p2]")
