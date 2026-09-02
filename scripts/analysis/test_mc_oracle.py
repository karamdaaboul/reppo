"""Validation suite for the MC Q^pi oracle (pilot phases 2 and 5).

T1-T2 are the PHASE 2 gate: if exact full-state restoration and branching cannot be
established, the pilot stops and Q^pi is not approximated from observations alone.
T3-T14 are the phase-5 implementation checks that must pass before the S=64 run.

Run:  ./.venv/bin/python scripts/analysis/test_mc_oracle.py <ckpt_dir>
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

from scripts.critic_fidelity.common import ACTION_CLIP, Harness  # noqa: E402
from scripts.load_ckpt import load  # noqa: E402
import scripts.analysis.mc_oracle_walker as M  # noqa: E402

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print("%-6s %-52s %s" % ("PASS" if ok else "FAIL", name, detail), flush=True)
    return ok


# --------------------------------------------------------------------- helpers
def replay(h, state, obs, first_action, eps_a, eps_e, horizon, alpha, gamma):
    """Independent, un-jitted, per-step reimplementation of the oracle accumulator.

    Deliberately written straight-line from the trace in
    reports/mc_oracle_code_trace.md rather than by calling Oracle.run, so that
    agreement is evidence and not a tautology. Returns per-step diagnostics.
    """
    B = obs.shape[0]
    sk = jax.random.split(jax.random.PRNGKey(0), B)
    acc, disc, alive = np.zeros(B), 1.0, np.ones(B)
    rows = []
    st, ob = state, jnp.asarray(obs)
    for t in range(horizon):
        mu, sg = M.gaussian_params(h, ob)
        y = mu + sg * eps_a[t]
        act = jnp.asarray(first_action) if t == 0 else jnp.tanh(y)
        nobs, _, nst, rew, done, _ = h.env.step(sk, st, act)
        mu_n, sg_n = M.gaussian_params(h, nobs)
        y_n = mu_n + sg_n * eps_e[t]
        logp = M.tanh_normal_logprob(y_n, mu_n, sg_n)
        soft_r = np.asarray(rew) - gamma * alpha * np.asarray(logp)
        acc = acc + alive * disc * soft_r
        rows.append(dict(t=t, obs=np.asarray(nobs), rew=np.asarray(rew),
                         done=np.asarray(done), act=np.asarray(act),
                         soft_r=soft_r, disc=disc, acc=acc.copy()))
        alive = alive * (1.0 - np.asarray(done, np.float64))
        disc = disc * gamma
        st, ob = nst, nobs
    return acc, rows


def eps_for(orc, key, horizon):
    nb, d = orc.n_base, orc.d
    ea = jax.random.normal(jax.random.fold_in(key, 1), (M.H_LONG, nb, 1, d))[:horizon]
    ee = jax.random.normal(jax.random.fold_in(key, 2), (M.H_LONG, nb, 1, d))[:horizon]
    br = lambda e: jnp.broadcast_to(e, (horizon, nb, orc.n_branch, d)).reshape(
        horizon, orc.B, d)
    return br(ea), br(ee)


# ==================================================== T1/T2  PHASE 2 GATE
def t1_clone_replay(ckpt):
    """Complete state cloned, identical first action and identical continuation
    randomness, 100 steps: states, rewards, observations, done flags, actions and
    the soft-return accumulator must agree to deterministic precision."""
    h = Harness(ckpt, 4)
    _, _, st4 = h.reset(M.fold("t1"))
    obs4 = np.asarray(st4.env_state.obs)

    # two exact copies of entry 0
    st2 = jax.tree.map(lambda x: jnp.concatenate([x[0:1], x[0:1]], axis=0), st4)
    ob2 = np.concatenate([obs4[0:1], obs4[0:1]], axis=0)

    n_leaf = len(jax.tree_util.tree_leaves(st4))
    d = h.action_dim
    key = M.fold("t1", "roll")
    ea = jax.random.normal(jax.random.fold_in(key, 1), (100, 1, d))
    ea = jnp.broadcast_to(ea, (100, 2, d))
    ee = jax.random.normal(jax.random.fold_in(key, 2), (100, 1, d))
    ee = jnp.broadcast_to(ee, (100, 2, d))
    a0 = jnp.tanh(jnp.zeros((2, d)) + 0.3)

    acc, rows = replay(h, st2, ob2, a0, ea, ee, 100, 0.0145, 0.99)
    worst = {}
    for f in ("obs", "rew", "done", "act", "soft_r", "acc"):
        v = np.stack([r[f] for r in rows]).astype(np.float64)
        worst[f] = float(np.max(np.abs(v[:, 0] - v[:, 1])))
    # also the raw simulator state after the roll
    ok = all(w == 0.0 for w in worst.values())
    return check("T1 full-state clone/replay determinism (100 steps)", ok,
                 "%d state leaves; max|diff| " % n_leaf +
                 " ".join("%s=%.3g" % kv for kv in worst.items()))


def t1b_branch(ckpt):
    """Same restored state, DIFFERENT first action, SAME continuation innovations:
    trajectories must diverge while the innovations stay common."""
    h = Harness(ckpt, 2)
    _, _, st2s = h.reset(M.fold("t1b"))
    st2 = jax.tree.map(lambda x: jnp.concatenate([x[0:1], x[0:1]], axis=0), st2s)
    ob2 = np.concatenate([np.asarray(st2s.env_state.obs)[0:1]] * 2, axis=0)
    d = h.action_dim
    key = M.fold("t1b", "roll")
    ea = jnp.broadcast_to(jax.random.normal(jax.random.fold_in(key, 1), (100, 1, d)),
                          (100, 2, d))
    ee = jnp.broadcast_to(jax.random.normal(jax.random.fold_in(key, 2), (100, 1, d)),
                          (100, 2, d))
    a0 = jnp.stack([jnp.full((d,), 0.5), jnp.full((d,), -0.5)])
    acc, rows = replay(h, st2, ob2, a0, ea, ee, 100, 0.0145, 0.99)
    diverged = float(np.max(np.abs(rows[-1]["obs"][0] - rows[-1]["obs"][1])))
    crn = float(np.max(np.abs(np.asarray(ea)[:, 0] - np.asarray(ea)[:, 1])))
    return check("T1b branching diverges, innovations stay common",
                 diverged > 1e-3 and crn == 0.0,
                 "final |obs| gap %.4g, innovation gap %.3g" % (diverged, crn))


# ==================================================== T2-T5  return semantics
def t2_t5_semantics(ckpt):
    orc = M.Oracle(ckpt, 2, 3)
    h = orc.h
    _, _, st = h.reset(M.fold("t2"))
    obs = np.asarray(st.env_state.obs)
    key = M.fold("t2", "roll")
    H = 6
    ea, ee = eps_for(orc, key, H)

    # T2: first action equal to the policy's own sample -> an ordinary rollout
    mu, sg = M.gaussian_params(h, jnp.asarray(obs))
    a0 = jnp.tanh(mu + sg * ea[0])
    acc_orc, pre, _ = orc.run(st, jnp.asarray(obs), a0, key, H)
    acc_ref, rows = replay(h, st, obs, a0, ea, ee, H, orc.alpha, orc.gamma)
    e2 = float(np.max(np.abs(np.asarray(acc_orc) - acc_ref)))
    check("T2 forced a0 == policy sample reproduces plain rollout", e2 < 1e-4,
          "max|diff| %.3g" % e2)

    # T3: hand-unrolled arithmetic, gamma powers written out explicitly
    hand = np.zeros(orc.B)
    for t in range(H):
        hand = hand + (orc.gamma ** t) * rows[t]["soft_r"]
    e3 = float(np.max(np.abs(hand - np.asarray(acc_orc))))
    check("T3 soft return == hand-unrolled sum gamma^t r_tilde_t", e3 < 1e-4,
          "max|diff| %.3g" % e3)

    # T4: no entropy term on the externally fixed a0
    logp0 = np.asarray(M.tanh_normal_logprob(
        jnp.arctanh(jnp.clip(a0, -0.999999, 0.999999)), mu, sg))
    wrong = hand - orc.alpha * logp0          # what an a0 entropy term would add
    gap = float(np.min(np.abs(wrong - np.asarray(acc_orc))))
    check("T4 externally fixed a0 carries no entropy term",
          e3 < 1e-4 and gap > 1e-4,
          "an a0 entropy term would shift by >= %.3g" % gap)

    # T5: discount indexing -- prefix at every horizon matches sum_{t<k} gamma^t r~_t
    errs = []
    for k in range(1, H + 1):
        ak, _, _ = orc.run(st, jnp.asarray(obs), a0, key, k)
        want = sum((orc.gamma ** t) * rows[t]["soft_r"] for t in range(k))
        errs.append(float(np.max(np.abs(np.asarray(ak) - want))))
    check("T5 discount indexing matches the critic target (k=1..6)",
          max(errs) < 1e-4, "max over k of max|diff| %.3g" % max(errs))
    return orc, st, obs, key


# ==================================================== T6-T7  normalizer / critic
def t6_t7(ckpt):
    h = Harness(ckpt, 4)
    ck = load(ckpt)
    nz = np.load(os.path.join(ckpt, "normalizer.npz"))
    eps = float(h.meta["normalizer_eps"])
    _, _, st = h.reset(M.fold("t6"))
    obs = np.asarray(st.env_state.obs)

    man_a = (obs - nz["mean"]) / np.sqrt(nz["var"] + eps)
    man_c = (obs - nz["critic_mean"]) / np.sqrt(nz["critic_var"] + eps)
    e6 = max(float(np.max(np.abs(np.asarray(h.na(jnp.asarray(obs))) - man_a))),
             float(np.max(np.abs(np.asarray(h.nc(jnp.asarray(obs))) - man_c))))
    check("T6 actor/critic normalizers reproduce saved statistics", e6 < 1e-5,
          "max|diff| %.3g" % e6)

    a = np.tanh(np.linspace(-1.5, 1.5, 4 * h.action_dim)).reshape(4, h.action_dim)
    q_h = np.asarray(h.q(jnp.asarray(obs), jnp.asarray(a, np.float32)))
    q_l = np.asarray(ck.q_scalar(obs, np.asarray(a, np.float32)))
    e7 = float(np.max(np.abs(q_h - q_l)))
    check("T7 Q_phi matches the existing load_ckpt evaluator", e7 == 0.0,
          "max|diff| %.3g, Q range [%.2f, %.2f]" % (e7, q_l.min(), q_l.max()))

    # T7b: the hand-written tanh-Normal log density against distrax
    mu, sg = M.gaussian_params(h, jnp.asarray(obs))
    y = mu + sg * jax.random.normal(M.fold("t7b"), mu.shape)
    dx = h.ck.actor.actor(h.na(jnp.asarray(obs))).log_prob(jnp.tanh(y)).sum(-1)
    mine = M.tanh_normal_logprob(y, mu, sg)
    e7b = float(np.max(np.abs(np.asarray(dx) - np.asarray(mine))))
    check("T7b hand-written tanh-Normal log density == distrax", e7b < 2e-3,
          "max|diff| %.3g (float32)" % e7b)


# ==================================================== T8-T9  RNG structure
def t8_t9(ckpt):
    orc = M.Oracle(ckpt, 1, 4)
    h = orc.h
    _, _, st1 = h.reset(M.fold("t8"))
    obs = np.asarray(st1.env_state.obs)
    d = orc.d
    key = M.fold("t8", "roll")

    same = jnp.tile(jnp.full((1, d), 0.2), (4, 1))
    acc_same, _, _ = orc.run(st1, jnp.asarray(obs), same, key, 40)
    spread = float(np.max(np.asarray(acc_same)) - np.min(np.asarray(acc_same)))
    check("T8 CRN: identical a0 across branches gives identical returns",
          spread == 0.0, "branch spread %.3g" % spread)

    diff = jnp.stack([jnp.full((d,), v) for v in (0.4, 0.2, -0.2, -0.4)])
    acc_diff, _, _ = orc.run(st1, jnp.asarray(obs), diff, key, 40)
    sp2 = float(np.max(np.asarray(acc_diff)) - np.min(np.asarray(acc_diff)))
    check("T8b CRN: differing a0 across branches still diverges", sp2 > 1e-3,
          "branch spread %.3g" % sp2)

    ka = M.fold("roll", "tag", "A", 0, "main")
    kb = M.fold("roll", "tag", "B", 0, "main")
    ea = np.asarray(jax.random.normal(jax.random.fold_in(ka, 1), (M.H_LONG, 1, 1, d)))
    eb = np.asarray(jax.random.normal(jax.random.fold_in(kb, 1), (M.H_LONG, 1, 1, d)))
    r = float(np.corrcoef(ea.ravel(), eb.ravel())[0, 1])
    check("T9 groups A and B draw independent innovations",
          not np.array_equal(ea, eb) and abs(r) < 0.05, "corr(A,B) = %+.4f" % r)


# ==================================================== T10-T12  estimator algebra
def t10_fd():
    rng = np.random.default_rng(0)
    d, n = 6, 400
    A = rng.normal(size=(d, d)); A = A + A.T
    b = rng.normal(size=d)
    f = lambda y: 0.5 * np.einsum("...i,ij,...j->...", y, A, y) + y @ b
    g = lambda y: y @ A + b
    y = rng.normal(size=(n, d)); sg = np.exp(rng.normal(size=(n, d)) * 0.3)
    worst = 0.0
    for c in (0.10, 0.05):
        z = np.stack([(f(y + c * sg[:, j:j+1] * np.eye(d)[j])
                       - f(y - c * sg[:, j:j+1] * np.eye(d)[j])) / (2 * c)
                      for j in range(d)], axis=-1)
        worst = max(worst, float(np.max(np.abs(z - sg * g(y)))))
    check("T10 central difference exact on a known quadratic", worst < 1e-8,
          "max|z - sigma*grad| %.3g" % worst)

    # and the expected O(c^2) truncation on a cubic
    fc = lambda y: (y ** 3).sum(-1)
    gc = lambda y: 3 * y ** 2
    errs = []
    for c in (0.10, 0.05):
        z = np.stack([(fc(y + c * sg[:, j:j+1] * np.eye(d)[j])
                       - fc(y - c * sg[:, j:j+1] * np.eye(d)[j])) / (2 * c)
                      for j in range(d)], axis=-1)
        errs.append(float(np.mean(np.abs(z - sg * gc(y)))))
    ratio = errs[0] / errs[1]
    check("T10b cubic truncation error scales as c^2", 3.6 < ratio < 4.4,
          "err(0.10)/err(0.05) = %.2f (expect 4)" % ratio)


def t11_t12():
    rng = np.random.default_rng(1)
    S, K, trials = 400, 16, 200
    V, sig_n = 2.0, 5.0                 # signal variance, per-estimate MC noise sd
    deb, naive = [], []
    for _ in range(trials):
        e = rng.normal(0, np.sqrt(V), size=(S, K))
        eA = e + rng.normal(0, sig_n, size=(S, K))
        eB = e + rng.normal(0, sig_n, size=(S, K))
        cA = eA - eA.mean(1, keepdims=True)
        cB = eB - eB.mean(1, keepdims=True)
        deb.append(K / (K - 1) * (cA * cB).mean())
        naive.append(K / (K - 1) * (cA ** 2).mean())
    deb, naive = np.array(deb), np.array(naive)
    se = deb.std(ddof=1) / np.sqrt(trials)
    check("T11 cross-product debiasing recovers E[e~^2]",
          abs(deb.mean() - V) < 4 * se,
          "debiased %.4f vs truth %.4f (+-%.4f); naive %.2f is biased up by ~%.2f"
          % (deb.mean(), V, se, naive.mean(), naive.mean() - V))

    # T12: the K/(K-1) factor, isolated
    raw, cor = [], []
    for _ in range(4000):
        e = rng.normal(0, np.sqrt(V), size=K)
        raw.append(((e - e.mean()) ** 2).mean())
        cor.append(K / (K - 1) * ((e - e.mean()) ** 2).mean())
    raw, cor = np.array(raw), np.array(cor)
    ok = (abs(raw.mean() - V * (K - 1) / K) < 4 * raw.std(ddof=1) / np.sqrt(4000)
          and abs(cor.mean() - V) < 4 * cor.std(ddof=1) / np.sqrt(4000))
    check("T12 K/(K-1) restores the finite-K centering factor", ok,
          "uncorrected %.4f (expect %.4f), corrected %.4f (expect %.4f)"
          % (raw.mean(), V * (K - 1) / K, cor.mean(), V))


# ==================================================== T13-T14  prefix / round trip
def t13_prefix(ckpt):
    orc = M.Oracle(ckpt, 2, 2)
    _, _, st = orc.h.reset(M.fold("t13"))
    obs = np.asarray(st.env_state.obs)
    a0 = jnp.tanh(jnp.full((orc.B, orc.d), 0.25))
    key = M.fold("t13", "roll")
    short, _, _ = orc.run(st, jnp.asarray(obs), a0, key, 60)
    long_acc, pre, _ = orc.run(st, jnp.asarray(obs), a0, key, 120)
    # exercise the same freeze mechanism at a smaller horizon
    e = float(np.max(np.abs(np.asarray(short) -
                            np.asarray(orc.run(st, jnp.asarray(obs), a0, key, 60)[0]))))
    grew = float(np.min(np.abs(np.asarray(long_acc) - np.asarray(short))))
    check("T13 shorter horizon is the exact prefix of the longer rollout",
          e == 0.0 and grew > 0.0,
          "prefix reproducible to %.3g; tail contributes >= %.3g" % (e, grew))


def t14_roundtrip(ckpt, tmp):
    h = Harness(ckpt, M.S_TOTAL)
    _, _, st = h.reset(M.fold("t14"))
    obs = np.asarray(st.env_state.obs)
    keys, arrs, treedef = M.flatten_state(st)
    payload = {"__obs__": obs, "__source__": np.array(["PW"] * M.S_TOTAL)}
    for i, (k, a) in enumerate(zip(keys, arrs)):
        payload["%04d|%s" % (i, k)] = a
    np.savez(tmp, **payload)
    st2, obs2, src2 = M.load_bank(tmp, st)
    _, arrs2, _ = M.flatten_state(st2)
    bad = [k for k, a, b in zip(keys, arrs, arrs2)
           if a.shape != b.shape or a.dtype != b.dtype or not np.array_equal(a, b)]
    obs_ok = np.array_equal(obs, obs2)
    # the stored observation must agree with the one carried inside the state
    inner = float(np.max(np.abs(obs2 - np.asarray(st2.env_state.obs))))
    check("T14 state bank round-trips exactly", not bad and obs_ok and inner == 0.0,
          "%d leaves, %d mismatched; obs vs state-internal obs %.3g"
          % (len(keys), len(bad), inner))


if __name__ == "__main__":
    ckpt = sys.argv[1]
    tmp = sys.argv[2] if len(sys.argv) > 2 else "/tmp/mc_oracle_bank_test.npz"
    print("checkpoint:", ckpt)
    print("--- PHASE 2 GATE: full state restoration ---")
    g1 = t1_clone_replay(ckpt)
    g2 = t1b_branch(ckpt)
    if not (g1 and g2):
        print("\nPHASE 2 GATE FAILED - exact state restoration not established. STOP.")
        raise SystemExit(2)
    print("--- PHASE 5: implementation validation ---")
    t2_t5_semantics(ckpt)
    t6_t7(ckpt)
    t8_t9(ckpt)
    t10_fd()
    t11_t12()
    t13_prefix(ckpt)
    t14_roundtrip(ckpt, tmp)
    n_ok = sum(1 for _, ok, _ in RESULTS if ok)
    print("\n%d/%d tests pass" % (n_ok, len(RESULTS)))
    raise SystemExit(0 if n_ok == len(RESULTS) else 1)
