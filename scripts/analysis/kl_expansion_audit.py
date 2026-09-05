#!/usr/bin/env python3
"""KL expansion audit: does the forward-KL gate term push pre-tanh sigma up?

Offline only. Loads exported checkpoints read-only, never mutates them, never trains.

Loss structure traced from src/jaxrl/reppo.py (not assumed):
  pathwise      objective = log_prob * sg(temperature) - value          (:1049)
                  entropy term  alpha*log_prob  acts on the policy;
                  pathwise-Q    -value
  weighted_mle  objective = -sum_i w_i * logp_theta_i                   (:985)
                  no policy-entropy term; target_entropy_loss (:1097) carries
                  stop_gradient(target_entropy) so it moves only the temperature.
  both          actor_kl_clip_mode == "clipped" (:1084):
                  where(kl < kl_bound, objective, kl*sg(lagrangian)*reduce_kl)
  kl            forward, sampled, reppo.py:889-917. logp_old at the UNCLIPPED sample,
                  logp_theta at the action clipped to +-(1-1e-4).
"""
import json, os, sys
import numpy as np
import jax, jax.numpy as jnp
import distrax
from flax import nnx
import optax

sys.path.insert(0, os.getcwd())
from scripts.load_ckpt import load

BANK      = "reports/artifacts/walker_fixed_state_bank.npz"
EXPORTS   = "exports"
SEEDS     = [int(x) for x in os.environ.get("KLAUDIT_SEEDS",
             "301,302,303,304,305,306,307,308").split(",")]
ARM_TAG   = {"pathwise": "pathwise_fa", "weighted_mle": "weighted_mle"}
RUN_TAG   = {"pathwise": "PW1", "weighted_mle": "WML32"}
FRACS     = [("p25", 5), ("p50", 10), ("final", 20)]
KL_BOUND  = 0.1
REDUCE_KL = 1.0
M         = 32
LR        = 3e-4
MAXGN     = 0.5
CLIP_A    = 1.0 - 1e-4
Y_CLIP    = float(np.arctanh(CLIP_A))
RUNDIR    = "/rwthfs/rz/cluster/hpcwork/qzi10910/reppo_runs/outputs/faithful_repair"
SEED_RNG  = 20260905
OUT       = "reports/artifacts/kl_expansion_audit.json"
D_ACT     = 6

res = {"config": dict(kl_bound=KL_BOUND, reduce_kl=REDUCE_KL, M=M, lr=LR,
                      max_grad_norm=MAXGN, rng=SEED_RNG, y_clip=Y_CLIP,
                      bank=BANK, fracs=[f for f, _ in FRACS])}

# ----------------------------------------------------------------- part 1
def kl_closed(mu_o, sg_o, mu_n, sg_n):
    return jnp.log(sg_n / sg_o) + (sg_o**2 + (mu_n - mu_o)**2) / (2 * sg_n**2) - 0.5

def kl_closed_reverse(mu_o, sg_o, mu_n, sg_n):
    return jnp.log(sg_o / sg_n) + (sg_n**2 + (mu_n - mu_o)**2) / (2 * sg_o**2) - 0.5

def repo_kl(mu_o, sg_o, mu_n, sg_n, key, m):
    old = distrax.Transformed(distrax.Normal(mu_o, sg_o), distrax.Tanh())
    new = distrax.Transformed(distrax.Normal(mu_n, sg_n), distrax.Tanh())
    a, lp_old = old.sample_and_log_prob(seed=key, sample_shape=(m,))
    a = jnp.clip(a, -CLIP_A, CLIP_A)
    return lp_old.mean(0) - new.log_prob(a).mean(0)

def part1():
    print("\n" + "=" * 74 + "\n1. ANALYTIC\n" + "=" * 74)
    grid = [(0.4, 0.0, 0.4), (0.4, 0.3, 0.5), (0.4, 0.3, 0.8), (1.0, 0.5, 0.7),
            (0.5, 0.0, 2.0), (0.3, 0.9, 0.35), (2.0, 1.0, 3.0), (0.45, 0.2, 0.45)]
    rows, worst = [], 0.0
    print("  sg_o  dmu  sg_n |      hand    autodiff     |err|  |   sig*   pressure")
    for sg_o, dmu, sg_n in grid:
        hand = 1.0 - (sg_o**2 + dmu**2) / sg_n**2
        g = float(jax.grad(lambda t: kl_closed(0.0, sg_o, dmu, sg_n * jnp.exp(t)))(0.0))
        worst = max(worst, abs(g - hand))
        p = "WIDEN" if hand < 0 else ("CONTRACT" if hand > 0 else "NONE")
        rows.append(dict(sg_o=sg_o, dmu=dmu, sg_n=sg_n, hand=hand, autodiff=g,
                         abs_err=abs(g - hand), fixed_point=float(np.sqrt(sg_o**2 + dmu**2)),
                         pressure=p))
        print("  %.2f %5.2f %5.2f | %+9.6f %+11.6f  %.2e | %6.4f  %s"
              % (sg_o, dmu, sg_n, hand, g, abs(g - hand), rows[-1]["fixed_point"], p))
    closed_pass = worst < 1e-5

    print("\n  repo sampled estimator (reppo.py:889-917), 400k draws per cell:")
    mc_rows, mc_worst = [], 0.0
    for sg_o, dmu, sg_n in grid:
        hand = 1.0 - (sg_o**2 + dmu**2) / sg_n**2
        k = jax.random.PRNGKey(0)
        g = float(jax.grad(lambda t: repo_kl(0.0, sg_o, dmu, sg_n * jnp.exp(t), k, 400_000))(0.0))
        wide = sg_n > 1.5 or sg_o > 1.5
        if not wide:
            mc_worst = max(mc_worst, abs(g - hand))
        mc_rows.append(dict(sg_o=sg_o, dmu=dmu, sg_n=sg_n, hand=hand, autodiff_mc=g,
                            abs_err=abs(g - hand), clip_regime=bool(wide)))
        print("    %.2f %5.2f %5.2f | hand %+9.6f  MC %+9.6f  |err| %.2e%s"
              % (sg_o, dmu, sg_n, hand, g, abs(g - hand),
                 "   <- tanh-clip regime" if wide else ""))
    verdict = "PASS" if (closed_pass and mc_worst < 5e-3) else "FAIL"
    print("\n  ANALYTIC_AUTODIFF_MATCH = %s" % verdict)
    print("    closed form  max|err| %.2e (tol 1e-5) ; repo sampled max|err| %.2e "
          "over non-clip cells (tol 5e-3)" % (worst, mc_worst))
    print("\n  FIXED POINT  d KL/d log sigma_n = 0  <=>  sigma_n^2 = sigma_o^2 + dmu^2")
    print("               sigma_n* = sqrt(sigma_o^2 + dmu^2) >= sigma_o, equality iff dmu=0")
    print("  SIGN         descent moves log sigma by -grad. When")
    print("               sigma_n^2 < sigma_o^2 + dmu^2 the gradient is NEGATIVE, the step")
    print("               is POSITIVE, and sigma INCREASES. Forward KL is minimised by widening.")
    print("  CONTRAST     reverse KL(new||old): d/dlog sigma_n = -1 + sigma_n^2/sigma_o^2,")
    print("               fixed point sigma_n* = sigma_o exactly. No widening incentive.")
    rk = float(jax.grad(lambda t: kl_closed_reverse(0.0, 0.4, 0.3, 0.4 * jnp.exp(t)))(0.0))
    print("               check sg_o=sg_n=0.4 dmu=0.3: %+.2e (expect 0)" % rk)
    print("  CEILING      the action clip at %.4f caps |y| at atanh = %.4f, so once"
          % (CLIP_A, Y_CLIP))
    print("               sigma_n >> %.2f the sampled second moment stops tracking sigma_n^2"
          % Y_CLIP)
    print("               and the same term becomes CONTRACTING. Quantified in part 2.")
    res["part1"] = dict(closed_form=rows, repo_sampled=mc_rows,
                        ANALYTIC_AUTODIFF_MATCH=verdict, closed_max_abs_err=worst,
                        mc_max_abs_err_nonclip=mc_worst, reverse_kl_check=rk)
    return verdict

# ----------------------------------------------------------------- helpers
def logp_tanh(y, mu, sg):
    lp = -0.5 * ((y - mu) / sg) ** 2 - jnp.log(sg) - 0.5 * jnp.log(2 * jnp.pi)
    return lp - (2.0 * (jnp.log(2.0) - y - jax.nn.softplus(-2.0 * y)))

def logp_tanh_at_a(a, mu, sg):
    return logp_tanh(jnp.arctanh(jnp.clip(a, -CLIP_A, CLIP_A)), mu, sg)

def make_norm(ckpt_dir, meta):
    nz = np.load(os.path.join(ckpt_dir, "normalizer.npz"))
    eps = meta["normalizer_eps"]
    m_, v_ = jnp.asarray(nz["mean"]), jnp.asarray(nz["var"])
    if not meta.get("normalize_env", True):
        return lambda s: jnp.asarray(s)
    return lambda s: (jnp.asarray(s) - m_) / jnp.sqrt(v_ + eps)

def logged(arm, seed):
    p = os.path.join(RUNDIR, "walker_%s_s%d" % (RUN_TAG[arm], seed), "metrics.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p)
    g = lambda k: np.asarray(z[k]).reshape(-1) if k in z.files else None
    return dict(sigma=g("train/pi_sigma_mean"), kl=g("train/kl"),
                gate=g("train/fr_gate_operator"), q25=g("train/fr_kl_q25"),
                q50=g("train/fr_kl_q50"), q75=g("train/fr_kl_q75"))

def sampled_kl(mu, sg, mu_o, sg_o, eps):
    """The gate's own estimator (reppo.py:889-917): M draws from pi_old, action clipped."""
    y = mu_o[None] + sg_o[None] * eps
    a = jnp.clip(jnp.tanh(y), -CLIP_A, CLIP_A)
    lp_old = logp_tanh(y, mu_o[None], sg_o[None]).sum(-1)
    lp_th = logp_tanh_at_a(a, mu[None], sg[None]).sum(-1)
    return lp_old.mean(0) - lp_th.mean(0)

def calibrate_old(mu, sg, seed, q25, q50, q75, drift, mu_o_fixed=None):
    """Surrogate for the frozen actor_target one outer iteration back.

    actor_target is NOT exported, so pi_old cannot be reconstructed exactly. It is
    rebuilt as
        sigma_old = sigma * exp(-drift + jitter),  drift = the run's OWN signed
                    per-iteration d log sigma here, so a widening run gets the
                    sigma_old < sigma it actually had;
        mu_old    = mu + c_n * sigma * z,  c_n lognormal(spread sp) across states.
    The two parameters (c, sp) are fitted so the bank reproduces that seed's logged
    per-state ANALYTIC KL distribution -- median to fr_kl_q50, interquartile ratio to
    fr_kl_q75/fr_kl_q25. A single c gives only chi-square spread (q75/q25 ~ 2.3), far
    narrower than the logged ~14, which is why the spread parameter is needed.

    The gate-fire fraction and the mean SAMPLED KL are then PREDICTIONS, reported
    against logged fr_gate_operator and train/kl as validation rather than fitted.
    Note the gate fires on the sampled M=32 KL, whose MC noise makes it broader than
    the analytic KL the quantiles describe: logged fr_kl_q50 (0.079) sits below the
    0.1 bound while fr_gate_operator is 0.55, which only the sampled estimator explains.
    """
    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(seed), 4)
    z_mu = jax.random.normal(k1, mu.shape)
    xi   = jax.random.normal(k2, (mu.shape[0], 1))
    jit  = max(abs(float(drift)), 1e-3)
    sg_old = sg * jnp.exp(-drift + jit * jax.random.normal(k3, sg.shape))
    eps  = jax.random.normal(k4, (M, *mu.shape))

    def analytic(c, sp):
        cn = c * jnp.exp(sp * xi - 0.5 * sp * sp)
        return jnp.sum(kl_closed(mu + cn * sg * z_mu, sg_old, mu, sg), axis=-1)

    if mu_o_fixed is not None:                 # sweep: hold |dmu| fixed, vary sigma
        k = sampled_kl(mu, sg, mu_o_fixed, sg_old, eps)
        return (mu_o_fixed, sg_old, float("nan"), float("nan"),
                float(jnp.mean(k)), float(jnp.mean(k >= KL_BOUND)))

    def fit_c(sp):                             # match the logged median
        lo, hi = 0.0, 12.0
        for _ in range(34):
            m = 0.5 * (lo + hi)
            if float(jnp.median(analytic(m, sp))) < q50: lo = m
            else: hi = m
        return 0.5 * (lo + hi)

    tgt_iqr = float(np.log(max(q75, 1e-9) / max(q25, 1e-9)))
    lo, hi = 0.0, 3.0                          # spread grows monotonically with sp
    for _ in range(20):
        sp = 0.5 * (lo + hi)
        k = analytic(fit_c(sp), sp)
        got = float(jnp.log(jnp.quantile(k, 0.75) / jnp.maximum(jnp.quantile(k, 0.25), 1e-9)))
        if got < tgt_iqr: lo = sp
        else: hi = sp
    sp = 0.5 * (lo + hi); c = fit_c(sp)
    cn = c * jnp.exp(sp * xi - 0.5 * sp * sp)
    mo = mu + cn * sg * z_mu
    ks = sampled_kl(mu, sg, mo, sg_old, eps)
    return (mo, sg_old, float(c), float(sp), float(jnp.mean(ks)),
            float(jnp.mean(ks >= KL_BOUND)))

# ----------------------------------------------------------------- parts 2 and 3
def analyse(ck, d, arm, bank, mu_o, sg_o, key):
    mu, sg = ck.policy_dist(bank)
    return analyse_at(ck, d, arm, bank, mu, sg, mu_o, sg_o, key)

def analyse_at(ck, d, arm, bank, mu, sg, mu_o, sg_o, key):
    nrm = make_norm(d, ck.meta)
    alpha = float(jnp.squeeze(ck.actor.temperature()))
    lagr  = float(jnp.squeeze(ck.actor.lagrangian()))
    eta   = max(float(jnp.squeeze(ck.actor.eta())), 1e-4) \
            if any("eta_param" in p for p in ck.meta["actor_leaf_paths"]) else 1.0
    kz, kp = jax.random.split(key)
    y_i = mu_o[None] + sg_o[None] * jax.random.normal(kz, (M, *mu.shape))
    a_i = jnp.clip(jnp.tanh(y_i), -CLIP_A, CLIP_A)
    lp_old_i = logp_tanh(y_i, mu_o[None], sg_o[None]).sum(-1)
    z_pw = jax.random.normal(kp, mu.shape)
    q_i = jax.vmap(lambda aa: ck.q_scalar(bank, aa))(a_i)
    w_i = jax.nn.softmax(q_i / eta, axis=0)
    clip_frac = float(jnp.mean(jnp.abs(y_i) > Y_CLIP))
    # the step-1 fixed point, per state per coordinate: sigma* = sqrt(sg_o^2 + dmu^2).
    # sigma < sigma*  <=>  the exact (unclipped) KL gradient is negative  <=>  WIDEN.
    fp = jnp.sqrt(sg_o**2 + (mu - mu_o)**2)
    fp_ratio = sg / fp
    zero = jnp.zeros(mu.shape)
    gsum = lambda f: jax.grad(lambda t: jnp.sum(f(t)))(zero)

    def kl_term(t):
        s = sg * jnp.exp(t)
        return lp_old_i.mean(0) - logp_tanh_at_a(a_i, mu[None], s[None]).sum(-1).mean(0)
    def kl_noclip(t):                       # same term with the action clip removed
        s = sg * jnp.exp(t)
        return lp_old_i.mean(0) - logp_tanh(y_i, mu[None], s[None]).sum(-1).mean(0)
    def wml_obj(t):
        s = sg * jnp.exp(t)
        lt = logp_tanh_at_a(a_i, mu[None], s[None]).sum(-1)
        return -jnp.sum(jax.lax.stop_gradient(w_i) * lt, axis=0)
    def pw_q(t):
        s = sg * jnp.exp(t)
        return -ck.q_scalar(bank, jnp.clip(jnp.tanh(mu + s * z_pw), -CLIP_A, CLIP_A))
    def pw_ent(t):
        s = sg * jnp.exp(t)
        return alpha * logp_tanh(mu + s * z_pw, mu, s).sum(-1)

    kl_val = kl_term(zero)
    g_kl = gsum(kl_term) * lagr * REDUCE_KL
    g_kl_nc = gsum(kl_noclip) * lagr * REDUCE_KL
    if arm == "weighted_mle":
        g_imp, g_ent = gsum(wml_obj), jnp.zeros(mu.shape)
    else:
        g_imp, g_ent = gsum(pw_q), gsum(pw_ent)
    return dict(fp_ratio=fp_ratio, mu=mu, sg=sg, nrm=nrm, alpha=alpha, lagr=lagr, eta=eta, a_i=a_i,
                lp_old_i=lp_old_i, w_i=w_i, z_pw=z_pw, kl=kl_val, clip_frac=clip_frac,
                g_imp=g_imp, g_ent=g_ent, g_kl=g_kl, g_kl_noclip=g_kl_nc)

def step_deltas(ck, d, arm, bank, A):
    """One optimizer step per component on temporary copies. No checkpoint mutated."""
    gdef, state, rest = nnx.split(ck.actor, nnx.Param, ...)
    base = float(np.median(np.asarray(A["sg"])))
    a_i, lp_old_i, w_i, z_pw = A["a_i"], A["lp_old_i"], A["w_i"], A["z_pw"]
    alpha, lagr, nrm = A["alpha"], A["lagr"], A["nrm"]

    def pieces(st):
        m = nnx.merge(gdef, st, rest)
        loc = m.actor_module(nrm(bank))
        mu, log_std = jnp.split(loc, 2, axis=-1)
        sg = jnp.exp(log_std) + m.min_std
        lt = logp_tanh_at_a(a_i, mu[None], sg[None]).sum(-1)
        kl = lp_old_i.mean(0) - lt.mean(0)
        if arm == "weighted_mle":
            imp = -jnp.sum(jax.lax.stop_gradient(w_i) * lt, axis=0)
            ent = jnp.zeros_like(imp)
        else:
            y = mu + sg * z_pw
            imp = -ck.q_scalar(bank, jnp.clip(jnp.tanh(y), -CLIP_A, CLIP_A))
            ent = alpha * logp_tanh(y, mu, sg).sum(-1)
        return imp, ent, imp + ent, kl * lagr * REDUCE_KL, kl < KL_BOUND

    def loss_of(which):
        def L(st):
            imp, ent, obj, klt, op = pieces(st)
            if which == "improvement": return jnp.mean(jnp.where(op, imp, 0.0))
            if which == "entropy":     return jnp.mean(jnp.where(op, ent, 0.0))
            if which == "kl":          return jnp.mean(jnp.where(op, 0.0, klt))
            # ungated: the gate closes on ~97% of states in WML, so the gated
            # improvement step is measured on a ~3% subset. These use every state.
            if which == "improvement_all": return jnp.mean(imp)
            if which == "entropy_all":     return jnp.mean(ent)
            if which == "kl_all":          return jnp.mean(klt)
            return jnp.mean(jnp.where(op, obj, klt))
        return L

    def med_after(st2):
        m2 = nnx.merge(gdef, st2, rest)
        _, ls2 = jnp.split(m2.actor_module(nrm(bank)), 2, axis=-1)
        return float(np.median(np.asarray(jnp.exp(ls2) + m2.min_std)))

    out = {}
    for which in ("improvement", "entropy", "kl", "full",
                  "improvement_all", "entropy_all", "kl_all"):
        gr = jax.grad(loss_of(which))(state)
        tx = optax.chain(optax.clip_by_global_norm(MAXGN), optax.adam(LR))
        upd, _ = tx.update(gr, tx.init(state), state)
        out["adam_" + which] = med_after(jax.tree.map(lambda a, b: a + b, state, upd)) - base
        # plain descent keeps gradient magnitude information, which adam normalises away
        out["sgd_" + which] = med_after(
            jax.tree.map(lambda a, g: a - LR * g, state, gr)) - base
    out["sigma0"] = base
    return out

def bank_rows(bank, arm, mode):
    """Neutral = all 3072 states (the arm-comparison protocol). On-arm = the 1536 states
    that arm's own rollouts visited, which is the population the training gradient
    actually saw. The bank is arm-major: PW first, WML second."""
    if mode == "neutral":
        return bank
    h = bank.shape[0] // 2
    return bank[:h] if arm == "pathwise" else bank[h:]

def parts_23(bank):
    print("\n" + "=" * 74 + "\n2. DECOMPOSE   d/d log sigma per state. DESCENT: grad<0 => WIDEN\n"
          + "=" * 74)
    print("  GATE_OPEN   kl <  %.2f  -> improvement branch" % KL_BOUND)
    print("  GATE_CLOSED kl >= %.2f  -> KL branch (improvement discarded)\n" % KL_BOUND)
    print("  arm  frac  sd | sigma0 | gate_cl | clipfr |    g_imp     g_ent      g_kl  "
          "| g_kl noclip | sg/sg* | frac<1")
    r2, r3 = [], []
    for mode in ("on_arm", "neutral"):
      for arm in ARM_TAG:
        for frac, idx in FRACS:
            for sd in SEEDS:
                bk = bank_rows(bank, arm, mode)
                d = os.path.join(EXPORTS, "WalkerRun_%s_s%d_%s" % (ARM_TAG[arm], sd, frac))
                if not os.path.isdir(d):
                    print("  %-4s %-5s %d | MISSING %s" % (arm[:4], frac, sd, d)); continue
                ck = load(d)
                L = logged(arm, sd)
                if not L or L["kl"] is None or L["sigma"] is None:
                    print("  %-4s %-5s %d | NO LOGGED METRICS -- skipped (no fabricated "
                          "targets)" % (arm[:4], frac, sd)); continue
                j = min(idx, len(L["kl"]) - 1)
                tkl = float(L["kl"][j])
                tgt = float(L["gate"][j]) if L["gate"] is not None else float("nan")
                qq = [float(L[q][j]) for q in ("q25", "q50", "q75")]
                mu, sg = ck.policy_dist(bk)
                lg = np.log(np.maximum(L["sigma"], 1e-8)) if L and L["sigma"] is not None else None
                jj = min(idx, len(lg) - 1) if lg is not None else None
                drift = float((lg[jj] - lg[max(jj - 1, 0)]) / 20.0) if jj else 0.0
                mo, so, c, sp, gotkl, gotgate = calibrate_old(
                    mu, sg, sd * 100 + idx, qq[0], qq[1], qq[2], drift)
                A = analyse(ck, d, arm, bk, mo, so, jax.random.PRNGKey(SEED_RNG + sd + idx))
                kl_s = np.asarray(A["kl"]); closed = kl_s >= KL_BOUND
                gi, ge, gk = (np.asarray(A["g_imp"]), np.asarray(A["g_ent"]),
                              np.asarray(A["g_kl"]))
                gnc = np.asarray(A["g_kl_noclip"]); fpr = np.asarray(A["fp_ratio"])
                fu = float(np.mean(gk[closed] < 0)) if closed.any() else float("nan")
                fu_nc = float(np.mean(gnc[closed] < 0)) if closed.any() else float("nan")
                pm = lambda x: "WIDEN" if np.median(x) < 0 else "CONTR"
                row = dict(mode=mode, arm=arm, frac=frac, seed=sd, sigma0=float(np.median(np.asarray(sg))),
                           target_kl=tkl, achieved_kl=gotkl, target_gate=tgt, drift=drift,
                           achieved_gate=gotgate, cal_c=c, cal_spread=sp,
                           gate_closed_frac=float(closed.mean()), clip_frac=A["clip_frac"],
                           alpha=A["alpha"], lagrangian=A["lagr"], eta=A["eta"],
                           g_imp_med=float(np.median(gi)), g_ent_med=float(np.median(ge)),
                           g_kl_med=float(np.median(gk)),
                           g_kl_noclip_med=float(np.median(gnc)),
                           g_imp_med_closed=float(np.median(gi[closed])) if closed.any() else None,
                           g_kl_med_closed=float(np.median(gk[closed])) if closed.any() else None,
                           frac_kl_up_closed=fu, frac_kl_up_closed_noclip=fu_nc,
                           fp_ratio_med=float(np.median(fpr)),
                           frac_below_fixed_point=float(np.mean(fpr < 1.0)),
                           pressure_imp=pm(gi), pressure_ent=pm(ge), pressure_kl=pm(gk))
                r2.append(row)
                print("  %-4s %-5s %d | %6.3f |  %.3f  | %.4f | %+9.4f %+9.4f %+9.4f "
                      "| %+11.4f | %6.4f | %.4f"
                      % (arm[:4], frac, sd, row["sigma0"], closed.mean(), A["clip_frac"],
                         np.median(gi), np.median(ge), np.median(gk), np.median(gnc),
                         np.median(fpr), float(np.mean(fpr < 1.0))))
                st = step_deltas(ck, d, arm, bk, A)
                r3.append(dict(mode=mode, arm=arm, frac=frac, seed=sd, **st))
    res["part2"], res["part3"] = r2, r3

    print("\n  --- part 2 medians over seeds ---")
    print("  arm            frac  | sigma0 | gate_cl(log) | clipfr |   g_imp    g_ent     g_kl "
          "| g_kl noclip | sg/sg* | frac<1 | frac_kl_up")
    for mode in ("on_arm", "neutral"):
      print("  -- bank: %s --" % mode)
      for arm in ARM_TAG:
        for frac, _ in FRACS:
            rr = [r for r in r2 if r["arm"] == arm and r["frac"] == frac
                  and r["mode"] == mode]
            if not rr: continue
            md = lambda k: np.median([r[k] for r in rr])
            print("  %-14s %-5s | %6.3f | %.3f(%.3f)  | %.4f | %+8.4f %+8.4f %+8.4f "
                  "| %+11.4f | %6.4f | %.4f | %.4f"
                  % (arm, frac, md("sigma0"), md("achieved_gate"), md("target_gate"),
                     md("clip_frac"), md("g_imp_med"), md("g_ent_med"), md("g_kl_med"),
                     md("g_kl_noclip_med"), md("fp_ratio_med"),
                     md("frac_below_fixed_point"), md("frac_kl_up_closed")))

    print("\n" + "=" * 74 + "\n3. PARAMETER STEP   temporary copies only; no checkpoint mutated\n"
          + "=" * 74)
    print("  adam = chain(clip_by_global_norm(%.1f), adam(%.0e)); sgd = -lr*grad" % (MAXGN, LR))
    print("  delta = median pre-tanh sigma after one step, minus before\n")
    print("  arm            frac  | sigma0 | GATED adam: d_imp     d_ent      d_kl    d_full"
          "  | UNGATED adam: d_imp     d_ent      d_kl")
    for mode in ("on_arm", "neutral"):
      print("  -- bank: %s --" % mode)
      for arm in ARM_TAG:
        for frac, _ in FRACS:
            rr = [r for r in r3 if r["arm"] == arm and r["frac"] == frac
                  and r["mode"] == mode]
            if not rr: continue
            md = lambda k: np.median([r[k] for r in rr])
            print("  %-14s %-5s | %6.3f | %+16.3e %+9.3e %+9.3e %+9.3e | %+16.3e %+9.3e %+9.3e"
                  % (arm, frac, md("sigma0"), md("adam_improvement"), md("adam_entropy"),
                     md("adam_kl"), md("adam_full"), md("adam_improvement_all"),
                     md("adam_entropy_all"), md("adam_kl_all")))
    return r2, r3

# ------------------------------------------------------- part 2b: sigma sweep
def sweep(bank_full):
    """Counterfactual: rescale the checkpoint's sigma and re-measure each term.

    The corrected runs widen from sigma~0.44 to ~3 before the first saved checkpoint
    (p25), so no export sits in the narrow regime where the widening began. Scaling the
    policy's log_std by a constant and holding mu fixed reconstructs that regime from a
    real checkpoint and locates the sigma at which each term changes sign.
    """
    print("\n" + "=" * 74 + "\n2b. SIGMA SWEEP  (counterfactual rescaling of the "
          "checkpoint sigma)\n" + "=" * 74)
    print("  factor applied to sigma; pi_old recalibrated at each factor; medians over seeds\n")
    print("  arm            fac |  sigma  | clipfr |   g_imp     g_ent      g_kl "
          "| g_kl noclip | sg/sg* | frac<1")
    rows = []
    for arm in ARM_TAG:
        for fac in (0.1, 0.25, 0.5, 1.0, 2.0, 4.0):
            acc = []
            for sd in SEEDS:
                d = os.path.join(EXPORTS, "WalkerRun_%s_s%d_p25" % (ARM_TAG[arm], sd))
                if not os.path.isdir(d): continue
                bank = bank_rows(bank_full, arm, "on_arm")
                ck = load(d)
                L = logged(arm, sd)
                if not L or L["kl"] is None: continue
                jj = min(5, len(L["kl"]) - 1)
                qq = [float(L[q][jj]) for q in ("q25", "q50", "q75")]
                mu, sg0 = ck.policy_dist(bank)
                mo1, _, _, _, _, _ = calibrate_old(mu, sg0, sd * 100 + 5, *qq, 0.0)
                sg = sg0 * fac                    # same absolute mean drift, new sigma
                mo, so, _, _, _, _ = calibrate_old(mu, sg, sd * 100 + 5, *qq, 0.0,
                                                   mu_o_fixed=mo1)
                A = analyse_at(ck, d, arm, bank, mu, sg, mo, so,
                               jax.random.PRNGKey(SEED_RNG + sd))
                acc.append((float(np.median(np.asarray(sg))), A["clip_frac"],
                            float(np.median(np.asarray(A["g_imp"]))),
                            float(np.median(np.asarray(A["g_ent"]))),
                            float(np.median(np.asarray(A["g_kl"]))),
                            float(np.median(np.asarray(A["g_kl_noclip"]))),
                            float(np.median(np.asarray(A["fp_ratio"]))),
                            float(np.mean(np.asarray(A["fp_ratio"]) < 1.0))))
            if not acc: continue
            m = np.median(np.asarray(acc), axis=0)
            rows.append(dict(arm=arm, factor=fac, sigma=m[0], clip_frac=m[1],
                             g_imp=m[2], g_ent=m[3], g_kl=m[4], g_kl_noclip=m[5],
                             fp_ratio=m[6], frac_below_fp=m[7]))
            print("  %-14s %4.2f | %7.4f | %.4f | %+8.4f %+9.4f %+9.4f | %+11.4f | %6.4f "
                  "| %.4f" % (arm, fac, m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7]))
    res["part2b"] = rows
    return rows

# ----------------------------------------------------------------- part 4
def part4():
    print("\n" + "=" * 74 + "\n4. RATCHET CHECK   21 logged eval points, each spanning 20 "
          "iterations\n" + "=" * 74)
    print("  Step-1 fixed point accumulated across iterations predicts")
    print("    sigma_{k+1}^2 = sigma_k^2 + dmu_k^2   =>   growth per iter = sqrt(1+(dmu/sigma)^2)")
    print("  Forward KL over d=%d dims with sigma_new ~ sigma_old gives (dmu/sigma)^2 ~ 2*kl/d.\n"
          % D_ACT)
    print("  arm            sd | kl_mean | (dmu/sg)^2 | gate | pred x/iter | pred cum | "
          "obs sigma_0->max->T | obs max/init")
    rows = []
    for arm in ARM_TAG:
        for sd in SEEDS:
            L = logged(arm, sd)
            if not L or L["kl"] is None or L["sigma"] is None: continue
            kl = np.asarray(L["kl"], float); sig = np.asarray(L["sigma"], float)
            gate = np.asarray(L["gate"], float) if L["gate"] is not None else None
            klm = float(np.mean(kl)); disp = 2.0 * klm / D_ACT
            per_it = float(np.sqrt(1.0 + disp))
            gf = float(np.mean(gate)) if gate is not None else 1.0
            pred = per_it ** (400 * gf)
            rows.append(dict(arm=arm, seed=sd, kl_mean=klm, disp_ratio=disp, gate_frac=gf,
                             pred_growth_per_iter=per_it, pred_cum_gated=pred,
                             sigma_init=float(sig[0]), sigma_max=float(sig.max()),
                             sigma_final=float(sig[-1]),
                             obs_growth_final=float(sig[-1] / sig[0]),
                             obs_growth_max=float(sig.max() / sig[0]),
                             ratio_obs_over_pred=float(sig.max() / sig[0]) / pred))
            print("  %-14s %d | %7.4f | %10.5f | %.3f | %11.5f | %8.2e | %5.3f->%6.3f->%6.3f "
                  "| %8.2f" % (arm, sd, klm, disp, gf, per_it, pred, sig[0], sig.max(),
                               sig[-1], sig.max() / sig[0]))
    res["part4"] = rows
    print()
    for arm in ARM_TAG:
        rr = [r for r in rows if r["arm"] == arm]
        if not rr: continue
        print("  %-14s median: obs max/init %8.2f x | unconstrained-ratchet pred %8.2e x "
              "| obs/pred %.3e" % (arm, np.median([r["obs_growth_max"] for r in rr]),
                                   np.median([r["pred_cum_gated"] for r in rr]),
                                   np.median([r["ratio_obs_over_pred"] for r in rr])))
    return rows

def main():
    v1 = part1()
    z = np.load(BANK)
    k = "states" if "states" in z.files else z.files[0]
    bank = jnp.asarray(z[k])
    print("\n  bank %s  shape %s" % (BANK, tuple(bank.shape)))
    ck0 = load(os.path.join(EXPORTS, "WalkerRun_weighted_mle_s%d_final" % SEEDS[0]))
    print("  actor min_std = %s   (cfg actor_min_std was 0.0)" % ck0.actor.min_std)
    res["min_std"] = float(ck0.actor.min_std)
    parts_23(bank)
    sweep(bank)
    part4()
    res["part1_verdict"] = v1
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1, default=float)
    print("\n  wrote %s" % OUT)

if __name__ == "__main__":
    main()
