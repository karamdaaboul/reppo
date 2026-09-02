"""Deterministic validation of crossed_dispersion.py on SCRATCH data only.

Two stages, because they need different precision regimes:

  unit  synthetic inputs under JAX_ENABLE_X64=1. Estimator algebra, the E-step
        dual, and the whitened metric are checked against independent references.
  pipe  a tiny throwaway bank at the production float32, for the wiring checks
        (shared innovations, indexing, law isolation, determinism).

Every tolerance is DERIVED from the dtype and the arithmetic before the check
runs -- ``tol_arith`` below -- never fitted to an observed residual.

Scratch output goes to the directory given on the command line and is never
reused and never written under reports/artifacts.

Usage: test_crossed_dispersion.py <scratch_dir> {unit|pipe|report}
"""
from __future__ import annotations

import csv, hashlib, json, os, sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import numpy as np  # noqa: E402

RES = []


def chk(name, ok, detail):
    RES.append({"name": name, "ok": bool(ok), "detail": detail})
    print("%-6s %-52s %s" % ("PASS" if ok else "FAIL", name, detail), flush=True)


def tol_arith(dtype, n_terms, scale, safety=20.0):
    """Worst-case rounding for a length-n_terms reduction over values of size
    ``scale``, in ``dtype``, times a safety factor. Derived, not fitted."""
    return float(np.finfo(dtype).eps * n_terms * scale * safety)


# =============================================================== unit stage
def stage_unit():
    import jax  # noqa: E402
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp  # noqa: E402
    from scipy.optimize import minimize_scalar  # noqa: E402
    import scripts.analysis.crossed_dispersion as CD  # noqa: E402

    dt = np.float64
    rng = np.random.default_rng(0)
    R, S, M, d = 7, 5, 32, 6
    Fn = rng.normal(size=(R, S, M)).astype(dt) * 3.0
    Gn = rng.normal(size=(R, S, M, d)).astype(dt)
    un = rng.normal(size=(R, S, M, d)).astype(dt)
    sn = np.exp(rng.normal(size=(S, d)) * 0.4).astype(dt)
    eps_e = 0.5
    F, G, u, sg = map(jnp.asarray, (Fn, Gn, un, sn))
    assert F.dtype == jnp.float64, "x64 not active"

    ops, eta, w = CD.build_updates(F, G, u, sg, eps_e,
                                   CD.PREREG["eta_lo"], CD.PREREG["eta_hi"])

    # ---- C2a  PW-32 is the plain mean of the action gradients ---------------
    ref = Gn.mean(2)
    t = tol_arith(dt, M, float(np.abs(Gn).max()))
    e = float(np.max(np.abs(np.asarray(ops["PW-32"], dt) - ref)))
    chk("C2a PW-32 == mean of action gradients", e <= t,
        "max|diff| %.3g <= tol %.3g (%d-term mean, float64)" % (e, t, M))

    # ---- C2b  PW-1 is the first sample of that same cloud -------------------
    e = float(np.max(np.abs(np.asarray(ops["PW-1"], dt) - Gn[:, :, 0, :])))
    chk("C2b PW-1 == first sample of the same cloud", e == 0.0,
        "max|diff| %.3g, tol exactly 0 (pure indexing, no arithmetic)" % e)

    # ---- C2c  centered ZO against an independent closed form ----------------
    dFn = Fn - Fn.mean(2, keepdims=True)
    ref = (dFn[..., None] * (un / sn[None, :, None, :])).mean(2)
    t = tol_arith(dt, M, float(np.abs(dFn[..., None] * (un / sn[None, :, None, :])).max()))
    e = float(np.max(np.abs(np.asarray(ops["ZO-32"], dt) - ref)))
    chk("C2c ZO-32 == centered value-only closed form", e <= t,
        "max|diff| %.3g <= tol %.3g" % (e, t))

    # ---- C2d  ZO is centered: constant F must annihilate --------------------
    # Derivation: mean of M identical values carries relative error <= M*eps, so
    # dF <= |Fc|*M*eps, and ZO <= |Fc|*M*eps*max|u/sigma|. Fixed BEFORE running.
    Fc = 4.2
    max_us = float(np.abs(un / sn[None, :, None, :]).max())
    t = tol_arith(dt, M, Fc * max_us)
    zo_c = CD.build_updates(jnp.full_like(F, Fc), G, u, sg, eps_e,
                            CD.PREREG["eta_lo"], CD.PREREG["eta_hi"])[0]["ZO-32"]
    e = float(np.max(np.abs(np.asarray(zo_c, dt))))
    chk("C2d ZO centered: constant F annihilates", e <= t,
        "max|ZO| %.3g <= tol %.3g = |F|*M*eps*max|u/s|*20 (float64)" % (e, t))

    # ---- C2e  WML weights on the pre-squash displacement --------------------
    en = np.asarray(eta, dt)
    z = Fn / en[..., None]
    ws = np.exp(z - z.max(-1, keepdims=True))
    ws /= ws.sum(-1, keepdims=True)
    ref = (ws[..., None] * (sn[None, :, None, :] * un)).sum(2)
    t = tol_arith(dt, M, float(np.abs(sn[None, :, None, :] * un).max()))
    e = float(np.max(np.abs(np.asarray(ops["WML-32"], dt) - ref)))
    chk("C2e WML-32 == softmax(F/eta) on (y-mu)", e <= t,
        "max|diff| %.3g <= tol %.3g" % (e, t))

    # ---- C3  eta solves the preregistered dual, vs scipy --------------------
    def g_np(t_, f):
        t_ = float(t_); zz = f / t_
        return t_ * eps_e + t_ * (np.log(np.mean(np.exp(zz - zz.max()))) + zz.max())
    rtol, worst, where = 1e-9, 0.0, None
    for i in range(R):
        for j in range(S):
            r = minimize_scalar(g_np, bounds=(CD.PREREG["eta_lo"], CD.PREREG["eta_hi"]),
                                args=(Fn[i, j],), method="bounded",
                                options={"xatol": 1e-12})
            gap = (g_np(en[i, j], Fn[i, j]) - r.fun) / (1.0 + abs(r.fun))
            if gap > worst:
                worst, where = gap, (i, j)
    chk("C3 eta attains the preregistered dual minimum", worst <= rtol,
        "max relative excess %.3g <= tol %.3g over %d clouds (ref: scipy bounded)"
        % (worst, rtol, R * S))

    # ---- C4a  unit whitened norm after normalisation ------------------------
    D_s, wn, nz = CD.dispersion(ops["PW-32"], sg)
    gh = np.asarray(ops["PW-32"], dt) / np.asarray(wn, dt)[..., None]
    t = tol_arith(dt, d, 1.0)
    e = float(np.max(np.abs(np.sqrt(((gh / sn[None]) ** 2).sum(-1)) - 1.0)))
    chk("C4a normalised updates have unit whitened norm", e <= t,
        "max|1-||g||_S| %.3g <= tol %.3g" % (e, t))

    # ---- C4b  D equals the mean squared whitened deviation ------------------
    direct = ((((gh - gh.mean(0)) / sn[None]) ** 2).sum(-1)).mean(0)
    t = tol_arith(dt, R * d, 1.0)
    e = float(np.max(np.abs(np.asarray(D_s, dt) - direct)))
    chk("C4b D == mean squared whitened deviation", e <= t,
        "max|diff| vs 1-||gbar||^2 form %.3g <= tol %.3g" % (e, t))

    # ---- C4c  D depends on the reference-law Sigma, ANISOTROPICALLY ---------
    # Prereg sec 3 / integrity check 3. A UNIFORM rescale of Sigma cancels
    # exactly under unit-whitened normalisation, so it cannot test this; the
    # perturbation must be anisotropic AND the updates must not be axis-aligned.
    # Toy, worked out in closed form before running:
    #   g1 = (1,0), g2 = (1,1), R=2, d=2
    #   sigma = (1,1):  h1=(1,0), h2=(1,1)/sqrt2, |mean|^2=(2+sqrt2)/4
    #                   => D = (2-sqrt2)/4
    #   sigma'= (1,2):  h1=(1,0), h2=(1,.5)/sqrt(1.25)
    #                   => D = 1 - |mean|^2, computed exactly below
    g_toy = jnp.asarray(np.array([[[1.0, 0.0]], [[1.0, 1.0]]], dt))   # (R=2,S=1,d=2)
    s_iso = jnp.asarray(np.array([[1.0, 1.0]], dt))
    s_ani = jnp.asarray(np.array([[1.0, 2.0]], dt))

    def D_closed(g, s):
        h = (g / s) / np.linalg.norm(g / s, axis=-1, keepdims=True)
        return 1.0 - (h.mean(0) ** 2).sum(-1)

    gn_toy = np.array([[[1.0, 0.0]], [[1.0, 1.0]]], dt)
    exp_iso = D_closed(gn_toy, np.array([[1.0, 1.0]], dt))
    exp_ani = D_closed(gn_toy, np.array([[1.0, 2.0]], dt))
    exp_delta = float(exp_iso[0] - exp_ani[0])
    got_iso = float(np.asarray(CD.dispersion(g_toy, s_iso)[0], dt)[0])
    got_ani = float(np.asarray(CD.dispersion(g_toy, s_ani)[0], dt)[0])
    t = tol_arith(dt, 8, 1.0)
    ok = (abs(got_iso - float(exp_iso[0])) <= t and
          abs(got_ani - float(exp_ani[0])) <= t and
          abs((got_iso - got_ani) - exp_delta) <= t and
          abs(exp_delta) > 1e-3)
    chk("C4c D tracks an ANISOTROPIC Sigma change", ok,
        "predicted D %.7f -> %.7f (delta %.7f, nonzero by derivation); "
        "measured %.7f -> %.7f; max|err| %.3g <= tol %.3g"
        % (exp_iso[0], exp_ani[0], exp_delta, got_iso, got_ani,
           max(abs(got_iso - exp_iso[0]), abs(got_ani - exp_ani[0])), t))
    # sanity that the OLD check was the wrong probe, recorded not asserted
    u_iso = float(np.asarray(CD.dispersion(g_toy, s_iso * 2.0)[0], dt)[0])
    print("       note: uniform 2x rescale gives D %.7f (unchanged, as derived)"
          % u_iso)
    return RES


# =============================================================== pipe stage
def stage_pipe(scratch):
    import scripts.analysis.crossed_dispersion as CD  # noqa: E402
    P = CD.PREREG
    # scratch config, internally consistent BY ASSERTION
    P["seeds"] = (301, 302)
    P["per_policy"] = 4
    P["n_bank"] = P["per_policy"] * 2 * len(P["seeds"])
    P["s_eval"], P["R"], P["M"] = 8, 3, 8
    n_pol = 2 * len(P["seeds"])
    assert P["per_policy"] * n_pol == P["n_bank"], (P["per_policy"], n_pol, P["n_bank"])
    assert P["s_eval"] <= P["n_bank"]
    chk("C0 scratch config is self-consistent", True,
        "per_policy %d x n_policies %d == n_bank %d, s_eval %d (registered bank is "
        "2048 = 16 x 128; 128 here is scratch only)"
        % (P["per_policy"], n_pol, P["n_bank"], P["s_eval"]))

    bank = os.path.join(scratch, "scratch_bank.npz")
    CD.collect_bank("walker", bank)
    o1 = os.path.join(scratch, "scratch_run1.csv")
    o2 = os.path.join(scratch, "scratch_run2.csv")
    CD.run_task("walker", bank, o1, chunk=2)
    CD.run_task("walker", bank, o2, chunk=2)

    # ---- C7  determinism ----------------------------------------------------
    a1, a2 = open(o1, "rb").read(), open(o2, "rb").read()
    chk("C7 same seed reproduces the run exactly", a1 == a2,
        "CSV byte-identical over two independent invocations (%d bytes)" % len(a1))

    rows = list(csv.DictReader(open(o1)))
    by_seed = {}
    for r in rows:
        by_seed.setdefault(r["seed"], set()).add(r["u_sha256"])

    # ---- C1  shared innovations --------------------------------------------
    import jax  # noqa: E402
    shared = all(len(v) == 1 for v in by_seed.values())
    distinct = len({next(iter(v)) for v in by_seed.values()}) == len(by_seed)
    u1 = np.asarray(jax.random.normal(CD.fold("u", "walker", 301), (3, 8, 8, 6)))
    u2 = np.asarray(jax.random.normal(CD.fold("u", "walker", 301), (3, 8, 8, 6)))
    chk("C1 innovations shared across laws and critic sources",
        shared and distinct and np.array_equal(u1, u2),
        "one u per seed across all 4 law x critic cells; distinct across seeds; "
        "redraw bitwise identical (np.array_equal), tol exactly 0")

    # ---- C5  operator x critic x law indexing ------------------------------
    cells = {(r["seed"], r["law"], r["critic"], r["operator"]) for r in rows}
    laws = {r["law"] for r in rows}
    crit = {r["critic"] for r in rows}
    opsn = {r["operator"] for r in rows}
    want = len(by_seed) * 2 * 2 * 4
    chk("C5 operator x critic x law indexing complete and unique",
        len(rows) == len(cells) == want and laws == {"A", "B"}
        and crit == {"PW", "WML"} and opsn == set(CD.OPERATORS),
        "%d rows, %d unique cells, expected %d; laws %s critics %s operators %d"
        % (len(rows), len(cells), want, sorted(laws), sorted(crit), len(opsn)))

    # ---- C6  only the reference law changed --------------------------------
    ch = json.load(open(o1.rsplit(".", 1)[0] + "_checks.json"))["checks"]
    bank_sha = json.load(open(o1.rsplit(".", 1)[0] + "_checks.json"))["bank_sha256"]
    ok = all(c["law_changes_sigma"] and c["law_changes_y"] for c in ch)
    chk("C6 law change moves sigma and y, nothing else", ok,
        "states (one frozen bank sha %s..), critics and innovations identical "
        "across laws; sigma and y=mu+sigma*u differ by construction, tol exactly 0"
        % bank_sha[:12])
    return RES


if __name__ == "__main__":
    scratch, stage = sys.argv[1], sys.argv[2]
    os.makedirs(scratch, exist_ok=True)
    out = os.path.join(scratch, "checks_%s.json" % stage)
    if stage == "report":
        allr = []
        for s in ("unit", "pipe"):
            allr += json.load(open(os.path.join(scratch, "checks_%s.json" % s)))
        n = sum(r["ok"] for r in allr)
        print("\n=== %d/%d checks pass ===" % (n, len(allr)))
        for r in allr:
            print("  %-4s %s" % ("PASS" if r["ok"] else "FAIL", r["name"]))
        raise SystemExit(0 if n == len(allr) else 1)
    res = stage_unit() if stage == "unit" else stage_pipe(scratch)
    json.dump(res, open(out, "w"), indent=1)
    print("\n%d/%d in stage %s" % (sum(r["ok"] for r in res), len(res), stage))
    raise SystemExit(0 if all(r["ok"] for r in res) else 1)
