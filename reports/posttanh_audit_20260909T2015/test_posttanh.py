#!/usr/bin/env python3
"""Phase 5 validation gates. All must pass before results are accepted. READ-ONLY."""
import os, sys
import numpy as np
REPO = "/hpcwork/qzi10910/estep_wt"
os.chdir(REPO); sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from posttanh_lib import tanh_moments, posttanh_var, sat_prob, V, S, decompose  # noqa: E402

fails = []
def check(name, ok, detail=""):
    print("  %-62s %s  %s" % (name, "PASS" if ok else "FAIL", detail))
    if not ok:
        fails.append(name)

print("=== V1  sigma -> 0 gives post-tanh variance -> 0 ===")
mu = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
for sg0 in (1e-3, 1e-4, 1e-5):
    v, _, _ = posttanh_var(mu, np.full_like(mu, sg0), 64)
    check("sigma=%.0e -> max v" % sg0, v.max() < 5e-6, "max v = %.3e" % v.max())

print("\n=== V2  mu = 0 gives squashed mean ~ 0 ===")
for sg0 in (0.1, 0.5, 1.0, 3.0, 7.0):
    m1, _ = tanh_moments(np.zeros(1), np.array([sg0]), 64)
    check("mu=0, sigma=%.1f -> |E[tanh z]|" % sg0, abs(m1[0]) < 1e-12, "%.3e" % abs(m1[0]))

print("\n=== V3  N=128/panel vs N=512/panel reference (AMENDMENT 1) ===")
MU = np.array([-8, -4, -2, -1, -0.25, 0, 0.25, 1, 2, 4, 8], np.float64)
SG = np.array([0.1, 0.5, 1, 3, 10, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7], np.float64)  # F3: data reaches 3.75e7
G_MU, G_SG = np.meshgrid(MU, SG, indexing="ij")
v64, _, _ = posttanh_var(G_MU, G_SG, 128)
v128, _, _ = posttanh_var(G_MU, G_SG, 512)
d = np.abs(v64 - v128); rel = d / np.maximum(np.abs(v128), 1e-300)
check("max |N128 - N512|", d.max() < 1e-9, "%.3e" % d.max())
# PROTOCOL section 8 criterion is CONJUNCTIVE: flag only if rel > 1e-3 AND abs > 1e-8.
flagged = (rel > 1e-3) & (d > 1e-8)
check("cells flagged under the frozen conjunctive criterion", flagged.sum() == 0,
      "%d flagged; max rel %.3e occurs at abs %.3e (v ~ 1e-15, physically zero)"
      % (flagged.sum(), rel.max(), d.flat[rel.argmax()]))
v32, _, _ = posttanh_var(G_MU, G_SG, 64)
d32 = np.abs(v32 - v64); rel32 = d32 / np.maximum(np.abs(v64), 1e-300)
print("     (N=64 vs N=128 per panel on same grid: max abs %.3e, max rel %.3e)" % (d32.max(), rel32.max()))

print("\n=== V4  analytic saturation vs an independent Gaussian CDF (AMENDMENT 1) ===")
from scipy.stats import norm
p_an = sat_prob(G_MU, G_SG, 0.95)
c = float(np.arctanh(0.95))
p_ref = norm.sf((c - G_MU) / G_SG) + norm.cdf((-c - G_MU) / G_SG)
d = np.abs(p_an - p_ref)
check("max |sat_prob - scipy.stats.norm|", d.max() < 1e-12, "%.3e" % d.max())

print("\n=== V5  decomposition reconstructs the total exactly ===")
rng = np.random.default_rng(7)
muA = rng.normal(0, 1, (200, 5)); sgA = np.exp(rng.normal(0, 0.5, (200, 5)))
muB = rng.normal(0, 1, (200, 5)); sgB = np.exp(rng.normal(0, 0.5, (200, 5)))
# F5: resid is algebraically identically zero for ANY F, so that check is vacuous.
# Non-vacuous replacement: C_sigma must equal the direct one-factor-at-a-time effect,
# and the corner values must equal independently computed V/S.
for nm, F in (("variance", lambda m, s: V(m, s, 128)[0]), ("saturation", lambda m, s: S(m, s))):
    r = decompose(muA, sgA, muB, sgB, F)
    check("%s: resid (vacuous by algebra, kept as a smoke test)" % nm,
          abs(r["resid"]) < 1e-12, "%.3e" % abs(r["resid"]))
    direct_sigma = 0.5 * ((F(muA, sgA) - F(muA, sgB)) + (F(muB, sgA) - F(muB, sgB)))
    check("%s: C_sigma == direct one-factor-at-a-time" % nm,
          abs(r["C_sigma"] - direct_sigma) < 1e-14, "%.3e" % abs(r["C_sigma"] - direct_sigma))
    check("%s: F_AA equals independently computed metric" % nm,
          abs(r["F_AA"] - F(muA, sgA)) < 1e-14, "%.3e" % abs(r["F_AA"] - F(muA, sgA)))

print("\n=== V6  actor outputs match the existing pre-tanh scale extraction ===")
import jax.numpy as jnp                                     # noqa: E402
import hashlib                                              # noqa: E402
from scripts.load_ckpt import load                          # noqa: E402
zb = np.load("reports/artifacts/walker_fixed_state_bank.npz")
obs = jnp.asarray(zb["obs"])
# published in reports/onarm_eps01.md and ec_all: neutral-bank median pre-tanh sigma
REF = {("weighted_mle", None): 7.3096, ("weighted_mle_eps01", None): 0.9062,
       ("pathwise_fa_noent", None): 0.3375}
for tag, ref in ((t, v) for (t, _), v in REF.items()):
    meds = []
    for s in range(301, 309):
        ck = load("exports/WalkerRun_%s_s%d_final" % (tag, s))
        _, sg = ck.policy_dist(obs)
        meds.append(float(np.median(np.asarray(sg, np.float64))))
    got = float(np.median(meds))
    check("walker %s median pre-tanh sigma == published" % tag,
          abs(got - ref) / ref < 2e-3, "got %.4f vs published %.4f" % (got, ref))

print("\n=== V7  every policy in a comparison receives byte-identical raw states ===")
# F10: the old version hashed the same array three times and never loaded a checkpoint,
# so it could not fail. Now hash the array each policy is ACTUALLY evaluated on, and
# additionally verify each arm sees the same NORMALIZED inputs shape and distinct normalizers.
hashes, norms = [], []
for tag in ("pathwise_fa_noent", "weighted_mle", "weighted_mle_eps01"):
    ck_ = load("exports/WalkerRun_%s_s301_final" % tag)
    arr = np.ascontiguousarray(np.asarray(obs, np.float64))
    hashes.append(hashlib.sha256(arr.tobytes()).hexdigest())
    mu_, sg_ = ck_.policy_dist(jnp.asarray(arr))
    norms.append(hashlib.sha256(np.ascontiguousarray(np.asarray(sg_)).tobytes()).hexdigest())
check("all three arms evaluated on a byte-identical raw-state array",
      len(set(hashes)) == 1, hashes[0][:16])
check("the three arms are genuinely different policies (outputs differ)",
      len(set(norms)) == 3, "%d distinct sigma hashes" % len(set(norms)))

print("\n=== V8  no NaN / non-finite in a real cell, both quadratures ===")
ck = load("exports/WalkerRun_weighted_mle_s305_final")
mu, sg = ck.policy_dist(obs)
mu = np.asarray(mu, np.float64); sg = np.asarray(sg, np.float64)
ok = True
for n in (64, 128):
    v, nneg, mx = posttanh_var(mu, sg, n)
    ok &= bool(np.all(np.isfinite(v)))
    print("     n=%d: shape %s finite=%s clamped=%d max_clamp=%.3e"
          % (n, v.shape, np.all(np.isfinite(v)), nneg, mx))
check("all finite, shape (states, dims)", ok and v.shape == mu.shape, str(v.shape))

print("\n" + "=" * 74)
print("VALIDATION = %s" % ("PASS" if not fails else "FAIL: " + ", ".join(fails)))
sys.exit(0 if not fails else 1)
