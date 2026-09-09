#!/usr/bin/env python3
"""Tests for the diagnostic forward-KL mean/covariance split.

Run: ./.venv/bin/python tests/test_kl_split.py
"""
import os, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import jax
jax.config.update("jax_enable_x64", True)   # the parts and their sum are close in float32
import jax.numpy as jnp
import numpy as np

# src/jaxrl/reppo.py imports the full training stack at module level (optuna,
# plotly, wandb, ...). The two functions under test are pure jnp and touch none of
# it, so any of those that is absent from THIS interpreter is stubbed rather than
# installed: the test then exercises the REAL repository function instead of a
# reimplementation of it, and mutates no environment. Every stub is reported, and
# nothing in src/ is ever stubbed.
import types
_stubbed = []
for _ in range(40):
    try:
        from src.jaxrl.reppo import gaussian_kl_diag, gaussian_kl_diag_split
        break
    except ModuleNotFoundError as e:
        name = e.name
        if not name or name.split(".")[0] == "src":
            raise
        m = types.ModuleType(name)
        m.__path__ = []                                  # a package, so submodules resolve
        m.__getattr__ = lambda attr: type("Stub", (), {})
        sys.modules[name] = m
        if "." in name:
            parent, child = name.rsplit(".", 1)
            setattr(sys.modules[parent], child, m)
        _stubbed.append(name)
else:
    raise RuntimeError("could not import src.jaxrl.reppo after 40 stubs")
if _stubbed:
    print("  (stubbed absent modules, none used by this test: %s)" % ", ".join(_stubbed))

RNG = np.random.default_rng(20260907)
TOL = 1e-10
fails = []


def check(name, cond, detail=""):
    print("  %-58s %s  %s" % (name, "PASS" if cond else "FAIL", detail))
    if not cond:
        fails.append(name)


def rand(shape, lo=0.05, hi=6.0):
    mu = RNG.normal(0.0, 2.0, shape)
    sg = RNG.uniform(lo, hi, shape)
    return jnp.asarray(mu), jnp.asarray(sg)


print("\nT1  mean_part + width_part == gaussian_kl_diag, random diagonal Gaussians")
worst = 0.0
for trial, shape in enumerate([(64, 6), (128, 21), (32, 16), (256, 3), (8, 103)]):
    mu0, sg0 = rand(shape)
    mu1, sg1 = rand(shape)
    kl = gaussian_kl_diag(mu0, sg0, mu1, sg1)
    m, w = gaussian_kl_diag_split(mu0, sg0, mu1, sg1)
    rel = float(jnp.max(jnp.abs(m + w - kl) / jnp.maximum(jnp.abs(kl), 1e-12)))
    worst = max(worst, rel)
    print("    shape %-10s  max rel resid %.3e" % (str(shape), rel))
check("T1 sum equals the analytic forward KL", worst < TOL, "worst rel %.3e" % worst)

print("\nT2  both parts vanish when pi_theta == pi_old")
mu0, sg0 = rand((512, 21))
m, w = gaussian_kl_diag_split(mu0, sg0, mu0, sg0)
kl = gaussian_kl_diag(mu0, sg0, mu0, sg0)
check("T2a mean_part  == 0", float(jnp.max(jnp.abs(m))) < TOL, "max %.3e" % float(jnp.max(jnp.abs(m))))
check("T2b width_part == 0", float(jnp.max(jnp.abs(w))) < TOL, "max %.3e" % float(jnp.max(jnp.abs(w))))
check("T2c analytic KL == 0", float(jnp.max(jnp.abs(kl))) < TOL, "max %.3e" % float(jnp.max(jnp.abs(kl))))

print("\nT3  each part isolates its own difference")
mu0, sg0 = rand((256, 6))
mu1 = mu0 + RNG.normal(0.0, 1.0, mu0.shape)      # means differ, widths identical
m, w = gaussian_kl_diag_split(mu0, sg0, jnp.asarray(mu1), sg0)
check("T3a width_part == 0 when sigmas match", float(jnp.max(jnp.abs(w))) < TOL,
      "max %.3e" % float(jnp.max(jnp.abs(w))))
_, sg1 = rand((256, 6))
m2, w2 = gaussian_kl_diag_split(mu0, sg0, mu0, sg1)   # widths differ, means identical
check("T3b mean_part  == 0 when means match", float(jnp.max(jnp.abs(m2))) < TOL,
      "max %.3e" % float(jnp.max(jnp.abs(m2))))

print("\nT4  ORIENTATION: the mean term is scaled by sigma_theta, not sigma_old")
mu0, sg0 = rand((256, 6), lo=0.2, hi=0.5)
mu1, sg1 = rand((256, 6), lo=2.0, hi=5.0)            # deliberately very different scales
m, _ = gaussian_kl_diag_split(mu0, sg0, mu1, sg1)
fwd = jnp.sum((mu0 - mu1) ** 2 / (2.0 * sg1**2), axis=-1)   # forward: sigma_theta
rev = jnp.sum((mu0 - mu1) ** 2 / (2.0 * sg0**2), axis=-1)   # reverse: sigma_old
check("T4a mean_part matches the FORWARD form", float(jnp.max(jnp.abs(m - fwd))) < TOL,
      "max %.3e" % float(jnp.max(jnp.abs(m - fwd))))
check("T4b mean_part differs from the REVERSE form",
      float(jnp.min(jnp.abs(m - rev))) > 1e-3,
      "min gap %.3e" % float(jnp.min(jnp.abs(m - rev))))

print("\nT5  both parts are non-negative")
neg_m = neg_w = 0
for _ in range(20):
    a0, b0 = rand((128, 12))
    a1, b1 = rand((128, 12))
    m, w = gaussian_kl_diag_split(a0, b0, a1, b1)
    neg_m += int(jnp.sum(m < -TOL)); neg_w += int(jnp.sum(w < -TOL))
check("T5a mean_part  >= 0 everywhere", neg_m == 0, "%d negatives" % neg_m)
check("T5b width_part >= 0 everywhere", neg_w == 0, "%d negatives" % neg_w)

print("\n" + "=" * 72)
print("KL_SPLIT_TESTS = %s" % ("PASS" if not fails else "FAIL " + ", ".join(fails)))
sys.exit(0 if not fails else 1)
