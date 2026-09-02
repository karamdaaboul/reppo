"""Zero-norm pathwise updates under the B law: is it tanh saturation?

Full R = 200, seed 301, law B only. Tests PW-1 and PW-32, and whether the
zero-norm mask coincides with all-coordinate saturation of the SAME cloud.
"""
import sys
import numpy as np
import jax, jax.numpy as jnp
sys.path.insert(0, "/rwthfs/rz/cluster/home/qzi10910/repos/reppo")
import scripts.analysis.crossed_dispersion as CD
from scripts.load_ckpt import load

z = np.load("reports/artifacts/cd_bank_walker_corrected.npz", allow_pickle=True)
raw = jnp.asarray(np.asarray(z["obs"], np.float32)[np.asarray(z["eval_idx"], np.int64)])
S, R, M = raw.shape[0], CD.PREREG["R"], CD.PREREG["M"]
cks = {a: load(CD.ckpt_dir("walker", a, 301)) for a in ("PW", "WML")}
d = int(cks["PW"].meta["action_dim"])
u = jax.random.normal(CD.fold("u", "walker", 301), (R, S, M, d))

mu, sigma = cks["WML"].policy_dist(raw)                       # B law
mu, sigma = jnp.asarray(mu), jnp.asarray(sigma)
y = mu[None, :, None, :] + sigma[None, :, None, :] * u
sat = jnp.abs(jnp.tanh(y)) > 0.999                            # (R,S,M,d)
print("seed 301, law B, R=%d S=%d M=%d" % (R, S, M))
print("per-coordinate saturation %.4f | all-d saturated per sample %.4f"
      % (float(jnp.mean(sat)), float(jnp.mean(sat.all(-1)))))

masks = {}
for csrc in ("PW", "WML"):
    F, G = CD._critic_batch(cks[csrc], raw, y, 25)
    for name, g in (("PW-1", G[:, :, 0, :]), ("PW-32", G.mean(2))):
        wn = jnp.sqrt(jnp.sum((g / sigma[None]) ** 2, -1))
        zm = wn <= 0
        masks[(csrc, name)] = np.asarray(zm)
        allsat = sat.all(-1)[:, :, 0] if name == "PW-1" else sat.all(-1).all(-1)
        both = float(jnp.mean(zm & allsat))
        zf = float(jnp.mean(zm))
        print("  critic %-4s %-6s zero frac %.5f (%d) | all-d saturated frac %.5f "
              "| zero AND saturated %.5f | P(saturated | zero) %.4f"
              % (csrc, name, zf, int(jnp.sum(zm)), float(jnp.mean(allsat)), both,
                 both / max(zf, 1e-12)))
for name in ("PW-1", "PW-32"):
    print("  %-6s zero mask identical across critic sources: %s"
          % (name, np.array_equal(masks[("PW", name)], masks[("WML", name)])))
