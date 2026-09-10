import numpy as np
print("=== D1: is hermgauss(n) itself breaking at large n? ===")
for n in (64, 128, 200, 400):
    x, w = np.polynomial.hermite.hermgauss(n)
    print("  n=%-4d finite weights: %-5s  sum(w)=%s  (exact sqrt(pi)=%.6f)"
          % (n, bool(np.all(np.isfinite(w))), np.sum(w), np.sqrt(np.pi)))

print("\n=== D2: GH convergence for E[tanh^2] at large sigma (mu=-4) ===")
mu, sg = -4.0, 10.0
for n in (32, 64, 96, 128):
    x, w = np.polynomial.hermite.hermgauss(n)
    z = mu + np.sqrt(2.0) * sg * x
    m1 = np.sum(w * np.tanh(z)) / np.sqrt(np.pi)
    m2 = np.sum(w * np.tanh(z) ** 2) / np.sqrt(np.pi)
    print("  GH n=%-4d  m1=%+.9f  m2=%.9f  v=%.9f" % (n, m1, m2, m2 - m1 * m1))

print("\n=== D3: 2-panel Gauss-Legendre split at the tanh transition ===")
def gl_moments(mu, sg, N=128, T=15.0):
    """E[f(mu+sg t)] over standard normal t, split at t=c where z=0."""
    mu = np.atleast_1d(np.asarray(mu, np.float64)); sg = np.atleast_1d(np.asarray(sg, np.float64))
    xg, wg = np.polynomial.legendre.leggauss(N)
    c = np.clip(-mu / sg, -T, T)
    phi = lambda t: np.exp(-0.5 * t * t) / np.sqrt(2 * np.pi)
    m1 = np.zeros_like(mu); m2 = np.zeros_like(mu)
    for a, b in ((np.full_like(c, -T), c), (c, np.full_like(c, T))):
        half = (b - a) / 2.0; mid = (a + b) / 2.0
        t = mid[..., None] + half[..., None] * xg
        z = mu[..., None] + sg[..., None] * t
        tt = np.tanh(z); ww = wg * half[..., None] * phi(t)
        m1 += np.sum(ww * tt, axis=-1); m2 += np.sum(ww * tt * tt, axis=-1)
    return m1, m2

for (mu, sg) in ((-4.0, 10.0), (0.0, 7.0), (2.0, 0.5), (-8.0, 0.1), (0.25, 3.0)):
    row = []
    for N in (64, 128, 256, 512):
        m1, m2 = gl_moments(mu, sg, N)
        row.append((m2 - m1 * m1)[0])
    print("  mu=%-6.2f sg=%-5.2f  v(N=64)=%.12f  N=128=%.12f  N=256=%.12f  N=512=%.12f"
          % (mu, sg, *row))
    print("      |N128-N256|=%.3e   |N256-N512|=%.3e" % (abs(row[1]-row[2]), abs(row[2]-row[3])))

print("\n=== D4: analytic saturation vs a VALID numerical reference (GL, not hermgauss-400) ===")
from scipy.special import erf
MU = np.array([-8,-4,-2,-1,-0.25,0,0.25,1,2,4,8], np.float64)
SG = np.array([0.1,0.25,0.5,1,2,3,5,7,10], np.float64)
G_MU, G_SG = np.meshgrid(MU, SG, indexing="ij")
c = float(np.arctanh(0.95))
Phi = lambda t: 0.5*(1.0+erf(t/np.sqrt(2.0)))
p_an = Phi((-c-G_MU)/G_SG) + 1.0 - Phi((c-G_MU)/G_SG)
# exact by construction: P(|z|>c) has a closed form; cross-check with GL on the indicator
xg, wg = np.polynomial.legendre.leggauss(2000)
T = 40.0
t = (xg*T)
ww = wg*T*np.exp(-0.5*t*t)/np.sqrt(2*np.pi)
z = G_MU[...,None] + G_SG[...,None]*t
p_num = np.sum(ww*(np.abs(np.tanh(z))>0.95), axis=-1)
d = np.abs(p_an-p_num)
print("  max |analytic - GL2000| = %.3e   (finite: %s)" % (d.max(), np.all(np.isfinite(d))))
