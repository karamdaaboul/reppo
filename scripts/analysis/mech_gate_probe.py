"""Counterfactual characterisation of the KL-gate estimator at N=16 vs N=32.

NOT a recovery of the historical gate-flip rate. No adjacent-iteration checkpoint
pair exists (snapshots are at iteration ~95/~190/399, meta.json checkpoint_frac
0.25/0.5/1.0), so pi_old/pi_new for a single outer iteration cannot be
reconstructed. This uses REAL (mu, sigma) from a corrected checkpoint on REAL bank
states, with pi_new calibrated to a target EXACT KL spanning the 0.1 bound, built
from BOTH mean and covariance change.

Read-only on checkpoints. Emits reports/artifacts/mech_gate_probe.json.
"""
from __future__ import annotations
import json, os, sys
import numpy as np
from scipy.optimize import brentq

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO); sys.path.insert(0, REPO)
from scripts.load_ckpt import load                                    # noqa: E402

BOUND, NREP, NSTATE = 0.1, 8000, 16
CKPT = "exports/WalkerRun_weighted_mle_s301_final"
BANK = "reports/artifacts/cd_bank_walker_corrected.npz"
KGRID = (0.02, 0.05, 0.08, 0.09, 0.095, 0.099, 0.101, 0.105, 0.11, 0.12, 0.15, 0.30)

def exact_kl(m0, s0, m1, s1):
    return float(np.sum(np.log(s1/s0) + (s0**2 + (m0-m1)**2)/(2*s1**2) - 0.5))

def make_new(m0, s0, K, mean_frac, rg):
    d = m0.shape[-1]; Km, Kc = K*mean_frac, K*(1-mean_frac)
    c = 1.0 if Kc <= 0 else brentq(lambda x: 0.5*d*(x**-2-1+2*np.log(x))-Kc, 1+1e-12, 20.0)
    s1 = s0*c
    u = rg.normal(size=d); u /= np.linalg.norm(u)
    sc = np.sqrt(2*Km/np.sum((u/s1)**2)) if Km > 0 else 0.0
    return m0 + sc*u, s1

def estimates(m0, s0, m1, s1, N, rg):
    """Exactly the implemented estimator: mean over N draws from pi_old of
    (log pi_old - log pi_new). The tanh Jacobian cancels, so this is the
    pre-squash Gaussian log-ratio (verified in reports/implementation_audit.md)."""
    u = rg.normal(size=(NREP, N, m0.shape[-1])); y = m0 + s0*u
    lo = np.sum(-0.5*((y-m0)/s0)**2 - np.log(s0), axis=-1)
    lt = np.sum(-0.5*((y-m1)/s1)**2 - np.log(s1), axis=-1)
    return (lo-lt).mean(axis=1)

def main():
    rg = np.random.default_rng(20260904)
    z = np.load(BANK, allow_pickle=True)
    obs = np.asarray(z["obs"], np.float32)[np.asarray(z["eval_idx"], np.int64)][:NSTATE]
    c = load(CKPT)
    mu, sg = c.policy_dist(obs)
    mu, sg = np.asarray(mu, np.float64), np.asarray(sg, np.float64)
    rows = []
    print("checkpoint %s   d=%d   sigma median %.3f min %.3f max %.3f"
          % (CKPT, mu.shape[-1], np.median(sg), sg.min(), sg.max()))
    print("  %-9s | %9s %9s %7s | %9s %9s | %9s %9s"
          % ("exact KL","flip N=16","flip N=32","ratio","fOpen16","fOpen32","fClos16","fClos32"))
    for K in KGRID:
        a = {16: [0,0,0], 32: [0,0,0]}; tot = 0; kes = []
        for i in range(mu.shape[0]):
            m1, s1 = make_new(mu[i], sg[i], K, 0.5, rg)
            ke = exact_kl(mu[i], sg[i], m1, s1); kes.append(ke); ge = ke >= BOUND
            for N in (16, 32):
                gs = estimates(mu[i], sg[i], m1, s1, N, rg) >= BOUND
                a[N][0] += int((gs != ge).sum())
                a[N][1] += int((np.logical_not(gs) & ge).sum())
                a[N][2] += int((gs & np.logical_not(ge)).sum())
            tot += NREP
        r = dict(exact_kl=float(np.mean(kes)), side="above" if np.mean(kes) >= BOUND else "below",
                 flip16=a[16][0]/tot, flip32=a[32][0]/tot,
                 ratio=a[16][0]/max(a[32][0], 1),
                 false_open16=a[16][1]/tot, false_open32=a[32][1]/tot,
                 false_closed16=a[16][2]/tot, false_closed32=a[32][2]/tot)
        rows.append(r)
        print("  %-9.3f | %9.4f %9.4f %7.3f | %9.4f %9.4f | %9.4f %9.4f"
              % (r["exact_kl"], r["flip16"], r["flip32"], r["ratio"],
                 r["false_open16"], r["false_open32"], r["false_closed16"], r["false_closed32"]))
    json.dump(dict(checkpoint=CKPT, bank=BANK, n_states=NSTATE, n_reps=NREP,
                   bound=BOUND, mean_frac=0.5, rows=rows,
                   disclaimer="counterfactual estimator characterisation; NOT the "
                              "historical gate-flip rate, which is not identifiable "
                              "from the reduced (21,1) logs"),
              open("reports/artifacts/mech_gate_probe.json", "w"), indent=1)
    print("\nwrote reports/artifacts/mech_gate_probe.json")

if __name__ == "__main__":
    main()
