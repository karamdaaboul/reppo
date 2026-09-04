"""Decompose the training-vs-probe ESS gap on ONE corrected Walker WML checkpoint.

Isolates: candidate count M, state distribution, generation semantics, and
sampling variability. Uses the exact training candidate generator and the
checkpoint's own eta. Read-only. Emits reports/artifacts/mech_ess_decomp.json.
"""
from __future__ import annotations
import json, os, sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO); sys.path.insert(0, REPO)
import jax, jax.numpy as jnp                                          # noqa: E402
from scripts.load_ckpt import load                                    # noqa: E402
from scripts.critic_fidelity.common import Harness, ACTION_CLIP       # noqa: E402

CKPT = "exports/WalkerRun_weighted_mle_s301_final"
ETA  = 0.024808276444673538      # eta_saved, reports/artifacts/cd_walker_corrected_diagnostics.csv
PILOT_BANK = "reports/artifacts/mc_oracle_state_bank.npz"
REP, NENV, DEPTHS = 20, 64, (50, 150, 300, 500, 700, 900, 1000)

def ess_over(c, obs, M, reps, key0=0):
    """Exact training generator: y = mu + sigma*u, a = tanh(y), w = softmax(Q/eta)."""
    mu, sg = c.policy_dist(jnp.asarray(obs))
    mu, sg = jnp.asarray(mu), jnp.asarray(sg)
    S, d = mu.shape
    out = []
    for r in range(reps):
        u = jax.random.normal(jax.random.PRNGKey(key0 + r), (M, S, d))
        a = jnp.tanh(mu[None] + sg[None] * u)
        ob = jnp.broadcast_to(jnp.asarray(obs)[None], (M, S, obs.shape[-1]))
        q = c.q_scalar(ob.reshape(-1, obs.shape[-1]), a.reshape(-1, d)).reshape(M, S)
        w = jax.nn.softmax(q / ETA, axis=0)
        out.append(np.asarray(1.0 / jnp.sum(w**2, axis=0)))
    return np.concatenate(out)

def stats(e, M):
    x = e / M
    return dict(mean=float(x.mean()), median=float(np.median(x)),
                p5=float(np.percentile(x, 5)), p95=float(np.percentile(x, 95)))

def main():
    c = load(CKPT)
    z = np.load(PILOT_BANK, allow_pickle=True)
    k = [n for n in z.files if z[n].ndim == 2 and z[n].shape[-1] == 24][0]
    obsP = np.asarray(z[k], np.float32)
    h = Harness(CKPT, NENV); key = jax.random.PRNGKey(7)
    k1, key = jax.random.split(key)
    o, _, st = h.reset(k1); snaps = {}
    for t in range(1, max(DEPTHS) + 1):
        ka, kb, key = jax.random.split(key, 3)
        a = jnp.clip(h.pi(o).sample(seed=ka), -ACTION_CLIP, ACTION_CLIP)
        o, _, st, _, _, _ = h.env.step(jax.random.split(kb, NENV), st, a)
        if t in DEPTHS:
            snaps[t] = np.asarray(o)
    obsT = np.concatenate([snaps[t] for t in DEPTHS], 0).astype(np.float32)

    out = dict(checkpoint=CKPT, eta=ETA, reps=REP,
               bank_pilot=dict(path=PILOT_BANK, key=k, shape=list(obsP.shape)),
               bank_episode=dict(depths=list(DEPTHS), n_env=NENV, shape=list(obsT.shape)))
    print("bank P (pilot, burn-in 50): %s   bank T (episode-spanning): %s"
          % (obsP.shape, obsT.shape))
    out["A_candidate_count_bankP"] = {}
    print("\nA. candidate count, SAME states (bank P)")
    for M in (16, 32):
        s = stats(ess_over(c, obsP, M, REP), M); out["A_candidate_count_bankP"]["M%d" % M] = s
        print("   M=%-3d ESS/M mean %.4f median %.4f p5 %.4f p95 %.4f"
              % (M, s["mean"], s["median"], s["p5"], s["p95"]))
    out["B_state_distribution_M32"] = {}
    print("\nB. state distribution, SAME M=32, SAME eta, SAME generator")
    for name, ob in (("pilot_burnin50", obsP), ("episode_spanning", obsT)):
        s = stats(ess_over(c, ob, 32, REP), 32); out["B_state_distribution_M32"][name] = s
        print("   %-18s ESS/M mean %.4f median %.4f p5 %.4f p95 %.4f"
              % (name, s["mean"], s["median"], s["p5"], s["p95"]))
    out["B2_per_depth_M32"] = {}
    print("\nB2. per rollout depth within bank T (M=32)")
    for i, t in enumerate(DEPTHS):
        s = stats(ess_over(c, obsT[i*NENV:(i+1)*NENV], 32, REP), 32)
        out["B2_per_depth_M32"]["depth_%d" % t] = s
        print("   depth %-5d ESS/M mean %.4f median %.4f" % (t, s["mean"], s["median"]))
    per = [float((ess_over(c, obsT, 32, 1, key0=100+r)/32).mean()) for r in range(REP)]
    out["D_sampling_variability"] = dict(per_draw_mean_min=min(per), per_draw_mean_max=max(per),
                                         sd=float(np.std(per)), n_draws=REP)
    print("\nD. sampling variability (bank T, M=32): min %.4f max %.4f sd %.5f"
          % (min(per), max(per), np.std(per)))
    out["reference"] = dict(
        ESS_training_M32_over_M=0.6403,
        ESS_training_source="reports/implementation_audit.md addendum A1.5 (commit e9b9f00)",
        ESS_Qphi_pilot_K16_over_M=0.099,
        ESS_pilot_source="reports/implementation_audit.md addendum A1.5 (commit e9b9f00)")
    json.dump(out, open("reports/artifacts/mech_ess_decomp.json", "w"), indent=1)
    print("\nwrote reports/artifacts/mech_ess_decomp.json")

if __name__ == "__main__":
    main()
