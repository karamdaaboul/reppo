#!/usr/bin/env python3
"""On-policy state banks for the eps_e = 0.1 arm (WML_eps01), all three tasks.

Completes the SECONDARY geometry endpoint registered in
docs/prereg_estep_concentration.md Sec. 4.2, to the spec frozen in
docs/protocol_onpolicy_banks.md: 192 states per (arm, seed), same depth
stratification, 32 parallel envs, stochastic actions clipped to +-0.999, frozen
normaliser, and fresh independent keys derived by sha256 (not Python's salted
hash), from a root distinct from every previous bank.

Rollouts collect states only. No training, no checkpoint written or modified.
"""
from __future__ import annotations
import hashlib, json, os, sys
import numpy as np

REPO = "/hpcwork/qzi10910/estep_wt"
os.chdir(REPO); sys.path.insert(0, REPO)
import jax, jax.numpy as jnp                                          # noqa: E402
from scripts.critic_fidelity.common import Harness, ACTION_CLIP       # noqa: E402

SEEDS = list(range(301, 309))
NENV = 32
ROOT = 20260909          # distinct from 20260905 (neutral) and 20260907 (new arms)
ARM = "WML_eps01"
TAG = "weighted_mle_eps01"
TASKS = {
 "walker": dict(env="WalkerRun",             depths=(50, 150, 300, 500, 700, 900)),
 "g1":     dict(env="G1JoystickFlatTerrain", depths=(50, 150, 300, 500, 700, 900)),
 "leap":   dict(env="LeapCubeRotateZAxis",   depths=(50, 100, 200, 300, 400, 480)),
}
OUT = "reports/artifacts/%s_onpolicy_bank_eps01.npz"


def block_key(task, arm, seed):
    off = int.from_bytes(hashlib.sha256(("%s|%s|%d" % (task, arm, seed)).encode()).digest()[:4],
                         "big") % (2 ** 31)
    return jax.random.fold_in(jax.random.PRNGKey(ROOT), off), off


def rollout(ckpt, depths, key):
    h = Harness(ckpt, NENV)
    k1, key = jax.random.split(key)
    o, _, st = h.reset(k1)
    got = []
    for t in range(1, max(depths) + 1):
        ka, kb, key = jax.random.split(key, 3)
        a = jnp.clip(h.pi(o).sample(seed=ka), -ACTION_CLIP, ACTION_CLIP)
        o, _, st, _, _, _ = h.env.step(jax.random.split(kb, NENV), st, a)
        if t in depths:
            got.append(np.asarray(o))
    return got


def main():
    me = os.path.abspath(__file__)
    print("script sha256: %s" % hashlib.sha256(open(me, "rb").read()).hexdigest())
    print("ROOT=%d  NENV=%d  arm=%s  SEEDS=%s" % (ROOT, NENV, ARM, SEEDS))
    man = {}
    for task, cfg in TASKS.items():
        out = OUT % task
        if os.path.exists(out):
            sys.exit("refusing to overwrite existing bank: %s" % out)
        print("\n########## %s (%s) depths=%s" % (task, cfg["env"], cfg["depths"]))
        ob, sr, dp = [], [], []
        for sd in SEEDS:
            ck = "exports/%s_%s_s%d_final" % (cfg["env"], TAG, sd)
            if not os.path.isdir(ck):
                sys.exit("missing checkpoint: %s" % ck)
            key, off = block_key(task, ARM, sd)
            for t, o in zip(cfg["depths"], rollout(ck, cfg["depths"], key)):
                ob.append(o); sr += ["%s-s%d" % (ARM, sd)] * NENV; dp += [t] * NENV
            print("    %-7s %-9s s%d off=%-10d states=%d"
                  % (task, ARM, sd, off, NENV * len(cfg["depths"])), flush=True)
        obs = np.concatenate(ob, 0).astype(np.float32)
        src, dep = np.array(sr), np.array(dp)
        np.savez(out, obs=obs, source=src, depth=dep, depths=np.array(cfg["depths"]),
                 n_env=np.array(NENV), root=np.array(ROOT),
                 keying=np.array("sha256-deterministic"))
        sha = hashlib.sha256(open(out, "rb").read()).hexdigest()
        n_exp = len(SEEDS) * NENV * len(cfg["depths"])
        assert obs.shape[0] == n_exp, (obs.shape, n_exp)
        assert bool(np.isfinite(obs).all())
        print("  BANK_PATH=%s\n  BANK_SHA256=%s\n  NUM_STATES=%d  finite=True"
              % (out, sha, obs.shape[0]))
        man[task] = dict(path=out, sha256=sha, n_states=int(obs.shape[0]), arm=ARM,
                         depths=list(cfg["depths"]), n_env=NENV, root=ROOT,
                         keying="sha256-deterministic")
    with open("reports/artifacts/onpolicy_banks_eps01_manifest.json", "w") as f:
        json.dump(dict(script_sha256=hashlib.sha256(open(me, "rb").read()).hexdigest(),
                       banks=man), f, indent=1)
    print("\nwrote reports/artifacts/onpolicy_banks_eps01_manifest.json")


if __name__ == "__main__":
    main()
