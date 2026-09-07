#!/usr/bin/env python3
"""Generate on-policy state banks for the two arms that had no own-rollout population.

Spec frozen in docs/protocol_onpolicy_banks.md. Mirrors
scripts/analysis/fixed_bank_width_saturation.py::build exactly, with one deliberate
deviation: the per-block RNG offset is derived by sha256 rather than by Python's salted
hash(), so these banks ARE byte-regenerable. See section 3 of the protocol.

Rollouts collect states only. No training, no checkpoint is written or modified.
"""
from __future__ import annotations
import hashlib, json, os, sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO); sys.path.insert(0, REPO)
import jax, jax.numpy as jnp                                          # noqa: E402
from scripts.critic_fidelity.common import Harness, ACTION_CLIP       # noqa: E402

SEEDS = list(range(301, 309))
NENV = 32
ROOT_ONPOLICY = 20260907          # distinct from the neutral banks' ROOT = 20260905
ARMS = {"PW_noent": "pathwise_fa_noent", "WML_ent": "weighted_mle_ent"}

TASKS = {
 "walker": dict(env="WalkerRun",              depths=(50, 150, 300, 500, 700, 900)),
 "g1":     dict(env="G1JoystickFlatTerrain",  depths=(50, 150, 300, 500, 700, 900)),
 "leap":   dict(env="LeapCubeRotateZAxis",    depths=(50, 100, 200, 300, 400, 480)),
}
OUT = "reports/artifacts/%s_onpolicy_bank_newarms.npz"


def block_key(task: str, arm: str, seed: int):
    """Deterministic, PYTHONHASHSEED-independent. Protocol section 3."""
    off = int.from_bytes(
        hashlib.sha256(("%s|%s|%d" % (task, arm, seed)).encode()).digest()[:4], "big"
    ) % (2**31)
    return jax.random.fold_in(jax.random.PRNGKey(ROOT_ONPOLICY), off), off


def rollout(ckpt: str, depths, key):
    """32 envs rolled to max(depths); observations captured at each listed depth."""
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
    print("ROOT_ONPOLICY=%d  NENV=%d  SEEDS=%s" % (ROOT_ONPOLICY, NENV, SEEDS))
    manifest = {}
    for task, cfg in TASKS.items():
        out = OUT % task
        if os.path.exists(out):
            print("\n########## %s -- EXISTS, refusing to overwrite: %s" % (task, out))
            sys.exit("refusing to overwrite an existing bank")
        print("\n########## %s  (%s)  depths=%s" % (task, cfg["env"], cfg["depths"]))
        obs_all, src_all, dep_all = [], [], []
        for arm, tag in ARMS.items():
            for sd in SEEDS:
                ckpt = "exports/%s_%s_s%d_final" % (cfg["env"], tag, sd)
                if not os.path.isdir(ckpt):
                    sys.exit("missing checkpoint: %s" % ckpt)
                key, off = block_key(task, arm, sd)
                got = rollout(ckpt, cfg["depths"], key)
                for t, o in zip(cfg["depths"], got):
                    obs_all.append(o)
                    src_all += ["%s-s%d" % (arm, sd)] * NENV
                    dep_all += [t] * NENV
                print("    %-7s %-9s s%d  off=%-10d states=%d"
                      % (task, arm, sd, off, NENV * len(cfg["depths"])), flush=True)
        obs = np.concatenate(obs_all, 0).astype(np.float32)
        src = np.array(src_all)
        dep = np.array(dep_all)
        np.savez(out, obs=obs, source=src, depth=dep,
                 depths=np.array(cfg["depths"]), n_env=np.array(NENV),
                 root=np.array(ROOT_ONPOLICY),
                 keying=np.array("sha256-deterministic"))
        sha = hashlib.sha256(open(out, "rb").read()).hexdigest()
        counts = {a: int((np.char.startswith(src, a + "-")).sum()) for a in ARMS}
        print("  BANK_PATH=%s" % out)
        print("  BANK_SHA256=%s" % sha)
        print("  NUM_STATES=%d  %s" % (obs.shape[0], counts))
        print("  DEPTH_DISTRIBUTION=" +
              " ".join("%d:%d" % (d, int((dep == d).sum())) for d in sorted(set(dep.tolist()))))
        assert obs.shape[0] == len(ARMS) * len(SEEDS) * NENV * len(cfg["depths"])
        assert all(v == len(SEEDS) * NENV * len(cfg["depths"]) for v in counts.values())
        manifest[task] = dict(path=out, sha256=sha, n_states=int(obs.shape[0]),
                              per_arm=counts, depths=list(cfg["depths"]),
                              n_env=NENV, root=ROOT_ONPOLICY,
                              keying="sha256-deterministic")
    with open("reports/artifacts/onpolicy_banks_manifest.json", "w") as f:
        json.dump(dict(script_sha256=hashlib.sha256(open(me, "rb").read()).hexdigest(),
                       banks=manifest), f, indent=1)
    print("\nwrote reports/artifacts/onpolicy_banks_manifest.json")


if __name__ == "__main__":
    main()
