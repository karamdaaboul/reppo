"""Budget and configuration assertions for the confirmatory M-sweep.

docs/prereg_m_sweep_confirmatory.md Sec. 4.2 requires that every batch, epoch and
iteration quantity be byte-identical across M, and Sec. 11 requires the smoke test
to print the RESOLVED values per arm and assert them against the WML-32 run's
values. This script is that assertion. It resolves the real Hydra config for each
arm -- it does not restate the numbers from the prereg -- so a config drift is
caught rather than papered over.

Usage: m_sweep_assert.py [--json OUT]
"""
from __future__ import annotations

import json, os, sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO)
from hydra import compose, initialize_config_dir  # noqa: E402

PREREG = "docs/prereg_m_sweep_confirmatory.md"
ALPHA = "0.014509912580251694"          # Sec. 2: one scalar, all eight controls
SEEDS = tuple(range(301, 309))
# Sec. 2 seed-to-partition table. M=128 pairs to each seed's control; M=512 is on
# c23g for every seed, with the odd-seed mismatch disclosed in Sec. 7 (iii).
CONTROL_PARTITION = {s: ("c25g" if s % 2 == 1 else "c23g") for s in SEEDS}
def partition_for(M, seed):
    return "c23g" if M == 512 else CONTROL_PARTITION[seed]

BASE = ["env=mjx_dmc", "env.name=WalkerRun",
        "experiment_overrides=mjx_dmc_large_data",
        "hyperparameters.actor_update_mode=weighted_mle",
        "hyperparameters.update_entropy_lagrangian=false",
        "hyperparameters.ent_start=" + ALPHA,
        "hyperparameters.faithful_same_point=true",
        "hyperparameters.fresh_minibatch_key=true",
        "hyperparameters.log_faithful_diag=true"]


def resolve(M, seed):
    with initialize_config_dir(config_dir=os.path.join(REPO, "config"),
                               version_base=None):
        c = compose(config_name="reppo",
                    overrides=BASE + ["seed=%d" % seed,
                                      "hyperparameters.estep_num_samples=%d" % M])
    h = c.hyperparameters
    ne, ns = int(h.num_envs), int(h.num_steps)
    nmb, nep = int(h.num_mini_batches), int(h.num_epochs)
    tts, nev = int(h.total_time_steps), int(h.num_eval)
    train_steps = tts // ns // ne
    eval_interval = train_steps // nev
    outer = train_steps // eval_interval + int(train_steps % eval_interval > 0)
    iters = outer * eval_interval
    return dict(
        M=int(h.estep_num_samples),
        states_per_iteration=ne * ns,
        states_per_minibatch=(ne * ns) // nmb,
        num_mini_batches=nmb, num_epochs=nep,
        updates_per_iteration=nep * nmb,
        iterations=iters,
        total_updates=iters * nep * nmb,
        env_steps=iters * ne * ns,
        eval_interval=eval_interval, num_eval=nev,
        freeze_sigma=h.get("freeze_sigma", "ABSENT"),
        sqrt_rho=float(h.get("sqrt_rho", -1)),
        ent_start=float(h.ent_start),
        update_entropy_lagrangian=bool(h.update_entropy_lagrangian),
        faithful_same_point=bool(h.faithful_same_point),
        fresh_minibatch_key=bool(h.fresh_minibatch_key),
        log_cov_diag=bool(h.get("log_cov_diag", False)),
        kl_num_samples=h.get("kl_num_samples", "ABSENT"),
        eps_e=float(h.eps_e), kl_bound=float(h.kl_bound),
    )


def tag(M, seed):
    v = "" if M == 32 else "_m%d" % M
    return "WalkerRun_weighted_mle%s_s%d" % (v, seed)


# every key that Sec. 4.2 requires byte-identical across M
INVARIANT = ["states_per_iteration", "states_per_minibatch", "num_mini_batches",
             "num_epochs", "updates_per_iteration", "iterations", "total_updates",
             "env_steps", "eval_interval", "num_eval", "freeze_sigma", "sqrt_rho",
             "ent_start", "update_entropy_lagrangian", "faithful_same_point",
             "fresh_minibatch_key", "log_cov_diag", "kl_num_samples", "eps_e",
             "kl_bound"]

def main():
    ref = resolve(32, 301)                       # the WML-32 control
    print("=== reference: WML-32 control, seed 301, resolved from config ===")
    for k in INVARIANT:
        print("  %-26s %s" % (k, ref[k]))
    print("  %-26s %s" % ("M", ref["M"]))
    print("  %-26s %s" % ("export tag", tag(32, 301)))

    ok, rows = True, []
    for M in (128, 512):
        for seed in SEEDS:
            r = resolve(M, seed)
            bad = [k for k in INVARIANT if r[k] != ref[k]]
            if r["M"] != M:
                bad.append("M(resolved=%s)" % r["M"])
            t = tag(M, seed)
            if "_m%d" % M not in t:
                bad.append("tag does not encode M")
            if t == tag(32, seed):
                bad.append("tag collides with the control")
            p = partition_for(M, seed)
            rows.append(dict(M=M, seed=seed, tag=t, partition=p,
                             control_partition=CONTROL_PARTITION[seed],
                             paired=(p == CONTROL_PARTITION[seed]),
                             mismatching=bad))
            ok &= not bad

    print("\n=== per-cell assertions (16 runs) ===")
    print("  %-4s %-5s %-34s %-6s %-8s %s" %
          ("M", "seed", "export tag", "part", "paired?", "budget vs WML-32"))
    for r in rows:
        print("  %-4d %-5d %-34s %-6s %-8s %s" %
              (r["M"], r["seed"], r["tag"], r["partition"],
               "yes" if r["paired"] else "NO (disclosed)",
               "IDENTICAL" if not r["mismatching"] else "MISMATCH " + ",".join(r["mismatching"])))

    mism = [r for r in rows if not r["paired"]]
    print("\n  hardware-paired cells: %d/16" % sum(r["paired"] for r in rows))
    print("  disclosed mismatches : %d  (prereg Sec. 2 / Sec. 7 condition iii)" % len(mism))
    for r in mism:
        print("     M=%d seed %d runs on %s, its M=128 counterpart on %s"
              % (r["M"], r["seed"], r["partition"], r["control_partition"]))

    if len(sys.argv) > 2 and sys.argv[1] == "--json":
        json.dump(dict(reference=ref, cells=rows), open(sys.argv[2], "w"),
                  indent=1, default=str)
    print("\n%s" % ("ALL BUDGET ASSERTIONS PASS" if ok else "ASSERTION FAILURE"))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
