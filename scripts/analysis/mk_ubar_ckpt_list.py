"""Build the registered Step 2/3 checkpoint list (docs/prereg_ubar_ratio.md Sec. 2).

Confirmatory seeds 101-108 for the unpadded tasks and 0-4 for the padded Walker set.
Seeds 201 (g1 pathwise) and 234 (hopper weighted_mle) are exploratory and excluded.
Also (re)writes reports/artifacts/ubar_checkpoint_audit.csv from meta.json.
"""
import csv, glob, json, os, collections

rows = []
for m in sorted(glob.glob("exports/*_final/meta.json")):
    d = json.load(open(m)); tag = m.split("/")[1]
    rows.append(dict(
        checkpoint=tag, env=d["env_name"], mode=d["actor_update_mode"], seed=d["seed"],
        action_dim=d["action_dim"], action_pad=d.get("action_pad", 0) or 0,
        actor_out_dim=d["actor_kwargs"]["action_dim"],
        critic_action_dim=d["critic_kwargs"]["action_dim"],
        M=d["estep_num_samples"], eps_e=d["eps_e"],
        min_std=d["actor_kwargs"]["min_std"], with_eta=d["actor_kwargs"].get("with_eta"),
        alpha_entropy=d["alpha_entropy"], alpha_kl=d["alpha_kl"],
        eta_final=(d.get("eta_curve") or [None])[-1], ess_final=d.get("ess_final"),
        final_return=d["final_eval_return"], obs_dim=d["obs_dim"],
        critic_obs_dim=d["critic_obs_dim"], max_episode_steps=d["max_episode_steps"],
        path="exports/" + tag))
os.makedirs("reports/artifacts", exist_ok=True)
with open("reports/artifacts/ubar_checkpoint_audit.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
    for r in rows: w.writerow(r)

CONF, PAD = set(range(101, 109)), set(range(5))
keep = sorted(r["path"] for r in rows
              if (r["action_pad"] == 0 and r["seed"] in CONF)
              or (r["action_pad"] == 16 and r["seed"] in PAD))
open("reports/artifacts/ubar_ckpt_list.txt", "w").write("\n".join(keep) + "\n")
print("exports scanned      : %d" % len(rows))
print("registered checkpoints: %d" % len(keep))
print("excluded (exploratory): %s"
      % [r["checkpoint"] for r in rows if r["path"] not in keep])
c = collections.Counter((r["env"], r["mode"], r["action_dim"], r["action_pad"])
                        for r in rows if r["path"] in keep)
for k, v in sorted(c.items(), key=lambda x: x[0][2]):
    print("  %-26s %-13s d=%-3d pad=%-3d n=%d" % (k[0], k[1], k[2], k[3], v))
print("distinct estimator-visible d: %s" % sorted({r["action_dim"] for r in rows}))
print("any d==21: %s" % any(r["action_dim"] == 21 for r in rows))
