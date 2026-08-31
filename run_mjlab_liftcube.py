"""Launcher: run the ORIGINAL REPPO torch trainer on mjlab's lift-cube task.

Purpose: separate "REPPO the algorithm cannot do contact-rich manipulation" from
"safe_rl's port has a bug". safe_rl's REPPO v24/v25 reach exactly 0.0000 success
on Mjlab-Lift-Cube-Yam while the task's registered PPO reaches ~0.80, so the next
question is whether the authors' own implementation behaves the same way.

Unlike run_mjlab_ant.py this needs NO `src` package swapping: Mjlab-Lift-Cube-Yam
is registered by mjlab itself, so unitree_rl_mjlab -- which owns a conflicting
top-level `src` and forced that repo's meta_path surgery -- is never imported.

Run with the dedicated venv (mjlab 1.2.0 pinned against mujoco/mujoco-warp 3.5.0
and warp-lang 1.12.1; newer mujoco-warp removed `ls_parallel` and breaks mjlab):

    cd /home/human/workspaces/reppo_original
    MUJOCO_GL=egl /home/human/venvs/reppo_ref/bin/python run_mjlab_liftcube.py \
        hyperparameters.total_time_steps=40000000

Any extra CLI args are appended verbatim to the hydra command line.
"""

import os
import sys

REPPO_ROOT = os.path.dirname(os.path.abspath(__file__))
TASK = os.environ.get("LIFTCUBE_TASK", "Mjlab-Lift-Cube-Yam")
NUM_ENVS = int(os.environ.get("LIFTCUBE_NUM_ENVS", "1024"))
SEED = int(os.environ.get("LIFTCUBE_SEED", "0"))
DEVICE = os.environ.get("LIFTCUBE_DEVICE", "cuda:0")

# ---- Phase 1: build the mjlab env ----
# NUM_ENVS must match hyperparameters.num_envs (default 1024): MjlabEnv consumes
# a pre-built env and reports env.num_envs, so a mismatch silently changes the
# trainer's batch shape rather than erroring.
import mjlab.tasks  # noqa: E402,F401  (import side effect registers every task)
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.tasks.registry import list_tasks, load_env_cfg  # noqa: E402

if TASK not in list_tasks():
    raise SystemExit(f"unknown task {TASK!r}; available: {', '.join(sorted(list_tasks()))}")

env_cfg = load_env_cfg(TASK)
env_cfg.scene.num_envs = NUM_ENVS
env_cfg.seed = SEED
env = ManagerBasedRlEnv(cfg=env_cfg, device=DEVICE, render_mode=None)
print(f"[launcher] built mjlab env {TASK}: {NUM_ENVS} envs, "
      f"{env.action_manager.total_action_dim} actions, max_ep_len {env.max_episode_length}",
      flush=True)

# ---- Phase 2: hand it to the reference wrapper ----
sys.path.insert(0, REPPO_ROOT)
from src.env_utils.torch_wrappers.mjlab_env import register_prebuilt  # noqa: E402

register_prebuilt(TASK, env)
print("[launcher] handed env to reppo wrapper; starting the authors' trainer", flush=True)

# ---- Phase 3: run their hydra main exactly as `python src/torchrl/reppo.py` ----
# hydra resolves the relative config_path filesystem-style only when the task
# module runs as __main__; imported, it demands an importable `config` module.
import runpy  # noqa: E402

user_args = sys.argv[1:]
user_keys = {a.split("=", 1)[0] for a in user_args if "=" in a}
defaults = [
    "env=mjlab_liftcube",
    f"env.name={TASK}",
    f"seed={SEED}",
    f"hyperparameters.num_envs={NUM_ENVS}",
    "wandb.mode=online",
    "wandb.entity=uqerh-kit",
    "wandb.project=mjlab",
    "name=reppo_original_liftcube",
]
# Drop any default the caller already set — hydra treats a repeated key as a
# duplicate override rather than a last-wins assignment.
sys.argv = ["reppo.py"] + [d for d in defaults if d.split("=", 1)[0] not in user_keys] + user_args
print(f"[launcher] hydra argv: {' '.join(sys.argv[1:])}", flush=True)
runpy.run_path(os.path.join(REPPO_ROOT, "src/torchrl/reppo.py"), run_name="__main__")
