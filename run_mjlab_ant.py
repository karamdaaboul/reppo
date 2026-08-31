"""Launcher: run the ORIGINAL REPPO torch trainer on an mjlab task.

Both this repo and unitree_rl_mjlab own a top-level `src` package, and the
latter is installed via a hook-based editable finder that hard-maps `src`.
Sequence: (1) build the mjlab env with unitree's `src`, (2) strip the finder
and purge the module cache, (3) register the pre-built env and run this
repo's hydra main with CLI overrides.
"""

import os
import sys

REPPO_ROOT = "/home/human/workspaces/reppo_original"
UNITREE_ROOT = "/home/human/workspaces/unitree_rl_mjlab"
TASK = "Ant-Flat"
NUM_ENVS = 1024

# ---- Phase 1: build the mjlab env (unitree `src` must win) ----
# The interpreter auto-prepends the script's directory, which would make `src`
# resolve to THIS repo. Strip it and force unitree first.
sys.path = [
    p for p in sys.path
    if os.path.abspath(p or os.getcwd()) != os.path.abspath(REPPO_ROOT)
]
sys.path.insert(0, UNITREE_ROOT)

import mjlab.tasks  # noqa: F401,E402
from mjlab.tasks.registry import load_env_cfg
import src.tasks  # noqa: F401  (registers Ant-* tasks)
from src.envs import build_env

env_cfg = load_env_cfg(TASK)
env_cfg.scene.num_envs = NUM_ENVS
env_cfg.seed = 0
env = build_env(env_cfg, "cuda:0", render_mode=None)
print(f"[launcher] built mjlab env {TASK} with {NUM_ENVS} envs")

# ---- Phase 2: swap `src` packages ----
sys.meta_path = [
    f for f in sys.meta_path
    if "unitree" not in (getattr(type(f), "__module__", "") or "") and "unitree" not in repr(f)
]
sys.path = [p for p in sys.path if os.path.abspath(p) != os.path.abspath(UNITREE_ROOT)]
for name in list(sys.modules):
    if name == "src" or name.startswith("src."):
        del sys.modules[name]
sys.path.insert(0, REPPO_ROOT)

from src.env_utils.torch_wrappers.mjlab_env import register_prebuilt  # noqa: E402

register_prebuilt(TASK, env)
print("[launcher] handed env to reppo wrapper; starting their trainer")

# ---- Phase 3: run their hydra main exactly as `python src/torchrl/reppo.py`
# (hydra resolves the relative config_path filesystem-style only when the task
# module runs as __main__; imported, it demands an importable `config` module).
import runpy  # noqa: E402

sys.argv = [
    "reppo.py",
    "env=mjlab",
    f"env.name={TASK}",
    "wandb.mode=online",
    "wandb.entity=uqerh-kit",
    "wandb.project=mjlab",
    "name=reppo_original_mjlab_ant",
]
runpy.run_path(REPPO_ROOT + "/src/torchrl/reppo.py", run_name="__main__")
