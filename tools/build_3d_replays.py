#!/usr/bin/env python
"""Build assets/data/sim_3d_replays.json for the 3D replay viewer from raw ManiSkill3 demos.

Source: the private HF dataset `haohw/behavior-uncloning-replay-data` (raw/maniskill3/...), e.g.
    hf download haohw/behavior-uncloning-replay-data --repo-type dataset --local-dir ./bu-rollouts
    python tools/build_3d_replays.py ./bu-rollouts

Per frame we keep the Franka joint positions (7 arm + 2 finger) and the cube pose, in ManiSkill
world coordinates (z up, metres; quaternions wxyz). Static geometry that is not recorded in the
env states (wall, pillars) is taken from the env definitions:
  - Push-Wall    : PushCubeObstacle-v1 wall, half sizes (0.005, 0.08, 0.02) at the origin.
  - Push-Pillars : three thin pillars at x = -0.06, y in {-0.102, 0, 0.102}; radius 0.006 m is
                   consistent with the closest cube approach in the demos (~0.027 m centre distance).
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import h5py
import numpy as np

logger = logging.getLogger(__name__)

OUT = Path(__file__).resolve().parents[1] / "assets/data/sim_3d_replays.json"
MAX_FRAMES = 160
PER_MODE = {"pushwall": 4, "pushpillars": 3}

TASKS = {
    "pushwall": {
        "label": "Push-Wall",
        "modes": [("left", "Left route", "push_wall/left_train.h5"), ("right", "Right route", "push_wall/right_train.h5")],
        "static": {"wall": {"center": [0.0, 0.0, 0.02], "half": [0.005, 0.08, 0.02]}, "goal_radius": 0.1},
    },
    "pushpillars": {
        "label": "Push-Pillars",
        "modes": [
            ("far_left", "Far left", "push_pillars/far_left_train.h5"),
            ("left_gap", "Left gap", "push_pillars/left_gap_train.h5"),
            ("right_gap", "Right gap", "push_pillars/right_gap_train.h5"),
            ("far_right", "Far right", "push_pillars/far_right_train.h5"),
        ],
        "static": {
            "pillars": [[-0.06, -0.102], [-0.06, 0.0], [-0.06, 0.102]],
            "pillar_radius": 0.006,
            "pillar_height": 0.05,
            "finish_line": {"x": 0.06, "half_width": 0.25},
        },
    },
}


def r(a: np.ndarray, nd: int = 4) -> list:
    return np.round(np.asarray(a, dtype=np.float64), nd).tolist()


def trajectory(g: h5py.Group) -> dict:
    """Trim at first success (+4 frames), downsample to <= MAX_FRAMES, keep qpos + cube pose."""
    arts = g["env_states/articulations/panda_wristcam"][:]  # (T, 31): root pose 13 | qpos 9 | qvel 9
    cube = g["env_states/actors/cube"][:]  # (T, 13): pos 3 | quat 4 (wxyz) | vel 6
    t_last = len(cube) - 1
    if "success" in g:
        s = np.asarray(g["success"][:])
        if s.any():
            t_last = min(t_last, int(np.argmax(s)) + 4)
    idx = np.unique(np.linspace(0, t_last, min(t_last + 1, MAX_FRAMES)).round().astype(int))
    out = {"qpos": r(arts[idx, 13:22]), "cube": r(cube[idx, :7])}
    if "goal_region" in g["env_states/actors"]:
        out["goal"] = r(g["env_states/actors/goal_region"][0, :3])
    return out


def main(root: Path) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raw = root / "raw/maniskill3"
    tasks = {}
    robot_root = None
    for tid, spec in TASKS.items():
        modes = []
        for mid, label, rel in spec["modes"]:
            with h5py.File(raw / rel, "r") as f:
                keys = sorted(f.keys(), key=lambda k: int(k.split("_")[1]))
                trajs = [trajectory(f[k]) for k in keys[: PER_MODE[tid]]]
                if robot_root is None:
                    robot_root = r(f[keys[0]]["env_states/articulations/panda_wristcam"][0, :7])
            modes.append({"id": mid, "label": label, "trajs": trajs})
            logger.info("%s/%s: %d trajectories, %s frames", tid, mid, len(trajs), [len(t["qpos"]) for t in trajs])
        tasks[tid] = {"label": spec["label"], "static": spec["static"], "modes": modes}
    data = {
        "version": 1,
        "source": "ManiSkill3 demonstration trajectories (raw env states); see tools/build_3d_replays.py",
        "robot_root": robot_root,
        "cube_half": 0.02,
        "tasks": tasks,
    }
    OUT.write_text(json.dumps(data, separators=(",", ":")))
    logger.info("wrote %s (%.0f KB)", OUT, OUT.stat().st_size / 1024)


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "./bu-rollouts"))
