#!/usr/bin/env python
"""Build assets/data/sim_task_replays.json from REAL trajectories.

Sources (on the TAMU HPRC cluster, paths absolute):
  - Push-Wall   : ManiSkill3 PushCubeObstacle demo h5 (mode-pure successful demos), cube top-down (x,y,yaw)
  - Push-Pillars: ManiSkill3 PushCube3Wall  demo h5 (4 routes), cube top-down (x,y,yaw)
  - Push-T      : MoRE-edited DP policy EVAL rollouts (v8 combined-close25 gamma sweep), pusher + T-block
  - Quadruped   : preserved verbatim from the existing JSON (already real Go1 base-pose rollouts)

Run with the openpi venv python (numpy + h5py):
  /home/haohw_tamu.edu/openpi/.venv/bin/python tools/build_sim_replays.py
Writes:  assets/data/sim_task_replays.real.json   (review, then mv into place)
"""
import json, os, glob, pickle
import numpy as np
import h5py

ROOT = "/scratch/project/prj-02-phai-lab/haohw/behavior-uncloning"
WEB  = "/scratch/project/prj-02-phai-lab/haohw/behavior-uncloning.github.io"
EXISTING = os.path.join(WEB, "assets/data/sim_task_replays.json")
OUT = os.path.join(WEB, "assets/data/sim_task_replays.real.json")

# ----------------------------------------------------------------------------- helpers
def quat_yaw(q):  # q = (...,4) as (w,x,y,z); planar yaw about z
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

def downsample(idx_len, max_n):
    if idx_len <= max_n:
        return np.arange(idx_len)
    return np.unique(np.linspace(0, idx_len - 1, max_n).round().astype(int))

def r4(v):
    return round(float(v), 4)

def spread_pick(seeds, n):
    """Pick n seeds spread across the sorted unique seed list."""
    s = sorted(set(seeds))
    if len(s) <= n:
        return s
    idx = np.linspace(0, len(s) - 1, n).round().astype(int)
    return [s[i] for i in sorted(set(idx))]

# ----------------------------------------------------------------------------- ManiSkill cube tasks
def maniskill_mode(h5_path, max_traj, max_pts):
    """Return list of rollout dicts {points:[[x,y,yaw]], success, seed, n} from a demo h5."""
    f = h5py.File(h5_path, "r")
    keys = sorted(f.keys(), key=lambda k: int(k.split("_")[1]))
    out = []
    for i, k in enumerate(keys[:max_traj]):
        g = f[k]
        cube = g["env_states/actors/cube"][:]          # (T,13) pos[3]+quat[4]+vel[6]
        # trim at first task success (+ short settle) so the path ends at the goal/finish
        t_last = len(cube) - 1
        if "success" in g:
            sarr = np.asarray(g["success"][:])
            if sarr.any():
                t_last = min(len(cube) - 1, int(np.argmax(sarr)) + 4)
        cube = cube[:t_last + 1]
        xy = cube[:, :2]
        yaw = quat_yaw(cube[:, 3:7])
        sel = downsample(len(xy), max_pts)
        pts = [[r4(xy[t, 0]), r4(xy[t, 1]), r4(yaw[t])] for t in sel]
        succ = bool(g["success"][-1]) if "success" in g else True
        out.append({"pts": pts, "success": succ})
    f.close()
    return out

def build_pushwall():
    modes_def = [("left", "Left route", "#236c9c", "data/maniskill3/obstacle-aux-data/dp_data/left_train.h5"),
                 ("right", "Right route", "#c9712f", "data/maniskill3/obstacle-aux-data/dp_data/right_train.h5")]
    modes = []
    allx, ally = [], []
    for mid, mlabel, color, rel in modes_def:
        rolls = maniskill_mode(os.path.join(ROOT, rel), max_traj=8, max_pts=180)
        ro = []
        for j, r in enumerate(rolls):
            ro.append({"id": f"{mid}_{j:02d}", "mode": mid, "seed": 100 + j,
                       "success": r["success"], "points": r["pts"]})
            for x, y, _ in r["pts"]:
                allx.append(x); ally.append(y)
        modes.append({"id": mid, "label": mlabel, "color": color, "rollouts": ro})
    # real PushCube goal_radius = 0.1 at (cube_x + 0.1 + r, cube_y); goal_region fixed at (0.15, 0)
    GOAL_R = 0.1
    xr = [r4(min(allx + [-0.005]) - 0.02), r4(max(allx + [0.15 + GOAL_R]) + 0.03)]
    yr = [r4(min(ally + [-GOAL_R]) - 0.03), r4(max(ally + [GOAL_R]) + 0.03)]
    scene = {"kind": "pushwall", "x_range": xr, "y_range": yr, "tick_step": 0.1,
             "wall": {"center": [0.0, 0.0], "size": [0.01, 0.16]},
             "goal": {"center": [0.15, 0.0], "radius": GOAL_R},
             "cube_size": 0.04}
    return {"label": "Push-Wall",
            "source": "Real ManiSkill3 PushCubeObstacle demonstration trajectories (mode-pure, successful); top-down cube pose (x, y, yaw).",
            "sample_rate_hz": 30, "scene": scene, "modes": modes}

def build_pushpillars():
    modes_def = [("far_left", "Far left", "#1f5f8b", "data/maniskill3/push_cube_3wall_dp_v5/far_left_train.h5"),
                 ("left_gap", "Left gap", "#2f8f8f", "data/maniskill3/push_cube_3wall_dp_v5/left_gap_train.h5"),
                 ("right_gap", "Right gap", "#c98a2f", "data/maniskill3/push_cube_3wall_dp_v5/right_gap_train.h5"),
                 ("far_right", "Far right", "#b5562f", "data/maniskill3/push_cube_3wall_dp_v5/far_right_train.h5")]
    modes = []
    allx, ally = [], []
    for mid, mlabel, color, rel in modes_def:
        rolls = maniskill_mode(os.path.join(ROOT, rel), max_traj=6, max_pts=180)
        ro = []
        for j, r in enumerate(rolls):
            ro.append({"id": f"{mid}_{j:02d}", "mode": mid, "seed": 500 + j,
                       "success": r["success"], "points": r["pts"]})
            for x, y, _ in r["pts"]:
                allx.append(x); ally.append(y)
        modes.append({"id": mid, "label": mlabel, "color": color, "rollouts": ro})
    xr = [r4(min(allx + [-0.18]) - 0.02), r4(max(allx + [0.06]) + 0.03)]
    yr = [r4(min(ally) - 0.03), r4(max(ally) + 0.03)]
    scene = {"kind": "pushpillars", "x_range": xr, "y_range": yr, "tick_step": 0.1,
             "pillars": [[-0.06, -0.102], [-0.06, 0.0], [-0.06, 0.102]],
             "pillar_radius": 0.006, "finish_x": 0.06, "cube_size": 0.04}
    return {"label": "Push-Pillars",
            "source": "Real ManiSkill3 PushCube3Wall demonstration trajectories (four routes, successful); top-down cube pose (x, y, yaw).",
            "sample_rate_hz": 30, "scene": scene, "modes": modes}

# ----------------------------------------------------------------------------- Push-T (edited policy rollouts)
MIXED_DEG = 30.0
PXT = 512.0
def cog_of_T(bx, by, bth):
    c, s = np.cos(bth), np.sin(bth)
    return np.array([bx - s * 45, by + c * 45])
def geom_mode(states, cap=50):
    cap = min(cap, len(states))
    if cap < 5:
        return -1
    a = states[:cap, 0:2]; b = states[:cap, 2:4]; th = states[:cap, 4]
    cog = np.stack([cog_of_T(b[t, 0], b[t, 1], th[t]) for t in range(cap)])
    rel = a - cog
    ang = np.unwrap(np.arctan2(rel[:, 1], rel[:, 0]))
    nw = np.degrees(ang - ang[0])
    if abs(nw.min()) > abs(nw.max()) + MIXED_DEG:
        return 1
    if abs(nw.max()) > abs(nw.min()) + MIXED_DEG:
        return 0
    return -1

def px_to_norm(px, py):
    return px / PXT, (PXT - py) / PXT       # x right, y up (flip image y)
def ang_to_deg(rad):
    return -np.degrees(rad)                  # negate to match y-flip

def collect_pusht(side_dirs, want_mode, n_pick, max_pts):
    cand = {}   # seed -> (cov, states, T_end, src)
    for d in side_dirs:
        for p in sorted(glob.glob(os.path.join(d, "**", "*.pkl"), recursive=True)):
            try:
                data = pickle.load(open(p, "rb"))
            except Exception:
                continue
            res = data.get("results") if isinstance(data, dict) else None
            if not (isinstance(res, list) and res and isinstance(res[0], dict) and "states" in res[0]):
                continue
            for r in res:
                rew = np.asarray(r["rewards"]); cov = float(rew.max())
                if cov < 0.95:
                    continue
                if geom_mode(r["states"]) != want_mode:
                    continue
                # trim to first frame coverage>=0.95 (+ short settle), so it ends at the goal
                hit = np.argmax(rew >= 0.95)
                t_end = min(len(r["states"]) - 1, int(hit) + 6)
                if t_end < 10:
                    continue
                seed = r.get("seed", len(cand))
                src = os.path.basename(os.path.dirname(p)) + "/" + os.path.basename(p)
                if seed not in cand or cov > cand[seed][0]:
                    cand[seed] = (cov, r["states"][:t_end + 1], t_end, src)
    seeds = spread_pick(list(cand.keys()), n_pick)
    rolls = []
    for j, seed in enumerate(seeds):
        cov, states, t_end, src = cand[seed]
        sel = downsample(len(states), max_pts)
        pusher = [[r4(px_to_norm(states[t, 0], states[t, 1])[0]),
                   r4(px_to_norm(states[t, 0], states[t, 1])[1]), 0.0] for t in sel]
        block = [[r4(px_to_norm(states[t, 2], states[t, 3])[0]),
                  r4(px_to_norm(states[t, 2], states[t, 3])[1]),
                  r4(ang_to_deg(states[t, 4]))] for t in sel]
        rolls.append({"seed": int(seed), "success": True, "cov": round(cov, 3), "src": src,
                      "pusher": pusher, "block": block})
    return rolls

def build_pusht():
    base = os.path.join(ROOT, "data/pusht_dp/aux_exp/rollout_eval_distill_v8_close25")
    left_dirs = sorted(glob.glob(os.path.join(base, "A_LEFT_*")))
    right_dirs = sorted(glob.glob(os.path.join(base, "A_RIGHT_*")))
    left = collect_pusht(left_dirs, 0, 8, 200)
    right = collect_pusht(right_dirs, 1, 8, 200)
    allx, ally, end_block = [], [], []
    modes = []
    for mid, mlabel, color, rolls in [("left", "Left (CCW)", "#236c9c", left),
                                      ("right", "Right (CW)", "#c9712f", right)]:
        ro = []
        for j, r in enumerate(rolls):
            ro.append({"id": f"{mid}_{j:02d}", "mode": mid, "seed": r["seed"],
                       "success": r["success"], "points": r["pusher"], "block": r["block"]})
            for x, y, _ in r["pusher"]:
                allx.append(x); ally.append(y)
            for x, y, _ in r["block"]:
                allx.append(x); ally.append(y)
            end_block.append(r["block"][-1])
        modes.append({"id": mid, "label": mlabel, "color": color, "rollouts": ro})
    eb = np.array(end_block)  # goal pose = where the T ends on success
    goal_center = [r4(np.median(eb[:, 0])), r4(np.median(eb[:, 1]))]
    goal_angle = r4(np.median(eb[:, 2]))
    xr = [r4(min(allx) - 0.04), r4(max(allx) + 0.04)]
    yr = [r4(min(ally) - 0.04), r4(max(ally) + 0.04)]
    scene = {"kind": "pusht", "x_range": xr, "y_range": yr, "tick_step": 0.2,
             "goal_t": {"center": goal_center, "angle_deg": goal_angle},
             "pusher_radius": 0.03}
    return {"label": "Push-T",
            "source": "Real MoRE-edited DP policy evaluation rollouts (v8 combined-close25 gamma sweep); pusher path + T-block pose, geometric wrap mode (CCW=left, CW=right).",
            "sample_rate_hz": 30, "scene": scene, "modes": modes}

# ----------------------------------------------------------------------------- main
def main():
    existing = json.load(open(EXISTING))
    tasks = {}
    tasks["pushwall"] = build_pushwall()
    tasks["pusht"] = build_pusht()
    tasks["pushpillars"] = build_pushpillars()
    tasks["quadruped"] = existing["tasks"]["quadruped"]   # preserve real Go1 data
    out = {"version": 2, "tasks": tasks}
    json.dump(out, open(OUT, "w"), separators=(",", ":"))

    # diagnostics
    print("WROTE", OUT, "(%.1f KB)" % (os.path.getsize(OUT) / 1024))
    for tk, tv in tasks.items():
        nm = len(tv["modes"]); nr = sum(len(m["rollouts"]) for m in tv["modes"])
        print(f"  {tk:12s} modes={nm} rollouts={nr} x_range={tv['scene']['x_range']} y_range={tv['scene']['y_range']}")
        for m in tv["modes"]:
            lens = [len(r["points"]) for r in m["rollouts"]]
            seeds = [r["seed"] for r in m["rollouts"]]
            print(f"      {m['id']:10s} n={len(m['rollouts'])} pts(min/max)={min(lens)}/{max(lens)} seeds={seeds}")

if __name__ == "__main__":
    main()
