// Push-Wall / Push-Pillars: recorded ManiSkill3 demonstrations replayed with the Franka arm.
// Data: assets/data/sim_3d_replays.json (tools/build_3d_replays.py) - per frame the 7+2 joint
// positions and the cube pose.
import * as THREE from "three";
import { SLATE, GOOD, loadMesh, modePaths, goalDisc, sample } from "./r3d-common.js";

const MESH_DIR = new URL("../3d/panda/", import.meta.url).href;

// panda_v3.urdf kinematic chain: [link mesh, joint origin xyz, joint origin rpy].
const CHAIN = [
  ["link1", [0, 0, 0.333], [0, 0, 0]],
  ["link2", [0, 0, 0], [-Math.PI / 2, 0, 0]],
  ["link3", [0, -0.316, 0], [Math.PI / 2, 0, 0]],
  ["link4", [0.0825, 0, 0], [Math.PI / 2, 0, 0]],
  ["link5", [-0.0825, 0.384, 0], [-Math.PI / 2, 0, 0]],
  ["link6", [0, 0, 0], [Math.PI / 2, 0, 0]],
  ["link7", [0.088, 0, 0], [Math.PI / 2, 0, 0]],
];

// URDF rpy is fixed-axis roll-pitch-yaw: R = Rz(yaw) Ry(pitch) Rx(roll) -> Euler order "ZYX".
const originNode = (xyz, rpy) => {
  const o = new THREE.Object3D();
  o.position.set(...xyz);
  o.rotation.set(rpy[0], rpy[1], rpy[2], "ZYX");
  return o;
};

const mesh = (name) => loadMesh(`${MESH_DIR}${name}.glb`);

/** Franka arm: {root, setQpos(q[9])}. Built once and shared by both tasks. */
let robotPromise = null;
const getRobot = (rootPose) => {
  robotPromise ??= (async () => {
    // Fetch all link meshes in parallel, then assemble the chain.
    const names = ["link0", ...CHAIN.map((c) => c[0]), "hand", "finger", "finger"];
    const meshes = await Promise.all(names.map(mesh));
    const root = new THREE.Group();
    root.position.set(rootPose[0], rootPose[1], rootPose[2]);
    root.quaternion.set(rootPose[4], rootPose[5], rootPose[6], rootPose[3]);
    root.add(meshes[0]);
    const joints = [];
    let parent = root;
    CHAIN.forEach(([, xyz, rpy], i) => {
      const origin = originNode(xyz, rpy);
      const motion = new THREE.Object3D();
      origin.add(motion);
      motion.add(meshes[1 + i]);
      parent.add(origin);
      joints.push(motion);
      parent = motion;
    });
    const hand = originNode([0, 0, 0.107], [0, 0, -Math.PI / 4]);
    parent.add(hand);
    hand.add(meshes[CHAIN.length + 1]);
    const fingers = [1, -1].map((dir, i) => {
      const base = originNode([0, 0, 0.0584], [0, 0, 0]);
      const slide = new THREE.Object3D();
      base.add(slide);
      const holder = new THREE.Object3D();
      if (dir < 0) holder.rotation.z = Math.PI;
      holder.add(meshes[CHAIN.length + 2 + i]);
      slide.add(holder);
      hand.add(base);
      return slide;
    });
    return {
      root,
      setQpos: (q) => {
        for (let i = 0; i < 7; i++) joints[i].rotation.z = q[i];
        fingers[0].position.y = q[7];
        fingers[1].position.y = -q[8];
      },
    };
  })().catch((e) => {
    robotPromise = null; // allow a retry on the next task switch
    throw e;
  });
  return robotPromise;
};

export const STAGE = {
  cam: [1.25, 1.2, 1.35],
  target: [-0.24, 0.3, 0],
  fog: [3.2, 6.5],
  shadow: { center: [0, 0, 0], half: 1, light: [0.9, 1.6, 0.6] },
  fit: 0.35,
  dist: [0.5, 3.5],
};

export const buildManiSkill = (data, id, colors, markDirty) => {
  const task = data.tasks[id];
  const st = task.static;
  const group = new THREE.Group();
  const slate = new THREE.MeshStandardMaterial({ color: SLATE, roughness: 0.6 });
  if (st.wall) {
    const [hx, hy, hz] = st.wall.half;
    const wall = new THREE.Mesh(new THREE.BoxGeometry(2 * hx, 2 * hy, 2 * hz), slate);
    wall.position.set(...st.wall.center);
    wall.castShadow = wall.receiveShadow = true;
    group.add(wall);
  }
  for (const [x, y] of st.pillars || []) {
    const p = new THREE.Mesh(new THREE.CylinderGeometry(st.pillar_radius, st.pillar_radius, st.pillar_height, 20), slate);
    p.rotation.x = Math.PI / 2; // cylinder axis y -> z
    p.position.set(x, y, st.pillar_height / 2);
    p.castShadow = true;
    group.add(p);
  }
  if (st.finish_line) {
    const line = new THREE.Mesh(
      new THREE.PlaneGeometry(0.006, 2 * st.finish_line.half_width),
      new THREE.MeshBasicMaterial({ color: GOOD, transparent: true, opacity: 0.85 })
    );
    line.position.set(st.finish_line.x, 0, 0.0008);
    group.add(line);
  }
  const goal = task.modes[0].trajs[0].goal;
  if (st.goal_radius && goal) group.add(goalDisc(goal[0], goal[1], st.goal_radius, 0.003));

  const modes = modePaths(task.modes, colors, (tr) => tr.cube, { radius: 0.0024, z: 0.0012, minStep: 0.002 });
  for (const m of modes) group.add(m.group);

  const half = data.cube_half;
  const cube = new THREE.Mesh(
    new THREE.BoxGeometry(2 * half, 2 * half, 2 * half),
    new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.45 })
  );
  cube.castShadow = true;
  group.add(cube);

  let robot = null;
  let current = null;
  let lastT = 0;
  getRobot(data.robot_root)
    .then((r) => {
      robot = r;
      group.add(r.root);
      scene.setTime(lastT);
      markDirty();
    })
    .catch((e) => console.warn("Franka meshes failed to load:", e));

  const scene = {
    group,
    modes,
    stage: STAGE,
    // The arm is shared by Push-Wall and Push-Pillars: re-attach it when this scene is shown.
    activate() {
      if (robot) {
        group.add(robot.root);
        scene.setTime(lastT);
      }
    },
    setCurrent(traj, color) {
      current = traj;
      cube.material.color.copy(color);
    },
    setTime(t) {
      lastT = t;
      if (!current) return;
      const { a, b, k } = sample(current.cube, t);
      cube.position.set(a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k, a[2] + (b[2] - a[2]) * k);
      cube.quaternion.set(a[4], a[5], a[6], a[3]);
      if (robot) {
        const q = sample(current.qpos, t);
        robot.setQpos(q.a.map((v, d) => v + (q.b[d] - v) * q.k));
      }
    },
  };
  return scene;
};
