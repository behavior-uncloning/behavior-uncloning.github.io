// Quadruped: recorded Unitree Go1 base trajectories (x, y, yaw at 50 Hz) replayed with the Go1
// model (MuJoCo Menagerie unitree_go1, BSD-3-Clause). Joint angles were not logged, so the legs
// follow a procedural trot whose phase is tied to the recorded base motion (illustrative only).
import * as THREE from "three";
import { SLATE, loadMesh, modePaths, goalDisc, sample, lerpAngle } from "./r3d-common.js";

const MESH_DIR = new URL("../3d/go1/", import.meta.url).href;
const HZ = 50;
const BODY_Z = 0.29; // trunk height above the floor
const LINK = 0.213; // thigh and calf lengths
const FOOT_R = 0.023;
const STRIDE = 0.32; // body travel per gait cycle (m)
const LIFT = 0.06; // swing-foot clearance (m)
const PILLAR_H = 0.4;

// [name, hip offset, hip mesh quaternion (x, y, z, w), thigh side (+1 left), thigh mesh, gait phase]
const LEGS = [
  ["FR", [0.1881, -0.04675, 0], [0, 0, 0, 1], -1, "thigh_mirror", 0],
  ["FL", [0.1881, 0.04675, 0], [0, 0, 0, 1], 1, "thigh", 0.5],
  ["RR", [-0.1881, -0.04675, 0], [0, 0, -1, 0], -1, "thigh_mirror", 0.5],
  ["RL", [-0.1881, 0.04675, 0], [0, 1, 0, 0], 1, "thigh", 0],
];

const material = new THREE.MeshStandardMaterial({ color: 0x3d434c, roughness: 0.45, metalness: 0.2 });
const mesh = (name) => loadMesh(`${MESH_DIR}${name}.glb`, material);

/** Go1 with hip / thigh / calf joints; returns {root, legs}. Meshes stream in asynchronously. */
const buildGo1 = (markDirty) => {
  const root = new THREE.Group();
  const attach = (parent, name, quat) => {
    mesh(name)
      .then((m) => {
        if (quat) m.quaternion.set(...quat);
        parent.add(m);
        markDirty();
      })
      .catch((e) => console.warn(`Go1 mesh ${name} failed to load:`, e));
  };
  attach(root, "trunk");
  const legs = LEGS.map(([, hipPos, hipQuat, side, thighMesh, phase]) => {
    const hip = new THREE.Object3D();
    hip.position.set(...hipPos);
    attach(hip, "hip", hipQuat);
    const thigh = new THREE.Object3D();
    thigh.position.set(0, 0.08 * side, 0);
    attach(thigh, thighMesh);
    const calf = new THREE.Object3D();
    calf.position.set(0, 0, -LINK);
    attach(calf, "calf");
    thigh.add(calf);
    hip.add(thigh);
    root.add(hip);
    return { thigh, calf, phase };
  });
  return { root, legs };
};

// Sagittal two-link IK (equal link lengths): foot at (fx, fz) relative to the thigh joint.
const legIK = (fx, fz) => {
  const d = Math.min(2 * LINK - 1e-4, Math.hypot(fx, fz));
  const knee = -2 * Math.acos(d / (2 * LINK));
  const dir = Math.atan2(-fx, -fz); // thigh+calf bisector: foot = d * (-sin, -cos)
  return [dir - knee / 2, knee];
};

// Per-trajectory gait phase (cycles) and stride amplitude, from the recorded base motion.
const gaitCache = new WeakMap();
const gaitFor = (traj) => {
  if (gaitCache.has(traj)) return gaitCache.get(traj);
  const p = traj.points;
  const n = p.length;
  const step = new Float32Array(n);
  for (let i = 1; i < n; i++) {
    const turn = Math.abs(lerpAngle(p[i - 1][2], p[i][2], 1) - p[i - 1][2]);
    step[i] = Math.hypot(p[i][0] - p[i - 1][0], p[i][1] - p[i - 1][1]) + 0.15 * turn;
  }
  const phase = new Float32Array(n);
  const amp = new Float32Array(n);
  for (let i = 1; i < n; i++) phase[i] = phase[i - 1] + step[i] / STRIDE;
  for (let i = 0; i < n; i++) {
    let s = 0;
    let c = 0;
    for (let j = Math.max(1, i - 6); j <= Math.min(n - 1, i + 6); j++, c++) s += step[j];
    const speed = c ? (s / c) * HZ : 0;
    const x = Math.min(1, speed / 0.25);
    amp[i] = x * x * (3 - 2 * x);
  }
  const g = { phase, amp };
  gaitCache.set(traj, g);
  return g;
};

export const STAGE = {
  cam: [1.95, 3.85, 4.3],
  target: [2.55, 0, 0.1],
  fog: [7, 15],
  shadow: { center: [2.6, 0, 0], half: 3.2, light: [0.9, 1.6, 0.6] },
  fit: 1,
  dist: [1.2, 9],
};

export const buildQuadruped = (task, colors, markDirty) => {
  const group = new THREE.Group();
  const r0 = task.modes[0].rollouts[0];
  const [px, py] = r0.pillar_xy || [2.5, 0];
  const [gx, gy] = r0.goal_xy || [4.5, 0];
  const sc = task.scene;

  const pillar = new THREE.Mesh(
    new THREE.CylinderGeometry(sc.pillar_radius, sc.pillar_radius, PILLAR_H, 48),
    new THREE.MeshStandardMaterial({ color: SLATE, roughness: 0.6 })
  );
  pillar.rotation.x = Math.PI / 2;
  pillar.position.set(px, py, PILLAR_H / 2);
  pillar.castShadow = pillar.receiveShadow = true;
  group.add(pillar);
  group.add(goalDisc(gx, gy, sc.goal_radius, 0.012));

  // Start pad at the mean start position.
  const starts = task.modes.flatMap((m) => m.rollouts.map((r) => r.points[0]));
  const sx = starts.reduce((s, p) => s + p[0], 0) / starts.length;
  const sy = starts.reduce((s, p) => s + p[1], 0) / starts.length;
  const pad = new THREE.Mesh(
    new THREE.RingGeometry(0.2, 0.212, 64),
    new THREE.MeshBasicMaterial({ color: 0x8b93a1, transparent: true, opacity: 0.6 })
  );
  pad.position.set(sx, sy, 0.0006);
  group.add(pad);

  const trajModes = task.modes.map((m) => ({ id: m.id, trajs: m.rollouts }));
  const modes = modePaths(trajModes, colors, (r) => r.points, { radius: 0.011, z: 0.004, minStep: 0.01 });
  for (const m of modes) group.add(m.group);

  const dog = buildGo1(markDirty);
  group.add(dog.root);

  let current = null;
  return {
    group,
    modes,
    stage: STAGE,
    setCurrent(traj) {
      current = traj;
    },
    setTime(t) {
      if (!current) return;
      const { a, b, k, i } = sample(current.points, t);
      const gait = gaitFor(current);
      const j = Math.min(current.points.length - 1, i + 1);
      const phase = gait.phase[i] + (gait.phase[j] - gait.phase[i]) * k;
      const amp = gait.amp[i] + (gait.amp[j] - gait.amp[i]) * k;
      const bob = 0.006 * amp * Math.cos(4 * Math.PI * phase);
      dog.root.position.set(a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k, BODY_Z + bob);
      dog.root.rotation.z = lerpAngle(a[2], b[2], k);
      for (const leg of dog.legs) {
        const psi = 2 * Math.PI * (phase + leg.phase);
        const fx = -(STRIDE / 4) * amp * Math.cos(psi);
        const fz = -(BODY_Z + bob - FOOT_R) + LIFT * amp * Math.max(0, Math.sin(psi));
        const [hip, knee] = legIK(fx, fz);
        leg.thigh.rotation.y = hip;
        leg.calf.rotation.y = knee;
      }
    },
  };
};
