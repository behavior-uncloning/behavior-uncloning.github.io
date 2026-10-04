// Push-T: recorded MoRE-edited Diffusion Policy rollouts (pusher position and T-block pose per
// frame) replayed on a tabletop board. Data: the explorer's sim_task_replays.json, normalised
// workspace coordinates in [0, 1] (y up); scaled here to an 80 cm board.
import * as THREE from "three";
import { GOOD, modePaths, sample, lerpAngle } from "./r3d-common.js";

const S = 0.8; // metres per normalised unit
const BASE = 0.01; // board top
const T_HEIGHT = 0.03;
const toX = (x) => (x - 0.5) * S;
const toY = (y) => (y - 0.5) * S;

// T outline in normalised units around the block pose (same geometry as the top-down view):
// bar 0.234 x 0.0586 with its top edge at the pose, stem 0.0586 x 0.176 below it.
const T_OUTLINE = [
  [-0.117, 0], [0.117, 0], [0.117, -0.0586], [0.0293, -0.0586],
  [0.0293, -0.2343], [-0.0293, -0.2343], [-0.0293, -0.0586], [-0.117, -0.0586],
];
const tShape = (inset = 0) => {
  const s = new THREE.Shape();
  T_OUTLINE.forEach(([x, y], i) => {
    // Pull each vertex toward the shape's interior by `inset` (keeps bevelled blocks true to size).
    const px = (x - Math.sign(x) * inset) * S;
    const py = (y === 0 ? y - inset : y + inset) * S;
    if (i) s.lineTo(px, py);
    else s.moveTo(px, py);
  });
  s.closePath();
  return s;
};

export const STAGE = {
  cam: [0.5, 1.12, 1.2],
  target: [0, 0, 0.02],
  fog: [3.2, 6.5],
  shadow: { center: [0, 0, 0], half: 0.6, light: [0.7, 1.6, 0.5] },
  fit: 0.6,
  dist: [0.45, 3],
};

export const buildPushT = (task, colors) => {
  const group = new THREE.Group();

  // Board: a low slab, slightly darker than the studio floor.
  const board = new THREE.Mesh(
    new THREE.BoxGeometry(S + 0.04, S + 0.04, BASE),
    new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 1 })
  );
  board.position.z = BASE / 2;
  board.receiveShadow = true;
  group.add(board);

  // Goal pose: flat translucent T with a crisp outline.
  const g = task.scene.goal_t;
  const goal = new THREE.Group();
  goal.add(
    new THREE.Mesh(
      new THREE.ShapeGeometry(tShape()),
      new THREE.MeshBasicMaterial({ color: GOOD, transparent: true, opacity: 0.16, depthWrite: false })
    )
  );
  const edge = new THREE.LineLoop(
    new THREE.BufferGeometry().setFromPoints(tShape().getPoints()),
    new THREE.LineBasicMaterial({ color: GOOD, transparent: true, opacity: 0.85 })
  );
  goal.add(edge);
  goal.position.set(toX(g.center[0]), toY(g.center[1]), BASE + 0.0006);
  goal.rotation.z = THREE.MathUtils.degToRad(g.angle_deg);
  group.add(goal);

  // Pusher paths, per mode.
  const trajModes = task.modes.map((m) => ({ id: m.id, trajs: m.rollouts }));
  const modes = modePaths(trajModes, colors, (r) => r.points.map((p) => [toX(p[0]), toY(p[1])]), {
    radius: 0.0028, z: BASE + 0.0012, minStep: 0.002,
  });
  for (const m of modes) group.add(m.group);

  // T-block.
  const bevel = 0.003;
  const block = new THREE.Mesh(
    new THREE.ExtrudeGeometry(tShape(bevel / S), {
      depth: T_HEIGHT - 2 * bevel, bevelEnabled: true, bevelThickness: bevel, bevelSize: bevel, bevelSegments: 2,
    }),
    new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.5 })
  );
  block.geometry.translate(0, 0, bevel);
  block.castShadow = true;
  block.receiveShadow = true;
  const blockPose = new THREE.Group();
  blockPose.position.z = BASE;
  blockPose.add(block);
  group.add(blockPose);

  // Pusher: a puck with a short handle (the Push-T agent is a disc, r = 0.03 units).
  const r = task.scene.pusher_radius * S;
  const steel = new THREE.MeshStandardMaterial({ color: 0x8f98a3, roughness: 0.35, metalness: 0.4 });
  const pusher = new THREE.Group();
  const puck = new THREE.Mesh(
    new THREE.CylinderGeometry(r, r, 0.03, 32),
    new THREE.MeshStandardMaterial({ color: 0x2b3240, roughness: 0.45 })
  );
  puck.rotation.x = Math.PI / 2;
  puck.position.z = 0.015;
  const handle = new THREE.Mesh(new THREE.CylinderGeometry(0.007, 0.007, 0.12, 16), steel);
  handle.rotation.x = Math.PI / 2;
  handle.position.z = 0.09;
  const knob = new THREE.Mesh(new THREE.SphereGeometry(0.012, 20, 12), steel);
  knob.position.z = 0.152;
  for (const m of [puck, handle, knob]) m.castShadow = true;
  pusher.add(puck, handle, knob);
  pusher.position.z = BASE;
  group.add(pusher);

  let current = null;
  return {
    group,
    modes,
    stage: STAGE,
    setCurrent(traj, color) {
      current = traj;
      block.material.color.copy(color);
    },
    setTime(t) {
      if (!current) return;
      const p = sample(current.points, t);
      pusher.position.x = toX(p.a[0] + (p.b[0] - p.a[0]) * p.k);
      pusher.position.y = toY(p.a[1] + (p.b[1] - p.a[1]) * p.k);
      const q = sample(current.block, t);
      blockPose.position.x = toX(q.a[0] + (q.b[0] - q.a[0]) * q.k);
      blockPose.position.y = toY(q.a[1] + (q.b[1] - q.a[1]) * q.k);
      blockPose.rotation.z = lerpAngle(THREE.MathUtils.degToRad(q.a[2]), THREE.MathUtils.degToRad(q.b[2]), q.k);
    },
  };
};
