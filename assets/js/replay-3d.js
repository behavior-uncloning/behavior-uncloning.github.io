// 3D replay of recorded ManiSkill3 demonstrations (Push-Wall, Push-Pillars) with the Franka arm.
// Driven by the mode explorer (assets/js/mode-explorer.js): it sets the task, the policy view, and
// the replay time. Data: assets/data/sim_3d_replays.json (tools/build_3d_replays.py).
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { MeshoptDecoder } from "three/addons/libs/meshopt_decoder.module.js";

const MESH_DIR = "assets/3d/panda/";
const BG = 0xfafbfc;
const WALL = 0x3a4252;
const GOOD = 0x3f8f5a;

// panda_v3.urdf kinematic chain: [link mesh, joint origin xyz, joint origin rpy, joint type].
const CHAIN = [
  ["link1", [0, 0, 0.333], [0, 0, 0], "revolute"],
  ["link2", [0, 0, 0], [-Math.PI / 2, 0, 0], "revolute"],
  ["link3", [0, -0.316, 0], [Math.PI / 2, 0, 0], "revolute"],
  ["link4", [0.0825, 0, 0], [Math.PI / 2, 0, 0], "revolute"],
  ["link5", [-0.0825, 0.384, 0], [-Math.PI / 2, 0, 0], "revolute"],
  ["link6", [0, 0, 0], [Math.PI / 2, 0, 0], "revolute"],
  ["link7", [0.088, 0, 0], [Math.PI / 2, 0, 0], "revolute"],
];

// URDF rpy is fixed-axis roll-pitch-yaw: R = Rz(yaw) Ry(pitch) Rx(roll) -> Euler order "ZYX".
const originNode = (xyz, rpy) => {
  const o = new THREE.Object3D();
  o.position.set(...xyz);
  o.rotation.set(rpy[0], rpy[1], rpy[2], "ZYX");
  return o;
};

const loader = new GLTFLoader().setMeshoptDecoder(MeshoptDecoder);
const meshCache = new Map();
const loadMesh = (name) => {
  if (!meshCache.has(name)) {
    meshCache.set(
      name,
      loader.loadAsync(`${MESH_DIR}${name}.glb`).then((g) => {
        g.scene.traverse((m) => {
          if (m.isMesh) {
            m.castShadow = true;
            m.receiveShadow = true;
          }
        });
        return g.scene;
      })
    );
  }
  return meshCache.get(name).then((s) => s.clone(true));
};

/** Franka arm: returns {root, setQpos(q[9])}. */
const buildRobot = async (rootPose) => {
  const root = new THREE.Group();
  root.position.set(rootPose[0], rootPose[1], rootPose[2]);
  root.quaternion.set(rootPose[4], rootPose[5], rootPose[6], rootPose[3]);
  root.add(await loadMesh("link0"));
  const joints = [];
  let parent = root;
  for (const [mesh, xyz, rpy] of CHAIN) {
    const origin = originNode(xyz, rpy);
    const motion = new THREE.Object3D();
    origin.add(motion);
    motion.add(await loadMesh(mesh));
    parent.add(origin);
    joints.push(motion);
    parent = motion;
  }
  const hand = originNode([0, 0, 0.107], [0, 0, -Math.PI / 4]);
  parent.add(hand);
  hand.add(await loadMesh("hand"));
  const fingers = [1, -1].map((dir) => {
    const base = originNode([0, 0, 0.0584], [0, 0, 0]);
    const slide = new THREE.Object3D();
    base.add(slide);
    const mesh = new THREE.Object3D();
    if (dir < 0) mesh.rotation.z = Math.PI;
    slide.add(mesh);
    loadMesh("finger").then((m) => mesh.add(m));
    hand.add(base);
    return { slide, dir };
  });
  return {
    root,
    setQpos: (q) => {
      for (let i = 0; i < 7; i++) joints[i].rotation.z = q[i];
      fingers[0].slide.position.y = q[7];
      fingers[1].slide.position.y = -q[8];
    },
  };
};

export class Replay3D {
  constructor(container, data, colors) {
    this.container = container;
    this.data = data;
    this.colors = colors;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setClearColor(BG);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.25;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.Fog(BG, 3.2, 6.5);
    // ManiSkill is z-up; three.js is y-up.
    this.world = new THREE.Group();
    this.world.rotation.x = -Math.PI / 2;
    this.scene.add(this.world);

    this.camera = new THREE.PerspectiveCamera(32, 16 / 10, 0.01, 20);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.maxPolarAngle = Math.PI * 0.47;
    this.controls.minDistance = 0.5;
    this.controls.maxDistance = 3.5;
    this.controls.addEventListener("change", () => (this.dirty = true));
    this.resetCamera();

    const hemi = new THREE.HemisphereLight(0xffffff, 0xd9dee6, 1.35);
    this.scene.add(hemi);
    const key = new THREE.DirectionalLight(0xffffff, 1.9);
    key.position.set(0.9, 1.6, 0.6);
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    key.shadow.radius = 6;
    key.shadow.bias = -0.0004;
    Object.assign(key.shadow.camera, { left: -1, right: 1, top: 1, bottom: -1, near: 0.1, far: 4 });
    this.scene.add(key);
    const fill = new THREE.DirectionalLight(0xffffff, 0.5);
    fill.position.set(-1, 0.8, -0.8);
    this.scene.add(fill);

    // Seamless studio floor: an unlit plane in the page colour plus a shadow-only layer.
    const floor = new THREE.Mesh(
      new THREE.PlaneGeometry(12, 12),
      new THREE.MeshBasicMaterial({ color: BG, toneMapped: false, fog: false })
    );
    this.world.add(floor);
    const shadow = new THREE.Mesh(new THREE.PlaneGeometry(12, 12), new THREE.ShadowMaterial({ opacity: 0.14 }));
    shadow.position.z = 0.0003;
    shadow.receiveShadow = true;
    this.world.add(shadow);

    this.taskGroup = new THREE.Group();
    this.world.add(this.taskGroup);
    this.ready = buildRobot(data.robot_root).then((robot) => {
      this.robot = robot;
      this.world.add(robot.root);
      this.dirty = true;
    });

    this.ro = new ResizeObserver(() => this.resize());
    this.ro.observe(container);
    this.resize();
    this.dirty = true;
  }

  resetCamera() {
    // three.js frame (y up): eye in front-right of the workspace, looking back toward the robot.
    this.camera.position.set(1.25, 1.2, 1.35);
    this.controls.target.set(-0.24, 0.3, 0.0);
    this.controls.update();
    this.dirty = true;
  }

  resize() {
    const w = this.container.clientWidth;
    const h = this.container.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.dirty = true;
  }

  setTask(id) {
    this.taskId = id;
    this.taskGroup.clear();
    const task = this.data.tasks[id];
    const st = task.static;
    const slate = new THREE.MeshStandardMaterial({ color: WALL, roughness: 0.6 });
    if (st.wall) {
      const [hx, hy, hz] = st.wall.half;
      const wall = new THREE.Mesh(new THREE.BoxGeometry(2 * hx, 2 * hy, 2 * hz), slate);
      wall.position.set(...st.wall.center);
      wall.castShadow = wall.receiveShadow = true;
      this.taskGroup.add(wall);
    }
    if (st.pillars) {
      for (const [x, y] of st.pillars) {
        const p = new THREE.Mesh(new THREE.CylinderGeometry(st.pillar_radius, st.pillar_radius, st.pillar_height, 20), slate);
        p.rotation.x = Math.PI / 2; // cylinder axis y -> z
        p.position.set(x, y, st.pillar_height / 2);
        p.castShadow = true;
        this.taskGroup.add(p);
      }
    }
    if (st.finish_line) {
      const line = new THREE.Mesh(
        new THREE.PlaneGeometry(0.006, 2 * st.finish_line.half_width),
        new THREE.MeshBasicMaterial({ color: GOOD, transparent: true, opacity: 0.85 })
      );
      line.position.set(st.finish_line.x, 0, 0.0008);
      this.taskGroup.add(line);
    }
    const goal = task.modes[0].trajs[0].goal;
    if (st.goal_radius && goal) {
      const disc = new THREE.Mesh(
        new THREE.CircleGeometry(st.goal_radius, 64),
        new THREE.MeshBasicMaterial({ color: GOOD, transparent: true, opacity: 0.12, depthWrite: false })
      );
      disc.position.set(goal[0], goal[1], 0.0006);
      this.taskGroup.add(disc);
      const ring = new THREE.Mesh(
        new THREE.RingGeometry(st.goal_radius - 0.003, st.goal_radius, 64),
        new THREE.MeshBasicMaterial({ color: GOOD, transparent: true, opacity: 0.7 })
      );
      ring.position.copy(disc.position);
      this.taskGroup.add(ring);
    }

    // Cube paths of every demo, per mode.
    this.modes = task.modes.map((m, mi) => {
      const color = new THREE.Color(this.colors[mi % this.colors.length]);
      const group = new THREE.Group();
      for (const tr of m.trajs) {
        // Drop near-duplicate consecutive samples (the cube rests at the start); repeated
        // control points make the Catmull-Rom tangents degenerate and the tube invisible.
        const pts = [];
        for (const c of tr.cube) {
          const v = new THREE.Vector3(c[0], c[1], 0.0012);
          if (!pts.length || v.distanceTo(pts[pts.length - 1]) > 0.002) pts.push(v);
        }
        if (pts.length < 2) continue;
        const geo = new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts), 160, 0.0024, 6, false);
        group.add(new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.7, depthWrite: false })));
      }
      this.taskGroup.add(group);
      return { id: m.id, group, color, trajs: m.trajs };
    });

    const half = this.data.cube_half;
    this.cube = new THREE.Mesh(
      new THREE.BoxGeometry(2 * half, 2 * half, 2 * half),
      new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.45 })
    );
    this.cube.castShadow = true;
    this.taskGroup.add(this.cube);
    this.dirty = true;
  }

  /** policy: "original" | "more"; target: mode id; cycle: replay index (advances each loop). */
  setView(policy, target, cycle) {
    if (!this.modes) return;
    const more = policy === "more";
    const visible = more ? this.modes.filter((m) => m.id === target) : this.modes;
    for (const m of this.modes) {
      const on = !more || m.id === target;
      for (const mesh of m.group.children) mesh.material.opacity = on ? 0.75 : 0.07;
    }
    // Original: alternate between modes; MoRE: only the target mode's demos.
    const mode = visible[cycle % visible.length];
    const k = Math.floor(cycle / visible.length) % mode.trajs.length;
    this.current = { traj: mode.trajs[k], color: mode.color };
    this.cube.material.color.copy(mode.color);
    this.dirty = true;
  }

  setTime(t) {
    if (!this.current || !this.robot) return;
    const tr = this.current.traj;
    const n = tr.qpos.length;
    const f = Math.min(n - 1, Math.max(0, t) * (n - 1));
    const i = Math.floor(f);
    const j = Math.min(n - 1, i + 1);
    const k = f - i;
    const q = tr.qpos[i].map((v, d) => v + (tr.qpos[j][d] - v) * k);
    this.robot.setQpos(q);
    const a = tr.cube[i];
    const b = tr.cube[j];
    this.cube.position.set(a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k, a[2] + (b[2] - a[2]) * k);
    this.cube.quaternion.set(a[4], a[5], a[6], a[3]);
    this.dirty = true;
  }

  render() {
    this.controls.update();
    if (!this.dirty) return;
    this.renderer.render(this.scene, this.camera);
    this.dirty = false;
  }
}

export const createReplay3D = async (container, colors) => {
  const data = await fetch("assets/data/sim_3d_replays.json").then((r) => r.json());
  const viewer = new Replay3D(container, data, colors);
  await viewer.ready;
  return viewer;
};
