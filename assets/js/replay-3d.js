// 3D replay of the four simulated tasks, driven by the mode explorer (assets/js/mode-explorer.js):
// it sets the task, the policy view, and the replay time. Each task scene lives in its own module:
//   Push-Wall / Push-Pillars -> r3d-panda.js (Franka arm, sim_3d_replays.json)
//   Push-T                   -> r3d-pusht.js (T-block + pusher, sim_task_replays.json)
//   Quadruped                -> r3d-go1.js   (Unitree Go1, sim_task_replays.json)
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { buildManiSkill } from "./r3d-panda.js";
import { buildPushT } from "./r3d-pusht.js";
import { buildQuadruped } from "./r3d-go1.js";

const BG = 0xfafbfc;
const DESIGN_ASPECT = 1.6; // stage cameras are framed for a 16:10 viewport

export class Replay3D {
  /** data3d: sim_3d_replays.json; replays: the explorer's sim_task_replays.json. */
  constructor(container, data3d, replays, colors) {
    this.container = container;
    this.data3d = data3d;
    this.replays = replays;
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
    // Task data is z-up; three.js is y-up.
    this.world = new THREE.Group();
    this.world.rotation.x = -Math.PI / 2;
    this.scene.add(this.world);

    this.camera = new THREE.PerspectiveCamera(32, 16 / 10, 0.01, 40);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.maxPolarAngle = Math.PI * 0.47;
    this.controls.addEventListener("change", () => (this.dirty = true));
    this.controls.addEventListener("start", () => (this.userMoved = true));

    this.scene.add(new THREE.HemisphereLight(0xffffff, 0xd9dee6, 1.35));
    this.key = new THREE.DirectionalLight(0xffffff, 1.9);
    this.key.castShadow = true;
    this.key.shadow.mapSize.set(2048, 2048);
    this.key.shadow.radius = 6;
    this.key.shadow.bias = -0.0004;
    this.scene.add(this.key, this.key.target);
    const fill = new THREE.DirectionalLight(0xffffff, 0.5);
    fill.position.set(-1, 0.8, -0.8);
    this.scene.add(fill);

    // Seamless studio floor: an unlit plane in the page colour plus a shadow-only layer.
    const floor = new THREE.Mesh(
      new THREE.PlaneGeometry(40, 40),
      new THREE.MeshBasicMaterial({ color: BG, toneMapped: false, fog: false })
    );
    this.world.add(floor);
    const shadow = new THREE.Mesh(new THREE.PlaneGeometry(40, 40), new THREE.ShadowMaterial({ opacity: 0.14 }));
    shadow.position.z = 0.0003;
    shadow.receiveShadow = true;
    this.world.add(shadow);

    this.taskGroup = new THREE.Group();
    this.world.add(this.taskGroup);

    this.ro = new ResizeObserver(() => this.resize());
    this.ro.observe(container);
    this.resize();
    this.dirty = true;
  }

  /** Camera, fog, distance limits and shadow frustum for the current task (three.js y-up frame). */
  applyStage(stage) {
    this.stage = stage;
    this.controls.minDistance = stage.dist[0];
    this.controls.maxDistance = stage.dist[1] * 1.8;
    const { center, half, light } = stage.shadow;
    const c = new THREE.Vector3(...center);
    const dir = new THREE.Vector3(...light).normalize();
    this.key.position.copy(c).addScaledVector(dir, 2 * half + 1);
    this.key.target.position.copy(c);
    Object.assign(this.key.shadow.camera, { left: -half, right: half, top: half, bottom: -half, near: 0.1, far: 4 * half + 3 });
    this.key.shadow.camera.updateProjectionMatrix();
    this.resetCamera();
  }

  resetCamera() {
    if (!this.stage) return;
    // Narrow viewports (phones) pull the camera back so the whole workspace stays in frame.
    // stage.fit (0..1) sets how strongly: 1 keeps the full width, smaller values crop the sides.
    const fit = Math.min(1.8, Math.max(1, DESIGN_ASPECT / this.camera.aspect) ** this.stage.fit);
    const target = new THREE.Vector3(...this.stage.target);
    this.camera.position.set(...this.stage.cam).sub(target).multiplyScalar(fit).add(target);
    this.controls.target.copy(target);
    this.scene.fog.near = this.stage.fog[0] * fit;
    this.scene.fog.far = this.stage.fog[1] * fit;
    this.userMoved = false;
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
    if (!this.userMoved) this.resetCamera();
    this.dirty = true;
  }

  buildScene(id) {
    const markDirty = () => (this.dirty = true);
    if (id === "pusht") return buildPushT(this.replays.tasks.pusht, this.colors);
    if (id === "quadruped") return buildQuadruped(this.replays.tasks.quadruped, this.colors, markDirty);
    return buildManiSkill(this.data3d, id, this.colors, markDirty);
  }

  /** Scenes are built once per task and reused (no GPU buffers are re-created on switches). */
  setTask(id) {
    if (id === this.taskId) return;
    this.taskId = id;
    this.scenes ??= {};
    this.task = this.scenes[id] ??= this.buildScene(id);
    this.taskGroup.clear();
    this.taskGroup.add(this.task.group);
    this.task.activate?.();
    this.applyStage(this.task.stage);
  }

  /** policy: "original" | "more"; target: mode id; cycle: replay index (advances each loop). */
  setView(policy, target, cycle) {
    if (!this.task) return;
    const modes = this.task.modes;
    const more = policy === "more";
    const visible = more ? modes.filter((m) => m.id === target) : modes;
    for (const m of modes) {
      const on = !more || m.id === target;
      for (const mesh of m.group.children) mesh.material.opacity = on ? 0.75 : 0.07;
    }
    // Original: alternate between modes; MoRE: only the target mode's rollouts.
    const mode = visible[cycle % visible.length];
    const k = Math.floor(cycle / visible.length) % mode.trajs.length;
    this.task.setCurrent(mode.trajs[k], mode.color);
    this.dirty = true;
  }

  setTime(t) {
    if (!this.task) return;
    this.task.setTime(t);
    this.dirty = true;
  }

  render() {
    this.controls.update();
    if (!this.dirty) return;
    this.renderer.render(this.scene, this.camera);
    this.dirty = false;
  }
}

export const createReplay3D = async (container, colors, replays) => {
  const data3d = await fetch(new URL("../data/sim_3d_replays.json", import.meta.url)).then((r) => r.json());
  return new Replay3D(container, data3d, replays, colors);
};
