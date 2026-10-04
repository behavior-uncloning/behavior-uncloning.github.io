// Shared helpers for the 3D replay scenes (assets/js/replay-3d.js and the r3d-*.js task scenes).
// World frame everywhere is z-up, metres (ManiSkill / MuJoCo convention).
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { MeshoptDecoder } from "three/addons/libs/meshopt_decoder.module.js";

export const SLATE = 0x3a4252;
export const GOOD = 0x3f8f5a;

const loader = new GLTFLoader().setMeshoptDecoder(MeshoptDecoder);
const meshCache = new Map();

/** Load a GLB once and hand out clones; `material` (optional) replaces the exported material. */
export const loadMesh = (url, material) => {
  const key = url + (material ? `#${material.uuid}` : "");
  if (!meshCache.has(key)) {
    meshCache.set(
      key,
      loader.loadAsync(url).then((g) => {
        g.scene.traverse((m) => {
          if (!m.isMesh) return;
          m.castShadow = true;
          m.receiveShadow = true;
          if (material) m.material = material;
        });
        return g.scene;
      })
      .catch((e) => {
        meshCache.delete(key); // allow a retry on the next request
        throw e;
      })
    );
  }
  return meshCache.get(key).then((s) => s.clone(true));
};

/** Translucent tube along a 2D path; near-duplicate samples are dropped so Catmull-Rom tangents stay valid. */
export const pathTube = (xy, color, { radius, z, minStep }) => {
  const pts = [];
  for (const p of xy) {
    const v = new THREE.Vector3(p[0], p[1], z);
    if (!pts.length || v.distanceTo(pts[pts.length - 1]) > minStep) pts.push(v);
  }
  if (pts.length < 2) return null;
  const segs = Math.min(400, pts.length * 3);
  const geo = new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts), segs, radius, 6, false);
  return new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.7, depthWrite: false }));
};

/** Mode paths: one translucent group of tubes per behavior mode. */
export const modePaths = (modes, colors, toXY, opts) =>
  modes.map((m, mi) => {
    const color = new THREE.Color(colors[mi % colors.length]);
    const group = new THREE.Group();
    for (const tr of m.trajs) {
      const tube = pathTube(toXY(tr), color, opts);
      if (tube) group.add(tube);
    }
    return { id: m.id, group, color, trajs: m.trajs };
  });

/** Goal region: soft disc plus a crisp ring, lying on the floor. */
export const goalDisc = (x, y, r, ringWidth) => {
  const g = new THREE.Group();
  const disc = new THREE.Mesh(
    new THREE.CircleGeometry(r, 64),
    new THREE.MeshBasicMaterial({ color: GOOD, transparent: true, opacity: 0.12, depthWrite: false })
  );
  const ring = new THREE.Mesh(
    new THREE.RingGeometry(r - ringWidth, r, 64),
    new THREE.MeshBasicMaterial({ color: GOOD, transparent: true, opacity: 0.7 })
  );
  g.add(disc, ring);
  g.position.set(x, y, 0.0006);
  return g;
};

/** Linear interpolation into a per-frame array at normalised time t in [0, 1]. */
export const sample = (arr, t) => {
  const n = arr.length;
  const f = Math.min(n - 1, Math.max(0, t) * (n - 1));
  const i = Math.floor(f);
  return { a: arr[i], b: arr[Math.min(n - 1, i + 1)], k: f - i, i };
};

export const lerpAngle = (a, b, k) => {
  let d = b - a;
  while (d > Math.PI) d -= 2 * Math.PI;
  while (d < -Math.PI) d += 2 * Math.PI;
  return a + d * k;
};
