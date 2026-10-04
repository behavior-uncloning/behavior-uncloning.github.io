// Mode explorer: top-down SVG replay of recorded trajectories for the four simulated tasks.
// "Original policy" shows every behavior mode; "MoRE-edited" keeps the chosen target mode and
// fades the rest. Data: assets/data/sim_task_replays.json (see tools/build_sim_replays.py).
(() => {
  const root = document.getElementById("explorer");
  if (!root) return;

  const SVGNS = "http://www.w3.org/2000/svg";
  const W = 960;
  const H = 600;
  const PAD = 46;
  // Validated categorical order (fixed, never cycled); identity is also carried by text labels.
  const MODE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"];
  // Table 1 of the paper: Avg SR / TCR over target modes.
  const RESULTS = {
    pusht: { orig: 50.0, more: 98.3, tcr: 98.7 },
    pushwall: { orig: 37.0, more: 80.7, tcr: 80.7 },
    pushpillars: { orig: 21.5, more: 72.3, tcr: 80.7 },
    quadruped: { orig: 50.0, more: 84.3, tcr: 84.7 }
  };
  const ORDER = ["pusht", "pushwall", "pushpillars", "quadruped"];
  const SOURCE = {
    pusht: "Trajectories: MoRE-edited Diffusion Policy rollouts (pusher path and T-block pose), grouped by wrap direction.",
    pushwall: "Trajectories: successful mode-pure demonstrations of each route (top-down cube path).",
    pushpillars: "Trajectories: successful mode-pure demonstrations of the four routes (top-down cube path).",
    quadruped: "Trajectories: recorded Unitree Go1 base poses from the evaluation rollouts."
  };
  const SOURCE_3D = {
    pusht: "3D: recorded MoRE-edited Diffusion Policy rollouts replayed on the board (T-block pose and pusher position per frame). Original alternates between wrap directions; a MoRE edit replays only the target mode.",
    pushwall: "3D: recorded ManiSkill3 demonstrations replayed with the Franka arm (joint angles and cube pose per frame). Original alternates between modes; a MoRE edit replays only the target mode.",
    pushpillars: "3D: recorded ManiSkill3 demonstrations replayed with the Franka arm (joint angles and cube pose per frame). Original alternates between modes; a MoRE edit replays only the target mode.",
    quadruped: "3D: recorded Unitree Go1 base trajectories (position and heading per frame). Joint angles were not logged, so the leg motion is an illustrative trot. Original alternates between routes; a MoRE edit replays only the target route."
  };
  const CYCLE_MS = 6500;
  const HOLD_MS = 1400;

  const el = (tag, attrs = {}, parent) => {
    const node = document.createElementNS(SVGNS, tag);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
    if (parent) parent.appendChild(node);
    return node;
  };

  const svg = root.querySelector("#ex-svg");
  const taskSeg = root.querySelector("#ex-tasks");
  const policySeg = root.querySelector("#ex-policy");
  const modeBox = root.querySelector("#ex-modes");
  const srBox = root.querySelector("#ex-sr");
  const note = root.querySelector("#ex-note");
  const playBtn = root.querySelector("#ex-play");
  const scrub = root.querySelector("#ex-scrub");
  const modeHeading = root.querySelector("#ex-mode-heading");
  const stage = root.querySelector(".explorer-stage");
  const box3d = root.querySelector("#ex-3d");
  const viewSeg = root.querySelector("#ex-view");
  const hint3d = root.querySelector("#ex-3d-hint");
  const HAS_3D = new Set(["pusht", "pushwall", "pushpillars", "quadruped"]);

  const state = { data: null, task: "pushwall", policy: "original", target: null, t: 0, playing: true, visible: false, view: "3d", cycle: 0 };
  let viewer = null; // Replay3D instance (lazy)
  let viewerLoading = null;
  let scene = null; // per-task drawing handles
  let last = 0;
  let holdUntil = 0;

  // ---------- geometry ----------
  const deg = (rad) => (rad * 180) / Math.PI;
  const viewFor = (task) => {
    const [x0, x1] = task.scene.x_range;
    const [y0, y1] = task.scene.y_range;
    const s = Math.min((W - 2 * PAD) / (x1 - x0), (H - 2 * PAD) / (y1 - y0));
    const ox = (W - (x1 - x0) * s) / 2 - x0 * s;
    const oy = (H - (y1 - y0) * s) / 2 + y1 * s;
    return { s, ox, oy, x0, x1, y0, y1 };
  };

  // Push-T block: pose is the bar-bottom centre; bar 0.234×0.0586, stem 0.0586×0.176 (normalised units).
  const tBlock = (g, cls) => {
    const grp = el("g", { class: cls }, g);
    el("rect", { x: -0.117, y: -0.0586, width: 0.234, height: 0.0586, rx: 0.006 }, grp);
    el("rect", { x: -0.0293, y: -0.2343, width: 0.0586, height: 0.176, rx: 0.006 }, grp);
    return grp;
  };

  const modeColor = (task, modeId) => MODE_COLORS[task.modes.findIndex((m) => m.id === modeId) % MODE_COLORS.length];

  // ---------- build a task scene ----------
  const build = () => {
    const task = state.data.tasks[state.task];
    const v = viewFor(task);
    svg.replaceChildren();
    const world = el("g", { transform: `translate(${v.ox} ${v.oy}) scale(${v.s} ${-v.s})` }, svg);

    // Faint grid.
    const step = task.scene.tick_step;
    const grid = el("g", { class: "ex-grid" }, world);
    for (let x = Math.ceil(v.x0 / step) * step; x <= v.x1 + 1e-9; x += step) el("line", { x1: x, y1: v.y0, x2: x, y2: v.y1 }, grid);
    for (let y = Math.ceil(v.y0 / step) * step; y <= v.y1 + 1e-9; y += step) el("line", { x1: v.x0, y1: y, x2: v.x1, y2: y }, grid);

    // Static scene geometry.
    const sc = task.scene;
    const fixtures = el("g", { class: "ex-fixtures" }, world);
    let unit = "10 cm";
    let unitLen = 0.1;
    if (sc.kind === "pushwall") {
      const [cx, cy] = sc.wall.center;
      const [w, h] = sc.wall.size;
      el("circle", { cx: sc.goal.center[0], cy: sc.goal.center[1], r: sc.goal.radius, class: "ex-goal" }, fixtures);
      el("rect", { x: cx - w / 2, y: cy - h / 2, width: w, height: h, class: "ex-wall" }, fixtures);
    } else if (sc.kind === "pushpillars") {
      el("line", { x1: v.x0, y1: sc.finish_y, x2: v.x1, y2: sc.finish_y, class: "ex-finish" }, fixtures);
      for (const [px, py] of sc.pillars) el("circle", { cx: px, cy: py, r: Math.max(sc.pillar_radius, 6 / v.s), class: "ex-wall" }, fixtures);
    } else if (sc.kind === "pusht") {
      const goal = tBlock(fixtures, "ex-goal-t");
      goal.setAttribute("transform", `translate(${sc.goal_t.center[0]} ${sc.goal_t.center[1]}) rotate(${sc.goal_t.angle_deg})`);
      unit = "0.2 units";
      unitLen = 0.2;
    } else if (sc.kind === "quadruped") {
      const r0 = task.modes[0].rollouts[0];
      const [px, py] = r0.pillar_xy || [2.5, 0];
      const [gx, gy] = r0.goal_xy || [4.5, 0];
      el("circle", { cx: gx, cy: gy, r: Math.max(sc.goal_radius, 0.18), class: "ex-goal" }, fixtures);
      el("circle", { cx: px, cy: py, r: sc.pillar_radius, class: "ex-wall" }, fixtures);
      unit = "1 m";
      unitLen = 1;
    }

    // Trajectories + moving bodies, grouped per mode.
    const modes = task.modes.map((m, mi) => {
      const color = MODE_COLORS[mi % MODE_COLORS.length];
      const g = el("g", { class: "ex-mode", "data-mode": m.id, style: `--c:${color}` }, world);
      const rollouts = m.rollouts.map((r) => {
        const d = r.points.map((p, i) => `${i ? "L" : "M"}${p[0].toFixed(4)},${p[1].toFixed(4)}`).join("");
        const ghost = el("path", { d, class: "ex-ghost", "vector-effect": "non-scaling-stroke" }, g);
        const trail = el("path", { d, class: "ex-trail", "vector-effect": "non-scaling-stroke" }, g);
        let body;
        if (sc.kind === "pusht") {
          body = el("g", { class: "ex-body" }, g);
          tBlock(body, "ex-tblock");
          el("circle", { r: sc.pusher_radius, class: "ex-pusher" }, body);
        } else if (sc.kind === "quadruped") {
          body = el("g", { class: "ex-body" }, g);
          el("rect", { x: -0.22, y: -0.1, width: 0.44, height: 0.2, rx: 0.06, class: "ex-dog" }, body);
          el("circle", { cx: 0.25, cy: 0, r: 0.055, class: "ex-dog" }, body);
        } else {
          const c = sc.cube_size;
          body = el("rect", { x: -c / 2, y: -c / 2, width: c, height: c, rx: c * 0.12, class: "ex-cube" }, g);
        }
        return { r, trail, ghost, body, len: 0 };
      });
      return { m, g, color, rollouts };
    });

    // Path lengths for the drawn-on trail. Strokes are non-scaling, so dashes are in screen
    // pixels: convert the world-unit length with the view scale.
    for (const mode of modes) for (const ro of mode.rollouts) {
      ro.len = ro.trail.getTotalLength() * v.s + 2;
      ro.trail.style.strokeDasharray = `${ro.len} ${ro.len}`;
    }

    // Scale bar (screen space).
    const barPx = unitLen * v.s;
    const sb = el("g", { class: "ex-scalebar", transform: `translate(${W - PAD - barPx} ${H - 18})` }, svg);
    el("line", { x1: 0, y1: 0, x2: barPx, y2: 0 }, sb);
    el("line", { x1: 0, y1: -4, x2: 0, y2: 4 }, sb);
    el("line", { x1: barPx, y1: -4, x2: barPx, y2: 4 }, sb);
    const label = el("text", { x: barPx / 2, y: -8, "text-anchor": "middle" }, sb);
    label.textContent = unit;

    svg.setAttribute("aria-label", `${task.label}: top-down trajectories of ${task.modes.length} behavior modes`);
    scene = { task, sc, modes };
    if (!state.target || !task.modes.some((m) => m.id === state.target)) state.target = task.modes[0].id;
  };

  // ---------- 3D replay ----------
  const use3d = () => state.view === "3d" && HAS_3D.has(state.task);

  const ensureViewer = () => {
    if (viewer || viewerLoading) return viewerLoading;
    box3d.classList.add("is-loading");
    viewerLoading = import(new URL("assets/js/replay-3d.js", document.baseURI).href)
      .then((mod) => mod.createReplay3D(box3d, MODE_COLORS, state.data))
      .then((v) => {
        viewer = v;
        box3d.classList.remove("is-loading");
        sync3d(true);
      })
      .catch((e) => {
        // WebGL or network unavailable: fall back to the top-down view.
        console.warn("3D replay disabled:", e);
        state.view = "2d";
        box3d.classList.remove("is-loading");
        applyStage();
      });
    return viewerLoading;
  };

  const sync3d = (taskChanged) => {
    if (!viewer || !use3d()) return;
    if (taskChanged || viewer.taskId !== state.task) viewer.setTask(state.task);
    viewer.setView(state.policy, state.target, state.cycle);
    viewer.setTime(state.t);
    viewer.render();
  };

  const applyStage = () => {
    const has = HAS_3D.has(state.task);
    viewSeg.hidden = !has;
    const on = use3d();
    stage.classList.toggle("is-3d", on);
    box3d.hidden = !on;
    hint3d.hidden = !on;
    for (const b of viewSeg.querySelectorAll("button")) b.setAttribute("aria-pressed", String(b.dataset.view === state.view));
    if (on) {
      ensureViewer();
      sync3d(true);
    }
  };

  // ---------- per-frame update ----------
  const poseAt = (r, t) => {
    const n = r.points.length;
    const f = Math.min(n - 1, t * (n - 1));
    const i = Math.floor(f);
    const a = r.points[i];
    const b = r.points[Math.min(n - 1, i + 1)];
    const k = f - i;
    return [a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k, a[2] + (b[2] - a[2]) * k, i];
  };

  const frame = () => {
    if (!scene) return;
    const { sc, modes } = scene;
    // Rollouts have different lengths; scale each to the longest so they move in real time.
    const maxN = Math.max(...modes.flatMap((m) => m.rollouts.map((ro) => ro.r.points.length)));
    for (const mode of modes) for (const ro of mode.rollouts) {
      const n = ro.r.points.length;
      const t = Math.min(1, (state.t * (maxN - 1)) / (n - 1));
      ro.trail.style.strokeDashoffset = `${ro.len * (1 - t)}`;
      const [x, y, a, i] = poseAt(ro.r, t);
      if (sc.kind === "pusht") {
        const bp = ro.r.block[Math.min(ro.r.block.length - 1, i)];
        ro.body.children[0].setAttribute("transform", `translate(${bp[0]} ${bp[1]}) rotate(${bp[2]})`);
        ro.body.children[1].setAttribute("cx", x);
        ro.body.children[1].setAttribute("cy", y);
      } else {
        ro.body.setAttribute("transform", `translate(${x} ${y}) rotate(${deg(a)})`);
      }
    }
    scrub.value = Math.round(state.t * 1000);
    if (viewer && use3d()) {
      viewer.setTime(state.t);
      viewer.render();
    }
  };

  // ---------- view state (policy / target) ----------
  const applyView = () => {
    if (!scene) return;
    const more = state.policy === "more";
    for (const mode of scene.modes) {
      mode.g.classList.toggle("is-faded", more && mode.m.id !== state.target);
      mode.g.classList.toggle("is-target", more && mode.m.id === state.target);
    }
    for (const b of policySeg.querySelectorAll("button")) b.setAttribute("aria-pressed", String(b.dataset.policy === state.policy));
    for (const b of modeBox.querySelectorAll("button")) {
      const on = b.dataset.mode === state.target;
      b.setAttribute("aria-pressed", String(on));
      b.classList.toggle("is-on", on);
    }
    modeBox.classList.toggle("is-original", !more);
    modeHeading.textContent = more ? "Target mode (MoRE edit)" : "Behavior modes in the data";
    const res = RESULTS[state.task];
    srBox.querySelector('[data-k="orig"] .ex-bar i').style.width = `${res.orig}%`;
    srBox.querySelector('[data-k="more"] .ex-bar i').style.width = `${res.more}%`;
    srBox.querySelector('[data-k="orig"] b').textContent = `${res.orig.toFixed(1)}%`;
    srBox.querySelector('[data-k="more"] b').textContent = `${res.more.toFixed(1)}%`;
    srBox.querySelector('[data-k="orig"]').classList.toggle("is-on", !more);
    srBox.querySelector('[data-k="more"]').classList.toggle("is-on", more);
    note.textContent = use3d() ? SOURCE_3D[state.task] : SOURCE[state.task];
    if (viewer && use3d()) viewer.setView(state.policy, state.target, state.cycle);
  };

  const renderModeButtons = () => {
    const task = state.data.tasks[state.task];
    modeBox.replaceChildren();
    task.modes.forEach((m, mi) => {
      const b = document.createElement("button");
      b.type = "button";
      b.dataset.mode = m.id;
      b.style.setProperty("--c", MODE_COLORS[mi % MODE_COLORS.length]);
      b.innerHTML = `<span class="sw"></span>${m.label}`;
      b.addEventListener("click", () => {
        state.target = m.id;
        state.policy = "more";
        restart();
      });
      modeBox.appendChild(b);
    });
  };

  const selectTask = (id) => {
    state.task = id;
    state.target = null;
    for (const b of taskSeg.querySelectorAll("button")) b.setAttribute("aria-pressed", String(b.dataset.task === id));
    build();
    renderModeButtons();
    applyStage();
    restart();
  };

  const restart = () => {
    state.t = 0;
    state.cycle = 0;
    holdUntil = 0;
    applyView();
    frame();
  };

  // ---------- loop ----------
  const tick = (now) => {
    const dt = last ? now - last : 0;
    last = now;
    if (state.playing && state.visible && !document.hidden) {
      if (state.t >= 1) {
        if (!holdUntil) holdUntil = now + HOLD_MS;
        else if (now >= holdUntil) {
          state.t = 0;
          state.cycle += 1;
          holdUntil = 0;
          if (viewer && use3d()) viewer.setView(state.policy, state.target, state.cycle);
        }
      } else {
        state.t = Math.min(1, state.t + dt / CYCLE_MS);
      }
      frame();
    } else if (viewer && use3d() && state.visible) {
      viewer.render(); // keep orbit damping smooth while paused
    }
    requestAnimationFrame(tick);
  };

  const setPlaying = (p) => {
    state.playing = p;
    playBtn.setAttribute("aria-label", p ? "Pause replay" : "Play replay");
    playBtn.classList.toggle("is-paused", !p);
  };

  // ---------- wire up ----------
  const init = (data) => {
    state.data = data;
    for (const id of ORDER) {
      const b = document.createElement("button");
      b.type = "button";
      b.dataset.task = id;
      b.textContent = data.tasks[id].label;
      b.addEventListener("click", () => selectTask(id));
      taskSeg.appendChild(b);
    }
    for (const b of policySeg.querySelectorAll("button")) {
      b.addEventListener("click", () => {
        state.policy = b.dataset.policy;
        restart();
      });
    }
    playBtn.addEventListener("click", () => setPlaying(!state.playing));
    for (const b of viewSeg.querySelectorAll("button")) {
      b.addEventListener("click", () => {
        state.view = b.dataset.view;
        applyStage();
        applyView();
      });
    }
    scrub.addEventListener("input", () => {
      setPlaying(false);
      state.t = Number(scrub.value) / 1000;
      frame();
    });
    new IntersectionObserver((entries) => {
      state.visible = entries.some((e) => e.isIntersecting);
    }).observe(root);
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) setPlaying(false);
    // Deep links, e.g. ?task=pushpillars&policy=more&target=left_gap
    const q = new URLSearchParams(location.search);
    if (ORDER.includes(q.get("task"))) state.task = q.get("task");
    if (q.get("view") === "2d") state.view = "2d";
    selectTask(state.task);
    if (q.get("policy") === "more") {
      state.policy = "more";
      if (q.get("target") && data.tasks[state.task].modes.some((m) => m.id === q.get("target"))) state.target = q.get("target");
      restart();
    }
    requestAnimationFrame(tick);
  };

  const load = () =>
    fetch("assets/data/sim_task_replays.json")
      .then((r) => r.json())
      .then(init)
      .catch(() => {
        note.textContent = "Trajectory data could not be loaded.";
      });
  // Defer the data and the 3D scene until the explorer is about to scroll into view.
  if ("IntersectionObserver" in window && !location.search.includes("task=")) {
    const io = new IntersectionObserver((entries) => {
      if (entries.some((e) => e.isIntersecting)) {
        io.disconnect();
        load();
      }
    }, { rootMargin: "800px 0px" });
    io.observe(root);
  } else {
    load();
  }
})();
