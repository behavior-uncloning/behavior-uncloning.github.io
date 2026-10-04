// MoRE vs. baselines (Table 1 of the paper, SR averaged over target modes).
// Small multiples, one per task; MoRE is the only colored series, the filtered-retrain
// reference is a dashed marker. Values appear on hover; the table below is the data view.
(() => {
  const host = document.getElementById("baseline-chart");
  if (!host) return;

  const METHODS = ["Original policy", "CPL", "DynaGuide", "MoRE"];
  const DATA = [
    { task: "Push-T", sr: [50.0, 90.3, 49.7, 98.3], ref: 100.0 },
    { task: "Push-Wall", sr: [37.0, 50.7, 70.0, 80.7], ref: 78.0 },
    { task: "Push-Pillars", sr: [21.5, 49.5, 34.3, 72.3], ref: 72.5 },
    { task: "Quadruped", sr: [50.0, 67.7, 50.3, 84.3], ref: 100.0 }
  ];

  const tip = document.createElement("div");
  tip.className = "bc-tip";
  tip.setAttribute("role", "status");
  host.appendChild(tip);

  const showTip = (target, html) => {
    tip.innerHTML = html;
    tip.style.opacity = "1";
    const hr = host.getBoundingClientRect();
    const tr = target.getBoundingClientRect();
    const x = Math.min(hr.width - tip.offsetWidth - 4, Math.max(4, tr.right - hr.left + 10));
    tip.style.transform = `translate(${x}px, ${tr.top - hr.top + tr.height / 2 - tip.offsetHeight / 2}px)`;
  };
  const hideTip = () => {
    tip.style.opacity = "0";
  };

  for (const d of DATA) {
    const panel = document.createElement("div");
    panel.className = "bc-panel";
    panel.innerHTML = `<h4>${d.task}</h4>`;
    const plot = document.createElement("div");
    plot.className = "bc-plot";
    const ref = document.createElement("div");
    ref.className = "bc-ref";
    ref.style.setProperty("--p", d.ref);
    ref.innerHTML = `<span>filtered retrain</span>`;
    ref.tabIndex = 0;
    const refTip = () => showTip(ref, `<b>Filtered-retrain reference</b>${d.ref.toFixed(1)}% SR · retrained from scratch on mode-filtered demos`);
    ref.addEventListener("mouseenter", refTip);
    ref.addEventListener("focus", refTip);
    ref.addEventListener("mouseleave", hideTip);
    ref.addEventListener("blur", hideTip);
    plot.appendChild(ref);

    METHODS.forEach((name, i) => {
      const row = document.createElement("div");
      row.className = "bc-row" + (name === "MoRE" ? " is-more" : "");
      row.tabIndex = 0;
      row.innerHTML = `<span class="bc-name">${name}</span><span class="bc-track"><i style="width:${d.sr[i]}%"></i></span>` +
        (name === "MoRE" ? `<b class="bc-val">${d.sr[i].toFixed(1)}</b>` : `<span class="bc-val"></span>`);
      const enter = () => showTip(row, `<b>${name} · ${d.task}</b>${d.sr[i].toFixed(1)}% SR (avg over target modes)`);
      row.addEventListener("mouseenter", enter);
      row.addEventListener("focus", enter);
      row.addEventListener("mouseleave", hideTip);
      row.addEventListener("blur", hideTip);
      plot.appendChild(row);
    });
    panel.appendChild(plot);
    host.appendChild(panel);
  }
})();
