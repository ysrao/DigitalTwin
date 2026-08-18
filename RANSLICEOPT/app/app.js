"use strict";

/* ---------------------------------------------------------------------
 * Band capability table (abstraction — see spec §12.2 Radio/Resource
 * Abstraction). Bandwidths in MHz, subcarrier spacing in kHz.
 * ------------------------------------------------------------------- */
const BAND_PARAMS = {
  FR1:    { bandwidths: [10, 20, 30, 40, 50, 60, 80, 100],        scs: [15, 30, 60] },
  FR2:    { bandwidths: [50, 100, 200, 400],                      scs: [60, 120] },
  FR3:    { bandwidths: [100, 200, 400, 800],                     scs: [120, 240] },
  SUBTHZ: { bandwidths: [400, 800, 1600, 2000, 3200],             scs: [480, 960] },
};

// 3GPP-style CQI -> spectral efficiency (bits/s/Hz), indices 1-15
const CQI_EFFICIENCY = [0.15, 0.23, 0.38, 0.60, 0.88, 1.18, 1.48, 1.91, 2.41, 2.73, 3.32, 3.90, 4.52, 5.12, 5.55];

const SLICE_META = {
  urllc: {
    label: "URLLC", cls: "urllc", color: "#e2452c", slaLabel: "Deadline (ms)", slaDefault: 1,
    extra: [
      { key: "maxActive", label: "Max active URLLC connections / cell", min: 0, step: 1 },
      { key: "packetSize", label: "Packet size (bytes)", min: 1, step: 1 },
    ],
  },
  embb: {
    label: "eMBB", cls: "embb", color: "#2f6fed", slaLabel: "Min throughput (Mbps)", slaDefault: 50,
    extra: [],
  },
  miot: {
    label: "mIoT/eMTC", cls: "miot", color: "#1a9d63", slaLabel: "Access success (%)", slaDefault: 99,
    extra: [
      { key: "maxActive", label: "Max active RedCap devices / cell", min: 0, step: 1 },
      { key: "minPrbPerDevice", label: "Min PRBs / RedCap device", min: 1, step: 1 },
    ],
  },
};

function defaultState() {
  return {
    cell: {
      name: "Cell-01",
      band: "FR3",
      bandwidth: 400,
      scs: 120,
      duplex: "TDD",
      tddRatio: 4,
      carriers: 2,
      mimo: 16,
      morphology: "dense-urban",
      radius: 200,
      ues: 150,
      cqi: 10,
    },
    slices: {
      urllc: { load: 15, sla: 1,  min: 15, target: 20, max: 35, borrow: true,  reclaim: "high",
                maxActive: 20, packetSize: 32 },
      embb:  { load: 60, sla: 50, min: 40, target: 55, max: 80, borrow: true,  reclaim: "medium" },
      miot:  { load: 25, sla: 99, min: 5,  target: 10, max: 20, borrow: true,  reclaim: "low",
                maxActive: 50, minPrbPerDevice: 2 },
    },
  };
}

let state = loadState() || defaultState();

/* ---------------------------------------------------------------------
 * Element refs
 * ------------------------------------------------------------------- */
const el = (id) => document.getElementById(id);
const cellFields = ["cellName", "band", "bandwidth", "scs", "duplex", "tddRatio", "carriers", "mimo", "morphology", "radius", "ues", "cqi"];

function init() {
  populateBandDependentSelects();
  bindCellFields();
  renderSliceCards();
  applyStateToInputs();
  wireGlobalButtons();
  recompute();
}

/* ---------------------------------------------------------------------
 * Band / bandwidth / SCS wiring
 * ------------------------------------------------------------------- */
function populateBandDependentSelects() {
  const band = state.cell.band;
  const params = BAND_PARAMS[band];

  const bwSel = el("bandwidth");
  bwSel.innerHTML = params.bandwidths.map(b => `<option value="${b}">${b} MHz</option>`).join("");
  if (!params.bandwidths.includes(state.cell.bandwidth)) state.cell.bandwidth = params.bandwidths[0];
  bwSel.value = state.cell.bandwidth;

  const scsSel = el("scs");
  scsSel.innerHTML = params.scs.map(s => `<option value="${s}">${s} kHz</option>`).join("");
  if (!params.scs.includes(state.cell.scs)) state.cell.scs = params.scs[0];
  scsSel.value = state.cell.scs;
}

function bindCellFields() {
  el("band").addEventListener("change", (e) => {
    state.cell.band = e.target.value;
    populateBandDependentSelects();
    recompute();
  });
  el("bandwidth").addEventListener("change", (e) => { state.cell.bandwidth = Number(e.target.value); recompute(); });
  el("scs").addEventListener("change", (e) => { state.cell.scs = Number(e.target.value); recompute(); });
  el("cellName").addEventListener("input", (e) => { state.cell.name = e.target.value; });
  el("duplex").addEventListener("change", (e) => {
    state.cell.duplex = e.target.value;
    el("tddRatioField").style.display = state.cell.duplex === "TDD" ? "" : "none";
    recompute();
  });
  el("morphology").addEventListener("change", (e) => { state.cell.morphology = e.target.value; recompute(); });

  bindRange("tddRatio", "tddRatioOut", (v) => { state.cell.tddRatio = v; return `${v}:1`; });
  bindRange("carriers", "carriersOut", (v) => { state.cell.carriers = v; return v; });
  bindRange("mimo", "mimoOut", (v) => { state.cell.mimo = v; return `${v}x`; });
  bindRange("radius", "radiusOut", (v) => { state.cell.radius = v; return v; });
  bindRange("ues", "uesOut", (v) => { state.cell.ues = v; return v; });
  bindRange("cqi", "cqiOut", (v) => { state.cell.cqi = v; return v; });
}

function bindRange(inputId, outId, apply) {
  const input = el(inputId);
  input.addEventListener("input", (e) => {
    const v = Number(e.target.value);
    el(outId).textContent = apply(v);
    recompute();
  });
}

/* ---------------------------------------------------------------------
 * Slice panel rendering
 * ------------------------------------------------------------------- */
function renderSliceCards() {
  const container = el("sliceList");
  container.innerHTML = "";

  Object.keys(SLICE_META).forEach((key) => {
    const meta = SLICE_META[key];
    const s = state.slices[key];

    const card = document.createElement("div");
    card.className = `slice-card ${meta.cls}`;
    card.innerHTML = `
      <div class="slice-head">
        <span class="slice-name">${meta.label}</span>
      </div>
      <div class="slice-fields">
        <label class="field">
          <span>Offered load (%)</span>
          <input type="number" min="0" max="100" data-k="load" value="${s.load}">
        </label>
        <label class="field">
          <span>${meta.slaLabel}</span>
          <input type="number" step="0.1" data-k="sla" value="${s.sla}">
        </label>
        ${meta.extra.map(f => `
        <label class="field">
          <span>${f.label}</span>
          <input type="number" min="${f.min}" step="${f.step}" data-k="${f.key}" value="${s[f.key]}">
        </label>`).join("")}
      </div>

      <div class="range-row">
        <div class="rr-label"><span>Minimum guarantee</span><b><span data-out="min">${s.min}</span>%</b></div>
        <input type="range" min="0" max="100" data-k="min" value="${s.min}">
      </div>
      <div class="range-row">
        <div class="rr-label"><span>Target share</span><b><span data-out="target">${s.target}</span>%</b></div>
        <input type="range" min="0" max="100" data-k="target" value="${s.target}">
      </div>
      <div class="range-row">
        <div class="rr-label"><span>Maximum cap</span><b><span data-out="max">${s.max}</span>%</b></div>
        <input type="range" min="0" max="100" data-k="max" value="${s.max}">
      </div>

      <div class="slice-toggles">
        <label><input type="checkbox" data-k="borrow" ${s.borrow ? "checked" : ""}> Borrowable when idle</label>
        <label>Reclaim priority
          <select data-k="reclaim">
            <option value="high" ${s.reclaim === "high" ? "selected" : ""}>High</option>
            <option value="medium" ${s.reclaim === "medium" ? "selected" : ""}>Medium</option>
            <option value="low" ${s.reclaim === "low" ? "selected" : ""}>Low</option>
          </select>
        </label>
      </div>
      <div class="slice-warn" data-out="warn"></div>
    `;

    container.appendChild(card);

    card.querySelectorAll("[data-k]").forEach((input) => {
      const k = input.dataset.k;
      const evt = input.type === "checkbox" ? "change" : (input.tagName === "SELECT" ? "change" : "input");
      input.addEventListener(evt, () => {
        let v = input.type === "checkbox" ? input.checked : input.value;
        if (input.type === "range" || input.type === "number") v = Number(v);
        s[k] = v;

        if (k === "min" || k === "target" || k === "max") {
          if (s.min > s.max) { if (k === "min") s.max = s.min; else s.min = s.max; }
          if (s.target < s.min) s.target = s.min;
          if (s.target > s.max) s.target = s.max;
          card.querySelector('[data-k="min"]').value = s.min;
          card.querySelector('[data-k="target"]').value = s.target;
          card.querySelector('[data-k="max"]').value = s.max;
          card.querySelector('[data-out="min"]').textContent = s.min;
          card.querySelector('[data-out="target"]').textContent = s.target;
          card.querySelector('[data-out="max"]').textContent = s.max;
        }
        recompute();
      });
    });
  });
}

/* ---------------------------------------------------------------------
 * Derived computation + validation
 * ------------------------------------------------------------------- */
function computePrbPool() {
  const { bandwidth, scs, carriers } = state.cell;
  const usableKHz = bandwidth * 1000 * 0.90; // ~10% guard-band abstraction
  const prbWidthKHz = scs * 12;
  const perCarrier = Math.floor(usableKHz / prbWidthKHz);
  return { perCarrier, total: perCarrier * carriers };
}

function computeSpectralEfficiencyRaw() {
  const base = CQI_EFFICIENCY[state.cell.cqi - 1];
  const mimoGain = Math.sqrt(state.cell.mimo); // diminishing-returns abstraction
  return base * mimoGain;
}

function computeSpectralEfficiency() {
  return computeSpectralEfficiencyRaw().toFixed(2);
}

/* ---------------------------------------------------------------------
 * Capacity guard — hard PRB floors for URLLC and RedCap/mIoT.
 *
 * URLLC: worst-case simultaneous arrival of `maxActive` connections must
 * each clear a `packetSize`-byte grant within the configured deadline.
 * TTI duration follows the NR slot-scaling identity TTI_ms = 15 / SCS_kHz,
 * so bits/PRB/TTI reduces to spectralEfficiency * 180 regardless of SCS —
 * kept as an explicit formula below rather than the constant, since band/
 * SCS selection still changes how many TTIs fit inside the deadline.
 *
 * RedCap/mIoT: each active reduced-capability device needs a minimum
 * per-TTI grant to stay connected; the floor is the concurrent count
 * times that minimum.
 * ------------------------------------------------------------------- */
function computeHardFloors(pool) {
  const { scs } = state.cell;
  const ttiMs = 15 / scs;
  const specEff = computeSpectralEfficiencyRaw();
  const prbBandwidthHz = scs * 1000 * 12;
  const bitsPerPrbPerTti = specEff * prbBandwidthHz * (ttiMs / 1000);

  const u = state.slices.urllc;
  const prbsPerConn = Math.max(1, Math.ceil((u.packetSize * 8) / bitsPerPrbPerTti));
  const ttisInDeadline = Math.max(1, Math.floor(u.sla / ttiMs));
  const urllcRequiredPrbs = Math.ceil((u.maxActive * prbsPerConn) / ttisInDeadline);

  const m = state.slices.miot;
  const miotRequiredPrbs = m.maxActive * m.minPrbPerDevice;

  const pct = (prbs) => (pool.total > 0 ? (prbs / pool.total) * 100 : 0);
  const configuredPrbs = (pctVal) => Math.round((pctVal / 100) * pool.total);

  return {
    ttiMs, prbsPerConn, ttisInDeadline,
    urllc: { requiredPrbs: urllcRequiredPrbs, floorPct: pct(urllcRequiredPrbs), configuredPrbs: configuredPrbs(u.min) },
    miot:  { requiredPrbs: miotRequiredPrbs,  floorPct: pct(miotRequiredPrbs),  configuredPrbs: configuredPrbs(m.min) },
  };
}

function applyStateToInputs() {
  el("cellName").value = state.cell.name;
  el("band").value = state.cell.band;
  el("duplex").value = state.cell.duplex;
  el("tddRatioField").style.display = state.cell.duplex === "TDD" ? "" : "none";
  el("tddRatio").value = state.cell.tddRatio;
  el("tddRatioOut").textContent = `${state.cell.tddRatio}:1`;
  el("carriers").value = state.cell.carriers;
  el("carriersOut").textContent = state.cell.carriers;
  el("mimo").value = state.cell.mimo;
  el("mimoOut").textContent = `${state.cell.mimo}x`;
  el("morphology").value = state.cell.morphology;
  el("radius").value = state.cell.radius;
  el("radiusOut").textContent = state.cell.radius;
  el("ues").value = state.cell.ues;
  el("uesOut").textContent = state.cell.ues;
  el("cqi").value = state.cell.cqi;
  el("cqiOut").textContent = state.cell.cqi;
}

function recompute() {
  const pool = computePrbPool();
  el("prbPerCarrier").textContent = pool.perCarrier.toLocaleString();
  el("prbTotal").textContent = pool.total.toLocaleString();
  el("specEff").textContent = `${computeSpectralEfficiency()} b/s/Hz`;

  renderShareChart(pool);
  renderCapacityGuard(pool);
  renderAllocTable(pool);
  renderValidation(pool);
  renderPerSliceWarnings();
}

function renderCapacityGuard(pool) {
  const floors = computeHardFloors(pool);

  const rows = [
    {
      key: "urllc", label: "URLLC", f: floors.urllc, configuredPct: state.slices.urllc.min,
      detail: `${state.slices.urllc.maxActive} active conn. × ${floors.prbsPerConn} PRB/conn, `
            + `${floors.ttisInDeadline} TTI(s) available within ${state.slices.urllc.sla} ms deadline `
            + `(TTI = ${floors.ttiMs.toFixed(3)} ms at ${state.cell.scs} kHz SCS).`,
    },
    {
      key: "miot", label: "RedCap / mIoT", f: floors.miot, configuredPct: state.slices.miot.min,
      detail: `${state.slices.miot.maxActive} active RedCap device(s) × ${state.slices.miot.minPrbPerDevice} min PRB/device.`,
    },
  ];

  el("capacityGuard").innerHTML = rows.map((r) => {
    const pass = r.f.requiredPrbs <= r.f.configuredPrbs;
    const floorPctCeil = Math.min(100, Math.ceil(r.f.floorPct));
    return `
      <div class="guard-item ${pass ? "guard-ok" : "guard-err"}">
        <div class="guard-head">
          <span>${pass ? "✓" : "✕"} ${r.label} hard floor</span>
          <span class="guard-pct">${r.f.requiredPrbs.toLocaleString()} PRB required (${r.f.floorPct.toFixed(1)}%) vs ${r.f.configuredPrbs.toLocaleString()} PRB configured (${r.configuredPct}%)</span>
        </div>
        <div class="guard-detail">${r.detail}</div>
        ${pass ? "" : `<button class="btn btn-fix" data-apply-slice="${r.key}" data-floor-pct="${floorPctCeil}">Apply required minimum (${floorPctCeil}%)</button>`}
      </div>`;
  }).join("");
}

function renderShareChart(pool) {
  const chart = el("shareChart");
  const legend = el("shareLegend");
  chart.innerHTML = "";
  legend.innerHTML = "";

  let sumTarget = 0;
  Object.keys(SLICE_META).forEach((key) => {
    const meta = SLICE_META[key];
    const s = state.slices[key];
    sumTarget += s.target;
    const seg = document.createElement("div");
    seg.className = "bar-seg";
    seg.style.width = `${s.target}%`;
    seg.style.background = meta.color;
    seg.title = `${meta.label}: ${s.target}%`;
    chart.appendChild(seg);
  });

  const remainder = Math.max(0, 100 - sumTarget);
  if (remainder > 0) {
    const seg = document.createElement("div");
    seg.className = "bar-seg";
    seg.style.width = `${remainder}%`;
    seg.style.background = "#dde2e8";
    seg.title = `Unallocated: ${remainder}%`;
    chart.appendChild(seg);
  }

  Object.keys(SLICE_META).forEach((key) => {
    const meta = SLICE_META[key];
    legend.innerHTML += `<span><span class="swatch" style="background:${meta.color}"></span>${meta.label} (${state.slices[key].target}%)</span>`;
  });
  if (remainder > 0) legend.innerHTML += `<span><span class="swatch" style="background:#dde2e8"></span>Unallocated (${remainder}%)</span>`;
}

function renderAllocTable(pool) {
  const tbody = document.querySelector("#allocTable tbody");
  tbody.innerHTML = "";
  Object.keys(SLICE_META).forEach((key) => {
    const meta = SLICE_META[key];
    const s = state.slices[key];
    const prbsAtTarget = Math.round((s.target / 100) * pool.total);
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${meta.label}</td><td>${s.min}%</td><td>${s.target}%</td><td>${s.max}%</td><td>${prbsAtTarget.toLocaleString()}</td>`;
    tbody.appendChild(tr);
  });
}

function renderValidation(pool) {
  const list = el("validationList");
  list.innerHTML = "";
  const sumMin = Object.values(state.slices).reduce((a, s) => a + s.min, 0);
  const sumTarget = Object.values(state.slices).reduce((a, s) => a + s.target, 0);

  const items = [];

  items.push(sumMin <= 100
    ? { level: "ok", text: `Sum of minimum guarantees is ${sumMin}% (≤ 100%).` }
    : { level: "err", text: `Sum of minimum guarantees is ${sumMin}% — exceeds 100% of PRB pool. Capacity conservation violated.` });

  items.push(sumTarget <= 100
    ? { level: "ok", text: `Sum of target shares is ${sumTarget}% (≤ 100%).` }
    : { level: "warn", text: `Sum of target shares is ${sumTarget}% — not all slices can operate at target simultaneously without borrowing.` });

  Object.keys(SLICE_META).forEach((key) => {
    const meta = SLICE_META[key];
    const s = state.slices[key];
    if (!(s.min <= s.target && s.target <= s.max)) {
      items.push({ level: "err", text: `${meta.label}: target must satisfy min ≤ target ≤ max.` });
    }
  });

  items.push(pool.total > 0
    ? { level: "ok", text: `Total PRB pool computed: ${pool.total.toLocaleString()} PRBs across ${state.cell.carriers} carrier(s).` }
    : { level: "err", text: "Computed PRB pool is zero for this bandwidth/numerology combination." });

  list.innerHTML = items.map(i => `<li class="v-${i.level}">${iconFor(i.level)} ${i.text}</li>`).join("");
}

function renderPerSliceWarnings() {
  document.querySelectorAll(".slice-card").forEach((card, idx) => {
    const key = Object.keys(SLICE_META)[idx];
    const s = state.slices[key];
    const warnEl = card.querySelector('[data-out="warn"]');
    if (!(s.min <= s.target && s.target <= s.max)) {
      warnEl.textContent = "Constraint violated: min ≤ target ≤ max required.";
    } else {
      warnEl.textContent = "";
    }
  });
}

function iconFor(level) {
  return level === "ok" ? "✓" : level === "warn" ? "⚠" : "✕";
}

/* ---------------------------------------------------------------------
 * Persistence / export
 * ------------------------------------------------------------------- */
const STORAGE_KEY = "ransliceopt_config_v1";

function loadState() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch { return null; }
}

function saveState() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
}

function wireGlobalButtons() {
  el("capacityGuard").addEventListener("click", (e) => {
    const btn = e.target.closest("[data-apply-slice]");
    if (!btn) return;
    const key = btn.dataset.applySlice;
    const pct = Math.min(100, Number(btn.dataset.floorPct));
    const s = state.slices[key];
    s.min = pct;
    if (s.target < s.min) s.target = s.min;
    if (s.max < s.min) s.max = s.min;
    renderSliceCards();
    recompute();
  });

  el("btn-save").addEventListener("click", () => {
    saveState();
    flashButton("btn-save", "Saved");
  });

  el("btn-reset").addEventListener("click", () => {
    if (!confirm("Reset configuration to defaults? This clears the saved local configuration.")) return;
    localStorage.removeItem(STORAGE_KEY);
    state = defaultState();
    populateBandDependentSelects();
    applyStateToInputs();
    renderSliceCards();
    recompute();
  });

  el("btn-export").addEventListener("click", () => {
    const payload = {
      configVersion: "1.0",
      generatedAt: new Date().toISOString(),
      cell: state.cell,
      derived: { prbPool: computePrbPool(), spectralEfficiency: computeSpectralEfficiency() },
      slices: state.slices,
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${state.cell.name || "ransliceopt-config"}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  });
}

function flashButton(id, text) {
  const btn = el(id);
  const orig = btn.textContent;
  btn.textContent = text;
  setTimeout(() => { btn.textContent = orig; }, 1200);
}

init();
