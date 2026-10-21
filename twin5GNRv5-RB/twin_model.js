/* twin_model.js - twin5GNR v5 link-budget and throughput model.
 * One source for the browser viewer and the Node cross-check; twin_model.py
 * mirrors it line for line. Everything is MODELLED, not measured.
 *
 * Per grid cell the ray tracer gives the downlink received power at a
 * reference 47 dBm boresight EIRP (3 sectors, TR 38.901 element, 4x4 array,
 * best sector, chosen tilt). This module turns that into per-device RSRP,
 * SINR and single-user / loaded throughput in both directions.
 *
 * Throughput terms come from the user's LTE link-budget workbook
 * (Qualcomm 80-W2559-5 Rev A): 36.942 CQI-efficiency curves (DL and UL),
 * the 2x2 SU-MIMO gain curve, the Vendor-B 4x4 gain curve, DL/UL overhead
 * tables, eNB NF 3 dB, target RoT 4 dB, PUSCH 1-RB requirement -0.7 dB,
 * PDCCH requirement -8.74 dB and the handset antenna fit capped at 2.14 dBi.
 * LTE-era terms applied to NR are a stated approximation.
 */
(function (root) {
  'use strict';
  const KT = -174.0;                 // dBm/Hz
  const P_COND_REF = 47.0 - 8.0 - 10 * Math.log10(16);   // 26.959 dBm
  const EIRP_REF = 47.0;
  const BS_NF = 3.0, ROT = 4.0;     // workbook UL
  const DL_REQ = -8.7356;           // workbook Tables!D46 (PDCCH, QPSK 1/10)
  const UL_REQ_1RB = -0.7;          // workbook PUSCH, 1 RB
  const TILTS = [0, 2, 4, 6, 8, 10];

  const BANDS = {
    f700:   { label: '0.7 GHz', nr: 'n28',     fghz: 0.7,   duplex: 'FDD', scs: 15,
              nrb: { 5: 25, 10: 52, 15: 79, 20: 106 }, bwdef: 10 },
    f2500:  { label: '2.5 GHz', nr: 'n41',     fghz: 2.5,   duplex: 'TDD', scs: 30,
              nrb: { 10: 24, 20: 51, 40: 106, 60: 162, 80: 217, 100: 273 }, bwdef: 60 },
    f3600:  { label: '3.6 GHz', nr: 'n48/n77', fghz: 3.6,   duplex: 'TDD', scs: 30,
              nrb: { 10: 24, 20: 51, 40: 106, 60: 162, 80: 217, 100: 273 }, bwdef: 40 },
    f7125:  { label: '7.125 GHz', nr: 'FR3',   fghz: 7.125, duplex: 'TDD', scs: 30,
              nrb: { 20: 51, 40: 106, 60: 162, 80: 217, 100: 273 }, bwdef: 100 },
    f28000: { label: '28 GHz', nr: 'n257',     fghz: 28.0,  duplex: 'TDD', scs: 120,
              nrb: { 50: 32, 100: 66, 200: 132, 400: 264 }, bwdef: 200 },
  };
  // TDD slot patterns; special slot 10 DL : 2 guard : 2 UL symbols
  const TDD = {
    DDDSU:      { dl: (3 + 10 / 14) / 5,  ul: (1 + 2 / 14) / 5 },
    DDDDDDDSUU: { dl: (7 + 10 / 14) / 10, ul: (2 + 2 / 14) / 10 },
    DDSUU:      { dl: (2 + 10 / 14) / 5,  ul: (2 + 2 / 14) / 5 },
  };

  // ---- device classes: mobility decides grid, losses and MIMO rank ----
  const ueAnt = f => Math.min(6.7292 * Math.log(f * 1000) - 51.61, 2.14);   // workbook fit
  const CLASSES = {
    cpe: { label: 'Rooftop CPE', mobility: 'fixed', grid: 'roof', cells: 'building',
           gain: b => (b.fghz >= 24 ? 20.0 : 9.0), ptx: b => (b.fghz >= 24 ? 23.0 : 26.0),
           nf: 7.0, body: 0.0, pen: () => 0.0, rx: b => (b.fghz >= 24 ? 2 : 4) },
    ped: { label: 'Outdoor handset', mobility: 'pedestrian, 3 km/h', grid: 'street', cells: 'outdoor',
           gain: b => ueAnt(b.fghz), ptx: () => 23.0, nf: 7.0, body: 0.0, pen: () => 0.0,
           rx: b => (b.fghz < 1.0 || b.fghz >= 24 ? 2 : 4) },
    indoor: { label: 'Indoor handset', mobility: 'nomadic', grid: 'street', cells: 'building',
           gain: b => ueAnt(b.fghz), ptx: () => 23.0, nf: 7.0, body: 0.0,
           pen: (b, o2i) => o2iLoss(b.fghz, o2i), rx: b => (b.fghz < 1.0 || b.fghz >= 24 ? 2 : 4) },
    veh: { label: 'In-vehicle handset', mobility: 'vehicular, 30-60 km/h', grid: 'street', cells: 'outdoor',
           gain: b => ueAnt(b.fghz), ptx: () => 23.0, nf: 7.0, body: 0.0, pen: () => 9.0,
           rx: b => Math.min(2, (b.fghz < 1.0 || b.fghz >= 24 ? 2 : 4)) },
  };

  // TR 38.901 O2I (same formulas as planner_v4.pl_tw) + mean indoor distance
  // loss 0.5 * E[min(U(0,25),U(0,25))] = 0.5 * 25/3 m
  function o2iLoss(f, model) {
    let lg, lc, wg, wc;
    if (model === 'high') { lg = 25.4 + 0.11 * f; lc = 5 + 4 * f; wg = 0.7; wc = 0.3; }
    else { lg = 2 + 0.2 * f; lc = 5 + 4 * f; wg = 0.3; wc = 0.7; }
    return 5 - 10 * Math.log10(wg * Math.pow(10, -lg / 10) + wc * Math.pow(10, -lc / 10)) + 0.5 * 25 / 3;
  }
  // Weissberger foliage loss on the LOS component, capped at 20 dB
  function foliageDb(f, d) {
    if (!(d > 0)) return 0;
    const k = Math.pow(f, 0.284);
    return Math.min(d < 14 ? 0.45 * k * d : 1.33 * k * Math.pow(d, 0.588), 20.0);
  }
  // workbook 36.942 efficiencies (bps/Hz per layer)
  const effDL = s => (s > 29.5 ? 5.6 : (s <= 5 ? 0.5326 * Math.exp(0.1713 * s) : Math.min(5.6, 0.182 * s + 0.3158)));
  const effUL = s => (s >= 46.8 ? 5.6 : (s <= 2.87 ? 0.3635 * Math.exp(0.1753 * s) : 0.1139 * s + 0.265));
  function mimoGain(s, layers) {
    if (layers <= 1) return 1;
    if (layers === 2) return s < 6.38 ? 1 : (s < 26.38 ? 0.0014 * s * s + 0.0063 * s + 0.9005 : 2);
    // 4 layers: workbook Vendor-B 4x4 curve
    if (s <= -5) return 0.9;
    if (s <= 16) return -0.0019 * s * s + 0.0736 * s + 1.3071;
    if (s <= 45) return -0.0017 * s * s + 0.1741 * s - 0.4252;
    return 3.97;
  }
  // workbook Tables: DL overhead, normal CP, 3 PDCCH symbols (15 MHz typo skipped)
  function ohDL(bw, ports) {
    const t = ports >= 4 ? [[5, 32.710], [10, 32.081], [1e9, 31.767]] : [[5, 30.386], [10, 29.729], [1e9, 29.400]];
    for (const [lim, v] of t) if (bw <= (lim === 10 ? 15 : lim)) return v / 100;
    return t[2][1] / 100;
  }
  const ohUL = bw => (bw <= 5 ? 22.119 : 22.341) / 100;

  /* evaluate one cell for one class and scenario.
     ref: DL received power at 47 dBm EIRP for the scenario's tilt (dBm, or null)
     phi: LOS share of power (0..1); depth: foliage depth on the direct path (m) */
  function cell(ref, phi, depth, cls, sc, common) {
    if (ref === null || ref === undefined || !isFinite(ref)) return null;
    const b = BANDS[sc.band], C = CLASSES[cls];
    const nrb = b.nrb[sc.bw];
    const bwOccHz = nrb * 12 * b.scs * 1e3;
    const veg = common.foliage ? foliageDb(b.fghz, depth) : 0;
    const vegAdj = veg > 0 ? 10 * Math.log10(1 - phi + phi * Math.pow(10, -veg / 10)) : 0;
    const losses = C.body + C.pen(b, common.o2i) - vegAdj;
    const g = C.gain(b);
    const pdl = ref + (sc.eirp - EIRP_REF) + g - losses;          // DL wideband received power
    const rsrp = pdl - 10 * Math.log10(12 * nrb);                 // per-RE RSRP
    const sinrDL = pdl - (KT + 10 * Math.log10(bwOccHz) + C.nf) - sc.iot;
    const pul = ref - P_COND_REF + C.ptx(b) + g - losses;         // UL received at the gNB
    const snrUL = pul - (KT + 10 * Math.log10(bwOccHz) + BS_NF + ROT);
    const share = b.duplex === 'FDD' ? { dl: 1, ul: 1 } : TDD[sc.tdd];
    const layers = Math.min(sc.ports, C.rx(b));
    const servedDL = sinrDL >= DL_REQ;
    const servedUL = snrUL + 10 * Math.log10(nrb) >= UL_REQ_1RB;
    const bwMHz = bwOccHz / 1e6;
    const dl1 = servedDL ? bwMHz * effDL(sinrDL) * mimoGain(sinrDL, layers) * (1 - ohDL(sc.bw, sc.ports)) * share.dl : 0;
    const ul1 = servedUL ? bwMHz * effUL(snrUL) * (1 - ohUL(sc.bw)) * share.ul : 0;
    return { rsrp, sinrDL, snrUL, servedDL, servedUL,
             dl: dl1 * (1 - sc.loadDL), ul: ul1 * (1 - sc.loadUL), dl1, ul1, veg };
  }

  /* aggregate one class over its cells inside the service radius */
  // indoor handsets: 'indoor' grid = power average over the building's facade
  // neighbourhood (all exposed sides, default); 'facade' = nearest open point
  const gridKey = (cls, common) => (cls === 'indoor' && common.indoorRef !== 'facade') ? 'indoor' : CLASSES[cls].grid;
  function classStats(grid, cls, sc, common) {
    const C = CLASSES[cls], g = grid[gridKey(cls, common)], ti = TILTS.indexOf(sc.tilt);
    let n = 0, sDL = 0, sUL = 0, tDL = 0, tUL = 0, cov = 0;
    const dls = [];
    for (let i = 0; i < grid.n2; i++) {
      if (grid.dist[i] > common.radius) continue;
      if ((C.cells === 'building') !== grid.bld[i]) continue;
      n++;
      const r = cell(g.ref[ti][i], g.phi[i], g.depth[i], cls, sc, common);
      if (!r) { dls.push(0); continue; }
      if (r.servedDL) sDL++;
      if (r.servedUL) sUL++;
      if (r.rsrp >= common.rsrpThr) cov++;
      tDL += r.dl; tUL += r.ul; dls.push(r.dl);
    }
    dls.sort((a, b) => a - b);
    const q = p => (dls.length ? dls[Math.min(dls.length - 1, Math.floor(p * dls.length))] : 0);
    return { n, servedDL: n ? sDL / n : 0, servedUL: n ? sUL / n : 0, cov: n ? cov / n : 0,
             dl: n ? tDL / n : 0, ul: n ? tUL / n : 0, dlMed: q(0.5), dlP10: q(0.1),
             ratio: tUL > 0 ? tDL / tUL : Infinity };
  }

  function blend(grid, sc, common) {
    const per = {}, out = { servedDL: 0, servedUL: 0, cov: 0, dl: 0, ul: 0 };
    let wsum = 0;
    for (const k of Object.keys(CLASSES)) {
      per[k] = classStats(grid, k, sc, common);
      const w = common.mix[k] || 0; wsum += w;
      for (const m of Object.keys(out)) out[m] += w * per[k][m];
    }
    for (const m of Object.keys(out)) out[m] = wsum ? out[m] / wsum : 0;
    out.ratio = out.ul > 0 ? out.dl / out.ul : Infinity;
    return { per, all: out };
  }

  const API = { BANDS, TDD, CLASSES, TILTS, gridKey, cell, classStats, blend, o2iLoss, foliageDb,
                effDL, effUL, mimoGain, ohDL, ohUL, ueAnt, DL_REQ, UL_REQ_1RB, P_COND_REF };
  if (typeof module !== 'undefined' && module.exports) module.exports = API;
  else root.TwinModel = API;
})(this);
