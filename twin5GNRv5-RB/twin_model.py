#!/usr/bin/env python3
"""twin_model.py - Python mirror of twin_model.js (same formulas, same order).

Also loads a site's v5 grids and prints a before/after upgrade report:

    python twin_model.py --site nw \
        --before band=f700,bw=10,eirp=47,tilt=4,ports=2 \
        --after  band=f3600,bw=40,eirp=47,tilt=4,ports=4,tdd=DDDSU \
        --mix cpe=30,indoor=40,ped=20,veg=10 --radius 1000

Scenario keys: band, bw (MHz), eirp (dBm total), tilt (0-10 step 2), ports
(2|4), tdd (DDDSU|DDDDDDDSUU|DDSUU), iot (dB), loaddl, loadul (0-1).
Everything is MODELLED, not measured. See twin_model.js for the sources.
"""
import argparse, json, math, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
KT = -174.0
P_COND_REF = 47.0 - 8.0 - 10 * math.log10(16)
EIRP_REF = 47.0
BS_NF, ROT = 3.0, 4.0
DL_REQ = -8.7356
UL_REQ_1RB = -0.7
TILTS = [0, 2, 4, 6, 8, 10]
BANDS = {
    'f700':   dict(label='0.7 GHz', fghz=0.7, duplex='FDD', scs=15, nrb={5: 25, 10: 52, 15: 79, 20: 106}, bwdef=10),
    'f2500':  dict(label='2.5 GHz', fghz=2.5, duplex='TDD', scs=30, nrb={10: 24, 20: 51, 40: 106, 60: 162, 80: 217, 100: 273}, bwdef=60),
    'f3600':  dict(label='3.6 GHz', fghz=3.6, duplex='TDD', scs=30, nrb={10: 24, 20: 51, 40: 106, 60: 162, 80: 217, 100: 273}, bwdef=40),
    'f7125':  dict(label='7.125 GHz', fghz=7.125, duplex='TDD', scs=30, nrb={20: 51, 40: 106, 60: 162, 80: 217, 100: 273}, bwdef=100),
    'f28000': dict(label='28 GHz', fghz=28.0, duplex='TDD', scs=120, nrb={50: 32, 100: 66, 200: 132, 400: 264}, bwdef=200),
}
TDD = {'DDDSU': ((3 + 10 / 14) / 5, (1 + 2 / 14) / 5),
       'DDDDDDDSUU': ((7 + 10 / 14) / 10, (2 + 2 / 14) / 10),
       'DDSUU': ((2 + 10 / 14) / 5, (2 + 2 / 14) / 5)}


def ue_ant(f):
    return min(6.7292 * math.log(f * 1000) - 51.61, 2.14)


def o2i_loss(f, model):
    if model == 'high':
        lg, lc, wg, wc = 25.4 + 0.11 * f, 5 + 4 * f, 0.7, 0.3
    else:
        lg, lc, wg, wc = 2 + 0.2 * f, 5 + 4 * f, 0.3, 0.7
    return 5 - 10 * math.log10(wg * 10 ** (-lg / 10) + wc * 10 ** (-lc / 10)) + 0.5 * 25 / 3


def rx_layers(f):
    return 2 if (f < 1.0 or f >= 24) else 4


CLASSES = {
    'cpe':    dict(label='Rooftop CPE', grid='roof', cells='building',
                   gain=lambda b: 20.0 if b['fghz'] >= 24 else 9.0,
                   ptx=lambda b: 23.0 if b['fghz'] >= 24 else 26.0,
                   nf=7.0, body=0.0, pen=lambda b, o: 0.0, rx=lambda b: 2 if b['fghz'] >= 24 else 4),
    'ped':    dict(label='Outdoor handset', grid='street', cells='outdoor', gain=lambda b: ue_ant(b['fghz']),
                   ptx=lambda b: 23.0, nf=7.0, body=0.0, pen=lambda b, o: 0.0, rx=lambda b: rx_layers(b['fghz'])),
    'indoor': dict(label='Indoor handset', grid='street', cells='building', gain=lambda b: ue_ant(b['fghz']),
                   ptx=lambda b: 23.0, nf=7.0, body=0.0, pen=lambda b, o: o2i_loss(b['fghz'], o),
                   rx=lambda b: rx_layers(b['fghz'])),
    'veh':    dict(label='In-vehicle handset', grid='street', cells='outdoor', gain=lambda b: ue_ant(b['fghz']),
                   ptx=lambda b: 23.0, nf=7.0, body=0.0, pen=lambda b, o: 9.0,
                   rx=lambda b: min(2, rx_layers(b['fghz']))),
}


def foliage_db(f, d):
    if not d > 0:
        return 0.0
    k = f ** 0.284
    return min(0.45 * k * d if d < 14 else 1.33 * k * d ** 0.588, 20.0)


def eff_dl(s):
    return 5.6 if s > 29.5 else (0.5326 * math.exp(0.1713 * s) if s <= 5 else min(5.6, 0.182 * s + 0.3158))


def eff_ul(s):
    return 5.6 if s >= 46.8 else (0.3635 * math.exp(0.1753 * s) if s <= 2.87 else 0.1139 * s + 0.265)


def mimo_gain(s, layers):
    if layers <= 1:
        return 1.0
    if layers == 2:
        return 1.0 if s < 6.38 else (0.0014 * s * s + 0.0063 * s + 0.9005 if s < 26.38 else 2.0)
    if s <= -5:
        return 0.9
    if s <= 16:
        return -0.0019 * s * s + 0.0736 * s + 1.3071
    if s <= 45:
        return -0.0017 * s * s + 0.1741 * s - 0.4252
    return 3.97


def oh_dl(bw, ports):
    t = [(5, 32.710), (15, 32.081), (1e9, 31.767)] if ports >= 4 else [(5, 30.386), (15, 29.729), (1e9, 29.400)]
    for lim, v in t:
        if bw <= lim:
            return v / 100


def oh_ul(bw):
    return (22.119 if bw <= 5 else 22.341) / 100


def cell(ref, phi, depth, cls, sc, common):
    if ref is None or not math.isfinite(ref):
        return None
    b, C = BANDS[sc['band']], CLASSES[cls]
    nrb = b['nrb'][sc['bw']]
    bw_occ = nrb * 12 * b['scs'] * 1e3
    veg = foliage_db(b['fghz'], depth) if common['foliage'] else 0.0
    veg_adj = 10 * math.log10(1 - phi + phi * 10 ** (-veg / 10)) if veg > 0 else 0.0
    losses = C['body'] + C['pen'](b, common['o2i']) - veg_adj
    g = C['gain'](b)
    pdl = ref + (sc['eirp'] - EIRP_REF) + g - losses
    rsrp = pdl - 10 * math.log10(12 * nrb)
    sinr_dl = pdl - (KT + 10 * math.log10(bw_occ) + C['nf']) - sc['iot']
    pul = ref - P_COND_REF + C['ptx'](b) + g - losses
    snr_ul = pul - (KT + 10 * math.log10(bw_occ) + BS_NF + ROT)
    sdl, sul = (1.0, 1.0) if b['duplex'] == 'FDD' else TDD[sc['tdd']]
    layers = min(sc['ports'], C['rx'](b))
    served_dl = sinr_dl >= DL_REQ
    served_ul = snr_ul + 10 * math.log10(nrb) >= UL_REQ_1RB
    mhz = bw_occ / 1e6
    dl1 = mhz * eff_dl(sinr_dl) * mimo_gain(sinr_dl, layers) * (1 - oh_dl(sc['bw'], sc['ports'])) * sdl if served_dl else 0.0
    ul1 = mhz * eff_ul(snr_ul) * (1 - oh_ul(sc['bw'])) * sul if served_ul else 0.0
    return dict(rsrp=rsrp, sinrDL=sinr_dl, snrUL=snr_ul, servedDL=served_dl, servedUL=served_ul,
                dl=dl1 * (1 - sc['loadDL']), ul=ul1 * (1 - sc['loadUL']), dl1=dl1, ul1=ul1, veg=veg)


def grid_key(cls, common):
    """Indoor handsets: 'indoor' = power average over the building's facade
    neighbourhood (signal entering through all exposed sides, default);
    'facade' option = the single nearest open point (conservative)."""
    if cls == 'indoor' and common.get('indoorRef', 'sides') != 'facade':
        return 'indoor'
    return CLASSES[cls]['grid']


def indoor_grid(ref_tilts, los_pw, iso, bld, depth, n):
    """Power-average the street-level signal of a building cell's facade point
    and its open 8-neighbours (40 m grid). ref_tilts: (n_tilt, n*n) dBm with NaN."""
    nt = ref_tilts.shape[0]
    out = np.full_like(ref_tilts, np.nan, dtype=float)
    phi = np.zeros(n * n); B = bld.reshape(n, n)
    lin = np.where(np.isfinite(ref_tilts), 10 ** (ref_tilts / 10), np.nan)
    for iy in range(n):
        for ix in range(n):
            i = iy * n + ix
            if not B[iy, ix]:
                continue
            idx = [i]
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    y, x = iy + dy, ix + dx
                    if (dy or dx) and 0 <= y < n and 0 <= x < n and not B[y, x]:
                        idx.append(y * n + x)
            idx = np.array(idx)
            for t in range(nt):
                v = lin[t, idx]; v = v[np.isfinite(v)]
                if v.size:
                    out[t, i] = 10 * np.log10(v.mean())
            s_iso = iso[idx].sum()
            phi[i] = los_pw[idx].sum() / s_iso if s_iso > 0 else 0.0
    return out, phi


def class_stats(grid, cls, sc, common):
    C = CLASSES[cls]; g = grid[grid_key(cls, common)]; ti = TILTS.index(sc['tilt'])
    n = sdl = sul = cov = 0; tdl = tul = 0.0; dls = []
    for i in range(grid['n2']):
        if grid['dist'][i] > common['radius']:
            continue
        if (C['cells'] == 'building') != bool(grid['bld'][i]):
            continue
        n += 1
        ref = g['ref'][ti][i]
        r = cell(None if np.isnan(ref) else float(ref), float(g['phi'][i]), float(g['depth'][i]), cls, sc, common)
        if r is None:
            dls.append(0.0); continue
        sdl += r['servedDL']; sul += r['servedUL']; cov += r['rsrp'] >= common['rsrpThr']
        tdl += r['dl']; tul += r['ul']; dls.append(r['dl'])
    dls.sort()
    q = lambda p: dls[min(len(dls) - 1, int(math.floor(p * len(dls))))] if dls else 0.0
    return dict(n=n, servedDL=sdl / n if n else 0, servedUL=sul / n if n else 0, cov=cov / n if n else 0,
                dl=tdl / n if n else 0, ul=tul / n if n else 0, dlMed=q(0.5), dlP10=q(0.1),
                ratio=tdl / tul if tul > 0 else float('inf'))


def blend(grid, sc, common):
    per = {k: class_stats(grid, k, sc, common) for k in CLASSES}
    out = dict(servedDL=0.0, servedUL=0.0, cov=0.0, dl=0.0, ul=0.0); w = 0.0
    for k, s in per.items():
        wk = common['mix'].get(k, 0.0); w += wk
        for m in out:
            out[m] += wk * s[m]
    for m in out:
        out[m] = out[m] / w if w else 0.0
    out['ratio'] = out['dl'] / out['ul'] if out['ul'] > 0 else float('inf')
    return dict(per=per, all=out)


# ---------------- grid loading ----------------
def load_grid(site, band):
    """Assemble one site's roof + street grids for one band from data_v5_*.npz."""
    veg = np.load(os.path.join(ROOT, f'data_veg_{site}.npz'))
    sites = {s['name'].split('-')[1].lower(): s for s in json.load(open(os.path.join(ROOT, 'data_sites_v4.json')))}
    grid = {}
    for rx in ('roof', 'street'):
        d = np.load(os.path.join(ROOT, f'data_v5_{site}_{band}_{rx}.npz'))
        iso = d['iso']
        phi = np.where(iso > 0, d['los_pw'] / np.where(iso > 0, iso, 1), 0.0)
        grid[rx] = dict(ref=[d['rsrp_tilt'][:, k].astype(float) for k in range(len(TILTS))],
                        phi=phi, depth=veg[f'depth_{rx}'])
        xs, ys, bld = d['xs'], d['ys'], d['indoor']
    s = grid['street']; n = len(xs)
    dstreet = np.load(os.path.join(ROOT, f'data_v5_{site}_{band}_street.npz'))
    ref_in, phi_in = indoor_grid(np.array(s['ref']), dstreet['los_pw'], dstreet['iso'], bld.astype(bool), s['depth'], n)
    grid['indoor'] = dict(ref=[ref_in[k] for k in range(len(TILTS))], phi=phi_in, depth=s['depth'])
    st = sites[site]
    from build_scene_v4 import to_local
    x0, y0 = to_local(st['lon'], st['lat'])
    X, Y = np.meshgrid(xs, ys)
    grid.update(n2=X.size, dist=np.hypot(X - x0, Y - y0).reshape(-1), bld=bld.astype(bool))
    return grid


def load_grid_blob(blob, site, band):
    """Decode the viewer's packed data exactly as the browser does (float32
    quantisation included), for the Python/JS cross-check."""
    import base64
    S = blob['sites'][site]; B = S['bands'][band]; n = S['n']; n2 = n * n
    dec = lambda s, dt: np.frombuffer(base64.b64decode(s), dtype=dt)
    bld = dec(S['bld'], 'u1') == 1
    iy, ix = np.divmod(np.arange(n2), n)
    x = (S['x0'] + ix * S['step']); y = (S['y0'] + iy * S['step'])
    dist = np.hypot(x - S['tx'][0], y - S['tx'][1]).astype(np.float32)
    grid = dict(n2=n2, dist=dist, bld=bld)
    for rx in ('roof', 'street', 'indoor'):
        q = dec(B[rx]['ref'], '<i2').reshape(len(TILTS), n2)
        ref = [np.where(q[t] == -32768, np.nan, (q[t] / 10).astype(np.float32)).astype(float) for t in range(len(TILTS))]
        phi = (dec(B[rx]['phi'], 'u1') / 255).astype(np.float32).astype(float)
        depth = dec(S['depth']['street' if rx == 'indoor' else rx], 'u1').astype(float)
        grid[rx] = dict(ref=ref, phi=phi, depth=depth)
    return grid


def parse_sc(s):
    sc = dict(tilt=4, eirp=47.0, ports=2, tdd='DDDSU', iot=0.0, loadDL=0.0, loadUL=0.0)
    for kv in s.split(','):
        k, v = kv.split('=')
        k = {'loaddl': 'loadDL', 'loadul': 'loadUL'}.get(k, k)
        sc[k] = v if k in ('band', 'tdd') else (int(v) if k in ('bw', 'tilt', 'ports') else float(v))
    sc.setdefault('bw', BANDS[sc['band']]['bwdef'])
    return sc


def main():
    sys.path.insert(0, ROOT)
    ap = argparse.ArgumentParser()
    ap.add_argument('--site', default='nw')
    ap.add_argument('--before', required=True)
    ap.add_argument('--after', required=True)
    ap.add_argument('--mix', default='cpe=30,indoor=40,ped=20,veh=10')
    ap.add_argument('--radius', type=float, default=1000)
    ap.add_argument('--o2i', default='low')
    ap.add_argument('--indoor-ref', default='sides', choices=['sides', 'facade'])
    ap.add_argument('--no-foliage', action='store_true')
    ap.add_argument('--rsrp-thr', type=float, default=-110)
    ap.add_argument('--json', action='store_true')
    a = ap.parse_args()
    common = dict(mix={k: float(v) for k, v in (kv.split('=') for kv in a.mix.split(','))}, indoorRef=a.indoor_ref,
                  radius=a.radius, o2i=a.o2i, foliage=not a.no_foliage, rsrpThr=a.rsrp_thr)
    res = {}
    for lab, s in (('before', a.before), ('after', a.after)):
        sc = parse_sc(s)
        res[lab] = dict(scenario=sc, **blend(load_grid(a.site, sc['band']), sc, common))
    if a.json:
        print(json.dumps(res, indent=1, default=float)); return
    print(f"Site {a.site.upper()}, radius {a.radius:.0f} m, mix {common['mix']}, O2I {a.o2i}, foliage {'on' if common['foliage'] else 'off'}")
    for lab in ('before', 'after'):
        print(f"  {lab:6s}: {res[lab]['scenario']}")
    print(f"{'class':20s} {'DL served':>14s} {'UL served':>14s} {'DL Mbps':>15s} {'UL Mbps':>13s} {'DL/UL':>12s}")
    for k in list(CLASSES) + ['all']:
        A = res['before']['per'][k] if k != 'all' else res['before']['all']
        B = res['after']['per'][k] if k != 'all' else res['after']['all']
        name = CLASSES[k]['label'] if k != 'all' else 'Blended (mix)'
        print(f"{name:20s} {A['servedDL']*100:5.1f}->{B['servedDL']*100:5.1f}% {A['servedUL']*100:5.1f}->{B['servedUL']*100:5.1f}% "
              f"{A['dl']:6.1f}->{B['dl']:6.1f} {A['ul']:5.1f}->{B['ul']:5.1f} {A['ratio']:5.2f}->{B['ratio']:5.2f}")


if __name__ == '__main__':
    main()
