#!/usr/bin/env python3
"""build_viewer_v5.py - pack the v5 grids and write the upgrade-planner viewer.

Reads every complete data_v5_<site>_<band>_<rx>.npz (both rx grids needed for
a site-band to be included), data_veg_<site>.npz and data_sites_v4.json, and
writes:
    twin_v5_data.json   compact data blob (also used by the Node / Python
                        cross-check)
    twin_viewer_v5.html standalone viewer: twin_model.js + data inlined,
                        open in any browser, no server
Encoding: DL received power at the 47 dBm reference per tilt as int16
(0.1 dB, -32768 = no ray path), LOS power share as uint8 (/255), foliage
depth as uint8 metres, building flag as uint8; all little-endian base64.
"""
import base64, json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from build_scene_v4 import to_local

SITES = ['nw', 'ne', 'sw', 'se']
BANDS = ['f700', 'f2500', 'f3600', 'f7125', 'f28000']
b64 = lambda a: base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def pack():
    sites = {s['name'].split('-')[1].lower(): s for s in json.load(open(os.path.join(ROOT, 'data_sites_v4.json')))}
    blob = {'sites': {}, 'version': 'v5-diffuse'}
    for s in SITES:
        vfile = os.path.join(ROOT, f'data_veg_{s}.npz')
        if not os.path.exists(vfile):
            continue
        veg = np.load(vfile)
        entry = {'bands': {}}
        for b in BANDS:
            files = {rx: os.path.join(ROOT, f'data_v5_{s}_{b}_{rx}.npz') for rx in ('roof', 'street')}
            ok = all(os.path.exists(f) and 'rsrp_tilt' in np.load(f).files for f in files.values())
            if not ok:
                continue
            band = {}
            for rx, f in files.items():
                d = np.load(f)
                r = d['rsrp_tilt']
                q = np.where(np.isnan(r), -32768, np.round(r * 10)).astype('<i2')
                iso = d['iso']
                phi = np.where(iso > 0, d['los_pw'] / np.where(iso > 0, iso, 1.0), 0.0)
                band[rx] = {'ref': b64(q.T), 'phi': b64(np.round(np.clip(phi, 0, 1) * 255).astype('u1'))}
                if 'xs' not in entry:
                    xs, ys = d['xs'], d['ys']
                    st = sites[s]
                    x0, y0 = to_local(st['lon'], st['lat'])
                    entry.update(n=int(len(xs)), x0=float(xs[0]), y0=float(ys[0]), step=float(xs[1] - xs[0]),
                                 tx=[float(x0), float(y0)], name=st['name'], mast=float(st['tx_agl_m']),
                                 bld=b64(d['indoor'].astype('u1')))
            # indoor: power average over each building's facade neighbourhood
            sys.path.insert(0, ROOT)
            import twin_model as TM
            ds = np.load(files['street'])
            rs = ds['rsrp_tilt'].T.astype(float)
            ref_in, phi_in = TM.indoor_grid(rs, ds['los_pw'], ds['iso'], ds['indoor'].astype(bool), None, int(len(ds['xs'])))
            q = np.where(np.isnan(ref_in), -32768, np.round(ref_in * 10)).astype('<i2')
            band['indoor'] = {'ref': b64(q), 'phi': b64(np.round(np.clip(phi_in, 0, 1) * 255).astype('u1'))}
            entry['bands'][b] = band
        entry['depth'] = {rx: b64(np.clip(np.round(veg[f'depth_{rx}']), 0, 255).astype('u1')) for rx in ('roof', 'street')}
        if entry['bands']:
            blob['sites'][s] = entry
    return blob


def main():
    blob = pack()
    js = json.dumps(blob, separators=(',', ':'))
    open(os.path.join(ROOT, 'twin_v5_data.json'), 'w').write(js)
    tpl = open(os.path.join(ROOT, 'viewer_v5_template.html')).read()
    model = open(os.path.join(ROOT, 'twin_model.js')).read()
    review = ''
    if '--review' in sys.argv:
        review = open(sys.argv[sys.argv.index('--review') + 1]).read()
    html = tpl.replace('/*__MODEL__*/', model).replace('"__DATA__"', js).replace('<!--__REVIEW__-->', review)
    out = sys.argv[sys.argv.index('--out') + 1] if '--out' in sys.argv else os.path.join(ROOT, 'twin_viewer_v5.html')
    open(out, 'w').write(html)
    have = {s: list(e['bands']) for s, e in blob['sites'].items()}
    print(f'wrote {out} ({len(html)//1024} KB); site-bands included: {have}')


if __name__ == '__main__':
    main()
