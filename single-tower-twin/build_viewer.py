#!/usr/bin/env python3
"""Build twin_viewer.html: interactive single-tower digital twin viewer.

Embeds the four precomputed Sionna RT grids (NW/NE/SW/SE) as JSON.
No dependencies, no server, no install - open in any browser.

Interactions:
  - site switcher (4 precomputed maps, instant)
  - EIRP slider 30-47 dBm (analytic dB shift, instant recolor + stats)
  - click-to-probe: bilinear RSRP/SNR/LOS/distance readout
"""
import json
import numpy as np

SITES = ['nw', 'ne', 'sw', 'se']
EIRP0 = 47.0
NOISE = -95.0  # dBm, 10 MHz, NF 9 dB

# viridis lookup, 33 entries
VIRIDIS = [
    (68, 1, 84), (72, 35, 116), (64, 67, 135), (52, 94, 141), (41, 120, 142),
    (32, 144, 140), (34, 167, 133), (68, 191, 112), (121, 209, 81), (189, 222, 38),
    (253, 231, 37),
]
# expand to 64 steps by linear interpolation
VIR = []
for i in range(len(VIRIDIS) - 1):
    a, b = VIRIDIS[i], VIRIDIS[i + 1]
    for k in range(6):
        f = k / 6
        VIR.append(tuple(int(a[j] + (b[j] - a[j]) * f) for j in range(3)))
VIR.append(VIRIDIS[-1])


def main():
    data = {}
    for s in SITES:
        d = np.load(f'data_twin_tower_{s}_f3600.npz')
        xs, ys = d['xs'], d['ys']
        rsrp = d['rsrp']
        los = d['los']
        meta = json.loads(str(d['meta']))
        n = len(xs)
        data[s] = {
            'n': n,
            'x0': round(float(xs[0]), 1), 'step': round(float(xs[1] - xs[0]), 2),
            'y0': round(float(ys[0]), 1),
            # RSRP at reference EIRP, 0.1 dB; NaN -> null
            'rsrp': [None if np.isnan(v) else round(float(v), 1)
                     for v in rsrp.reshape(-1)],
            'los': ''.join('1' if v else '0' for v in los.reshape(-1)),
            'tx': [round(float(xs[n // 2]), 1), round(float(ys[n // 2]), 1)],
            'txname': meta['site'],
        }
    # tx is grid centre by construction; store actual TX local coords instead
    sites_json = json.load(open('data_sites_v4.json'))
    import sys
    sys.path.insert(0, '.')
    from build_scene_v4 import to_local
    for sj in sites_json:
        key = sj['name'].split('-')[1].lower()
        if key in data:
            x, y = to_local(sj['lon'], sj['lat'])
            data[key]['tx'] = [round(float(x), 1), round(float(y), 1)]

    html = _HTML.replace('__DATA__', json.dumps(data, separators=(',', ':')))
    html = html.replace('__VIR__', json.dumps(VIR))
    open('twin_viewer.html', 'w').write(html)
    print('wrote v4/twin_viewer.html (%d KB)' % (len(html) // 1024))


_HTML = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Single-tower digital twin - interactive viewer</title>
<style>
body{font-family:system-ui,-apple-system,sans-serif;max-width:1060px;margin:18px auto;padding:0 14px;color:#222}
h2{margin:0 0 4px}.sub{color:#555;margin:0 0 12px;font-size:14px}
#wrap{display:flex;gap:18px;flex-wrap:wrap}
#map{border:1px solid #ccc;cursor:crosshair;touch-action:manipulation}
#panel{min-width:250px;flex:1}
.btns{margin-bottom:10px}
.btns button{margin:0 6px 6px 0;padding:7px 12px;font-size:14px;cursor:pointer;border:1px solid #999;background:#f4f4f4;border-radius:6px}
.btns button.on{background:#1a73e8;color:#fff;border-color:#1a73e8}
label{font-size:14px}
input[type=range]{width:100%}
#probe{background:#f7f9fc;border:1px solid #ddd;border-radius:8px;padding:10px 12px;margin:10px 0;font-size:14px;line-height:1.65}
#probe b{font-variant-numeric:tabular-nums}
#stats{font-size:14px;line-height:1.7}
.bar{height:8px;background:linear-gradient(90deg,#440154,#21918c,#fde725);border-radius:4px;margin:4px 0 2px}
.scale{display:flex;justify-content:space-between;font-size:12px;color:#555}
.foot{margin-top:14px;font-size:12px;color:#777}
</style></head><body>
<h2>Single-tower digital twin</h2>
<p class="sub">Sionna RT 2.2.0 ray tracing over Rancho Bernardo real terrain (USGS 3DEP) + LiDAR buildings &mdash;
3.6&nbsp;GHz CBRS, RX 6&nbsp;m above surface, max-depth 3 + diffraction. Click the map to probe any point.</p>
<div id="wrap">
<div><canvas id="map" width="620" height="620"></canvas>
<div class="bar"></div><div class="scale"><span>-110 dBm</span><span>RSRP</span><span>-50 dBm</span></div></div>
<div id="panel">
<div class="btns" id="sitebtns"></div>
<label>EIRP: <b id="eirpv">47.0</b> dBm</label>
<input type="range" id="eirp" min="30" max="47" step="0.5" value="47">
<div id="probe">Click anywhere on the map to probe signal.</div>
<div id="stats"></div>
</div></div>
<p class="foot">White cells: no ray-traced path found (genuine dead zones at depth 3). EIRP slider is an analytic dB
shift &mdash; no re-solve needed. Full grids: v4/data_twin_tower_&lt;site&gt;.npz</p>
<script>
const DATA=__DATA__, VIR=__VIR__;
const NOISE=-95, VMIN=-110, VMAX=-50;
let site='nw', eirp=47;
const cv=document.getElementById('map'), ctx=cv.getContext('2d');
const N=DATA[site].n, PX=cv.width/N;

function color(v){ // v in dBm
  if(v==null) return [255,255,255];
  let t=(v-VMIN)/(VMAX-VMIN); t=Math.max(0,Math.min(1,t));
  return VIR[Math.round(t*(VIR.length-1))];
}
function draw(){
  const d=DATA[site], img=ctx.createImageData(cv.width,cv.height);
  const sh=eirp-47;
  for(let iy=0; iy<N; iy++) for(let ix=0; ix<N; ix++){
    const v=d.rsrp[iy*N+ix], c=color(v==null?null:v+sh);
    for(let py=0; py<PX; py++) for(let px=0; px<PX; px++){
      const o=(((N-1-iy)*PX+py)*cv.width+ix*PX+px)*4;
      img.data[o]=c[0]; img.data[o+1]=c[1]; img.data[o+2]=c[2]; img.data[o+3]=255;
    }
  }
  ctx.putImageData(img,0,0);
  // TX marker
  const tx=km(d.tx);
  ctx.fillStyle='#e53935';
  ctx.beginPath();
  ctx.moveTo(tx[0],tx[1]-9); ctx.lineTo(tx[0]-7,tx[1]+6); ctx.lineTo(tx[0]+7,tx[1]+6);
  ctx.closePath(); ctx.fill();
  ctx.strokeStyle='#fff'; ctx.lineWidth=1.5; ctx.stroke();
  stats();
}
function km(p){ const d=DATA[site];
  return [(p[0]-d.x0)/1000*PX*N/ N*0+ (p[0]-d.x0)/(d.step*N)*cv.width,
          cv.height-(p[1]-d.y0)/(d.step*N)*cv.height]; }
function stats(){
  const d=DATA[site], sh=eirp-47;
  let c90=0,c100=0,n=0,snrs=[];
  for(const v of d.rsrp){ if(v==null) continue; n++; const r=v+sh;
    if(r>=-90)c90++; if(r>=-100)c100++; snrs.push(r-NOISE); }
  snrs.sort((a,b)=>a-b);
  const med=snrs.length?snrs[Math.floor(snrs.length/2)]:NaN;
  document.getElementById('stats').innerHTML=
    '<b>'+d.txname+'</b> &mdash; '+n.toLocaleString()+' cells<br>'+
    'Area &ge; &minus;90 dBm: <b>'+(100*c90/n).toFixed(1)+'%</b><br>'+
    'Area &ge; &minus;100 dBm: <b>'+(100*c100/n).toFixed(1)+'%</b><br>'+
    'Median SNR: <b>'+med.toFixed(1)+' dB</b>';
}
function probe(ev){
  const r=cv.getBoundingClientRect();
  const px=(ev.clientX-r.left)*cv.width/r.width, py=(ev.clientY-r.top)*cv.height/r.height;
  const d=DATA[site];
  const fx=px/PX, fy=(cv.height-py)/PX; // grid coords, fy=0 south
  const ix=Math.floor(fx), iy=Math.floor(fy);
  if(ix<0||iy<0||ix>=N-1||iy>=N-1) return;
  const ax=fx-ix, ay=fy-iy, sh=eirp-47;
  const g=(x,y)=>d.rsrp[y*N+x];
  const v00=g(ix,iy),v10=g(ix+1,iy),v01=g(ix,iy+1),v11=g(ix+1,iy+1);
  const el=document.getElementById('probe');
  if(v00==null&&v10==null&&v01==null&&v11==null){
    el.innerHTML='No ray-traced path at this point &mdash; dead zone.'; return; }
  const vals=[v00,v10,v01,v11].filter(v=>v!=null);
  const v=vals.reduce((a,b)=>a+b,0)/vals.length+sh;
  const losc=[d.los[iy*N+ix],d.los[iy*N+ix+1],d.los[(iy+1)*N+ix],d.los[(iy+1)*N+ix+1]];
  const los=losc.some(c=>c==='1');
  const mx=d.x0+(fx+0.5)*d.step, my=d.y0+(fy+0.5)*d.step;
  const dist=Math.hypot(mx-d.tx[0],my-d.tx[1]);
  // crosshair
  draw(); ctx.strokeStyle='#fff'; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(px-12,py); ctx.lineTo(px+12,py);
  ctx.moveTo(px,py-12); ctx.lineTo(px,py+12); ctx.stroke();
  ctx.strokeStyle='#333'; ctx.beginPath(); ctx.arc(px,py,3,0,7); ctx.stroke();
  el.innerHTML='RSRP: <b>'+v.toFixed(1)+' dBm</b> &nbsp; SNR: <b>'+(v-NOISE).toFixed(1)+' dB</b><br>'+
    (los?'LOS path present':'NLOS only')+' &nbsp;&middot;&nbsp; '+dist.toFixed(0)+' m from '+d.txname;
}
cv.addEventListener('click',probe);
document.getElementById('eirp').addEventListener('input',e=>{
  eirp=parseFloat(e.target.value);
  document.getElementById('eirpv').textContent=eirp.toFixed(1); draw(); });
const bb=document.getElementById('sitebtns');
for(const s of Object.keys(DATA)){
  const b=document.createElement('button'); b.textContent=DATA[s].txname;
  b.className=s===site?'on':'';
  b.onclick=()=>{site=s;eirp=47;document.getElementById('eirp').value=47;
    document.getElementById('eirpv').textContent='47.0';
    [...bb.children].forEach(x=>x.className='');b.className='on';draw();};
  bb.appendChild(b);
}
draw();
</script></body></html>
"""

if __name__ == '__main__':
    main()
