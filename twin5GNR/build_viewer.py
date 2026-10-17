#!/usr/bin/env python3
"""Build twin_viewer.html: interactive multi-band digital twin viewer.

Embeds precomputed Sionna RT grids (4 sites x 4 bands) as JSON.
No dependencies, no server, no install - open in any browser.

Interactions:
  - site switcher (4 precomputed maps, instant)
  - band switcher: 0.7 / 2.5 / 3.6 / 7.125 GHz (precomputed, instant)
  - EIRP slider 30-47 dBm (analytic dB shift, instant recolor + stats)
  - bandwidth slider 5-100 MHz (analytic noise floor -> SNR, instant)
  - click-to-probe: RSRP/SNR/LOS/distance readout
"""
import json
import numpy as np

SITES = ['nw', 'ne', 'sw', 'se']
BANDS = [('f700', 0.7, '0.7 GHz'), ('f2500', 2.5, '2.5 GHz'),
         ('f3600', 3.6, '3.6 GHz'), ('f7125', 7.125, '7.125 GHz')]
EIRP0 = 47.0

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
        for fstr, fghz, blabel in BANDS:
            key = f'{s}_{fstr}'
            d = np.load(f'v4/data_twin_tower_{s}_{fstr}.npz')
            xs, ys = d['xs'], d['ys']
            rsrp = d['rsrp']
            los = d['los']
            meta = json.loads(str(d['meta']))
            n = len(xs)
            data[key] = {
                'n': n,
                'x0': round(float(xs[0]), 1), 'step': round(float(xs[1] - xs[0]), 2),
                'y0': round(float(ys[0]), 1),
                # RSRP at reference EIRP, 0.1 dB; NaN -> null
                'rsrp': [None if np.isnan(v) else round(float(v), 1)
                         for v in rsrp.reshape(-1)],
                'los': ''.join('1' if v else '0' for v in los.reshape(-1)),
                'tx': [round(float(xs[n // 2]), 1), round(float(ys[n // 2]), 1)],
                'txname': meta['site'],
                'band': blabel,
                'mat': meta.get('mat_mode', 'itu'),
            }
    # tx is grid centre by construction; store actual TX local coords instead
    sites_json = json.load(open('v4/data_sites_v4.json'))
    import sys
    sys.path.insert(0, 'v4')
    from build_scene_v4 import to_local
    for sj in sites_json:
        key = sj['name'].split('-')[1].lower()
        x, y = to_local(sj['lon'], sj['lat'])
        for fstr, _, _ in BANDS:
            k = f'{key}_{fstr}'
            if k in data:
                data[k]['tx'] = [round(float(x), 1), round(float(y), 1)]

    html = _HTML.replace('__DATA__', json.dumps(data, separators=(',', ':')))
    html = html.replace('__VIR__', json.dumps(VIR))
    html = html.replace('__BANDS__', json.dumps(
        [{'f': fstr, 'label': blabel} for fstr, _, blabel in BANDS]))
    open('v4/twin_viewer.html', 'w').write(html)
    print('wrote v4/twin_viewer.html (%d KB, %d maps)' % (len(html) // 1024, len(data)))


_HTML = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>twin5GNR - interactive viewer</title>
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
<h2>twin5GNR</h2>
<p class="sub">Sionna RT 2.2.0 ray tracing over Rancho Bernardo real terrain (USGS 3DEP) + LiDAR buildings &mdash;
5G NR bands 0.7 / 2.5 / 3.6 / 7.125&nbsp;GHz, RX 6&nbsp;m above surface, max-depth 3 + diffraction.
Click the map to probe any point.</p>
<div id="wrap">
<div><canvas id="map" width="612" height="612"></canvas>
<div class="bar"></div><div class="scale"><span>-110 dBm</span><span>RSRP</span><span>-50 dBm</span></div></div>
<div id="panel">
<div class="btns" id="sitebtns"></div>
<div class="btns" id="bandbtns"></div>
<label>EIRP: <b id="eirpv">47.0</b> dBm</label>
<input type="range" id="eirp" min="30" max="47" step="0.5" value="47">
<label>Bandwidth: <b id="bwv">10</b> MHz</label>
<input type="range" id="bw" min="0" max="4" step="1" value="1">
<div id="probe">Click anywhere on the map to probe signal.</div>
<div id="stats"></div>
</div></div>
<p class="foot">White cells: no ray-traced path found (genuine dead zones at depth 3). EIRP slider is an analytic dB
shift; bandwidth slider moves the noise floor only &mdash; no re-solve needed. 0.7&nbsp;GHz uses ITU-R P.2040
materials clamped to their 1&nbsp;GHz boundary values (below the recommendation's defined range).
Full grids: v4/data_twin_tower_&lt;site&gt;_&lt;band&gt;.npz</p>
<script>
const DATA=__DATA__, VIR=__VIR__, BANDS=__BANDS__;
const VMIN=-110, VMAX=-50, BWS=[5,10,20,40,100];
let site='nw', band='f3600', eirp=47, bwi=1;
const key=()=>site+'_'+band;
const noise=()=>-174+10*Math.log10(BWS[bwi]*1e6)+9;
const cv=document.getElementById('map'), ctx=cv.getContext('2d');
const N=DATA[key()].n, PX=12; // cv.width == N*PX exactly: integer pixel writes

function color(v){ // v in dBm
  if(v==null) return [255,255,255];
  let t=(v-VMIN)/(VMAX-VMIN); t=Math.max(0,Math.min(1,t));
  return VIR[Math.round(t*(VIR.length-1))];
}
function draw(){
  const d=DATA[key()], img=ctx.createImageData(cv.width,cv.height);
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
function km(p){ const d=DATA[key()];
  return [(p[0]-d.x0)/(d.step*N)*cv.width, cv.height-(p[1]-d.y0)/(d.step*N)*cv.height]; }
function stats(){
  const d=DATA[key()], sh=eirp-47, nz=noise();
  let c90=0,c100=0,n=0,snrs=[];
  for(const v of d.rsrp){ if(v==null) continue; n++; const r=v+sh;
    if(r>=-90)c90++; if(r>=-100)c100++; snrs.push(r-nz); }
  snrs.sort((a,b)=>a-b);
  const med=snrs.length?snrs[Math.floor(snrs.length/2)]:NaN;
  document.getElementById('stats').innerHTML=
    '<b>'+d.txname+'</b> &mdash; '+d.band+' &mdash; '+n.toLocaleString()+' cells<br>'+
    'Area &ge; &minus;90 dBm: <b>'+(100*c90/n).toFixed(1)+'%</b><br>'+
    'Area &ge; &minus;100 dBm: <b>'+(100*c100/n).toFixed(1)+'%</b><br>'+
    'Median SNR ('+BWS[bwi]+' MHz): <b>'+med.toFixed(1)+' dB</b>';
}
function probeAt(px,py,redraw){
  const d=DATA[key()];
  const fx=px/PX, fy=(cv.height-py)/PX; // grid coords, fy=0 south
  const ix=Math.floor(fx), iy=Math.floor(fy);
  if(ix<0||iy<0||ix>=N-1||iy>=N-1) return;
  const sh=eirp-47, nz=noise();
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
  if(redraw) draw();
  // crosshair
  ctx.strokeStyle='#fff'; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(px-12,py); ctx.lineTo(px+12,py);
  ctx.moveTo(px,py-12); ctx.lineTo(px,py+12); ctx.stroke();
  ctx.strokeStyle='#333'; ctx.beginPath(); ctx.arc(px,py,3,0,7); ctx.stroke();
  el.innerHTML='RSRP: <b>'+v.toFixed(1)+' dBm</b> &nbsp; SNR: <b>'+(v-nz).toFixed(1)+' dB</b><br>'+
    (los?'LOS path present':'NLOS only')+' &nbsp;&middot;&nbsp; '+dist.toFixed(0)+' m from '+d.txname;
}
let lastProbe=null;
function probe(ev){
  const r=cv.getBoundingClientRect();
  const px=(ev.clientX-r.left)*cv.width/r.width, py=(ev.clientY-r.top)*cv.height/r.height;
  lastProbe=[px,py]; probeAt(px,py,true);
}
cv.addEventListener('click',probe);
document.getElementById('eirp').addEventListener('input',e=>{
  eirp=parseFloat(e.target.value);
  document.getElementById('eirpv').textContent=eirp.toFixed(1); draw();
  if(lastProbe) probeAt(lastProbe[0],lastProbe[1],false); });
document.getElementById('bw').addEventListener('input',e=>{
  bwi=parseInt(e.target.value);
  document.getElementById('bwv').textContent=BWS[bwi]; draw();
  if(lastProbe) probeAt(lastProbe[0],lastProbe[1],false); });
function mkbtns(elid, items, cur, onpick){
  const bb=document.getElementById(elid); bb.innerHTML='';
  for(const it of items){
    const b=document.createElement('button'); b.textContent=it.label;
    b.className=it.id===cur?'on':'';
    b.onclick=()=>{onpick(it.id);
      document.getElementById('eirp').value=47; eirp=47;
      document.getElementById('eirpv').textContent='47.0';
      lastProbe=null;
      document.getElementById('probe').innerHTML='Click anywhere on the map to probe signal.';
      [...bb.children].forEach(x=>x.className='');b.className='on';draw();};
    bb.appendChild(b);
  }
}
const SITENAMES={nw:'Site-NW',ne:'Site-NE',sw:'Site-SW',se:'Site-SE'};
function buildbtns(){
  mkbtns('sitebtns', Object.keys(SITENAMES).map(s=>({id:s,label:SITENAMES[s]})), site, id=>{site=id;});
  mkbtns('bandbtns', BANDS.map(b=>({id:b.f,label:b.label})), band, id=>{band=id;});
}
buildbtns();
draw();
</script></body></html>
"""

if __name__ == '__main__':
    main()
