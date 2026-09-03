/* Multi-Tier Digital Twin v2 browser engine.
 * A deterministic aggregate screening implementation for editable GitHub Pages
 * use. It intentionally does not claim numerical parity with the Python twin. */
(() => {
  'use strict';
  const P = window.PRECOMPUTED || {};
  const clamp = (x, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, x));
  const deep = x => JSON.parse(JSON.stringify(x));
  const value = id => Number(document.getElementById(id).value);
  const POLICIES = ['static', 'demand_follow', 'rule_based', 'ppo', 'dqn'];
  const TEMPLATES = [
    [.55,.25,.20], [.40,.40,.20], [.35,.25,.40], [.70,.20,.10],
    [.25,.60,.15], [.20,.20,.60], [.46,.34,.20], [.34,.46,.20], [.34,.26,.40]
  ];
  const ADV_DEFAULTS = {};

  function rng(seed) {
    let a = seed >>> 0;
    return () => {
      a |= 0; a = a + 0x6D2B79F5 | 0;
      let t = Math.imul(a ^ a >>> 15, 1 | a);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }

  function normal(random) {
    const u = Math.max(1e-9, random()), v = Math.max(1e-9, random());
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  function baseTier(name) {
    const previews = Object.values(P.preview || {});
    for (const p of previews) {
      const t = p.plan.tiers.find(x => x.tier === name);
      if (t) return t;
    }
    throw new Error(`Missing reference tier ${name}`);
  }

  function buildAdvancedRows() {
    const rows = TIER_DEFAULTS.map((d, i) => {
      const t = baseTier(d.tier);
      ADV_DEFAULTS[d.tier] = {
        band: t.band_ghz, scs: t.numerology_khz, prbs: t.prbs_per_cell,
        tx: t.tx_power_dbm, height: t.antenna_height_m, gain: t.antenna_gain_dbi,
        nf: t.noise_figure_db, maxse: ({macro_low:3,macro_mid:5.5,umb_6g:7,
          mmwave:8,wifi7_indoor:7.5,ntn_leo:2.5})[d.tier],
        mimoEff: .75, cio: ({macro_low:0,macro_mid:8,umb_6g:14,mmwave:18,
          wifi7_indoor:0,ntn_leo:0})[d.tier], x: .15 + .12*i, y: .5
      };
      const a = ADV_DEFAULTS[d.tier];
      return `<tr data-adv-tier="${d.tier}"><td>${d.tier}</td>
        <td><input data-a="band" type="number" value="${a.band}" step="0.1"></td>
        <td><select data-a="scs">${[15,30,60,120].map(x=>`<option${x===a.scs?' selected':''}>${x}</option>`).join('')}</select></td>
        <td><select data-a="mode"><option value="manual">Manual</option><option value="auto">Auto</option></select></td>
        <td><input data-a="prbs" type="number" value="${a.prbs}" min="1"></td>
        <td><input data-a="tx" type="number" value="${a.tx}" step="1"></td>
        <td><input data-a="height" type="number" value="${a.height}" step="1"></td>
        <td><input data-a="gain" type="number" value="${a.gain}" step="1"></td>
        <td><input data-a="nf" type="number" value="${a.nf}" step="0.5"></td>
        <td><input data-a="maxse" type="number" value="${a.maxse}" step="0.1"></td>
        <td><input data-a="mimoEff" type="number" value="${a.mimoEff}" step="0.05" min="0.1" max="1"></td>
        <td><input data-a="cio" type="number" value="${a.cio}" step="1"></td>
        <td><input data-a="x" type="number" value="${a.x.toFixed(2)}" step="0.05"></td>
        <td><input data-a="y" type="number" value="${a.y.toFixed(2)}" step="0.05"></td></tr>`;
    }).join('');
    document.getElementById('advancedTierRows').innerHTML = rows;
  }

  function readAdvanced(name) {
    const tr = document.querySelector(`[data-adv-tier="${name}"]`);
    const get = k => tr.querySelector(`[data-a="${k}"]`).value;
    return {band:+get('band'), scs:+get('scs'), mode:get('mode'), prbs:+get('prbs'),
      tx:+get('tx'), height:+get('height'), gain:+get('gain'), nf:+get('nf'),
      maxse:+get('maxse'), mimoEff:+get('mimoEff'), cio:+get('cio'), x:+get('x'), y:+get('y')};
  }

  function browserConfig() {
    const scenario = document.getElementById('scenario').value;
    const p = P.preview[scenario];
    const area = p.plan.area_km;
    const basics = {};
    document.querySelectorAll('#tierRows tr').forEach(tr => {
      const get = k => tr.querySelector(`[data-f="${k}"]`).value;
      basics[tr.dataset.tier] = {cells:+get('cells'), morphology:get('scenario'),
        bandwidth:+get('bandwidth_mhz'), carriers:+get('carriers'),
        layers:+get('mimo_layers'), mode:get('mimo_mode')};
    });
    const seeds = document.getElementById('evaluation_seeds').value.split(',')
      .map(x=>Number(x.trim())).filter(Number.isFinite).slice(0,20);
    if (!seeds.length) seeds.push(value('seed'));
    return {scenario, area, basics, advanced:Object.fromEntries(TIER_DEFAULTS.map(t=>[t.tier,readAdvanced(t.tier)])),
      sessions:value('n_sessions'), indoor:value('indoor_fraction'), steps:Math.max(2,value('episode_steps')),
      ticks:Math.max(1,value('control_interval_ticks')), tickSeconds:value('tick_seconds'),
      rain:value('rain_rate_mm_h'), doppler:value('doppler_precompensation'), episodes:Math.max(1,value('train_episodes')),
      seeds, interference:value('interference')===1, shadow:value('shadow_fading')===1,
      margin:value('link_margin_db'), a3:value('a3_offset_db'), hysteresis:value('a3_hysteresis_db'),
      ttt:value('a3_ttt_ms'), a5Serving:value('a5_serving_dbm'), a5Neighbour:value('a5_neighbour_dbm'),
      minRsrp:value('min_rsrp_dbm'), wifiBias:value('wifi_bias_db'),
      ntnThreshold:value('ntn_threshold_dbm'), ntnSpeed:value('ntn_speed_kmh'),
      material:value('material_gain_pct'), continuityGuard:value('continuity_guard_pp')};
  }

  function calculatedPreview(c) {
    const out = deep(P.preview[c.scenario]);
    const areaKm2 = c.area[0] * c.area[1];
    const tiers = [];
    for (const def of TIER_DEFAULTS) {
      const b = c.basics[def.tier], a = c.advanced[def.tier], ref = baseTier(def.tier);
      if (b.cells <= 0) continue;
      const prbBW = def.tier === 'wifi7_indoor' ? 2.0 : 12 * a.scs / 1000;
      const prbs = a.mode === 'auto' ? Math.max(1, Math.floor(.95*b.bandwidth*b.carriers/prbBW)) : Math.max(1,Math.round(a.prbs));
      const mimoGain = Math.max(1,b.layers)*a.mimoEff*(b.mode==='MU'?1.15:1);
      const peak = prbs*prbBW*a.maxse*mimoGain;
      const noiseDelta = -10*Math.log10(prbBW/ref.prb_bandwidth_mhz) - (a.nf-ref.noise_figure_db);
      const budgetDelta = (a.tx-ref.tx_power_dbm)+(a.gain-ref.antenna_gain_dbi)+noiseDelta-(c.margin-8);
      const exponent = b.morphology==='RMa'?3.0:b.morphology==='InH'?2.6:b.morphology==='UMi'?3.3:3.5;
      let radius = ref.ntn ? ref.slant_range_km*1000 : ref.cell_radius_m*Math.pow(10,budgetDelta/(10*exponent))*Math.pow(ref.band_ghz/a.band,.35);
      const interferenceFactor = c.interference ? clamp(ref.loaded_cell_capacity_mbps/ref.peak_cell_capacity_mbps,.25,1) : 1;
      const rainFactor = 1-clamp(c.rain*Math.max(0,a.band-10)/4000,0,.65);
      const loaded = peak*interferenceFactor*rainFactor;
      const t = {...ref, band_ghz:a.band, bandwidth_mhz:b.bandwidth, carriers:b.carriers,
        aggregate_bandwidth_mhz:b.bandwidth*b.carriers, numerology_khz:a.scs,
        prb_bandwidth_mhz:prbBW, prbs_per_cell:prbs, mimo:`${b.layers}x ${b.mode}`,
        mimo_capacity_gain:+mimoGain.toFixed(3), morphology:b.morphology,
        pathloss_model:def.tier==='ntn_leo'?'TR38.811-NTN':`TR38.901-${b.morphology}`,
        cells:b.cells, antenna_height_m:a.height, tx_power_dbm:a.tx, antenna_gain_dbi:a.gain,
        noise_figure_db:a.nf, peak_cell_capacity_mbps:+peak.toFixed(1),
        tier_capacity_mbps:+(peak*b.cells).toFixed(1), loaded_cell_capacity_mbps:+loaded.toFixed(1),
        loaded_tier_capacity_mbps:+(loaded*b.cells).toFixed(1), centre_x_km:a.x, centre_y_km:a.y,
        cell_radius_m:+radius.toFixed(1), area_covered:ref.ntn?1:clamp(b.cells*Math.PI*(radius/1000)**2/areaKm2,0,.95)};
      tiers.push(t);
    }
    const eligible = slice => tiers.filter(t=>t.slices.includes(slice));
    const sum = (xs,k) => xs.reduce((z,x)=>z+x[k],0);
    out.plan.tiers=tiers; out.plan.total_cells=sum(tiers,'cells'); out.plan.area_km=c.area; out.plan.area_km2=areaKm2;
    out.plan.capacity={network_mbps:sum(tiers,'tier_capacity_mbps'),network_loaded_mbps:sum(tiers,'loaded_tier_capacity_mbps'),
      by_tier_mbps:Object.fromEntries(tiers.map(t=>[t.tier,t.tier_capacity_mbps])),
      by_tier_loaded_mbps:Object.fromEntries(tiers.map(t=>[t.tier,t.loaded_tier_capacity_mbps])),
      by_slice_mbps:Object.fromEntries(SLICES.map(s=>[s,sum(eligible(s),'tier_capacity_mbps')])),
      by_slice_loaded_mbps:Object.fromEntries(SLICES.map(s=>[s,sum(eligible(s),'loaded_tier_capacity_mbps')])),
      basis:'JS v2 aggregate calculation: theoretical peak and interference/rain-derated loaded capacity'};
    out.plan.coverage={area_km2:areaKm2,fade_margin:c.shadow?'90th-percentile screening margin':'disabled',
      by_tier:Object.fromEntries(tiers.map(t=>[t.tier,t.area_covered])),
      terrestrial_best:Math.max(0,...tiers.filter(t=>!t.ntn).map(t=>t.area_covered)),ntn_backstop:tiers.some(t=>t.ntn)};
    out.capacity=out.plan.capacity; out.coverage=out.plan.coverage;
    return out;
  }

  function scenarioMix(name) {
    return ({urban_dense:[.55,.15,.30],highway_mobility:[.60,.30,.10],indoor_enterprise:[.55,.15,.30],
      coverage_hole:[.35,.10,.55],rain_fade:[.50,.15,.35],cellular_only:[.55,.15,.30]})[name] || [.55,.15,.30];
  }

  function stateAt(c, random, t) {
    const mix=scenarioMix(c.scenario), phase=2*Math.PI*t/c.steps;
    const load=clamp(.72+.22*Math.sin(phase-1)+.08*normal(random),.25,1.35);
    const d=mix.map((m,i)=>Math.max(.01,m*load*(1+.18*Math.sin(phase+i*1.7))));
    const z=d.reduce((a,b)=>a+b,0); return [...d.map(x=>x/z),load,c.indoor,c.rain/50,t/c.steps,1];
  }

  function softmax(xs) { const m=Math.max(...xs), e=xs.map(x=>Math.exp(x-m)), z=e.reduce((a,b)=>a+b,0); return e.map(x=>x/z); }
  function dot(a,b){return a.reduce((z,x,i)=>z+x*b[i],0);}
  function choose(p,random){let x=random();for(let i=0;i<p.length;i++){x-=p[i];if(x<=0)return i;}return p.length-1;}

  function outcome(c, preview, state, alloc) {
    const totalLoaded=preview.plan.capacity.network_loaded_mbps;
    const rates=[5,1,.02], mix=scenarioMix(c.scenario);
    const offered=mix.map((m,i)=>c.sessions*m*rates[i]*state[3]);
    const capable=SLICES.map((s,i)=>preview.plan.tiers.filter(t=>t.slices.includes(s)).reduce((z,t)=>z+t.loaded_tier_capacity_mbps*alloc[i],0));
    const served=offered.map((x,i)=>Math.min(x,capable[i]));
    const sat=offered.map((x,i)=>clamp(served[i]/Math.max(x,1e-9)));
    const util=clamp(served.reduce((a,b)=>a+b,0)/Math.max(totalLoaded,1));
    const fairness=(sat.reduce((a,b)=>a+b,0)**2)/(3*sat.reduce((z,x)=>z+x*x,0)+1e-9);
    const violation=(1-sat[0])*.35+(1-sat[1])*.45+(1-sat[2])*.20;
    const energy=clamp(.30+.65*util);
    return {offered,served,sat,util,fairness,energy,reward:.45*sat[0]+.35*sat[1]+.20*sat[2]+.06*util+.04*fairness-.12*violation-.03*energy};
  }

  function train(c, preview, algorithm, seed) {
    const random=rng(seed+(algorithm==='ppo'?17:29)), n=8, weights=TEMPLATES.map(()=>Array(n).fill(0).map(()=>normal(random)*.02));
    for(let ep=0;ep<c.episodes;ep++) for(let t=0;t<c.steps;t++) {
      const s=stateAt(c,random,t), scores=weights.map(w=>dot(w,s)); let action;
      if(algorithm==='ppo') action=choose(softmax(scores),random);
      else action=random()<Math.max(.05,.45*(1-ep/c.episodes))?Math.floor(random()*TEMPLATES.length):scores.indexOf(Math.max(...scores));
      const r=outcome(c,preview,s,TEMPLATES[action]).reward;
      if(algorithm==='ppo') {
        const p=softmax(scores), baseline=p.reduce((z,x,i)=>z+x*outcome(c,preview,s,TEMPLATES[i]).reward,0);
        weights.forEach((w,i)=>w.forEach((_,j)=>w[j]+=.06*(r-baseline)*((i===action?1:0)-p[i])*s[j]));
      } else weights[action].forEach((_,j)=>weights[action][j]+=.045*(r-scores[action])*s[j]);
    }
    return s => {const q=weights.map(w=>dot(w,s));return q.indexOf(Math.max(...q));};
  }

  function evaluate(c, preview, policy, seed, learned=null) {
    const random=rng(seed+500), sums={reward:0,sat:[0,0,0],offered:[0,0,0],served:[0,0,0],util:0,fair:0,energy:0};
    let last=0;
    for(let t=0;t<c.steps;t++) {
      const s=stateAt(c,random,t); let alloc;
      if(policy==='static') alloc=TEMPLATES[0];
      else if(policy==='demand_follow') alloc=s.slice(0,3);
      else if(policy==='rule_based') {const d=s.slice(0,3).map((x,i)=>Math.max(x,[.30,.25,.20][i])),z=d.reduce((a,b)=>a+b,0);alloc=d.map(x=>x/z);}
      else {last=learned(s);alloc=TEMPLATES[last];}
      const o=outcome(c,preview,s,alloc); sums.reward+=o.reward;sums.util+=o.util;sums.fair+=o.fairness;sums.energy+=o.energy;
      for(let i=0;i<3;i++){sums.sat[i]+=o.sat[i];sums.offered[i]+=o.offered[i];sums.served[i]+=o.served[i];}
    }
    const n=c.steps, mobility=c.scenario==='highway_mobility'?.16:.035;
    const tuningPenalty=Math.abs(c.a3-3)*.002+Math.abs(c.hysteresis-1)*.002+Math.abs(c.ttt-160)/100000
      +Math.abs(c.a5Serving+118)*.0005+Math.abs(c.a5Neighbour+108)*.0005;
    const hoSuccess=clamp(.992-mobility-tuningPenalty+(policy==='rule_based'?.012:policy==='ppo'||policy==='dqn'?.008:0));
    const continuity=clamp(1-mobility*.08-(1-hoSuccess)*.08);
    const tiers=preview.plan.tiers, tierTotal=tiers.reduce((z,t)=>z+t.loaded_tier_capacity_mbps,0);
    const share=Object.fromEntries(tiers.map(t=>[t.tier,t.loaded_tier_capacity_mbps/Math.max(1,tierTotal)]));
    const intent=Object.fromEntries(tiers.map(t=>[t.tier,policy==='static'?TEMPLATES[0]:policy==='demand_follow'?scenarioMix(c.scenario):policy==='rule_based'?TEMPLATES[6]:TEMPLATES[last]]));
    return {mean_reward:sums.reward/n,mean_satisfaction:Object.fromEntries(SLICES.map((s,i)=>[s,sums.sat[i]/n])),
      sla_compliance:Object.fromEntries(SLICES.map((s,i)=>[s,clamp((sums.sat[i]/n-.7)/.3)])),
      offered_mbps:Object.fromEntries(SLICES.map((s,i)=>[s,sums.offered[i]])),served_mbps:Object.fromEntries(SLICES.map((s,i)=>[s,sums.served[i]])),
      network_utilization:sums.util/n,jain_fairness:sums.fair/n,mean_access_latency_ms:1.2+8*(1-sums.sat[1]/n)+mobility*12,
      energy_fraction:sums.energy/n,mean_spectral_efficiency:3.5,unattached_sessions:c.sessions*(1-continuity),
      coverage_gap_mbps:Object.fromEntries(SLICES.map(s=>[s,0])),capacity_gap_mbps:Object.fromEntries(SLICES.map((s,i)=>[s,Math.max(0,sums.offered[i]-sums.served[i])/n])),
      tier_headroom_mbps:Object.fromEntries(tiers.map(t=>[t.tier,t.loaded_tier_capacity_mbps*(1-sums.util/n)])),tier_intent:intent,tier_session_share:share,
      handover:{attempts:Math.round(c.sessions*mobility),success_rate:hoSuccess,failure_rate:1-hoSuccess,ping_pong_rate:clamp(.018+c.hysteresis*.004),radio_link_failures:Math.round(c.sessions*mobility*(1-hoSuccess)),mean_interruption_ms:40+c.ttt*.1},
      session_continuity:{overall:continuity,eMBB:continuity,URLLC:clamp(continuity-.002),mIoT:clamp(continuity+.002)},posture_mix:{balanced:n}};
  }

  function comparisonForSeed(c, preview, seed) {
    const ppo=train(c,preview,'ppo',seed), dqn=train(c,preview,'dqn',seed), results={};
    for(const k of POLICIES) results[k]=evaluate(c,preview,k,seed,k==='ppo'?ppo:k==='dqn'?dqn:null);
    const baseline=results.rule_based.mean_reward;
    for(const r of Object.values(results)) r.gain_vs_rule_based_pct=100*(r.mean_reward-baseline)/Math.max(Math.abs(baseline),1e-9);
    return results;
  }

  function stat(xs) {const mean=xs.reduce((a,b)=>a+b,0)/xs.length;if(xs.length<2)return{mean,ci95_low:mean,ci95_high:mean};const sd=Math.sqrt(xs.reduce((z,x)=>z+(x-mean)**2,0)/(xs.length-1)),t=xs.length===5?2.776:1.96,h=t*sd/Math.sqrt(xs.length);return{mean,ci95_low:mean-h,ci95_high:mean+h};}
  function validation(c,runs){const summary={};for(const k of POLICIES){summary[k]={};for(const m of ['mean_reward','session_continuity','handover_success']){const xs=runs.map(r=>m==='mean_reward'?r[k].mean_reward:m==='session_continuity'?r[k].session_continuity.overall:r[k].handover.success_rate);summary[k][m]=stat(xs);}summary[k].mean_access_latency_ms=stat(runs.map(r=>r[k].mean_access_latency_ms));}
    let pw=0,dw=0,ties=0;runs.forEach(r=>{const d=r.ppo.mean_reward-r.dqn.mean_reward;if(Math.abs(d)<1e-9)ties++;else if(d>0)pw++;else dw++;});return{seeds:c.seeds,training_episodes_per_seed:c.episodes,scope:'browser JS aggregate sensitivity; not Python parity or convergence evidence',summary,ppo_reward_wins:pw,dqn_reward_wins:dw,ties};}

  function buildComparison(c) {
    const preview=calculatedPreview(c), runs=c.seeds.map(seed=>comparisonForSeed(c,preview,seed)), results=runs[0];
    const best=POLICIES.reduce((a,b)=>results[a].mean_reward>=results[b].mean_reward?a:b), learned=results.ppo.mean_reward>=results.dqn.mean_reward?'ppo':'dqn';
    const deterministic=['static','demand_follow','rule_based'].reduce((a,b)=>results[a].mean_reward>=results[b].mean_reward?a:b);
    const comparator=results[deterministic], gain=100*(results[learned].mean_reward-comparator.mean_reward)/Math.max(Math.abs(comparator.mean_reward),1e-9),cd=100*(results[learned].session_continuity.overall-comparator.session_continuity.overall);
    let status,reason,recommended=deterministic;if(cd < -c.continuityGuard){status='UNDERPERFORMING';reason=`Learned policy violates the continuity guard against ${LABEL[deterministic]}.`;}else if(gain<c.material){status=gain<0?'UNDERPERFORMING':'NO MATERIAL BENEFIT';reason=`Learned-policy gain is below the threshold versus ${LABEL[deterministic]}.`;}else{status=`PASS — ${learned.toUpperCase()} preferred`;reason=`Learned policy clears the reward and continuity guards against ${LABEL[deterministic]} in this browser model.`;recommended=learned;}
    return {implementation:'multi_tier_v2_browser_js_aggregate',evidence_class:'synthetic_browser_screening',architecture:'centralized aggregate branching approximation',scenario:c.scenario,scenario_label:P.compare[c.scenario].scenario_label,scenario_description:P.compare[c.scenario].scenario_description,
      config:c,plan:preview.plan,training:{ppo:{algorithm:'linear-softmax PPO approximation',episodes:c.episodes},dqn:{algorithm:'linear Double-DQN approximation',episodes:c.episodes}},
      action_space:{type:'browser aggregate template selection',branches:preview.plan.tiers.length+1,branch_sizes:preview.plan.tiers.map(()=>9).concat(4),per_tier_templates:9,postures:['balanced','wifi-first','terrestrial-first','ntn-resilience'],equivalent_joint_actions:Math.pow(9,preview.plan.tiers.length)*4},observation_dim:8,results,best_policy:best,
      decision:{status,reason,best_learned:learned,gain_vs_rule_based_pct:gain,continuity_delta_pp:cd,handover_success_delta_pp:100*(results[learned].handover.success_rate-results.rule_based.handover.success_rate),recommended_policy:recommended},
      feasibility:P.compare[c.scenario].feasibility,validation:validation(c,runs),limitations:['Browser v2 is an aggregate JavaScript screening model and is not numerically identical to the Python twin.','Synthetic sessions; not carrier evidence.','PRACH contention, collisions, retries, ACB and access failures are not modelled.','Linear browser PPO/DQN approximations do not establish neural-agent convergence.','Propagation and handover outcomes are screening approximations; no ray tracing or RRC signalling simulation.']};
  }

  function optimizer(c, objective) {
    const preview=calculatedPreview(c), base=evaluate(c,preview,'rule_based',c.seeds[0]), fields={handover:['a3','hysteresis','ttt'],throughput:['wifiBias','minRsrp'],latency:['ttt','a3'],coverage:['ntnThreshold','minRsrp']}[objective];
    const grids={a3:[1,2,3,4.5,6],hysteresis:[.5,1,2,3],ttt:[40,100,160,320,640],wifiBias:[2,6,10,14],minRsrp:[-135,-130,-125,-120],ntnThreshold:[-126,-120,-114,-108]};
    const score=r=>objective==='handover'?1.5*r.session_continuity.overall+r.handover.success_rate-.8*r.handover.ping_pong_rate+.5*r.mean_reward:objective==='throughput'?Object.values(r.served_mbps).reduce((a,b)=>a+b,0)/1000+.5*r.mean_satisfaction.eMBB:objective==='latency'?-r.mean_access_latency_ms/5+.5*r.mean_satisfaction.URLLC:2+r.session_continuity.overall;
    const initial=Object.fromEntries(fields.map(k=>[k,c[k]]));let current={...initial},bestScore=score(base),history=[],evaluations=1;
    for(const field of fields){let best=current[field];for(const v of grids[field]){const trial={...c,...current,[field]:v};const r=evaluate(trial,calculatedPreview(trial),'rule_based',c.seeds[0]);const s=score(r);evaluations++;history.push({knob:field,value:v,score:s,objective_kpi:objective==='handover'?r.handover.success_rate:objective==='throughput'?Object.values(r.served_mbps).reduce((a,b)=>a+b,0):objective==='latency'?r.mean_access_latency_ms:r.session_continuity.overall,continuity:r.session_continuity.overall});if(s>bestScore){bestScore=s;best=v;}}current[field]=best;}
    const tunedC={...c,...current},tuned=evaluate(tunedC,calculatedPreview(tunedC),'rule_based',c.seeds[0]);const labels={handover:'Handover thresholds',throughput:'Throughput (eMBB)',latency:'Latency (URLLC)',coverage:'Coverage (mIoT / edge)'},kpi=r=>objective==='handover'?r.handover.success_rate:objective==='throughput'?Object.values(r.served_mbps).reduce((a,b)=>a+b,0):objective==='latency'?r.mean_access_latency_ms:r.session_continuity.overall;
    return {implementation:'browser_js_coordinate_search',objective,objective_label:labels[objective],scenario:c.scenario,scenario_label:P.compare[c.scenario].scenario_label,feasibility:P.compare[c.scenario].feasibility,thresholds_can_help:P.compare[c.scenario].feasibility.thresholds_can_help,evaluations,search_seconds:0,knobs:fields,baseline:{values:initial,score:score(base),objective_kpi:kpi(base),continuity:base.session_continuity.overall,handover_success:base.handover.success_rate,mean_access_latency_ms:base.mean_access_latency_ms,served_mbps:base.served_mbps},optimized:{values:current,score:bestScore,objective_kpi:kpi(tuned),continuity:tuned.session_continuity.overall,handover_success:tuned.handover.success_rate,mean_access_latency_ms:tuned.mean_access_latency_ms,served_mbps:tuned.served_mbps},improvement:{score:bestScore-score(base),objective_kpi_delta:kpi(tuned)-kpi(base),continuity_pp:100*(tuned.session_continuity.overall-base.session_continuity.overall)},history,limitations:['Browser aggregate coordinate search; one pass is not a global optimum.']};
  }

  async function execute(kind) {
    const c=browserConfig();
    if(kind==='preview'||kind==='all') renderPreview(calculatedPreview(c));
    if(kind==='compare'||kind==='all') render(buildComparison(c));
    if(kind==='optimize'||kind==='all') renderOptimization(optimizer(c,document.getElementById('objective').value));
    document.getElementById('health').textContent='Browser JS engine · calculated locally';
  }

  function install() {
    buildAdvancedRows();
    document.querySelectorAll('#controls input, #controls select, #advancedCfg input, #advancedCfg select').forEach(el=>el.disabled=false);
    document.getElementById('tierReset').style.display='inline';
    document.getElementById('datasetMeta').textContent=`Reference ${P.metadata.model_version} · generated ${P.metadata.generated_utc} · config ${P.metadata.config_id}. Browser edits create new local results.`;
    document.getElementById('preview').onclick=()=>withButton(document.getElementById('preview'),'Calculating…',()=>execute('preview'),{done:'Capacity calculated',flashIds:['capPanel']});
    document.getElementById('run').onclick=()=>withButton(document.getElementById('run'),'Training locally…',()=>execute('compare'),{done:'Comparison complete',flashIds:['statusPanel','kpiPanel','validationPanel']});
    document.getElementById('optimize').onclick=()=>withButton(document.getElementById('optimize'),'Searching locally…',()=>execute('optimize'),{done:'Optimizer complete',flashIds:['optPanel']});
    document.getElementById('runAll').onclick=()=>withButton(document.getElementById('runAll'),'Calculating all…',()=>execute('all'),{done:'All JS calculations complete',flashIds:['capPanel','statusPanel','kpiPanel','optPanel']});
    document.getElementById('scenario').onchange=()=>{const f=SCENARIOS.find(s=>s.id===document.getElementById('scenario').value);document.getElementById('scenarioNote').textContent=f?f.description:'';execute('all');};
    document.getElementById('objective').onchange=()=>execute('optimize');
    document.getElementById('tierReset').onclick=()=>{buildTierRows();buildAdvancedRows();execute('all');};
    setTimeout(()=>execute('all'),180);
  }
  install();
})();
