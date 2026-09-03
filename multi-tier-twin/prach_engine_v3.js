/* mIoT PRACH overload engine for Multi-Tier Digital Twin v3.
 * Aggregate RAO simulator: arrivals, ACB, preamble occupancy, collisions,
 * retry backoff, retry exhaustion, and lightweight learned access control. */
(() => {
  'use strict';
  const ACTIONS = [
    {p:.01,b:1,w:18},{p:.02,b:1,w:16},{p:.05,b:2,w:14},
    {p:.10,b:3,w:12},{p:.20,b:4,w:10},{p:.40,b:3,w:7},{p:1,b:1,w:2}
  ];
  const clamp=(x,a=0,b=1)=>Math.max(a,Math.min(b,x));
  const rng=seed=>{let s=seed>>>0;return()=>{s=(Math.imul(1664525,s)+1013904223)>>>0;return s/4294967296;};};
  const normal=r=>Math.sqrt(-2*Math.log(Math.max(r(),1e-9)))*Math.cos(2*Math.PI*r());
  const poisson=(l,r)=>l>30?Math.max(0,Math.round(l+Math.sqrt(l)*normal(r))):(()=>{let p=1,k=0,L=Math.exp(-l);do{k++;p*=r();}while(p>L);return k-1;})();
  const binomial=(n,p,r)=>clamp(Math.round(n*p+Math.sqrt(Math.max(0,n*p*(1-p)))*normal(r)),0,n);
  const dot=(a,b)=>a.reduce((z,x,i)=>z+x*b[i],0);
  const softmax=x=>{const m=Math.max(...x),e=x.map(v=>Math.exp(clamp(v-m,-30,30))),z=e.reduce((a,b)=>a+b,0);return e.map(v=>v/z);};
  const choose=(p,r)=>{let u=r();for(let i=0;i<p.length;i++){u-=p[i];if(u<=0)return i;}return p.length-1;};
  const percentile=(a,p)=>{if(!a.length)return 0;const x=[...a].sort((q,w)=>q-w);return x[Math.min(x.length-1,Math.ceil(p*x.length)-1)];};

  function stormWeights(c){const x=[];for(let i=0;i<c.duration;i++){const u=(i+.5)/c.duration;x.push(c.profile==='moderate'?1:Math.pow(u,2)*Math.pow(1-u,c.profile==='extreme'?5:3));}const z=x.reduce((a,b)=>a+b,0);return x.map(v=>v/z);}
  function arrivals(c,seed){const out=Array(c.horizon).fill(0),r=rng(seed),w=stormWeights(c);let left=c.devices;for(let i=0;i<c.duration;i++){const n=i===c.duration-1?left:Math.min(left,Math.round(c.devices*w[i]));out[c.start+i]+=n;left-=n;}for(let t=0;t<c.horizon;t++)out[t]+=poisson(c.background,r);return out;}
  function features(backlog,lastCollision,lastIdle,t,c){return[clamp(backlog/Math.max(1,c.devices),0,2),lastCollision,lastIdle,t/c.horizon,clamp(backlog/Math.max(1,c.preambles*8),0,2),t>=c.start&&t<c.start+c.duration?1:0,1];}
  function makeModel(seed){const r=rng(seed);return{w:ACTIONS.map(()=>Array(7).fill(0).map(()=>normal(r)*.03)),baseline:0};}
  function learnedAction(model,s,kind,r,training,episode,c){const scores=model.w.map(w=>dot(w,s));if(kind==='ppo'){const p=softmax(scores);return{a:training?choose(p,r):p.indexOf(Math.max(...p)),p};}const eps=training?Math.max(.05,.6*(1-episode/Math.max(1,c.episodes))):0;return{a:r()<eps?Math.floor(r()*ACTIONS.length):scores.indexOf(Math.max(...scores)),p:null};}

  function simulate(kind,c,seed,model=null,training=false,episode=0){
    const r=rng(seed),schedule=arrivals(c,seed),q=Array.from({length:c.horizon+82},()=>[]),trace=[],delays=[];
    let backlog=0,successes=0,failures=0,retries=0,collisionTx=0,admittedTx=0,lastCollision=0,lastIdle=1,peak=0,clearance=c.horizon;
    let ctrl={p:kind==='static'?c.fixedAcb:1,b:c.fixedBarring,w:c.fixedBackoff};
    for(let t=0;t<c.horizon;t++){
      const n=schedule[t];if(n){q[t].push({n,born:t,attempt:0});backlog+=n;}
      const s=features(backlog,lastCollision,lastIdle,t,c);let selected=null;
      if(kind==='demand_follow'){ctrl.p=clamp(c.demandTarget*c.preambles/Math.max(1,backlog),.005,1);ctrl.w=c.fixedBackoff;}
      if(kind==='rule_based'){ctrl.p=clamp(.55*c.preambles/Math.max(1,backlog),.005,1);if(lastCollision>c.ruleHigh)ctrl.w=clamp(ctrl.w+3,1,40);else if(lastCollision<c.ruleLow)ctrl.w=clamp(ctrl.w-1,1,40);}
      if(kind==='ppo'||kind==='dqn'){selected=learnedAction(model,s,kind,r,training,episode,c);ctrl={...ACTIONS[selected.a]};}
      const due=q[t];let admitted=[];
      for(const cohort of due){const a=binomial(cohort.n,ctrl.p,r),barred=cohort.n-a;if(barred){const at=Math.min(q.length-1,t+1+Math.floor(r()*Math.max(1,ctrl.b)));q[at].push({...cohort,n:barred});}if(a)admitted.push({...cohort,n:a});}
      const total=admitted.reduce((z,x)=>z+x.n,0);admittedTx+=total;
      const expected=total?total*Math.exp(-(total-1)/Math.max(1,c.preambles)):0;
      const stepSuccess=clamp(Math.round(expected+Math.sqrt(Math.max(1,expected*.2))*normal(r)),0,total);let actualStepSuccess=0;
      for(let i=0;i<admitted.length;i++){const cohort=admitted[i],succ=Math.min(cohort.n,Math.round(stepSuccess*cohort.n/Math.max(1,total)));actualStepSuccess+=succ;const coll=cohort.n-succ;if(succ){successes+=succ;backlog-=succ;for(let j=0;j<succ;j++)delays.push(t-cohort.born+1);}if(coll){collisionTx+=coll;retries+=coll;if(cohort.attempt+1>=c.maxAttempts){failures+=coll;backlog-=coll;}else{const at=Math.min(q.length-1,t+1+Math.floor(r()*Math.max(1,ctrl.w)));q[at].push({n:coll,born:cohort.born,attempt:cohort.attempt+1});}}}
      lastCollision=total?(total-actualStepSuccess)/total:0;lastIdle=clamp(Math.exp(-total/Math.max(1,c.preambles)));
      peak=Math.max(peak,backlog);if(t>=c.start+c.duration&&backlog===0&&clearance===c.horizon)clearance=t;
      const reward=actualStepSuccess/Math.max(1,c.preambles)-1.2*lastCollision-1.5*backlog/Math.max(1,c.devices)-2.5*failures/Math.max(1,c.devices);
      if(training&&selected){if(kind==='ppo'){model.baseline=.95*model.baseline+.05*reward;const adv=clamp(reward-model.baseline,-3,3);model.w.forEach((w,a)=>w.forEach((_,j)=>w[j]+=.025*adv*((a===selected.a?1:0)-selected.p[a])*s[j]));}else{const qv=dot(model.w[selected.a],s),err=clamp(reward-qv,-4,4);model.w[selected.a].forEach((_,j)=>model.w[selected.a][j]+=.018*err*s[j]);}}
      trace.push({rao:t,backlog,collision:lastCollision,admission:ctrl.p});
    }
    failures+=backlog;
    return{successRate:successes/Math.max(1,successes+failures),successes,failures,collisionRate:collisionTx/Math.max(1,admittedTx),retries,meanDelayMs:(delays.reduce((a,b)=>a+b,0)/Math.max(1,delays.length))*1000/c.raosPerSec,p95DelayMs:percentile(delays,.95)*1000/c.raosPerSec,peakBacklog:peak,clearanceRao:clearance,trace};
  }

  function train(kind,c,seed){const model=makeModel(seed^(kind==='ppo'?0x51f15e:0x77191));for(let ep=0;ep<c.episodes;ep++){const scale=.7+.6*((ep*37)%11)/10,tc={...c,devices:Math.round(c.devices*scale),seed:seed+ep*7919};simulate(kind,tc,tc.seed,model,true,ep);}return model;}
  function compare(c,seed){const ppo=train('ppo',c,seed),dqn=train('dqn',c,seed),out={};for(const k of ['static','demand_follow','rule_based','ppo','dqn'])out[k]=simulate(k,c,seed,k==='ppo'?ppo:k==='dqn'?dqn:null);return out;}
  window.PrachV2={compare};
})();
