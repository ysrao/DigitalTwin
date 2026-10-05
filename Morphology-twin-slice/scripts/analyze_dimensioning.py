import json,sys,numpy as np
from scipy.stats import wilcoxon
rng=np.random.default_rng(0)
def load(paths):
    R=[]
    for p in paths: R+=json.load(open(p))
    return R
def earn(r,cu=50.0,ce=5.0):
    e=0.030*r['served_embb']-ce*r['viol_embb']; u=120-cu*r['viol_urllc']; m=0.002*r['served_miot']-1*r['viol_miot']
    return e,u,m
def boot(x,B=10000):
    x=np.asarray(x,float); idx=rng.integers(0,len(x),(B,len(x))); m=x[idx].mean(1); return np.percentile(m,[2.5,97.5])
def boot_ratio(a,b,B=10000):
    a=np.asarray(a,float); b=np.asarray(b,float); idx=rng.integers(0,len(a),(B,len(a)))
    r=a[idx].mean(1)/np.where(b[idx].mean(1)==0,np.nan,b[idx].mean(1)); return np.nanpercentile(r,[2.5,97.5])
def summary(R,label):
    out={"label":label,"n":len(R)}
    key=lambda r:(r['layout'],r['seed'])
    S={}
    for r in R: S.setdefault((r['embb_min'],r['urllc_min']),{})[key(r)]=r
    cells={}
    for k,v in sorted(S.items()):
        vs=list(v.values())
        cells[f"e{int(k[0]*100)}/u{int(k[1]*100)}"]={m:[float(np.mean([x[m] for x in vs])),*map(float,boot([x[m] for x in vs]))] for m in ['viol_urllc','viol_embb','violations','viol_miot','eff_kb_per_prb','served_kb','prb_util']}
        cells[f"e{int(k[0]*100)}/u{int(k[1]*100)}"]['n']=len(vs)
    out['cells']=cells
    a,b=S[(0.3,0.1)],S[(0.3,0.3)]; ks=sorted(a)
    du=[a[k]['viol_urllc']-b[k]['viol_urllc'] for k in ks]; de=[b[k]['viol_embb']-a[k]['viol_embb'] for k in ks]
    out['urllc10to30_at_e30']={"urllc_saved":float(np.mean(du)),"embb_cost":float(np.mean(de)),
        "p_urllc":float(wilcoxon(du).pvalue) if any(du) else None,"p_embb":float(wilcoxon(de).pvalue) if any(de) else None,
        "exch_ratio_of_means":float(np.mean(de)/np.mean(du)) if np.mean(du)!=0 else None,
        "exch_CI":list(map(float,boot_ratio(de,du))) if np.mean(du)!=0 else None,
        "pairs_urllc_changed":int(sum(1 for x in du if x!=0)),
        "mean_per_pair_ratio":float(np.mean([e/u for e,u in zip(de,du) if u!=0])) if any(du) else None}
    a,b=S[(0.3,0.1)],S[(0.6,0.1)]
    d=[b[k]['violations']-a[k]['violations'] for k in ks]
    out['embb30to60_at_u10']={"total_a":float(np.mean([a[k]['violations'] for k in ks])),"total_b":float(np.mean([b[k]['violations'] for k in ks])),
        "diff":float(np.mean(d)),"median":float(np.median(d)),"frac_improve":float(np.mean([x<0 for x in d])),"p":float(wilcoxon(d).pvalue) if any(d) else None,"n":len(d)}
    # earnings by URLLC floor (lambda=10), and lambda sweep
    E={}
    for u in (0.1,0.2,0.3):
        rs=[r for r in R if r['urllc_min']==u]; ee=np.array([earn(r) for r in rs])
        E[f"u{int(u*100)}"]={"embb":float(ee[:,0].mean()),"urllc":float(ee[:,1].mean()),"miot":float(ee[:,2].mean()),"total":float(ee.sum(1).mean()),"sd":float(ee.sum(1).std(ddof=1)),"n":len(rs)}
    out['earnings']=E
    lam=np.arange(1,50.01,0.1); best=[]
    for L in lam:
        t={u:np.mean([sum(earn(r,cu=5*L)) for r in R if r['urllc_min']==u]) for u in (0.1,0.2,0.3)}
        best.append(max(t,key=t.get))
    sw=[(round(float(lam[i]),1),best[i]) for i in range(len(lam)) if i==0 or best[i]!=best[i-1]]
    out['lambda_switch']=sw
    # paired earnings diffs at lambda=10
    P={}
    for u in (0.1,0.2,0.3):
        for r in R:
            if r['urllc_min']==u: P.setdefault(u,{})[(r['layout'],r['seed'],r['embb_min'])]=sum(earn(r))
    for x,y in ((0.1,0.2),(0.2,0.3)):
        kk=sorted(P[x]); d=[P[x][k]-P[y][k] for k in kk]
        out[f"earn_u{int(x*100)}_minus_u{int(y*100)}"]={"mean":float(np.mean(d)),"p":float(wilcoxon(d).pvalue)}
    return out
if __name__=="__main__":
    print(json.dumps(summary(load(sys.argv[2:]),sys.argv[1]),indent=1))
