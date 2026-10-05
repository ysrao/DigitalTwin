"""v2.0 figures (antenna-fixed results). Writes paper/figs/fig1..fig5.
Colours: URLLC orange, eMBB blue, mIoT green, total dark grey."""
import json, glob, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
rng = np.random.default_rng(1)
C = {"viol_urllc": "#dd6b20", "viol_embb": "#2b6cb0", "viol_miot": "#38a169", "violations": "#4a5568"}
def load(exp): R=[]; [R.extend(json.load(open(p))) for p in sorted(glob.glob(f"data/v2_{exp}_w*.json"))]; return R
def ci(x):
    x=np.asarray(x,float); m=x[rng.integers(0,len(x),(5000,len(x)))].mean(1); return x.mean(), x.mean()-np.percentile(m,2.5), np.percentile(m,97.5)-x.mean()
def style(a): a.spines[['top','right']].set_visible(False)
GRID=[(e,u) for e in (0.30,0.45,0.60) for u in (0.10,0.20,0.30)]
U=load("urban")
# Fig 1: four panels
fig,ax=plt.subplots(1,4,figsize=(15,3.6))
for a,(k,t) in zip(ax,[("viol_urllc","(a) URLLC"),("viol_embb","(b) eMBB"),("viol_miot","(c) mIoT"),("violations","(d) Total")]):
    st=[ci([r[k] for r in U if (r['embb_min'],r['urllc_min'])==g]) for g in GRID]
    a.bar(range(9),[s[0] for s in st],yerr=[[s[1] for s in st],[s[2] for s in st]],color=C[k],capsize=2)
    a.set_title(t+" violations/run",fontsize=10); a.set_xticks(range(9)); a.set_xticklabels([f"e{int(e*100)}/u{int(u*100)}" for e,u in GRID],rotation=45,fontsize=7)
    a.set_ylabel("violations/run"); style(a)
    if k=="viol_miot" and max(s[0] for s in st)==0: a.set_ylim(0,1); a.text(4,0.5,"0 in all runs",ha="center",fontsize=9,color="#38a169")
fig.tight_layout(); fig.savefig("paper/figs/fig1_dimensioning.png",dpi=200); plt.close(fig)
# Fig 2: per-slice violations and efficiency vs URLLC floor (urban)
fig,ax=plt.subplots(1,2,figsize=(11,3.6)); fl=[0.1,0.2,0.3]; w=0.27
for j,(s,k) in enumerate([("eMBB","embb"),("URLLC","urllc"),("mIoT","miot")]):
    v=[np.mean([r[f"viol_{k}"] for r in U if r['urllc_min']==u]) for u in fl]
    e=[np.mean([r[f"served_{k}"]/max(1e-9,r[f"prb_{k}"]) for r in U if r['urllc_min']==u and r[f"prb_{k}"]>0]) for u in fl]
    ax[0].bar(np.arange(3)+(j-1)*w,v,w,color=C[f"viol_{k}"],label=s); ax[1].bar(np.arange(3)+(j-1)*w,e,w,color=C[f"viol_{k}"],label=s)
for a,t,y in ((ax[0],"(a) Violations per slice","violations/run"),(ax[1],"(b) Per-slice efficiency","kb / PRB")):
    a.set_xticks(range(3)); a.set_xticklabels(["10%","20%","30%"]); a.set_xlabel("URLLC floor"); a.set_title(t,fontsize=10); a.set_ylabel(y); a.legend(fontsize=8,frameon=False); style(a)
fig.tight_layout(); fig.savefig("paper/figs/fig2_per_slice.png",dpi=200); plt.close(fig)
# Fig 3: PRACH (v2 corrected channel, 60-480 UEs; v1 full-range probe for reference)
P=[]; [P.extend(json.load(open(f))) for f in sorted(glob.glob("data/v2_prach_w*.json"))]
V=json.load(open("data/v1_superseded/prach_extended.json")) if glob.glob("data/v1_superseded/prach_extended.json") else json.load(open("data/prach_extended.json"))
nsV=sorted(set(r['n_ues'] for r in V)); x={n:i for i,n in enumerate(nsV)}
ns=sorted(set(r['n_ues'] for r in P)); st=[ci([100*r['coll_rate'] for r in P if r['n_ues']==n]) for n in ns]
fig,a=plt.subplots(figsize=(9,3.4))
a.bar([x[n] for n in ns],[s[0] for s in st],yerr=[[s[1] for s in st],[s[2] for s in st]],color="#38a169",capsize=2,label="v2.0 corrected channel (10 seeds)")
a.plot(range(len(nsV)),[100*np.mean([r['coll_rate'] for r in V if r['n_ues']==n]) for n in nsV],"o--",color="#a0aec0",ms=4,label="v1.x channel, superseded (reference)")
a.axhline(20,ls="--",color="#c53030",lw=1,label="20% congestion threshold"); a.set_xticks(range(len(nsV))); a.set_xticklabels(nsV,rotation=45,fontsize=8)
a.set_xlabel("Total UEs (30% mIoT)"); a.set_ylabel("PRACH collision rate (%)"); a.legend(fontsize=8,frameon=False,loc="upper left"); style(a)
fig.tight_layout(); fig.savefig("paper/figs/fig3_prach.png",dpi=200); plt.close(fig)
# Fig 4: earnings-optimal floor vs lambda (urban)
def earn(r,cu): return 0.030*r['served_embb']-5*r['viol_embb']+120-cu*r['viol_urllc']+0.002*r['served_miot']-r['viol_miot']
lam=np.arange(1,50.01,0.1); best=[]
for L in lam:
    t={u:np.mean([earn(r,5*L) for r in U if r['urllc_min']==u]) for u in fl}; best.append(max(t,key=t.get))
fig,a=plt.subplots(figsize=(9,2.6)); cols={0.1:"#90cdf4",0.2:"#f6ad55",0.3:"#fc8181"}
b=np.array(best); edges=[0]+[i for i in range(1,len(b)) if b[i]!=b[i-1]]+[len(b)]
seen=set()
for i0,i1 in zip(edges[:-1],edges[1:]):
    u=b[i0]; lab=None if u in seen else f"{int(u*100)}% floor optimal"; seen.add(u)
    a.axvspan(lam[i0],lam[i1-1]+0.1,color=cols[u],label=lab)
a.set_yticks([]); a.set_xlabel("λ = c_URLLC / c_eMBB"); a.set_xlim(1,50); a.legend(fontsize=8,frameon=False,loc="upper center",bbox_to_anchor=(0.5,-0.32),ncol=3); style(a)
fig.tight_layout(); fig.savefig("paper/figs/fig4_lambda.png",dpi=200,bbox_inches="tight"); plt.close(fig)
# Fig 5: morphologies
sets=[("Urban (UMa, ISD 500 m)",U,"#2b6cb0"),("Suburban (RMa, ISD 1000 m)",load("suburban"),"#dd6b20"),
      ("Rural, uniform load (RMa, ISD 1732 m)",load("rural_uniform"),"#805ad5"),("Rural, traffic profile (30/20/50 mix)",load("rural_profile"),"#b7791f")]
fig,ax=plt.subplots(1,3,figsize=(13,4)); w=0.2
for j,(lab,R,c) in enumerate(sets):
    for p,k in enumerate(["viol_urllc","viol_embb"]):
        st=[ci([r[k] for r in R if r['urllc_min']==u and r['embb_min']==0.3]) for u in fl]
        ax[p].bar(np.arange(3)+(j-1.5)*w,[s[0] for s in st],w,yerr=[[s[1] for s in st],[s[2] for s in st]],color=c,capsize=2,label=lab)
    st=[ci([r['violations'] for r in R if r['urllc_min']==0.1 and r['embb_min']==e]) for e in (0.3,0.45,0.6)]
    ax[2].bar(np.arange(3)+(j-1.5)*w,[s[0] for s in st],w,yerr=[[s[1] for s in st],[s[2] for s in st]],color=c,capsize=2)
for p,t in enumerate(["(a) URLLC violations/run","(b) eMBB violations/run"]):
    ax[p].set_title(t,fontsize=10); ax[p].set_xticks(range(3)); ax[p].set_xticklabels(["10%","20%","30%"]); ax[p].set_xlabel("URLLC floor (eMBB floor 30%)")
ax[2].set_title("(c) Total violations/run vs eMBB floor",fontsize=10); ax[2].set_xticks(range(3)); ax[2].set_xticklabels(["30%","45%","60%"]); ax[2].set_xlabel("eMBB floor (URLLC floor 10%)")
for a in ax: a.set_ylabel("violations/run"); style(a)
fig.legend(*ax[0].get_legend_handles_labels(),loc="upper center",ncol=4,frameon=False,fontsize=8.5)
fig.tight_layout(rect=(0,0,1,0.88)); fig.savefig("paper/figs/fig5_morphology.png",dpi=200); plt.close(fig)
print("figs ok")
