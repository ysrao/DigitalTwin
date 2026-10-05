import json,sys,numpy as np,matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
rng=np.random.default_rng(1)
def load(ps):
    R=[]; [R.extend(json.load(open(p))) for p in ps]; return R
sets=[("Urban (UMa, ISD 500 m)",load(["data/stage_e_dimensioning_fr3_20seed.json"]),"#2b6cb0"),
      ("Suburban, same traffic (RMa, ISD 1000 m)",load(["data/stage_e_dimensioning_suburban_w0.json","data/stage_e_dimensioning_suburban_w1.json"]),"#dd6b20"),
      ("Suburban, recalibrated load (RMa, ×8/9)",load(["data/stage_e_dimensioning_suburban_heavy_w0.json","data/stage_e_dimensioning_suburban_heavy_w1.json"]),"#38a169"),
      ("Rural, uniform load (RMa, ISD 1732 m, ×8/9)",load(["data/stage_e_dimensioning_suburban_rural_w0.json","data/stage_e_dimensioning_suburban_rural_w1.json"]),"#805ad5"),
      ("Rural, traffic profile (30/20/50 mix)",load(["data/stage_e_dimensioning_suburban_ruralprofile_w0.json","data/stage_e_dimensioning_suburban_ruralprofile_w1.json"]),"#b7791f")]
def ci(x):
    x=np.asarray(x,float); m=x[rng.integers(0,len(x),(5000,len(x)))].mean(1); return x.mean(),*np.percentile(m,[2.5,97.5])
fig,ax=plt.subplots(1,3,figsize=(13,4.2))
floors=[0.1,0.2,0.3]; w=0.16
for j,(lab,R,c) in enumerate(sets):
    for p,(key,ttl) in enumerate([("viol_urllc","(a) URLLC violations/run"),("viol_embb","(b) eMBB violations/run")]):
        st=[ci([r[key] for r in R if r['urllc_min']==u and r['embb_min']==0.3]) for u in floors]
        m=[s[0] for s in st]; lo=[s[0]-s[1] for s in st]; hi=[s[2]-s[0] for s in st]
        ax[p].bar(np.arange(3)+(j-2)*w,m,w,yerr=[lo,hi],color=c,capsize=3,label=lab)
        ax[p].set_title(ttl,fontsize=10)
    st=[ci([r['violations'] for r in R if r['urllc_min']==0.1 and r['embb_min']==e]) for e in (0.3,0.45,0.6)]
    m=[s[0] for s in st]; ax[2].bar(np.arange(3)+(j-2)*w,m,w,yerr=[[s[0]-s[1] for s in st],[s[2]-s[0] for s in st]],color=c,capsize=3)
    ax[2].set_title("(c) Total violations/run vs eMBB floor",fontsize=10)
for p in (0,1):
    ax[p].set_xticks(range(3)); ax[p].set_xticklabels(["10%","20%","30%"]); ax[p].set_xlabel("URLLC floor (eMBB floor 30%)")
ax[2].set_xticks(range(3)); ax[2].set_xticklabels(["30%","45%","60%"]); ax[2].set_xlabel("eMBB floor (URLLC floor 10%)")
for a in ax: a.set_ylabel("violations/run"); a.spines[['top','right']].set_visible(False)
fig.legend(*ax[0].get_legend_handles_labels(),loc="upper center",ncol=3,frameon=False,fontsize=8.5)
fig.tight_layout(rect=(0,0,1,0.84)); fig.savefig("paper/figs/morphology.png",dpi=200)
print("ok")
