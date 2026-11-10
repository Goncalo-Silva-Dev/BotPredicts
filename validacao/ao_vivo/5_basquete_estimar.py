import pandas as pd, numpy as np, sys
from scipy.stats import norm
from analise import basquete
df=pd.read_pickle(sys.argv[1] if len(sys.argv)>1 else "NBA.pkl")
wf=basquete.walk_forward(df)
q=df.dropna(subset=["home_q1","home_q4"]).copy()
m=wf.merge(q[["date","home","away"]+[f"{s}_q{k}" for s in("home","away") for k in range(1,5)]],on=["date","home","away"])
m["season_test"]=pd.to_datetime(m["date"])>=pd.Timestamp("2025-07-01")
print(len(wf),len(m),m.season_test.sum())
rows=[]
for k in (1,2,3):
    lead=sum(m[f"home_q{i}"]-m[f"away_q{i}"] for i in range(1,k+1)); tot=sum(m[f"home_q{i}"]+m[f"away_q{i}"] for i in range(1,k+1))
    m[f"lead{k}"]=lead; m[f"tot{k}"]=tot
fits={}
tr=m[~m.season_test]
for k in (1,2,3):
    tau=1-k/4
    lead,tot=tr[f"lead{k}"],tr[f"tot{k}"]
    remM=tr.real_margin-lead; remT=tr.real_total-tot
    X=np.c_[tr.margin,lead]; a,b=np.linalg.lstsq(X,remM,rcond=None)[0]; s=np.std(remM-X@[a,b])/tr.sd_margin.mean()
    dev=tot-(1-tau)*tr.total
    X2=np.c_[tr.total,dev]; c,d=np.linalg.lstsq(X2,remT,rcond=None)[0]; u=np.std(remT-X2@[c,d])/tr.sd_total.mean()
    fits[tau]=(a,b,s,c,d,u); print(f"tau={tau}: margem a={a:.3f} b={b:.3f} sd={s:.3f} | total c={c:.3f} d={d:.3f} sd={u:.3f}")
import json; json.dump({str(k):v for k,v in fits.items()},open(sys.argv[1].replace('.pkl','')+"_fits.json" if len(sys.argv)>1 else "NBA_fits.json","w"))
te=m[m.season_test]
def ll(p,y): p=np.clip(p,1e-6,1-1e-6); return -(y*np.log(p)+(1-y)*np.log(1-p)).mean()
y=te.home_win.astype(float)
print("antes do jogo: log loss %.3f acerto %.3f"%(ll(te.p_home,y),((te.p_home>.5)==y).mean()))
for k in (1,2,3):
    tau=1-k/4; a,b,s,c,d,u=fits[tau]
    lead,tot=te[f"lead{k}"],te[f"tot{k}"]
    mu=a*te.margin+b*lead+lead; sd=s*te.sd_margin
    p=1-norm.cdf((0-mu)/sd)  # (empate -> metade)
    naive_mu=lead+tau*te.margin; naive_sd=np.sqrt(tau)*te.sd_margin; pn=1-norm.cdf(-naive_mu/naive_sd)
    T=tot+c*te.total+d*(tot-(1-tau)*te.total); line=np.floor(tot+tau*te.total)+0.5
    po=1-norm.cdf((line-T)/(u*te.sd_total)); yo=(te.real_total>line).astype(float)
    pon=1-norm.cdf((line-(tot+tau*te.total))/(np.sqrt(tau)*te.sd_total))
    print(f"fim do {k}.º quarto: vitória log loss simples {ll(pn,y):.3f} afinado {ll(p,y):.3f} acerto {((p>.5)==y).mean():.3f} | total (linha ao meio) simples {ll(pon,yo):.3f} afinado {ll(po,yo):.3f}")
    for lo,hi in [(0,.1),(.1,.3),(.3,.5),(.5,.7),(.7,.9),(.9,1.01)]:
        mm=(p>=lo)&(p<hi); print(f"    {lo:.1f}-{hi:.1f}: previsto {p[mm].mean():.3f} aconteceu {y[mm].mean():.3f} n={mm.sum()}")
