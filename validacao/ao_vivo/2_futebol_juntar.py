import os
BACKTEST = os.environ.get("BACKTEST_DIR", "data/backtest")  # CSVs de "python -m analise.cli backtest"
import csv, json, re, sys, numpy as np

from analise.aovivo import similarity
from scipy.optimize import fsolve
from scipy.stats import poisson
from datetime import date
E=json.load(open('soccer.json'))
by={}
for e in E:
    d=date.fromisoformat(e['date'][:10]); by.setdefault(e['league'],[]).append((d,e))
def inv(ph,pa):
    g=np.arange(15)
    def f(x):
        l,m=np.exp(x); M=np.outer(poisson.pmf(g,l),poisson.pmf(g,m))
        return [np.tril(M,-1).sum()-ph, np.triu(M,1).sum()-pa]
    x=fsolve(f,[np.log(1.4),np.log(1.1)]); return [float(v) for v in np.exp(x)]
def mn(dc):
    m=re.match(r"(\d+)'(?:\+(\d+)')?",dc); b=int(m.group(1)); s=int(m.group(2)) if m.group(2) else 0
    return b,s
out=[]
for k in ["POR","ESP","ENG","FRA","ITA","GER","USA","BRA"]:
    for r in csv.DictReader(open(f"{BACKTEST}/{k}.csv")):
        if r["date"]<"2023-07-01" or not r["model_home"]: continue
        d=date.fromisoformat(r["date"]); best=None
        for d2,e in by.get(k,[]):
            if abs((d2-d).days)>1: continue
            s=min(similarity(r['home'],e['home']),similarity(r['away'],e['away']))
            if s>=0.5 and (best is None or s>best[0]): best=(s,e)
        if not best: continue
        e=best[1]
        if e['status']!='STATUS_FULL_TIME': continue
        ev=[]
        for x in e['det']:
            if x['so']: continue
            b,s=mn(x['dc'])
            if b>90 or (b>45 and b<=45 and False): pass
            half=1 if (b<=45) else 2
            if b>90: continue  # (prolongamento)
            side='h' if x['team']==e['hid'] else 'a'
            if x['goal']: ev.append(('g',half,b,s,side))
            elif x['red']: ev.append(('r',half,b,s,side))
        o=int(r['outcome'])
        m=re.match(r"90'\+(\d+)'",e['dclock'] or ''); L2=int(m.group(1)) if m else 0
        out.append({"league":k,"date":r["date"],"lam":inv(float(r['model_home']),float(r['model_away'])),
          "p":[float(r['model_home']),float(r['model_draw']),float(r['model_away'])],"hs":e['hs'],"as":e['as'],"L2":L2,"ev":ev})
json.dump(out,open("games.json","w")); print(len(out))
