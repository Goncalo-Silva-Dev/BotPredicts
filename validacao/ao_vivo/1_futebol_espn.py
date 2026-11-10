import os
BACKTEST = os.environ.get("BACKTEST_DIR", "data/backtest")  # CSVs de "python -m analise.cli backtest"
import csv, json, sys, time, requests
from datetime import date, timedelta
from concurrent.futures import ThreadPoolExecutor
SL={"POR":"por.1","ESP":"esp.1","ENG":"eng.1","FRA":"fra.1","ITA":"ita.1","GER":"ger.1","USA":"usa.1","BRA":"bra.1"}
jobs=set()
for k,s in SL.items():
    for r in csv.DictReader(open(f"{BACKTEST}/{k}.csv")):
        if r["date"]>="2023-07-01":
            d=date.fromisoformat(r["date"])
            jobs.add((k,d)); 
            if k in("USA","BRA"): jobs.add((k,d+timedelta(days=1)))
def get(job):
    k,d=job
    for i in range(4):
        try:
            r=requests.get(f"https://site.api.espn.com/apis/site/v2/sports/soccer/{SL[k]}/scoreboard",params={"dates":d.strftime("%Y%m%d")},timeout=30)
            r.raise_for_status(); out=[]
            for ev in r.json().get("events",[]):
                c=ev["competitions"][0]; st=ev["status"]
                if not st["type"].get("completed"): continue
                t={x["homeAway"]:x for x in c["competitors"]}
                out.append({"league":k,"id":ev["id"],"date":ev["date"],"home":t["home"]["team"]["displayName"],"away":t["away"]["team"]["displayName"],
                  "hid":t["home"]["team"]["id"],"hs":int(t["home"]["score"]),"as":int(t["away"]["score"]),"status":st["type"]["name"],"clock":st.get("clock"),"dclock":st.get("displayClock"),
                  "det":[{"t":x["type"]["text"],"c":x["clock"]["value"],"dc":x["clock"]["displayValue"],"team":x.get("team",{}).get("id"),"goal":x.get("scoringPlay"),"red":x.get("redCard"),"og":x.get("ownGoal"),"so":x.get("shootout")} for x in c.get("details",[])]})
            return out
        except Exception as e:
            time.sleep(2**i)
    print("fail",job,file=sys.stderr); return []
with ThreadPoolExecutor(10) as p: res=[e for l in p.map(get,sorted(jobs)) for e in l]
u={e["id"]:e for e in res}
json.dump(list(u.values()),open("soccer.json","w"))
print(len(jobs),len(u))
