import sys, pandas as pd
from datetime import date
from analise.ingest import basket, hockey
lg=basket.BASKET_LEAGUES
for k in ("NBA","NBL"):
    df=basket.fetch_espn(lg[k], date(2022,9,1), date(2026,11,9)); df.to_pickle(f"{k}.pkl"); print(k,len(df), df.filter(like="_q").notna().sum().min())
h=hockey.fetch_games(hockey.HOCKEY_LEAGUES["NHL"], date(2026,11,9), seasons=4)
h.to_pickle("NHL.pkl"); print("NHL",len(h), list(h.columns))
