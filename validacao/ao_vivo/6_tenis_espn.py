from datetime import date
from analise.ingest import tennis
from analise import tenis
m=tennis.fetch_matches(date(2026,11,10), years=4); m.to_pickle("tennis_matches.pkl")
res=tenis.walk_forward(m, start="2025-01-01", with_markets=False); res.to_pickle("tennis_wf.pkl"); print(len(m),len(res))
