"""Teste das percentagens ao vivo com jogos passados (usa as funções de aovivo_modelo.py).

Dados (gerados pelos scripts desta pasta, a partir da ESPN e do backtest do modelo):
  games.json        futebol: 8 662 jogos com o minuto de cada golo e expulsão e as
                    probabilidades de antes do jogo do backtest (só jogos anteriores)
  NBA.pkl, NHL.pkl, NHL_wf.pkl, tennis_matches.pkl, tennis_wf.pkl
Uso: python validar.py PASTA_DOS_DADOS
"""
import json, sys, collections
import numpy as np, pandas as pd
from scipy.stats import norm
from analise import aovivo_modelo as am
from analise.models import tennis as tm

D = sys.argv[1]
def ll(p, y):
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9); y = np.asarray(y, float)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
def calib(p, y, bins=((0, .1), (.1, .3), (.3, .5), (.5, .7), (.7, .9), (.9, 1.01))):
    p, y = np.asarray(p), np.asarray(y, float)
    return [(f"{lo:.0%}-{min(hi,1):.0%}", float(p[(p >= lo) & (p < hi)].mean()), float(y[(p >= lo) & (p < hi)].mean()), int(((p >= lo) & (p < hi)).sum()))
            for lo, hi in bins if ((p >= lo) & (p < hi)).any()]
out = {}

# ---------- futebol ----------
G = [g for g in json.load(open(f"{D}/games.json")) if g["date"] >= "2025-07-01"]
CHECK = [("15'", 1, 15, None), ("30'", 1, 30, None), ("Intervalo", 2, 45, None), ("60'", 2, 60, None),
         ("75'", 2, 75, None), ("85'", 2, 85, None)]
def state(g, half, minute):
    gh = ga = rh = ra = 0
    for kind, h, m, s, side in g["ev"]:
        if minute == 45 and half == 2:  # intervalo: tudo da 1.ª parte
            before = h == 1
        else:
            before = h < half or (h == half and s == 0 and m <= minute)
        if not before: continue
        if kind == "g": gh, ga = (gh + 1, ga) if side == "h" else (gh, ga + 1)
        else: rh, ra = (rh + 1, ra) if side == "h" else (rh, ra + 1)
    return gh, ga, rh, ra
rows = []; pre = []
for g in G:
    o = 0 if g["hs"] > g["as"] else (1 if g["hs"] == g["as"] else 2)
    pre.append(-np.log(g["p"][o]))
    for name, h, m, s in CHECK:
        gh, ga, rh, ra = state(g, h, m)
        r = am.soccer_live(*g["lam"], gh, ga, rh, ra, h, m, s)
        p = [r["casa"], r["empate"], r["fora"]]
        rows.append((name, p, o, r["mais_2_5"], g["hs"] + g["as"] > 2.5))
soc = {"jogos": len(G), "antes": float(np.mean(pre)), "momentos": []}
for name, *_ in CHECK:
    R = [r for r in rows if r[0] == name]
    soc["momentos"].append({"momento": name, "logloss_1x2": float(np.mean([-np.log(max(r[1][r[2]], 1e-9)) for r in R])),
                            "acerto": float(np.mean([int(np.argmax(r[1]) == r[2]) for r in R])),
                            "logloss_mais_2_5": ll([r[3] for r in R], [r[4] for r in R])})
ps = np.concatenate([r[1] for r in rows]); ys = np.concatenate([np.eye(3)[r[2]] for r in rows])
soc["calibracao_1x2"] = calib(ps, ys, ((0, .05), (.05, .15), (.15, .3), (.3, .5), (.5, .7), (.7, .85), (.85, .95), (.95, 1.01)))
soc["calibracao_mais_2_5"] = calib([r[3] for r in rows], [r[4] for r in rows])
out["futebol"] = soc

# ---------- basquetebol (NBA) ----------
from analise import basquete
df = pd.read_pickle(f"{D}/NBA.pkl"); wf = basquete.walk_forward(df)
q = df.dropna(subset=["home_q1", "home_q4"])
m = wf.merge(q[["date", "home", "away"] + [f"{s}_q{k}" for s in ("home", "away") for k in range(1, 5)]], on=["date", "home", "away"])
m = m[pd.to_datetime(m["date"]) >= pd.Timestamp("2025-07-01")]
y = m.home_win.astype(float).to_numpy()
bk = {"jogos": len(m), "antes": ll(m.p_home, y), "acerto_antes": float(((m.p_home > .5) == y).mean()), "momentos": []}
for k in (1, 2, 3):
    lead = sum(m[f"home_q{i}"] - m[f"away_q{i}"] for i in range(1, k + 1)).to_numpy()
    pts = sum(m[f"home_q{i}"] + m[f"away_q{i}"] for i in range(1, k + 1)).to_numpy()
    res = [am.basket_live(r.margin, r.total, r.sd_margin, r.sd_total, l, t, 1 - k / 4)
           for r, l, t in zip(m.itertuples(), lead, pts)]
    p = np.array([r["casa"] for r in res])
    line = np.floor(pts + (1 - k / 4) * m.total.to_numpy()) + 0.5
    po = np.array([1 - norm.cdf((L - r["total"]) / r["sd_total"]) for r, L in zip(res, line)])
    bk["momentos"].append({"momento": f"Fim do {k}.º quarto", "logloss": ll(p, y), "acerto": float(((p > .5) == y).mean()),
                           "calibracao": calib(p, y), "total_linha_logloss": ll(po, m.real_total.to_numpy() > line)})
out["basquetebol"] = bk

# ---------- hóquei (NHL) ----------
nh = pd.read_pickle(f"{D}/NHL.pkl"); wfh = pd.read_pickle(f"{D}/NHL_wf.pkl")
key = ["date", "home_goals", "away_goals", "home_reg", "away_reg"]
mh = wfh.merge(nh[key + ["home_p1", "away_p1", "home_p2", "away_p2"]].drop_duplicates(key, keep=False), on=key)
mh = mh[pd.to_datetime(mh["date"]) >= pd.Timestamp("2025-07-01")].dropna(subset=["home_p1", "home_p2"])
y = (mh.home_goals > mh.away_goals).astype(float).to_numpy()
hk = {"jogos": len(mh), "antes": ll(mh["home"], y), "momentos": []}
for k in (1, 2):
    p = np.array([am.hockey_live(r.lam_home, r.lam_away, r.home_p1 + (r.home_p2 if k == 2 else 0),
                                 r.away_p1 + (r.away_p2 if k == 2 else 0), k + 1, 20.0)["casa"] for r in mh.itertuples()])
    hk["momentos"].append({"momento": f"Fim do {k}.º período", "logloss": ll(p, y), "acerto": float(((p > .5) == y).mean()),
                           "calibracao": calib(p, y)})
out["hoquei"] = hk

# ---------- ténis ----------
tmx = pd.read_pickle(f"{D}/tennis_matches.pkl"); twf = pd.read_pickle(f"{D}/tennis_wf.pkl")
t = twf.merge(tmx[["id"] + [f"{s}_s{k}" for s in ("home", "away") for k in (1, 2, 3)]], on="id")
t = t[(t.status == "final") & (t.best_of == 3)].dropna(subset=["home_s1", "home_s2"])
tn = {"jogos": len(t), "momentos": []}
for k, name in ((1, "Fim do 1.º set"), (2, "Fim do 2.º set (1-1)")):
    P, Y, C = [], [], []
    for r in t.itertuples():
        sets = [(int(getattr(r, f"home_s{i}")), int(getattr(r, f"away_s{i}"))) for i in range(1, k + 1)]
        if k == 2 and (sets[0][0] > sets[0][1]) == (sets[1][0] > sets[1][1]): continue
        if any(not am_ok for am_ok in [all(0 <= x <= 7 for s in sets for x in s)]): continue
        base, sigma = tm.POINT_MODEL[(r.league, r.surface)]
        P.append(am.tennis_live(base, sigma, tm.serve_diff(r.casa, base, sigma), 3, sets, (0, 0))["casa"])
        Y.append(float(r.y == 1)); C.append(r.casa)
    tn["momentos"].append({"momento": name, "jogos": len(Y), "antes": ll(C, Y), "logloss": ll(P, Y),
                           "acerto": float(((np.array(P) > .5) == np.array(Y)).mean()), "calibracao": calib(P, Y)})
out["tenis"] = tn
json.dump(out, open("resultados.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(out, indent=1, ensure_ascii=False))
