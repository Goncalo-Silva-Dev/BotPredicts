"""Registo das previsões publicadas e comparação com os resultados finais.

Cada jogo fica guardado com a última previsão feita antes do início: depois da hora do jogo
a previsão nunca mais muda. Os ficheiros (um JSON por mês do jogo) vivem no ramo "historico"
do repositório, por isso cada gravação fica com data e hora no histórico do git.
"""
import glob
import json
import math
import os
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd

LISBON = ZoneInfo("Europe/Lisbon")
DAYS_AHEAD = 7  # só se guardam os jogos dos próximos 7 dias
GIVE_UP_DAYS = 14  # sem resultado ao fim de 14 dias (adiado, nome diferente): fica de fora


def kickoff(day, clock) -> datetime:
    """Início do jogo em UTC; sem hora conta a meia-noite (hora de Portugal) desse dia, para
    que só valham as previsões feitas na véspera ou antes."""
    d = pd.Timestamp(day).date()
    hh = mm = 0
    if isinstance(clock, str) and len(clock) >= 5 and clock[2] == ":":
        hh, mm = int(clock[:2]), int(clock[3:5])
    return datetime(d.year, d.month, d.day, hh, mm, tzinfo=LISBON).astimezone(timezone.utc)


def _r(x) -> float:
    return round(float(x), 4)


def football_probs(g) -> dict:
    p = {"1x2": [_r(g["casa"]), _r(g["empate"]), _r(g["fora"])],
         "golos": [_r(g["golos_casa"]), _r(g["golos_fora"])],
         "ambas": _r(g["ambas_marcam"])}
    for line in ("1.5", "2.5", "3.5"):
        if pd.notna(g.get(f"mais_{line}")):
            p[f"mais_{line}"] = _r(g[f"mais_{line}"])
    if pd.notna(g.get("dnb_casa")):
        p["dnb_casa"] = _r(g["dnb_casa"])
    if pd.notna(g.get("p1_casa")):
        p["intervalo"] = [_r(g["p1_casa"]), _r(g["p1_empate"]), _r(g["p1_fora"])]
    if pd.notna(g.get("mercado_casa")):
        p["mercado_1x2"] = [_r(g["mercado_casa"]), _r(g["mercado_empate"]), _r(g["mercado_fora"])]
    return p


def basket_probs(g) -> dict:
    mk = g["mercados"]
    hcp, tot = mk["handicap"][len(mk["handicap"]) // 2], mk["total"][len(mk["total"]) // 2]
    return {"vencedor": _r(g["casa"]), "pontos": [_r(g["pontos_casa"]), _r(g["pontos_fora"])],
            # handicap da casa: [linha, probabilidade de a diferença final ficar acima]
            "handicap": [float(hcp[0]), _r(hcp[1])], "total": [float(tot[0]), _r(tot[1])],
            "total_80": [[_r(g["total_min"]), _r(g["total_max"])], 0.8]}


def hockey_probs(g) -> dict:
    mk = g["mercados"]
    tot = mk["total"][len(mk["total"]) // 2]
    puck = next(p for side, line, p in mk["puck_line"] if side == "casa" and line == -1.5)
    return {"vencedor": _r(g["casa"]), "60min": [_r(g["reg_casa"]), _r(g["reg_empate"]), _r(g["reg_fora"])],
            "golos": [_r(g["golos_casa"]), _r(g["golos_fora"])],
            "prolongamento": _r(mk["prolongamento"]), "total": [float(tot[0]), _r(tot[1])],
            "puck_casa": [-1.5, _r(puck)], "ambas": _r(mk["ambas_marcam"])}


PROBS = {"futebol": football_probs, "basquetebol": basket_probs, "hoquei": hockey_probs}


def load(folder: str) -> dict:
    store = {}
    for path in sorted(glob.glob(os.path.join(folder, "previsoes", "*.json"))):
        with open(path, encoding="utf-8") as f:
            for rec in json.load(f):
                store[rec["id"]] = rec
    return store


def save(store: dict, folder: str) -> None:
    by_month = {}
    for rec in store.values():
        by_month.setdefault(rec["data"][:7], []).append(rec)
    os.makedirs(os.path.join(folder, "previsoes"), exist_ok=True)
    for month, recs in by_month.items():
        recs.sort(key=lambda r: (r["data"], r["hora"] or "", r["id"]))
        with open(os.path.join(folder, "previsoes", f"{month}.json"), "w", encoding="utf-8") as f:
            json.dump(recs, f, ensure_ascii=False, indent=1)
            f.write("\n")


def record(store: dict, preds: pd.DataFrame, sport: str, now: datetime) -> int:
    """Guarda (ou atualiza) a previsão dos jogos que ainda não começaram."""
    if preds is None or preds.empty:
        return 0
    n = 0
    for g in preds.to_dict("records"):
        if pd.isna(g.get("casa")):
            continue
        clock = g.get("time") if isinstance(g.get("time"), str) else None
        start = kickoff(g["date"], clock)
        if not now < start <= now + timedelta(days=DAYS_AHEAD):
            continue
        day = str(pd.Timestamp(g["date"]).date())
        key = "|".join([sport, g["league"], day, g["home"], g["away"]])
        old = store.get(key)
        if old and now >= kickoff(old["data"], old["hora"]):
            continue  # o jogo já começou na hora antiga: a previsão fica como estava
        store[key] = {"id": key, "desporto": sport, "liga": g["league"], "data": day, "hora": clock,
                      "casa": g["home"], "fora": g["away"],
                      "feita_em": now.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
                      "p": PROBS[sport](g), "estado": "pendente", "resultado": None}
        n += 1
    return n


def resolve(store: dict, results: pd.DataFrame, sport: str, now: datetime) -> int:
    """Junta o resultado final aos jogos guardados.

    results tem league, date, home, away e as colunas do resultado de cada desporto:
    futebol home_goals, away_goals (e ao intervalo home_goals_ht, away_goals_ht);
    basquetebol home_points, away_points; hóquei home_goals, away_goals (finais) e
    home_reg, away_reg (60 minutos). A data pode diferir 1 dia (fusos das fontes).
    """
    pending = [r for r in store.values() if r["desporto"] == sport and r["estado"] == "pendente"]
    if not pending:
        return 0
    index = {}
    if results is not None and not results.empty:
        for g in results.to_dict("records"):
            index.setdefault((g["league"], g["home"], g["away"]), []).append(g)
    n = 0
    for rec in pending:
        day = pd.Timestamp(rec["data"]).date()
        found = [g for g in index.get((rec["liga"], rec["casa"], rec["fora"]), [])
                 if abs((pd.Timestamp(g["date"]).date() - day).days) <= 1]
        res = _result(found[0], sport) if found else None
        if res:
            rec["estado"], rec["resultado"] = "terminado", res
            n += 1
        elif now > kickoff(rec["data"], rec["hora"]) + timedelta(days=GIVE_UP_DAYS):
            rec["estado"] = "sem resultado"
    return n


def _num(v):
    return None if v is None or pd.isna(v) else int(v)


def _result(g, sport) -> dict | None:
    if sport == "basquetebol":
        cols = {"casa": "home_points", "fora": "away_points"}
    elif sport == "hoquei":
        cols = {"casa": "home_goals", "fora": "away_goals", "casa_60": "home_reg", "fora_60": "away_reg"}
    else:
        cols = {"casa": "home_goals", "fora": "away_goals", "casa_int": "home_goals_ht",
                "fora_int": "away_goals_ht"}
    res = {k: _num(g.get(c)) for k, c in cols.items()}
    if res["casa"] is None or res["fora"] is None:
        return None
    return {k: v for k, v in res.items() if v is not None}


def _1x2(h, a) -> int:
    return 0 if h > a else (1 if h == a else 2)


# (nome, chave guardada, tipo, resultado): "1x2" tem 3 probabilidades e o resultado é o índice
# certo; "sim" é uma probabilidade (ou [linha, probabilidade]) e o resultado é verdadeiro/falso.
# Um resultado None quer dizer que a aposta é devolvida (p. ex. empate anula aposta com empate).
MARKETS = {
    "futebol": [
        ("Vencedor (1X2)", "1x2", "1x2", lambda r, _: _1x2(r["casa"], r["fora"])),
        ("Ao intervalo (1X2)", "intervalo", "1x2",
         lambda r, _: _1x2(r["casa_int"], r["fora_int"]) if "casa_int" in r else None),
        ("Mais de 1,5 golos", "mais_1.5", "sim", lambda r, _: r["casa"] + r["fora"] > 1.5),
        ("Mais de 2,5 golos", "mais_2.5", "sim", lambda r, _: r["casa"] + r["fora"] > 2.5),
        ("Mais de 3,5 golos", "mais_3.5", "sim", lambda r, _: r["casa"] + r["fora"] > 3.5),
        ("Ambas marcam", "ambas", "sim", lambda r, _: r["casa"] > 0 and r["fora"] > 0),
        ("Empate anula aposta (casa)", "dnb_casa", "sim",
         lambda r, _: None if r["casa"] == r["fora"] else r["casa"] > r["fora"]),
    ],
    "basquetebol": [
        ("Vencedor", "vencedor", "sim", lambda r, _: r["casa"] > r["fora"]),
        ("Handicap da casa (linha do modelo)", "handicap", "sim", lambda r, line: r["casa"] - r["fora"] > line),
        ("Total de pontos (linha do modelo)", "total", "sim", lambda r, line: r["casa"] + r["fora"] > line),
        ("Total dentro do intervalo de 80%", "total_80", "sim",
         lambda r, band: band[0] <= r["casa"] + r["fora"] <= band[1]),
    ],
    "hoquei": [
        ("Vencedor (com prolongamento)", "vencedor", "sim", lambda r, _: r["casa"] > r["fora"]),
        ("Resultado aos 60 minutos (1X2)", "60min", "1x2",
         lambda r, _: _1x2(r["casa_60"], r["fora_60"]) if "casa_60" in r else None),
        ("Vai a prolongamento", "prolongamento", "sim",
         lambda r, _: r["casa_60"] == r["fora_60"] if "casa_60" in r else None),
        ("Total de golos (linha do modelo)", "total", "sim", lambda r, line: r["casa"] + r["fora"] > line),
        ("Puck line -1,5 da casa", "puck_casa", "sim", lambda r, line: r["casa"] - r["fora"] > -line),
        ("Ambas marcam (60 minutos)", "ambas", "sim",
         lambda r, _: r["casa_60"] > 0 and r["fora_60"] > 0 if "casa_60" in r else None),
    ],
}
MAIN = {"futebol": "1x2", "basquetebol": "vencedor", "hoquei": "vencedor"}
EPS = 1e-6


def evaluate(store: dict) -> pd.DataFrame:
    """Uma linha por jogo terminado e mercado: acerto, confiança e log loss."""
    rows = []
    for rec in store.values():
        if rec["estado"] != "terminado":
            continue
        res = rec["resultado"]
        for name, key, kind, outcome in MARKETS[rec["desporto"]]:
            if key not in rec["p"]:
                continue
            value = rec["p"][key]
            line, p = (value[0], value[1]) if kind == "sim" and isinstance(value, list) else (None, value)
            y = outcome(res, line)
            if y is None:
                continue
            rows.append({"desporto": rec["desporto"], "liga": rec["liga"], "data": rec["data"],
                         "mercado": name, "chave": key, **_score(kind, p, y)})
        if "mercado_1x2" in rec["p"]:  # as casas de apostas nos mesmos jogos, para comparar
            y = _1x2(res["casa"], res["fora"])
            rows.append({"desporto": rec["desporto"], "liga": rec["liga"], "data": rec["data"],
                         "mercado": "Casas de apostas (1X2)", "chave": "mercado_1x2",
                         **_score("1x2", rec["p"]["mercado_1x2"], y)})
    cols = ["desporto", "liga", "data", "mercado", "chave", "acerto", "confianca", "logloss", "referencia"]
    return pd.DataFrame(rows, columns=cols)


def _score(kind, p, y) -> dict:
    if kind == "1x2":
        pick = max(range(3), key=lambda i: p[i])
        return {"acerto": float(pick == y), "confianca": p[pick],
                "logloss": -math.log(max(p[y], EPS)), "referencia": math.log(3)}
    y = bool(y)
    pick = p >= 0.5
    return {"acerto": float(pick == y), "confianca": p if pick else 1 - p,
            "logloss": -math.log(max(p if y else 1 - p, EPS)), "referencia": math.log(2)}


def table(ev: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    if ev.empty:
        return pd.DataFrame(columns=by + ["jogos", "acerto", "confianca", "logloss", "referencia"])
    return ev.groupby(by, sort=False).agg(
        jogos=("acerto", "size"), acerto=("acerto", "mean"), confianca=("confianca", "mean"),
        logloss=("logloss", "mean"), referencia=("referencia", "mean")).reset_index()


# "Jogos mais seguros": vitória do favorito com 65% ou mais; senão, dupla hipótese do favorito
# (favorito ou empate) com 80% ou mais. No backtest de 15 872 jogos acertaram 77% (15% dos jogos) e 85% (17%).
SAFE_WIN, SAFE_DOUBLE = 0.65, 0.80
# Só as ligas desse teste (têm remates à baliza); na MLS a vitória com 65% ou mais acertou só 57%
SAFE_LEAGUES = {"POR", "ESP", "ENG", "FRA", "ITA", "GER", "TUR", "NED", "BEL"}


def safe_pick(p1x2) -> tuple[str, int, float] | None:
    """("vitoria" ou "dupla", 0 casa / 2 fora, probabilidade) ou None se o jogo não é claro."""
    side = 0 if p1x2[0] >= p1x2[2] else 2
    if p1x2[side] < p1x2[1]:
        return None  # o mais provável é o empate
    if p1x2[side] >= SAFE_WIN:
        return "vitoria", side, float(p1x2[side])
    if p1x2[side] + p1x2[1] >= SAFE_DOUBLE:
        return "dupla", side, float(p1x2[side] + p1x2[1])
    return None


def safe_evaluate(store: dict) -> pd.DataFrame:
    """Jogos de futebol terminados que estavam nos "mais seguros", e se a escolha acertou."""
    rows = []
    for rec in store.values():
        if rec["desporto"] != "futebol" or rec["estado"] != "terminado" or rec["liga"] not in SAFE_LEAGUES:
            continue
        pick = safe_pick(rec["p"]["1x2"])
        if pick is None:
            continue
        kind, side, p = pick
        real = _1x2(rec["resultado"]["casa"], rec["resultado"]["fora"])
        ok = real == side if kind == "vitoria" else real != 2 - side
        rows.append({"tipo": kind, "liga": rec["liga"], "data": rec["data"], "acerto": float(ok),
                     "confianca": p})
    return pd.DataFrame(rows, columns=["tipo", "liga", "data", "acerto", "confianca"])


BUCKETS = [(0.0, 0.5, "Menos de 50%"), (0.5, 0.6, "50% a 60%"), (0.6, 0.7, "60% a 70%"),
           (0.7, 0.8, "70% a 80%"), (0.8, 0.9, "80% a 90%"), (0.9, 1.01, "90% ou mais")]


def calibration(ev: pd.DataFrame) -> pd.DataFrame:
    """Previsões do modelo agrupadas pela confiança: quem diz 70% deve acertar 7 em 10."""
    ev = ev[ev["chave"] != "mercado_1x2"]
    rows = []
    for lo, hi, label in BUCKETS:
        part = ev[(ev["confianca"] >= lo) & (ev["confianca"] < hi)]
        if len(part):
            rows.append({"grupo": label, "previsoes": len(part), "confianca": part["confianca"].mean(),
                         "acerto": part["acerto"].mean()})
    return pd.DataFrame(rows, columns=["grupo", "previsoes", "confianca", "acerto"])
