"""Probabilidades do Dixon-Coles para os próximos jogos."""
from datetime import date, timedelta

import pandas as pd

from analise.models.dixon_coles import DixonColes, fair_probs


def predict(
    history: pd.DataFrame,
    fixtures: pd.DataFrame,
    as_of: date | None = None,
    half_life_days: float = 180.0,
    window_days: int = 730,
    min_team_matches: int = 5,
) -> pd.DataFrame:
    """Uma linha por jogo de fixtures (uma só liga), treinando com history até as_of.

    Jogos com equipas que têm menos de min_team_matches no treino ficam com as
    probabilidades vazias (p. ex. equipas acabadas de subir).
    """
    as_of = pd.Timestamp(as_of or date.today())
    train = history.dropna(subset=["home_goals", "away_goals"]).copy()
    train["date"] = pd.to_datetime(train["date"])
    train = train[(train["date"] < as_of) & (train["date"] >= as_of - timedelta(days=window_days))]
    counts = pd.concat([train["home"], train["away"]]).value_counts()
    model = DixonColes(half_life_days=half_life_days).fit(train, as_of=as_of)

    rows = []
    for g in fixtures.itertuples(index=False):
        row = {"league": g.league, "round": getattr(g, "round", None), "date": g.date,
               "time": g.time, "home": g.home, "away": g.away}
        if min(counts.get(g.home, 0), counts.get(g.away, 0)) >= min_team_matches:
            p = model.predict(g.home, g.away)
            row.update({
                "casa": p["home"], "empate": p["draw"], "fora": p["away"],
                "mais_2_5": p["over_2_5"], "ambas_marcam": p["btts"],
                "golos_casa": p["goals_home"], "golos_fora": p["goals_away"],
                "resultados": ", ".join(f"{s} ({q:.0%})" for s, q in model.top_scores(g.home, g.away)),
            })
        odds = (g.odds_home, g.odds_draw, g.odds_away)
        if all(pd.notna(o) and o > 1 for o in odds):
            m = fair_probs(*odds)
            row.update({"mercado_casa": m[0], "mercado_empate": m[1], "mercado_fora": m[2]})
        rows.append(row)
    return pd.DataFrame(rows)
