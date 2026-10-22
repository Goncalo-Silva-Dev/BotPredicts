"""Previsões e backtest de basquetebol com os ratings de ataque e defesa."""
from datetime import date, timedelta

import numpy as np
import pandas as pd

from analise.models.basket import BasketRatings


def _train(games: pd.DataFrame, as_of, window_days: int) -> pd.DataFrame:
    played = games.dropna(subset=["home_points", "away_points"])
    d = pd.to_datetime(played["date"])
    return played[(d < as_of) & (d >= as_of - timedelta(days=window_days))]


def predict_upcoming(games: pd.DataFrame, today: date | None = None, half_life_days=120.0,
                     window_days=540, min_team_games=5) -> pd.DataFrame:
    """Uma linha por jogo ainda por jogar a partir de hoje."""
    today = pd.Timestamp(today or date.today())
    train = _train(games, today, window_days)
    future = games[games["home_points"].isna() & (pd.to_datetime(games["date"]) >= today)]
    if future.empty or len(train) < 30:
        return pd.DataFrame()
    counts = pd.concat([train["home"], train["away"]]).value_counts()
    model = BasketRatings(half_life_days=half_life_days).fit(train, as_of=today)
    rows = []
    for g in future.itertuples(index=False):
        row = {"league": g.league, "round": g.round, "date": g.date, "time": g.time,
               "home": g.home, "away": g.away}
        if min(counts.get(g.home, 0), counts.get(g.away, 0)) >= min_team_games:
            p = model.predict(g.home, g.away)
            row.update({
                "casa": p["home"], "fora": p["away"], "pontos_casa": p["points_home"],
                "pontos_fora": p["points_away"], "diferenca": p["margin"], "total": p["total"],
                "total_min": p["total_low"], "total_max": p["total_high"],
            })
        rows.append(row)
    return pd.DataFrame(rows)


def walk_forward(games: pd.DataFrame, half_life_days=120.0, window_days=540, min_train=150,
                 min_team_games=5) -> pd.DataFrame:
    """Para cada semana treina só com o passado e prevê os jogos dessa semana."""
    played = games.dropna(subset=["home_points", "away_points"]).copy()
    played["week"] = pd.to_datetime(played["date"]).dt.to_period("W-SUN").dt.start_time
    rows = []
    for week, wk in played.groupby("week"):
        train = _train(played, week, window_days)
        if len(train) < min_train:
            continue
        counts = pd.concat([train["home"], train["away"]]).value_counts()
        model = BasketRatings(half_life_days=half_life_days).fit(train, as_of=week)
        for g in wk.itertuples(index=False):
            if min(counts.get(g.home, 0), counts.get(g.away, 0)) < min_team_games:
                continue
            p = model.predict(g.home, g.away)
            rows.append({"league": g.league, "date": g.date, "home": g.home, "away": g.away,
                         "p_home": p["home"], "margin": p["margin"], "total": p["total"],
                         "home_win": g.home_points > g.away_points,
                         "real_margin": g.home_points - g.away_points,
                         "real_total": g.home_points + g.away_points})
    return pd.DataFrame(rows)


def summary(res: pd.DataFrame) -> dict:
    p = res["p_home"].clip(1e-6, 1 - 1e-6).to_numpy()
    y = res["home_win"].to_numpy(dtype=float)
    base = y.mean()  # referência: dar sempre a percentagem média de vitórias em casa
    return {
        "jogos": len(res),
        "acerto": float(((p > 0.5) == (y == 1)).mean()),
        "logloss": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()),
        "logloss_base": float(-(y * np.log(base) + (1 - y) * np.log(1 - base)).mean()),
        "brier": float(((p - y) ** 2).mean()),
        "erro_diferenca": float((res["margin"] - res["real_margin"]).abs().mean()),
        "erro_total": float((res["total"] - res["real_total"]).abs().mean()),
    }
