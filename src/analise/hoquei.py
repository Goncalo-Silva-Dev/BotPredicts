"""Previsões e backtest de hóquei no gelo com o modelo dos 60 minutos e do fim de jogo."""
from datetime import date, timedelta

import numpy as np
import pandas as pd

from analise.models.hockey import HockeyModel


def _train(games: pd.DataFrame, as_of, window_days: int) -> pd.DataFrame:
    played = games.dropna(subset=["home_reg", "away_reg"])
    d = pd.to_datetime(played["date"])
    return played[(d < as_of) & (d >= as_of - timedelta(days=window_days))]


def predict_upcoming(games: pd.DataFrame, today: date | None = None, half_life_days=120.0,
                     window_days=540, min_team_games=5) -> pd.DataFrame:
    """Uma linha por jogo ainda por jogar a partir de hoje."""
    today = pd.Timestamp(today or date.today())
    train = _train(games, today, window_days)
    future = games[games["home_goals"].isna() & (pd.to_datetime(games["date"]) >= today)]
    if future.empty or len(train) < 30:
        return pd.DataFrame()
    counts = pd.concat([train["home"], train["away"]]).value_counts()
    model = HockeyModel(half_life_days=half_life_days).fit(train, as_of=today)
    rows = []
    for g in future.itertuples(index=False):
        row = {"league": g.league, "round": g.round, "date": g.date, "time": g.time,
               "home": g.home, "away": g.away}
        if min(counts.get(g.home, 0), counts.get(g.away, 0)) >= min_team_games:
            p = model.predict(g.home, g.away)
            periods, most = model.periods(g.home, g.away)
            row.update({
                "casa": p["home"], "fora": p["away"], "reg_casa": p["reg_home"],
                "reg_empate": p["reg_draw"], "reg_fora": p["reg_away"],
                "golos_casa": p["goals_home"], "golos_fora": p["goals_away"],
                "periodos": periods, "periodo_mais_golos": most,
                "mercados": model.markets(g.home, g.away),
            })
        rows.append(row)
    return pd.DataFrame(rows)


def walk_forward(games: pd.DataFrame, half_life_days=120.0, window_days=540, min_train=150,
                 min_team_games=5, with_periods=False) -> pd.DataFrame:
    """Para cada semana treina só com o passado e prevê os jogos dessa semana."""
    played = games.dropna(subset=["home_reg", "away_reg"]).copy()
    played["week"] = pd.to_datetime(played["date"]).dt.to_period("W-SUN").dt.start_time
    rows = []
    for week, wk in played.groupby("week"):
        train = _train(played, week, window_days)
        if len(train) < min_train:
            continue
        counts = pd.concat([train["home"], train["away"]]).value_counts()
        model = HockeyModel(half_life_days=half_life_days).fit(train, as_of=week)
        for g in wk.itertuples(index=False):
            if min(counts.get(g.home, 0), counts.get(g.away, 0)) < min_team_games:
                continue
            p, mk = model.predict(g.home, g.away), model.markets(g.home, g.away)
            row = {"league": g.league, "date": g.date, "home": g.home, "away": g.away,
                   **p, "markets": mk, "home_reg": g.home_reg, "away_reg": g.away_reg,
                   "home_goals": g.home_goals, "away_goals": g.away_goals, "ending": g.ending}
            if with_periods:
                row["periods"] = model.periods(g.home, g.away, n=8000)[0]
                row["real_periods"] = [(getattr(g, f"home_p{k}"), getattr(g, f"away_p{k}"))
                                       for k in (1, 2, 3)]
            rows.append(row)
    return pd.DataFrame(rows)


def summary(res: pd.DataFrame) -> dict:
    p = res["home"].clip(1e-6, 1 - 1e-6).to_numpy()
    y = (res["home_goals"] > res["away_goals"]).to_numpy(dtype=float)
    base = y.mean()  # referência: dar sempre a percentagem média de vitórias em casa
    reg = np.select([res["home_reg"] > res["away_reg"], res["home_reg"] == res["away_reg"]], [0, 1], 2)
    p3 = res[["reg_home", "reg_draw", "reg_away"]].to_numpy()
    freq = np.bincount(reg, minlength=3) / len(reg)
    return {
        "jogos": len(res),
        "acerto": float(((p > 0.5) == (y == 1)).mean()),
        "logloss": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()),
        "logloss_base": float(-(y * np.log(base) + (1 - y) * np.log(1 - base)).mean()),
        "logloss_60": float(-np.log(p3[np.arange(len(reg)), reg]).mean()),
        "logloss_60_base": float(-np.log(freq[reg]).mean()),
        "empates_previstos": float(res["reg_draw"].mean()), "empates_reais": float(freq[1]),
    }
