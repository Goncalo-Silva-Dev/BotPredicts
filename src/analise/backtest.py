"""Backtest walk-forward do Dixon-Coles contra as odds de fecho sem margem.

Os jogos são agrupados por semana (aproximação à jornada, que os CSVs não
trazem). Para cada semana o modelo é treinado só com jogos anteriores a ela e
prevê os jogos dessa semana; a referência é o mercado de fecho (Pinnacle quando
existe, senão a média) com a margem retirada.
"""
from datetime import timedelta

import numpy as np
import pandas as pd

from analise.models.dixon_coles import DixonColes, fair_probs, log_loss

OUTCOMES = ["home", "draw", "away"]


def _market_probs(row) -> tuple[float, float, float] | None:
    for kind in ("pin", "avg"):
        odds = [row[f"odds_{o}_{kind}"] for o in OUTCOMES]
        if all(pd.notna(v) and v > 1 for v in odds):
            return fair_probs(*odds)
    return None


def _outcome(hg: int, ag: int) -> int:
    return 0 if hg > ag else (1 if hg == ag else 2)


def walk_forward(
    df: pd.DataFrame,
    half_life_days: float = 180.0,
    window_days: int = 730,
    min_train: int = 300,
    min_team_matches: int = 5,
    start=None,
) -> pd.DataFrame:
    """Uma linha por jogo previsto, com probabilidades do modelo e do mercado.

    df: jogos de uma só liga com date, home, away, home_goals, away_goals e odds.
    Jogos de equipas com menos de min_team_matches no treino (p. ex. promovidas)
    ficam de fora; a cobertura vem no resumo.
    """
    df = df.dropna(subset=["home_goals", "away_goals"]).copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df["week"] = df["date"].dt.to_period("W-SUN").dt.start_time

    rows = []
    for week, games in df.groupby("week"):
        if start is not None and week < pd.Timestamp(start):
            continue
        train = df[(df["date"] < week) & (df["date"] >= week - timedelta(days=window_days))]
        if len(train) < min_train:
            continue
        counts = pd.concat([train["home"], train["away"]]).value_counts()
        model = DixonColes(half_life_days=half_life_days).fit(train, as_of=week)
        for g in games.itertuples(index=False):
            row = g._asdict()
            known = min(counts.get(g.home, 0), counts.get(g.away, 0)) >= min_team_matches
            market = _market_probs(row)
            p = model.predict(g.home, g.away) if known else None
            rows.append({
                "date": g.date.date(), "home": g.home, "away": g.away,
                "outcome": _outcome(g.home_goals, g.away_goals),
                "model_home": p and p["home"], "model_draw": p and p["draw"],
                "model_away": p and p["away"],
                "market_home": market and market[0], "market_draw": market and market[1],
                "market_away": market and market[2],
            })
    return pd.DataFrame(rows)


def _probs(res: pd.DataFrame, who: str) -> np.ndarray:
    return res[[f"{who}_{o}" for o in OUTCOMES]].to_numpy(dtype=float)


def brier(probs: np.ndarray, outcomes: np.ndarray) -> float:
    onehot = np.eye(3)[outcomes]
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def rps(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Ranked probability score: tem em conta que o empate fica entre casa e fora."""
    onehot = np.eye(3)[outcomes]
    diff = np.cumsum(probs - onehot, axis=1)[:, :2]
    return float((diff ** 2).sum(axis=1).mean() / 2)


def summary(res: pd.DataFrame) -> dict:
    """Métricas só nos jogos em que modelo e mercado têm previsão."""
    both = res.dropna(subset=["model_home", "market_home"])
    y = both["outcome"].to_numpy(dtype=int)
    out = {"jogos": len(res), "avaliados": len(both)}
    for who in ("model", "market"):
        p = _probs(both, who)
        out[f"logloss_{who}"] = log_loss(p, y) if len(both) else np.nan
        out[f"brier_{who}"] = brier(p, y) if len(both) else np.nan
        out[f"rps_{who}"] = rps(p, y) if len(both) else np.nan
    return out


def calibration(res: pd.DataFrame, who: str = "model", bins: int = 10) -> pd.DataFrame:
    """Probabilidade prevista média vs frequência observada, juntando os 3 resultados."""
    both = res.dropna(subset=["model_home", "market_home"])
    p = _probs(both, who).ravel()
    hit = np.eye(3)[both["outcome"].to_numpy(dtype=int)].ravel()
    edges = np.linspace(0, 1, bins + 1)
    b = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    table = pd.DataFrame({"bin": b, "prevista": p, "observada": hit}).groupby("bin").agg(
        prevista=("prevista", "mean"), observada=("observada", "mean"), n=("observada", "size"))
    table.index = [f"{edges[i]:.1f}-{edges[i + 1]:.1f}" for i in table.index]
    return table
