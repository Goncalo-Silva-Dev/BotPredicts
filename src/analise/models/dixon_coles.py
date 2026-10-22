"""Modelo de Dixon-Coles (1997) para resultados de futebol.

Golos de cada equipa ~ Poisson(ataque * defesa adversária * vantagem casa), com
correção rho para os resultados baixos (0-0, 1-0, 0-1, 1-1) e jogos antigos
pesados menos (decaimento exponencial com meia-vida configurável).
"""
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

MAX_GOALS = 10


def _tau(x, y, lam, mu, rho):
    t = np.ones_like(lam)
    t = np.where((x == 0) & (y == 0), 1 - lam * mu * rho, t)
    t = np.where((x == 0) & (y == 1), 1 + lam * rho, t)
    t = np.where((x == 1) & (y == 0), 1 + mu * rho, t)
    t = np.where((x == 1) & (y == 1), 1 - rho, t)
    return t


@dataclass
class DixonColes:
    half_life_days: float = 180.0
    teams_: list | None = None
    attack_: np.ndarray | None = None
    defence_: np.ndarray | None = None
    home_: float = 0.0
    rho_: float = 0.0

    def fit(self, df: pd.DataFrame, as_of: date | None = None) -> "DixonColes":
        """df precisa de: date, home, away, home_goals, away_goals."""
        df = df.dropna(subset=["home_goals", "away_goals"])
        as_of = pd.Timestamp(as_of or df["date"].max())
        age = (as_of - pd.to_datetime(df["date"])).dt.days.to_numpy()
        w = np.exp(-np.log(2) * np.clip(age, 0, None) / self.half_life_days)

        self.teams_ = sorted(set(df["home"]) | set(df["away"]))
        idx = {t: i for i, t in enumerate(self.teams_)}
        hi = df["home"].map(idx).to_numpy()
        ai = df["away"].map(idx).to_numpy()
        x = df["home_goals"].to_numpy(dtype=int)
        y = df["away_goals"].to_numpy(dtype=int)
        n = len(self.teams_)

        def unpack(p):
            att = np.append(p[: n - 1], -p[: n - 1].sum())  # soma zero para identificar o modelo
            dfc = p[n - 1 : 2 * n - 1]
            return att, dfc, p[-2], p[-1]

        def nll(p):
            att, dfc, home, rho = unpack(p)
            lam = np.exp(att[hi] + dfc[ai] + home)
            mu = np.exp(att[ai] + dfc[hi])
            tau = np.clip(_tau(x, y, lam, mu, rho), 1e-10, None)
            ll = np.log(tau) + poisson.logpmf(x, lam) + poisson.logpmf(y, mu)
            return -(w * ll).sum()

        p0 = np.concatenate([np.zeros(n - 1), np.zeros(n), [0.25, -0.05]])
        bounds = [(None, None)] * (2 * n - 1) + [(None, None), (-0.3, 0.3)]
        res = minimize(nll, p0, method="L-BFGS-B", bounds=bounds)
        self.attack_, self.defence_, self.home_, self.rho_ = unpack(res.x)
        return self

    def score_matrix(self, home: str, away: str) -> np.ndarray:
        """Probabilidade de cada resultado exato; [i, j] = casa marca i, fora marca j."""
        h, a = self.teams_.index(home), self.teams_.index(away)
        lam = np.exp(self.attack_[h] + self.defence_[a] + self.home_)
        mu = np.exp(self.attack_[a] + self.defence_[h])
        g = np.arange(MAX_GOALS + 1)
        m = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))
        xx, yy = np.meshgrid(g, g, indexing="ij")
        m *= _tau(xx, yy, np.full_like(m, lam), np.full_like(m, mu), self.rho_)
        return m / m.sum()

    def predict(self, home: str, away: str) -> dict:
        m = self.score_matrix(home, away)
        g = np.arange(MAX_GOALS + 1)
        total = g[:, None] + g[None, :]
        return {
            "home": float(np.tril(m, -1).sum()),
            "draw": float(np.trace(m)),
            "away": float(np.triu(m, 1).sum()),
            "over_2_5": float(m[total > 2.5].sum()),
            "btts": float(m[1:, 1:].sum()),
            "goals_home": float((g[:, None] * m).sum()),
            "goals_away": float((g[None, :] * m).sum()),
        }

    def top_scores(self, home: str, away: str, n: int = 3) -> list[tuple[str, float]]:
        """Os n resultados exatos mais prováveis, p. ex. [("1-0", 0.12), ...]."""
        m = self.score_matrix(home, away)
        best = np.argsort(m, axis=None)[::-1][:n]
        return [(f"{i}-{j}", float(m[i, j])) for i, j in zip(*np.unravel_index(best, m.shape))]


def fair_probs(odds_home: float, odds_draw: float, odds_away: float) -> tuple[float, float, float]:
    """Probabilidades implícitas sem a margem da casa (normalização proporcional)."""
    inv = np.array([1 / odds_home, 1 / odds_draw, 1 / odds_away])
    p = inv / inv.sum()
    return tuple(float(v) for v in p)


def log_loss(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """probs: (n, 3) para casa/empate/fora; outcomes: índice 0, 1 ou 2."""
    p = np.clip(probs[np.arange(len(outcomes)), outcomes], 1e-12, 1)
    return float(-np.log(p).mean())
