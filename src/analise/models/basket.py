"""Ratings de ataque e defesa para basquetebol.

Pontos da casa = média + vantagem casa + ataque(casa) - defesa(fora) + ruído
Pontos de fora = média + ataque(fora) - defesa(casa) + ruído

Ajustado por mínimos quadrados pesados (jogos antigos pesam menos, com
meia-vida configurável) e uma pequena penalização ridge que puxa as equipas
com poucos jogos para a média. A diferença e o total de pontos seguem uma
normal com o desvio padrão dos resíduos, daí a probabilidade de vitória.
"""
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
from scipy.stats import norm


@dataclass
class BasketRatings:
    half_life_days: float = 120.0
    ridge: float = 2.0
    teams_: list | None = None
    mean_: float = 0.0
    home_: float = 0.0
    attack_: np.ndarray | None = None
    defence_: np.ndarray | None = None
    sd_margin_: float = 12.0
    sd_total_: float = 18.0

    def fit(self, df: pd.DataFrame, as_of: date | None = None) -> "BasketRatings":
        """df precisa de: date, home, away, home_points, away_points."""
        df = df.dropna(subset=["home_points", "away_points"])
        as_of = pd.Timestamp(as_of or df["date"].max())
        age = (as_of - pd.to_datetime(df["date"])).dt.days.to_numpy()
        w = np.exp(-np.log(2) * np.clip(age, 0, None) / self.half_life_days)

        self.teams_ = sorted(set(df["home"]) | set(df["away"]))
        idx = {t: i for i, t in enumerate(self.teams_)}
        hi = df["home"].map(idx).to_numpy()
        ai = df["away"].map(idx).to_numpy()
        n, g = len(self.teams_), len(df)

        # Colunas: média, casa, ataque[n], defesa[n]; uma linha por pontuação
        X = np.zeros((2 * g, 2 + 2 * n))
        rows = np.arange(g)
        X[rows, 0] = X[g + rows, 0] = 1
        X[rows, 1] = 1
        X[rows, 2 + hi] = 1
        X[rows, 2 + n + ai] = -1
        X[g + rows, 2 + ai] = 1
        X[g + rows, 2 + n + hi] = -1
        y = np.concatenate([df["home_points"].to_numpy(float), df["away_points"].to_numpy(float)])
        ww = np.concatenate([w, w])

        penalty = np.diag([0, 0] + [self.ridge] * (2 * n))
        A = X.T @ (X * ww[:, None]) + penalty
        beta = np.linalg.solve(A, X.T @ (ww * y))
        self.mean_, self.home_ = beta[0], beta[1]
        self.attack_, self.defence_ = beta[2 : 2 + n], beta[2 + n :]

        pred = X @ beta
        margin_res = (y[:g] - y[g:]) - (pred[:g] - pred[g:])
        total_res = (y[:g] + y[g:]) - (pred[:g] + pred[g:])
        self.sd_margin_ = float(np.sqrt(np.average(margin_res ** 2, weights=w)))
        self.sd_total_ = float(np.sqrt(np.average(total_res ** 2, weights=w)))
        return self

    def expected_points(self, home: str, away: str) -> tuple[float, float]:
        h, a = self.teams_.index(home), self.teams_.index(away)
        ph = self.mean_ + self.home_ + self.attack_[h] - self.defence_[a]
        pa = self.mean_ + self.attack_[a] - self.defence_[h]
        return float(ph), float(pa)

    def predict(self, home: str, away: str) -> dict:
        ph, pa = self.expected_points(home, away)
        p_home = float(norm.cdf((ph - pa) / self.sd_margin_))
        return {
            "home": p_home,
            "away": 1 - p_home,
            "points_home": ph,
            "points_away": pa,
            "margin": ph - pa,
            "total": ph + pa,
            "total_low": ph + pa - 1.2816 * self.sd_total_,  # intervalo de 80%
            "total_high": ph + pa + 1.2816 * self.sd_total_,
        }
