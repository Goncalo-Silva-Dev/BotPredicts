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

# Cada período em função do jogo inteiro, estimado com ~3500 jogos da NBA, EuroLeague e
# EuroCup com pontos por quarto (os valores quase não mudam de liga para liga):
# (fração da diferença esperada, fração do total, desvio da diferença e do total em
# proporção aos do jogo inteiro; o do total alargado 10% para o intervalo de 80% acertar
# em 80% dos jogos de teste). O 4.º quarto tem menos diferença: quem ganha abranda.
PERIODS = {
    "1.º quarto": (0.275, 0.251, 0.565, 0.462),
    "2.º quarto": (0.258, 0.251, 0.574, 0.476),
    "3.º quarto": (0.195, 0.245, 0.588, 0.458),
    "4.º quarto": (0.222, 0.247, 0.570, 0.509),
    "1.ª parte": (0.533, 0.502, 0.765, 0.691),
    "2.ª parte": (0.417, 0.492, 0.775, 0.707),
}


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

    def predict_periods(self, home: str, away: str) -> list[dict]:
        """Vencedor (com empate, que num período acontece) e pontos de cada quarto e parte."""
        ph, pa = self.expected_points(home, away)
        out = []
        for name, (f_margin, f_total, r_margin, r_total) in PERIODS.items():
            margin, total = f_margin * (ph - pa), f_total * (ph + pa)
            sd = r_margin * self.sd_margin_
            # a diferença é um número inteiro: empate = diferença entre -0.5 e 0.5
            p_home = float(1 - norm.cdf((0.5 - margin) / sd))
            p_away = float(norm.cdf((-0.5 - margin) / sd))
            half = 1.2816 * r_total * self.sd_total_
            out.append({
                "periodo": name, "casa": p_home, "empate": 1 - p_home - p_away, "fora": p_away,
                "pontos_casa": (total + margin) / 2, "pontos_fora": (total - margin) / 2,
                "total": total, "total_min": total - half, "total_max": total + half,
            })
        return out
