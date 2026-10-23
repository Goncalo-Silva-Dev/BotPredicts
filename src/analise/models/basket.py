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
from scipy.stats import multivariate_normal, norm

# Cada período em função do jogo inteiro, estimado com ~3500 jogos da NBA, EuroLeague e
# EuroCup com pontos por quarto (os valores quase não mudam de liga para liga):
# (fração da diferença esperada, fração do total, desvio da diferença e do total em
# proporção aos do jogo inteiro). O 4.º quarto tem menos diferença: quem ganha abranda.
PERIODS = {
    "1.º quarto": (0.275, 0.251, 0.565, 0.420),
    "2.º quarto": (0.258, 0.251, 0.574, 0.433),
    "3.º quarto": (0.195, 0.245, 0.588, 0.416),
    "4.º quarto": (0.222, 0.247, 0.570, 0.463),
    "1.ª parte": (0.533, 0.502, 0.765, 0.628),
    "2.ª parte": (0.417, 0.492, 0.775, 0.643),
}

# Os resíduos do próprio ajuste subestimam o erro em jogos futuros (as forças das equipas
# também são estimadas). Em 7875 jogos de teste das 12 ligas, prevendo cada semana só com
# o passado, o erro real foi 1,075 vezes o da diferença e 1,08 vezes o do total.
SD_MARGIN_FACTOR = 1.075
SD_TOTAL_FACTOR = 1.08

# Diferença da 1.ª parte vs. diferença final: correlação dos desvios ao previsto
# (0,60 a 0,66 na NBA, EuroLeague e EuroCup)
H1_FULL_CORR = 0.64


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
        self.sd_margin_ = SD_MARGIN_FACTOR * float(np.sqrt(np.average(margin_res ** 2, weights=w)))
        self.sd_total_ = SD_TOTAL_FACTOR * float(np.sqrt(np.average(total_res ** 2, weights=w)))
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

    def markets(self, home: str, away: str) -> dict:
        """Outras apostas do jogo inteiro, todas da mesma normal da diferença e do total.

        As linhas acabam em ,5 (como nas casas de apostas) e ficam à volta do previsto."""
        ph, pa = self.expected_points(home, away)
        m, t = ph - pa, ph + pa
        sm, st = self.sd_margin_, self.sd_total_
        # pontos de uma equipa = (total ± diferença) / 2
        s_team = float(np.sqrt(st ** 2 + sm ** 2) / 2)
        over = lambda mean, sd, line: float(1 - norm.cdf((line - mean) / sd))  # noqa: E731
        around = lambda x, steps: [np.floor(x) + 0.5 + k for k in steps]  # noqa: E731

        out = {
            # handicap da casa: "casa -L" ganha se a diferença final for maior que L
            "handicap": [(float(line), over(m, sm, line)) for line in around(m, (-6, -3, 0, 3, 6))],
            "total": [(float(line), over(t, st, line)) for line in around(t, (-10, -5, 0, 5, 10))],
            "pontos_casa": [(float(line), over(ph, s_team, line)) for line in around(ph, (-5, 0, 5))],
            "pontos_fora": [(float(line), over(pa, s_team, line)) for line in around(pa, (-5, 0, 5))],
        }
        # margem de vitória (sem empate: o prolongamento decide, por isso tira-se a fatia do 0)
        tie = float(norm.cdf(0.5, m, sm) - norm.cdf(-0.5, m, sm))
        bands = [(1, 5), (6, 10), (11, 15), (16, 20), (21, None)]
        margin = []
        for side, sign in (("casa", 1), ("fora", -1)):
            for lo, hi in bands:
                a, b = lo - 0.5, (hi + 0.5) if hi else np.inf
                p = (norm.cdf(b, sign * m, sm) - norm.cdf(a, sign * m, sm)) / (1 - tie)
                margin.append((side, f"{lo}-{hi}" if hi else f"{lo}+", float(p)))
        out["margem"] = margin

        # intervalo/final: diferença na 1.ª parte e no fim como normal bivariada
        f1, _, r1, _ = PERIODS["1.ª parte"]
        mean = [f1 * m, m]
        s1 = r1 * sm
        cov = [[s1 ** 2, H1_FULL_CORR * s1 * sm], [H1_FULL_CORR * s1 * sm, sm ** 2]]
        mvn = multivariate_normal(mean, cov)
        cdf = lambda x, y: float(mvn.cdf([x, y]))  # noqa: E731
        cut = {"2": (-np.inf, -0.5), "X": (-0.5, 0.5), "1": (0.5, np.inf)}
        htft = {}
        for k, (lo, hi) in cut.items():
            p_h1 = float(norm.cdf(hi, f1 * m, s1) - norm.cdf(lo, f1 * m, s1))
            hi_c = min(hi, 1e6) if np.isfinite(hi) else 1e6
            lo_c = max(lo, -1e6) if np.isfinite(lo) else -1e6
            p_h1_away = cdf(hi_c, 0) - cdf(lo_c, 0)  # 1.ª parte nesse intervalo e perde no fim
            htft[f"{k}/2"] = max(p_h1_away, 0.0)
            htft[f"{k}/1"] = max(p_h1 - p_h1_away, 0.0)
        total_p = sum(htft.values())
        out["intervalo_final"] = {k: v / total_p for k, v in htft.items()}
        return out
