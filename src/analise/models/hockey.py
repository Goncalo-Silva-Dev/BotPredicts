"""Modelo de hóquei no gelo: golos nos 60 minutos, fim de jogo, prolongamento e penáltis.

1. Forças de ataque e defesa de cada equipa com o Dixon-Coles do futebol, ajustado aos golos
   dos 60 minutos (os do prolongamento e dos penáltis ficam de fora).
2. No hóquei o fim do jogo é especial: quem perde por 1 ou 2 tira o guarda-redes, o que dá
   golos na baliza deserta (a vantagem cresce) ou empates tardios. Com só Poisson, o modelo
   previa 17% de empates aos 60 minutos e na NHL são 23%. Por isso os golos "normais" seguem
   uma Poisson e no fim aplica-se uma transição: com vantagem de 1, quem ganha marca mais um
   com probabilidade p1 e quem perde empata com q1; com vantagem de 2, p2 e q2.
   Os 5 números foram estimados por máxima verosimilhança em 2511 jogos de teste da NHL
   (cada um previsto só com os jogos anteriores) e confirmados em jogos que não entraram
   na estimativa.
3. Empate aos 60 minutos: prolongamento e penáltis. A casa ganhou 52% (586 jogos); a força
   das equipas quase não ajudou a prever quem ganha, por isso fica uma constante.
"""
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd
from scipy.stats import poisson

from analise.models.dixon_coles import DixonColes

MAX_GOALS = 11
# (s, p1, q1, p2, q2): s encolhe os golos "normais", porque o fim de jogo acrescenta alguns
END_KERNEL = (0.9588, 0.3053, 0.2182, 0.3803, 0.1648)
OT_HOME = 0.5205
SHRINK_TOTAL, SHRINK_DIFF = 0.25, 0.75
# Parte dos golos "normais" no 1.º e 2.º período (casa, fora); o 3.º fica com o resto e
# com os golos do fim de jogo
PERIOD_SHARES = {"home": (0.313, 0.365), "away": (0.306, 0.358)}
PERIOD_NAMES = ["1.º período", "2.º período", "3.º período"]


def end_of_game(m: np.ndarray, p1: float, q1: float, p2: float, q2: float) -> np.ndarray:
    """Aplica a transição do fim de jogo a uma matriz de resultados [casa, fora]."""
    n = m.shape[0]
    out = np.zeros((n + 1, n + 1))
    for h in range(n):
        for a in range(n):
            v, d = m[h, a], h - a
            if abs(d) not in (1, 2):
                out[h, a] += v
                continue
            lead, trail = (p1, q1) if abs(d) == 1 else (p2, q2)
            if d > 0:
                out[h + 1, a] += v * lead
                out[h, a + 1] += v * trail
            else:
                out[h, a + 1] += v * lead
                out[h + 1, a] += v * trail
            out[h, a] += v * (1 - lead - trail)
    return out


@dataclass
class HockeyModel:
    half_life_days: float = 120.0
    kernel: tuple = END_KERNEL
    ot_home: float = OT_HOME
    dc_: DixonColes = field(default=None)

    @property
    def teams_(self) -> list:
        return self.dc_.teams_

    def fit(self, df: pd.DataFrame, as_of: date | None = None) -> "HockeyModel":
        """df precisa de: date, home, away, home_reg, away_reg (golos nos 60 minutos)."""
        reg = df.dropna(subset=["home_reg", "away_reg"]).rename(
            columns={"home_goals": "final_home", "away_goals": "final_away"}).rename(
            columns={"home_reg": "home_goals", "away_reg": "away_goals"})
        self.dc_ = DixonColes(half_life_days=self.half_life_days).fit(reg, as_of=as_of)
        pairs = np.array([self._raw(h, a) for h, a in zip(reg["home"], reg["away"])])
        self.mean_sum_, self.mean_diff_ = float(pairs.sum(1).mean()), float((pairs[:, 0] - pairs[:, 1]).mean())
        return self

    def _raw(self, home, away):
        return np.log(self.dc_.expected_goals(home, away))

    def expected_goals(self, home: str, away: str) -> tuple[float, float]:
        """Golos "normais" esperados (antes do fim de jogo).

        As forças do Dixon-Coles exageram as diferenças: nos jogos de teste, a soma dos
        golos esperados acertou mais com só 25% do desvio para a média da liga, e a
        diferença com 75% (por isso os totais variam pouco de jogo para jogo, como na NHL)."""
        lh, la = self._raw(home, away)
        s = self.mean_sum_ + SHRINK_TOTAL * (lh + la - self.mean_sum_)
        d = self.mean_diff_ + SHRINK_DIFF * (lh - la - self.mean_diff_)
        return float(np.exp((s + d) / 2)) * self.kernel[0], float(np.exp((s - d) / 2)) * self.kernel[0]

    def regulation_matrix(self, home: str, away: str) -> np.ndarray:
        """Probabilidade de cada resultado aos 60 minutos; [i, j] = casa i, fora j."""
        lam, mu = self.expected_goals(home, away)
        g = np.arange(MAX_GOALS)
        return end_of_game(np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu)), *self.kernel[1:])

    def predict(self, home: str, away: str) -> dict:
        m = self.regulation_matrix(home, away)
        g = np.arange(m.shape[0])
        diff = g[:, None] - g[None, :]
        reg_home, tie = float(m[diff > 0].sum()), float(m[diff == 0].sum())
        reg_away = 1 - reg_home - tie
        goals_home, goals_away = float((g[:, None] * m).sum()), float((g[None, :] * m).sum())
        return {
            # vencedor com prolongamento e penáltis
            "home": reg_home + tie * self.ot_home, "away": reg_away + tie * (1 - self.ot_home),
            # 60 minutos
            "reg_home": reg_home, "reg_draw": tie, "reg_away": reg_away,
            # golos esperados no resultado final (quem ganha no prolongamento/penáltis soma 1)
            "goals_home": goals_home + tie * self.ot_home,
            "goals_away": goals_away + tie * (1 - self.ot_home),
        }

    def markets(self, home: str, away: str) -> dict:
        """Mercados do jogo; as linhas de golos contam com o prolongamento e os penáltis,
        como nas casas de apostas (o vencedor no prolongamento ou nos penáltis soma 1 golo)."""
        m = self.regulation_matrix(home, away)
        g = np.arange(m.shape[0])
        diff = g[:, None] - g[None, :]
        total = g[:, None] + g[None, :]
        tie = float(m[diff == 0].sum())
        reg_home = float(m[diff > 0].sum())
        reg_away = 1 - reg_home - tie
        q = self.ot_home

        def final_total_over(line):  # empate aos 60 minutos -> mais 1 golo no fim
            return float(m[(total + (diff == 0)) > line].sum())

        def team_over(side, line):
            goals = g[:, None] + 0 * g[None, :] if side == "home" else 0 * g[:, None] + g[None, :]
            win_ot = q if side == "home" else 1 - q
            no_tie = float(m[(diff != 0) & (goals > line)].sum())
            # no empate a equipa fica com +1 se ganhar no prolongamento/penáltis
            tie_over = float(m[(diff == 0) & (goals > line)].sum()) + \
                win_ot * float(m[(diff == 0) & (goals + 1 > line) & (goals <= line)].sum())
            return no_tie + tie_over

        exp_total = float((total * m).sum()) + tie
        center = np.floor(exp_total) + 0.5
        return {
            "dupla": {"1X": reg_home + tie, "12": reg_home + reg_away, "X2": tie + reg_away},
            "prolongamento": tie,
            # puck line: o prolongamento decide sempre por 1, por isso -1,5 só se ganha nos 60'
            "puck_line": [
                ("casa", -1.5, float(m[diff >= 2].sum())),
                ("casa", +1.5, 1 - float(m[diff <= -2].sum())),
                ("fora", -1.5, float(m[diff <= -2].sum())),
                ("fora", +1.5, 1 - float(m[diff >= 2].sum())),
                ("casa", -2.5, float(m[diff >= 3].sum())),
                ("fora", -2.5, float(m[diff <= -3].sum())),
            ],
            "total": [(line, final_total_over(line)) for line in center + np.array([-2, -1, 0, 1, 2])],
            "golos_casa": [(line, team_over("home", line)) for line in (1.5, 2.5, 3.5, 4.5)],
            "golos_fora": [(line, team_over("away", line)) for line in (1.5, 2.5, 3.5, 4.5)],
            "ambas_marcam": float(m[1:, 1:].sum()),
            "resultados": [(f"{i}-{j}", float(m[i, j])) for i, j in
                           sorted(np.ndindex(*m.shape), key=lambda ij: -m[ij])[:4]],
        }

    def periods(self, home: str, away: str, n: int = 40_000, seed: int = 7) -> list[dict]:
        """Vencedor e golos de cada período, por simulação (o fim de jogo cai no 3.º)."""
        lam, mu = self.expected_goals(home, away)
        rng = np.random.default_rng(seed)
        sh, sa = PERIOD_SHARES["home"], PERIOD_SHARES["away"]
        shares_h, shares_a = (*sh, 1 - sum(sh)), (*sa, 1 - sum(sa))
        gh = np.stack([rng.poisson(lam * s, n) for s in shares_h])
        ga = np.stack([rng.poisson(mu * s, n) for s in shares_a])
        # fim de jogo: com o resultado antes dele, sorteia golo de quem ganha ou de quem perde
        p1, q1, p2, q2 = self.kernel[1:]
        d = gh.sum(0) - ga.sum(0)
        lead = np.where(np.abs(d) == 1, p1, np.where(np.abs(d) == 2, p2, 0.0))
        trail = np.where(np.abs(d) == 1, q1, np.where(np.abs(d) == 2, q2, 0.0))
        u = rng.random(n)
        leader_scores, trailer_scores = u < lead, (u >= lead) & (u < lead + trail)
        gh[2] += np.where(d > 0, leader_scores, trailer_scores) * (d != 0)
        ga[2] += np.where(d < 0, leader_scores, trailer_scores) * (d != 0)
        out = []
        for k, name in enumerate(PERIOD_NAMES):
            h, a = gh[k], ga[k]
            tot = h + a
            out.append({
                "periodo": name, "casa": float((h > a).mean()), "empate": float((h == a).mean()),
                "fora": float((h < a).mean()), "golos_casa": float(h.mean()), "golos_fora": float(a.mean()),
                "mais_0_5": float((tot > 0.5).mean()), "mais_1_5": float((tot > 1.5).mean()),
                "mais_2_5": float((tot > 2.5).mean()),
            })
        # período com mais golos (empate se dois ou mais tiverem o máximo)
        tots = gh + ga
        best = tots.max(axis=0)
        ties = (tots == best).sum(axis=0) > 1
        most = {name: float(((tots[k] == best) & ~ties).mean()) for k, name in enumerate(PERIOD_NAMES)}
        most["Empate"] = float(ties.mean())
        return out, most
