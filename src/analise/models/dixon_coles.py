"""Modelo de Dixon-Coles (1997) para resultados de futebol.

Golos de cada equipa ~ Poisson(ataque * defesa adversária * vantagem casa), com
correção rho para os resultados baixos (0-0, 1-0, 0-1, 1-1) e jogos antigos
pesados menos (decaimento exponencial com meia-vida configurável).

Quando há remates à baliza, as forças são estimadas com uma mistura de golos e
remates à baliza convertidos em golos (metade de cada): os remates têm menos
sorte que os golos e o backtest walk-forward melhora em todas as ligas com esses dados.
"""
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import poisson

# Fração dos golos marcados na 1.ª parte (casa, fora) quando a liga não tem resultados
# ao intervalo (Brasil e MLS no football-data); média das ligas europeias
HT_SHARE = (0.446, 0.439)
MAX_GOALS = 10


def _tau(x, y, lam, mu, rho):
    t = np.ones_like(lam)
    t = np.where((x == 0) & (y == 0), 1 - lam * mu * rho, t)
    t = np.where((x == 0) & (y == 1), 1 + lam * rho, t)
    t = np.where((x == 1) & (y == 0), 1 + mu * rho, t)
    t = np.where((x == 1) & (y == 1), 1 - rho, t)
    return t


def _shot_targets(df: pd.DataFrame, x: np.ndarray, y: np.ndarray, goals_weight: float):
    """Golos misturados com remates à baliza × (golos por remate à baliza na amostra).
    Jogos sem remates ficam só com os golos."""
    if goals_weight >= 1 or "home_shots_target" not in df:
        return None
    hs = pd.to_numeric(df["home_shots_target"], errors="coerce").to_numpy(dtype=float)
    as_ = pd.to_numeric(df["away_shots_target"], errors="coerce").to_numpy(dtype=float)
    ok = ~np.isnan(hs) & ~np.isnan(as_)
    if ok.mean() < 0.5 or hs[ok].sum() + as_[ok].sum() == 0:
        return None
    k = (x[ok].sum() + y[ok].sum()) / (hs[ok].sum() + as_[ok].sum())
    a = goals_weight
    return (np.where(ok, a * x + (1 - a) * k * np.nan_to_num(hs), x),
            np.where(ok, a * y + (1 - a) * k * np.nan_to_num(as_), y))


@dataclass
class DixonColes:
    half_life_days: float = 180.0
    # peso dos golos reais face aos remates à baliza (1 = só golos)
    goals_weight: float = 0.5
    teams_: list | None = None
    attack_: np.ndarray | None = None
    defence_: np.ndarray | None = None
    home_: float = 0.0
    rho_: float = 0.0

    def fit(self, df: pd.DataFrame, as_of: date | None = None) -> "DixonColes":
        """df precisa de: date, home, away, home_goals, away_goals (e, se houver,
        home_shots_target, away_shots_target)."""
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
        shots = _shot_targets(df, x, y, self.goals_weight)
        # com remates, as forças vêm dos alvos misturados (sem rho) e o rho dos golos a seguir
        joint = shots is None
        tx, ty = (x, y) if joint else shots

        def unpack(p):
            att = np.append(p[: n - 1], -p[: n - 1].sum())  # soma zero para identificar o modelo
            dfc = p[n - 1 : 2 * n - 1]
            return att, dfc, p[2 * n - 1], (p[2 * n] if joint else 0.0)

        def nll(p):
            # log-verosimilhança de Poisson (sem constantes) com gradiente analítico
            att, dfc, home, rho = unpack(p)
            log_lam, log_mu = att[hi] + dfc[ai] + home, att[ai] + dfc[hi]
            lam, mu = np.exp(log_lam), np.exp(log_mu)
            val = (w * (tx * log_lam - lam + ty * log_mu - mu)).sum()
            g_lam, g_mu = w * (tx - lam), w * (ty - mu)
            g_rho = []
            if joint:
                tau = np.clip(_tau(x, y, lam, mu, rho), 1e-10, None)
                val += (w * np.log(tau)).sum()
                m00, m01 = (x == 0) & (y == 0), (x == 0) & (y == 1)
                m10, m11 = (x == 1) & (y == 0), (x == 1) & (y == 1)
                wt = w / tau
                g_lam -= np.where(m00, wt * lam * mu * rho, 0) - np.where(m01, wt * lam * rho, 0)
                g_mu -= np.where(m00, wt * lam * mu * rho, 0) - np.where(m10, wt * mu * rho, 0)
                g_rho = [(np.where(m00, -wt * lam * mu, 0) + np.where(m01, wt * lam, 0)
                          + np.where(m10, wt * mu, 0) - np.where(m11, wt, 0)).sum()]
            g_att = np.bincount(hi, g_lam, n) + np.bincount(ai, g_mu, n)
            g_def = np.bincount(ai, g_lam, n) + np.bincount(hi, g_mu, n)
            grad = np.concatenate([g_att[:-1] - g_att[-1], g_def, [g_lam.sum()], g_rho])
            return -val, -grad

        p0 = np.concatenate([np.zeros(n - 1), np.zeros(n), [0.25], [-0.05] if joint else []])
        bounds = [(None, None)] * (2 * n) + ([(-0.3, 0.3)] if joint else [])
        res = minimize(nll, p0, jac=True, method="L-BFGS-B", bounds=bounds)
        self.attack_, self.defence_, self.home_, self.rho_ = unpack(res.x)
        if not joint:
            lam = np.exp(self.attack_[hi] + self.defence_[ai] + self.home_)
            mu = np.exp(self.attack_[ai] + self.defence_[hi])
            self.rho_ = float(minimize_scalar(
                lambda r: -(w * np.log(np.clip(_tau(x, y, lam, mu, r), 1e-10, None))).sum(),
                bounds=(-0.3, 0.3), method="bounded").x)
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

    def markets(self, home: str, away: str) -> dict:
        """Mercados que saem da matriz de resultados exatos do jogo inteiro."""
        m = self.score_matrix(home, away)
        g = np.arange(MAX_GOALS + 1)
        hg, ag = np.meshgrid(g, g, indexing="ij")
        total, diff = hg + ag, hg - ag
        p = lambda mask: float(m[mask].sum())  # noqa: E731
        out = {
            "dupla_1x": p(diff >= 0), "dupla_12": p(diff != 0), "dupla_x2": p(diff <= 0),
            # empate anula aposta: o empate devolve o dinheiro, por isso fica de fora
            "dnb_casa": p(diff > 0) / max(p(diff != 0), 1e-12),
            "casa_sem_sofrer": p((diff > 0) & (ag == 0)), "fora_sem_sofrer": p((diff < 0) & (hg == 0)),
            "casa_marca": p(hg > 0), "fora_marca": p(ag > 0),
        }
        for line in (0.5, 1.5, 2.5, 3.5, 4.5):
            out[f"mais_{line}"] = p(total > line)
        for line in (0.5, 1.5, 2.5):
            out[f"casa_mais_{line}"] = p(hg > line)
            out[f"fora_mais_{line}"] = p(ag > line)
        for n in range(4):
            out[f"golos_{n}"] = p(total == n)
        out["golos_4+"] = p(total >= 4)
        # handicap europeu de -1 golo para a casa: casa ganha por 2+, empate ganha por 1
        out["hcp_casa_-1"], out["hcp_empate_-1"], out["hcp_fora_+1"] = p(diff >= 2), p(diff == 1), p(diff <= 0)
        out["hcp_casa_+1"], out["hcp_empate_+1"], out["hcp_fora_-1"] = p(diff >= 0), p(diff == -1), p(diff <= -2)
        return out

    def half_time_full_time(self, home: str, away: str, share_home: float = HT_SHARE[0],
                            share_away: float = HT_SHARE[1]) -> dict:
        """Intervalo/final (9 combinações), com as partes como Poissons independentes."""
        lam, mu = self.expected_goals(home, away)
        g = np.arange(MAX_GOALS + 1)

        def half(sh, sa):
            m = np.outer(poisson.pmf(g, lam * sh), poisson.pmf(g, mu * sa))
            return m / m.sum()

        first, second = half(share_home, share_away), half(1 - share_home, 1 - share_away)
        d = g[:, None] - g[None, :]
        n = 2 * MAX_GOALS + 1
        # distribuição da diferença de golos em cada parte, de -MAX a +MAX
        d1 = np.bincount((d + MAX_GOALS).ravel(), weights=first.ravel(), minlength=n)
        d2 = np.bincount((d + MAX_GOALS).ravel(), weights=second.ravel(), minlength=n)
        diffs = np.arange(-MAX_GOALS, MAX_GOALS + 1)
        sign = lambda x: "1" if x > 0 else ("X" if x == 0 else "2")  # noqa: E731
        out = {f"{a}/{b}": 0.0 for a in "1X2" for b in "1X2"}
        for i, x in enumerate(diffs):
            for j, y in enumerate(diffs):
                out[f"{sign(x)}/{sign(x + y)}"] += float(d1[i] * d2[j])
        return out

    def expected_goals(self, home: str, away: str) -> tuple[float, float]:
        h, a = self.teams_.index(home), self.teams_.index(away)
        return (float(np.exp(self.attack_[h] + self.defence_[a] + self.home_)),
                float(np.exp(self.attack_[a] + self.defence_[h])))

    def halves(self, home: str, away: str, share_home: float = HT_SHARE[0],
               share_away: float = HT_SHARE[1]) -> dict:
        """Mercados por parte. Os golos de cada parte seguem Poissons independentes com
        uma fração fixa dos golos esperados no jogo (share_* é a fração da 1.ª parte)."""
        lam, mu = self.expected_goals(home, away)
        g = np.arange(MAX_GOALS + 1)
        out = {}
        for key, sh, sa in (("p1", share_home, share_away), ("p2", 1 - share_home, 1 - share_away)):
            m = np.outer(poisson.pmf(g, lam * sh), poisson.pmf(g, mu * sa))
            m /= m.sum()
            total = g[:, None] + g[None, :]
            out.update({
                f"{key}_casa": float(np.tril(m, -1).sum()), f"{key}_empate": float(np.trace(m)),
                f"{key}_fora": float(np.triu(m, 1).sum()),
                f"{key}_mais_0_5": float(m[total > 0.5].sum()),
                f"{key}_mais_1_5": float(m[total > 1.5].sum()),
                f"{key}_golos": lam * sh + mu * sa,
            })
        return out

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
