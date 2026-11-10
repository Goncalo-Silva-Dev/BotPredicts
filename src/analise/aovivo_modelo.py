"""Percentagens ao vivo: as probabilidades de antes do jogo atualizadas com o resultado e o tempo.

As contas são feitas no browser (aovivo.js), com os números de cada jogo que vão na página;
este módulo tem a mesma matemática em Python, que serviu para afinar e testar os modelos com
jogos passados (ver validacao/ao_vivo/) e que os testes comparam com o JavaScript.

Futebol: golos que faltam como processo minuto a minuto. A taxa de cada equipa é a do modelo
  (golos esperados no jogo) vezes o peso de cada minuto (há mais golos no fim de cada parte e
  nos descontos), vezes um fator do marcador (quem perde ataca mais) e outro das expulsões.
  Os pesos e fatores foram estimados em 5 313 jogos com o minuto de cada golo (ESPN, 8 ligas,
  2023-24 e 2024-25) e testados em 3 349 jogos de 2025-26 que não entraram na estimativa.
Basquetebol: diferença e total que faltam como normais; média e desvio dependem do tempo que
  falta e do marcador (quem vai à frente tende a perder parte da vantagem). Estimado com 4 000
  jogos da NBA e testado em 1 609 da época seguinte.
Hóquei: golos "normais" que faltam (Poisson, pela parte de cada período), mais o fim de jogo
  com a baliza deserta, como no modelo de antes do jogo.
Ténis: modelo de pontos (serviço) do jogo por sets e jogos; a forma do dia é aprendida com os
  sets e jogos já disputados (regra de Bayes sobre os 5 cenários do modelo de antes do jogo).
"""
import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.stats import norm

from analise.models import tennis as tennis_model
from analise.models.hockey import END_KERNEL, OT_HOME, PERIOD_SHARES

# --- Futebol -------------------------------------------------------------------------------
# Parte dos golos de um jogo em cada minuto: [0..44] = minutos 1 a 45, [45] = descontos da
# 1.ª parte (todos), [46..90] = minutos 46 a 90, [91] = descontos da 2.ª parte (todos).
SOCCER_PROFILE = [
    0.00624, 0.00687, 0.00739, 0.00809, 0.00848, 0.00893, 0.00909, 0.00895, 0.00909, 0.00918,
    0.00902, 0.00909, 0.00953, 0.00942, 0.00955, 0.0097, 0.00987, 0.00995, 0.00995, 0.00975,
    0.00976, 0.00952, 0.00967, 0.00969, 0.00962, 0.00955, 0.00949, 0.00967, 0.00971, 0.01007,
    0.00998, 0.00988, 0.00933, 0.00921, 0.00935, 0.00966, 0.00999, 0.01005, 0.01016, 0.01005,
    0.00978, 0.00988, 0.01003, 0.01014, 0.01001, 0.03663, 0.00695, 0.00818, 0.00943, 0.0108,
    0.01102, 0.01145, 0.01141, 0.01151, 0.0118, 0.01198, 0.01204, 0.01197, 0.0119, 0.01178,
    0.01161, 0.0112, 0.01163, 0.01153, 0.01123, 0.01104, 0.01101, 0.01062, 0.01081, 0.01058,
    0.01086, 0.01088, 0.01077, 0.01059, 0.0107, 0.01054, 0.0105, 0.01029, 0.01043, 0.01031,
    0.01035, 0.01044, 0.0109, 0.01076, 0.01065, 0.01089, 0.01087, 0.01067, 0.01082, 0.01126,
    0.01121, 0.07679,
]
# Fator da taxa de golos de uma equipa pelo marcador, do ponto de vista dela: a perder por 2
# ou mais, por 1, empatado, a ganhar por 1, por 2 ou mais
SOCCER_STATE = [1.119, 1.161, 1.051, 0.927, 0.925]
# Fator pelas expulsões: com menos jogadores, igual, com mais jogadores
SOCCER_RED = [0.514, 1.0, 1.65]
# Probabilidade de os descontos chegarem a cada minuto (1.º, 2.º, ...): 1.ª e 2.ª parte
SOCCER_STOP1 = [1, .88, .62, .47, .33, .22, .17, .09, .04, .02, .01]
SOCCER_STOP2 = [0.998, 0.992, 0.984, 0.951, 0.855, 0.686, 0.485, 0.317, 0.193, 0.114, 0.065,
                0.039, 0.024, 0.015, 0.008, 0.004, 0.002, 0.001, 0.001, 0.001, 0.001]
MAX_GOALS = 12


def soccer_steps(half: int, minute: int, stop: int | None) -> list[float]:
    """Peso de cada minuto que falta. minute = minutos já jogados na parte (0 a 45 / 45 a 90);
    stop = minutos de descontos já jogados (None se ainda não se está nos descontos)."""
    prof, s1, s2 = SOCCER_PROFILE, SOCCER_STOP1, SOCCER_STOP2
    c1, c2 = prof[45] / sum(s1), prof[91] / sum(s2)

    def stoppage(surv, c, done):
        base = surv[done - 1] if 0 < done <= len(surv) else (1.0 if done == 0 else surv[-1])
        return [c * surv[u] / base for u in range(done, len(surv))]
    w = []
    if half == 1:
        if stop is None:
            w += [prof[m - 1] for m in range(minute + 1, 46)] + stoppage(s1, c1, 0)
        else:
            w += stoppage(s1, c1, stop)
        w += [prof[m] for m in range(46, 91)] + stoppage(s2, c2, 0)
    elif stop is None:
        w += [prof[m] for m in range(max(minute + 1, 46), 91)] + stoppage(s2, c2, 0)
    else:
        w += stoppage(s2, c2, stop)
    return w


def soccer_added(lh, la, gh, ga, rh, ra, steps) -> np.ndarray:
    """[i, j] = probabilidade de a casa marcar mais i e a equipa de fora mais j até ao fim."""
    n = MAX_GOALS
    p = np.zeros((n, n))
    p[0, 0] = 1.0
    h, a = np.indices((n, n))
    d = np.clip(gh + h - ga - a, -2, 2)
    red_h = SOCCER_RED[max(-1, min(1, ra - rh)) + 1]
    red_a = SOCCER_RED[max(-1, min(1, rh - ra)) + 1]
    mh = np.array(SOCCER_STATE)[d + 2] * red_h
    ma = np.array(SOCCER_STATE)[-d + 2] * red_a
    for w in steps:
        ph, pa = 1 - np.exp(-lh * w * mh), 1 - np.exp(-la * w * ma)
        nxt = p * (1 - ph) * (1 - pa)
        nxt[1:, :] += (p * ph * (1 - pa))[:-1, :]
        nxt[:, 1:] += (p * (1 - ph) * pa)[:, :-1]
        nxt[1:, 1:] += (p * ph * pa)[:-1, :-1]
        p = nxt
    return p


def soccer_live(lh, la, gh, ga, rh, ra, half, minute, stop) -> dict:
    p = soccer_added(lh, la, gh, ga, rh, ra, soccer_steps(half, minute, stop))
    h, a = np.indices(p.shape)
    d, t = gh + h - ga - a, gh + ga + h + a
    return {"casa": float(p[d > 0].sum()), "empate": float(p[d == 0].sum()), "fora": float(p[d < 0].sum()),
            "mais_2_5": float(p[t > 2.5].sum()),
            "ambas": float(p[(gh + h > 0) & (ga + a > 0)].sum())}


# --- Basquetebol ----------------------------------------------------------------------------
# Parte do jogo que falta -> (a, b, s, c, d, u):
# diferença que falta = a * diferença prevista + b * vantagem atual, desvio s * desvio do modelo;
# total que falta = c * total previsto + d * (pontos até agora - previstos até agora), desvio u.
BASKET_FITS = {
    1.0: (1.0, 0.0, 1.0, 1.0, 0.0, 1.0),
    0.75: (0.745, -0.256, 0.877, 0.749, 0.128, 0.842),
    0.5: (0.532, -0.211, 0.759, 0.499, 0.052, 0.71),
    0.25: (0.233, -0.116, 0.546, 0.246, 0.041, 0.526),
    0.0: (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
}


def basket_coefs(tau: float) -> tuple:
    """Coeficientes para a parte do jogo que falta (0 a 1), por interpolação; os desvios
    interpolam-se ao quadrado (variâncias)."""
    tau = min(max(tau, 0.0), 1.0)
    keys = sorted(BASKET_FITS)
    hi = next(k for k in keys if k >= tau)
    lo = keys[max(keys.index(hi) - 1, 0)]
    f = 0.0 if hi == lo else (tau - lo) / (hi - lo)
    a0, a1 = BASKET_FITS[lo], BASKET_FITS[hi]
    lin = lambda i: a0[i] + f * (a1[i] - a0[i])  # noqa: E731
    var = lambda i: math.sqrt(a0[i] ** 2 + f * (a1[i] ** 2 - a0[i] ** 2))  # noqa: E731
    return lin(0), lin(1), var(2), lin(3), lin(4), var(5)


def basket_live(margin, total, sd_margin, sd_total, lead, points, tau) -> dict:
    """margin/total previstos antes do jogo; lead = casa - fora agora; points = total agora."""
    a, b, s, c, d, u = basket_coefs(tau)
    mu = lead + a * margin + b * lead
    sd = max(s * sd_margin, 0.05)
    p_ot = float(norm.cdf(margin * math.sqrt(5 / 48) / sd_margin))
    over, tie = 1 - norm.cdf((0.5 - mu) / sd), norm.cdf((0.5 - mu) / sd) - norm.cdf((-0.5 - mu) / sd)
    tot = points + c * total + d * (points - (1 - tau) * total)
    return {"casa": float(over + tie * p_ot), "margem": float(mu), "sd_margem": sd,
            "total": float(tot), "sd_total": max(u * sd_total, 0.05)}


# --- Hóquei ---------------------------------------------------------------------------------
def hockey_remaining_share(side: str, period: int, minutes_left: float) -> float:
    sh = list(PERIOD_SHARES[side]) + [1 - sum(PERIOD_SHARES[side])]
    if period > 3:
        return 0.0
    return sh[period - 1] * minutes_left / 20 + sum(sh[period:])


def hockey_live(lam_h, lam_a, gh, ga, period, minutes_left) -> dict:
    from scipy.stats import poisson
    rh = lam_h * hockey_remaining_share("home", period, minutes_left)
    ra = lam_a * hockey_remaining_share("away", period, minutes_left)
    n = 14
    g = np.arange(n)
    m = np.outer(poisson.pmf(g, rh), poisson.pmf(g, ra))
    left = (3 - period) * 20 + minutes_left if period <= 3 else 0.0
    k = min(1.0, left / 3.0)  # o fim de jogo (baliza deserta) acontece nos últimos minutos
    p1, q1, p2, q2 = (x * k for x in END_KERNEL[1:])
    out = np.zeros((n + 1, n + 1))
    for i in range(n):
        for j in range(n):
            v, diff = m[i, j], (gh + i) - (ga + j)
            if abs(diff) not in (1, 2):
                out[i, j] += v
                continue
            lead, trail = (p1, q1) if abs(diff) == 1 else (p2, q2)
            if diff > 0:
                out[i + 1, j] += v * lead
                out[i, j + 1] += v * trail
            else:
                out[i, j + 1] += v * lead
                out[i + 1, j] += v * trail
            out[i, j] += v * (1 - lead - trail)
    h, a = np.indices(out.shape)
    diff = gh + h - ga - a
    tie = float(out[diff == 0].sum())
    return {"casa": float(out[diff > 0].sum()) + tie * OT_HOME, "reg_casa": float(out[diff > 0].sum()),
            "reg_empate": tie, "reg_fora": float(out[diff < 0].sum())}


# --- Ténis ----------------------------------------------------------------------------------
TENNIS_Z = [float(z) for z in tennis_model._HERMITE_Z]
TENNIS_W = [float(w) for w in tennis_model._HERMITE_W]


def _set_tables(pa: float, pb: float):
    """Para um set: vitória de A a partir de (i, j) e probabilidade de chegar a (i, j),
    conforme quem serve o 1.º jogo do set."""
    ga, gb = tennis_model.game_prob(pa), 1 - tennis_model.game_prob(pb)

    @lru_cache(None)
    def win(i, j, a_serves):
        if (i >= 6 and i - j >= 2) or i == 7:
            return 1.0
        if (j >= 6 and j - i >= 2) or j == 7:
            return 0.0
        if i == 6 and j == 6:
            x, y = (pa, 1 - pb) if a_serves else (1 - pb, pa)
            t = tennis_model.tiebreak_prob(x, y)
            return t if a_serves else 1 - t
        p = ga if a_serves else gb
        return p * win(i + 1, j, not a_serves) + (1 - p) * win(i, j + 1, not a_serves)

    def reach(a_first):
        r = np.zeros((8, 8))
        r[0, 0] = 1.0
        for n in range(13):
            for i in range(0, n + 1):
                j = n - i
                if i > 7 or j > 7 or r[i, j] == 0:
                    continue
                if (i >= 6 and i - j >= 2) or (j >= 6 and j - i >= 2) or i == 7 or j == 7:
                    continue
                if i == 6 and j == 6:
                    t = win(6, 6, a_first)  # 12 jogos: quem começou o set serve no tie-break
                    r[7, 6] += r[6, 6] * t
                    r[6, 7] += r[6, 6] * (1 - t)
                    continue
                a_serves = a_first if n % 2 == 0 else not a_first
                p = ga if a_serves else gb
                r[i + 1, j] += r[i, j] * p
                r[i, j + 1] += r[i, j] * (1 - p)
        return r
    return win, reach(True), reach(False)


def tennis_live(base, sigma, d0, best_of, sets_done, current) -> dict:
    """sets_done: [(jogos A, jogos B), ...] dos sets acabados; current: (i, j) do set a decorrer."""
    need = best_of // 2 + 1
    sa = sum(1 for x, y in sets_done if x > y)
    sb = len(sets_done) - sa
    num_match = num_set = den = 0.0
    for z, w in zip(TENNIS_Z, TENNIS_W):
        d = d0 + sigma * z
        pa, pb = min(max(base + d / 2, 0.2), 0.97), min(max(base - d / 2, 0.2), 0.97)
        win, r_a, r_b = _set_tables(round(pa, 3), round(pb, 3))
        lik = 1.0
        for x, y in sets_done:
            lik *= max(0.5 * (r_a[x, y] + r_b[x, y]), 1e-9)
        i, j = current
        la, lb = r_a[i, j], r_b[i, j]
        if la + lb <= 0:
            la = lb = 1.0
        lik *= 0.5 * (la + lb)
        # quem serve agora: se A serviu primeiro no set, serve A com i + j par
        a_now_if_a_first = (i + j) % 2 == 0
        pc = (la * win(i, j, a_now_if_a_first) + lb * win(i, j, not a_now_if_a_first)) / (la + lb)
        ps = 0.5 * (win(0, 0, True) + win(0, 0, False))

        @lru_cache(None)
        def match(x, y):
            if x == need:
                return 1.0
            if y == need:
                return 0.0
            return ps * match(x + 1, y) + (1 - ps) * match(x, y + 1)
        pm = pc * match(sa + 1, sb) + (1 - pc) * match(sa, sb + 1)
        num_match += w * lik * pm
        num_set += w * lik * pc
        den += w * lik
    return {"casa": num_match / den, "set": num_set / den}


# --- Dados para a página --------------------------------------------------------------------
JS_PATH = Path(__file__).with_name("aovivo.js")


def constants() -> dict:
    return {
        "soccer": {"profile": SOCCER_PROFILE, "state": SOCCER_STATE, "red": SOCCER_RED,
                   "stop1": SOCCER_STOP1, "stop2": SOCCER_STOP2},
        "basket": {str(k): v for k, v in BASKET_FITS.items()},
        "hockey": {"kernel": list(END_KERNEL[1:]), "ot": OT_HOME,
                   "home": list(PERIOD_SHARES["home"]), "away": list(PERIOD_SHARES["away"])},
        "tennis": {"z": TENNIS_Z, "w": TENNIS_W},
    }


def script() -> str:
    """O JavaScript dos resultados e das percentagens ao vivo, com as constantes dos modelos."""
    return JS_PATH.read_text(encoding="utf-8").replace("__LV__", json.dumps(constants(), separators=(",", ":")))


def _r(x, k=4):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), k)


def card_params(sport: str, g) -> dict | None:
    """Números de antes do jogo que o browser precisa para as contas ao vivo."""
    try:
        if sport == "futebol":
            if g.get("golos_casa") is None or math.isnan(g["golos_casa"]):
                return None
            return {"k": "f", "l": [_r(g["golos_casa"]), _r(g["golos_fora"])]}
        if sport == "basquetebol":
            mk = g.get("mercados")
            if not isinstance(mk, dict) or g.get("sd_diferenca") is None or math.isnan(g["sd_diferenca"]):
                return None
            return {"k": "b", "m": _r(g["diferenca"], 2), "t": _r(g["total"], 2),
                    "sm": _r(g["sd_diferenca"], 3), "st": _r(g["sd_total"], 3),
                    "q": 10 if g["league"] != "NBA" else 12,
                    "hl": [l for l, _ in mk["handicap"]], "tl": [l for l, _ in mk["total"]]}
        if sport == "hoquei":
            if g.get("lam_casa") is None or math.isnan(g["lam_casa"]):
                return None
            mk = g["mercados"]
            return {"k": "h", "l": [_r(g["lam_casa"]), _r(g["lam_fora"])], "tl": [l for l, _ in mk["total"]]}
        if sport == "tenis":
            base, sigma = tennis_model.POINT_MODEL[(g["league"], g["surface"])]
            p3 = g.get("p_bo3", g["casa"])
            return {"k": "t", "b": base, "s": sigma, "d": _r(tennis_model.serve_diff(p3, base, sigma), 5),
                    "n": int(g["best_of"]), "id": str(g.get("home_id") or ""),
                    "nh": str(g["home"]), "na": str(g["away"])}
    except (KeyError, TypeError, ValueError):
        return None
    return None
