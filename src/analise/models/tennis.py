"""Modelo de ténis: Elo (geral e por piso) para quem ganha, e um modelo de pontos para o resto.

1. Elo: cada jogador tem uma força geral e uma por piso (terra, relva, rápido). Depois de cada
   jogo, quem ganha sobe e quem perde desce, mais quando o resultado surpreende e quando o
   jogador tem poucos jogos (K = 250 / (jogos + 5) ** 0.4, como no Elo de ténis da FiveThirtyEight).
2. Modelo de pontos: a probabilidade de vitória à melhor de 3 sets do Elo passa a uma
   probabilidade de ganhar cada ponto no serviço (base do circuito e do piso ± diferença dos
   jogadores). Com ela calcula-se, sem simulações, a distribuição exata de jogos, sets e
   tie-breaks: resultado em sets, total de jogos, handicap de jogos, 1.º set, etc.
"""
from functools import lru_cache

import numpy as np

# (probabilidade média de ganhar um ponto no serviço, desvio da forma do dia) por circuito e
# piso, afinados no backtest para que o total de jogos e os tie-breaks batam certo
POINT_MODEL = {
    ("ATP", "rapido"): (0.64, 0.07), ("ATP", "terra"): (0.63, 0.07), ("ATP", "relva"): (0.65, 0.07),
    ("WTA", "rapido"): (0.58, 0.10), ("WTA", "terra"): (0.56, 0.11), ("WTA", "relva"): (0.60, 0.10),
}
MAX_GAMES = 7 * 5 + 1


def game_prob(p: float) -> float:
    """Probabilidade de quem serve ganhar o jogo, com probabilidade p de ganhar cada ponto."""
    q = 1 - p
    deuce = p * p / (1 - 2 * p * q)
    return p ** 4 * (1 + 4 * q + 10 * q * q) + 20 * p ** 3 * q ** 3 * deuce


def tiebreak_prob(x: float, y: float) -> float:
    """A ganha o tie-break; x = A ganha o ponto quando serve, y = A ganha quando B serve.
    A serve o 1.º ponto, depois cada um serve 2 seguidos."""
    dp = np.zeros((8, 8))
    dp[0, 0] = 1.0
    win = 0.0
    for n in range(12):  # até 6-6
        for i in range(max(0, n - 6), min(n, 6) + 1):
            j = n - i
            if j > 6 or dp[i, j] == 0:
                continue
            p = x if ((n + 1) // 2) % 2 == 0 else y
            for di, pr in ((1, p), (0, 1 - p)):
                a, b = i + di, j + 1 - di
                if a == 7 and b <= 5:
                    win += dp[i, j] * pr
                elif b == 7 and a <= 5:
                    pass
                else:
                    dp[a, b] += dp[i, j] * pr
    both, none = x * y, (1 - x) * (1 - y)
    return win + dp[6, 6] * both / (both + none)


@lru_cache(maxsize=200_000)
def set_outcomes(pa: float, pb: float, a_serves: bool) -> tuple:
    """((jogos A, jogos B, houve tie-break, probabilidade), ...) de um set; pa e pb são as
    probabilidades de cada um ganhar um ponto no seu serviço."""
    ga, gb = game_prob(pa), 1 - game_prob(pb)  # A ganha o jogo quando serve / quando B serve
    dp = np.zeros((7, 7))
    dp[0, 0] = 1.0
    out = []
    for n in range(12):
        for i in range(max(0, n - 6), min(n, 6) + 1):
            j = n - i
            if j > 6 or dp[i, j] == 0:
                continue
            if (i == 6 and j <= 4) or (j == 6 and i <= 4) or (i == 7) or (j == 7):
                continue
            p = ga if (n % 2 == 0) == a_serves else gb
            for di, pr in ((1, p), (0, 1 - p)):
                a, b = i + di, j + 1 - di
                if (a == 6 and b <= 4) or (b == 6 and a <= 4):
                    out.append((a, b, False, dp[i, j] * pr))
                elif a == 7 or b == 7:
                    out.append((a, b, False, dp[i, j] * pr))
                else:
                    dp[a, b] += dp[i, j] * pr
    # 7-5 e 5-7 vêm de 6-5 / 5-6; 6-6 vai a tie-break
    tie = dp[6, 6]
    if tie > 0:
        x, y = (pa, 1 - pb) if a_serves else (1 - pb, pa)
        t = tiebreak_prob(x, y)  # probabilidade de quem serve primeiro no tie-break o ganhar
        t_a = t if a_serves else 1 - t
        out += [(7, 6, True, tie * t_a), (6, 7, True, tie * (1 - t_a))]
    return tuple(out)


def match_distribution(pa: float, pb: float, best_of: int = 3) -> dict:
    """Distribuição exata do jogo. Quem serve primeiro é sorteado (metade das vezes cada um)."""
    need = best_of // 2 + 1
    games = np.zeros((MAX_GAMES, MAX_GAMES))
    sets = {}
    first_set = 0.0
    tiebreak = 0.0
    for a_first in (True, False):
        # estado: (sets A, sets B, A serve no início do set, houve tie-break) -> jogos A x jogos B
        states = {(0, 0, a_first, False): _point(0, 0)}
        while states:
            nxt = {}
            for (sa, sb, a_serves, tb), mat in states.items():
                for x, y, is_tb, pr in set_outcomes(round(pa, 4), round(pb, 4), a_serves):
                    w = 0.5 * pr
                    if sa == 0 and sb == 0 and x > y:
                        first_set += w
                    na, nb = sa + (x > y), sb + (y > x)
                    shifted = np.zeros_like(mat)
                    shifted[x:, y:] = mat[:MAX_GAMES - x, :MAX_GAMES - y]
                    shifted *= pr
                    serve_next = a_serves if (x + y) % 2 == 0 else not a_serves
                    t = tb or is_tb
                    if na == need or nb == need:
                        games += 0.5 * shifted
                        sets[(na, nb)] = sets.get((na, nb), 0.0) + 0.5 * shifted.sum()
                        if t:
                            tiebreak += 0.5 * shifted.sum()
                    else:
                        key = (na, nb, serve_next, t)
                        nxt[key] = nxt[key] + shifted if key in nxt else shifted
            states = nxt
    win = sum(p for (a, b), p in sets.items() if a > b)
    return {"win": win, "sets": dict(sorted(sets.items(), key=lambda kv: (-kv[0][0], kv[0][1]))),
            "games": games, "first_set": first_set, "tiebreak": tiebreak}


def _point(i, j):
    m = np.zeros((MAX_GAMES, MAX_GAMES))
    m[i, j] = 1.0
    return m


# Forma do dia: a diferença entre os jogadores no serviço varia de jogo para jogo (normal com
# desvio sigma). Sem isto o modelo de pontos dá jogos equilibrados a mais (mais jogos e mais
# tie-breaks do que na realidade, sobretudo na WTA).
_HERMITE_Z, _HERMITE_W = np.polynomial.hermite_e.hermegauss(5)
_HERMITE_W = _HERMITE_W / _HERMITE_W.sum()


def mixed_distribution(base: float, d0: float, sigma: float, best_of: int = 3) -> dict:
    """match_distribution com a diferença no serviço d ~ Normal(d0, sigma)."""
    parts = []
    for z, w in zip(_HERMITE_Z, _HERMITE_W) if sigma > 0 else [(0.0, 1.0)]:
        d = d0 + sigma * z
        pa, pb = min(max(base + d / 2, 0.2), 0.97), min(max(base - d / 2, 0.2), 0.97)
        parts.append((w, match_distribution(round(pa, 3), round(pb, 3), best_of)))
    sets = {}
    for w, dist in parts:
        for k, v in dist["sets"].items():
            sets[k] = sets.get(k, 0.0) + w * v
    win = sum(p for (a, b), p in sets.items() if a > b)
    return {"win": win, "sets": dict(sorted(sets.items(), key=lambda kv: (-kv[0][0], kv[0][1]))),
            "games": sum(w * dist["games"] for w, dist in parts),
            "first_set": sum(w * dist["first_set"] for w, dist in parts),
            "tiebreak": sum(w * dist["tiebreak"] for w, dist in parts)}


@lru_cache(maxsize=64)
def _bo3_curve(base: float, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Probabilidade de vitória à melhor de 3 em função da diferença média no serviço."""
    ds = np.linspace(0, 0.40, 81)
    probs = np.array([mixed_distribution(base, d, sigma, 3)["win"] for d in ds])
    return ds, probs


def serve_diff(p_bo3: float, base: float, sigma: float) -> float:
    """Diferença média no serviço (A menos B) que dá p_bo3 à melhor de 3."""
    ds, probs = _bo3_curve(round(base, 4), round(sigma, 4))
    p = min(max(p_bo3, 1e-4), 1 - 1e-4)
    d = float(np.interp(abs(p - 0.5) + 0.5, probs, ds))
    return d if p >= 0.5 else -d


def predict_distribution(p_bo3: float, league: str, surface: str, best_of: int) -> dict:
    base, sigma = POINT_MODEL[(league, surface)]
    return mixed_distribution(base, serve_diff(p_bo3, base, sigma), sigma, best_of)


class TennisElo:
    """Elo geral e por piso; prever com predict() e atualizar com update() jogo a jogo."""

    def __init__(self, surface_weight: float = 0.5, initial: float = 1500.0, k_scale: float = 250.0,
                 k_offset: float = 5.0, k_shape: float = 0.4, mov: float = 0.0,
                 mov_scale: float = 3.0, new_offset: float = 0.0):
        self.mov, self.mov_scale, self.new_offset = mov, mov_scale, new_offset
        self.w, self.start = surface_weight, initial
        self.k_scale, self.k_offset, self.k_shape = k_scale, k_offset, k_shape
        self.rating, self.n = {}, {}

    def _r(self, key):
        # quem aparece pela primeira vez (muitas vezes vindo da qualificação) começa abaixo da média
        return self.rating.get(key, self.start - self.new_offset)

    def strength(self, player: str, surface: str) -> float:
        return (1 - self.w) * self._r(player) + self.w * self._r((player, surface))

    def games(self, player: str) -> int:
        return self.n.get(player, 0)

    def predict(self, a: str, b: str, surface: str) -> float:
        """Probabilidade de A ganhar à melhor de 3 sets."""
        diff = self.strength(a, surface) - self.strength(b, surface)
        return 1 / (1 + 10 ** (-diff / 400))

    def _k(self, key):
        return self.k_scale / (self.n.get(key, 0) + self.k_offset) ** self.k_shape

    def update(self, a: str, b: str, surface: str, a_won: bool, expected: float | None = None,
               game_share: float | None = None):
        """expected: probabilidade de A ganhar no formato do jogo (melhor de 3 ou de 5);
        game_share: parte dos jogos ganhos por A (conta com peso self.mov no resultado)."""
        y = 1.0 if a_won else 0.0
        if game_share is not None and self.mov > 0:
            y = (1 - self.mov) * y + self.mov * min(max(0.5 + self.mov_scale * (game_share - 0.5), 0.0), 1.0)
        for ka, kb in ((a, b), ((a, surface), (b, surface))):
            e = 1 / (1 + 10 ** (-(self._r(ka) - self._r(kb)) / 400))
            if expected is not None and ka == a:
                e = expected
            delta_a, delta_b = self._k(ka) * (y - e), self._k(kb) * (y - e)
            self.rating[ka] = self._r(ka) + delta_a
            self.rating[kb] = self._r(kb) - delta_b
            self.n[ka] = self.n.get(ka, 0) + 1
            self.n[kb] = self.n.get(kb, 0) + 1


def markets(dist: dict, best_of: int) -> dict:
    """Mercados a partir da distribuição exata: linhas como as das casas de apostas."""
    g = dist["games"]
    ga = np.arange(MAX_GAMES)
    total = np.zeros(2 * MAX_GAMES)
    diff = {}
    for i in range(MAX_GAMES):
        for j in range(MAX_GAMES):
            if g[i, j] > 0:
                total[i + j] += g[i, j]
                diff[i - j] = diff.get(i - j, 0.0) + g[i, j]
    mean_total = float((np.arange(len(total)) * total).sum())
    centre = int(round(mean_total)) + 0.5
    total_lines = [c for c in np.arange(centre - 3, centre + 3.5, 1.0) if c > 12]
    over = lambda line: float(total[int(np.ceil(line)):].sum())
    diffs = np.array(sorted(diff))
    probs = np.array([diff[d] for d in diffs])
    hcp = lambda line: float(probs[diffs + line > 0].sum())  # A com handicap `line` ganha
    a_games = g.sum(axis=1)
    b_games = g.sum(axis=0)
    mean_a, mean_b = float((ga * a_games).sum()), float((ga * b_games).sum())
    pl_lines = lambda m: [x + 0.5 for x in range(int(m) - 2, int(m) + 2) if x >= 2]
    sets_needed = best_of // 2 + 1
    return {
        "jogos_esperados": [mean_a, mean_b], "total_esperado": mean_total,
        "total": [(float(l), over(l)) for l in total_lines],
        "handicap": [(float(l), hcp(l)) for l in np.arange(-5.5, 6.0, 1.0)],
        "jogos_casa": [(float(l), float(a_games[int(np.ceil(l)):].sum())) for l in pl_lines(mean_a)],
        "jogos_fora": [(float(l), float(b_games[int(np.ceil(l)):].sum())) for l in pl_lines(mean_b)],
        "sets": [(f"{a}-{b}", p) for (a, b), p in dist["sets"].items()],
        "primeiro_set": dist["first_set"],
        "tiebreak": dist["tiebreak"],
        "todos_os_sets": sum(p for (a, b), p in dist["sets"].items() if min(a, b) == sets_needed - 1),
        "casa_ganha_set": 1 - dist["sets"].get((0, sets_needed), 0.0),
        "fora_ganha_set": 1 - dist["sets"].get((sets_needed, 0), 0.0),
    }
