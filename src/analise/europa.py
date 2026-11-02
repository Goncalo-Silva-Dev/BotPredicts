"""Competições europeias (Liga dos Campeões, Liga Europa e Liga Conferência).

As equipas vêm de dezenas de ligas, por isso um modelo treinado só com os jogos europeus
(10 a 20 por equipa e por época) fica muito fraco. O modelo das taças é um Dixon-Coles
único treinado com os jogos europeus e com os campeonatos nacionais que temos: os jogos
europeus ligam as ligas entre si e os campeonatos dão a força de cada equipa no dia a dia.

Os nomes das equipas no betexplorer (taças) e no football-data.co.uk (campeonatos) nem
sempre coincidem; team_map liga-os com regras conservadoras, porque juntar duas equipas
diferentes (p. ex. Sparta Praga e Sparta Roterdão) estraga as duas.
"""
import re
import unicodedata

import pandas as pd

CUPS = ("UCL", "UEL", "UECL")

# Nome no betexplorer -> (liga, nome no histórico) quando as regras não chegam
ALIASES = {
    "Manchester City": ("ENG", "Man City"),
    "Manchester Utd": ("ENG", "Man United"),
    "Nottingham": ("ENG", "Nott'm Forest"),
    "Newcastle Utd": ("ENG", "Newcastle"),
    "PSG": ("FRA", "Paris SG"),
    "Sporting CP": ("POR", "Sp Lisbon"),
    "Braga": ("POR", "Sp Braga"),
    "Vitoria Guimaraes": ("POR", "Guimaraes"),
    "Atl. Madrid": ("ESP", "Ath Madrid"),
    "Real Sociedad": ("ESP", "Sociedad"),
    "Rayo Vallecano": ("ESP", "Vallecano"),
    "Celta Vigo": ("ESP", "Celta"),
    "Espanyol": ("ESP", "Espanol"),
    "Eintracht Frankfurt": ("GER", "Ein Frankfurt"),
    "Bayer Leverkusen": ("GER", "Leverkusen"),
    "B. Monchengladbach": ("GER", "M'gladbach"),
    "Olympiacos Piraeus": ("GRE", "Olympiakos"),
    "AEK Athens": ("GRE", "AEK"),
    "Basaksehir": ("TUR", "Buyuksehyr"),
    "Adana Demirspor": ("TUR", "Ad. Demirspor"),
    "Royale Union SG": ("BEL", "St. Gilloise"),
    "Club Brugge KV": ("BEL", "Club Brugge"),
    "Cercle Brugge KSV": ("BEL", "Cercle Brugge"),
    "PSV": ("NED", "PSV Eindhoven"),
    "G.A. Eagles": ("NED", "Go Ahead Eagles"),
    "Dundee Utd": ("SCO", "Dundee United"),
}

# Palavras que não identificam um clube (para a comparação de nomes)
_STOP = {"fc", "cf", "ac", "as", "sc", "afc", "kv", "ksv", "fk", "sk", "if", "ff", "bk", "cp", "club"}


def _tokens(name: str) -> frozenset:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    return frozenset(t for t in re.sub(r"[^a-z0-9]", " ", s).split() if t not in _STOP)


def team_map(cup_names, domestic: dict[str, set[str]]) -> dict[str, tuple[str, str]]:
    """Liga cada equipa das taças à (liga, nome) do campeonato, só quando é claro.

    Regras, por ordem: ALIASES; nome igual (sem acentos, pontuação nem "FC"/"KV"...) numa
    só liga; um nome contido no outro (p. ex. "Bodo/Glimt") quando só há um candidato
    de cada lado. O resto fica sem campeonato e conta só com os jogos europeus.
    """
    pairs = [(league, name) for league, names in domestic.items() for name in names]
    out, used = {}, set()
    for cup in cup_names:
        target = ALIASES.get(cup)
        if target and target[1] in domestic.get(target[0], ()):
            out[cup] = target
            used.add(target)
    by_tokens: dict[frozenset, list] = {}
    for p in pairs:
        by_tokens.setdefault(_tokens(p[1]), []).append(p)
    left = [c for c in cup_names if c not in out]
    for cup in left:
        same = [p for p in by_tokens.get(_tokens(cup), []) if p not in used]
        if len(same) == 1 and _tokens(cup):
            out[cup] = same[0]
            used.add(same[0])
    left = [c for c in left if c not in out]
    free = [p for p in pairs if p not in used]
    for cup in left:
        tc = _tokens(cup)
        cands = [p for p in free if _tokens(p[1]) and (_tokens(p[1]) <= tc or tc <= _tokens(p[1]))]
        if len(cands) != 1:
            continue
        tp = _tokens(cands[0][1])
        rivals = [c for c in left if tp <= _tokens(c) or _tokens(c) <= tp]
        if rivals == [cup]:
            out[cup] = cands[0]
    return out


# Campeonatos de fora da Europa ou que não são a 1.ª divisão (não ajudam a medir as equipas
# das taças e podiam repetir nomes)
NOT_DOMESTIC = {"USA", "BRA", "SAU", "PO2", "U23"}
# Os jogos europeus pesam mais do que os do campeonato: são os únicos que comparam ligas
CUP_WEIGHT = 2.0
# Parâmetros escolhidos no teste walk-forward de 2022-23 a 2026-27 (3410 jogos): a meia-vida,
# o peso e a penalização mudam pouco o resultado (log loss 0,948 a 0,961); o que mais ajudou foi
# juntar os campeonatos (0,99 -> 0,96) e ajustar o rho à parte. Mais 33 campeonatos pequenos
# do betexplorer (Leste europeu, Balcãs...) pioraram o teste, por isso ficaram de fora.
HALF_LIFE = 365.0
# penalização das forças (equipas com poucos jogos ficam perto da média); rho à parte, porque
# o ajuste conjunto falha com mais de mil equipas; remates à baliza onde os houver
MODEL_OPTIONS = {"ridge": 0.5, "joint_rho": False, "min_shots_share": 0.1}
# jogos mínimos de cada equipa no treino (campeonato + taças) para haver previsão
MIN_MATCHES = 3


def training_history(all_matches: pd.DataFrame) -> pd.DataFrame:
    """Jogos das três taças + campeonatos, com os nomes do campeonato trocados pelos das taças.

    Equipas do campeonato sem correspondência ficam "LIGA:nome", para não se confundirem com
    equipas de outros países com o mesmo nome.
    """
    # cup: vantagem de casa e nível de golos próprios das taças (ver DixonColes.fit)
    cups = all_matches[all_matches["league"].isin(CUPS)].assign(weight=CUP_WEIGHT, cup=True)
    dom = all_matches[~all_matches["league"].isin(CUPS) & ~all_matches["league"].isin(NOT_DOMESTIC)]
    names = {league: set(g["home"]) | set(g["away"]) for league, g in dom.groupby("league")}
    mapping = {(l, n): cup for cup, (l, n) in
               team_map(sorted(set(cups["home"]) | set(cups["away"])), names).items()}
    dom = dom.assign(
        home=[mapping.get((l, n), f"{l}:{n}") for l, n in zip(dom["league"], dom["home"])],
        away=[mapping.get((l, n), f"{l}:{n}") for l, n in zip(dom["league"], dom["away"])],
        weight=1.0, cup=False)
    return pd.concat([cups, dom], ignore_index=True)


class OddsRatings:
    """Força de cada equipa tirada das odds de fecho dos jogos JÁ JOGADOS (nunca do jogo a prever).

    As odds de um jogo resumem o que o mercado sabia das duas equipas (lesões, forma, plantel);
    juntando muitos jogos, cada equipa fica com uma força que vale também contra adversários de
    outras ligas. Para cada jogo, y = ln(P(casa) / P(fora)) sem margem; ajusta-se
    y ≈ força casa − força fora + vantagem de casa (+ extra nas taças) por mínimos quadrados
    pesados (os jogos recentes e os das taças pesam mais) com uma pequena penalização.
    O empate vem de uma curva ajustada às odds das taças: P(empate) em função de |y|.
    """

    def __init__(self, half_life_days=365.0, ridge=0.1, cup_weight=4.0):
        self.half_life_days, self.ridge, self.cup_weight = half_life_days, ridge, cup_weight

    def fit(self, train: pd.DataFrame, as_of) -> "OddsRatings":
        import numpy as np
        from scipy.sparse import csr_matrix, hstack

        from analise.models.dixon_coles import fair_probs

        df = train.copy()
        probs = []
        for r in df.to_dict("records"):
            p = (np.nan,) * 3
            for kind in ("pin", "avg"):
                o = [r.get(f"odds_{x}_{kind}") for x in ("home", "draw", "away")]
                if all(v is not None and pd.notna(v) and v > 1 for v in o):
                    p = fair_probs(*o)
                    break
            probs.append(p)
        df[["ph", "pd", "pa"]] = probs
        df = df.dropna(subset=["ph"])
        cup = df["cup"].fillna(False).to_numpy(dtype=bool) if "cup" in df else np.zeros(len(df), bool)
        y = np.log(df["ph"].to_numpy() / df["pa"].to_numpy())
        age = (pd.Timestamp(as_of) - pd.to_datetime(df["date"])).dt.days.to_numpy()
        w = np.exp(-np.log(2) * np.clip(age, 0, None) / self.half_life_days)
        w = w * np.where(cup, self.cup_weight, 1.0)

        self.teams_ = sorted(set(df["home"]) | set(df["away"]))
        idx = {t: i for i, t in enumerate(self.teams_)}
        n, m = len(self.teams_), len(df)
        hi, ai = df["home"].map(idx).to_numpy(), df["away"].map(idx).to_numpy()
        rows = np.arange(m)
        a = csr_matrix((np.r_[np.ones(m), -np.ones(m)], (np.r_[rows, rows], np.r_[hi, ai])), shape=(m, n))
        a = hstack([a, csr_matrix(np.ones((m, 1))), csr_matrix(cup[:, None].astype(float))]).tocsr()
        wa = a.multiply(w[:, None]).T
        mat = (wa @ a).toarray()
        mat[np.arange(n), np.arange(n)] += self.ridge
        mat[n:, n:] += np.eye(2) * 1e-6  # sem jogos de campeonato, casa e casa-taça confundem-se
        theta = np.linalg.solve(mat, wa @ y)
        self.rating_, self.home_, self.cup_home_ = theta[:n], theta[n], theta[n + 1]
        # curva do empate (só taças; se houver poucas, todos os jogos)
        sel = cup if cup.sum() >= 200 else np.ones(m, bool)
        x = np.c_[np.ones(sel.sum()), np.abs(y[sel]), y[sel] ** 2]
        self.draw_ = np.linalg.lstsq(x, df["pd"].to_numpy()[sel], rcond=None)[0]
        return self

    def predict(self, home: str, away: str, cup: bool = True) -> tuple[float, float, float] | None:
        import numpy as np
        if home not in self.teams_ or away not in self.teams_:
            return None
        y = (self.rating_[self.teams_.index(home)] - self.rating_[self.teams_.index(away)]
             + self.home_ + (self.cup_home_ if cup else 0.0))
        draw = float(np.clip(self.draw_[0] + self.draw_[1] * abs(y) + self.draw_[2] * y * y, 0.03, 0.4))
        r = float(np.exp(y))
        home_p = (1 - draw) * r / (1 + r)
        return home_p, draw, 1 - draw - home_p


# Peso das forças tiradas das odds no 1X2 final (o resto é o Dixon-Coles dos golos); no teste
# walk-forward o melhor ficou entre 0,6 e 0,8
ODDS_WEIGHT = 0.8


def align_with_odds(model, train, fixtures, as_of) -> None:
    """Desloca a diferença de golos esperada de cada jogo para que vitória casa − vitória fora
    do Dixon-Coles fique igual à mistura com as forças das odds; assim todos os mercados
    (dupla hipótese, handicaps, por parte...) continuam coerentes com o 1X2."""
    from scipy.optimize import brentq

    ratings = OddsRatings().fit(train, as_of)
    for g in fixtures.itertuples(index=False):
        if g.home not in model.teams_ or g.away not in model.teams_:
            continue
        r = ratings.predict(g.home, g.away)
        if r is None:
            continue
        p = model.predict(g.home, g.away)
        target = ODDS_WEIGHT * (r[0] - r[2]) + (1 - ODDS_WEIGHT) * (p["home"] - p["away"])

        def gap(shift):
            model.set_shift(g.home, g.away, shift)
            q = model.predict(g.home, g.away)
            return q["home"] - q["away"] - target

        try:
            model.set_shift(g.home, g.away, brentq(gap, -4.0, 4.0, xtol=1e-4))
        except ValueError:  # alvo fora do alcance: fica sem ajuste
            model.set_shift(g.home, g.away, 0.0)


def walk_forward(all_matches: pd.DataFrame, start, window_days: int = 730) -> pd.DataFrame:
    """Teste: cada semana de jogos das taças prevista só com os jogos anteriores, com o mesmo
    código das previsões do site; junta as odds de fecho (sem margem) para comparar."""
    import numpy as np

    from analise import previsoes
    from analise.models.dixon_coles import fair_probs

    hist = training_history(all_matches)
    hist["date"] = pd.to_datetime(hist["date"])
    cups = hist[hist["league"].isin(CUPS)].dropna(subset=["home_goals", "away_goals"])
    cups = cups[cups["date"] >= pd.Timestamp(start)]
    weeks = cups["date"].dt.to_period("W-SUN").dt.start_time
    rows = []
    for week, games in cups.groupby(weeks):
        fx = games.rename(columns={f"odds_{o}_avg": f"odds_{o}" for o in ("home", "draw", "away")})
        fx = fx.assign(time=None, date=fx["date"].dt.date)
        res = previsoes.predict(hist, fx, as_of=week.date(), half_life_days=HALF_LIFE,
                                window_days=window_days, min_team_matches=MIN_MATCHES,
                                model_options=MODEL_OPTIONS, prepare=align_with_odds)
        for g, p in zip(games.itertuples(index=False), res.to_dict("records")):
            odds = (g.odds_home_avg, g.odds_draw_avg, g.odds_away_avg)
            m = fair_probs(*odds) if all(pd.notna(o) and o > 1 for o in odds) else (np.nan,) * 3
            rows.append({"league": g.league, "date": g.date.date(), "home": g.home, "away": g.away,
                         "home_goals": g.home_goals, "away_goals": g.away_goals,
                         "casa": p.get("casa"), "empate": p.get("empate"), "fora": p.get("fora"),
                         "mais_2_5": p.get("mais_2_5"), "ambas_marcam": p.get("ambas_marcam"),
                         "mercado_casa": m[0], "mercado_empate": m[1], "mercado_fora": m[2]})
    return pd.DataFrame(rows)


def summary(res: pd.DataFrame) -> dict:
    """Acerto do favorito e log loss do modelo e das casas nos jogos com as duas previsões."""
    import numpy as np

    b = res.dropna(subset=["casa", "mercado_casa"])
    y = np.select([b["home_goals"] > b["away_goals"], b["home_goals"] == b["away_goals"]], [0, 1], 2)
    out = {"jogos": len(b)}
    for who, cols in (("modelo", ["casa", "empate", "fora"]),
                      ("casas", ["mercado_casa", "mercado_empate", "mercado_fora"])):
        p = b[cols].to_numpy(dtype=float)
        out[f"acerto_{who}"] = float((p.argmax(1) == y).mean()) if len(b) else np.nan
        out[f"logloss_{who}"] = float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean()) if len(b) else np.nan
    p = b[["casa", "empate", "fora"]].to_numpy(dtype=float)
    strong = p.max(1) >= 0.65
    out["favorito_65"] = float(strong.mean()) if len(b) else np.nan
    out["acerto_favorito_65"] = float((p.argmax(1) == y)[strong].mean()) if strong.any() else np.nan
    goals = b["home_goals"] + b["away_goals"]
    out["acerto_mais_2_5"] = float(((b["mais_2_5"] > 0.5) == (goals > 2.5)).mean()) if len(b) else np.nan
    out["acerto_ambas"] = float(((b["ambas_marcam"] > 0.5) == ((b["home_goals"] > 0) & (b["away_goals"] > 0))).mean()) if len(b) else np.nan
    return out
