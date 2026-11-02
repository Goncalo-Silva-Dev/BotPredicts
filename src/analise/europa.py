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
