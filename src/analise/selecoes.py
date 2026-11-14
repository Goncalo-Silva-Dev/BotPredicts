"""Liga das Nações (UEFA Nations League): jogos de seleções.

As seleções jogam pouco (8 a 12 jogos por ano), por isso a força de cada uma vem de todos os
jogos internacionais dos últimos anos e não só da Liga das Nações: qualificações e fases
finais do Europeu e do Mundial (todas as confederações) e amigáveis. Tudo do betexplorer,
com as odds 1X2 de fecho (média do mercado).

No histórico ficam com a liga "UNL" os jogos da Liga das Nações e com "SEL" os outros jogos
internacionais (só servem para treinar o modelo). A época ("season") diz de onde vem cada
jogo: "2024-25" (edição da Liga das Nações), "euro-2024", "mundial-2026", "amigaveis-2025".
"""
import os
import re
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd
import requests
from sqlalchemy import delete

from analise.db import matches
from analise.ingest import betexplorer

NATIONS = "UNL"
OTHER = "SEL"

# Fases finais em campo neutro: (primeiro dia, último dia, seleções da casa). Na fase final do
# Euro 2020 os anfitriões jogaram em casa os jogos da fase de grupos.
FINALS = {
    "euro-2016": ("2016-06-10", "2016-07-10", {"France"}),
    "euro-2020": ("2021-06-11", "2021-07-11", {"England", "Italy", "Germany", "Spain", "Netherlands",
                                                "Hungary", "Denmark", "Scotland", "Russia",
                                                "Azerbaijan", "Romania"}),
    "euro-2024": ("2024-06-14", "2024-07-14", {"Germany"}),
    "mundial-2018": ("2018-06-14", "2018-07-15", {"Russia"}),
    "mundial-2022": ("2022-11-20", "2022-12-18", {"Qatar"}),
    "mundial-2026": ("2026-06-11", "2026-07-19", {"USA", "Mexico", "Canada"}),
    # fase final da Liga das Nações (meias-finais, 3.º lugar e final)
    "2018-19": ("2019-06-05", "2019-06-09", {"Portugal"}),
    "2020-21": ("2021-10-06", "2021-10-10", {"Italy"}),
    "2022-23": ("2023-06-14", "2023-06-18", {"Netherlands"}),
    "2024-25": ("2025-06-04", "2025-06-08", {"Germany"}),
}

# Seleções jovens, femininas ou de outras modalidades que aparecem nos amigáveis
_NOT_SENIOR = re.compile(r"\b(?:U\d{2}|W|Olympic|B)$|\bWomen\b", re.I)


@dataclass(frozen=True)
class Source:
    path: str          # caminho no betexplorer
    season: str        # rótulo da época no histórico
    league: str        # NATIONS ou OTHER


def sources(today: date, first_year: int) -> list[Source]:
    """Todas as competições a recolher. As que ainda não existem no site são saltadas."""
    out = [Source("football/europe/uefa-nations-league", "atual", NATIONS)]
    for y in range(2018, today.year + 1, 2):
        if y + 1 >= first_year:
            out.append(Source(f"football/europe/uefa-nations-league-{y}-{y + 1}", f"{y}-{(y + 1) % 100:02d}",
                              NATIONS))
    for y in range(2016, today.year + 3, 4):
        if y >= first_year:
            out.append(Source(f"football/europe/euro-{y}", f"euro-{y}", OTHER))
    for y in range(2018, today.year + 5, 4):
        if y >= first_year:
            name = "world-cup" if y <= 2022 else "world-championship"
            out.append(Source(f"football/world/{name}-{y}", f"mundial-{y}", OTHER))
    out.append(Source("football/world/friendly-international", "amigaveis", OTHER))
    for y in range(first_year, today.year + 1):
        out.append(Source(f"football/world/friendly-international-{y}", f"amigaveis-{y}", OTHER))
    return out


_STAGE = re.compile(r'href="\?stage=([A-Za-z0-9]+)(?:&(?:amp;)?activecountry=([A-Za-z0-9]+))?"')


def fetch_source(src: Source, today: date) -> pd.DataFrame:
    """Todas as fases de uma competição (qualificação de cada confederação, fase final...).

    Os amigáveis mostram um mês de cada vez; "month=all" dá o ano inteiro.
    """
    url = betexplorer.BETEXPLORER.format(path=src.path, page="results")
    params = {"month": "all"} if "friendly" in src.path else {}
    try:
        first = betexplorer.get(url, **params)
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            return pd.DataFrame(columns=betexplorer.FOOTBALL_COLUMNS)
        raise
    pages = [first]
    for stage, country in dict.fromkeys(_STAGE.findall(first)):
        extra = {"stage": stage, **({"activecountry": country} if country else {}), **params}
        pages.append(betexplorer.get(url, **extra))
    df = pd.concat([betexplorer.parse_football_results(p, today, src.league) for p in pages],
                   ignore_index=True)
    return df.drop_duplicates(subset=["date", "home", "away"])


# Uma competição sem jogos há mais de 240 dias está acabada (a maior pausa é a do inverno entre
# a qualificação e o play-off): fica guardada em disco para não voltar a pedir ao site
DONE_DAYS = 240


def _cached(src: Source, today: date, cache: str | None) -> pd.DataFrame:
    file = os.path.join(cache, src.path.replace("/", "_") + ".csv") if cache else None
    if file and os.path.exists(file):
        old = pd.read_csv(file)
        old["date"] = pd.to_datetime(old["date"]).dt.date
        return old.assign(league=src.league)
    df = fetch_source(src, today)
    finished = not df.empty and max(df["date"]) < today - timedelta(days=DONE_DAYS)
    if file and finished and src.season not in ("atual", "amigaveis"):
        os.makedirs(cache, exist_ok=True)
        df.to_csv(file, index=False)
    return df


def _season(src: Source, day: date) -> str:
    """Rótulo da época de cada jogo; nas páginas sem ano (edição atual), tirado da data."""
    if src.season == "amigaveis":
        return f"amigaveis-{day.year}"
    if src.season == "atual":  # a Liga das Nações começa em setembro dos anos pares
        y = day.year if day.month >= 7 else day.year - 1
        y -= y % 2
        return f"{y}-{(y + 1) % 100:02d}"
    return src.season


def neutral(season: str, day: date, home: str) -> bool:
    final = FINALS.get(season)
    if not final:
        return False
    start, end, hosts = final
    return date.fromisoformat(start) <= day <= date.fromisoformat(end) and home not in hosts


def collect(today: date | None = None, first_year: int | None = None,
            cache: str | None = None) -> pd.DataFrame:
    """Todos os jogos internacionais no esquema da tabela matches (sem prolongamentos)."""
    today = today or date.today()
    first_year = first_year or today.year - 6
    cache = cache if cache is not None else os.environ.get("SELECOES_CACHE_DIR")
    frames = []
    for src in sources(today, first_year):
        df = _cached(src, today, cache)
        # jogos decididos no prolongamento/penáltis ou atribuídos: o resultado não é o dos 90'
        df = df.dropna(subset=["home_goals", "away_goals"])
        df = df[df["extra"].isna()] if "extra" in df else df
        df = df[~df["home"].str.contains(_NOT_SENIOR) & ~df["away"].str.contains(_NOT_SENIOR)]
        if df.empty:
            continue
        seasons = [_season(src, d) for d in df["date"]]
        frames.append(pd.DataFrame({
            "league": src.league, "season": seasons, "date": df["date"].to_numpy(),
            "home": df["home"].to_numpy(), "away": df["away"].to_numpy(),
            "home_goals": df["home_goals"].astype(int).to_numpy(),
            "away_goals": df["away_goals"].astype(int).to_numpy(),
            "odds_home_avg": df["odds_home"].to_numpy(), "odds_draw_avg": df["odds_draw"].to_numpy(),
            "odds_away_avg": df["odds_away"].to_numpy(),
            "neutral": [int(neutral(s, d, h)) for s, d, h in zip(seasons, df["date"], df["home"])],
            "source": "betexplorer.com",
        }))
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = out[pd.to_datetime(out["date"]).dt.year >= first_year]
    # nos amigáveis aparecem clubes e seleções que não são da FIFA (Catalunha, Martinica...):
    # ficam só os jogos entre seleções que jogaram alguma competição oficial
    friendly = out["season"].str.startswith("amigaveis")
    official = set(out.loc[~friendly, "home"]) | set(out.loc[~friendly, "away"])
    out = out[~friendly | (out["home"].isin(official) & out["away"].isin(official))]
    # o mesmo jogo pode aparecer em duas páginas (p. ex. edição atual e com ano): fica o da
    # Liga das Nações, que é o que interessa ao histórico de acertos
    out = out.sort_values("league", key=lambda s: s != NATIONS)
    return out.drop_duplicates(subset=["date", "home", "away"]).reset_index(drop=True)


def load_history(engine, today: date | None = None, first_year: int | None = None) -> int:
    """Substitui no histórico todos os jogos de seleções (UNL e SEL)."""
    df = collect(today, first_year)
    if df.empty:
        return 0
    records = df.astype(object).where(df.notna(), None).to_dict("records")
    with engine.begin() as conn:
        conn.execute(delete(matches).where(matches.c.league.in_([NATIONS, OTHER])))
        conn.execute(matches.insert(), records)
    return len(records)


# ---------------------------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------------------------

# Os amigáveis valem menos: as seleções fazem muitas trocas e poupam jogadores
FRIENDLY_WEIGHT = 0.5
HALF_LIFE = 730.0
WINDOW_DAYS = 4 * 365
MODEL_OPTIONS = {"ridge": 0.3, "joint_rho": False}
MIN_MATCHES = 4
# forças tiradas das odds de fecho (jogos já jogados); ver europa.OddsRatings
RATINGS = {"half_life_days": 730.0, "ridge": 0.1, "cup_weight": 1.0, "use_weight": True}
ODDS_WEIGHT = 0.6


def training_history(all_matches: pd.DataFrame) -> pd.DataFrame:
    """Jogos de seleções (UNL + SEL) com o peso de cada um."""
    df = all_matches[all_matches["league"].isin([NATIONS, OTHER])].copy()
    friendly = df["season"].astype(str).str.startswith("amigaveis")
    return df.assign(weight=friendly.map({True: FRIENDLY_WEIGHT, False: 1.0}).astype(float))


def prepare(model, train, fixtures, as_of, ratings=None, odds_weight=None) -> None:
    from analise import europa
    europa.align_with_odds(model, train, fixtures, as_of,
                           ratings=ratings or europa.OddsRatings(**RATINGS),
                           odds_weight=ODDS_WEIGHT if odds_weight is None else odds_weight)


def predict(history: pd.DataFrame, fixtures: pd.DataFrame, as_of, **kw) -> pd.DataFrame:
    from analise import previsoes
    opts = {"half_life_days": HALF_LIFE, "window_days": WINDOW_DAYS, "min_team_matches": MIN_MATCHES,
            "model_options": MODEL_OPTIONS, "prepare": prepare, **kw}
    return previsoes.predict(history, fixtures, as_of=as_of, **opts)


def walk_forward(all_matches: pd.DataFrame, start, leagues=(NATIONS,), **kw) -> pd.DataFrame:
    """Teste: cada semana de jogos prevista só com os jogos anteriores, com o mesmo código das
    previsões do site; junta as odds de fecho (sem margem) para comparar com as casas."""
    import numpy as np

    from analise.models.dixon_coles import fair_probs

    hist = training_history(all_matches)
    hist["date"] = pd.to_datetime(hist["date"])
    test = hist[hist["league"].isin(leagues) & (hist["date"] >= pd.Timestamp(start))]
    weeks = test["date"].dt.to_period("W-SUN").dt.start_time
    rows = []
    for week, games in test.groupby(weeks):
        fx = games.rename(columns={f"odds_{o}_avg": f"odds_{o}" for o in ("home", "draw", "away")})
        fx = fx.assign(time=None, date=fx["date"].dt.date)
        res = predict(hist, fx, week.date(), **kw)
        for g, p in zip(games.itertuples(index=False), res.to_dict("records")):
            odds = (g.odds_home_avg, g.odds_draw_avg, g.odds_away_avg)
            m = fair_probs(*odds) if all(pd.notna(o) and o > 1 for o in odds) else (np.nan,) * 3
            rows.append({"league": g.league, "season": g.season, "date": g.date.date(), "home": g.home,
                         "away": g.away, "home_goals": g.home_goals, "away_goals": g.away_goals,
                         "casa": p.get("casa"), "empate": p.get("empate"), "fora": p.get("fora"),
                         "mais_2_5": p.get("mais_2_5"), "ambas_marcam": p.get("ambas_marcam"),
                         "mercado_casa": m[0], "mercado_empate": m[1], "mercado_fora": m[2]})
    return pd.DataFrame(rows)
