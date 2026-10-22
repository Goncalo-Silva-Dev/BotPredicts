"""Jogos que faltam: calendário completo do openfootball + odds do football-data.co.uk."""
import logging
from datetime import date, timedelta

import pandas as pd
from sqlalchemy import select

from analise.db import matches
from analise.ingest import football_data_uk, openfootball
from analise.leagues import League

log = logging.getLogger(__name__)
ODDS = ["odds_home", "odds_draw", "odds_away"]


def _recent_teams(engine, league: League, today: date) -> set[str]:
    since = today - timedelta(days=730)
    with engine.connect() as conn:
        rows = conn.execute(select(matches.c.home, matches.c.away).where(
            matches.c.league == league.key, matches.c.date >= since))
        return {team for r in rows for team in r}


def upcoming(engine, leagues: list[League], today: date | None = None) -> pd.DataFrame:
    """Todos os jogos a partir de hoje, com a jornada e as odds quando já existem.

    Ligas sem calendário no openfootball (MLS) ficam só com os jogos que o
    football-data.co.uk já publicou (normalmente a semana seguinte).
    """
    today = today or date.today()
    fd = football_data_uk.fetch_fixtures(leagues)
    fd = fd[fd["date"] >= today]
    frames = []
    for league in leagues:
        if not league.of_code:
            continue
        try:
            season = openfootball.fetch_season(league, today)
        except Exception as e:  # sem calendário: fica só com o football-data
            log.warning("%s: sem calendário do openfootball (%s)", league.key, e)
            continue
        season = season[~season["played"] & (season["date"] >= today)]
        names = openfootball.match_teams(
            sorted(set(season["home"]) | set(season["away"])), _recent_teams(engine, league, today))
        season = season.assign(home=season["home"].map(lambda n: names.get(n, n)),
                               away=season["away"].map(lambda n: names.get(n, n)))
        frames.append(season.drop(columns="played"))

    full = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["league", "round", "date", "time", "home", "away"])
    # Odds (e hora mais certa) do football-data para os jogos que já as têm
    key = ["league", "home", "away"]
    full = full.merge(fd[key + ["time"] + ODDS].rename(columns={"time": "fd_time"}), on=key, how="left")
    full["time"] = full["fd_time"].fillna(full["time"])
    full = full.drop(columns="fd_time")
    extra = fd.merge(full[key], on=key, how="left", indicator=True)
    extra = extra[extra["_merge"] == "left_only"].drop(columns="_merge").assign(round=None)
    out = pd.concat([full, extra[full.columns]], ignore_index=True)
    return out.sort_values(["date", "time", "league"], na_position="last").reset_index(drop=True)
