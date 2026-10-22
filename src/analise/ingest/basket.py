"""Jogos de basquetebol (passados e futuros) das fontes grátis.

- NBA: API pública da ESPN, um pedido por dia de calendário.
- EuroLeague e EuroCup: API oficial da Euroleague Basketball, um pedido por época.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd
import requests

ESPN_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/{slug}/scoreboard"
EUROLEAGUE_URL = "https://api-live.euroleague.net/v2/competitions/{comp}/seasons/{comp}{year}/games"
COLUMNS = ["league", "date", "time", "round", "home", "away", "home_points", "away_points"]


@dataclass(frozen=True)
class BasketLeague:
    key: str
    name: str
    source: str  # "espn" ou "euroleague"
    code: str  # slug da ESPN ou código da competição na Euroleague
    # Meses sem jogos (a ESPN pede um pedido por dia, por isso saltamos o defeso)
    off_months: tuple = ()


BASKET_LEAGUES = {
    l.key: l
    for l in [
        BasketLeague("NBA", "NBA", "espn", "nba", off_months=(7, 8, 9)),
        BasketLeague("EUR", "EuroLeague", "euroleague", "E"),
        BasketLeague("EUC", "EuroCup", "euroleague", "U"),
    ]
}


def season_start(today: date) -> int:
    """Ano em que começou a época atual (as épocas começam no verão)."""
    return today.year if today.month >= 7 else today.year - 1


def _espn_day(league: BasketLeague, day: date) -> list[dict]:
    resp = requests.get(ESPN_URL.format(slug=league.code),
                        params={"dates": day.strftime("%Y%m%d")}, timeout=30)
    resp.raise_for_status()
    rows = []
    for ev in resp.json().get("events", []):
        comp = ev["competitions"][0]
        teams = {c["homeAway"]: c for c in comp["competitors"]}
        done = ev["status"]["type"].get("completed", False)
        when = pd.Timestamp(ev["date"]).tz_convert("Europe/Lisbon")
        rows.append({
            "league": league.key, "date": when.date(), "time": when.strftime("%H:%M"),
            "round": None,
            "home": teams["home"]["team"]["displayName"], "away": teams["away"]["team"]["displayName"],
            "home_points": float(teams["home"]["score"]) if done else None,
            "away_points": float(teams["away"]["score"]) if done else None,
        })
    return rows


def fetch_espn(league: BasketLeague, start: date, end: date) -> pd.DataFrame:
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    days = [d for d in days if d.month not in league.off_months]
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = [r for day_rows in pool.map(lambda d: _espn_day(league, d), days) for r in day_rows]
    return pd.DataFrame(rows, columns=COLUMNS)


def fetch_euroleague(league: BasketLeague, year: int) -> pd.DataFrame:
    resp = requests.get(EUROLEAGUE_URL.format(comp=league.code, year=year), timeout=30)
    resp.raise_for_status()
    rows = []
    for g in resp.json().get("data", []):
        when = pd.Timestamp(g["utcDate"]).tz_convert("Europe/Lisbon")
        played = g.get("played", False)
        rows.append({
            "league": league.key, "date": when.date(),
            "time": when.strftime("%H:%M") if g.get("confirmedHour") else None,
            "round": g.get("roundName"),
            "home": g["local"]["club"]["name"], "away": g["road"]["club"]["name"],
            "home_points": float(g["local"]["score"]) if played else None,
            "away_points": float(g["road"]["score"]) if played else None,
        })
    return pd.DataFrame(rows, columns=COLUMNS)


def fetch_games(league: BasketLeague, today: date | None = None, seasons: int = 3) -> pd.DataFrame:
    """Jogos das últimas `seasons` épocas até ao fim da atual (com os que faltam)."""
    today = today or date.today()
    first = season_start(today) - seasons + 1
    if league.source == "espn":
        df = fetch_espn(league, date(first, 7, 1), date(season_start(today) + 1, 6, 30))
    else:
        df = pd.concat([fetch_euroleague(league, y) for y in range(first, season_start(today) + 1)],
                       ignore_index=True)
    return df.drop_duplicates(subset=["date", "home", "away"]).sort_values("date").reset_index(drop=True)
