"""Jogos de hóquei no gelo da ESPN (a mesma API pública da NBA), com golos por período.

Cada jogo traz o resultado final (com prolongamento e penáltis: quem ganha nos penáltis
fica com mais um golo, como nas casas de apostas) e os golos de cada período, de onde sai
o resultado nos 60 minutos.
"""
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd
import requests

ESPN_URL = "https://site.api.espn.com/apis/site/v2/sports/hockey/{slug}/scoreboard"
PERIODS = [f"{side}_p{p}" for p in range(1, 4) for side in ("home", "away")]
COLUMNS = ["league", "date", "time", "round", "home", "away", "home_goals", "away_goals",
           "home_reg", "away_reg", "ending"] + PERIODS
# Tipos de época da ESPN: 1 pré-época (fica de fora: equipas cheias de reservas),
# 2 fase regular, 3 playoffs
SEASON_TYPES = {2, 3}


@dataclass(frozen=True)
class HockeyLeague:
    key: str
    name: str
    slug: str  # slug da ESPN
    off_months: tuple = ()  # meses sem jogos (um pedido por dia, por isso salta-se o defeso)


HOCKEY_LEAGUES = {l.key: l for l in [
    HockeyLeague("NHL", "NHL (EUA e Canadá)", "nhl", off_months=(7, 8)),
]}


def season_start(today: date) -> int:
    """Ano em que começou a época atual (a NHL começa em outubro e acaba em junho)."""
    return today.year if today.month >= 7 else today.year - 1


def parse_espn_day(events: list[dict], league_key: str) -> list[dict]:
    rows = []
    for ev in events:
        if (ev.get("season") or {}).get("type") not in SEASON_TYPES:
            continue
        comp = ev["competitions"][0]
        teams = {c["homeAway"]: c for c in comp["competitors"]}
        if set(teams) != {"home", "away"}:
            continue
        status = ev["status"]
        done = status["type"].get("completed", False)
        when = pd.Timestamp(ev["date"]).tz_convert("Europe/Lisbon")
        row = {
            "league": league_key, "date": when.date(), "time": when.strftime("%H:%M"),
            "round": "Playoffs" if ev["season"]["type"] == 3 else None,
            "home": teams["home"]["team"]["displayName"], "away": teams["away"]["team"]["displayName"],
            "home_goals": None, "away_goals": None, "home_reg": None, "away_reg": None, "ending": None,
        }
        if done:
            lines = {side: [float(ls["value"]) for ls in teams[side].get("linescores") or []]
                     for side in ("home", "away")}
            if min(len(lines["home"]), len(lines["away"])) < 3:
                continue  # sem golos por período não dá para saber os 60 minutos
            periods = status.get("period") or len(lines["home"])
            row.update({
                "home_goals": float(teams["home"]["score"]), "away_goals": float(teams["away"]["score"]),
                "home_reg": sum(lines["home"][:3]), "away_reg": sum(lines["away"][:3]),
                # REG nos 60 minutos, OT no prolongamento, SO nos penáltis
                "ending": "REG" if periods <= 3 else ("SO" if "SO" in status["type"].get("detail", "")
                                                      else "OT"),
            })
            for side in ("home", "away"):
                for p in range(3):
                    row[f"{side}_p{p + 1}"] = lines[side][p]
        rows.append(row)
    return rows


def _espn_day(league: HockeyLeague, day: date, tries: int = 3) -> list[dict]:
    for attempt in range(tries):  # com ~900 pedidos, uma ligação cortada de vez em quando é normal
        try:
            resp = requests.get(ESPN_URL.format(slug=league.slug),
                                params={"dates": day.strftime("%Y%m%d")}, timeout=30)
            resp.raise_for_status()
            return parse_espn_day(resp.json().get("events", []), league.key)
        except requests.RequestException:
            if attempt == tries - 1:
                raise
            time.sleep(2 ** attempt)
    return []


def fetch_games(league: HockeyLeague, today: date | None = None, seasons: int = 3) -> pd.DataFrame:
    """Jogos das últimas `seasons` épocas até ao fim da atual (com os que faltam)."""
    today = today or date.today()
    start = date(season_start(today) - seasons + 1, 9, 1)
    end = date(season_start(today) + 1, 6, 30)
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    days = [d for d in days if d.month not in league.off_months]
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = [r for day in pool.map(lambda d: _espn_day(league, d), days) for r in day]
    df = pd.DataFrame(rows, columns=COLUMNS)
    df = df.drop_duplicates(subset=["date", "home", "away"])
    return df.sort_values(["date", "time"]).reset_index(drop=True)
