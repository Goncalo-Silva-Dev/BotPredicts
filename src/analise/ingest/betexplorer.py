"""Resultados e calendários do betexplorer.com, para ligas sem fonte oficial que se possa usar.

Serve o basquetebol (CBA, PBA) e o futebol (Liga Portugal 2, Liga Revelação Sub-23 e o
calendário das ligas que o openfootball não tem). O caminho é o do site, p. ex.
"basketball/china/cba" ou "football/portugal/liga-portugal-2".
"""
import html as html_lib
import re
from datetime import date, timedelta

import pandas as pd
import requests

HEADERS = {"User-Agent": "BotPredicts/0.1 (+https://github.com/Goncalo-Silva-Dev/BotPredicts)"}
COLUMNS = ["league", "date", "time", "round", "home", "away", "home_points", "away_points"]
BETEXPLORER = "https://www.betexplorer.com/{path}/{page}/"
_BE_RESULT = re.compile(
    r'class="in-match"><span>(?:<strong>)?([^<]+)(?:</strong>)?</span> - <span>(?:<strong>)?([^<]+)'
    r'(?:</strong>)?</span></a></td><td class="h-text-center"><a [^>]*>(\d+):(\d+)'
    r'.*?h-text-right h-text-no-wrap">([^<]*)<', re.S)
_BE_FIXTURE = re.compile(
    r'table-main__datetime">([^<]*)</td>\s*<td class="h-text-left"><a [^>]*class="in-match">'
    r'<span>([^<]+)</span> - <span>([^<]+)</span>', re.S)
# o betexplorer mostra as horas sempre em UTC+1, também no inverno
BETEXPLORER_TZ = "Etc/GMT-1"


RELATIVE_DAYS = {"Yesterday": -1, "Today": 0, "Tomorrow": 1}


def _be_day(text: str, today: date) -> date:
    """"Today", "Yesterday", "Tomorrow", "21.12." (ano corrente, que o site omite) ou "04.01.2027"."""
    if text in RELATIVE_DAYS:
        return today + timedelta(days=RELATIVE_DAYS[text])
    d, m, *y = [p for p in text.split(".") if p]
    return date(int(y[0]) if y else today.year, int(m), int(d))


def parse_betexplorer(page: str, today: date, league_key: str) -> pd.DataFrame:
    """Resultados de uma página do betexplorer (o prolongamento conta)."""
    rows = [{"league": league_key, "date": _be_day(when.strip(), today), "time": None, "round": None,
             "home": html_lib.unescape(home).strip(), "away": html_lib.unescape(away).strip(),
             "home_points": float(hp), "away_points": float(ap)}
            for home, away, hp, ap, when in _BE_RESULT.findall(page)]
    return pd.DataFrame(rows, columns=COLUMNS)


def parse_betexplorer_fixtures(page: str, today: date, league_key: str) -> pd.DataFrame:
    """Jogos marcados; uma célula de data vazia repete a da linha anterior."""
    rows, last = [], None
    for when, home, away in _BE_FIXTURE.findall(page):
        when = html_lib.unescape(when).strip() or last
        last = when
        if not when:
            continue
        day_text, _, clock = when.partition(" ")
        day = _be_day(day_text, today)
        stamp = pd.Timestamp(f"{day} {clock}", tz=BETEXPLORER_TZ).tz_convert("Europe/Lisbon")
        rows.append({"league": league_key, "date": stamp.date(), "time": stamp.strftime("%H:%M"),
                     "round": None, "home": html_lib.unescape(home).strip(),
                     "away": html_lib.unescape(away).strip(),
                     "home_points": None, "away_points": None})
    return pd.DataFrame(rows, columns=COLUMNS)


def _be_get(path: str, page: str, **params) -> str:
    resp = requests.get(BETEXPLORER.format(path=path, page=page), params=params, headers=HEADERS,
                        timeout=60)
    resp.raise_for_status()
    return resp.text


def fetch_betexplorer(path: str, league_key: str, today: date | None = None,
                      fixtures: bool = False) -> pd.DataFrame:
    """Todas as fases (fase regular, playoffs...) de um torneio do betexplorer."""
    today = today or date.today()
    first = _be_get(path, "results")
    pages = [first] + [_be_get(path, "results", stage=stage)
                       for stage in dict.fromkeys(re.findall(r'href="\?stage=([A-Za-z0-9]+)"', first))]
    frames = [parse_betexplorer(p, today, league_key) for p in pages]
    if fixtures:
        frames.append(parse_betexplorer_fixtures(_be_get(path, "fixtures"), today, league_key))
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates(subset=["date", "home", "away"])


def fetch_fixtures(path: str, league_key: str, today: date | None = None) -> pd.DataFrame:
    """Só os jogos marcados (hora de Portugal)."""
    return parse_betexplorer_fixtures(_be_get(path, "fixtures"), today or date.today(), league_key)
