"""Resultados e calendários do betexplorer.com, para ligas sem fonte oficial que se possa usar.

Serve o basquetebol (CBA, PBA) e o futebol (Liga Portugal 2, Liga Revelação Sub-23, Arábia
Saudita, taças europeias e o calendário das ligas que o openfootball não tem). O caminho é o do site, p. ex.
"basketball/china/cba" ou "football/portugal/liga-portugal-2".
"""
import html as html_lib
import re
import time
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


_BE_ROW = re.compile(r"<tr>(.*?)</tr>", re.S)
_BE_TEAMS = re.compile(r'class="in-match"><span>(?:<strong>)?([^<]+)(?:</strong>)?</span> - '
                       r'<span>(?:<strong>)?([^<]+)(?:</strong>)?</span>')
_BE_SCORE = re.compile(r'<td class="h-text-center"><a [^>]*>(\d+):(\d+)\s*(?:<span[^>]*>([^<]*)</span>)?')
_BE_ODD = re.compile(r'data-odd="([\d.]+)"')
_BE_DAY = re.compile(r'h-text-right h-text-no-wrap">([^<]*)<')
FOOTBALL_COLUMNS = ["league", "date", "home", "away", "home_goals", "away_goals", "extra",
                    "odds_home", "odds_draw", "odds_away"]


def parse_football_results(page: str, today: date, league_key: str) -> pd.DataFrame:
    """Resultados de futebol com as odds 1X2 de fecho (média) quando o site as mostra.

    extra é "AET" (prolongamento) ou "PEN" (penáltis): nesses jogos o resultado mostrado
    não é o dos 90 minutos, por isso não deve contar para os mercados do tempo regulamentar.
    """
    rows = []
    for row in _BE_ROW.findall(page):
        teams, score, day = _BE_TEAMS.search(row), _BE_SCORE.search(row), _BE_DAY.search(row)
        if not (teams and score and day):
            continue
        odds = [float(o) for o in _BE_ODD.findall(row)]
        odds = odds if len(odds) == 3 else [None] * 3
        rows.append({"league": league_key, "date": _be_day(day.group(1).strip(), today),
                     "home": html_lib.unescape(teams.group(1)).strip(),
                     "away": html_lib.unescape(teams.group(2)).strip(),
                     "home_goals": int(score.group(1)), "away_goals": int(score.group(2)),
                     "extra": (score.group(3) or "").strip() or None,
                     "odds_home": odds[0], "odds_draw": odds[1], "odds_away": odds[2]})
    return pd.DataFrame(rows, columns=FOOTBALL_COLUMNS)


def fetch_football_results(path: str, league_key: str, today: date | None = None) -> pd.DataFrame:
    """Todas as fases de um torneio de futebol, com odds e indicação de prolongamento."""
    today = today or date.today()
    first = _be_get(path, "results")
    pages = [first] + [_be_get(path, "results", stage=stage)
                       for stage in dict.fromkeys(re.findall(r'href="\?stage=([A-Za-z0-9]+)"', first))]
    df = pd.concat([parse_football_results(p, today, league_key) for p in pages], ignore_index=True)
    return df.drop_duplicates(subset=["date", "home", "away"])


def parse_betexplorer_fixtures(page: str, today: date, league_key: str) -> pd.DataFrame:
    """Jogos marcados; uma célula de data vazia repete a da linha anterior.

    Quando o site já mostra as odds 1X2 (média do mercado), ficam em odds_home/draw/away.
    """
    rows, last = [], None
    for m in _BE_FIXTURE.finditer(page):
        when, home, away = m.groups()
        when = html_lib.unescape(when).strip() or last
        last = when
        if not when:
            continue
        end = page.find("</tr>", m.end())
        odds = [float(o) for o in _BE_ODD.findall(page[m.end():end if end >= 0 else None])]
        odds = odds if len(odds) == 3 else [None] * 3
        day_text, _, clock = when.partition(" ")
        day = _be_day(day_text, today)
        stamp = pd.Timestamp(f"{day} {clock}", tz=BETEXPLORER_TZ).tz_convert("Europe/Lisbon")
        rows.append({"league": league_key, "date": stamp.date(), "time": stamp.strftime("%H:%M"),
                     "round": None, "home": html_lib.unescape(home).strip(),
                     "away": html_lib.unescape(away).strip(),
                     "home_points": None, "away_points": None,
                     "odds_home": odds[0], "odds_draw": odds[1], "odds_away": odds[2]})
    return pd.DataFrame(rows, columns=COLUMNS + ["odds_home", "odds_draw", "odds_away"])


# O site responde "429 Too Many Requests" a rajadas de pedidos (o futebol e o basquetebol
# fazem dezenas seguidas na mesma corrida): espaçar os pedidos e, se recusar, esperar e repetir.
MIN_INTERVAL = 1.5
RETRY_WAITS = (10, 30, 60, 120)
_last_request = 0.0


def get(url: str, **params) -> str:
    global _last_request
    for wait in (*RETRY_WAITS, None):
        time.sleep(max(0.0, _last_request + MIN_INTERVAL - time.monotonic()))
        resp = requests.get(url, params=params, headers=HEADERS, timeout=60)
        _last_request = time.monotonic()
        if resp.status_code not in (429, 503) or wait is None:
            break
        retry_after = resp.headers.get("Retry-After", "")
        time.sleep(min(int(retry_after), 300) if retry_after.isdigit() else wait)
    resp.raise_for_status()
    return resp.text


def _be_get(path: str, page: str, **params) -> str:
    return get(BETEXPLORER.format(path=path, page=page), **params)


def fetch_betexplorer(path: str, league_key: str, today: date | None = None,
                      fixtures: bool = False) -> pd.DataFrame:
    """Todas as fases (fase regular, playoffs...) de um torneio do betexplorer."""
    today = today or date.today()
    first = _be_get(path, "results")
    pages = [first] + [_be_get(path, "results", stage=stage)
                       for stage in dict.fromkeys(re.findall(r'href="\?stage=([A-Za-z0-9]+)"', first))]
    frames = [parse_betexplorer(p, today, league_key) for p in pages]
    if fixtures:
        frames.append(parse_betexplorer_fixtures(_be_get(path, "fixtures"), today, league_key)[COLUMNS])
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates(subset=["date", "home", "away"])


def fetch_fixtures(path: str, league_key: str, today: date | None = None) -> pd.DataFrame:
    """Só os jogos marcados (hora de Portugal)."""
    return parse_betexplorer_fixtures(_be_get(path, "fixtures"), today or date.today(), league_key)
