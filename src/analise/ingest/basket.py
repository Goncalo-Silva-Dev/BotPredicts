"""Jogos de basquetebol (passados e futuros) das fontes grátis.

- NBA: API pública da ESPN, um pedido por dia de calendário.
- EuroLeague e EuroCup: API oficial da Euroleague Basketball, um pedido por época.
- NBB (Brasil): tabela de jogos do site da LNB, uma página por época.
- ACB (Espanha): calendário do site oficial, uma página por época.
- Lega Basket Serie A (Itália): API do site oficial, um pedido por jornada.
"""
import html as html_lib
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd
import requests

ESPN_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/{slug}/scoreboard"
EUROLEAGUE_URL = "https://api-live.euroleague.net/v2/competitions/{comp}/seasons/{comp}{year}/games"
# Identificação honesta do projeto (alguns sites recusam o agente por omissão do requests)
HEADERS = {"User-Agent": "BotPredicts/0.1 (+https://github.com/Goncalo-Silva-Dev/BotPredicts)"}
COLUMNS = ["league", "date", "time", "round", "home", "away", "home_points", "away_points"]


@dataclass(frozen=True)
class BasketLeague:
    key: str
    name: str
    source: str  # "espn", "euroleague", "nbb", "acb" ou "lba"
    code: str  # slug da ESPN ou código da competição na Euroleague
    # Meses sem jogos (a ESPN pede um pedido por dia, por isso saltamos o defeso)
    off_months: tuple = ()


BASKET_LEAGUES = {
    l.key: l
    for l in [
        BasketLeague("NBA", "NBA", "espn", "nba", off_months=(7, 8, 9)),
        BasketLeague("EUR", "EuroLeague", "euroleague", "E"),
        BasketLeague("EUC", "EuroCup", "euroleague", "U"),
        BasketLeague("ACB", "Liga Endesa (Espanha)", "acb", ""),
        BasketLeague("LBA", "Lega Basket Serie A (Itália)", "lba", ""),
        BasketLeague("NBB", "NBB (Brasil)", "nbb", ""),
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
    elif league.source == "acb":
        df = fetch_acb(league, range(first, season_start(today) + 1))
    elif league.source == "lba":
        df = fetch_lba(league, range(first, season_start(today) + 1))
    elif league.source == "nbb":
        df = fetch_nbb(league, range(first, season_start(today) + 1))
    else:
        df = pd.concat([fetch_euroleague(league, y) for y in range(first, season_start(today) + 1)],
                       ignore_index=True)
    return df.drop_duplicates(subset=["date", "home", "away"]).sort_values("date").reset_index(drop=True)


NBB_URL = "https://lnb.com.br/nbb/tabela-de-jogos/"


def _nbb_seasons(html: str) -> dict[int, str]:
    """Ano de início da época -> id da época no site da LNB."""
    found = re.findall(r'value="(\d+)"[^>]*>\s*<label[^>]*>NBB (\d{4})/\d{4}<', html)
    return {int(year): sid for sid, year in found}


def parse_nbb(html: str, league_key: str = "NBB") -> pd.DataFrame:
    rows = []
    for row in re.findall(r'<tr[^>]*>\s*<td[^>]*data-real-id="\d+".*?</tr>', html, re.S):
        when = re.search(r"(\d{2}/\d{2}/\d{4})</span>\s*<span[^>]*>(\d{2}:\d{2})?", row)
        teams = re.findall(r'class="team-shortname">([^<]+)<', row)
        if not when or len(teams) < 2:
            continue
        score = re.search(r'class="score_value.*?</td>', row, re.S).group(0)
        pts = re.findall(r'<span class="(home|away)">(\d+)</span>', score)
        rnd = re.search(r'class="number">(\d+)', row)
        small = re.search(r'matche_for_small">(.*?)</td>', row, re.S)
        codes = re.findall(r"<strong[^>]*>(.*?)</strong>", small.group(1), re.S) if small else []
        codes = [re.sub(r"<[^>]+>", "", c).strip() for c in codes]
        played = len(pts) == 2
        rows.append({
            "league": league_key,
            "date": pd.to_datetime(when.group(1), dayfirst=True).date(),
            "time": when.group(2),
            "round": f"Round {rnd.group(1)}" if rnd else None,
            "home": html_lib.unescape(teams[0].strip()), "away": html_lib.unescape(teams[1].strip()),
            "home_points": float(pts[0][1]) if played else None,
            "away_points": float(pts[1][1]) if played else None,
            "home_code": codes[0] if len(codes) == 2 else None,
            "away_code": codes[1] if len(codes) == 2 else None,
        })
    return pd.DataFrame(rows, columns=COLUMNS + ["home_code", "away_code"])


def _unify_names(df: pd.DataFrame) -> pd.DataFrame:
    """Os patrocínios mudam o nome das equipas de época para época; a sigla (MOG, COR...)
    mantém-se, por isso usamos o nome mais recente de cada sigla."""
    long = pd.concat([
        df[["date", "home_code", "home"]].set_axis(["date", "code", "name"], axis=1),
        df[["date", "away_code", "away"]].set_axis(["date", "code", "name"], axis=1)]).dropna()
    latest = long.sort_values("date").groupby("code")["name"].last()
    df = df.copy()
    df["home"] = df["home_code"].map(latest).fillna(df["home"])
    df["away"] = df["away_code"].map(latest).fillna(df["away"])
    return df[COLUMNS]


def fetch_nbb(league: BasketLeague, years: range) -> pd.DataFrame:
    first = requests.get(NBB_URL, headers=HEADERS, timeout=60)
    first.raise_for_status()
    ids = _nbb_seasons(first.text)
    frames = []
    for year in years:
        if year not in ids:
            continue
        resp = requests.get(NBB_URL, params={"season[]": ids[year]}, headers=HEADERS,
                            timeout=60)
        resp.raise_for_status()
        frames.append(parse_nbb(resp.text, league.key))
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    return _unify_names(pd.concat(frames, ignore_index=True))


ACB_URL = "https://acb.com/es/liga/calendario"
_ACB_MATCH = re.compile(
    r'\{\\"id\\":(\d+),\\"homeTeam\\":[^{}]*?\\"homeTeamScore\\":(null|\d+),'
    r'\\"awayTeamScore\\":(null|\d+),\\"startDateTime\\":\\"([^\\]+)\\",\\"matchStatus\\":\\"(\w+)')


def _acb_seasons(html: str) -> dict[int, str]:
    found = re.findall(r'\\"id\\":(\d+),\\"startYear\\":(\d{4}),\\"endYear\\"', html)
    return {int(year): sid for sid, year in found}


def parse_acb(html: str, league_key: str = "ACB") -> pd.DataFrame:
    """O calendário da ACB vem numa página Next.js: as equipas e a jornada estão no HTML e a
    data e o resultado nos dados embutidos; juntamos os dois pelo id do jogo."""
    data = {int(m[0]): m[1:] for m in _ACB_MATCH.findall(html)}
    rows = []
    for section in html.split('id="calendar-round-')[1:]:
        rnd = re.search(r'roundTitle">([^<]+)<', section)
        for block in section.split("__roundMatch__teams")[1:]:
            names = re.findall(r'teamName--fullName">([^<]+)<', block)
            # O id do clube no link da equipa não muda com os patrocínios
            codes = list(dict.fromkeys(re.findall(r"/equipos/[a-z0-9-]+-(\d+)\?", block)))
            mid = re.search(r"partidos/[a-z0-9-]+-(\d+)/", block)
            if len(names) < 2 or not mid or int(mid.group(1)) not in data:
                continue
            hs, as_, start, status = data[int(mid.group(1))]
            when = pd.Timestamp(start).tz_convert("Europe/Lisbon")
            done = status == "FINALIZED" and hs != "null"
            rows.append({
                "league": league_key, "date": when.date(),
                "time": when.strftime("%H:%M") if status != "UNSCHEDULED" else None,
                "round": re.sub(r"^Jornada\s*", "Round ", rnd.group(1).strip()) if rnd else None,
                "home": html_lib.unescape(names[0]), "away": html_lib.unescape(names[1]),
                "home_points": float(hs) if done else None,
                "away_points": float(as_) if done else None,
                "home_code": codes[0] if len(codes) > 1 else None,
                "away_code": codes[1] if len(codes) > 1 else None,
            })
    return pd.DataFrame(rows, columns=COLUMNS + ["home_code", "away_code"])


def fetch_acb(league: BasketLeague, years: range) -> pd.DataFrame:
    first = requests.get(ACB_URL, headers=HEADERS, timeout=60)
    first.raise_for_status()
    ids = _acb_seasons(first.text)
    frames = []
    for year in years:
        if year not in ids:
            continue
        resp = requests.get(ACB_URL, params={"temporada": ids[year]}, headers=HEADERS, timeout=60)
        resp.raise_for_status()
        frames.append(parse_acb(resp.text, league.key))
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    return _unify_names(pd.concat(frames, ignore_index=True))


LBA_API = "https://www.legabasket.it/api/championships"


def _lba_get(path: str, **params) -> dict:
    resp = requests.get(f"{LBA_API}/{path}", params=params, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    return resp.json()


def parse_lba(matches: list[dict], league_key: str = "LBA") -> pd.DataFrame:
    rows = []
    for m in matches:
        when = pd.Timestamp(m["match_datetime"]).tz_convert("Europe/Lisbon")
        done = str(m.get("game_status")) == "2"
        rows.append({
            "league": league_key, "date": when.date(), "time": when.strftime("%H:%M"),
            "round": f"Round {m['day_serial']}",
            "home": m["h_team_name"], "away": m["v_team_name"],
            "home_points": float(m["home_final_score"]) if done else None,
            "away_points": float(m["visitor_final_score"]) if done else None,
            "home_code": m.get("h_club_code"), "away_code": m.get("v_club_code"),
        })
    return pd.DataFrame(rows, columns=COLUMNS + ["home_code", "away_code"])


def fetch_lba(league: BasketLeague, years: range) -> pd.DataFrame:
    """Lega Basket Serie A: só a fase regular, uma chamada por jornada."""
    comps = _lba_get("get-championships", cs_id=1, items=1000)["competitions"]
    ids = {c["year"]: c["id"] for c in comps if c["ctype_code"] == "RS"}
    matches = []
    for year in years:
        if year not in ids:
            continue
        days = _lba_get("get-championships-calendar-by-id", id=ids[year])["filters"]["days"]
        with ThreadPoolExecutor(max_workers=6) as pool:
            for day in pool.map(lambda d: _lba_get("get-championships-calendar-by-id",
                                                   id=ids[year], d=d["event_serial"]), days):
                matches.extend(day.get("matches") or [])
    if not matches:
        return pd.DataFrame(columns=COLUMNS)
    return _unify_names(parse_lba(matches, league.key))
