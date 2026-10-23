"""Jogos de basquetebol (passados e futuros) das fontes grátis.

- NBA e NBL (Austrália): API pública da ESPN, um pedido por dia de calendário.
- EuroLeague e EuroCup: API oficial da Euroleague Basketball, um pedido por época.
- NBB (Brasil): tabela de jogos do site da LNB, uma página por época.
- ACB (Espanha): calendário do site oficial, uma página por época.
- Lega Basket Serie A (Itália): API do site oficial, um pedido por jornada.
- Liga Betclic Masculina (Portugal): resultados e calendário do site da FPB.
- KBL (Coreia do Sul): API do site oficial, um pedido por época.
- Liga ABA (Sérvia e Balcãs): calendário do site oficial, uma página por época.
- B.League B1 (Japão): calendário do site oficial, um pedido por dia de jogos.
- Betclic ÉLITE (França): API do site da LNB, com o token público que o site dá a quem o visita.
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
# Pontos por quarto, só nas fontes que os dão (ESPN e Euroleague); servem para calibrar os períodos
QUARTERS = [f"{side}_q{q}" for q in range(1, 5) for side in ("home", "away")]


@dataclass(frozen=True)
class BasketLeague:
    key: str
    name: str
    source: str  # "espn", "euroleague", "nbb", "acb", "lba", "fpb" ou "lnb"
    code: str  # slug da ESPN ou código da competição na Euroleague
    # Meses sem jogos (a ESPN pede um pedido por dia, por isso saltamos o defeso)
    off_months: tuple = ()


BASKET_LEAGUES = {
    l.key: l
    for l in [
        BasketLeague("NBA", "NBA", "espn", "nba", off_months=(7, 8, 9)),
        BasketLeague("NBL", "NBL (Austrália)", "espn", "nbl", off_months=(5, 6, 7, 8)),
        BasketLeague("EUR", "EuroLeague", "euroleague", "E"),
        BasketLeague("EUC", "EuroCup", "euroleague", "U"),
        BasketLeague("ACB", "Liga Endesa (Espanha)", "acb", ""),
        BasketLeague("LBA", "Lega Basket Serie A (Itália)", "lba", ""),
        BasketLeague("LPB", "Liga Betclic (Portugal)", "fpb", ""),
        BasketLeague("NBB", "NBB (Brasil)", "nbb", ""),
        BasketLeague("LNB", "Betclic ÉLITE (França)", "lnb", "PROA"),
        BasketLeague("BLG", "B.League (Japão)", "bleague", "2"),
        BasketLeague("ABA", "Liga ABA (Sérvia e Balcãs)", "aba", ""),
        BasketLeague("KBL", "KBL (Coreia do Sul)", "kbl", ""),
        BasketLeague("SBL", "Basketligan (Suécia)", "sbl", ""),
        BasketLeague("KOR", "Korisliiga (Finlândia)", "kori", ""),
        BasketLeague("CBA", "CBA (China)", "cba", ""),
        BasketLeague("PBA", "PBA (Filipinas)", "pba", ""),
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
        if done:
            for side in ("home", "away"):
                for ls in teams[side].get("linescores") or []:
                    if 1 <= ls.get("period", 0) <= 4:
                        rows[-1][f"{side}_q{ls['period']}"] = float(ls["value"])
    return rows


def fetch_espn(league: BasketLeague, start: date, end: date) -> pd.DataFrame:
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    days = [d for d in days if d.month not in league.off_months]
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = [r for day_rows in pool.map(lambda d: _espn_day(league, d), days) for r in day_rows]
    return pd.DataFrame(rows, columns=COLUMNS + QUARTERS)


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
        if played:
            for side, team in (("home", g["local"]), ("away", g["road"])):
                partials = team.get("partials") or {}
                for q in range(1, 5):
                    if partials.get(f"partials{q}") is not None:
                        rows[-1][f"{side}_q{q}"] = float(partials[f"partials{q}"])
    return pd.DataFrame(rows, columns=COLUMNS + QUARTERS)


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
    elif league.source == "fpb":
        df = fetch_fpb(league, range(first, season_start(today) + 1))
    elif league.source == "nbb":
        df = fetch_nbb(league, range(first, season_start(today) + 1))
    elif league.source == "pba":
        df = fetch_pba(league, range(first, season_start(today) + 1), today)
    elif league.source == "cba":
        df = fetch_cba(league, range(first, season_start(today) + 1), today)
    elif league.source == "kori":
        df = fetch_kori(league, range(first, season_start(today) + 1))
    elif league.source == "sbl":
        df = fetch_sbl(league, range(first, season_start(today) + 1))
    elif league.source == "kbl":
        df = fetch_kbl(league, range(first, season_start(today) + 1))
    elif league.source == "aba":
        df = fetch_aba(league, range(first, season_start(today) + 1))
    elif league.source == "bleague":
        df = fetch_bleague(league, range(first, season_start(today) + 1))
    elif league.source == "lnb":
        df = fetch_lnb(league, range(first, season_start(today) + 1))
    else:
        df = pd.concat([fetch_euroleague(league, y) for y in range(first, season_start(today) + 1)],
                       ignore_index=True)
    return df.drop_duplicates(subset=["date", "home", "away"]).sort_values("date").reset_index(drop=True)


NBB_URL = "https://lnb.com.br/nbb/tabela-de-jogos/"


def _nbb_seasons(html: str) -> dict[int, str]:
    """Ano de início da época -> id da época no site da LNB."""
    found = re.findall(r'value="(\d+)"[^>]*>\s*<label[^>]*>NBB (\d{4})/\d{4}<', html)
    return {int(year): sid for sid, year in found}


def _brasilia_to_lisbon(day: str, clock: str | None):
    d = pd.to_datetime(day, dayfirst=True)
    if not clock:
        return d.date(), None
    t = pd.Timestamp(f"{d.date()} {clock}", tz="America/Sao_Paulo").tz_convert("Europe/Lisbon")
    return t.date(), t.strftime("%H:%M")


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
            # horas de Brasília, passadas para a hora de Portugal
            "date": _brasilia_to_lisbon(when.group(1), when.group(2))[0],
            "time": _brasilia_to_lisbon(when.group(1), when.group(2))[1],
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


FPB_AJAX = "https://www.fpb.pt/wp-admin/admin-ajax.php"
# Liga Betclic Masculina por época; a pesquisa do site não devolve todas, por isso as
# conhecidas ficam aqui e as novas são procuradas pelo nome
FPB_LPB_IDS = {2024: "10390", 2025: "10902", 2026: "11440"}
# O site da FPB usa o nome completo nalgumas épocas e o curto noutras
FPB_ALIASES = {
    "Futebol Clube do Porto": "FC Porto", "Sporting Clube Portugal": "Sporting CP",
    "Sporting Clube de Braga": "SC Braga", "Vitória Sport Clube": "Vitória SC",
    "Galitos BARREIRO ACEDE": "Galitos BARREIRO", "Queluz O NOSSO PREGO": "Queluz | O Nosso Prego",
    "CD Póvoa ESC ONLINE": "CD Póvoa / Esc Online",
}
PT_MONTHS = {"JAN": 1, "FEV": 2, "MAR": 3, "ABR": 4, "MAI": 5, "JUN": 6, "JUL": 7, "AGO": 8,
             "SET": 9, "OUT": 10, "NOV": 11, "DEZ": 12}


def _fpb_lpb_id(year: int) -> str | None:
    if year in FPB_LPB_IDS:
        return FPB_LPB_IDS[year]
    resp = requests.post(FPB_AJAX, headers=HEADERS, timeout=60, data={
        "action": "get_competicoes", "epoca": f"{year}/{year + 1}", "escalao": "Sénior",
        "genero": "masculino", "radio": "true", "associacao": "50"})
    resp.raise_for_status()
    found = re.findall(r'data-id="(\d+)" name="comp-option" value="Liga Betclic Masculina"',
                       resp.text)
    return found[0] if found else None


def parse_fpb(html: str, league_key: str = "LPB") -> pd.DataFrame:
    rows = []
    for day in html.split('class="day-wrapper')[1:]:
        d = re.search(r'class="date">\s*(\d{1,2}) ([A-Z]{3}) (\d{4})', day)
        if not d:
            continue
        when = date(int(d.group(3)), PT_MONTHS[d.group(2)], int(d.group(1)))
        for game in day.split('class="game-wrapper ')[1:]:
            names = [n.strip() for n in re.findall(r'class="fullName">([^<]+)<', game)]
            pts = [p.strip() for p in re.findall(r'class="results_text[^"]*">([^<]*)<', game)]
            hour = re.search(r"<h3>\s*(\d{1,2})H(\d{2})", game)
            if len(names) < 2:
                continue
            played = len(pts) == 2 and all(p.isdigit() for p in pts)
            rows.append({
                "league": league_key, "date": when,
                "time": f"{int(hour.group(1)):02d}:{hour.group(2)}" if hour else None,
                "round": None,
                "home": FPB_ALIASES.get(html_lib.unescape(names[0]), html_lib.unescape(names[0])),
                "away": FPB_ALIASES.get(html_lib.unescape(names[1]), html_lib.unescape(names[1])),
                "home_points": float(pts[0]) if played else None,
                "away_points": float(pts[1]) if played else None,
            })
    return pd.DataFrame(rows, columns=COLUMNS)


def fetch_fpb(league: BasketLeague, years: range) -> pd.DataFrame:
    """Liga Betclic Masculina (LPB): resultados e calendário do site da FPB."""
    frames = []
    for year in years:
        cid = _fpb_lpb_id(year)
        if not cid:
            continue
        period = {"period[time_option]": "fromInit", "period[from_date]": f"01-08-{year}",
                  "period[to_date]": f"31-07-{year + 1}"}
        for action in ("get_results", "get_more_days"):
            resp = requests.get(FPB_AJAX, headers=HEADERS, timeout=60,
                                params={"action": action, "competicao": cid, **period})
            resp.raise_for_status()
            result = resp.json()["result"]
            frames.append(parse_fpb("".join(result) if isinstance(result, list) else result,
                                    league.key))
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.concat(frames, ignore_index=True)
    # O mesmo jogo pode vir nas duas listas; fica a versão com resultado
    df = df.sort_values("home_points", na_position="last")
    return df.drop_duplicates(subset=["date", "home", "away"]).reset_index(drop=True)


LNB_TOKEN_URL = "https://lnb.fr/api/token"
LNB_CALENDAR_URL = "https://api-prod.lnb.fr/match/v3/getCalendar"


def parse_lnb(days: list[dict], league_key: str = "LNB") -> pd.DataFrame:
    rows = []
    for day in days:
        for m in day["data"]:
            if len(m.get("teams") or []) != 2:
                continue
            home, away = m["teams"]  # a primeira equipa é a da casa ("Dijon - Le Mans")
            when = pd.Timestamp(m["match_time_utc"]).tz_convert("Europe/Lisbon")
            done = m.get("match_status") == "COMPLETE"
            rows.append({
                "league": league_key, "date": when.date(), "time": when.strftime("%H:%M"),
                "round": f"Round {m['round_number']}" if m.get("round_number") else None,
                "home": home["team_name"], "away": away["team_name"],
                "home_points": float(home["score_string"]) if done else None,
                "away_points": float(away["score_string"]) if done else None,
                "home_code": home.get("club_external_id"), "away_code": away.get("club_external_id"),
            })
    return pd.DataFrame(rows, columns=COLUMNS + ["home_code", "away_code"])


def fetch_lnb(league: BasketLeague, years: range) -> pd.DataFrame:
    """Betclic ÉLITE: o calendário de cada época num só pedido."""
    resp = requests.get(LNB_TOKEN_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    token = resp.json()["token"]
    headers = {**HEADERS, "Authorization": f"Bearer {token}", "device_type": "web",
               "language_code": "fr"}
    days = []
    for year in years:
        body = {"year": year, "competition_abbrev": league.code, "division_external_id": 0,
                "team_external_id": 0, "round_number": 0, "phase_id": 0,
                "tournament_number": 0, "direction": "initial", "limit": 1000}
        resp = requests.post(LNB_CALENDAR_URL, json=body, headers=headers, timeout=60)
        resp.raise_for_status()
        days.extend(resp.json().get("data") or [])
    if not days:
        return pd.DataFrame(columns=COLUMNS)
    return _unify_names(parse_lnb(days, league.key))


BLEAGUE_URL = "https://www.bleague.jp/schedule/"
_BLEAGUE_GAME = re.compile(r'<li class="list-item" id="(\d+)">(.*?)class="data-link"', re.S)


def parse_bleague(html: str, day: date, league_key: str = "BLG") -> pd.DataFrame:
    rows = []
    for _, block in _BLEAGUE_GAME.findall(html):
        names = re.findall(r'team-name">([^<]*)<', block)
        codes = re.findall(r'/logo/\d+/s/(\w+)\.png', block)
        if len(names) != 2 or any(BLEAGUE_SKIP.match(c) for c in codes):
            continue
        scores = re.findall(r'(home|away)-score[^"]*"><span>(\d+)</span>', block)
        pts = {side: float(v) for side, v in scores}
        clock = re.search(r"<span>(\d\d:\d\d)</span>", block)
        when = None
        if clock:  # horas do Japão
            when = pd.Timestamp(f"{day} {clock.group(1)}", tz="Asia/Tokyo").tz_convert("Europe/Lisbon")
        rows.append({
            "league": league_key, "date": when.date() if when is not None else day,
            "time": when.strftime("%H:%M") if when is not None else None, "round": None,
            "home": BLEAGUE_NAMES.get(codes[0], names[0]) if len(codes) == 2 else names[0],
            "away": BLEAGUE_NAMES.get(codes[1], names[1]) if len(codes) == 2 else names[1],
            "home_points": pts.get("home"), "away_points": pts.get("away"),
        })
    return pd.DataFrame(rows, columns=COLUMNS)


def _bleague_get(**params) -> str:
    resp = requests.get(BLEAGUE_URL, params={"tab": 1, "club": "", **params}, headers=HEADERS,
                        timeout=60)
    resp.raise_for_status()
    return resp.text


def fetch_bleague(league: BasketLeague, years: range) -> pd.DataFrame:
    """B1: a página de cada mês diz em que dias há jogos; depois um pedido por dia."""
    days = []
    for year in years:
        for month in (9, 10, 11, 12, 1, 2, 3, 4, 5):
            html = _bleague_get(year=year, mon=month, event=league.code)
            cal_year = year if month >= 7 else year + 1
            for d in sorted(set(re.findall(r'data-day="(\d+)"', html))):
                days.append((year, date(cal_year, month, int(d))))
    with ThreadPoolExecutor(max_workers=6) as pool:
        frames = list(pool.map(lambda yd: parse_bleague(
            _bleague_get(year=yd[0], mon=yd[1].month, day=yd[1].day, event=league.code),
            yd[1], league.key), days))
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)


# Nomes das equipas pela sigla do logótipo (o site só os dá em japonês); uma sigla nova
# aparece com o nome japonês até ser acrescentada aqui
BLEAGUE_NAMES = {
    "ac": "Altiri Chiba", "an": "Akita Northern Happinets", "at": "Alvark Tokyo",
    "bw": "Shinshu Brave Warriors", "cj": "Chiba Jets", "dd": "Nagoya Diamond Dolphins",
    "fe": "FE Nagoya", "gc": "Gunma Crane Thunders", "hd": "Hiroshima Dragonflies",
    "ir": "Ibaraki Robots", "ka": "Koshigaya Alphas", "kb": "Kawasaki Brave Thunders",
    "kh": "Kyoto Hannaryz", "lh": "Levanga Hokkaido", "ls": "Shiga Lakes",
    "ns": "Kobe Storks", "nv": "Nagasaki Velca", "oe": "Osaka Evessa", "rg": "Ryukyu Golden Kings",
    "se": "Sendai 89ers", "sg": "Saga Ballooners", "sm": "SeaHorses Mikawa",
    "sn": "San-en NeoPhoenix", "sr": "Sunrockers Shibuya", "ss": "Shimane Susanoo Magic",
    "tg": "Toyama Grouses", "ub": "Utsunomiya Brex", "yb": "Yokohama B-Corsairs",
}
# Jogos das estrelas e seleções que aparecem no calendário da B1 (siglas a1, a2, a5...)
BLEAGUE_SKIP = re.compile(r"^a\d+$")


ABA_URL = "https://www.aba-liga.com/calendar/{season}/1/"
_ABA_ROW = re.compile(
    r'class="visible-xs">\s*<a[^>]*>\s*([^<]*?)\s*<span>:</span>\s*([^<]*?)</a>.*?'
    r'class="hidden-xs">\s*<a[^>]*>\s*([^<]*?)\s*<span>:</span>\s*([^<]*?)</a>.*?'
    r'class="scoretable">(.*?)</td>\s*<td class="locationtable">\s*([^<]*?)\s*</td>', re.S)


def parse_aba(html: str, league_key: str = "ABA") -> pd.DataFrame:
    rows = []
    # o calendário vem em blocos ("ROUND 5", "Top8 - R1"...); só a fase regular interessa,
    # porque nas outras fases as equipas ainda não estão definidas
    parts = re.split(r'aria-controls="#collapse_\d+"[^>]*>\s*([^<]*?)\s*<', html)
    for label, chunk in zip(parts[1::2], parts[2::2]):
        rnd = re.fullmatch(r"ROUND (\d+)", label.strip())
        if not rnd:
            continue
        for hc, ac, home, away, score, when in _ABA_ROW.findall(chunk):
            # "Friday, 02.10.2026 18:30 CET"; mais à frente só há o dia (ou um intervalo de dias)
            m = re.search(r"(\d\d)\.(\d\d)\.(\d{4})(?: (\d\d:\d\d))?", when)
            if not m:  # jogos ainda sem data ("TBA")
                continue
            day, clock = f"{m[3]}-{m[2]}-{m[1]}", None
            if m[4]:
                local = pd.Timestamp(f"{day} {m[4]}", tz="Europe/Belgrade").tz_convert("Europe/Lisbon")
                day, clock = str(local.date()), local.strftime("%H:%M")
            pts = re.search(r"(\d+)\s*:\s*(\d+)", score)
            rows.append({
                "league": league_key, "date": date.fromisoformat(day), "time": clock,
                "round": f"Round {rnd[1]}", "home": home.strip(), "away": away.strip(),
                "home_points": float(pts[1]) if pts else None,
                "away_points": float(pts[2]) if pts else None,
                "home_code": hc.strip(), "away_code": ac.strip(),
            })
    return pd.DataFrame(rows, columns=COLUMNS + ["home_code", "away_code"])


def fetch_aba(league: BasketLeague, years: range) -> pd.DataFrame:
    """Liga ABA: a página de cada época tem a fase regular inteira."""
    frames = []
    for year in years:
        resp = requests.get(ABA_URL.format(season=year - 2000), headers=HEADERS, timeout=60)
        resp.raise_for_status()
        frames.append(parse_aba(resp.text, league.key))
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    return _unify_names(pd.concat(frames, ignore_index=True))


KBL_URL = "https://api.kbl.or.kr/match/list"
# o site manda estes cabeçalhos em todos os pedidos (sem eles a API responde 400)
KBL_HEADERS = {**HEADERS, "Channel": "WEB", "TeamCode": "XX", "lang": "ko"}
KBL_NAMES = {
    "06": "Suwon KT Sonicboom", "10": "Ulsan Hyundai Mobis Phoebus", "16": "Wonju DB Promy",
    "35": "Seoul Samsung Thunders", "50": "Changwon LG Sakers", "55": "Seoul SK Knights",
    "60": "Busan KCC Egis", "64": "Daegu KOGAS Pegasus", "66": "Goyang Sono Skygunners",
    "70": "Anyang Jung Kwan Jang Red Boosters",
}
# 01 fase regular, 03 playoffs, 04 final; ficam de fora a liga D, o EASL e o jogo das estrelas
KBL_GAME_CODES = {"01", "03", "04"}


def parse_kbl(games: list[dict], league_key: str = "KBL") -> pd.DataFrame:
    rows = []
    for g in games:
        if g.get("seasonGrade") != 1 or g.get("gameCode") not in KBL_GAME_CODES:
            continue
        start = g.get("gameStart") or ""
        when = pd.Timestamp(f"{g['gameDate']} {start[:2]}:{start[2:]}" if len(start) == 4
                            else g["gameDate"], tz="Asia/Seoul")
        done = g.get("isEnded") == 1 and g.get("scoreH") is not None
        rows.append({
            "league": league_key,
            "date": when.tz_convert("Europe/Lisbon").date() if len(start) == 4 else when.date(),
            "time": when.tz_convert("Europe/Lisbon").strftime("%H:%M") if len(start) == 4 else None,
            "round": None,
            "home": KBL_NAMES.get(g["tcodeH"], g["tnameH"]),
            "away": KBL_NAMES.get(g["tcodeA"], g["tnameA"]),
            "home_points": float(g["scoreH"]) if done else None,
            "away_points": float(g["scoreA"]) if done else None,
        })
    return pd.DataFrame(rows, columns=COLUMNS)


def fetch_kbl(league: BasketLeague, years: range) -> pd.DataFrame:
    games = []
    for year in years:
        resp = requests.get(KBL_URL, params={"fromDate": f"{year}0901", "toDate": f"{year + 1}0630"},
                            headers=KBL_HEADERS, timeout=60)
        resp.raise_for_status()
        games.extend(resp.json() or [])
    return parse_kbl(games, league.key)


SBL_API = "https://www.sblherr.se/api/sports-v2"


def parse_sbl(games: list[dict], league_key: str = "SBL") -> pd.DataFrame:
    rows = []
    for g in games:
        home, away = g["homeTeamInfo"], g["awayTeamInfo"]
        when = pd.Timestamp(g["rawStartDateTime"]).tz_convert("Europe/Lisbon")
        done = g.get("state") == "post-game" and home.get("score") is not None
        rows.append({
            "league": league_key, "date": when.date(), "time": when.strftime("%H:%M"),
            "round": None, "home": home["names"]["long"], "away": away["names"]["long"],
            "home_points": float(home["score"]) if done else None,
            "away_points": float(away["score"]) if done else None,
        })
    return pd.DataFrame(rows, columns=COLUMNS)


def fetch_sbl(league: BasketLeague, years: range) -> pd.DataFrame:
    """Basketligan (Suécia): fase regular e playoffs de cada época, um pedido por fase."""
    def get(path, **params):
        resp = requests.get(f"{SBL_API}/{path}", params=params, headers=HEADERS, timeout=60)
        resp.raise_for_status()
        return resp.json()

    filters = get("season-series-game-types-filter")
    series = filters["series"][0]["uuid"]
    seasons = {int(s["code"]): s["uuid"] for s in filters["season"] if s["code"].isdigit()}
    games = []
    for year in years:
        if year not in seasons:
            continue
        types = get("season-series-game-types-filter", season=seasons[year], series=series)
        for t in types["gameType"]:
            games.extend(get("game-schedule", seasonUuid=seasons[year], seriesUuid=series,
                             gameTypeUuid=t["uuid"]).get("gameInfo") or [])
    return parse_sbl(games, league.key)


KORI_API = "https://koripallo-api.torneopal.net/taso/rest"
# chave pública que o site de resultados manda em todos os pedidos
KORI_HEADERS = {**HEADERS, "accept": "json/df8e84j9xtdz269euy3h",
                "Referer": "https://tulospalvelu.basket.fi/"}
KORI_CATEGORY = "4"  # Korisliiga (masculino)


def parse_kori(matches: list[dict], league_key: str = "KOR") -> pd.DataFrame:
    rows = []
    for m in matches:
        if m.get("match_type") != "single" or not m.get("date"):
            continue  # "series" é o resumo de cada eliminatória dos playoffs
        clock = (m.get("time") or "")[:5]
        when = pd.Timestamp(f"{m['date']} {clock or '00:00'}", tz="Europe/Helsinki")
        when = when.tz_convert("Europe/Lisbon") if clock else when
        done = m.get("status") == "Played" and str(m.get("fs_A", "")).isdigit()
        rows.append({
            "league": league_key, "date": when.date(), "time": when.strftime("%H:%M") if clock else None,
            "round": None, "home": m["team_A_name"], "away": m["team_B_name"],
            "home_points": float(m["fs_A"]) if done else None,
            "away_points": float(m["fs_B"]) if done else None,
        })
    return pd.DataFrame(rows, columns=COLUMNS)


def fetch_kori(league: BasketLeague, years: range) -> pd.DataFrame:
    """Korisliiga (Finlândia): todos os jogos de cada época num pedido (a API só tem desde 2025-26)."""
    matches = []
    for year in years:
        resp = requests.get(f"{KORI_API}/getMatches", headers=KORI_HEADERS, timeout=60, params={
            "competition_id": f"huki{year % 100:02d}{(year + 1) % 100:02d}", "category_id": KORI_CATEGORY})
        resp.raise_for_status()
        matches.extend(resp.json().get("matches") or [])  # épocas antigas: erro sem jogos
    return parse_kori(matches, league.key)


BETEXPLORER = "https://www.betexplorer.com/basketball/{path}/{page}/"
_BE_RESULT = re.compile(
    r'class="in-match"><span>(?:<strong>)?([^<]+)(?:</strong>)?</span> - <span>(?:<strong>)?([^<]+)'
    r'(?:</strong>)?</span></a></td><td class="h-text-center"><a [^>]*>(\d+):(\d+)'
    r'.*?h-text-right h-text-no-wrap">([^<]*)<', re.S)
_BE_FIXTURE = re.compile(
    r'table-main__datetime">([^<]*)</td>\s*<td class="h-text-left"><a [^>]*class="in-match">'
    r'<span>([^<]+)</span> - <span>([^<]+)</span>', re.S)
# o betexplorer mostra as horas sempre em UTC+1, também no inverno
BETEXPLORER_TZ = "Etc/GMT-1"


def _be_day(text: str, today: date) -> date:
    """"Today", "Yesterday", "21.12." (ano corrente, que o site omite) ou "04.01.2027"."""
    if text.startswith(("Today", "Yesterday")):
        return today - timedelta(days=text.startswith("Yesterday"))
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


CBA_SCHEDULE = "https://portal-server.cbaleague.com/home/home_schedules"
# ID de cada clube no site oficial -> nome do betexplorer (os nomes oficiais mudam com o patrocinador)
CBA_TEAMS = {
    29115: "Beijing", 29117: "Xinjiang", 29118: "Jiangsu Dragons", 29124: "Guangdong",
    29125: "Shanghai", 29127: "Sichuan", 29128: "Zhejiang Guangsha", 29129: "Liaoning",
    29130: "Shandong", 29131: "Shenzhen", 29132: "Shanxi Zhongyu", 29133: "Nanjing Tongxi",
    29134: "Fujian", 29135: "Qingdao", 29136: "Beijing Royal Fighters", 29137: "Jilin",
    29138: "Tianjin", 29139: "Guangzhou", 29140: "Zhejiang Chouzhou", 100074683: "Ningbo Rockets",
}
CBA_SKIP_TYPES = {3}  # 3: torneio de pré-época por estações (campos neutros)
# Em 2025-26 a CBA passou às regras FIBA (quartos de 10 minutos): o total caiu de ~207 para
# ~178 pontos. No teste, usar só jogos desde então foi melhor do que juntar ou reescalar os antigos.
CBA_FIRST_SEASON = 2025


def parse_cba(games: list[dict], league_key: str = "CBA") -> pd.DataFrame:
    rows = []
    for g in games:
        if g.get("ScheduleTypeID") in CBA_SKIP_TYPES:
            continue
        when = pd.Timestamp(f"{g['dates']} {g.get('time') or '00:00'}", tz="Asia/Shanghai")
        if g.get("time"):
            when = when.tz_convert("Europe/Lisbon")
        done = g.get("Status") == 4 and g.get("HomeTeamScore") is not None
        rows.append({
            "league": league_key, "date": when.date(),
            "time": when.strftime("%H:%M") if g.get("time") else None,
            "round": f"Round {g['Round']}" if g.get("Round") else None,
            "home": CBA_TEAMS.get(g["HomeTeamID"], g["HomeTeamName"]),
            "away": CBA_TEAMS.get(g["VisitingTeamID"], g["VisitingTeamName"]),
            "home_points": float(g["HomeTeamScore"]) if done else None,
            "away_points": float(g["VisitingTeamScore"]) if done else None,
        })
    return pd.DataFrame(rows, columns=COLUMNS)


def fetch_cba(league: BasketLeague, years: range, today: date | None = None) -> pd.DataFrame:
    """Época atual (com o calendário que falta) do site oficial; épocas passadas do betexplorer."""
    today = today or date.today()
    current = season_start(today)
    frames = [fetch_betexplorer(f"china/cba-{y}-{y + 1}", league.key, today)
              for y in years if CBA_FIRST_SEASON <= y < current]
    if current in years:
        resp = requests.get(CBA_SCHEDULE, headers=HEADERS, timeout=60)
        resp.raise_for_status()
        frames.append(parse_cba(resp.json().get("data") or [], league.key))
    return pd.concat(frames, ignore_index=True)


def fetch_pba(league: BasketLeague, years: range, today: date | None = None) -> pd.DataFrame:
    """PBA (Filipinas): três torneios por época (Philippine, Commissioner's e Governors' Cup).
    O site oficial tem proteção anti-robôs, por isso vem tudo do betexplorer."""
    today = today or date.today()
    resp = requests.get("https://www.betexplorer.com/basketball/philippines/", headers=HEADERS,
                        timeout=60)
    resp.raise_for_status()
    slugs = dict.fromkeys(re.findall(r'href="/basketball/philippines/(pba-[a-z0-9-]+)/"', resp.text))
    df = pd.concat([fetch_betexplorer(f"philippines/{s}", league.key, today, fixtures=True)
                    for s in slugs], ignore_index=True)
    return df[pd.to_datetime(df["date"]) >= pd.Timestamp(years.start, 7, 1)]
