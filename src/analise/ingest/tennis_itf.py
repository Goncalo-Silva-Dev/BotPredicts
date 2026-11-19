"""Ténis abaixo da ATP e da WTA, do betexplorer.com: torneios femininos ITF (W15 a W100) e
Challengers masculinos, que a ESPN não tem e que as casas de apostas oferecem todos os dias.

O betexplorer tem uma página por dia com todos os resultados de ténis (com os sets e o piso) e uma
página com os próximos jogos, já com as odds médias das casas. Para o Elo contam todos os jogos
de singulares que lá estão (femininos: WTA, WTA 125 e ITF; masculinos: ATP, Challenger e ITF),
porque os jogadores passam de uns torneios para os outros; no site só aparecem os ITF femininos e
os Challengers (os outros já vêm da ESPN).

Os dias passados não mudam: cada dia fica guardado em `cache` (pasta) para não voltar a pedi-lo.
"""
import html as html_lib
import os
import re
import time
import unicodedata
from datetime import date, timedelta

import pandas as pd
import requests

from analise.ingest import betexplorer
from analise.ingest.tennis import COLUMNS, MAX_SETS

RESULTS_URL = "https://www.betexplorer.com/tennis/results/"
NEXT_URL = "https://www.betexplorer.com/tennis/next/"
YEARS = 2  # anos de jogos para o Elo
# categorias de singulares no betexplorer e a liga com que ficam. No site só aparecem os ITF
# femininos e os Challengers masculinos (ATP e WTA já vêm da ESPN); as outras servem para o Elo.
CATEGORIES = {"ITF Women - Singles": "ITF", "Challenger Women - Singles": "W125",
              "WTA - Singles": "WTAB",
              "Challenger Men - Singles": "CH", "ITF Men - Singles": "ITFM", "ATP - Singles": "ATPB"}
SHOWN = ["ITF", "CH"]  # ligas que aparecem no site
MEN = {"CH", "ITFM", "ATPB"}  # os homens têm o identificador com "m:" (nunca se cruzam com elas)
CACHE_VERSION = "v2"  # v1 só tinha os jogos femininos
SURFACES = {"clay": "terra", "grass": "relva"}  # o resto (hard, indoor) é piso rápido

_TBODY = re.compile(r"<tbody[^>]*>(.*?)</tbody>", re.S)
_TOURNAMENT = re.compile(r'tournament--tennis">.*?</i>([^<]*)</a>', re.S)
_ROW = re.compile(r"<tr data-fro.*?</tr>", re.S)
_DT = re.compile(r'data-dt="(\d+),(\d+),(\d+),(\d+),(\d+)"')
_STATUS = re.compile(r'table-main__time[^"]*">([^<]*)<')
_LINK = re.compile(r'href="/tennis/[^/]+/[^/]+/([^/]+)/([A-Za-z0-9]+)/"')
_PLAYER = re.compile(r'teamLine--(home|away)">(.*?)</span>(?=<span class="table-main__teamLine|</a>)', re.S)
_RESULT = re.compile(r'table-main__result">\s*<a[^>]*>(.*?)</a>', re.S)
_PARTIAL = re.compile(r'table-main__partial[^>]*>(.*?)</td>', re.S)
_ODD = re.compile(r'data-odd="([\d.]+)"')


def _plain(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "-", text).strip("-")


def _surname(short: str) -> list[str]:
    """"Lee C. Y." -> ["lee"]; "Huang Yujia" -> ["huang", "yujia"]."""
    return [t for t in _plain(re.sub(r"\b\w\.", "", short)).split("-") if t]


def _has(part: list[str], name: list[str]) -> int:
    """2 se o nome abre esta parte, 1 se a fecha (nome próprio primeiro), 0 se não está lá."""
    n = len(name)
    return 2 if part[:n] == name else (1 if part[-n:] == name else 0)


def player_ids(slug: str, home: str, away: str) -> tuple[str, str]:
    """Separa o caminho do jogo ("garland-joanna-broadus-savannah") nos nomes completos das duas
    jogadoras, que servem de identificador (o nome curto "Zhang S." repete-se). Umas vêm com o
    apelido primeiro e outras com o nome próprio primeiro, por isso o identificador junta as
    palavras por ordem alfabética. Se não der para separar, fica o nome curto."""
    tokens = slug.split("-")
    hs, as_ = _surname(home), _surname(away)
    best, score = None, 0
    # com uma inicial no nome curto ("Hageman M."), o nome completo tem mais do que o apelido
    hmin, amin = len(hs) + ("." in home), len(as_) + ("." in away)
    for i in range(hmin, len(tokens) - amin + 1):
        a, b = _has(tokens[:i], hs), _has(tokens[i:], as_)
        if hs and as_ and a and b and a + b > score:
            best, score = i, a + b
    if best is None:
        return _plain(home), _plain(away)
    return "-".join(sorted(tokens[:best])), "-".join(sorted(tokens[best:]))


def _sets(partial: str) -> list[tuple[float, float]]:
    partial = re.sub(r"<sup>\d+</sup>", "", partial)
    return [(float(a), float(b)) for a, b in re.findall(r"(\d+):(\d+)", partial)]


def parse_page(page: str) -> pd.DataFrame:
    """Jogos de singulares de uma página do betexplorer (resultados ou próximos jogos)."""
    rows = []
    for body in _TBODY.findall(page):
        m = _TOURNAMENT.search(body)
        if not m:
            continue
        title = html_lib.unescape(m.group(1)).strip()
        category, _, rest = title.partition(": ")
        league = CATEGORIES.get(category)
        if league is None:
            continue
        name, _, surf = rest.rpartition(", ")
        surface = SURFACES.get(surf.strip().lower(), "rapido")
        for row in _ROW.findall(body):
            link, dt, status = _LINK.search(row), _DT.search(row), _STATUS.search(row)
            players = dict((side, re.sub("<[^>]+>", "", txt)) for side, txt in _PLAYER.findall(row))
            if not (link and dt and status and len(players) == 2):
                continue
            home, away = (html_lib.unescape(players[s]).strip() for s in ("home", "away"))
            if not home or not away:
                continue
            d, mo, y, h, mi = (int(x) for x in dt.groups())
            # o betexplorer mostra as horas em UTC+1 (também no inverno)
            when = pd.Timestamp(y, mo, d, h, mi, tz=betexplorer.BETEXPLORER_TZ).tz_convert("Europe/Lisbon")
            home_id, away_id = player_ids(link.group(1), home, away)
            if league in MEN:
                home_id, away_id = "m:" + home_id, "m:" + away_id
            res = _RESULT.search(row)
            result = re.sub("<[^>]+>", "", res.group(1)).strip() if res else ""
            out = {"id": "be-" + link.group(2), "league": league, "date": when.date(),
                   "time": when.strftime("%H:%M"), "tournament": name.strip(), "round": "",
                   "surface": surface, "best_of": 3, "home": home, "away": away,
                   "home_id": home_id, "away_id": away_id, "home_sets": None, "away_sets": None,
                   "winner": None, "odds_home": None, "odds_away": None}
            st = status.group(1).strip()
            if st == "FIN":
                if "CAN" in result or "WO" in result or "INT" in result or not result:
                    continue  # cancelado, falta de comparência ou interrompido
                strong = [s for s, txt in _PLAYER.findall(row) if "<strong>" in txt]
                if len(strong) != 1:
                    continue
                out["winner"] = 0 if strong[0] == "home" else 1
                out["status"] = "desistencia" if "RET" in result else "final"
                part = _PARTIAL.search(row)
                sets = _sets(part.group(1)) if part else []
                if out["status"] == "final":
                    for k, (a, b) in enumerate(sets[:MAX_SETS], start=1):
                        out[f"home_s{k}"], out[f"away_s{k}"] = a, b
                    out["home_sets"] = float(sum(a > b for a, b in sets))
                    out["away_sets"] = float(sum(b > a for a, b in sets))
            elif re.fullmatch(r"\d\d:\d\d", st):
                out["status"] = "por jogar"
                odds = _ODD.findall(row)
                if len(odds) >= 2:
                    out["odds_home"], out["odds_away"] = float(odds[0]), float(odds[1])
            else:
                continue  # a decorrer ou adiado
            rows.append(out)
    return pd.DataFrame(rows, columns=COLUMNS + ["winner", "odds_home", "odds_away"])


def _day(day: date, cache: str | None) -> pd.DataFrame:
    path = os.path.join(cache, f"{CACHE_VERSION}-{day.isoformat()}.csv") if cache else None
    if path and os.path.exists(path):
        return pd.read_csv(path, dtype={"id": str, "home_id": str, "away_id": str},
                           parse_dates=["date"]).assign(date=lambda d: d["date"].dt.date)
    for attempt in range(3):
        try:
            df = parse_page(betexplorer.get(RESULTS_URL, year=day.year, month=day.month, day=day.day))
            break
        except requests.RequestException as e:
            if attempt == 2:  # um dia que falhe não estraga o resto (e não fica guardado)
                print(f"ténis ITF: falhou {day} ({e.__class__.__name__})", flush=True)
                return pd.DataFrame(columns=COLUMNS + ["winner", "odds_home", "odds_away"])
            time.sleep(10 * (attempt + 1))
    if path:
        os.makedirs(cache, exist_ok=True)
        df.to_csv(path, index=False)
    return df


def fetch_matches(today: date | None = None, years: int = YEARS, cache: str | None = None) -> pd.DataFrame:
    """Resultados dos últimos `years` anos e os próximos jogos (com as odds médias das casas)."""
    today = today or date.today()
    cache = cache if cache is not None else os.environ.get("TENNIS_CACHE_DIR")
    frames = []
    for n in range(int(365.25 * years), -1, -1):
        day = today - timedelta(days=n)
        # os 2 últimos dias ainda podem mudar (jogos acabados depois), não ficam guardados
        frames.append(_day(day, cache if n >= 3 else None))
    frames.append(parse_page(betexplorer.get(NEXT_URL)))
    df = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    # o mesmo jogo pode estar em dois dias (fusos) ou nos próximos jogos e nos resultados
    df["_done"] = df["status"] != "por jogar"
    df = df.sort_values("_done").drop_duplicates(subset=["id"], keep="last").drop(columns="_done")
    return df.sort_values(["date", "time", "id"]).reset_index(drop=True)
