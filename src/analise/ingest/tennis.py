"""Jogos de ténis (singulares) da ESPN: ATP e WTA, com o resultado de cada set.

Um pedido por ano e circuito traz todos os torneios desse ano (ATP 250 a Grand Slams, WTA 125
a Grand Slams, com a qualificação). Os Challengers e os ITF não estão na ESPN. Os torneios
mistos (Grand Slams, Indian Wells, Miami, Madrid, Roma...) vêm nos dois circuitos, por isso
cada jogo fica só uma vez (pelo id da ESPN).
"""
import time
import unicodedata
from dataclasses import dataclass
from datetime import date

import pandas as pd
import requests

ESPN_URL = "https://site.api.espn.com/apis/site/v2/sports/tennis/{slug}/scoreboard"
SLAMS = ("australian open", "roland garros", "wimbledon", "us open")
MAX_SETS = 5
COLUMNS = (["id", "league", "date", "time", "tournament", "round", "surface", "best_of",
            "home", "away", "home_id", "away_id", "status", "home_sets", "away_sets"]
           + [f"{side}_s{k}" for k in range(1, MAX_SETS + 1) for side in ("home", "away")])


@dataclass(frozen=True)
class TennisLeague:
    key: str
    name: str
    grouping: str  # tipo de jogo na ESPN


TENNIS_LEAGUES = {l.key: l for l in [
    TennisLeague("ATP", "ATP (masculino)", "mens-singles"),
    TennisLeague("WTA", "WTA (feminino)", "womens-singles"),
]}
GROUPING_TO_LEAGUE = {l.grouping: l.key for l in TENNIS_LEAGUES.values()}

ROUNDS = {
    "Round 1": "1.ª ronda", "Round 2": "2.ª ronda", "Round 3": "3.ª ronda", "Round 4": "4.ª ronda",
    "Round of 128": "1.ª ronda", "Round of 64": "2.ª ronda", "Round of 32": "Ronda de 32",
    "Round of 16": "Oitavos de final", "Quarterfinal": "Quartos de final",
    "Quarterfinals": "Quartos de final", "Semifinal": "Meias-finais", "Semifinals": "Meias-finais",
    "Final": "Final", "Round Robin": "Fase de grupos",
    "Qualifying 1st Round": "Qualificação, 1.ª ronda", "Qualifying 2nd Round": "Qualificação, 2.ª ronda",
    "Qualifying 3rd Round": "Qualificação, 3.ª ronda", "Qualifying Final": "Qualificação, ronda final",
}


def _plain(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode()
    return text.lower()


# Piso dos torneios: a ESPN não o dá. Terra batida e relva por cidade (com os meses em que lá se
# joga nesse piso); o resto é piso rápido (duro, ao ar livre ou em pavilhão).
CLAY = {
    "houston": None, "marrakech": None, "bucharest": None, "monte-carlo": None, "munich": None,
    "barcelona": None, "madrid": (3, 4, 5), "rome": (4, 5, 7), "geneva": None, "hamburg": None,
    "paris": (5, 6), "bastad": None, "gstaad": None, "umag": None, "kitzbuhel": None,
    "estoril": None, "buenos aires": None, "rio de janeiro": None, "santiago,": None,
    "charleston": None, "bogota": None, "la bisbal": None, "rouen": None, "oeiras": (4, 5),
    "saint malo": None, "saint-malo": None, "vic,": None, "parma": None, "rabat": None,
    "strasbourg": None, "bari": None, "foggia": None, "makarska": None, "valencia": None,
    "grado": None, "contrexeville": None, "iasi": None, "porto,": (6, 7), "prague": None,
    "palermo": None, "warsaw": None, "montreux": None, "tolentino": None, "rende": None,
    "istanbul": (4, 5, 7), "antalya": (2, 3, 4, 5), "dubrovnik": None, "modena": None,
    "brescia": None, "figueira": None, "athens": (5, 6, 7), "targu mures": None,
    "ljubljana": None, "cali,": None, "barranquilla": None, "samsun": None, "stuttgart": (4,),
    "linz": (4,), "florianopolis": None, "bucuresti": None, "lyon": (5, 6), "cordoba": None,
    "san sebastian": None, "budapest": None, "belgrade": (4, 5), "caldas da rainha": None,
    "lausanne": None,
}
GRASS = {
    "'s-hertogenbosch": None, "halle": None, "london": (6, 7), "eastbourne": None,
    "mallorca": (6,), "stuttgart": (6,), "berlin": None, "nottingham": None, "bad homburg": None,
    "birmingham": None, "ilkley": None, "newport": None, "queen's": None,
}


def _matches(table: dict, venue: str, month: int) -> bool:
    for city, months in table.items():
        if city in venue and (months is None or month in months):
            return True
    return False


def surface(tournament: str, venue: str | None, month: int) -> str:
    """'terra', 'relva' ou 'rapido' (duro e pavilhão)."""
    name, where = _plain(tournament), _plain(venue or "") + ","
    if "wimbledon" in name:
        return "relva"
    if "roland garros" in name:
        return "terra"
    if "australian open" in name or "us open" in name:
        return "rapido"
    # na relva olha-se primeiro (Estugarda tem terra em abril e relva em junho)
    if _matches(GRASS, where, month):
        return "relva"
    if _matches(CLAY, where, month):
        return "terra"
    return "rapido"


def _sets(competitor: dict) -> list[float]:
    return [float(ls["value"]) for ls in competitor.get("linescores") or [] if "value" in ls]


def parse_year(events: list[dict]) -> list[dict]:
    rows = []
    for ev in events:
        tournament = ev.get("name") or ""
        slam = any(s in _plain(tournament) for s in SLAMS)
        for grouping in ev.get("groupings") or []:
            league = GROUPING_TO_LEAGUE.get(grouping.get("grouping", {}).get("slug"))
            if league is None:
                continue
            for c in grouping.get("competitions") or []:
                players = sorted(c.get("competitors") or [], key=lambda x: x.get("order", 0))
                if len(players) != 2 or any(p.get("type") != "athlete" for p in players):
                    continue
                ids = [str(p.get("id", "")) for p in players]
                names = [(p.get("athlete") or {}).get("displayName") for p in players]
                if any(i.startswith("-") or not i for i in ids) or any(not n or n == "TBD" for n in names):
                    continue  # jogo ainda sem jogadores (quadro por preencher)
                status = c["status"]["type"]["name"]
                when = pd.Timestamp(c.get("date") or ev["date"]).tz_convert("Europe/Lisbon")
                rnd = (c.get("round") or {}).get("displayName") or ""
                venue = (c.get("venue") or {}).get("fullName")
                qualifying = rnd.startswith("Qualifying")
                row = {
                    "id": str(c["id"]), "league": league, "date": when.date(),
                    "time": when.strftime("%H:%M") if c.get("timeValid", True) else None,
                    "tournament": tournament, "round": ROUNDS.get(rnd, rnd),
                    "surface": surface(tournament, venue, when.month),
                    "best_of": 5 if slam and league == "ATP" and not qualifying else 3,
                    "home": names[0], "away": names[1], "home_id": ids[0], "away_id": ids[1],
                    "status": {"STATUS_FINAL": "final", "STATUS_RETIRED": "desistencia",
                               "STATUS_WALKOVER": "falta de comparencia"}.get(
                                   status, "por jogar" if c["status"]["type"].get("state") == "pre"
                                   else ("a decorrer" if c["status"]["type"].get("state") == "in" else status)),
                    "home_sets": None, "away_sets": None,
                }
                if status in ("STATUS_FINAL", "STATUS_RETIRED"):
                    a, b = _sets(players[0]), _sets(players[1])
                    for k, (x, y) in enumerate(zip(a, b), start=1):
                        if k <= MAX_SETS:
                            row[f"home_s{k}"], row[f"away_s{k}"] = x, y
                    won = [ls.get("winner") for ls in players[0].get("linescores") or []]
                    lost = [ls.get("winner") for ls in players[1].get("linescores") or []]
                    row["home_sets"] = float(sum(1 for w in won if w))
                    row["away_sets"] = float(sum(1 for w in lost if w))
                    # vencedor do jogo (numa desistência pode ter menos sets)
                    row["winner"] = 0 if players[0].get("winner") else (1 if players[1].get("winner") else None)
                rows.append(row)
    return rows


def _get(slug: str, year: int, tries: int = 4) -> list[dict]:
    for attempt in range(tries):
        try:
            resp = requests.get(ESPN_URL.format(slug=slug), params={"dates": str(year)}, timeout=120)
            resp.raise_for_status()
            return resp.json().get("events", [])
        except requests.RequestException:
            if attempt == tries - 1:
                raise
            time.sleep(2 ** attempt)
    return []


def fetch_matches(today: date | None = None, years: int = 4) -> pd.DataFrame:
    """Jogos de singulares dos últimos `years` anos civis (o atual incluído), dos dois circuitos."""
    today = today or date.today()
    rows = []
    for year in range(today.year - years + 1, today.year + 1):
        for slug in ("atp", "wta"):
            rows += parse_year(_get(slug, year))
    return to_frame(rows)


def to_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=COLUMNS + ["winner"])
    df = df.drop_duplicates(subset=["id"], keep="last")
    return df.sort_values(["date", "time", "id"], na_position="last").reset_index(drop=True)
