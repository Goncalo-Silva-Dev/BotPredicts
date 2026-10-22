"""Calendário completo da época a partir do openfootball (football.json no GitHub).

Os nomes das equipas no openfootball são os oficiais ("Sport Lisboa e Benfica") e
no football-data.co.uk são abreviados ("Benfica"); match_teams liga uns aos outros.
"""
import difflib
import re
import unicodedata
from datetime import date

import pandas as pd
import requests

from analise.leagues import League

BASE_URL = "https://raw.githubusercontent.com/openfootball/football.json/master"

# Pares que a comparação automática não acerta: nome openfootball -> football-data
ALIASES = {
    "Sporting Clube de Portugal": "Sp Lisbon",
    "Athletic Club": "Ath Bilbao",
    "Club Atlético de Madrid": "Ath Madrid",
    "FC Internazionale Milano": "Inter",
    "Paris Saint-Germain FC": "Paris SG",
    "Olympique Lyonnais": "Lyon",
    "CA Paranaense": "Athletico-PR",
    "CA Mineiro": "Atletico-MG",
    "Nottingham Forest FC": "Nott'm Forest",
    "Manchester City FC": "Man City",
    "Manchester United FC": "Man United",
    "RCD Espanyol de Barcelona": "Espanol",
    "Stade Rennais FC 1901": "Rennes",
}

# Palavras que não distinguem clubes
STOP = {
    "fc", "cf", "ac", "as", "sc", "cd", "ud", "rc", "rcd", "ca", "cr", "ec", "se", "ss", "ssc",
    "us", "afc", "acf", "bc", "ogc", "sco", "aj", "es", "gd", "cs", "fbc", "fr", "af", "fbpa",
    "club", "clube", "de", "da", "do", "del", "la", "e", "calcio", "futbol", "balompie",
    "real", "sport", "sporting", "united", "city", "town", "athletic", "atletico", "olympique",
    "stade", "racing", "hotspur", "hove", "albion", "alsace",
}


def season_path(league: League, today: date) -> str:
    if league.fd_extra:  # ligas com calendário anual (Brasil)
        return str(today.year)
    start = today.year if today.month >= 7 else today.year - 1
    return f"{start}-{(start + 1) % 100:02d}"


def _norm(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9 ]", " ", s)


def _tokens(name: str) -> set[str]:
    return {t for t in _norm(name).split() if t not in STOP and not t.isdigit()}


def _score(of_name: str, fd_name: str) -> float:
    a, b = _tokens(of_name), _tokens(fd_name)
    overlap = len(a & b) / len(b) if b else 0.0
    ratio = difflib.SequenceMatcher(None, " ".join(sorted(a)), " ".join(sorted(b))).ratio()
    prefix = any(x.startswith(y) or y.startswith(x) for x in a for y in b if min(len(x), len(y)) >= 4)
    return overlap + 0.5 * ratio + 0.3 * prefix


def match_teams(of_names, fd_names) -> dict[str, str]:
    """Liga cada nome do openfootball ao nome do histórico (ou fica de fora se não houver)."""
    fd_names = set(fd_names)
    mapping = {n: ALIASES[n] for n in of_names if ALIASES.get(n) in fd_names}
    left_of = [n for n in of_names if n not in mapping]
    left_fd = fd_names - set(mapping.values())
    pairs = sorted(((_score(o, f), o, f) for o in left_of for f in left_fd), reverse=True)
    for score, o, f in pairs:
        if score < 0.6:
            break
        if o not in mapping and f in left_fd:
            mapping[o] = f
            left_fd.discard(f)
    return mapping


def fetch_season(league: League, today: date | None = None) -> pd.DataFrame:
    """Todos os jogos da época atual, com os nomes originais do openfootball."""
    today = today or date.today()
    url = f"{BASE_URL}/{season_path(league, today)}/{league.of_code}.json"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    rows = [{
        "league": league.key,
        "round": m.get("round"),
        "date": date.fromisoformat(m["date"]),
        "time": m.get("time"),
        "home": m["team1"],
        "away": m["team2"],
        "played": bool(m.get("score")),
    } for m in resp.json().get("matches", [])]
    return pd.DataFrame(rows)
