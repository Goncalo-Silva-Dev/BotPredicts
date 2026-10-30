"""Resultados ao vivo: liga cada jogo do site ao jogo da ESPN (API pública, sem chave).

Quando o site é gerado, procura-se o id da ESPN de cada jogo dos próximos dias pelos nomes
das equipas (as fontes escrevem-nos de maneira diferente: "Man United" e "Manchester United").
Depois é a página que, aberta no browser, vai buscar o resultado à ESPN a cada 30 segundos.
"""
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from difflib import SequenceMatcher

import pandas as pd
import requests

SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/{slug}/scoreboard"
# Ligas do site que a ESPN tem; as outras ficam sem resultado ao vivo
ESPN_SLUGS = {
    "POR": "soccer/por.1", "ESP": "soccer/esp.1", "ENG": "soccer/eng.1", "FRA": "soccer/fra.1",
    "ITA": "soccer/ita.1", "GER": "soccer/ger.1", "USA": "soccer/usa.1", "BRA": "soccer/bra.1",
    "TUR": "soccer/tur.1", "NED": "soccer/ned.1", "BEL": "soccer/bel.1", "SAU": "soccer/ksa.1",
    "NBA": "basketball/nba", "NBL": "basketball/nbl", "NHL": "hockey/nhl",
}
# Palavras que não distinguem equipas (siglas de clube e afins)
NOISE = {"fc", "cf", "sc", "ac", "afc", "cd", "ud", "sd", "rc", "ca", "ec", "se", "sv", "vfb", "vfl",
         "tsg", "fsv", "bc", "kv", "krc", "kaa", "rsc", "sk", "fk", "as", "ss", "ssc", "us", "ogc",
         "losc", "de", "da", "do", "di", "del", "la", "le", "club", "clube", "futebol", "calcio",
         "1", "04", "05", "07", "1899", "1846", "1907", "1913", "sp", "rj", "sad", "the"}
# Abreviaturas do football-data.co.uk
ALIASES = {"man": "manchester", "utd": "united", "nott'm": "nottingham", "ath": "atletico",
           "ein": "eintracht", "m'gladbach": "monchengladbach", "sp": "sporting", "st": "saint",
           "inter": "internazionale", "psg": "paris", "wolves": "wolverhampton"}
# Nomes inteiros que não se parecem nada entre as fontes
SAME_TEAM = {"la coruna": "deportivo", "lafc": "los angeles", "espanol": "espanyol"}
MIN_SCORE = 0.5  # semelhança mínima de cada equipa (0 a 1)
MAX_GAP = 3  # dias de diferença aceites entre a data do site e a da ESPN


def _words(name: str) -> list[str]:
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower().strip()
    name = SAME_TEAM.get(name, name)
    words = [ALIASES.get(w, w) for w in re.split(r"[^a-z0-9']+", name) if w]
    words = [w.replace("'", "") for w in words]
    return [w for w in words if w and w not in NOISE] or words


def similarity(a: str, b: str) -> float:
    """0 a 1: quão parecidos são dois nomes da mesma equipa escritos por fontes diferentes."""
    wa, wb = _words(a), _words(b)
    if not wa or not wb:
        return 0.0
    if " ".join(wa) == " ".join(wb):
        return 1.0

    def best(w, others):  # palavra igual, uma é o início da outra (Atl./Atletico) ou parecida
        return max((1.0 if w == o else 0.9 if len(min(w, o, key=len)) >= 3 and
                    (w.startswith(o) or o.startswith(w)) else SequenceMatcher(None, w, o).ratio())
                   for o in others)
    # cada palavra do nome mais curto tem de aparecer no mais longo
    short, long_ = sorted((wa, wb), key=len)
    words = sum(best(w, long_) for w in short) / len(short)
    whole = SequenceMatcher(None, "".join(wa), "".join(wb)).ratio()
    return max(words * (0.85 + 0.15 * len(short) / len(long_)), whole)


def parse_events(events: list[dict], slug: str, day: str) -> list[dict]:
    rows = []
    for ev in events:
        comp = (ev.get("competitions") or [{}])[0]
        teams = {c.get("homeAway"): c["team"] for c in comp.get("competitors", [])}
        if set(teams) != {"home", "away"}:
            continue
        start = pd.Timestamp(ev["date"])
        rows.append({"id": str(ev["id"]), "ref": f"{slug}|{day}|{ev['id']}|{start:%Y-%m-%dT%H:%MZ}",
                     "start": start.tz_convert("Europe/Lisbon"),
                     "home": [teams["home"].get(k) or "" for k in ("displayName", "shortDisplayName", "name")],
                     "away": [teams["away"].get(k) or "" for k in ("displayName", "shortDisplayName", "name")]})
    return rows


def _day(slug: str, day: date, tries: int = 3) -> list[dict]:
    param = day.strftime("%Y%m%d")
    for attempt in range(tries):
        try:
            resp = requests.get(SCOREBOARD.format(slug=slug), params={"dates": param}, timeout=20)
            resp.raise_for_status()
            return parse_events(resp.json().get("events", []), slug, param)
        except requests.RequestException:
            if attempt == tries - 1:
                return []  # sem resultado ao vivo nesse dia; o resto do site não depende disto
            time.sleep(2 ** attempt)
    return []


def fetch_events(slug: str, start: date, end: date) -> list[dict]:
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        found = [e for day in pool.map(lambda d: _day(slug, d), days) for e in day]
    unique = {}
    for e in found:  # o mesmo jogo pode aparecer em dois dias seguidos
        unique.setdefault(e["id"], e)
    return list(unique.values())


def match(games: pd.DataFrame, events: list[dict]) -> pd.Series:
    """Referência da ESPN ("slug|dia|id|início em UTC") de cada jogo, ou None; um jogo da ESPN só serve um."""
    def sim(name, names):
        return max(similarity(name, n) for n in names if n)
    pairs = []
    for idx, g in games.iterrows():
        day = pd.Timestamp(g["date"]).date()
        for e in events:
            gap = abs((e["start"].date() - day).days)
            if gap > MAX_GAP:  # outro fuso ou jogo mudado de dia depois do calendário publicado
                continue
            h, a = sim(g["home"], e["home"]), sim(g["away"], e["away"])
            if min(h, a) >= MIN_SCORE:
                pairs.append((h + a - 0.15 * gap, idx, e["id"], e["ref"]))
    out, used = {}, set()
    for _, idx, eid, ref in sorted(pairs, key=lambda p: -p[0]):
        if idx not in out and eid not in used:
            out[idx] = ref
            used.add(eid)
    return pd.Series([out.get(i) for i in games.index], index=games.index, dtype=object)


def attach(df: pd.DataFrame, today: date, days: int = 3) -> pd.DataFrame:
    """Junta a coluna "espn" aos jogos que começam até `days` dias depois de hoje."""
    if df is None or df.empty:
        return df
    df = df.copy()
    df["espn"] = None
    dates = pd.to_datetime(df["date"]).dt.date
    soon = (dates >= today - timedelta(days=1)) & (dates <= today + timedelta(days=days))
    for key, slug in ESPN_SLUGS.items():
        rows = soon & (df["league"] == key)
        if not rows.any():
            continue
        events = fetch_events(slug, today - timedelta(days=1 + MAX_GAP),
                              today + timedelta(days=days + MAX_GAP))
        found = match(df[rows], events)
        df.loc[found.index, "espn"] = found
        print(f"ao vivo {key}: {found.notna().sum()} de {rows.sum()} jogos ligados à ESPN")
    return df
