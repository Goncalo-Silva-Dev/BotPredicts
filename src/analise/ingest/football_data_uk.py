"""Histórico de resultados e odds de fecho do football-data.co.uk."""
import io
from datetime import date

import pandas as pd
import requests
from sqlalchemy import delete

from analise.db import matches
from analise.ingest import betexplorer
from analise.leagues import League

BASE_URL = "https://football-data.co.uk"

# Colunas de odds por ordem de preferência: fecho primeiro, depois abertura
# (as épocas antigas não têm odds de fecho da média do mercado).
ODDS_AVG = [("AvgCH", "AvgCD", "AvgCA"), ("AvgH", "AvgD", "AvgA"), ("BbAvH", "BbAvD", "BbAvA")]
ODDS_PIN = [("PSCH", "PSCD", "PSCA"), ("PSH", "PSD", "PSA")]


def season_code(start_year: int) -> str:
    """2025 -> '2526', o formato dos URLs das ligas principais."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def season_label(start_year: int, calendar_year: bool) -> str:
    return str(start_year) if calendar_year else f"{start_year}-{(start_year + 1) % 100:02d}"


def csv_url(league: League, start_year: int | None = None) -> str:
    if league.fd_extra:
        return f"{BASE_URL}/new/{league.fd_code}.csv"
    return f"{BASE_URL}/mmz4281/{season_code(start_year)}/{league.fd_code}.csv"


def fetch_csv(url: str) -> pd.DataFrame:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    # Os ficheiros vêm em latin-1, alguns com BOM, e por vezes com colunas vazias no fim
    text = resp.content.removeprefix(b"\xef\xbb\xbf").decode("latin-1")
    return pd.read_csv(io.StringIO(text), on_bad_lines="skip")


def _pick(df: pd.DataFrame, choices) -> pd.DataFrame:
    """Para cada linha usa o primeiro trio de colunas de odds que tiver valores."""
    out = pd.DataFrame(index=df.index, columns=["h", "d", "a"], dtype=float)
    for cols in choices:
        if not all(c in df.columns for c in cols):
            continue
        part = df[list(cols)].apply(pd.to_numeric, errors="coerce")
        part.columns = ["h", "d", "a"]
        out = out.fillna(part)
    return out


def _int(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(pd.NA, index=df.index, dtype="Int64")
    return pd.to_numeric(df[col], errors="coerce").astype("Int64")


def normalize(df: pd.DataFrame, league: League, season: str | None = None) -> pd.DataFrame:
    """Converte um CSV do football-data.co.uk para o esquema da tabela matches."""
    if league.fd_extra:
        df = df.rename(columns={"Home": "HomeTeam", "Away": "AwayTeam", "HG": "FTHG", "AG": "FTAG"})
        seasons = df["Season"].astype(str).str.slice(0, 4)
    else:
        seasons = pd.Series(season, index=df.index)

    df = df.dropna(subset=["HomeTeam", "AwayTeam", "Date"])
    seasons = seasons.loc[df.index]
    avg = _pick(df, ODDS_AVG)
    pin = _pick(df, ODDS_PIN)

    out = pd.DataFrame({
        "league": league.key,
        "season": seasons,
        "date": pd.to_datetime(df["Date"], dayfirst=True, format="mixed").dt.date,
        "home": df["HomeTeam"].str.strip(),
        "away": df["AwayTeam"].str.strip(),
        "home_goals": _int(df, "FTHG"),
        "away_goals": _int(df, "FTAG"),
        "home_goals_ht": _int(df, "HTHG"),
        "away_goals_ht": _int(df, "HTAG"),
        "home_shots": _int(df, "HS"),
        "away_shots": _int(df, "AS"),
        "home_shots_target": _int(df, "HST"),
        "away_shots_target": _int(df, "AST"),
        "odds_home_avg": avg["h"], "odds_draw_avg": avg["d"], "odds_away_avg": avg["a"],
        "odds_home_pin": pin["h"], "odds_draw_pin": pin["d"], "odds_away_pin": pin["a"],
        "source": "football-data.co.uk",
    })
    return out.drop_duplicates(subset=["league", "date", "home", "away"])


def save(engine, df: pd.DataFrame) -> int:
    """Substitui as épocas presentes em df (idempotente)."""
    if df.empty:
        return 0
    records = df.astype(object).where(df.notna(), None).to_dict("records")
    with engine.begin() as conn:
        for (league, season) in df[["league", "season"]].drop_duplicates().itertuples(index=False):
            conn.execute(delete(matches).where(matches.c.league == league, matches.c.season == season))
        conn.execute(matches.insert(), records)
    return len(records)


def from_betexplorer(df: pd.DataFrame, league: League, season: str) -> pd.DataFrame:
    """Resultados do betexplorer no esquema da tabela matches (sem intervalo, remates nem odds)."""
    df = df.dropna(subset=["home_points", "away_points"])
    out = pd.DataFrame({
        "league": league.key, "season": season, "date": df["date"],
        "home": df["home"], "away": df["away"],
        "home_goals": df["home_points"].astype("Int64"), "away_goals": df["away_points"].astype("Int64"),
        "source": "betexplorer.com",
    })
    return out.drop_duplicates(subset=["league", "date", "home", "away"])


def load_history(engine, league: League, n_seasons: int, today: date | None = None) -> int:
    today = today or date.today()
    if league.fd_code is None:
        current = today.year if today.month >= 7 else today.year - 1
        total = 0
        for start in range(current - n_seasons + 1, current + 1):
            path = league.be_path if start == current else league.be_history.format(
                y=start, y1=start + 1)
            try:
                df = betexplorer.fetch_betexplorer(path, league.key, today)
            except requests.HTTPError as e:
                if e.response is None or e.response.status_code != 404:  # 404: época que o site não tem
                    print(f"{league.key} {start}: falhou a recolha ({e})")
                continue
            total += save(engine, from_betexplorer(df, league, season_label(start, calendar_year=False)))
        return total
    if league.fd_extra:
        df = normalize(fetch_csv(csv_url(league)), league)
        first = str(today.year - n_seasons + 1)
        return save(engine, df[df["season"] >= first])

    # A época europeia começa em julho
    current = today.year if today.month >= 7 else today.year - 1
    total = 0
    for start in range(current - n_seasons + 1, current + 1):
        raw = fetch_csv(csv_url(league, start))
        total += save(engine, normalize(raw, league, season_label(start, calendar_year=False)))
    return total


def fetch_fixtures(leagues: list[League]) -> pd.DataFrame:
    """Próximos jogos (com odds atuais) das ligas pedidas.

    O site atualiza estes ficheiros ao longo da semana; os jogos de fim de
    semana costumam aparecer à quinta ou sexta.
    """
    frames = []
    main = [l for l in leagues if not l.fd_extra and l.fd_code]
    if main:
        df = fetch_csv(f"{BASE_URL}/fixtures.csv")
        df.columns = df.columns.str.replace("﻿", "").str.strip()
        codes = {l.fd_code: l.key for l in main}
        df = df[df["Div"].isin(codes)].assign(league=lambda d: d["Div"].map(codes))
        frames.append(df)
    extra = [l for l in leagues if l.fd_extra]
    if extra:
        df = fetch_csv(f"{BASE_URL}/new_league_fixtures.csv")
        df.columns = df.columns.str.replace("﻿", "").str.strip()
        countries = {l.fd_country: l.key for l in extra}
        df = df[df["Country"].isin(countries)].assign(league=lambda d: d["Country"].map(countries))
        frames.append(df.rename(columns={"Home": "HomeTeam", "Away": "AwayTeam"}))

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    cols = ["league", "date", "time", "home", "away", "odds_home", "odds_draw", "odds_away"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    avg = _pick(df, ODDS_PIN[1:] + ODDS_AVG[1:2])
    return pd.DataFrame({
        "league": df["league"],
        "date": pd.to_datetime(df["Date"], dayfirst=True, format="mixed").dt.date,
        "time": df.get("Time"),
        "home": df["HomeTeam"].str.strip(),
        "away": df["AwayTeam"].str.strip(),
        "odds_home": avg["h"], "odds_draw": avg["d"], "odds_away": avg["a"],
    })[cols].sort_values(["date", "time", "league"]).reset_index(drop=True)
