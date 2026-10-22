import io
from datetime import date

import pandas as pd
from sqlalchemy import select

from analise.db import get_engine, matches
from analise.ingest import football_data_uk as fd
from analise.leagues import LEAGUES

MAIN_CSV = """Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HS,AS,HST,AST,AvgH,AvgD,AvgA,PSCH,PSCD,PSCA,AvgCH,AvgCD,AvgCA
P1,09/08/2025,20:15,Benfica ,Porto,2,1,H,14,9,6,3,2.10,3.40,3.60,2.05,3.50,3.80,2.08,3.45,3.70
P1,10/08/2025,18:00,Sporting,Braga,1,1,D,12,10,4,4,1.80,3.70,4.50,,,,,,
"""

EXTRA_CSV = """Country,League,Season,Date,Time,Home,Away,HG,AG,Res,PSCH,PSCD,PSCA,MaxCH,MaxCD,MaxCA,AvgCH,AvgCD,AvgCA
Brazil,Serie A,2024,13/04/2024,21:00,Flamengo,Palmeiras,2,0,H,2.20,3.30,3.40,2.30,3.40,3.60,2.15,3.25,3.35
Brazil,Serie A,2025,29/03/2025,16:00,Santos,Gremio,1,2,A,2.50,3.20,2.90,2.60,3.30,3.00,2.45,3.15,2.85
"""


def read(text):
    return pd.read_csv(io.StringIO(text))


def test_season_code():
    assert fd.season_code(2025) == "2526"
    assert fd.season_code(1999) == "9900"
    assert fd.csv_url(LEAGUES["POR"], 2025).endswith("/mmz4281/2526/P1.csv")
    assert fd.csv_url(LEAGUES["BRA"]).endswith("/new/BRA.csv")


def test_normalize_main_league_falls_back_to_opening_odds():
    df = fd.normalize(read(MAIN_CSV), LEAGUES["POR"], "2025-26")
    first, second = df.iloc[0], df.iloc[1]
    assert first["home"] == "Benfica"
    assert first["date"] == date(2025, 8, 9)
    assert first["odds_home_avg"] == 2.08  # fecho
    assert second["odds_home_avg"] == 1.80  # sem fecho -> abertura
    assert pd.isna(second["odds_home_pin"])
    assert set(df["season"]) == {"2025-26"}


def test_normalize_extra_league_uses_calendar_seasons():
    df = fd.normalize(read(EXTRA_CSV), LEAGUES["BRA"])
    assert list(df["season"]) == ["2024", "2025"]
    assert list(df["home_goals"]) == [2, 1]
    assert df.iloc[1]["odds_away_pin"] == 2.90


def test_save_is_idempotent(tmp_path):
    engine = get_engine(f"sqlite:///{tmp_path}/t.db")
    df = fd.normalize(read(MAIN_CSV), LEAGUES["POR"], "2025-26")
    fd.save(engine, df)
    fd.save(engine, df)
    with engine.connect() as conn:
        rows = conn.execute(select(matches)).all()
    assert len(rows) == 2
