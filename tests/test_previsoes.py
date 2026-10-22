from datetime import date

import pandas as pd

from analise import previsoes
from analise.ingest import football_data_uk
from analise.leagues import LEAGUES


def test_predict_gives_probabilities_and_skips_unknown_teams(league):
    fixtures = pd.DataFrame([
        {"league": "X", "date": date(2026, 1, 1), "time": "20:00", "home": "T0", "away": "T1",
         "odds_home": 2.0, "odds_draw": 3.5, "odds_away": 4.0},
        {"league": "X", "date": date(2026, 1, 1), "time": "20:00", "home": "T0", "away": "Nova",
         "odds_home": None, "odds_draw": None, "odds_away": None},
    ])
    res = previsoes.predict(league, fixtures, as_of=date(2025, 1, 1))
    first = res.iloc[0]
    assert abs(first["casa"] + first["empate"] + first["fora"] - 1) < 1e-9
    assert 0 < first["mais_2_5"] < 1 and 0 < first["ambas_marcam"] < 1
    assert first["resultados"].count("%") == 3
    assert abs(first["mercado_casa"] + first["mercado_empate"] + first["mercado_fora"] - 1) < 1e-9
    assert pd.isna(res.iloc[1]["casa"])


def test_fetch_fixtures_keeps_only_our_leagues(monkeypatch):
    main = pd.DataFrame({
        "Div": ["P1", "E1"], "Date": ["24/10/2026", "24/10/2026"], "Time": ["20:30", "15:00"],
        "HomeTeam": ["Benfica", "Leeds"], "AwayTeam": ["Porto", "Hull"],
        "PSH": [2.4, 1.8], "PSD": [3.3, 3.5], "PSA": [3.0, 4.5],
    })
    extra = pd.DataFrame({
        "Country": ["Brazil", "Japan"], "League": ["Serie A", "J1 League"],
        "Date": ["25/10/2026", "25/10/2026"], "Time": ["21:00", "06:00"],
        "Home": ["Flamengo", "Kashima"], "Away": ["Palmeiras", "Urawa"],
        "AvgH": [2.1, 2.0], "AvgD": [3.2, 3.3], "AvgA": [3.6, 3.7],
    })
    monkeypatch.setattr(football_data_uk, "fetch_csv",
                        lambda url: main if url.endswith("/fixtures.csv") else extra)
    res = football_data_uk.fetch_fixtures([LEAGUES["POR"], LEAGUES["BRA"]])
    assert res[["league", "home", "away"]].values.tolist() == [
        ["POR", "Benfica", "Porto"], ["BRA", "Flamengo", "Palmeiras"]]
    assert res["odds_home"].tolist() == [2.4, 2.1]
