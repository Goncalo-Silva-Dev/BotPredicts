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
    assert abs(first["p1_casa"] + first["p1_empate"] + first["p1_fora"] - 1) < 1e-9
    assert first["p1_empate"] > first["empate"] and first["p1_golos"] < first["p2_golos"]
    assert abs(first["dupla_1x"] - (first["casa"] + first["empate"])) < 1e-9
    assert abs(first["mais_2.5"] - first["mais_2_5"]) < 1e-9
    assert abs(sum(first[f"golos_{n}"] for n in ["0", "1", "2", "3", "4+"]) - 1) < 1e-9
    assert abs(sum(first["intervalo_final"].values()) - 1) < 1e-6
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


def test_render_page(league):
    from analise import pagina
    fixtures = pd.DataFrame([
        {"league": "POR", "date": date(2026, 1, 1), "time": "20:00", "home": "T0", "away": "T1",
         "odds_home": 2.0, "odds_draw": 3.5, "odds_away": 4.0},
        {"league": "POR", "date": date(2026, 1, 1), "time": None, "home": "T2", "away": "Nova",
         "odds_home": None, "odds_draw": None, "odds_away": None},
    ])
    res = previsoes.predict(league, fixtures, as_of=date(2025, 1, 1))
    site = pagina.render_site(res)
    out = site["futebol/por.html"]
    assert "T0 – T1" in out and "Liga Portugal" in out and "Sem previsão" in out
    assert "Por parte" in out and "Por período" not in out and "1.ª parte" in out
    assert "Mais apostas" in out and "Intervalo / final" in out and "Handicap -1 casa" in out
    assert "Antes de usares estas previsões" in site["index.html"]
    assert 'href="futebol/por.html"' in site["index.html"]
    assert 'href="../basquetebol/nba.html"' in out
    assert "Sem jogos marcados" in pagina.render_site(pd.DataFrame())["futebol/eng.html"]
    assert '"home": "T0"' in pagina.to_json(res)


def test_match_teams_links_official_names():
    from analise.ingest.openfootball import match_teams
    of = ["Sport Lisboa e Benfica", "Sporting Clube de Portugal", "Sporting Clube de Braga",
          "FC Porto", "Vitória Guimarães", "CF Estrela da Amadora", "Equipa Nova FC"]
    fd = ["Benfica", "Sp Lisbon", "Sp Braga", "Porto", "Guimaraes", "Estrela"]
    assert match_teams(of, fd) == {
        "Sport Lisboa e Benfica": "Benfica", "Sporting Clube de Portugal": "Sp Lisbon",
        "Sporting Clube de Braga": "Sp Braga", "FC Porto": "Porto",
        "Vitória Guimarães": "Guimaraes", "CF Estrela da Amadora": "Estrela"}


def test_upcoming_joins_calendar_and_odds(monkeypatch):
    from analise import calendario
    from analise.db import get_engine, metadata
    from analise.ingest import openfootball

    engine = get_engine("sqlite://")
    metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(calendario.matches.insert(), [
            {"league": "POR", "season": "2026-27", "date": date(2026, 9, 1), "home": h, "away": a,
             "source": "t"} for h, a in [("Benfica", "Porto"), ("Sp Lisbon", "Benfica")]])
    season = pd.DataFrame([
        {"league": "POR", "round": "Matchday 9", "date": date(2026, 10, 25), "time": None,
         "home": "Sport Lisboa e Benfica", "away": "FC Porto", "played": False},
        {"league": "POR", "round": "Matchday 10", "date": date(2026, 11, 1), "time": None,
         "home": "Sporting Clube de Portugal", "away": "Sport Lisboa e Benfica", "played": False},
        {"league": "POR", "round": "Matchday 1", "date": date(2026, 8, 8), "time": None,
         "home": "FC Porto", "away": "Sport Lisboa e Benfica", "played": True},
    ])
    fd = pd.DataFrame([
        {"league": "POR", "date": date(2026, 10, 25), "time": "20:30", "home": "Benfica",
         "away": "Porto", "odds_home": 2.2, "odds_draw": 3.3, "odds_away": 3.4},
        {"league": "USA", "date": date(2026, 10, 24), "time": "23:00", "home": "A", "away": "B",
         "odds_home": 2.0, "odds_draw": 3.5, "odds_away": 3.5},
    ])
    monkeypatch.setattr(openfootball, "fetch_season", lambda league, today: season)
    monkeypatch.setattr(calendario.football_data_uk, "fetch_fixtures", lambda leagues: fd)
    res = calendario.upcoming(engine, [LEAGUES["POR"], LEAGUES["USA"]], today=date(2026, 10, 22))
    assert res[["league", "home", "away"]].values.tolist() == [
        ["USA", "A", "B"], ["POR", "Benfica", "Porto"], ["POR", "Sp Lisbon", "Benfica"]]
    benfica = res.iloc[1]
    assert benfica["time"] == "20:30" and benfica["odds_home"] == 2.2
    assert benfica["round"] == "Matchday 9"
    assert pd.isna(res.iloc[2]["odds_home"])


def test_openfootball_local_times_become_portugal_time():
    from analise.calendario import _to_lisbon
    games = pd.DataFrame([
        {"date": date(2026, 10, 24), "time": "20:45"},
        {"date": date(2026, 10, 24), "time": None},
    ])
    rome = _to_lisbon(games, "Europe/Rome")
    assert rome["time"].tolist()[0] == "19:45" and rome["time"].isna().tolist()[1]
    late = _to_lisbon(pd.DataFrame([{"date": date(2026, 11, 4), "time": "21:30"}]), "America/Sao_Paulo")
    assert late.iloc[0]["date"] == date(2026, 11, 5) and late.iloc[0]["time"] == "00:30"
