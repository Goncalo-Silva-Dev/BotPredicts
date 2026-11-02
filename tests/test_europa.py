from datetime import date

import pandas as pd

from analise import europa
from analise.ingest import betexplorer, football_data_uk
from analise.leagues import LEAGUES

ROW = ('<tr><td class="h-text-left"><a href="/x/" class="in-match"><span><strong>{home}</strong></span>'
       ' - <span>{away}</span></a></td><td class="h-text-center"><a href="/x/">{score}</a></td>'
       '<td class="table-main__odds colored"><span><span><span data-odd="2.10"></span></span></span></td>'
       '<td class="table-main__odds" data-odd="3.53"></td><td class="table-main__odds" data-odd="3.58"></td>'
       '<td class="h-text-right h-text-no-wrap">{when}</td></tr>')


def test_parse_football_results_reads_odds_and_extra_time():
    page = (ROW.format(home="PSG", away="Inter", score="5:0", when="31.05.2025")
            + ROW.format(home="Inter", away="Barcelona",
                         score='4:3 <span title="After Extra Time">AET</span>', when="Yesterday"))
    df = betexplorer.parse_football_results(page, date(2026, 11, 2), "UCL")
    assert list(df["home_goals"]) == [5, 4]
    assert (df.loc[0, "odds_home"], df.loc[0, "odds_draw"], df.loc[0, "odds_away"]) == (2.10, 3.53, 3.58)
    assert pd.isna(df.loc[0, "extra"]) and df.loc[1, "extra"] == "AET"
    assert df.loc[1, "date"] == date(2026, 11, 1)
    # o prolongamento não é o resultado dos 90 minutos: não entra no histórico
    saved = football_data_uk.from_betexplorer(df, LEAGUES["UCL"], "2025-26")
    assert list(saved["home"]) == ["PSG"] and saved.iloc[0]["odds_home_avg"] == 2.10


def test_fixtures_keep_odds_when_shown():
    row = ('<tr><td class="table-main__datetime">{when}</td>\n<td class="h-text-left"><a href="/x/" '
           'class="in-match"><span>{home}</span> - <span>Stuttgart</span></a></td>{odds}</tr>')
    odds = "".join(f'<td class="table-main__odds"><button data-odd="{o}"></button></td>'
                   for o in ("2.18", "4.04", "2.91"))
    page = row.format(when="Tomorrow 18:45", home="Galatasaray", odds=odds) + row.format(
        when="&nbsp;", home="Benfica", odds="")
    fx = betexplorer.parse_betexplorer_fixtures(page, date(2026, 11, 2), "UCL")
    assert list(fx["odds_home"].fillna(0)) == [2.18, 0]
    assert list(fx["time"]) == ["17:45", "17:45"]


def test_team_map_is_conservative():
    domestic = {
        "ENG": {"Man City", "Arsenal"}, "NED": {"Sparta Rotterdam", "PSV Eindhoven"},
        "NOR": {"Bodo/Glimt"}, "ESP": {"Celta"}, "GRE": {"AEK"},
    }
    cups = ["Manchester City", "Arsenal", "Sparta Prague", "PSV", "Bodo/Glimt", "Celta Vigo",
            "AEK Athens", "AEK Larnaca"]
    m = europa.team_map(cups, domestic)
    assert m["Manchester City"] == ("ENG", "Man City") and m["Arsenal"] == ("ENG", "Arsenal")
    assert m["PSV"] == ("NED", "PSV Eindhoven") and m["Bodo/Glimt"] == ("NOR", "Bodo/Glimt")
    assert m["Celta Vigo"] == ("ESP", "Celta") and m["AEK Athens"] == ("GRE", "AEK")
    # nomes parecidos de clubes diferentes ficam por ligar
    assert "Sparta Prague" not in m and "AEK Larnaca" not in m


def test_training_history_links_cups_and_leagues():
    games = pd.DataFrame([
        {"league": "UCL", "date": date(2026, 10, 1), "home": "Manchester City", "away": "Sparta Prague",
         "home_goals": 3, "away_goals": 0},
        {"league": "ENG", "date": date(2026, 10, 4), "home": "Man City", "away": "Arsenal",
         "home_goals": 1, "away_goals": 1},
        {"league": "NED", "date": date(2026, 10, 4), "home": "Sparta Rotterdam", "away": "Ajax",
         "home_goals": 0, "away_goals": 2},
        {"league": "USA", "date": date(2026, 10, 4), "home": "Arsenal", "away": "X",
         "home_goals": 0, "away_goals": 2},
    ])
    h = europa.training_history(games)
    assert set(h["league"]) == {"UCL", "ENG", "NED"}
    eng = h[h["league"] == "ENG"].iloc[0]
    assert eng["home"] == "Manchester City" and eng["away"] == "ENG:Arsenal"
    assert h[h["league"] == "NED"].iloc[0]["home"] == "NED:Sparta Rotterdam"
    assert h[h["league"] == "UCL"].iloc[0]["weight"] == europa.CUP_WEIGHT
