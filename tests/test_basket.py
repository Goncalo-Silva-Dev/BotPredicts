from datetime import date, timedelta

import numpy as np
import pandas as pd

from analise.models.basket import BasketRatings


def _season(seed=3):
    rng = np.random.default_rng(seed)
    teams = [f"B{i}" for i in range(12)]
    att, dfc = rng.normal(0, 4, 12), rng.normal(0, 4, 12)
    rows, d = [], date(2025, 10, 20)
    for _ in range(3):
        for i in range(12):
            for j in range(12):
                if i != j:
                    rows.append({
                        "date": d, "home": teams[i], "away": teams[j],
                        "home_points": 112 + 3 + att[i] - dfc[j] + rng.normal(0, 10),
                        "away_points": 112 + att[j] - dfc[i] + rng.normal(0, 10)})
                    d += timedelta(days=1)
    return pd.DataFrame(rows), att, dfc


def test_recovers_strengths_and_home_advantage():
    df, att, dfc = _season()
    m = BasketRatings(half_life_days=10_000, ridge=0.1).fit(df)
    assert abs(m.home_ - 3) < 1.5
    assert abs(m.mean_ - 112) < 2
    order = [int(t[1:]) for t in m.teams_]
    assert np.corrcoef(m.attack_, att[order])[0, 1] > 0.9
    assert np.corrcoef(m.defence_, dfc[order])[0, 1] > 0.9
    assert 12 < m.sd_margin_ < 16  # duas pontuações com ruído 10 -> ~14


def test_predict_is_consistent():
    df, att, dfc = _season()
    m = BasketRatings().fit(df)
    strong, weak = f"B{int(np.argmax(att - dfc))}", f"B{int(np.argmin(att - dfc))}"
    p = m.predict(strong, weak)
    assert p["home"] > 0.75 and abs(p["home"] + p["away"] - 1) < 1e-12
    assert p["margin"] > 0 and p["total_low"] < p["total"] < p["total_high"]
    assert m.predict(weak, strong)["home"] < p["home"]


NBB_ROW = """<tr class="with-hotel">
<td class="position_value show-for-medium" data-real-id="{id}">1</td>
<td class="date_value show-for-medium"><span class="">{d}</span>
<span class="">16:00</span></td>
<td class="home_team_value"><span class="team-shortname">{home}</span></td>
<td class="score_value show-for-medium">{score}</td>
<td class="visitor_team_value"><span class="team-shortname">{away}</span></td>
<td class="hide-for-medium matche_for_small"><strong class="x"> {hc} <img alt=""></strong>
<a>x</a><strong class="x"><img alt=""> {ac} </strong></td>
<td class="game_value"><span class="number">3ª</span> RODADA</td>
</tr>"""


def test_parse_nbb_and_unify_sponsor_names():
    from analise.ingest.basket import _unify_names, parse_nbb
    played = '<a><span class="home">84</span> X <span class="away">73</span></a>'
    html = "".join([
        NBB_ROW.format(id=1, d="17/10/2025", home="Mogi", away="Corinthians", score=played,
                       hc="MOG", ac="COR"),
        NBB_ROW.format(id=2, d="20/10/2026", home="Desk Manager Mogi Basquete",
                       away="Corinthians", score="", hc="MOG", ac="COR"),
    ])
    df = _unify_names(parse_nbb(html))
    assert df["home"].tolist() == ["Desk Manager Mogi Basquete"] * 2
    assert df["home_points"].iloc[0] == 84 and pd.isna(df["home_points"].iloc[1])
    assert df["round"].iloc[0] == "Round 3" and df["time"].iloc[0] == "20:00"


def test_parse_lnb_home_team_first_and_unify():
    from analise.ingest.basket import _unify_names, parse_lnb

    def team(name, club, score):
        return {"team_name": name, "club_external_id": club, "score_string": score}

    days = [
        {"date": "2025-09-26", "data": [{
            "match_time_utc": "2025-09-26T18:30:00.000Z", "match_status": "COMPLETE",
            "round_number": 1, "teams": [team("JDA Dijon", 27, "89"), team("Le Mans", 38, "91")]}]},
        {"date": "2026-10-23", "data": [{
            "match_time_utc": "2026-10-23T18:00:00.000Z", "match_status": "SCHEDULED",
            "round_number": 5, "teams": [team("Le Mans", 38, ""), team("Dijon", 27, "")]}]},
    ]
    df = _unify_names(parse_lnb(days))
    assert df[["home", "away"]].values.tolist() == [["Dijon", "Le Mans"], ["Le Mans", "Dijon"]]
    assert df["home_points"].iloc[0] == 89 and pd.isna(df["home_points"].iloc[1])
    assert df["time"].iloc[0] == "19:30" and df["round"].iloc[1] == "Round 5"


def test_periods_are_consistent_with_the_game():
    df, att, dfc = _season()
    m = BasketRatings().fit(df)
    strong, weak = f"B{int(np.argmax(att - dfc))}", f"B{int(np.argmin(att - dfc))}"
    periods = {p["periodo"]: p for p in m.predict_periods(strong, weak)}
    assert len(periods) == 6
    for p in periods.values():
        assert abs(p["casa"] + p["empate"] + p["fora"] - 1) < 1e-12 and 0 < p["empate"] < 0.15
        assert p["casa"] > p["fora"] and p["total_min"] < p["total"] < p["total_max"]
    game = m.predict(strong, weak)
    assert periods["1.º quarto"]["casa"] < periods["1.ª parte"]["casa"] < game["home"]
    assert abs(periods["1.ª parte"]["total"] + periods["2.ª parte"]["total"] - game["total"]) < 0.01 * game["total"]


def test_markets_are_consistent_with_the_game():
    df, att, dfc = _season()
    m = BasketRatings().fit(df)
    strong, weak = f"B{int(np.argmax(att - dfc))}", f"B{int(np.argmin(att - dfc))}"
    mk, game = m.markets(strong, weak), m.predict(strong, weak)
    for key in ("handicap", "total", "pontos_casa", "pontos_fora"):
        lines = [line for line, _ in mk[key]]
        probs = [p for _, p in mk[key]]
        assert all(line % 1 == 0.5 for line in lines) and lines == sorted(lines)
        assert all(a > b for a, b in zip(probs, probs[1:]))  # linha mais alta, menos provável
    # "casa -0,5" no handicap é ganhar o jogo
    assert abs(dict(mk["handicap"]).get(-0.5, game["home"]) - game["home"]) < 0.02
    assert abs(sum(p for *_, p in mk["margem"]) - 1) < 1e-9
    home_bands = sum(p for side, _, p in mk["margem"] if side == "casa")
    assert abs(home_bands - game["home"]) < 0.02
    htft = mk["intervalo_final"]
    assert abs(sum(htft.values()) - 1) < 1e-9 and htft["1/1"] == max(htft.values())


def test_reserve_games_roundtrip(tmp_path):
    from analise.cli import _reserve_games
    df = pd.DataFrame([{"league": "LNB", "date": date(2026, 10, 23), "time": "19:00",
                        "round": "Round 5", "home": "Le Mans", "away": "Dijon",
                        "home_points": None, "away_points": None}])
    df.to_csv(tmp_path / "LNB.csv", index=False)
    back = _reserve_games(str(tmp_path), "LNB")
    assert back["date"].iloc[0] == date(2026, 10, 23) and back["time"].iloc[0] == "19:00"
    assert _reserve_games(str(tmp_path), "NBB") is None and _reserve_games(None, "LNB") is None


BLG_GAME = """<li class="list-item" id="{id}"><div class="inner"><a class="data-game" href="#">
<div class="game "><span class="team home"><span class="team-logo"><img src="https://x/logo/2026/s/{hc}.png"></span>
<span class="team-name">{hn}</span></span><span class="point font-blg">{score}</span>
<span class="team away"><span class="team-name">{an}</span><span class="team-logo"><img src="https://x/logo/2026/s/{ac}.png"></span></span></div>
<div class="info"><div class="info-arena"><span>滋賀県</span><span>19:05</span></div></div></a>
<div class="data-link"></div></div></li>"""


def test_parse_bleague_names_scores_and_skips_all_star():
    from analise.ingest.basket import parse_bleague
    score = ('<span class="number home-score "><span>81</span></span><span class="hifun"></span>'
             '<span class="number away-score gr"><span>52</span></span>')
    html = (BLG_GAME.format(id=1, hc="cj", hn="千葉J", ac="kh", an="京都", score=score)
            + BLG_GAME.format(id=2, hc="ls", hn="滋賀", ac="zz", an="新", score="")
            + BLG_GAME.format(id=3, hc="a1", hn="B.BLACK", ac="a2", an="B.WHITE", score=score))
    df = parse_bleague(html, date(2026, 10, 23))
    assert df[["home", "away"]].values.tolist() == [["Chiba Jets", "Kyoto Hannaryz"], ["Shiga Lakes", "新"]]
    assert df["home_points"].iloc[0] == 81 and pd.isna(df["home_points"].iloc[1])
    assert df["time"].iloc[0] == "11:05" and str(df["date"].iloc[0]) == "2026-10-23"


def test_parse_kbl_keeps_first_division_league_games():
    from analise.ingest.basket import parse_kbl
    base = {"seasonGrade": 1, "gameCode": "01", "gameDate": "20251003", "gameStart": "1400",
            "tcodeH": "50", "tnameH": "창원 LG", "tcodeA": "55", "tnameA": "서울 SK",
            "scoreH": 81, "scoreA": 89, "isEnded": 1}
    games = [base, {**base, "seasonGrade": 2}, {**base, "gameCode": "07"},
             {**base, "gameDate": "20261024", "isEnded": 0, "scoreH": 0, "scoreA": 0}]
    df = parse_kbl(games)
    assert len(df) == 2 and df["home"].iloc[0] == "Changwon LG Sakers"
    assert df["time"].iloc[0] == "06:00" and df["home_points"].iloc[0] == 81
    assert pd.isna(df["home_points"].iloc[1])


def test_parse_sbl_utc_to_lisbon_and_unplayed():
    from analise.ingest.basket import parse_sbl

    def team(name, score):
        return {"names": {"long": name}, "score": score}

    games = [
        {"rawStartDateTime": "2025-09-18T17:04:00.000Z", "state": "post-game",
         "homeTeamInfo": team("Uppsala Basket", 90), "awayTeamInfo": team("Sloga Uppsala", 84)},
        {"rawStartDateTime": "2027-01-10T15:00:00.000Z", "state": "pre-game",
         "homeTeamInfo": team("BC Luleå", None), "awayTeamInfo": team("Borås Basket", None)},
    ]
    df = parse_sbl(games)
    assert list(df["time"]) == ["18:04", "15:00"]  # hora de verão e de inverno em Lisboa
    assert df.loc[0, "home_points"] == 90 and pd.isna(df.loc[1, "home_points"])


def test_parse_kori_skips_series_summaries():
    from analise.ingest.basket import parse_kori

    base = {"status": "Played", "team_A_name": "Kobrat", "team_B_name": "Tapiolan Honka"}
    matches = [
        {**base, "match_type": "single", "date": "2025-10-03", "time": "18:30:00", "fs_A": "74", "fs_B": "80"},
        {**base, "match_type": "series", "date": "", "time": "", "fs_A": "4", "fs_B": "1"},
        {**base, "match_type": "single", "date": "2027-01-15", "time": "18:30:00", "status": "Fixture",
         "fs_A": "", "fs_B": ""},
    ]
    df = parse_kori(matches)
    assert len(df) == 2 and list(df["time"]) == ["16:30", "16:30"]  # Helsínquia está 2 h à frente
    assert df.loc[0, "away_points"] == 80 and pd.isna(df.loc[1, "home_points"])


def test_parse_cba_maps_club_ids_and_skips_preseason():
    from analise.ingest.basket import parse_cba

    game = {"ScheduleTypeID": 1, "Status": 4, "Round": 1, "dates": "2026-10-16", "time": "19:35",
            "HomeTeamID": 29125, "HomeTeamName": "上海久事", "HomeTeamScore": 102,
            "VisitingTeamID": 29131, "VisitingTeamName": "深圳马可波罗", "VisitingTeamScore": 68}
    df = parse_cba([game, {**game, "ScheduleTypeID": 3}])
    assert len(df) == 1
    assert (df.loc[0, "home"], df.loc[0, "away"], df.loc[0, "time"]) == ("Shanghai", "Shenzhen", "12:35")


def test_parse_betexplorer_results_and_fixtures():
    from analise.ingest.betexplorer import parse_betexplorer, parse_betexplorer_fixtures

    result = ('<td class="h-text-left"><a href="/x/" class="in-match"><span><strong>Zhejiang Guangsha'
              '</strong></span> - <span>Beijing</span></a></td><td class="h-text-center"><a href="/x/">'
              '117:104 <span title="After Extra Time">AET</span></a></td><td class="table-main__odds">'
              '</td><td class="h-text-right h-text-no-wrap">{when}</td>')
    page = result.format(when="19.04.") + result.format(when="12.12.2025") + result.format(when="Yesterday")
    df = parse_betexplorer(page, date(2026, 10, 23), "CBA")
    assert list(df["date"]) == [date(2026, 4, 19), date(2025, 12, 12), date(2026, 10, 22)]
    assert df.loc[0, "home"] == "Zhejiang Guangsha" and df.loc[0, "home_points"] == 117

    row = ('<td class="table-main__datetime">{when}</td> <td class="h-text-left"><a href="/x/" '
           'class="in-match"><span>Liaoning</span> - <span>Sichuan</span></a></td>')
    page = row.format(when="04.01.2027 12:35") + row.format(when="&nbsp;") + row.format(when="30.10. 12:35")
    fx = parse_betexplorer_fixtures(page, date(2026, 10, 23), "CBA")
    # UTC+1 fixo: no inverno são 11:35 em Lisboa, no verão 12:35
    assert list(fx["time"]) == ["11:35", "11:35", "11:35"]
    assert list(fx["date"]) == [date(2027, 1, 4), date(2027, 1, 4), date(2026, 10, 30)]
