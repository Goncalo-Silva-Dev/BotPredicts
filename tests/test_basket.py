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
    assert df["round"].iloc[0] == "Round 3" and df["time"].iloc[0] == "16:00"


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
