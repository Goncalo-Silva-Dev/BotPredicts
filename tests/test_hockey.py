from datetime import date, timedelta

import numpy as np
import pandas as pd

from analise import hoquei, pagina
from analise.ingest.hockey import parse_espn_day
from analise.models.hockey import HockeyModel, end_of_game


def _season(seed=5):
    rng = np.random.default_rng(seed)
    teams = [f"H{i}" for i in range(10)]
    att, dfc = rng.normal(0, 0.15, 10), rng.normal(0, 0.15, 10)
    rows, d = [], date(2025, 10, 1)
    for _ in range(4):
        for i in range(10):
            for j in range(10):
                if i != j:
                    h = rng.poisson(np.exp(1.1 + 0.05 + att[i] + dfc[j]))
                    a = rng.poisson(np.exp(1.1 + att[j] + dfc[i]))
                    tie = h == a
                    rows.append({"league": "NHL", "date": d, "time": "00:00", "round": None,
                                 "home": teams[i], "away": teams[j], "home_reg": h, "away_reg": a,
                                 "home_goals": h + (tie and rng.random() < 0.5),
                                 "away_goals": a + 0, "ending": "OT" if tie else "REG"})
                    if tie and rows[-1]["home_goals"] == h:
                        rows[-1]["away_goals"] = a + 1
                    d += timedelta(days=1)
    return pd.DataFrame(rows), att - dfc


def test_end_of_game_keeps_probability_and_adds_ties():
    g = np.arange(8)
    m = np.outer(np.exp(-3) * 3.0 ** g / np.cumprod(np.r_[1, g[1:]]), np.exp(-2.8) * 2.8 ** g / np.cumprod(np.r_[1, g[1:]]))
    out = end_of_game(m, 0.3, 0.2, 0.38, 0.16)
    assert abs(out.sum() - m.sum()) < 1e-12
    assert np.trace(out) > np.trace(m)  # empates tardios


def test_model_probabilities_are_consistent():
    df, strength = _season()
    m = HockeyModel(half_life_days=10_000).fit(df)
    strong, weak = f"H{int(np.argmax(strength))}", f"H{int(np.argmin(strength))}"
    p, mk = m.predict(strong, weak), m.markets(strong, weak)
    assert abs(p["home"] + p["away"] - 1) < 1e-9
    assert abs(p["reg_home"] + p["reg_draw"] + p["reg_away"] - 1) < 1e-9
    assert p["home"] > m.predict(weak, strong)["home"]
    assert abs(mk["prolongamento"] - p["reg_draw"]) < 1e-12
    pl = {(s, l): v for s, l, v in mk["puck_line"]}
    assert abs(pl[("casa", -1.5)] + pl[("fora", 1.5)] - 1) < 1e-9
    assert pl[("casa", -1.5)] < p["reg_home"] and pl[("casa", -2.5)] < pl[("casa", -1.5)]
    totals = [v for _, v in mk["total"]]
    assert all(a > b for a, b in zip(totals, totals[1:]))
    periods, most = m.periods(strong, weak, n=5000)
    assert len(periods) == 3 and abs(sum(most.values()) - 1) < 1e-9
    assert all(abs(x["casa"] + x["empate"] + x["fora"] - 1) < 1e-9 for x in periods)


def test_predict_upcoming_and_page():
    df, _ = _season()
    last = df["date"].max()
    future = pd.DataFrame([{"league": "NHL", "date": last + timedelta(days=2), "time": "00:00",
                            "round": None, "home": "H1", "away": "H2"}])
    res = hoquei.predict_upcoming(pd.concat([df, future], ignore_index=True), last + timedelta(days=1))
    assert len(res) == 1 and 0 < res.iloc[0]["casa"] < 1
    site = pagina.render_site(pd.DataFrame(), hockey=res)
    page = site["hoquei/nhl.html"]
    assert "H1 – H2" in page and "Puck line" in page and "3.º período" in page
    assert "🏒 Hóquei" in site["index.html"]


def test_parse_espn_regulation_overtime_and_shootout():
    def ev(detail, period, home, away, lines_h, lines_a, season=2, done=True):
        return {"date": "2026-10-24T23:00Z", "season": {"type": season},
                "status": {"period": period, "type": {"completed": done, "detail": detail}},
                "competitions": [{"competitors": [
                    {"homeAway": "home", "score": str(home), "team": {"displayName": "Casa"},
                     "linescores": [{"value": v} for v in lines_h]},
                    {"homeAway": "away", "score": str(away), "team": {"displayName": "Fora"},
                     "linescores": [{"value": v} for v in lines_a]}]}]}
    rows = parse_espn_day([
        ev("Final", 3, 4, 2, [1, 2, 1], [0, 1, 1]),
        ev("Final/OT", 4, 3, 2, [1, 1, 0, 1], [0, 2, 0, 0]),
        ev("Final/SO", 5, 2, 3, [1, 0, 1, 0, 0], [1, 1, 0, 0, 1]),
        ev("Final", 3, 1, 0, [1, 0, 0], [0, 0, 0], season=1),  # pré-época fica de fora
        ev("Scheduled", 0, 0, 0, [], [], done=False),
    ], "NHL")
    assert [r["ending"] for r in rows[:3]] == ["REG", "OT", "SO"]
    assert (rows[1]["home_reg"], rows[1]["away_reg"]) == (2, 2)
    assert (rows[2]["home_reg"], rows[2]["away_reg"], rows[2]["away_goals"]) == (2, 2, 3)
    assert rows[0]["date"] == date(2026, 10, 25) and rows[0]["time"] == "00:00"  # hora de Portugal
    assert len(rows) == 4 and rows[3]["home_goals"] is None
