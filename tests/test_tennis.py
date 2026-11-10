from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd

from analise import pagina, registo, tenis
from analise.ingest.tennis import parse_year, surface, to_frame
from analise.models.tennis import (TennisElo, game_prob, markets, match_distribution,
                                   mixed_distribution, predict_distribution, tiebreak_prob)


def test_point_model_is_consistent():
    assert abs(game_prob(0.5) - 0.5) < 1e-12
    assert game_prob(0.65) > 0.65
    assert abs(tiebreak_prob(0.6, 0.4) - 0.5) < 1e-9  # iguais: 50%
    for best_of in (3, 5):
        d = match_distribution(0.62, 0.62, best_of)
        assert abs(d["games"].sum() - 1) < 1e-9 and abs(sum(d["sets"].values()) - 1) < 1e-9
        assert abs(d["win"] - 0.5) < 1e-9 and abs(d["first_set"] - 0.5) < 1e-9
    bo3, bo5 = match_distribution(0.65, 0.61, 3), match_distribution(0.65, 0.61, 5)
    assert 0.5 < bo3["win"] < bo5["win"]  # à melhor de 5 o melhor jogador ganha mais vezes
    mix = mixed_distribution(0.6, 0.04, 0.1, 3)
    assert abs(mix["games"].sum() - 1) < 1e-9


def test_distribution_keeps_elo_probability_and_markets_add_up():
    for p in (0.3, 0.5, 0.72, 0.9):
        d = predict_distribution(p, "ATP", "terra", 3)
        assert abs(d["win"] - p) < 0.01
    d5 = predict_distribution(0.7, "ATP", "relva", 5)
    assert d5["win"] > 0.7
    mk = markets(predict_distribution(0.7, "WTA", "rapido", 3), 3)
    assert abs(sum(p for _, p in mk["sets"]) - 1) < 1e-9
    over = [p for _, p in mk["total"]]
    assert all(a >= b for a, b in zip(over, over[1:]))
    hcp = [p for _, p in mk["handicap"]]
    assert all(a <= b for a, b in zip(hcp, hcp[1:]))
    assert mk["casa_ganha_set"] > mk["fora_ganha_set"]


def test_surface_from_tournament_and_city():
    assert surface("Wimbledon", "London, Great Britain", 7) == "relva"
    assert surface("Roland Garros", "Paris, France", 5) == "terra"
    assert surface("Rolex Paris Masters", "Paris, France", 10) == "rapido"
    assert surface("Porsche Tennis Grand Prix", "Stuttgart, Germany", 4) == "terra"
    assert surface("Boss Open", "Stuttgart, Germany", 6) == "relva"
    assert surface("Mutua Madrid Open", "Madrid, Spain", 4) == "terra"
    assert surface("Australian Open", "Melbourne, Australia", 1) == "rapido"


def _competition(cid, a, b, status="STATUS_FINAL", sets=None, rnd="Round 1", when="2026-05-26T10:00Z"):
    comps = []
    for order, (pid, name) in enumerate((a, b), start=1):
        c = {"id": pid, "type": "athlete", "order": order, "athlete": {"displayName": name},
             "winner": order == 1 and status == "STATUS_FINAL"}
        if sets:
            c["linescores"] = [{"value": s[order - 1], "winner": s[order - 1] > s[2 - order]} for s in sets]
        comps.append(c)
    state = "post" if status != "STATUS_SCHEDULED" else "pre"
    return {"id": cid, "date": when, "timeValid": True, "venue": {"fullName": "Paris, France"},
            "status": {"type": {"name": status, "state": state}}, "round": {"displayName": rnd},
            "competitors": comps}


def test_parse_year_reads_sets_and_skips_unknown_players():
    events = [{"name": "Roland Garros", "date": "2026-05-24T00:00Z", "groupings": [
        {"grouping": {"slug": "mens-singles"}, "competitions": [
            _competition("1", ("10", "A"), ("11", "B"), sets=[(6, 4), (3, 6), (7, 6), (6, 2)]),
            _competition("2", ("-4", "TBD"), ("11", "B"), status="STATUS_SCHEDULED", rnd="Final")]},
        {"grouping": {"slug": "mens-doubles"}, "competitions": [
            _competition("3", ("12", "C"), ("13", "D"), sets=[(6, 4), (6, 4)])]},
        {"grouping": {"slug": "womens-singles"}, "competitions": [
            _competition("4", ("20", "E"), ("21", "F"), status="STATUS_SCHEDULED", rnd="Qualifying Final")]},
    ]}]
    df = to_frame(parse_year(events))
    assert list(df["id"]) == ["1", "4"]
    m = df.iloc[0]
    assert m.league == "ATP" and m.best_of == 5 and m.surface == "terra"
    assert (m.home_sets, m.away_sets, m.winner) == (3, 1, 0)
    assert m.home_s3 == 7 and m.away_s3 == 6
    w = df.iloc[1]
    assert w.league == "WTA" and w.best_of == 3 and w.status == "por jogar"
    assert w["round"] == "Qualificação, ronda final"


def _matches(n_days=200, seed=3):
    """Jogadores com forças diferentes; jogos à melhor de 3 em piso rápido."""
    rng = np.random.default_rng(seed)
    skill = {f"P{i}": rng.normal(0, 0.03) for i in range(12)}
    rows, start = [], date(2025, 1, 1)
    for k in range(n_days * 3):
        a, b = rng.choice(list(skill), 2, replace=False)
        pa, pb = 0.62 + skill[a] - skill[b], 0.62 + skill[b] - skill[a]
        d = match_distribution(round(pa, 3), round(pb, 3), 3)
        win = rng.random() < d["win"]
        sets = [(6, 4), (6, 3)] if win else [(4, 6), (3, 6)]
        rows.append({"id": str(k), "league": "ATP", "date": start + timedelta(days=k // 3),
                     "time": "12:00", "tournament": "Torneio", "round": "1.ª ronda",
                     "surface": "rapido", "best_of": 3, "home": a, "away": b, "home_id": a,
                     "away_id": b, "status": "final", "home_sets": 2.0 if win else 0.0,
                     "away_sets": 0.0 if win else 2.0, "home_s1": sets[0][0], "away_s1": sets[0][1],
                     "home_s2": sets[1][0], "away_s2": sets[1][1], "winner": 0 if win else 1})
    df = pd.DataFrame(rows)
    for k in range(3, 6):
        df[f"home_s{k}"] = np.nan
        df[f"away_s{k}"] = np.nan
    return df, skill


def test_elo_learns_who_is_better():
    df, skill = _matches()
    elo = tenis.fit(df)
    best, worst = max(skill, key=skill.get), min(skill, key=skill.get)
    assert elo.predict(best, worst, "rapido") > 0.6
    e = TennisElo()
    e.update("x", "y", "terra", True)
    assert e.strength("x", "terra") > e.strength("y", "terra")


def test_walk_forward_predict_page_and_track():
    df, skill = _matches(120)
    res = tenis.walk_forward(df, start="2025-03-01")
    s = tenis.summary(res)
    assert s["jogos"] > 100 and 0 < s["logloss"] < 0.75
    assert not tenis.calibration(res).empty

    today = date(2025, 6, 1)
    upcoming = df.iloc[:2].copy()
    upcoming["id"] = ["u1", "u2"]
    upcoming["date"] = [today, today + timedelta(days=1)]
    upcoming["status"] = "por jogar"
    upcoming[["home_sets", "away_sets", "winner", "home_s1", "away_s1", "home_s2", "away_s2"]] = np.nan
    all_matches = pd.concat([df, upcoming], ignore_index=True)
    preds = tenis.predict_upcoming(all_matches, today)
    assert len(preds) == 2 and ((preds["casa"] + preds["fora"]).sub(1).abs() < 1e-9).all()

    pages = pagina.render_site(pd.DataFrame(), tennis=preds)
    assert "tenis/atp.html" in pages and "tenis/wta.html" in pages
    html = pages["tenis/atp.html"]
    assert "Mais apostas" in html and "Total de jogos" in html and "Resultado em sets" in html
    assert "🎾 Ténis" in pages["index.html"]

    now = datetime(2025, 5, 31, 12, tzinfo=timezone.utc)
    store = {}
    assert registo.record(store, preds, "tenis", now) == 2
    first = preds.iloc[0]
    result = pd.DataFrame([{"league": "ATP", "date": today, "home": first.home, "away": first.away,
                            "status": "final", "home_sets": 2.0, "away_sets": 1.0,
                            "home_s1": 6, "away_s1": 7, "home_s2": 6, "away_s2": 2, "home_s3": 6,
                            "away_s3": 4, "home_s4": np.nan, "away_s4": np.nan, "home_s5": np.nan,
                            "away_s5": np.nan}])
    assert registo.resolve(store, result, "tenis", now + timedelta(days=2)) == 1
    rec = next(r for r in store.values() if r["estado"] == "terminado")
    assert rec["resultado"] == {"casa": 2, "fora": 1, "jogos_casa": 18, "jogos_fora": 13,
                                "set1": 1, "tiebreaks": 1}
    ev = registo.evaluate(store)
    assert set(ev["mercado"]) >= {"Vencedor", "Vencedor do 1.º set", "Há tie-break", "Vai a todos os sets"}
    assert "🎾 Ténis" in pagina.render_site(pd.DataFrame(), track=store)["historico.html"]
