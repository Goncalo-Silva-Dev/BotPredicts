from datetime import date, datetime, timedelta, timezone

import pandas as pd

from analise import basquete, hoquei, pagina, previsoes, registo
from tests.test_basket import _season as basket_season
from tests.test_hockey import _season as hockey_season

NOW = datetime(2026, 10, 27, 6, 0, tzinfo=timezone.utc)


def _football(league, day, clock="20:00", odds=True):
    fixtures = pd.DataFrame([
        {"league": "POR", "date": day, "time": clock, "home": "T0", "away": "T1",
         "odds_home": 2.0 if odds else None, "odds_draw": 3.5 if odds else None,
         "odds_away": 4.0 if odds else None}])
    return previsoes.predict(league, fixtures, as_of=date(2025, 1, 1))


def test_kickoff_is_lisbon_time():
    assert registo.kickoff(date(2026, 7, 1), "20:00") == datetime(2026, 7, 1, 19, 0, tzinfo=timezone.utc)
    assert registo.kickoff(date(2026, 12, 1), "20:00") == datetime(2026, 12, 1, 20, 0, tzinfo=timezone.utc)
    assert registo.kickoff(date(2026, 12, 1), None) == datetime(2026, 12, 1, 0, 0, tzinfo=timezone.utc)


def test_prediction_freezes_at_kickoff(league):
    store = {}
    assert registo.record(store, _football(league, date(2026, 10, 27)), "futebol", NOW) == 1
    (rec,) = store.values()
    first = rec["p"]["1x2"]
    assert rec["feita_em"] == "2026-10-27T06:00Z" and rec["estado"] == "pendente"
    # mais tarde, antes do jogo, a previsão é substituída pela mais recente
    changed = _football(league, date(2026, 10, 27)).assign(casa=0.9, empate=0.05, fora=0.05)
    registo.record(store, changed, "futebol", NOW + timedelta(hours=10))
    assert rec["id"] in store and store[rec["id"]]["p"]["1x2"] == [0.9, 0.05, 0.05]
    # depois do início (20:00 em Lisboa = 20:00 UTC no inverno) já não muda
    registo.record(store, changed.assign(casa=0.1), "futebol", NOW + timedelta(hours=15))
    assert store[rec["id"]]["p"]["1x2"][0] == 0.9 and first[0] != 0.9


def test_skips_started_far_and_untimed_same_day(league):
    store = {}
    registo.record(store, _football(league, date(2026, 10, 27), "05:00"), "futebol", NOW)
    registo.record(store, _football(league, date(2026, 11, 20)), "futebol", NOW)
    registo.record(store, _football(league, date(2026, 10, 27), None), "futebol", NOW)
    assert store == {}
    registo.record(store, _football(league, date(2026, 10, 28), None), "futebol", NOW)
    assert len(store) == 1


def test_resolve_evaluate_and_page(league, tmp_path):
    store = {}
    registo.record(store, _football(league, date(2026, 10, 27)), "futebol", NOW)
    registo.record(store, _football(league, date(2026, 10, 28), odds=False).assign(home="T2", away="T3"),
                   "futebol", NOW)
    results = pd.DataFrame([
        {"league": "POR", "date": date(2026, 10, 27), "home": "T0", "away": "T1", "home_goals": 2,
         "away_goals": 1, "home_goals_ht": 0, "away_goals_ht": 0},
        {"league": "POR", "date": date(2026, 10, 29), "home": "T2", "away": "T3", "home_goals": 1,
         "away_goals": 1, "home_goals_ht": None, "away_goals_ht": None}])  # 1 dia de diferença
    assert registo.resolve(store, results, "futebol", NOW + timedelta(days=3)) == 2
    ev = registo.evaluate(store)
    one = ev[ev["mercado"] == "Vencedor (1X2)"]
    assert len(one) == 2 and len(ev[ev["chave"] == "mercado_1x2"]) == 1
    # empate anula aposta com empate não conta; sem intervalo também não
    assert len(ev[ev["chave"] == "dnb_casa"]) == 1 and len(ev[ev["chave"] == "intervalo"]) == 1
    assert ev.loc[ev["chave"] == "mais_2.5", "acerto"].notna().all()
    assert (ev["logloss"] > 0).all() and (ev["confianca"] >= 1 / 3).all()

    registo.save(store, str(tmp_path))
    again = registo.load(str(tmp_path))
    assert again == store
    page = pagina.render_site(pd.DataFrame(), track=again)["historico.html"]
    assert "Histórico de acertos" in page and "Casas: acerto" in page and "2–1" in page
    assert "Liga Portugal" in page and "Ao intervalo (1X2)" in page


def test_unresolved_game_is_dropped_after_two_weeks(league):
    store = {}
    registo.record(store, _football(league, date(2026, 10, 27)), "futebol", NOW)
    registo.resolve(store, pd.DataFrame(), "futebol", NOW + timedelta(days=5))
    assert next(iter(store.values()))["estado"] == "pendente"
    registo.resolve(store, pd.DataFrame(), "futebol", NOW + timedelta(days=20))
    assert next(iter(store.values()))["estado"] == "sem resultado"
    assert registo.evaluate(store).empty


def test_basket_and_hockey_markets():
    games, *_ = basket_season()
    games = games.assign(league="NBA", round=None, time="01:00")
    last = games["date"].max()
    day = last + timedelta(days=2)
    future = pd.DataFrame([{"league": "NBA", "date": day, "time": "01:00", "round": None,
                            "home": "B1", "away": "B2", "home_points": None, "away_points": None}])
    hoops = basquete.predict_upcoming(pd.concat([games, future], ignore_index=True), last + timedelta(days=1))
    hg, _ = hockey_season()
    hlast = hg["date"].max()
    hfuture = pd.DataFrame([{"league": "NHL", "date": day, "time": "01:00", "round": None,
                             "home": "H1", "away": "H2"}])
    ice = hoquei.predict_upcoming(pd.concat([hg, hfuture], ignore_index=True), hlast + timedelta(days=1))
    now = registo.kickoff(day, "01:00") - timedelta(hours=10)
    store = {}
    assert registo.record(store, hoops, "basquetebol", now) == 1
    assert registo.record(store, ice, "hoquei", now) == 1
    basket_res = pd.DataFrame([{"league": "NBA", "date": day, "home": "B1", "away": "B2",
                                "home_points": 101.0, "away_points": 99.0}])
    hockey_res = pd.DataFrame([{"league": "NHL", "date": day, "home": "H1", "away": "H2",
                                "home_goals": 3, "away_goals": 2, "home_reg": 2, "away_reg": 2}])
    later = now + timedelta(days=1)
    assert registo.resolve(store, basket_res, "basquetebol", later) == 1
    assert registo.resolve(store, hockey_res, "hoquei", later) == 1
    ev = registo.evaluate(store).set_index(["desporto", "chave"])
    assert len(ev.loc["basquetebol"]) == 4 and len(ev.loc["hoquei"]) == 6
    # empate aos 60 minutos: "vai a prolongamento" aconteceu
    pro = store[next(k for k in store if k.startswith("hoquei"))]["p"]["prolongamento"]
    assert ev.loc[("hoquei", "prolongamento"), "acerto"] == float(pro >= 0.5) * 1.0
    page = pagina.render_site(pd.DataFrame(), track=store)["historico.html"]
    assert "Puck line -1,5 da casa" in page and "Total dentro do intervalo de 80%" in page


def test_safe_pick_rules():
    assert registo.safe_pick([0.70, 0.20, 0.10]) == ("vitoria", 0, 0.70)
    assert registo.safe_pick([0.15, 0.25, 0.60]) == ("dupla", 2, 0.85)
    assert registo.safe_pick([0.45, 0.30, 0.25]) is None
    assert registo.safe_pick([0.30, 0.40, 0.30]) is None  # o empate é o mais provável


def test_safe_page_and_history():
    now = datetime(2026, 10, 30, 8, 0, tzinfo=timezone.utc)
    preds = pd.DataFrame([
        {"league": "POR", "date": date(2026, 10, 31), "time": "20:30", "home": "Benfica", "away": "Arouca",
         "casa": 0.78, "empate": 0.14, "fora": 0.08},
        {"league": "ENG", "date": date(2026, 11, 1), "time": "15:00", "home": "Burnley", "away": "Arsenal",
         "casa": 0.15, "empate": 0.24, "fora": 0.61},
        {"league": "ESP", "date": date(2026, 11, 1), "time": "15:00", "home": "Getafe", "away": "Celta",
         "casa": 0.40, "empate": 0.30, "fora": 0.30},
        {"league": "POR", "date": date(2026, 12, 20), "time": "20:30", "home": "Porto", "away": "Arouca",
         "casa": 0.80, "empate": 0.12, "fora": 0.08},  # fora dos 7 dias
        {"league": "USA", "date": date(2026, 11, 1), "time": "20:00", "home": "Miami", "away": "Austin",
         "casa": 0.80, "empate": 0.12, "fora": 0.08}])  # MLS não entra
    page = pagina._safe_page(preds, now)
    assert "<b>Benfica</b>" in page and "<b>Arsenal ou empate</b>" in page
    assert "Getafe" not in page and "Porto" not in page and "Miami" not in page

    store = {"a": {"desporto": "futebol", "estado": "terminado", "liga": "POR", "data": "2026-10-31",
                   "p": {"1x2": [0.78, 0.14, 0.08]}, "resultado": {"casa": 1, "fora": 1}},
             "b": {"desporto": "futebol", "estado": "terminado", "liga": "ENG", "data": "2026-11-01",
                   "p": {"1x2": [0.15, 0.24, 0.61]}, "resultado": {"casa": 1, "fora": 1}}}
    safe = registo.safe_evaluate(store).set_index("tipo")
    assert safe.loc["vitoria", "acerto"] == 0 and safe.loc["dupla", "acerto"] == 1
