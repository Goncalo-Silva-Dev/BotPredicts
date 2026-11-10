import pandas as pd

from analise import aovivo, pagina


def _event(eid, when, home, away):
    return {"id": eid, "date": when, "status": {"type": {"state": "pre"}},
            "competitions": [{"competitors": [
                {"homeAway": "home", "team": {"displayName": home, "shortDisplayName": home}},
                {"homeAway": "away", "team": {"displayName": away, "shortDisplayName": away}}]}]}


def test_similarity_across_sources():
    for a, b in [("Sp Lisbon", "Sporting CP"), ("Man United", "Manchester United"),
                 ("Nott'm Forest", "Nottingham Forest"), ("Ath Bilbao", "Athletic Club"),
                 ("Paris SG", "Paris Saint-Germain"), ("Gremio", "Grêmio"), ("FC Koln", "FC Cologne"),
                 ("La Coruna", "Deportivo"), ("Los Angeles FC", "LAFC")]:
        assert aovivo.similarity(a, b) >= aovivo.MIN_SCORE, (a, b)
    for a, b in [("Vitoria", "Grêmio"), ("Benfica", "FC Porto"), ("Man United", "Man City")]:
        assert aovivo.similarity(a, b) < aovivo.similarity(a, a)
    assert aovivo.similarity("Man United", "Manchester City") < aovivo.similarity("Man United", "Manchester United")


def test_match_pairs_each_game_once():
    events = aovivo.parse_events([
        _event("1", "2026-10-31T15:00Z", "Manchester United", "Chelsea"),
        _event("2", "2026-10-31T17:30Z", "Manchester City", "Arsenal"),
        _event("3", "2026-11-03T20:00Z", "Coritiba", "Grêmio"),
    ], "soccer/eng.1", "20261031")
    games = pd.DataFrame([
        {"league": "ENG", "date": "2026-10-31", "home": "Man City", "away": "Arsenal"},
        {"league": "ENG", "date": "2026-10-31", "home": "Man United", "away": "Chelsea"},
        {"league": "ENG", "date": "2026-10-31", "home": "Coritiba", "away": "Vitoria"},  # só a casa bate
        {"league": "ENG", "date": "2026-10-20", "home": "Man United", "away": "Chelsea"},  # longe demais
    ])
    refs = aovivo.match(games, events).tolist()
    assert refs[0] == "soccer/eng.1|20261031|2|2026-10-31T17:30Z"
    assert refs[1].split("|")[2] == "1"
    assert refs[2] is None and refs[3] is None


def test_page_has_live_hooks_only_for_matched_games():
    g = {"league": "NHL", "date": "2026-10-31", "time": "23:00", "home": "Boston Bruins",
         "away": "Chicago Blackhawks", "casa": float("nan")}
    plain = pagina.render_site(pd.DataFrame(), hockey=pd.DataFrame([g]))["hoquei/nhl.html"]
    assert "Boston Bruins – Chicago Blackhawks" in plain and "data-espn=" not in plain
    live = pagina.render_site(pd.DataFrame(), hockey=pd.DataFrame([{**g, "espn": "hockey/nhl|20261031|9|2026-10-31T23:00Z"}]))
    page = live["hoquei/nhl.html"]
    assert 'data-espn="hockey/nhl|20261031|9|2026-10-31T23:00Z"' in page
    assert '<span class="sc">–</span>' in page and "a cada 30 segundos" in page
    assert "não tem resultados ao vivo" in live["futebol/po2.html"]
