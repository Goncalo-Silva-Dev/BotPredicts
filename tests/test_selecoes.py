from datetime import date

import numpy as np
import pandas as pd

from analise import selecoes
from analise.ingest import betexplorer
from analise.models.dixon_coles import DixonColes


def test_sources_cover_nations_league_qualifiers_and_friendlies():
    paths = {s.path for s in selecoes.sources(date(2026, 11, 14), 2020)}
    assert "football/europe/uefa-nations-league" in paths
    assert "football/europe/uefa-nations-league-2024-2025" in paths
    assert "football/europe/euro-2024" in paths and "football/world/world-championship-2026" in paths
    assert "football/world/friendly-international-2025" in paths
    assert "football/europe/uefa-nations-league-2018-2019" not in paths  # antes de first_year


def test_season_labels_and_neutral_finals():
    cur = selecoes.Source("football/europe/uefa-nations-league", "atual", selecoes.NATIONS)
    assert selecoes._season(cur, date(2026, 11, 13)) == "2026-27"
    assert selecoes._season(cur, date(2027, 3, 20)) == "2026-27"
    fr = selecoes.Source("football/world/friendly-international", "amigaveis", selecoes.OTHER)
    assert selecoes._season(fr, date(2026, 6, 1)) == "amigaveis-2026"
    assert selecoes.neutral("euro-2024", date(2024, 6, 20), "Spain")
    assert not selecoes.neutral("euro-2024", date(2024, 6, 20), "Germany")  # anfitrião
    assert not selecoes.neutral("euro-2024", date(2023, 9, 8), "Spain")  # qualificação
    assert not selecoes.neutral("2026-27", date(2026, 11, 13), "Spain")


def test_not_senior_teams_are_filtered():
    names = pd.Series(["Spain U21", "Spain W", "Spain", "Mali", "Catalonia Olympic"])
    assert list(names.str.contains(selecoes._NOT_SENIOR)) == [True, True, False, False, True]


def test_training_history_weights_friendlies():
    df = pd.DataFrame([
        {"league": "UNL", "season": "2026-27", "home": "A", "away": "B"},
        {"league": "SEL", "season": "amigaveis-2026", "home": "A", "away": "C"},
        {"league": "SEL", "season": "euro-2024", "home": "B", "away": "C"},
        {"league": "POR", "season": "2026-27", "home": "X", "away": "Y"},
    ])
    h = selecoes.training_history(df)
    assert list(h["weight"]) == [1.0, selecoes.FRIENDLY_WEIGHT, 1.0]


def test_fixtures_follow_all_stages(monkeypatch):
    row = ('<tr><td class="table-main__datetime">Tomorrow 18:45</td>\n<td class="h-text-left"><a href="/x/" '
           'class="in-match"><span>{h}</span> - <span>{a}</span></a></td></tr>')
    pages = {None: '<a href="?stage=AAA">League A</a><a href="?stage=BBB">League B</a>'
                   + row.format(h="Spain", a="England"),
             "AAA": row.format(h="Spain", a="England"), "BBB": row.format(h="Malta", a="Gibraltar")}
    monkeypatch.setattr(betexplorer, "_be_get", lambda path, page, stage=None: pages[stage])
    fx = betexplorer.fetch_fixtures("football/europe/uefa-nations-league", "UNL", date(2026, 11, 14),
                                    all_stages=True)
    assert list(fx["home"]) == ["Spain", "Malta"]
    assert len(betexplorer.fetch_fixtures("x", "UNL", date(2026, 11, 14))) == 1


def test_neutral_games_do_not_count_for_home_advantage():
    rng = np.random.default_rng(3)
    rows = []
    for i in range(3000):
        h, a = rng.choice(6, 2, replace=False)
        neutral = i % 2
        lam = np.exp(0.1 + (0.0 if neutral else 0.4))
        rows.append({"date": date(2026, 1, 1), "home": f"T{h}", "away": f"T{a}",
                     "home_goals": rng.poisson(lam), "away_goals": rng.poisson(np.exp(0.1)),
                     "neutral": neutral})
    df = pd.DataFrame(rows)
    with_flag = DixonColes(joint_rho=False).fit(df)
    without = DixonColes(joint_rho=False).fit(df.drop(columns="neutral"))
    assert abs(with_flag.home_ - 0.4) < 0.08
    assert without.home_ < with_flag.home_ - 0.1
