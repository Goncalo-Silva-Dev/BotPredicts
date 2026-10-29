from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from analise.models.dixon_coles import DixonColes, fair_probs, log_loss


@pytest.fixture(scope="module")
def simulated():
    """Liga simulada com forças conhecidas: A forte, D fraca, vantagem casa 0.3."""
    rng = np.random.default_rng(0)
    attack = {"A": 0.5, "B": 0.1, "C": -0.1, "D": -0.5}
    defence = {"A": -0.4, "B": 0.0, "C": 0.1, "D": 0.3}
    rows, d = [], date(2024, 1, 1)
    for _ in range(40):
        for h in attack:
            for a in attack:
                if h == a:
                    continue
                lam = np.exp(attack[h] + defence[a] + 0.3)
                mu = np.exp(attack[a] + defence[h])
                rows.append((d, h, a, rng.poisson(lam), rng.poisson(mu)))
                d += timedelta(days=1)
    return pd.DataFrame(rows, columns=["date", "home", "away", "home_goals", "away_goals"])


def test_fit_recovers_known_strengths(simulated):
    model = DixonColes(half_life_days=10_000).fit(simulated)
    att = dict(zip(model.teams_, model.attack_))
    assert att["A"] > att["B"] > att["C"] > att["D"]
    assert model.home_ == pytest.approx(0.3, abs=0.1)


def test_predictions_are_probabilities(simulated):
    model = DixonColes().fit(simulated)
    p = model.predict("A", "D")
    assert p["home"] + p["draw"] + p["away"] == pytest.approx(1, abs=1e-9)
    assert p["home"] > 0.6
    assert 0 < p["over_2_5"] < 1 and 0 < p["btts"] < 1


def test_fair_probs_removes_margin():
    p = fair_probs(2.0, 3.4, 3.8)
    assert sum(p) == pytest.approx(1)
    assert p[0] > p[2]


def test_log_loss():
    probs = np.array([[0.5, 0.3, 0.2], [0.2, 0.3, 0.5]])
    assert log_loss(probs, np.array([0, 2])) == pytest.approx(-np.log(0.5))


def test_shots_on_target_move_strengths(simulated):
    """Com os mesmos golos, a equipa com mais remates à baliza fica com mais ataque."""
    df = simulated.assign(home_shots_target=simulated["home_goals"] * 3 + 1,
                          away_shots_target=simulated["away_goals"] * 3 + 1)
    df.loc[df["home"] == "C", "home_shots_target"] += 6
    goals = DixonColes(goals_weight=1.0).fit(df)
    mixed = DixonColes().fit(df)
    gap = lambda m: dict(zip(m.teams_, m.attack_))["C"] - dict(zip(m.teams_, m.attack_))["B"]  # noqa: E731
    assert gap(mixed) > gap(goals)
    p = mixed.predict("A", "D")
    assert p["home"] + p["draw"] + p["away"] == pytest.approx(1, abs=1e-9)
    assert -0.3 <= mixed.rho_ <= 0.3


def test_without_shots_uses_goals_only(simulated):
    a = DixonColes().fit(simulated)
    b = DixonColes(goals_weight=1.0).fit(simulated)
    assert a.predict("A", "D")["home"] == pytest.approx(b.predict("A", "D")["home"])
