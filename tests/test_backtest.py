from datetime import timedelta

import numpy as np
import pandas as pd

from analise import backtest


def test_walk_forward_only_uses_past(league):
    res = backtest.walk_forward(league, min_train=150)
    first = pd.Timestamp(res["date"].min())
    # O primeiro jogo previsto vem depois de haver treino suficiente
    assert (pd.to_datetime(league["date"]) < first).sum() >= 150
    assert res[["model_home", "model_draw", "model_away"]].sum(axis=1).round(9).eq(1).all()
    assert res["market_home"].round(6).eq(0.45).all()


def test_summary_and_calibration(league):
    res = backtest.walk_forward(league, min_train=150)
    s = backtest.summary(res)
    assert s["avaliados"] == len(res) > 0
    assert 0.8 < s["logloss_model"] < 1.3
    cal = backtest.calibration(res)
    assert cal["n"].sum() == 3 * s["avaliados"]


def test_unknown_teams_are_skipped(league):
    extra = league.iloc[[-1]].assign(home="Promovida", date=league["date"].max() + timedelta(days=1))
    res = backtest.walk_forward(pd.concat([league, extra]), min_train=150)
    assert res[res["home"] == "Promovida"]["model_home"].isna().all()


def test_scores():
    probs = np.array([[1.0, 0, 0], [0, 0, 1.0]])
    y = np.array([0, 2])
    assert backtest.brier(probs, y) == 0
    assert backtest.rps(probs, y) == 0
    # Prever empate num jogo ganho em casa custa menos no RPS do que prever fora
    assert backtest.rps(np.array([[0, 1.0, 0]]), np.array([0])) < backtest.rps(
        np.array([[0, 0, 1.0]]), np.array([0]))
