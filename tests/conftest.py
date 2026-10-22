from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def league():
    """Duas épocas simuladas de 10 equipas, com odds tiradas das forças verdadeiras."""
    rng = np.random.default_rng(1)
    teams = [f"T{i}" for i in range(10)]
    att, dfc = rng.normal(0, 0.3, 10), rng.normal(0, 0.3, 10)
    rows, d = [], date(2023, 8, 1)
    for _ in range(4):
        for i in range(10):
            for j in range(10):
                if i == j:
                    continue
                lam, mu = np.exp(att[i] + dfc[j] + 0.25), np.exp(att[j] + dfc[i])
                p = np.array([0.45, 0.27, 0.28]) * 1.05
                rows.append({
                    "date": d, "home": teams[i], "away": teams[j],
                    "home_goals": rng.poisson(lam), "away_goals": rng.poisson(mu),
                    "odds_home_pin": 1 / p[0], "odds_draw_pin": 1 / p[1], "odds_away_pin": 1 / p[2],
                    "odds_home_avg": None, "odds_draw_avg": None, "odds_away_avg": None,
                })
                d += timedelta(days=1)
    return pd.DataFrame(rows)
