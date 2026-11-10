"""Percentagens ao vivo: a matemática em Python e a do browser (aovivo.js) têm de dar o mesmo."""
import json
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

from analise import aovivo_modelo as am
from analise.models import tennis as tm


def test_soccer_kickoff_close_to_pregame_and_sums_to_one():
    r = am.soccer_live(1.6, 1.1, 0, 0, 0, 0, 1, 0, None)
    assert abs(r["casa"] + r["empate"] + r["fora"] - 1) < 1e-6
    # no início o modelo ao vivo fica perto do Poisson de antes do jogo
    g = np.arange(12)
    from scipy.stats import poisson
    m = np.outer(poisson.pmf(g, 1.6), poisson.pmf(g, 1.1))
    assert abs(r["casa"] - np.tril(m, -1).sum()) < 0.03


def test_soccer_moves_the_right_way():
    early = am.soccer_live(1.4, 1.4, 1, 0, 0, 0, 1, 20, None)["casa"]
    late = am.soccer_live(1.4, 1.4, 1, 0, 0, 0, 2, 85, None)["casa"]
    stoppage = am.soccer_live(1.4, 1.4, 1, 0, 0, 0, 2, 90, 6)["casa"]
    assert 0.5 < early < late < stoppage < 1
    # uma expulsão da casa baixa as hipóteses dela
    red = am.soccer_live(1.4, 1.4, 0, 0, 1, 0, 1, 30, None)
    even = am.soccer_live(1.4, 1.4, 0, 0, 0, 0, 1, 30, None)
    assert red["casa"] < even["casa"] and red["fora"] > even["fora"]


def test_soccer_steps_total_weight():
    w = am.soccer_steps(1, 0, None)
    assert 1.0 < sum(w) < 1.05  # o fator do marcador faz a média voltar a ~1
    assert sum(am.soccer_steps(2, 90, 30)) == 0


def test_basket_end_of_game_and_halftime():
    r = am.basket_live(4, 220, 13, 18, 10, 110, 0.5)
    assert 0.8 < r["casa"] < 0.95
    assert am.basket_live(-3, 220, 13, 18, 1, 200, 0.0)["casa"] > 0.99


def test_hockey_and_tennis_sanity():
    h = am.hockey_live(3.0, 2.8, 2, 0, 3, 5.0)
    assert h["casa"] > 0.9
    base, sigma = tm.POINT_MODEL[("ATP", "rapido")]
    d0 = tm.serve_diff(0.6, base, sigma)
    start = am.tennis_live(base, sigma, d0, 3, [], (0, 0))["casa"]
    # no início do jogo dá a probabilidade de antes do jogo
    assert abs(start - tm.mixed_distribution(base, d0, sigma, 3)["win"]) < 0.01
    after = am.tennis_live(base, sigma, d0, 3, [(6, 2)], (3, 1))["casa"]
    assert after > 0.85


CASES = {
    "soccer": [
        [{"l": [1.6, 1.1]}, {"gh": 0, "ga": 0, "rh": 0, "ra": 0, "half": 1, "minute": 10, "stop": None}],
        [{"l": [1.2, 1.9]}, {"gh": 1, "ga": 1, "rh": 0, "ra": 1, "half": 2, "minute": 70, "stop": None}],
        [{"l": [2.0, 0.8]}, {"gh": 0, "ga": 1, "rh": 0, "ra": 0, "half": 1, "minute": 45, "stop": 2}],
        [{"l": [1.5, 1.5]}, {"gh": 2, "ga": 1, "rh": 0, "ra": 0, "half": 2, "minute": 90, "stop": 4}],
    ],
    "basket": [
        [{"m": 4.5, "t": 228, "sm": 13.5, "st": 18.7}, {"lead": -6, "points": 55, "tau": 0.6}],
        [{"m": -2, "t": 170, "sm": 12, "st": 16}, {"lead": 3, "points": 150, "tau": 0.1}],
    ],
    "hockey": [
        [{"l": [3.1, 2.7], "tl": [5.5, 6.5]}, {"gh": 1, "ga": 0, "period": 2, "left": 12.5}],
        [{"l": [2.9, 3.0], "tl": [5.5]}, {"gh": 2, "ga": 2, "period": 3, "left": 1.5}],
    ],
    "tennis": [
        [{"b": 0.64, "s": 0.07, "d": 0.05, "n": 3}, {"done": [[4, 6]], "cur": [2, 1]}],
        [{"b": 0.56, "s": 0.11, "d": -0.03, "n": 3}, {"done": [], "cur": [5, 5]}],
        [{"b": 0.65, "s": 0.07, "d": 0.0, "n": 5}, {"done": [[7, 6], [3, 6]], "cur": [0, 0]}],
    ],
}


def _python(sport, par, st):
    if sport == "soccer":
        r = am.soccer_live(*par["l"], st["gh"], st["ga"], st["rh"], st["ra"], st["half"], st["minute"], st["stop"])
        return [r["casa"], r["empate"], r["fora"], r["mais_2_5"], r["ambas"]]
    if sport == "basket":
        r = am.basket_live(par["m"], par["t"], par["sm"], par["st"], st["lead"], st["points"], st["tau"])
        return [r["casa"], r["margem"], r["total"]]
    if sport == "hockey":
        r = am.hockey_live(*par["l"], st["gh"], st["ga"], st["period"], st["left"])
        return [r["casa"], r["reg_casa"], r["reg_empate"]]
    r = am.tennis_live(par["b"], par["s"], par["d"], par["n"], [tuple(x) for x in st["done"]], tuple(st["cur"]))
    return [r["casa"], r["set"]]


JS_PICK = {
    "soccer": "r => [r.casa, r.empate, r.fora, r.mais[2.5], r.ambas]",
    "basket": "r => [r.casa, r.margem, r.total]",
    "hockey": "r => [r.casa, r.reg_casa, r.reg_empate]",
    "tennis": "r => [r.casa, r.set]",
}


@pytest.mark.skipif(shutil.which("node") is None, reason="sem Node.js")
def test_javascript_matches_python(tmp_path):
    js = tmp_path / "live.js"
    js.write_text(am.script(), encoding="utf-8")
    runner = tmp_path / "run.js"
    runner.write_text(
        "const L = require(%s); const cases = %s; const pick = {%s}; const out = {};\n"
        "for (const k in cases) out[k] = cases[k].map(([p, s]) => pick[k](L[k](p, s)));\n"
        "console.log(JSON.stringify(out));"
        % (json.dumps(str(js)), json.dumps(CASES), ",".join(f"{k}: {v}" for k, v in JS_PICK.items())),
        encoding="utf-8")
    got = json.loads(subprocess.run(["node", str(runner)], capture_output=True, text=True, check=True).stdout)
    for sport, cases in CASES.items():
        for (par, st), js_vals in zip(cases, got[sport]):
            py_vals = _python(sport, par, st)
            assert np.allclose(js_vals, py_vals, atol=2e-4), (sport, st, js_vals, py_vals)


def test_card_params_and_page_attributes():
    from analise import pagina
    g = pd.Series({"league": "POR", "date": "2026-11-11", "time": "20:15", "home": "Benfica", "away": "Porto",
                   "golos_casa": 1.5, "golos_fora": 1.2, "espn": "soccer/por.1|20261111|1|2026-11-11T20:15Z"})
    attr = pagina._live(g, "futebol")
    assert 'data-espn="soccer/por.1' in attr and "data-m=" in attr and "&quot;k&quot;" in attr
    assert am.card_params("futebol", pd.Series({"golos_casa": float("nan")})) is None
