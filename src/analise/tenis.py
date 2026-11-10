"""Previsões e backtest de ténis: Elo para o vencedor e modelo de pontos para sets e jogos."""
from datetime import date

import numpy as np
import pandas as pd

from analise.models.tennis import TennisElo, markets, predict_distribution

YEARS = 5  # anos de jogos no site (no teste, começar em 2019 ou em 2022 deu o mesmo acerto)
MIN_GAMES = 10  # jogadores com menos jogos no histórico ficam com aviso (força pouco conhecida)
ELO_OPTIONS = {"surface_weight": 0.5, "mov": 0.7}
USE_RETIREMENTS = True  # quem desiste perde, para o Elo (mas os sets e jogos não contam)


def _played(matches: pd.DataFrame) -> pd.DataFrame:
    keep = ["final", "desistencia"] if USE_RETIREMENTS else ["final"]
    return matches[matches["status"].isin(keep) & matches["winner"].notna()]


def _order(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(_t=df["time"].fillna("23:59")).sort_values(["date", "_t", "id"]).drop(columns="_t")


def predict_match(elo: TennisElo, home_id, away_id, league, surface, best_of) -> dict:
    p3 = elo.predict(home_id, away_id, surface)
    dist = predict_distribution(p3, league, surface, int(best_of))
    return {"casa": float(dist["win"]), "fora": 1 - float(dist["win"]), "dist": dist}


def fit(matches: pd.DataFrame, until=None, **options) -> TennisElo:
    """Elo com todos os jogos terminados antes de `until` (data)."""
    elo = TennisElo(**{**ELO_OPTIONS, **options})
    played = _order(_played(matches))
    if until is not None:
        played = played[pd.to_datetime(played["date"]) < pd.Timestamp(until)]
    for g in played.itertuples(index=False):
        _update(elo, g)
    return elo


def _update(elo, g, expected=None):
    share = None
    if g.status == "final":
        a = sum(getattr(g, f"home_s{k}") for k in range(1, 6) if pd.notna(getattr(g, f"home_s{k}")))
        b = sum(getattr(g, f"away_s{k}") for k in range(1, 6) if pd.notna(getattr(g, f"away_s{k}")))
        share = a / (a + b) if a + b > 0 else None
    elo.update(g.home_id, g.away_id, g.surface, g.winner == 0, expected, share)


def predict_upcoming(matches: pd.DataFrame, today: date | None = None) -> pd.DataFrame:
    """Uma linha por jogo ainda por jogar (com os dois jogadores já conhecidos)."""
    today = pd.Timestamp(today or date.today())
    future = matches[(matches["status"] == "por jogar") & (pd.to_datetime(matches["date"]) >= today)]
    if future.empty:
        return pd.DataFrame()
    elo = fit(matches, until=None)
    rows = []
    for g in future.itertuples(index=False):
        p = predict_match(elo, g.home_id, g.away_id, g.league, g.surface, g.best_of)
        mk = markets(p["dist"], int(g.best_of))
        rows.append({
            # para o resultado e as percentagens ao vivo (id do jogo na ESPN e Elo à melhor de 3)
            "espn_id": str(g.id), "home_id": str(g.home_id),
            "p_bo3": elo.predict(g.home_id, g.away_id, g.surface),
            "league": g.league, "round": g.tournament, "fase": g.round, "date": g.date, "time": g.time,
            "home": g.home, "away": g.away, "surface": g.surface, "best_of": int(g.best_of),
            "casa": p["casa"], "fora": p["fora"],
            "jogos_casa": elo.games(g.home_id), "jogos_fora": elo.games(g.away_id),
            "elo_casa": elo.strength(g.home_id, g.surface), "elo_fora": elo.strength(g.away_id, g.surface),
            "mercados": mk,
        })
    return pd.DataFrame(rows)


def walk_forward(matches: pd.DataFrame, start="2025-01-01", with_markets=True, **options) -> pd.DataFrame:
    """Cada jogo é previsto só com os jogos anteriores (o Elo é atualizado depois de cada um)."""
    elo = TennisElo(**{**ELO_OPTIONS, **options})
    start = pd.Timestamp(start)
    rows = []
    for g in _order(_played(matches)).itertuples(index=False):
        evaluate = pd.Timestamp(g.date) >= start
        p3 = elo.predict(g.home_id, g.away_id, g.surface)
        if int(g.best_of) == 3 and not evaluate:
            _update(elo, g, p3)  # à melhor de 3 o Elo já dá a probabilidade: poupa o resto
            continue
        if int(g.best_of) == 3 and not with_markets:
            p = {"casa": p3}
        else:
            p = predict_match(elo, g.home_id, g.away_id, g.league, g.surface, g.best_of)
        if evaluate:
            row = {"id": g.id, "league": g.league, "date": g.date, "surface": g.surface,
                   "best_of": int(g.best_of), "status": g.status, "home": g.home, "away": g.away,
                   "casa": p["casa"], "y": float(g.winner == 0),
                   "n_casa": elo.games(g.home_id), "n_fora": elo.games(g.away_id),
                   "home_sets": g.home_sets, "away_sets": g.away_sets}
            if with_markets and g.status == "final":
                sets = [(getattr(g, f"home_s{k}"), getattr(g, f"away_s{k}")) for k in range(1, 6)]
                sets = [(a, b) for a, b in sets if pd.notna(a) and pd.notna(b)]
                mk = markets(p["dist"], int(g.best_of))
                row.update({"mk": mk, "real_games": (sum(a for a, _ in sets), sum(b for _, b in sets)),
                            "real_first": sets[0][0] > sets[0][1] if sets else None,
                            "real_tb": any(max(a, b) == 7 and min(a, b) == 6 for a, b in sets)})
            rows.append(row)
        _update(elo, g, p["casa"])
    return pd.DataFrame(rows)


def _ll(p, y):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, dtype=float)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def summary(res: pd.DataFrame) -> dict:
    p, y = res["casa"].to_numpy(), res["y"].to_numpy()
    fav = np.maximum(p, 1 - p)
    out = {"jogos": len(res), "acerto": float(((p >= 0.5) == (y == 1)).mean()),
           "confianca": float(fav.mean()), "logloss": _ll(p, y), "logloss_base": float(np.log(2)),
           "brier": float(((p - y) ** 2).mean())}
    strong = fav >= 0.75
    out["fav75_jogos"] = int(strong.sum())
    out["fav75_acerto"] = float(((p >= 0.5) == (y == 1))[strong].mean()) if strong.any() else float("nan")
    known = (res["n_casa"] >= 10) & (res["n_fora"] >= 10)
    out["conhecidos_acerto"] = float(((p >= 0.5) == (y == 1))[known].mean())
    out["conhecidos_jogos"] = int(known.sum())
    if "mk" in res:
        m = res.dropna(subset=["mk"])
        if len(m):
            out.update(market_summary(m))
    return out


def market_summary(m: pd.DataFrame) -> dict:
    """Mercados de jogos e sets: probabilidade prevista média contra frequência real."""
    tot_p, tot_y, hc_p, hc_y, fs_p, fs_y, tb_p, tb_y, mean_pred, mean_real = ([] for _ in range(10))
    for r in m.itertuples(index=False):
        mk, (ga, gb) = r.mk, r.real_games
        line, p = min(mk["total"], key=lambda lp: abs(lp[1] - 0.5))  # linha mais equilibrada
        tot_p.append(p), tot_y.append(ga + gb > line)
        line, p = min(mk["handicap"], key=lambda lp: abs(lp[1] - 0.5))
        hc_p.append(p), hc_y.append(ga - gb + line > 0)
        fs_p.append(mk["primeiro_set"]), fs_y.append(bool(r.real_first))
        tb_p.append(mk["tiebreak"]), tb_y.append(bool(r.real_tb))
        mean_pred.append(mk["total_esperado"]), mean_real.append(ga + gb)
    sets_ll = []
    for r in m.itertuples(index=False):
        probs = dict(r.mk["sets"])
        key = f"{int(r.home_sets)}-{int(r.away_sets)}"
        sets_ll.append(-np.log(max(probs.get(key, 1e-6), 1e-6)))
    acc = lambda p, y: float(((np.array(p) >= 0.5) == np.array(y)).mean())
    return {
        "total_jogos_prev": float(np.mean(mean_pred)), "total_jogos_real": float(np.mean(mean_real)),
        "linha_total_acerto": acc(tot_p, tot_y), "linha_total_ll": _ll(tot_p, tot_y),
        "handicap_acerto": acc(hc_p, hc_y), "handicap_ll": _ll(hc_p, hc_y),
        "primeiro_set_acerto": acc(fs_p, fs_y), "primeiro_set_ll": _ll(fs_p, fs_y),
        "tiebreak_prev": float(np.mean(tb_p)), "tiebreak_real": float(np.mean(tb_y)),
        "tiebreak_ll": _ll(tb_p, tb_y),
        "sets_ll": float(np.mean(sets_ll)),
    }


def calibration(res: pd.DataFrame) -> pd.DataFrame:
    """Jogos agrupados pela probabilidade do favorito: quem diz 70% deve acertar 7 em 10."""
    p, y = res["casa"].to_numpy(), res["y"].to_numpy()
    fav = np.maximum(p, 1 - p)
    hit = (p >= 0.5) == (y == 1)
    rows = []
    for lo, hi in [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]:
        m = (fav >= lo) & (fav < hi)
        if m.any():
            rows.append({"grupo": f"{lo:.0%}-{min(hi, 1):.0%}", "jogos": int(m.sum()),
                         "previsto": float(fav[m].mean()), "acertou": float(hit[m].mean())})
    return pd.DataFrame(rows)
