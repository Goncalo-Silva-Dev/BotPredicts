import argparse
import os
from datetime import date, datetime, timedelta, timezone

import pandas as pd
from sqlalchemy import select

from analise import backtest, basquete, calendario, europa, hoquei, pagina, previsoes, registo, tenis
from analise.ingest import basket, hockey, tennis
from analise.db import get_engine, matches
from analise.ingest import football_data_uk
from analise.leagues import LEAGUES, TRAINING_LEAGUES


def main(argv=None):
    parser = argparse.ArgumentParser(prog="analise")
    sub = parser.add_subparsers(dest="cmd", required=True)
    hist = sub.add_parser("historico", help="Descarrega o histórico do football-data.co.uk")
    hist.add_argument("--epocas", type=int, default=6)
    hist.add_argument("--ligas", nargs="*", default=list(LEAGUES) + list(TRAINING_LEAGUES),
                      choices=list(LEAGUES) + list(TRAINING_LEAGUES))
    bt = sub.add_parser("backtest", help="Compara o Dixon-Coles com as odds de fecho sem margem")
    bt.add_argument("--ligas", nargs="*", default=list(LEAGUES), choices=list(LEAGUES))
    bt.add_argument("--meia-vida", type=float, default=180.0, help="dias")
    bt.add_argument("--janela", type=int, default=730, help="dias de histórico no treino")
    bt.add_argument("--desde", help="só avalia jogos a partir desta data (AAAA-MM-DD)")
    bt.add_argument("--peso-golos", type=float, default=0.5,
                    help="peso dos golos face aos remates à baliza (1 = só golos)")
    bt.add_argument("--saida", default="data/backtest", help="pasta para os CSVs por jogo")
    prev = sub.add_parser("previsoes", help="Probabilidades para os próximos jogos")
    prev.add_argument("--ligas", nargs="*", default=list(LEAGUES), choices=list(LEAGUES))
    prev.add_argument("--dias", type=int, default=7, help="só jogos nos próximos N dias")
    prev.add_argument("--meia-vida", type=float, default=180.0, help="dias")
    prev.add_argument("--janela", type=int, default=730, help="dias de histórico no treino")
    prev.add_argument("--jogo", nargs=2, metavar=("CASA", "FORA"),
                      help="prevê um jogo à escolha em vez do calendário (precisa de 1 liga)")
    prev.add_argument("--saida", default="data/previsoes.csv")
    bb = sub.add_parser("basquete-backtest", help="Testa o modelo de basquetebol em jogos passados")
    bb.add_argument("--ligas", nargs="*", default=list(basket.BASKET_LEAGUES),
                    choices=list(basket.BASKET_LEAGUES))
    bb.add_argument("--epocas", type=int, default=3)
    hb = sub.add_parser("hoquei-backtest", help="Testa o modelo de hóquei em jogos passados")
    hb.add_argument("--ligas", nargs="*", default=list(hockey.HOCKEY_LEAGUES),
                    choices=list(hockey.HOCKEY_LEAGUES))
    hb.add_argument("--epocas", type=int, default=3)
    tn = sub.add_parser("tenis-backtest", help="Testa o modelo de ténis em jogos passados")
    tn.add_argument("--anos", type=int, default=8, help="anos de jogos (o Elo aquece nos primeiros)")
    tn.add_argument("--desde", default="2025-01-01", help="primeiro jogo avaliado (AAAA-MM-DD)")
    tn.add_argument("--saida", default="data/backtest/TENIS.csv")
    tb = sub.add_parser("tacas-backtest", help="Testa o modelo das taças europeias em jogos passados")
    tb.add_argument("--desde", default="2022-07-01", help="primeiro jogo avaliado (AAAA-MM-DD)")
    tb.add_argument("--saida", default="data/backtest/TACAS.csv")
    pag = sub.add_parser("pagina", help="Gera a página web com as previsões")
    pag.add_argument("--dias", type=int, help="só jogos nos próximos N dias (por omissão, todos)")
    pag.add_argument("--saida", default="site", help="pasta da página")
    pag.add_argument("--dados-basquete", default=os.environ.get("BASKET_DATA_DIR"),
                     help="pasta com CSVs de reserva por liga (p. ex. NBB.csv) para as ligas "
                          "cujo site recusa a recolha")
    pag.add_argument("--historico", default=os.environ.get("HISTORICO_DIR"),
                     help="pasta do registo das previsões (ramo 'historico'): guarda as de hoje, "
                          "junta os resultados e gera a página do histórico")
    bd = sub.add_parser("basquete-dados", help="Guarda os jogos de basquetebol em CSV por liga")
    bd.add_argument("--ligas", nargs="*", default=["NBB", "LNB"], choices=list(basket.BASKET_LEAGUES))
    bd.add_argument("--saida", default="dados-basquete")
    args = parser.parse_args(argv)

    engine = get_engine()
    if args.cmd == "historico":
        for key in args.ligas:
            n = football_data_uk.load_history(engine, {**LEAGUES, **TRAINING_LEAGUES}[key], args.epocas)
            print(f"{key}: {n} jogos")
    elif args.cmd == "backtest":
        run_backtest(engine, args)
    elif args.cmd == "previsoes":
        run_previsoes(engine, args)
    elif args.cmd == "basquete-backtest":
        run_basquete_backtest(args)
    elif args.cmd == "hoquei-backtest":
        run_hoquei_backtest(args)
    elif args.cmd == "tenis-backtest":
        run_tenis_backtest(args)
    elif args.cmd == "tacas-backtest":
        run_tacas_backtest(engine, args)
    elif args.cmd == "pagina":
        run_pagina(engine, args)
    elif args.cmd == "basquete-dados":
        run_basquete_dados(args)


def run_backtest(engine, args):
    os.makedirs(args.saida, exist_ok=True)
    rows, all_res = [], []
    for key in args.ligas:
        with engine.connect() as conn:
            df = pd.read_sql(select(matches).where(matches.c.league == key), conn)
        if df.empty:
            print(f"{key}: sem dados (corre primeiro 'historico')")
            continue
        res = backtest.walk_forward(
            df, half_life_days=args.meia_vida, window_days=args.janela, start=args.desde,
            goals_weight=args.peso_golos)
        res.insert(0, "league", key)
        res.to_csv(os.path.join(args.saida, f"{key}.csv"), index=False)
        all_res.append(res)
        rows.append({"liga": key, **backtest.summary(res)})
        print(f"{key}: {rows[-1]['avaliados']} jogos avaliados")
    if not all_res:
        return
    total = pd.concat(all_res)
    rows.append({"liga": "TOTAL", **backtest.summary(total)})
    pd.set_option("display.width", 160)
    print()
    print(pd.DataFrame(rows).set_index("liga").round(4).to_string())
    for who in ("model", "market"):
        print(f"\nCalibração ({'modelo' if who == 'model' else 'mercado'})")
        print(backtest.calibration(total, who).round(3).to_string())


def run_previsoes(engine, args):
    today = date.today()
    if args.jogo:
        if len(args.ligas) != 1:
            print("Com --jogo indica uma liga, p. ex. --ligas POR")
            return
        fixtures = pd.DataFrame([{
            "league": args.ligas[0], "date": today, "time": None, "home": args.jogo[0],
            "away": args.jogo[1], "odds_home": None, "odds_draw": None, "odds_away": None}])
    else:
        fixtures = football_data_uk.fetch_fixtures([LEAGUES[k] for k in args.ligas])
        fixtures = fixtures[(fixtures["date"] >= today)
                            & (fixtures["date"] <= today + timedelta(days=args.dias))]
    res = predict_fixtures(engine, fixtures, args.meia_vida, args.janela, today)
    if res.empty:
        print("Sem jogos marcados no football-data.co.uk para estas ligas e datas.")
        return
    os.makedirs(os.path.dirname(args.saida) or ".", exist_ok=True)
    res.to_csv(args.saida, index=False)
    pct = ["casa", "empate", "fora", "mais_2_5", "ambas_marcam"]
    show = res[["league", "date", "home", "away"] + pct + ["resultados"]].copy()
    show[pct] = (show[pct] * 100).round(0)
    pd.set_option("display.width", 200)
    print(show.to_string(index=False))
    missing = res[res["casa"].isna()]
    if not missing.empty:
        print("\nSem previsão (equipa com menos de 5 jogos no treino ou nome diferente do histórico):")
        print(", ".join(f"{g.home} - {g.away}" for g in missing.itertuples()))
    print(f"\nGuardado em {args.saida} (inclui golos esperados e probabilidades do mercado)")



def predict_fixtures(engine, fixtures, half_life_days, window_days, today):
    out = []
    cup_history = None
    for key, games in fixtures.groupby("league"):
        if key in europa.CUPS:
            # um só modelo para as três taças, com os campeonatos nacionais (ver europa.py)
            if cup_history is None:
                since = today - timedelta(days=window_days + 1)
                with engine.connect() as conn:
                    cup_history = europa.training_history(
                        pd.read_sql(select(matches).where(matches.c.date >= since), conn))
            if not cup_history["league"].isin(europa.CUPS).any():
                print(f"{key}: sem histórico das taças (corre primeiro 'historico')")
                continue
            out.append(previsoes.predict(
                cup_history, games, as_of=today, half_life_days=europa.HALF_LIFE,
                window_days=window_days, min_team_matches=europa.MIN_MATCHES,
                model_options=europa.MODEL_OPTIONS, prepare=europa.align_with_odds))
            continue
        with engine.connect() as conn:
            hist = pd.read_sql(select(matches).where(matches.c.league == key), conn)
        if hist.empty:
            print(f"{key}: sem histórico (corre primeiro 'historico')")
            continue
        out.append(previsoes.predict(
            hist, games, as_of=today, half_life_days=half_life_days, window_days=window_days))
    if not out:
        return pd.DataFrame()
    res = pd.concat(out).sort_values(["date", "time", "league"], na_position="last")
    for col in ["casa", "empate", "fora", "mais_2_5", "ambas_marcam", "resultados"]:
        if col not in res:
            res[col] = None
    return res


def run_tacas_backtest(engine, args):
    """Precisa do histórico com épocas suficientes antes de --desde (p. ex. historico --epocas 7)."""
    with engine.connect() as conn:
        everything = pd.read_sql(select(matches), conn)
    res = europa.walk_forward(everything, args.desde)
    os.makedirs(os.path.dirname(args.saida) or ".", exist_ok=True)
    res.to_csv(args.saida, index=False)
    rows = [{"liga": k, **europa.summary(g)} for k, g in res.groupby("league")]
    rows.append({"liga": "TOTAL", **europa.summary(res)})
    pd.set_option("display.width", 200)
    print(pd.DataFrame(rows).set_index("liga").round(3).to_string())


def run_pagina(engine, args):
    today = date.today()
    fixtures = calendario.upcoming(engine, list(LEAGUES.values()), today)
    if args.dias:
        fixtures = fixtures[fixtures["date"] <= today + timedelta(days=args.dias)]
    res = predict_fixtures(engine, fixtures, 180.0, 730, today)
    basket_games, hockey_games = [], []
    hoops = basket_predictions(today, args.dados_basquete, basket_games)
    if args.dias and not hoops.empty:
        hoops = hoops[hoops["date"] <= today + timedelta(days=args.dias)]
    ice = hockey_predictions(today, hockey_games)
    if args.dias and not ice.empty:
        ice = ice[ice["date"] <= today + timedelta(days=args.dias)]
    tennis_games = []
    courts = tennis_predictions(today, tennis_games)
    if args.dias and not courts.empty:
        courts = courts[courts["date"] <= today + timedelta(days=args.dias)]
    track = None
    if args.historico:
        track = update_track(engine, args.historico,
                             {"futebol": res, "basquetebol": hoops, "hoquei": ice, "tenis": courts},
                             {"basquetebol": basket_games, "hoquei": hockey_games, "tenis": tennis_games})
    for path, page in pagina.render_site(res, hoops, hockey=ice, track=track, tennis=courts).items():
        full = os.path.join(args.saida, path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(page)
    with open(os.path.join(args.saida, "previsoes.json"), "w", encoding="utf-8") as f:
        f.write(pagina.to_json(res))
    with open(os.path.join(args.saida, "basquetebol.json"), "w", encoding="utf-8") as f:
        f.write(pagina.to_json(hoops))
    with open(os.path.join(args.saida, "hoquei.json"), "w", encoding="utf-8") as f:
        f.write(pagina.to_json(ice))
    with open(os.path.join(args.saida, "tenis.json"), "w", encoding="utf-8") as f:
        f.write(pagina.to_json(courts))
    print(f"{len(res)} jogos de futebol, {len(hoops)} de basquetebol, {len(ice)} de hóquei e "
          f"{len(courts)} de ténis em {args.saida}/")


def update_track(engine, folder, preds, games):
    """Guarda as previsões dos jogos que ainda não começaram e junta os resultados finais."""
    now = datetime.now(timezone.utc)
    store = registo.load(folder)
    for sport, df in preds.items():
        n = registo.record(store, df, sport, now)
        print(f"histórico: {n} previsões de {sport} guardadas")
    pending = [r["data"] for r in store.values() if r["estado"] == "pendente"]
    if pending:
        since = date.fromisoformat(min(pending)) - timedelta(days=2)
        with engine.connect() as conn:
            football = pd.read_sql(select(matches).where(matches.c.date >= since), conn)
        results = {"futebol": football}
        for sport, frames in games.items():
            frames = [f for f in frames if not f.empty]
            results[sport] = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        for sport, df in results.items():
            n = registo.resolve(store, df, sport, now)
            print(f"histórico: {n} resultados de {sport} juntados")
    registo.save(store, folder)
    return store


def tennis_predictions(today, collect=None):
    try:  # se a ESPN falhar, a página sai sem ténis em vez de não sair
        matches = tennis.fetch_matches(today, years=tenis.YEARS)
    except Exception as e:
        print(f"ténis: falhou a recolha ({e})")
        return pd.DataFrame()
    if collect is not None:
        collect.append(matches)
    preds = tenis.predict_upcoming(matches, today)
    if preds.empty:
        return preds
    return preds.sort_values(["date", "time"], na_position="last")


def run_tenis_backtest(args):
    matches = tennis.fetch_matches(years=args.anos)
    res = tenis.walk_forward(matches, start=args.desde)
    os.makedirs(os.path.dirname(args.saida) or ".", exist_ok=True)
    res.drop(columns=["mk"], errors="ignore").to_csv(args.saida, index=False)
    rows = [{"grupo": f"{k[0]} {k[1]}", **tenis.summary(g)} for k, g in res.groupby(["league", "surface"])]
    rows += [{"grupo": k, **tenis.summary(g)} for k, g in res.groupby("league")]
    rows.append({"grupo": "TOTAL", **tenis.summary(res)})
    pd.set_option("display.width", 200)
    print(pd.DataFrame(rows).set_index("grupo").round(3).T.to_string())
    print("\nCalibração (vencedor)")
    print(tenis.calibration(res).round(3).to_string(index=False))


def hockey_predictions(today, collect=None):
    out = []
    for league in hockey.HOCKEY_LEAGUES.values():
        try:  # uma liga que falhe não estraga a página
            games = hockey.fetch_games(league, today)
        except Exception as e:
            print(f"{league.key}: falhou a recolha ({e})")
            continue
        if collect is not None:
            collect.append(games)
        out.append(hoquei.predict_upcoming(games, today))
    out = [o for o in out if not o.empty]
    if not out:
        return pd.DataFrame()
    return pd.concat(out).sort_values(["date", "time"], na_position="last")


def basket_predictions(today, reserve_dir=None, collect=None):
    out = []
    for league in basket.BASKET_LEAGUES.values():
        try:  # uma liga que falhe não estraga a página
            # betexplorer: 2 épocas chegam para a janela do modelo e poupam pedidos ao site
            games = basket.fetch_games(league, today, seasons=2 if league.source == "betexplorer" else 3)
        except Exception as e:
            print(f"{league.key}: falhou a recolha ({e})")
            games = _reserve_games(reserve_dir, league.key)
            if games is None:
                continue
            print(f"{league.key}: a usar a cópia guardada ({len(games)} jogos)")
        if collect is not None:
            collect.append(games)
        out.append(basquete.predict_upcoming(games, today))
    out = [o for o in out if not o.empty]
    if not out:
        return pd.DataFrame()
    return pd.concat(out).sort_values(["date", "time"], na_position="last")


def _reserve_games(folder, key):
    """Cópia guardada pela tarefa diária (ramo 'dados'), para sites que recusam o GitHub."""
    path = os.path.join(folder, f"{key}.csv") if folder else None
    if not path or not os.path.exists(path):
        return None
    games = pd.read_csv(path, dtype={"time": str, "round": str})
    games["date"] = pd.to_datetime(games["date"]).dt.date
    return games


def run_basquete_dados(args):
    os.makedirs(args.saida, exist_ok=True)
    for key in args.ligas:
        games = basket.fetch_games(basket.BASKET_LEAGUES[key])
        games.to_csv(os.path.join(args.saida, f"{key}.csv"), index=False)
        print(f"{key}: {len(games)} jogos")


def run_basquete_backtest(args):
    rows = []
    for key in args.ligas:
        games = basket.fetch_games(basket.BASKET_LEAGUES[key], seasons=args.epocas)
        res = basquete.walk_forward(games)
        rows.append({"liga": key, **basquete.summary(res)})
    pd.set_option("display.width", 160)
    print(pd.DataFrame(rows).set_index("liga").round(3).to_string())



def run_hoquei_backtest(args):
    rows = []
    for key in args.ligas:
        games = hockey.fetch_games(hockey.HOCKEY_LEAGUES[key], seasons=args.epocas)
        rows.append({"liga": key, **hoquei.summary(hoquei.walk_forward(games))})
    pd.set_option("display.width", 160)
    print(pd.DataFrame(rows).set_index("liga").round(3).to_string())


if __name__ == "__main__":
    main()
