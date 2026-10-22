import argparse
import os
from datetime import date, timedelta

import pandas as pd
from sqlalchemy import select

from analise import backtest, basquete, calendario, pagina, previsoes
from analise.ingest import basket
from analise.db import get_engine, matches
from analise.ingest import football_data_uk
from analise.leagues import LEAGUES


def main(argv=None):
    parser = argparse.ArgumentParser(prog="analise")
    sub = parser.add_subparsers(dest="cmd", required=True)
    hist = sub.add_parser("historico", help="Descarrega o histórico do football-data.co.uk")
    hist.add_argument("--epocas", type=int, default=6)
    hist.add_argument("--ligas", nargs="*", default=list(LEAGUES), choices=list(LEAGUES))
    bt = sub.add_parser("backtest", help="Compara o Dixon-Coles com as odds de fecho sem margem")
    bt.add_argument("--ligas", nargs="*", default=list(LEAGUES), choices=list(LEAGUES))
    bt.add_argument("--meia-vida", type=float, default=180.0, help="dias")
    bt.add_argument("--janela", type=int, default=730, help="dias de histórico no treino")
    bt.add_argument("--desde", help="só avalia jogos a partir desta data (AAAA-MM-DD)")
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
    pag = sub.add_parser("pagina", help="Gera a página web com as previsões")
    pag.add_argument("--dias", type=int, help="só jogos nos próximos N dias (por omissão, todos)")
    pag.add_argument("--saida", default="site", help="pasta da página")
    args = parser.parse_args(argv)

    engine = get_engine()
    if args.cmd == "historico":
        for key in args.ligas:
            n = football_data_uk.load_history(engine, LEAGUES[key], args.epocas)
            print(f"{key}: {n} jogos")
    elif args.cmd == "backtest":
        run_backtest(engine, args)
    elif args.cmd == "previsoes":
        run_previsoes(engine, args)
    elif args.cmd == "basquete-backtest":
        run_basquete_backtest(args)
    elif args.cmd == "pagina":
        run_pagina(engine, args)


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
            df, half_life_days=args.meia_vida, window_days=args.janela, start=args.desde)
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
    for key, games in fixtures.groupby("league"):
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


def run_pagina(engine, args):
    today = date.today()
    fixtures = calendario.upcoming(engine, list(LEAGUES.values()), today)
    if args.dias:
        fixtures = fixtures[fixtures["date"] <= today + timedelta(days=args.dias)]
    res = predict_fixtures(engine, fixtures, 180.0, 730, today)
    hoops = basket_predictions(today)
    if args.dias and not hoops.empty:
        hoops = hoops[hoops["date"] <= today + timedelta(days=args.dias)]
    os.makedirs(args.saida, exist_ok=True)
    with open(os.path.join(args.saida, "index.html"), "w", encoding="utf-8") as f:
        f.write(pagina.render(res, hoops))
    with open(os.path.join(args.saida, "previsoes.json"), "w", encoding="utf-8") as f:
        f.write(pagina.to_json(res))
    with open(os.path.join(args.saida, "basquetebol.json"), "w", encoding="utf-8") as f:
        f.write(pagina.to_json(hoops))
    print(f"{len(res)} jogos de futebol e {len(hoops)} de basquetebol em {args.saida}/index.html")


def basket_predictions(today):
    out = []
    for league in basket.BASKET_LEAGUES.values():
        try:  # uma liga que falhe não estraga a página
            games = basket.fetch_games(league, today)
        except Exception as e:
            print(f"{league.key}: falhou a recolha ({e})")
            continue
        out.append(basquete.predict_upcoming(games, today))
    out = [o for o in out if not o.empty]
    if not out:
        return pd.DataFrame()
    return pd.concat(out).sort_values(["date", "time"], na_position="last")


def run_basquete_backtest(args):
    rows = []
    for key in args.ligas:
        games = basket.fetch_games(basket.BASKET_LEAGUES[key], seasons=args.epocas)
        res = basquete.walk_forward(games)
        rows.append({"liga": key, **basquete.summary(res)})
    pd.set_option("display.width", 160)
    print(pd.DataFrame(rows).set_index("liga").round(3).to_string())


if __name__ == "__main__":
    main()
