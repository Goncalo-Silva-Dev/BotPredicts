import argparse
import os

import pandas as pd
from sqlalchemy import select

from analise import backtest
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
    args = parser.parse_args(argv)

    engine = get_engine()
    if args.cmd == "historico":
        for key in args.ligas:
            n = football_data_uk.load_history(engine, LEAGUES[key], args.epocas)
            print(f"{key}: {n} jogos")
    elif args.cmd == "backtest":
        run_backtest(engine, args)


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


if __name__ == "__main__":
    main()
