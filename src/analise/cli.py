import argparse

from analise.db import get_engine
from analise.ingest import football_data_uk
from analise.leagues import LEAGUES


def main(argv=None):
    parser = argparse.ArgumentParser(prog="analise")
    sub = parser.add_subparsers(dest="cmd", required=True)
    hist = sub.add_parser("historico", help="Descarrega o histórico do football-data.co.uk")
    hist.add_argument("--epocas", type=int, default=6)
    hist.add_argument("--ligas", nargs="*", default=list(LEAGUES), choices=list(LEAGUES))
    args = parser.parse_args(argv)

    engine = get_engine()
    if args.cmd == "historico":
        for key in args.ligas:
            n = football_data_uk.load_history(engine, LEAGUES[key], args.epocas)
            print(f"{key}: {n} jogos")


if __name__ == "__main__":
    main()
