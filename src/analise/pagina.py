"""Site estático com as previsões (para o GitHub Pages).

index.html é o aviso inicial; cada liga tem a sua página em futebol/, basquetebol/ ou hoquei/,
com um menu comum no topo.
"""
import html
import re
import json
from zoneinfo import ZoneInfo
import math
from datetime import datetime, timezone

import pandas as pd

from analise import registo
from analise.ingest.basket import BASKET_LEAGUES
from analise.ingest.hockey import HOCKEY_LEAGUES
from analise.leagues import LEAGUES

LISBON = ZoneInfo("Europe/Lisbon")

FLAGS = {"POR": "🇵🇹", "PO2": "🇵🇹", "U23": "🇵🇹", "TUR": "🇹🇷", "NED": "🇳🇱", "BEL": "🇧🇪", "ESP": "🇪🇸", "ENG": "🏴", "FRA": "🇫🇷", "ITA": "🇮🇹", "USA": "🇺🇸", "BRA": "🇧🇷",
         "NBA": "🇺🇸", "NHL": "🇺🇸", "NBL": "🇦🇺", "BLG": "🇯🇵", "ABA": "🇷🇸", "KBL": "🇰🇷", "SBL": "🇸🇪", "KOR": "🇫🇮", "CBA": "🇨🇳", "PBA": "🇵🇭", "EUR": "🇪🇺", "EUC": "🇪🇺", "NBB": "🇧🇷", "ACB": "🇪🇸", "LBA": "🇮🇹", "LPB": "🇵🇹", "LNB": "🇫🇷"}
WEEKDAYS = ["seg", "ter", "qua", "qui", "sex", "sáb", "dom"]

CSS = """
:root { --bg:#f7f7f5; --card:#fff; --text:#1d1d1b; --muted:#6b6b66; --line:#e4e3de;
  --home:#2f6fdb; --draw:#9a9a92; --away:#d9563b; }
@media (prefers-color-scheme: dark) { :root { --bg:#161615; --card:#21211f; --text:#ecebe6;
  --muted:#a3a29b; --line:#34342f; --home:#6b9cf0; --draw:#77776f; --away:#ef7d63; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--text);
  font:15px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif; }
main { max-width:860px; margin:0 auto; padding:24px 16px 48px; }
h1 { font-size:24px; margin:0 0 4px; }
h2 { font-size:17px; margin:28px 0 10px; }
.sub { color:var(--muted); margin:0 0 8px; }
.note { color:var(--muted); font-size:13px; }
.game { background:var(--card); border:1px solid var(--line); border-radius:10px;
  padding:12px 14px; margin-bottom:10px; }
.top { display:flex; justify-content:space-between; gap:12px; flex-wrap:wrap; }
.teams { font-weight:600; }
.when { color:var(--muted); font-size:13px; white-space:nowrap; }
.bar { display:flex; height:26px; border-radius:6px; overflow:hidden; margin:10px 0 6px;
  font-size:13px; font-weight:600; color:#fff; }
.bar span { display:flex; align-items:center; justify-content:center; min-width:34px; }
.h { background:var(--home); } .d { background:var(--draw); } .a { background:var(--away); }
.facts { display:flex; flex-wrap:wrap; gap:4px 16px; font-size:13px; color:var(--muted); }
.facts b { color:var(--text); font-weight:600; }
details { margin-bottom:8px; }
summary { cursor:pointer; padding:8px 2px; font-weight:600; }
summary span { color:var(--muted); font-weight:400; font-size:13px; margin-left:6px; }
details.per { margin:8px 0 0; }
details.per summary { padding:2px 0; font-size:13px; color:var(--home); }
.per .wrap { overflow-x:auto; }
.per table { border-collapse:collapse; font-size:13px; width:100%; margin-top:4px; }
.per th, .per td { padding:4px 8px; text-align:right; border-bottom:1px solid var(--line);
  white-space:nowrap; }
.per th { color:var(--muted); font-weight:500; }
.per th:first-child, .per td:first-child { text-align:left; }
.mk { display:grid; grid-template-columns:repeat(auto-fill, minmax(190px, 1fr)); gap:8px 20px;
  margin-top:6px; font-size:13px; }
.mk h4 { margin:4px 0 2px; font-size:12px; font-weight:600; color:var(--muted); }
.mk div div { display:flex; justify-content:space-between; border-bottom:1px solid var(--line);
  padding:2px 0; }
header { background:var(--card); border-bottom:1px solid var(--line); position:sticky; top:0;
  z-index:5; }
header nav { max-width:860px; margin:0 auto; padding:0 16px; display:flex; gap:4px;
  align-items:center; flex-wrap:wrap; }
header a, header summary { color:var(--text); text-decoration:none; font-weight:600;
  padding:12px 10px; display:block; }
header a.brand { padding-left:0; margin-right:auto; }
header details { position:relative; margin:0; }
header summary { list-style:none; }
header summary::-webkit-details-marker { display:none; }
header summary::after { content:" ▾"; color:var(--muted); }
header .menu { position:absolute; right:0; top:100%; background:var(--card); min-width:240px;
  border:1px solid var(--line); border-radius:10px; padding:6px; box-shadow:0 8px 24px #0002; }
header .menu a { font-weight:500; padding:8px 10px; border-radius:6px; }
header .menu a:hover, header .menu a.on { background:var(--bg); }
.grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(220px,1fr)); gap:10px;
  margin:10px 0 24px; }
.grid a { background:var(--card); border:1px solid var(--line); border-radius:10px;
  padding:12px 14px; color:var(--text); text-decoration:none; font-weight:600; }
.grid a span { display:block; color:var(--muted); font-weight:400; font-size:13px; }
.warn { background:var(--card); border:1px solid var(--line); border-left:4px solid var(--away);
  border-radius:10px; padding:14px 16px; margin:16px 0; }
.warn li { margin:6px 0; }
.empty { background:var(--card); border:1px dashed var(--line); border-radius:10px; padding:16px; }
.tbl { overflow-x:auto; background:var(--card); border:1px solid var(--line); border-radius:10px;
  margin:8px 0 16px; }
.tbl table { border-collapse:collapse; font-size:13px; width:100%; }
.tbl th, .tbl td { padding:6px 10px; text-align:right; border-bottom:1px solid var(--line);
  white-space:nowrap; }
.tbl tr:last-child td { border-bottom:0; }
.tbl th { color:var(--muted); font-weight:500; }
.tbl th:first-child, .tbl td:first-child { text-align:left; }
.tbl.games th, .tbl.games td { text-align:left; }
.note a { color:var(--home); }
.ok { color:#2e9e5b; font-weight:600; } .ko { color:var(--away); font-weight:600; }
"""


def _pct(v) -> str:
    return "–" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v * 100:.0f}%"


def _n(v: float, digits: int = 1) -> str:
    """Número com vírgula decimal, como se escreve em Portugal."""
    return f"{v:.{digits}f}".replace(".", ",")


def _line(x: float) -> str:
    return _n(x, 1)


def _when(d, t) -> str:
    d = pd.Timestamp(d)
    hour = f" {t}" if isinstance(t, str) and t else ""
    return f"{WEEKDAYS[d.weekday()]} {d.day:02d}/{d.month:02d}{hour}"


# Nomes como se escrevem em Portugal e no Brasil (as fontes inglesas tiram os acentos)
TEAM_NAMES = {
    "Sp Lisbon": "Sporting", "Sp Braga": "Braga", "Guimaraes": "Vitória SC",
    "Famalicao": "Famalicão", "Maritimo": "Marítimo", "Academico Viseu": "Académico de Viseu",
    "Estrela": "Estrela da Amadora", "Pacos Ferreira": "Paços de Ferreira",
    "Sao Paulo": "São Paulo", "Gremio": "Grêmio", "Vitoria": "Vitória", "Atletico-MG": "Atlético-MG",
    "Athletico-PR": "Athletico-PR", "Flamengo RJ": "Flamengo", "Botafogo RJ": "Botafogo",
    "Chapecoense-SC": "Chapecoense", "Cuiaba": "Cuiabá", "Goias": "Goiás", "Ceara": "Ceará",
    "America MG": "América-MG", "Avai": "Avaí",
    "Academica": "Académica", "Leixoes": "Leixões", "Lusitania FC": "Lusitânia",
    "SC Farense": "Farense", "Os Belenenses": "Belenenses", "Sporting CP B": "Sporting B",
    "Sporting CP": "Sporting",
}


def _team(name: str) -> str:
    if name.endswith(" U23"):  # equipas da Liga Revelação
        base = name.removesuffix(" U23")
        return html.escape(f"{TEAM_NAMES.get(base, base)} Sub-23")
    return html.escape(TEAM_NAMES.get(name, name))


def _game(g) -> str:
    teams = f"{_team(g['home'])} – {_team(g['away'])}"
    head = f'<div class="top"><span class="teams">{teams}</span>' \
           f'<span class="when">{_when(g["date"], g.get("time"))}</span></div>'
    if pd.isna(g.get("casa")):
        return f'<div class="game">{head}<p class="note">Sem previsão: uma das equipas ' \
               f'tem poucos jogos no histórico.</p></div>'
    bar = "".join(
        f'<span class="{c}" style="flex:{g[k]:.4f}" title="{label}">{_pct(g[k])}</span>'
        for c, k, label in [("h", "casa", "Vitória da casa"), ("d", "empate", "Empate"),
                            ("a", "fora", "Vitória de fora")])
    facts = [
        f"Golos esperados <b>{_n(g['golos_casa'])} – {_n(g['golos_fora'])}</b>",
        f"Mais de 2,5 <b>{_pct(g['mais_2_5'])}</b>",
        f"Ambas marcam <b>{_pct(g['ambas_marcam'])}</b>",
        f"Resultados <b>{html.escape(g['resultados'])}</b>",
    ]
    if pd.notna(g.get("mercado_casa")):
        facts.append("Casas de apostas <b>" + " / ".join(
            _pct(g[k]) for k in ("mercado_casa", "mercado_empate", "mercado_fora")) + "</b>")
    spans = "".join(f"<span>{f}</span>" for f in facts)
    return f'<div class="game">{head}<div class="bar">{bar}</div><div class="facts">{spans}</div>' \
           f'{_halves_table(g)}{_markets(g)}</div>'


def _markets(g) -> str:
    """Mais apostas do jogo inteiro, em blocos pequenos."""
    if pd.isna(g.get("dupla_1x")):
        return ""
    home, away = _team(g["home"]), _team(g["away"])
    blocks = [
        ("Dupla hipótese", [("Casa ou empate (1X)", "dupla_1x"), ("Casa ou fora (12)", "dupla_12"),
                            ("Empate ou fora (X2)", "dupla_x2")]),
        ("Empate anula aposta", [(home, g["dnb_casa"]), (away, 1 - g["dnb_casa"])]),
        ("Total de golos", [(f"Mais de {_line(x)}", f"mais_{x}") for x in (0.5, 1.5, 2.5, 3.5, 4.5)]),
        ("Número de golos", [("0", "golos_0"), ("1", "golos_1"), ("2", "golos_2"),
                             ("3", "golos_3"), ("4 ou mais", "golos_4+")]),
        (f"Golos de {home}", [(f"Mais de {_line(x)}", f"casa_mais_{x}") for x in (0.5, 1.5, 2.5)]),
        (f"Golos de {away}", [(f"Mais de {_line(x)}", f"fora_mais_{x}") for x in (0.5, 1.5, 2.5)]),
        ("Handicap -1 casa", [(f"{home} -1", "hcp_casa_-1"), ("Empate (-1)", "hcp_empate_-1"),
                              (f"{away} +1", "hcp_fora_+1")]),
        ("Handicap +1 casa", [(f"{home} +1", "hcp_casa_+1"), ("Empate (+1)", "hcp_empate_+1"),
                              (f"{away} -1", "hcp_fora_-1")]),
        ("Ganha sem sofrer", [(home, "casa_sem_sofrer"), (away, "fora_sem_sofrer")]),
    ]
    htft = g.get("intervalo_final")
    if isinstance(htft, dict):
        names = {"1": "Casa", "X": "Empate", "2": "Fora"}
        best = sorted(htft.items(), key=lambda kv: -kv[1])
        blocks.append(("Intervalo / final", [
            (f"{names[k[0]]} / {names[k[2]]}", v) for k, v in best]))
    html_blocks = []
    for title, rows in blocks:
        cells = "".join(f"<div><span>{label}</span><b>{_pct(g[v] if isinstance(v, str) else v)}</b></div>"
                        for label, v in rows)
        html_blocks.append(f"<div><h4>{title}</h4>{cells}</div>")
    return f'<details class="per"><summary>Mais apostas</summary><div class="mk">' \
           f'{"".join(html_blocks)}</div></details>'


def _per_table(head: list[str], rows: list[list[str]], title: str = "Por período") -> str:
    th = "".join(f"<th>{h}</th>" for h in head)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<details class="per"><summary>{title}</summary><div class="wrap"><table>' \
           f'<tr>{th}</tr>{body}</table></div></details>'


def _halves_table(g) -> str:
    if pd.isna(g.get("p1_casa")):
        return ""
    rows = [[name, _pct(g[f"{k}_casa"]), _pct(g[f"{k}_empate"]), _pct(g[f"{k}_fora"]),
             _n(g[f"{k}_golos"]), _pct(g[f"{k}_mais_0_5"]), _pct(g[f"{k}_mais_1_5"])]
            for name, k in (("1.ª parte", "p1"), ("2.ª parte", "p2"))]
    return _per_table(["", "Casa", "Empate", "Fora", "Golos esperados", "Mais de 0,5", "Mais de 1,5"], rows,
                      "Por parte")


def _periods_table(g) -> str:
    periods = g.get("periodos")
    if not isinstance(periods, list):
        return ""
    rows = [[p["periodo"], _pct(p["casa"]), _pct(p["empate"]), _pct(p["fora"]),
             f"{p['pontos_casa']:.0f} – {p['pontos_fora']:.0f}",
             f"{p['total']:.0f} ({p['total_min']:.0f} a {p['total_max']:.0f})"] for p in periods]
    return _per_table(["", "Casa", "Empate", "Fora", "Pontos", "Total (80%)"], rows)


def _basket_game(g) -> str:
    teams = f"{html.escape(g['home'])} – {html.escape(g['away'])}"
    head = f'<div class="top"><span class="teams">{teams}</span>' \
           f'<span class="when">{_when(g["date"], g.get("time"))}</span></div>'
    if pd.isna(g.get("casa")):
        return f'<div class="game">{head}<p class="note">Sem previsão: uma das equipas ' \
               f'tem poucos jogos no histórico.</p></div>'
    bar = "".join(
        f'<span class="{c}" style="flex:{g[k]:.4f}" title="{label}">{_pct(g[k])}</span>'
        for c, k, label in [("h", "casa", "Vitória da casa"), ("a", "fora", "Vitória de fora")])
    fav = g["home"] if g["diferenca"] >= 0 else g["away"]
    facts = [
        f"Pontos esperados <b>{g['pontos_casa']:.0f} – {g['pontos_fora']:.0f}</b>",
        f"Diferença <b>{html.escape(fav)} por {_n(abs(g['diferenca']))}</b>",
        f"Total <b>{g['total']:.0f}</b> (80%: {g['total_min']:.0f} a {g['total_max']:.0f})",
    ]
    spans = "".join(f"<span>{f}</span>" for f in facts)
    return f'<div class="game">{head}<div class="bar">{bar}</div><div class="facts">{spans}</div>' \
           f'{_periods_table(g)}{_basket_markets(g)}</div>'


def _signed(x: float) -> str:
    return ("+" if x > 0 else "−") + _line(abs(x))


def _basket_markets(g) -> str:
    """Mais apostas de basquetebol (jogo inteiro, prolongamento incluído)."""
    mk = g.get("mercados")
    if not isinstance(mk, dict):
        return ""
    home, away = html.escape(g["home"]), html.escape(g["away"])
    blocks = [
        ("Handicap", [(f"{home} {_signed(-line)}", p) for line, p in mk["handicap"]]),
        ("Handicap (equipa de fora)", [(f"{away} {_signed(line)}", 1 - p) for line, p in mk["handicap"]]),
        ("Total de pontos", [(f"Mais de {_line(line)}", p) for line, p in mk["total"]]),
        (f"Pontos de {home}", [(f"Mais de {_line(line)}", p) for line, p in mk["pontos_casa"]]),
        (f"Pontos de {away}", [(f"Mais de {_line(line)}", p) for line, p in mk["pontos_fora"]]),
        (f"Margem de vitória: {home}", [(f"{b} pontos", p) for side, b, p in mk["margem"] if side == "casa"]),
        (f"Margem de vitória: {away}", [(f"{b} pontos", p) for side, b, p in mk["margem"] if side == "fora"]),
    ]
    names = {"1": "Casa", "X": "Empate", "2": "Fora"}
    best = sorted(mk["intervalo_final"].items(), key=lambda kv: -kv[1])
    blocks.append(("Intervalo / final", [(f"{names[k[0]]} / {names[k[2]]}", v) for k, v in best]))
    html_blocks = []
    for title, rows in blocks:
        cells = "".join(f"<div><span>{label}</span><b>{_pct(v)}</b></div>" for label, v in rows)
        html_blocks.append(f"<div><h4>{title}</h4>{cells}</div>")
    return f'<details class="per"><summary>Mais apostas</summary><div class="mk">' \
           f'{"".join(html_blocks)}</div></details>'


def _hockey_game(g) -> str:
    teams = f"{_team(g['home'])} – {_team(g['away'])}"
    head = f'<div class="top"><span class="teams">{teams}</span>' \
           f'<span class="when">{_when(g["date"], g.get("time"))}</span></div>'
    if pd.isna(g.get("casa")):
        return f'<div class="game">{head}<p class="note">Sem previsão: uma das equipas ' \
               f'tem poucos jogos no histórico.</p></div>'
    bar = "".join(
        f'<span class="{c}" style="flex:{g[k]:.4f}" title="{label}">{_pct(g[k])}</span>'
        for c, k, label in [("h", "casa", "Vitória da casa (com prolongamento e penáltis)"),
                            ("a", "fora", "Vitória de fora (com prolongamento e penáltis)")])
    mk = g["mercados"]
    facts = [
        f"Golos esperados <b>{_n(g['golos_casa'])} – {_n(g['golos_fora'])}</b>",
        "60 minutos <b>" + " / ".join(_pct(g[k]) for k in ("reg_casa", "reg_empate", "reg_fora")) + "</b>",
        f"Vai a prolongamento <b>{_pct(mk['prolongamento'])}</b>",
        "Resultados (60') <b>" + ", ".join(s for s, _ in mk["resultados"][:3]) + "</b>",
    ]
    spans = "".join(f"<span>{f}</span>" for f in facts)
    return f'<div class="game">{head}<div class="bar">{bar}</div><div class="facts">{spans}</div>' \
           f'{_hockey_periods(g)}{_hockey_markets(g)}</div>'


def _hockey_periods(g) -> str:
    periods = g.get("periodos")
    if not isinstance(periods, list):
        return ""
    rows = [[p["periodo"], _pct(p["casa"]), _pct(p["empate"]), _pct(p["fora"]),
             f"{_n(p['golos_casa'])} – {_n(p['golos_fora'])}", _pct(p["mais_0_5"]),
             _pct(p["mais_1_5"]), _pct(p["mais_2_5"])] for p in periods]
    most = g.get("periodo_mais_golos")
    if isinstance(most, dict):
        rows.append(["Período com mais golos"] + [""] * 7)
        rows += [[f"&nbsp;&nbsp;{html.escape(k)}", _pct(v)] + [""] * 6 for k, v in most.items()]
    return _per_table(["", "Casa", "Empate", "Fora", "Golos", "Mais de 0,5", "Mais de 1,5",
                       "Mais de 2,5"], rows)


def _hockey_markets(g) -> str:
    """Mais apostas de hóquei; as linhas de golos contam com o prolongamento e os penáltis."""
    mk = g.get("mercados")
    if not isinstance(mk, dict):
        return ""
    home, away = _team(g["home"]), _team(g["away"])
    names = {"casa": home, "fora": away}
    blocks = [
        ("Resultado aos 60 minutos", [(home, g["reg_casa"]), ("Empate", g["reg_empate"]), (away, g["reg_fora"])]),
        ("Dupla hipótese (60 minutos)", [("Casa ou empate (1X)", mk["dupla"]["1X"]),
                                         ("Casa ou fora (12)", mk["dupla"]["12"]),
                                         ("Empate ou fora (X2)", mk["dupla"]["X2"])]),
        ("Puck line", [(f"{names[side]} {_signed(line)}", p) for side, line, p in mk["puck_line"]]),
        ("Total de golos", [(f"Mais de {_line(line)}", p) for line, p in mk["total"]]),
        (f"Golos de {home}", [(f"Mais de {_line(line)}", p) for line, p in mk["golos_casa"]]),
        (f"Golos de {away}", [(f"Mais de {_line(line)}", p) for line, p in mk["golos_fora"]]),
        ("Outras", [("Ambas marcam (60 minutos)", mk["ambas_marcam"]),
                    ("Há prolongamento", mk["prolongamento"]), ("Não há prolongamento", 1 - mk["prolongamento"])]),
        ("Resultado exato aos 60 minutos", mk["resultados"]),
    ]
    html_blocks = []
    for title, rows in blocks:
        cells = "".join(f"<div><span>{label}</span><b>{_pct(v)}</b></div>" for label, v in rows)
        html_blocks.append(f"<div><h4>{title}</h4>{cells}</div>")
    return f'<details class="per"><summary>Mais apostas</summary><div class="mk">' \
           f'{"".join(html_blocks)}</div></details>'


def _round_label(r, dates: pd.Series) -> str:
    if isinstance(r, str) and r:
        label = re.sub(r"^(Matchday|Round)\s*", "Jornada ", r)
    else:
        monday = pd.Timestamp(dates.min()) - pd.Timedelta(days=pd.Timestamp(dates.min()).weekday())
        label = f"Semana de {monday:%d/%m}"
    first, last = pd.Timestamp(dates.min()), pd.Timestamp(dates.max())
    span = f"{first:%d/%m}" if first == last else f"{first:%d/%m} a {last:%d/%m}"
    return f"{html.escape(label)}<span>{span} · {len(dates)} jogos</span>"


def _league_rounds(games: pd.DataFrame, card=_game) -> str:
    """Um bloco que abre e fecha por jornada; a primeira vem aberta."""
    games = games.copy()
    games["_round"] = games["round"] if "round" in games else None
    week = pd.to_datetime(games["date"]).dt.to_period("W-SUN").astype(str)
    games["_group"] = games["_round"].fillna("semana " + week)
    order = games.groupby("_group", sort=False)["date"].min().sort_values().index
    blocks = []
    for i, key in enumerate(order):
        g = games[games["_group"] == key]
        cards = "".join(card(x) for x in g.to_dict("records"))
        is_open = " open" if i == 0 else ""
        blocks.append(f"<details{is_open}><summary>{_round_label(g['_round'].iloc[0], g['date'])}"
                      f"</summary>{cards}</details>")
    return "".join(blocks)


FOOTBALL_NOTE = """Barras: vitória da casa · empate · vitória de fora. Modelo Dixon-Coles com os
últimos 2 anos de resultados. As odds das casas de apostas aparecem quando já existem (perto do
jogo). No teste com 11 984 jogos passados o modelo acertou bem nas percentagens, mas ficou um
pouco atrás das odds de fecho das casas de apostas. Em "Por parte": resultado ao intervalo e
golos em cada parte, com a parte dos golos que cada liga costuma marcar na 1.ª parte; ao
intervalo o favorito acerta em cerca de 45% dos jogos, porque o empate é muito comum. Em "Mais
apostas": dupla hipótese, totais, golos por equipa, handicap e intervalo/final, todos tirados do
mesmo modelo; no teste com 2500 jogos da última época as percentagens bateram certo com o que
aconteceu (por exemplo, mais de 3,5 golos: 29,5% previsto, 29,9% real)."""

BASKET_NOTE = """Barras: vitória da casa · vitória de fora. Forças de ataque e defesa de cada
equipa com os resultados das últimas épocas (os jogos recentes pesam mais). A incerteza foi
acertada com 7875 jogos passados das 12 ligas, cada um previsto só com os jogos anteriores, para
que o intervalo de 80% do total apanhe de facto cerca de 80% dos jogos. Em "Por período":
quartos e partes, calibrados com 3500 jogos com pontos por quarto. Um período é muito mais
incerto que o jogo: no teste o favorito ganhou o quarto em 52% a 55% das vezes e a parte em
56% a 57%. Em "Mais apostas": handicaps, total de pontos, pontos de cada equipa, margem de
vitória e intervalo/final; no teste com 3513 jogos da NBA, EuroLeague e EuroCup as percentagens
bateram certo com o que aconteceu (por exemplo, handicaps a 66%: ganharam 67%)."""

# Notas por liga de futebol (teste com jogos passados de cada liga, prevendo só com o passado)
NO_HT = (" Esta fonte não tem o resultado ao intervalo, por isso \"Por parte\" usa a parte dos golos "
         "que as outras ligas marcam na 1.ª parte (cerca de 44%).")
FOOTBALL_LEAGUE_NOTES = {
    "TUR": " Nesta liga, no teste com 384 jogos, o favorito do modelo ganhou 49% das vezes.",
    "NED": " Nesta liga, no teste com 363 jogos, o favorito do modelo ganhou 51% das vezes.",
    "BEL": " Nesta liga, no teste com 375 jogos, o favorito do modelo ganhou 48% das vezes.",
    "U23": " Nesta liga, no teste com 250 jogos, o favorito do modelo ganhou 50% das vezes." + NO_HT,
    "PO2": (" Atenção: na Liga Portugal 2 o modelo não funcionou bem no teste com 341 jogos (o "
            "favorito ganhou só 41% das vezes e as percentagens foram piores do que usar as médias "
            "da liga). Aqui as percentagens valem pouco: usa-as com muita cautela." + NO_HT),
}

HOCKEY_NOTE = """Barras: vitória da casa · vitória de fora, já com prolongamento e penáltis. Forças
de ataque e defesa de cada equipa com os golos dos 60 minutos das últimas épocas (os jogos
recentes pesam mais), mais o fim de jogo do hóquei: quem perde tira o guarda-redes, o que dá
golos na baliza deserta ou empates tardios. No teste com 2766 jogos passados da NHL, cada um
previsto só com os jogos anteriores, o favorito ganhou 55% das vezes (a NHL é das ligas mais
equilibradas e imprevisíveis), o empate aos 60 minutos foi previsto em 23,5% dos jogos e
aconteceu em 23,1%, e a puck line -1,5 da casa foi prevista a 31,6% e ganhou 32,4%. Nas casas de
apostas, o total de golos e a puck line contam com o prolongamento e os penáltis (quem ganha
fica com mais 1 golo), e é assim que estão aqui. O "Resultado aos 60 minutos" e o resultado
exato são só do tempo regulamentar. Em "Por período" as percentagens ficam perto das médias da
liga: um período tem poucos golos e é muito difícil de prever."""

# Percentagem de vencedores certos no teste com jogos passados (basquete-backtest)
BASKET_ACCURACY = {"NBA": 66, "EUR": 64, "EUC": 65, "ACB": 69, "LBA": 71, "LPB": 75, "NBB": 73, "LNB": 71, "NBL": 59, "BLG": 67, "ABA": 80, "KBL": 58, "SBL": 73, "KOR": 57, "CBA": 71, "PBA": 66}


def _page(title: str, body: str, root: str, current: str, updated: datetime) -> str:
    """Página completa com o menu; root é o caminho relativo até à raiz do site."""
    def menu(folder, leagues):
        return "".join(
            f'<a href="{root}{folder}/{k.lower()}.html"{" class=on" if current == k else ""}>'
            f"{FLAGS.get(k, '')} {html.escape(l.name)}</a>" for k, l in leagues.items())
    return f"""<!doctype html>
<html lang="pt-PT">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)} · BotPredicts</title>
<style>{CSS}</style>
</head>
<body>
<header><nav>
<a class="brand" href="{root}index.html">BotPredicts</a>
<details><summary>⚽ Futebol</summary><div class="menu">{menu("futebol", LEAGUES)}</div></details>
<details><summary>🏀 Basquetebol</summary><div class="menu">{menu("basquetebol", BASKET_LEAGUES)}</div></details>
<details><summary>🏒 Hóquei</summary><div class="menu">{menu("hoquei", HOCKEY_LEAGUES)}</div></details>
<a href="{root}historico.html">📊 Histórico</a>
</nav></header>
<main>
{body}
<p class="note">Atualizado {updated.astimezone(LISBON):%d/%m/%Y %H:%M} (hora de Portugal) · Dados: football-data.co.uk,
openfootball, ESPN (NBA e NHL), Euroleague Basketball e sites oficiais das ligas.</p>
</main>
<script>
// Fecha os outros menus quando se abre um
document.querySelectorAll("header details").forEach(d => d.addEventListener("toggle", () => {{
  if (d.open) document.querySelectorAll("header details").forEach(o => {{ if (o !== d) o.open = false; }});
}}));
</script>
</body>
</html>
"""


def _league_page(key, league, games, card, note, root, updated) -> str:
    if games.empty:
        content = '<div class="empty">Sem jogos marcados nesta liga por agora.</div>'
    else:
        content = _league_rounds(games, card)
    body = (f"<h1>{FLAGS.get(key, '')} {html.escape(league.name)}</h1>"
            f'<p class="note">{note}</p>{content}')
    return _page(league.name, body, root, key, updated)


def _index(preds, basket, hockey, updated) -> str:
    def cards(folder, leagues, df):
        out = []
        for k, l in leagues.items():
            n = int((df["league"] == k).sum()) if not df.empty else 0
            out.append(f'<a href="{folder}/{k.lower()}.html">{FLAGS.get(k, "")} '
                       f"{html.escape(l.name)}<span>{n} {"jogo" if n == 1 else "jogos"} com previsão</span></a>")
        return "".join(out)
    body = f"""<h1>Previsões de futebol, basquetebol e hóquei no gelo</h1>
<div class="warn">
<strong>Antes de usares estas previsões</strong>
<ul>
<li>São probabilidades calculadas por modelos estatísticos a partir de resultados passados.
Não são certezas nem dicas garantidas: um jogo com 70% para uma equipa perde-se 3 vezes em 10.</li>
<li>No teste com milhares de jogos passados os modelos acertaram bem nas percentagens, mas no
futebol ficaram um pouco atrás das odds das casas de apostas. Isto quer dizer que, por si só,
não dão lucro a apostar.</li>
<li>Os modelos só conhecem resultados. Não sabem de lesões, castigos, rotações nem mudanças de
treinador, e os jogos mais distantes usam a forma atual das equipas.</li>
<li>Este site é informativo e não tem ligação às ligas, aos clubes nem a casas de apostas.</li>
<li>Apostar só a partir dos 18 anos e só o que podes perder. Se o jogo deixar de ser diversão,
procura ajuda: em Portugal, o SICAD apoia casos de jogo excessivo.</li>
</ul>
</div>
<h2>⚽ Futebol</h2><div class="grid">{cards("futebol", LEAGUES, preds)}</div>
<h2>🏀 Basquetebol</h2><div class="grid">{cards("basquetebol", BASKET_LEAGUES, basket)}</div>
<h2>🏒 Hóquei no gelo</h2><div class="grid">{cards("hoquei", HOCKEY_LEAGUES, hockey)}</div>
<h2>📊 Histórico</h2><div class="grid"><a href="historico.html">Acertos das previsões publicadas
<span>todas guardadas antes de cada jogo e comparadas com o resultado final</span></a></div>"""
    return _page("Início", body, "", "", updated)


SPORTS = {"futebol": ("⚽ Futebol", LEAGUES), "basquetebol": ("🏀 Basquetebol", BASKET_LEAGUES),
          "hoquei": ("🏒 Hóquei no gelo", HOCKEY_LEAGUES)}
STORE_URL = "https://github.com/Goncalo-Silva-Dev/BotPredicts/commits/historico"

TRACK_NOTE = f"""Duas vezes por dia, quando o site atualiza, fica guardada a previsão de cada jogo
dos próximos 7 dias. Quando o jogo começa a previsão fica congelada: conta a última feita antes do
início, e só entram jogos depois de terminados. Jogos sem hora marcada só contam com previsões da
véspera ou antes. As gravações ficam no ramo <a href="{STORE_URL}">historico</a> do GitHub, com data
e hora, para qualquer pessoa confirmar que nada foi mexido depois do jogo."""

TRACK_HELP = """<b>Acerto</b>: quantas vezes aconteceu o lado que o modelo achou mais provável.
<b>Confiança</b>: a percentagem média que o modelo deu a esse lado; num modelo bem afinado as duas
ficam perto (quem diz 70% deve acertar cerca de 7 em 10). <b>Pontuação</b> (log loss): mede todas
as percentagens e não só o favorito; quanto mais baixa melhor. <b>À sorte</b> é a pontuação de quem
dá a mesma percentagem a todos os resultados (1,10 nos mercados de 3 resultados, 0,69 nos de
sim/não). Com poucas dezenas de jogos os números ainda mudam muito."""


def _tbl(head: list[str], rows: list[list[str]], cls: str = "") -> str:
    th = "".join(f"<th>{h}</th>" for h in head)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tbl {cls}"><table><tr>{th}</tr>{body}</table></div>'


def _stat_rows(t: pd.DataFrame, label) -> list[list[str]]:
    return [[label(r), f"{r.jogos}", _pct(r.acerto), _pct(r.confianca), _n(r.logloss, 3),
             _n(r.referencia, 2)] for r in t.itertuples()]


def _pick_label(rec) -> tuple[str, bool | None]:
    """Favorito do mercado principal e se acertou."""
    p, res = rec["p"], rec["resultado"]
    if rec["desporto"] == "futebol":
        names = [_team(rec["casa"]), "Empate", _team(rec["fora"])]
        i = max(range(3), key=lambda k: p["1x2"][k])
        real = 0 if res["casa"] > res["fora"] else (1 if res["casa"] == res["fora"] else 2)
        return f"{names[i]} {_pct(p['1x2'][i])}", i == real
    home = p["vencedor"] >= 0.5
    name = _team(rec["casa"] if home else rec["fora"])
    return f"{name} {_pct(max(p['vencedor'], 1 - p['vencedor']))}", home == (res["casa"] > res["fora"])


def _track_page(store: dict, updated: datetime) -> str:
    ev = registo.evaluate(store)
    recs = list(store.values())
    pending = sum(r["estado"] == "pendente" for r in recs)
    body = ['<h1>📊 Histórico de acertos</h1>',
            '<p class="sub">As previsões publicadas no site comparadas com o resultado final.</p>',
            f'<p class="note">{TRACK_NOTE}</p>']
    if recs:
        since = min(r["feita_em"] for r in recs)
        since = datetime.fromisoformat(since.replace("Z", "+00:00")).astimezone(LISBON)
        body.append(f'<p class="note">Registo desde {since:%d/%m/%Y}. {pending} '
                    f'{"jogo guardado à espera" if pending == 1 else "jogos guardados à espera"} '
                    f"do resultado.</p>")
    if ev.empty:
        body.append('<div class="empty">Ainda não há jogos terminados com previsão guardada. Os '
                    "primeiros resultados aparecem aqui depois dos próximos jogos.</div>")
        return _page("Histórico", "".join(body), "", "historico", updated)

    main = ev[ev["chave"] == ev["desporto"].map(registo.MAIN)]
    cards = []
    for sport, (title, _) in SPORTS.items():
        part = main[main["desporto"] == sport]
        if len(part):
            cards.append(f'<a href="#{sport}">{title}<span>{len(part)} jogos · acertou o vencedor em '
                         f'{_pct(part["acerto"].mean())}</span></a>')
    body.append(f'<div class="grid">{"".join(cards)}</div>')
    body.append(f'<p class="note">{TRACK_HELP}</p>')

    head = ["Mercado", "Jogos", "Acerto", "Confiança", "Pontuação", "À sorte"]
    for sport, (title, leagues) in SPORTS.items():
        part = ev[ev["desporto"] == sport]
        if part.empty:
            continue
        body.append(f'<h2 id="{sport}">{title}</h2>')
        body.append(_tbl(head, _stat_rows(registo.table(part, ["mercado"]), lambda r: html.escape(r.mercado))))
        by_league = registo.table(part[part["chave"] == registo.MAIN[sport]], ["liga"])
        rows = _stat_rows(by_league, lambda r: f"{FLAGS.get(r.liga, '')} {html.escape(leagues[r.liga].name)}"
                          if r.liga in leagues else html.escape(r.liga))
        league_head = ["Liga (vencedor)"] + head[1:]
        if sport == "futebol":
            market = registo.table(part[part["chave"] == "mercado_1x2"], ["liga"]).set_index("liga")
            league_head += ["Casas: jogos", "Casas: acerto", "Casas: pontuação"]
            for row, liga in zip(rows, by_league["liga"]):
                m = market.loc[liga] if liga in market.index else None
                row += ["–", "–", "–"] if m is None else [f"{m.jogos}", _pct(m.acerto), _n(m.logloss, 3)]
        body.append(_tbl(league_head, rows))
    if (ev["chave"] == "mercado_1x2").any():
        body.append('<p class="note">"Casas" são as odds das casas de apostas sem margem, nos jogos de '
                    "futebol em que já existiam quando a previsão foi guardada.</p>")

    cal = registo.calibration(ev)
    body.append("<h2>Percentagens bem afinadas?</h2>")
    body.append('<p class="note">Todas as previsões de todos os mercados, agrupadas pela percentagem '
                "que o modelo deu ao lado escolhido.</p>")
    body.append(_tbl(["Confiança do modelo", "Previsões", "Média prevista", "Aconteceu"],
                     [[r.grupo, f"{r.previsoes}", _pct(r.confianca), _pct(r.acerto)] for r in cal.itertuples()]))

    done = sorted((r for r in recs if r["estado"] == "terminado"),
                  key=lambda r: (r["data"], r["hora"] or ""), reverse=True)[:40]
    rows = []
    for rec in done:
        label, ok = _pick_label(rec)
        res = rec["resultado"]
        rows.append([f"{_when(rec['data'], rec['hora'])}",
                     f"{FLAGS.get(rec['liga'], '')} {_team(rec['casa'])} – {_team(rec['fora'])}",
                     label, f"{res['casa']}–{res['fora']}",
                     '<span class="ok">✓</span>' if ok else '<span class="ko">✗</span>'])
    body.append("<h2>Últimos jogos</h2>")
    body.append(_tbl(["Data", "Jogo", "Favorito do modelo", "Resultado", ""], rows, "games"))
    return _page("Histórico", "".join(body), "", "historico", updated)


def render_site(preds: pd.DataFrame, basket: pd.DataFrame | None = None,
                updated: datetime | None = None, hockey: pd.DataFrame | None = None,
                track: dict | None = None) -> dict[str, str]:
    """Todas as páginas do site: {caminho relativo: html}; track é o registo das previsões."""
    updated = updated or datetime.now(timezone.utc)
    basket = basket if basket is not None else pd.DataFrame()
    hockey = hockey if hockey is not None else pd.DataFrame()
    pages = {"index.html": _index(preds, basket, hockey, updated),
             "historico.html": _track_page(track or {}, updated)}
    for key, league in LEAGUES.items():
        games = preds[preds["league"] == key] if not preds.empty else preds
        pages[f"futebol/{key.lower()}.html"] = _league_page(
            key, league, games, _game, FOOTBALL_NOTE + FOOTBALL_LEAGUE_NOTES.get(key, ""), "../",
            updated)
    for key, league in BASKET_LEAGUES.items():
        games = basket[basket["league"] == key] if not basket.empty else basket
        note = BASKET_NOTE
        if key in BASKET_ACCURACY:
            note += f" No teste com jogos passados acertou o vencedor em {BASKET_ACCURACY[key]}% dos jogos."
        pages[f"basquetebol/{key.lower()}.html"] = _league_page(
            key, league, games, _basket_game, note, "../", updated)
    for key, league in HOCKEY_LEAGUES.items():
        games = hockey[hockey["league"] == key] if not hockey.empty else hockey
        pages[f"hoquei/{key.lower()}.html"] = _league_page(
            key, league, games, _hockey_game, HOCKEY_NOTE, "../", updated)
    return pages


def to_json(preds: pd.DataFrame) -> str:
    records = preds.astype(object).where(preds.notna(), None).to_dict("records")
    return json.dumps(records, ensure_ascii=False, default=str, indent=1)
