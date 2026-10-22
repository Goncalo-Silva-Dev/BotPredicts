"""Site estático com as previsões (para o GitHub Pages).

index.html é o aviso inicial; cada liga tem a sua página em futebol/ ou basquetebol/,
com um menu comum no topo.
"""
import html
import re
import json
import math
from datetime import datetime, timezone

import pandas as pd

from analise.ingest.basket import BASKET_LEAGUES
from analise.leagues import LEAGUES

FLAGS = {"POR": "🇵🇹", "ESP": "🇪🇸", "ENG": "🏴", "FRA": "🇫🇷", "ITA": "🇮🇹", "USA": "🇺🇸", "BRA": "🇧🇷",
         "NBA": "🇺🇸", "EUR": "🇪🇺", "EUC": "🇪🇺", "NBB": "🇧🇷", "ACB": "🇪🇸", "LBA": "🇮🇹", "LPB": "🇵🇹"}
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
"""


def _pct(v) -> str:
    return "–" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v * 100:.0f}%"


def _when(d, t) -> str:
    d = pd.Timestamp(d)
    hour = f" {t}" if isinstance(t, str) and t else ""
    return f"{WEEKDAYS[d.weekday()]} {d.day:02d}/{d.month:02d}{hour}"


def _game(g) -> str:
    teams = f"{html.escape(g['home'])} – {html.escape(g['away'])}"
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
        f"Golos esperados <b>{g['golos_casa']:.1f} – {g['golos_fora']:.1f}</b>",
        f"Mais de 2.5 <b>{_pct(g['mais_2_5'])}</b>",
        f"Ambas marcam <b>{_pct(g['ambas_marcam'])}</b>",
        f"Resultados <b>{html.escape(g['resultados'])}</b>",
    ]
    if pd.notna(g.get("mercado_casa")):
        facts.append("Casas de apostas <b>" + " / ".join(
            _pct(g[k]) for k in ("mercado_casa", "mercado_empate", "mercado_fora")) + "</b>")
    spans = "".join(f"<span>{f}</span>" for f in facts)
    return f'<div class="game">{head}<div class="bar">{bar}</div><div class="facts">{spans}</div></div>'


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
        f"Diferença <b>{html.escape(fav)} por {abs(g['diferenca']):.1f}</b>",
        f"Total <b>{g['total']:.0f}</b> (80%: {g['total_min']:.0f} a {g['total_max']:.0f})",
    ]
    spans = "".join(f"<span>{f}</span>" for f in facts)
    return f'<div class="game">{head}<div class="bar">{bar}</div><div class="facts">{spans}</div></div>'


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
pouco atrás das odds de fecho das casas de apostas."""

BASKET_NOTE = """Barras: vitória da casa · vitória de fora. Ratings de ataque e defesa de cada
equipa com os resultados das últimas épocas (os jogos recentes pesam mais)."""

# Percentagem de vencedores certos no teste com jogos passados (basquete-backtest)
BASKET_ACCURACY = {"NBA": 66, "EUR": 64, "EUC": 65, "ACB": 69, "LBA": 71, "LPB": 75, "NBB": 73}


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
</nav></header>
<main>
{body}
<p class="note">Atualizado {updated:%d/%m/%Y %H:%M} UTC · Dados: football-data.co.uk,
openfootball, ESPN, Euroleague Basketball e sites oficiais das ligas.</p>
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


def _index(preds, basket, updated) -> str:
    def cards(folder, leagues, df):
        out = []
        for k, l in leagues.items():
            n = int((df["league"] == k).sum()) if not df.empty else 0
            out.append(f'<a href="{folder}/{k.lower()}.html">{FLAGS.get(k, "")} '
                       f"{html.escape(l.name)}<span>{n} {"jogo" if n == 1 else "jogos"} com previsão</span></a>")
        return "".join(out)
    body = f"""<h1>Previsões de futebol e basquetebol</h1>
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
<h2>🏀 Basquetebol</h2><div class="grid">{cards("basquetebol", BASKET_LEAGUES, basket)}</div>"""
    return _page("Início", body, "", "", updated)


def render_site(preds: pd.DataFrame, basket: pd.DataFrame | None = None,
                updated: datetime | None = None) -> dict[str, str]:
    """Todas as páginas do site: {caminho relativo: html}."""
    updated = updated or datetime.now(timezone.utc)
    basket = basket if basket is not None else pd.DataFrame()
    pages = {"index.html": _index(preds, basket, updated)}
    for key, league in LEAGUES.items():
        games = preds[preds["league"] == key] if not preds.empty else preds
        pages[f"futebol/{key.lower()}.html"] = _league_page(
            key, league, games, _game, FOOTBALL_NOTE, "../", updated)
    for key, league in BASKET_LEAGUES.items():
        games = basket[basket["league"] == key] if not basket.empty else basket
        note = BASKET_NOTE
        if key in BASKET_ACCURACY:
            note += f" No teste com jogos passados acertou o vencedor em {BASKET_ACCURACY[key]}% dos jogos."
        pages[f"basquetebol/{key.lower()}.html"] = _league_page(
            key, league, games, _basket_game, note, "../", updated)
    return pages


def to_json(preds: pd.DataFrame) -> str:
    records = preds.astype(object).where(preds.notna(), None).to_dict("records")
    return json.dumps(records, ensure_ascii=False, default=str, indent=1)
