"""Página estática com as previsões dos próximos jogos (para o GitHub Pages)."""
import html
import re
import json
import math
from datetime import datetime, timezone

import pandas as pd

from analise.ingest.basket import BASKET_LEAGUES
from analise.leagues import LEAGUES

FLAGS = {"POR": "🇵🇹", "ESP": "🇪🇸", "ENG": "🏴", "FRA": "🇫🇷", "ITA": "🇮🇹", "USA": "🇺🇸", "BRA": "🇧🇷",
         "NBA": "🇺🇸", "EUR": "🇪🇺", "EUC": "🇪🇺", "NBB": "🇧🇷"}
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
nav { display:flex; gap:16px; margin:16px 0 0; font-weight:600; }
nav a { color:var(--home); text-decoration:none; }
h1.sport { font-size:21px; margin:36px 0 0; padding-top:12px; border-top:1px solid var(--line); }
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


def _sections(preds: pd.DataFrame, leagues: dict, card) -> list[str]:
    out = []
    for key, league in leagues.items():
        games = preds[preds["league"] == key] if not preds.empty else preds
        if not games.empty:
            out.append(f"<h2>{FLAGS.get(key, '')} {html.escape(league.name)}</h2>"
                       f"{_league_rounds(games, card)}")
    return out


def render(preds: pd.DataFrame, basket: pd.DataFrame | None = None,
           updated: datetime | None = None) -> str:
    updated = updated or datetime.now(timezone.utc)
    basket = basket if basket is not None else pd.DataFrame()
    football = _sections(preds, LEAGUES, _game) or [
        '<div class="empty">Não foi possível obter o calendário dos próximos '
        'jogos. A página volta a tentar na próxima atualização.</div>']
    hoops = _sections(basket, BASKET_LEAGUES, _basket_game) or [
        '<div class="empty">Sem jogos de basquetebol nesta atualização.</div>']

    return f"""<!doctype html>
<html lang="pt-PT">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Previsões BotPredicts</title>
<style>{CSS}</style>
</head>
<body><main>
<h1>Previsões de todos os jogos que faltam</h1>
<p class="sub">Atualizado {updated:%d/%m/%Y %H:%M} UTC</p>
<p class="note">São probabilidades, não certezas, e usam a forma atual das equipas: os jogos
mais distantes vão mudar à medida que a época avança.</p>
<nav><a href="#futebol">⚽ Futebol</a><a href="#basquetebol">🏀 Basquetebol</a></nav>

<h1 class="sport" id="futebol">⚽ Futebol</h1>
<p class="note">Barras: vitória da casa · empate · vitória de fora. Modelo Dixon-Coles com os
últimos 2 anos de resultados. As odds das casas de apostas aparecem quando já existem (perto do
jogo). No teste com 11 984 jogos passados o modelo acertou bem nas percentagens mas ficou um
pouco atrás das odds de fecho das casas de apostas, por isso ainda não serve para apostar.</p>
{''.join(football)}

<h1 class="sport" id="basquetebol">🏀 Basquetebol</h1>
<p class="note">Barras: vitória da casa · vitória de fora. Ratings de ataque e defesa de cada
equipa com os últimos 18 meses de resultados. No teste com jogos passados acertou o vencedor em
66% dos jogos da NBA, 64% da EuroLeague e 65% da EuroCup.</p>
{''.join(hoops)}

<p class="note">Dados: football-data.co.uk, openfootball, ESPN e Euroleague Basketball.</p>
</main></body>
</html>
"""


def to_json(preds: pd.DataFrame) -> str:
    records = preds.astype(object).where(preds.notna(), None).to_dict("records")
    return json.dumps(records, ensure_ascii=False, default=str, indent=1)
