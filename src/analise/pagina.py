"""Página estática com as previsões dos próximos jogos (para o GitHub Pages)."""
import html
import json
import math
from datetime import datetime, timezone

import pandas as pd

from analise.leagues import LEAGUES

FLAGS = {"POR": "🇵🇹", "ESP": "🇪🇸", "ENG": "🏴", "FRA": "🇫🇷", "ITA": "🇮🇹", "USA": "🇺🇸", "BRA": "🇧🇷"}
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


def render(preds: pd.DataFrame, updated: datetime | None = None) -> str:
    updated = updated or datetime.now(timezone.utc)
    sections = []
    if preds.empty:
        sections.append('<div class="empty">Ainda não há jogos marcados para os próximos dias. '
                        'O calendário do fim de semana costuma chegar à quinta ou sexta.</div>')
    for key, league in LEAGUES.items():
        games = preds[preds["league"] == key] if not preds.empty else preds
        if games.empty:
            continue
        cards = "".join(_game(g) for g in games.to_dict("records"))
        sections.append(f"<h2>{FLAGS.get(key, '')} {html.escape(league.name)}</h2>{cards}")

    return f"""<!doctype html>
<html lang="pt-PT">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Previsões BotPredicts</title>
<style>{CSS}</style>
</head>
<body><main>
<h1>Previsões dos próximos jogos</h1>
<p class="sub">Atualizado {updated:%d/%m/%Y %H:%M} UTC · modelo Dixon-Coles treinado com os
últimos 2 anos de resultados</p>
<p class="note">Barras: vitória da casa · empate · vitória de fora. São probabilidades, não
certezas. No teste com 11 984 jogos passados o modelo acertou bem nas percentagens mas ficou um
pouco atrás das odds de fecho das casas de apostas, por isso ainda não serve para apostar.</p>
{''.join(sections)}
<p class="note">Dados: football-data.co.uk.</p>
</main></body>
</html>
"""


def to_json(preds: pd.DataFrame) -> str:
    records = preds.astype(object).where(preds.notna(), None).to_dict("records")
    return json.dumps(records, ensure_ascii=False, default=str, indent=1)
