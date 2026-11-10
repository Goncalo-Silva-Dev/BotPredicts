# Percentagens ao vivo: estimação e teste com jogos passados

As contas ao vivo estão em `src/analise/aovivo_modelo.py` (Python) e `src/analise/aovivo.js`
(browser); os testes garantem que dão o mesmo. Os números dos modelos foram estimados e
testados com estes scripts (correr por ordem, numa pasta de trabalho):

1. `1_futebol_espn.py`: minuto de cada golo e expulsão (ESPN) dos jogos do backtest desde 2023-07.
2. `2_futebol_juntar.py`: junta-os às probabilidades de antes do jogo do backtest
   (`BACKTEST_DIR`, CSVs de `python -m analise.cli backtest`) e tira os golos esperados.
3. `3_futebol_estimar.py`: peso de cada minuto, fator do marcador e das expulsões
   (estimados só com 2023-24 e 2024-25).
4. `4_basquete_hoquei_espn.py`: NBA, NBL e NHL com pontos/golos por período.
5. `5_basquete_estimar.py NBA.pkl`: coeficientes do basquetebol (épocas antes de 2025-26).
6. `6_tenis_espn.py`: jogos de ténis e previsões de antes do jogo (Elo, só jogos anteriores).
7. `validar.py .`: testa tudo em jogos de 2025-26 (ténis: desde 2025) que não entraram na
   estimação; resultados em `resultados.json`.

## Resultados (log loss: quanto mais baixo, melhor)

| Desporto | Jogos de teste | Antes do jogo | Momento | Ao vivo | Acerto ao vivo |
|---|---|---|---|---|---|
| Futebol (8 ligas) | 3 349 | 0,993 | 30' / intervalo / 75' | 0,902 / 0,815 / 0,580 | 57% / 62% / 76% |
| Basquetebol (NBA) | 1 609 | 0,621 | fim do 1.º / 2.º / 3.º quarto | 0,576 / 0,490 / 0,375 | 70% / 76% / 82% |
| Hóquei (NHL) | 1 400 | 0,692 | fim do 1.º / 2.º período | 0,610 / 0,438 | 66% / 78% |
| Ténis (ATP e WTA) | 17 269 | 0,627 | fim do 1.º set | 0,451 | 82% |

Calibração (quando o site diz X%, acontece em quantos?): no futebol, os resultados dados
entre 70% e 85% aconteceram 74% das vezes (previsto 77%) e os dados entre 15% e 30%
aconteceram 23% (previsto 23%); o "mais de 2,5 golos" entre 10% e 30% aconteceu 25%
(previsto 21%). Ver `resultados.json` para todas as faixas.
