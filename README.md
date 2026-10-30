# Análise de jogos

Recolha de dados, modelo de previsões, relatórios por jogo e app web para futebol
(Portugal, Espanha, Inglaterra, França, Itália, Alemanha, MLS, Brasil, Arábia Saudita) e basquetebol (NBA, EuroLeague, EuroCup, Espanha, Itália,
Portugal, França e Brasil), e hóquei no gelo (NHL).

## Começar

```bash
pip install -e ".[dev]"
cp .env.example .env
python -m analise.cli historico --epocas 6   # descarrega e guarda o histórico
python -m analise.cli backtest               # Dixon-Coles vs odds de fecho sem margem
python -m analise.cli backtest --peso-golos 1   # o mesmo, só com golos (sem remates à baliza)
python -m analise.cli previsoes              # probabilidades dos jogos dos próximos 7 dias
python -m analise.cli previsoes --ligas POR --jogo Benfica Porto   # um jogo à escolha
python -m analise.cli basquete-backtest      # ligas de basquetebol em jogos passados
python -m analise.cli hoquei-backtest         # NHL em jogos passados
python -m analise.cli pagina                 # gera site/index.html com todos os jogos que faltam
pytest
```

## Fontes (fase 1, gratuitas)

| Fonte | Uso |
| --- | --- |
| football-data.co.uk | Histórico de resultados, estatísticas e odds de fecho |
| football-data.org | Calendário e resultados atuais (sem MLS) |
| API-Football | MLS, alinhamentos, lesões (100 pedidos/dia) |
| openfootball (football.json) | Calendário completo da época (todas menos a MLS) |
| ESPN (API pública) | NBA: calendário e resultados |
| Euroleague Basketball (api-live.euroleague.net) | EuroLeague e EuroCup |
| Understat | xG (Inglaterra, Espanha, Itália, França) |
| The Odds API | Odds atuais (500 créditos/mês) |

## Página das previsões

O workflow `.github/workflows/previsoes.yml` corre às 6h e às 16h (UTC), descarrega o
histórico, calcula as previsões de todos os jogos que faltam na época (calendário do
openfootball; a MLS só tem os jogos já publicados no football-data.co.uk) e publica `site/`
no GitHub Pages.
Para funcionar, em Settings → Pages escolhe "Source: GitHub Actions".
