# Análise de jogos

Recolha de dados, modelo de previsões, relatórios por jogo e app web para futebol
(Portugal, Espanha, Inglaterra, França, Itália, MLS, Brasil) e, mais tarde, basquetebol.

## Começar

```bash
pip install -e ".[dev]"
cp .env.example .env
python -m analise.cli historico --epocas 6   # descarrega e guarda o histórico
pytest
```

## Fontes (fase 1, gratuitas)

| Fonte | Uso |
| --- | --- |
| football-data.co.uk | Histórico de resultados, estatísticas e odds de fecho |
| football-data.org | Calendário e resultados atuais (sem MLS) |
| API-Football | MLS, alinhamentos, lesões (100 pedidos/dia) |
| Understat | xG (Inglaterra, Espanha, Itália, França) |
| The Odds API | Odds atuais (500 créditos/mês) |
