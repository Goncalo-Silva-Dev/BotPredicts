from dataclasses import dataclass


@dataclass(frozen=True)
class League:
    key: str
    name: str
    # Código no football-data.co.uk: divisão (ligas principais) ou país (ligas extra);
    # None para as ligas que só existem no betexplorer
    fd_code: str | None
    # Ligas extra têm um ficheiro único com todas as épocas e calendário anual
    fd_extra: bool = False
    # País no ficheiro de próximos jogos das ligas extra (new_league_fixtures.csv)
    fd_country: str | None = None
    # Ficheiro do calendário completo no openfootball (sem MLS)
    of_code: str | None = None
    # Fuso das horas do openfootball (hora local da liga); o football-data usa hora do
    # Reino Unido, que é sempre igual à de Portugal continental
    tz: str = "Europe/Lisbon"
    # Torneio da época atual no betexplorer: calendário das ligas sem openfootball e, nas
    # ligas sem football-data, também os resultados
    be_path: str | None = None
    # Torneios das épocas passadas no betexplorer ({y} e {y1}: anos de início e fim)
    be_history: str | None = None


LEAGUES = {
    l.key: l
    for l in [
        League("POR", "Liga Portugal", "P1", of_code="pt.1"),
        League("ESP", "LaLiga", "SP1", of_code="es.1", tz="Europe/Madrid"),
        League("ENG", "Premier League", "E0", of_code="en.1", tz="Europe/London"),
        League("FRA", "Ligue 1", "F1", of_code="fr.1", tz="Europe/Paris"),
        League("ITA", "Serie A", "I1", of_code="it.1", tz="Europe/Rome"),
        League("USA", "MLS", "USA", fd_extra=True, fd_country="USA", be_path="football/usa/mls"),
        League("BRA", "Brasileirão Série A", "BRA", fd_extra=True, fd_country="Brazil", of_code="br.1",
               tz="America/Sao_Paulo"),
        League("TUR", "Süper Lig", "T1", be_path="football/turkey/super-lig"),
        League("NED", "Eredivisie", "N1", of_code="nl.1", tz="Europe/Amsterdam"),
        League("BEL", "Jupiler Pro League", "B1", be_path="football/belgium/jupiler-pro-league"),
        League("PO2", "Liga Portugal 2", None, be_path="football/portugal/liga-portugal-2",
               be_history="football/portugal/liga-portugal-2-{y}-{y1}"),
        League("U23", "Liga Revelação Sub-23", None, be_path="football/portugal/liga-next-gen-u23",
               be_history="football/portugal/liga-revelacao-u23-{y}-{y1}"),
    ]
}
