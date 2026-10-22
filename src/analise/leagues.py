from dataclasses import dataclass


@dataclass(frozen=True)
class League:
    key: str
    name: str
    # Código no football-data.co.uk: divisão (ligas principais) ou país (ligas extra)
    fd_code: str
    # Ligas extra têm um ficheiro único com todas as épocas e calendário anual
    fd_extra: bool = False
    # País no ficheiro de próximos jogos das ligas extra (new_league_fixtures.csv)
    fd_country: str | None = None


LEAGUES = {
    l.key: l
    for l in [
        League("POR", "Liga Portugal", "P1"),
        League("ESP", "LaLiga", "SP1"),
        League("ENG", "Premier League", "E0"),
        League("FRA", "Ligue 1", "F1"),
        League("ITA", "Serie A", "I1"),
        League("USA", "MLS", "USA", fd_extra=True, fd_country="USA"),
        League("BRA", "Brasileirão Série A", "BRA", fd_extra=True, fd_country="Brazil"),
    ]
}
