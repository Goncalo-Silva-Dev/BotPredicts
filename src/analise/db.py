import os

from sqlalchemy import (
    Column, Date, Float, Integer, MetaData, String, Table, UniqueConstraint, create_engine, inspect,
    text,
)

metadata = MetaData()

matches = Table(
    "matches",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("league", String, nullable=False),
    Column("season", String, nullable=False),
    Column("date", Date, nullable=False),
    Column("home", String, nullable=False),
    Column("away", String, nullable=False),
    Column("home_goals", Integer),
    Column("away_goals", Integer),
    Column("home_goals_ht", Integer),  # golos ao intervalo
    Column("away_goals_ht", Integer),
    Column("home_shots", Integer),
    Column("away_shots", Integer),
    Column("home_shots_target", Integer),
    Column("away_shots_target", Integer),
    # Odds 1X2 de fecho: média do mercado e Pinnacle
    Column("odds_home_avg", Float),
    Column("odds_draw_avg", Float),
    Column("odds_away_avg", Float),
    Column("odds_home_pin", Float),
    Column("odds_draw_pin", Float),
    Column("odds_away_pin", Float),
    Column("source", String, nullable=False),
    UniqueConstraint("league", "date", "home", "away"),
)


def get_engine(url: str | None = None):
    url = url or os.environ.get("DATABASE_URL", "sqlite:///data/analise.db")
    if url.startswith("sqlite:///"):
        os.makedirs(os.path.dirname(url.removeprefix("sqlite:///")) or ".", exist_ok=True)
    engine = create_engine(url)
    metadata.create_all(engine)
    _add_missing_columns(engine)
    return engine


def _add_missing_columns(engine) -> None:
    """Bases criadas antes de haver colunas novas: acrescenta-as (ficam vazias até ao
    próximo 'historico')."""
    existing = {c["name"] for c in inspect(engine).get_columns("matches")}
    with engine.begin() as conn:
        for col in matches.columns:
            if col.name not in existing:
                conn.execute(text(f"ALTER TABLE matches ADD COLUMN {col.name} {col.type.compile(engine.dialect)}"))
