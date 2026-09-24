"""SQLite engine/session setup - swaps to Postgres+pgvector by changing
DATABASE_URL once volume justifies it (see CLAUDE.md roadmap, Phase 5).
"""
import os

from sqlmodel import Session, SQLModel, create_engine

# DETECTIVE_DATABASE_URL lets a test run or a scratch server use its own database
# (conftest.py sets it, so the suite can never touch detective.db); unset, it is the
# same relative detective.db as always.
DATABASE_URL = os.environ.get("DETECTIVE_DATABASE_URL", "sqlite:///detective.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

# Columns added to an existing table after a database was first created.
# SQLModel's create_all only creates missing *tables*, never missing columns,
# so an install that predates a column would otherwise crash on first query.
# (table, column, SQLite column DDL). A stopgap - real migrations (Alembic)
# arrive with the Postgres move, if that ever happens.
_COLUMN_MIGRATIONS = [
    ("source", "date_note", "VARCHAR"),  # Phase 5
]


def _add_missing_columns():
    with engine.begin() as conn:
        for table, column, ddl in _COLUMN_MIGRATIONS:
            existing = {row[1] for row in conn.exec_driver_sql(f"PRAGMA table_info({table})")}
            if existing and column not in existing:
                conn.exec_driver_sql(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)
    _add_missing_columns()


def get_session():
    with Session(engine) as session:
        yield session
