"""
SQLite database setup using SQLAlchemy.

Uses SQLAlchemy's declarative mapping with a synchronous engine.
The database file is created automatically in the data directory.
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from backend.app.utils.config import get_settings
from backend.app.utils.logging import get_logger

logger = get_logger(__name__)


class Base(DeclarativeBase):
    """SQLAlchemy declarative base for all ORM models."""
    pass


def get_db_url() -> str:
    """Return the SQLite database URL."""
    settings = get_settings()
    db_path = settings.data_dir / "sentinel_cd.db"
    return f"sqlite:///{db_path}"


def create_db_engine():
    """Create and return the SQLAlchemy engine."""
    url = get_db_url()
    engine = create_engine(
        url,
        connect_args={"check_same_thread": False},  # Required for SQLite
        echo=False,
    )
    logger.debug("Database engine created: %s", url)
    return engine


# Module-level singletons
_engine = None
_SessionLocal = None


def get_engine():
    """Return the singleton database engine, creating it if necessary."""
    global _engine
    if _engine is None:
        _engine = create_db_engine()
    return _engine


def get_session_factory():
    """Return the session factory."""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=get_engine())
    return _SessionLocal


def get_db_session() -> Session:
    """Return a new database session (caller must close it)."""
    return get_session_factory()()


def init_database() -> None:
    """
    Create all database tables.
    Safe to call multiple times — tables are only created if they don't exist.
    """
    # Import models so SQLAlchemy knows about them
    from backend.app.storage import models  # noqa: F401

    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialized: %s", get_db_url())
