"""Neo4j connection helpers."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from neo4j import GraphDatabase, Driver, Session

from .config import settings

_driver: Driver | None = None


def get_driver() -> Driver:
    """Return a lazily initialised Neo4j driver."""
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(
            settings.neo4j.uri, auth=(settings.neo4j.user, settings.neo4j.password)
        )
    return _driver


@contextmanager
def get_session() -> Iterator[Session]:
    """Context manager that yields a database session and closes it afterwards."""
    driver = get_driver()
    session = driver.session(database=settings.neo4j.database)
    try:
        yield session
    finally:
        session.close()


def close_driver() -> None:
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None

