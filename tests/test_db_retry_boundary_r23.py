"""Hermetic regression: transient markers must not span the type/message boundary.

Round 23 / r23. No live Neo4j, no driver, no sleep: only ``Exception`` objects
and the pure :func:`auto_ingest.shorts.db_retry.is_transient` predicate.
"""
from auto_ingest.shorts import db_retry


class Service(Exception):
    """Type name is the first half of the ``ServiceUnavailable`` marker."""


class Session(Exception):
    """Type name is the first half of the ``SessionExpired`` marker."""


class SessionExpiredError(Exception):
    """Type name carries a whole marker on its own."""


def test_marker_must_not_span_type_and_message_boundary():
    # The only occurrence of "ServiceUnavailable" is across the artificial
    # concatenation of type name and message. These are ordinary, non-transient
    # errors and must not trigger a 12-attempt backoff retry storm.
    assert not db_retry.is_transient(Service("Unavailable: bad credentials"))
    assert not db_retry.is_transient(
        Session("Expired? no - this is a plain query syntax error"))


def test_whole_marker_in_type_or_message_still_detected():
    # Controls: a real marker wholly inside either field is still transient.
    assert db_retry.is_transient(SessionExpiredError("plain syntax error"))
    assert db_retry.is_transient(
        ValueError("neo4j.exceptions.TransientError while querying"))
    assert not db_retry.is_transient(ValueError("plain syntax error"))
