"""sweep40x — birdcam protect/unprotect must not claim success for an unknown event.

The GET /events/{event_id} contract 404s when no such event exists.  The
protect/unprotect endpoints used to return ``{"ok": True}`` unconditionally,
even though ``set_event_protected`` runs a MATCH+SET that changes zero rows.
That is a vacuous success: an operator cannot tell "protection applied" from
"nothing happened".  All inputs are in-memory fakes; no Neo4j, network or
filesystem state is touched.
"""
import pytest
from fastapi.testclient import TestClient

from birdcam.api import create_app


class Repo:
    """Minimal in-memory stand-in for DetectionGraphRepository."""

    def __init__(self, known=()):
        self.known = set(known)
        self.writes = []

    def list_events(self, camera_id=None, limit=100):
        return [{"event_id": e} for e in sorted(self.known)]

    def get_event(self, event_id):
        return {"event_id": event_id} if event_id in self.known else None

    def set_event_protected(self, event_id, protected):
        self.writes.append((event_id, protected))


@pytest.fixture
def repo():
    return Repo({"e1"})


@pytest.fixture
def client(repo):
    return TestClient(create_app(repo))


@pytest.mark.parametrize("action", ["protect", "unprotect"])
def test_unknown_event_never_reports_success(client, repo, action):
    response = client.post(f"/events/ghost/{action}")
    assert response.status_code == 404, (action, response.status_code, response.json())
    assert repo.writes == [], "no protection write should target a nonexistent event"


@pytest.mark.parametrize(
    "action,value", [("protect", True), ("unprotect", False)]
)
def test_known_event_still_succeeds(client, repo, action, value):
    response = client.post(f"/events/e1/{action}")
    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert repo.writes == [("e1", value)]


def test_get_unknown_event_control(client):
    # Pre-existing contract: GET of an unknown event is a 404.
    assert client.get("/events/ghost").status_code == 404
