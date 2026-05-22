from fastapi.testclient import TestClient
from birdcam.api import create_app

class Repo:
    def list_events(self, camera_id=None, limit=100): return [{'event_id':'e1'}]
    def get_event(self, event_id): return {'event_id':event_id}
    def set_event_protected(self, event_id, protected): self.last=(event_id,protected)

def test_api_graph_endpoints():
    repo=Repo(); c=TestClient(create_app(repo))
    assert c.get('/events').status_code == 200
    assert c.get('/events/e1').json()['event_id'] == 'e1'
    assert c.post('/events/e1/protect').status_code == 200
