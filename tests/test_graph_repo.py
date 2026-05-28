from birdcam.graph.repository import DetectionGraphRepository
from birdcam.graph.schema import init_schema
from birdcam.graph.outbox import GraphOutbox

class FakeDriver:
    def __init__(self, fail=False): self.calls=[]; self.fail=fail
    def execute(self, q, **p):
        self.calls.append((q,p))
        if self.fail: raise RuntimeError('down')
        if 'RETURN e' in q: return [{'e': {'event_id': p.get('event_id','e1')}}]
        return []

def test_schema_init_runs_queries():
    d=FakeDriver(); init_schema(d); assert len(d.calls) >= 10

def test_outbox_on_failure(tmp_path):
    out=GraphOutbox(str(tmp_path/'o.sqlite'))
    repo=DetectionGraphRepository(FakeDriver(fail=True), outbox=out, use_outbox_on_failure=True)
    repo.set_event_protected('e1', True)
    assert len(out.pending()) == 1

def test_replay_outbox(tmp_path):
    out=GraphOutbox(str(tmp_path/'o.sqlite')); out.append('x', {'query':'RETURN 1', 'params':{}})
    repo=DetectionGraphRepository(FakeDriver(), outbox=out, use_outbox_on_failure=True)
    assert repo.replay_outbox() == 1
    assert out.pending() == []
