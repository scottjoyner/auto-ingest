from __future__ import annotations
from dataclasses import asdict
from datetime import datetime
from birdcam.models import EventSummary

class DetectionGraphRepository:
    def __init__(self, driver, outbox=None, use_outbox_on_failure:bool=True):
        self.driver = driver
        self.outbox = outbox
        self.use_outbox_on_failure = use_outbox_on_failure

    def _run(self, op_type, query, payload):
        try:
            return self.driver.execute(query, **payload)
        except Exception as e:
            if self.outbox and self.use_outbox_on_failure:
                self.outbox.append(op_type, {"query": query, "params": payload}, str(e))
                return []
            raise

    def upsert_camera(self, camera: dict) -> None:
        q = """MERGE (c:Camera {camera_id:$camera_id}) SET c += $props, c.updated_at=$now, c.created_at=coalesce(c.created_at,$now)"""
        self._run("upsert_camera", q, {"camera_id": camera["camera_id"], "props": camera, "now": datetime.utcnow().isoformat()})

    def create_detection_event(self, event: EventSummary) -> None:
        payload = asdict(event); now = datetime.utcnow().isoformat()
        q = """MERGE (e:DetectionEvent {event_id:$event_id}) SET e += $props, e.updated_at=$now, e.created_at=coalesce(e.created_at,$now) WITH e MATCH (c:Camera {camera_id:$camera_id}) MERGE (c)-[:PRODUCED_EVENT]->(e)"""
        payload.update({"camera_id":event.camera_id,"start_epoch_ms":int(event.start_time.timestamp()*1000),"end_epoch_ms":int(event.end_time.timestamp()*1000)})
        self._run("create_event", q, {"event_id": event.event_id, "camera_id": event.camera_id, "props": payload, "now": now})

    def add_detection(self, detection: dict) -> None:
        q = """MERGE (d:Detection {detection_id:$detection_id}) SET d += $props WITH d MATCH (e:DetectionEvent {event_id:$event_id}) MERGE (e)-[:HAS_DETECTION]->(d)"""
        self._run("add_detection", q, {"detection_id": detection["detection_id"], "event_id": detection["event_id"], "props": detection})

    def attach_clip(self, event_id: str, clip: dict) -> None:
        q = """MERGE (c:Clip {clip_id:$clip_id}) SET c += $clip WITH c MATCH (e:DetectionEvent {event_id:$event_id}) MERGE (e)-[:HAS_CLIP]->(c)"""
        self._run("attach_clip", q, {"clip_id":clip["clip_id"], "clip":clip, "event_id":event_id})

    def attach_thumbnail(self, event_id: str, thumbnail: dict) -> None:
        q = """MERGE (t:Thumbnail {thumbnail_id:$thumbnail_id}) SET t += $thumb WITH t MATCH (e:DetectionEvent {event_id:$event_id}) MERGE (e)-[:HAS_THUMBNAIL]->(t)"""
        self._run("attach_thumbnail", q, {"thumbnail_id":thumbnail["thumbnail_id"], "thumb":thumbnail, "event_id":event_id})

    def list_events(self, camera_id=None, limit=100):
        if camera_id:
            q="MATCH (c:Camera {camera_id:$camera_id})-[:PRODUCED_EVENT]->(e:DetectionEvent) RETURN e ORDER BY e.start_epoch_ms DESC LIMIT $limit"
            rs=self.driver.execute(q,camera_id=camera_id,limit=limit)
        else:
            q="MATCH (e:DetectionEvent) RETURN e ORDER BY e.start_epoch_ms DESC LIMIT $limit"
            rs=self.driver.execute(q,limit=limit)
        return [r['e'] for r in rs]

    def get_event(self,event_id:str):
        rs=self.driver.execute("MATCH (e:DetectionEvent {event_id:$event_id}) RETURN e", event_id=event_id)
        return rs[0]['e'] if rs else None

    def set_event_protected(self,event_id:str,protected:bool):
        self._run("protect","MATCH (e:DetectionEvent {event_id:$event_id}) SET e.protected=$protected",{"event_id":event_id,"protected":protected})

    def replay_outbox(self, limit:int=100):
        if not self.outbox: return 0
        count=0
        for id_, op, payload, _ in self.outbox.pending(limit):
            try:
                import json
                p=json.loads(payload)
                self.driver.execute(p['query'], **p['params'])
                self.outbox.mark_done(id_); count += 1
            except Exception as e:
                self.outbox.mark_failed(id_, str(e))
        return count
