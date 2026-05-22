from __future__ import annotations
from datetime import datetime
from birdcam.config import Settings
from birdcam.stream import VideoStream
from birdcam.events import EventManager
from birdcam.storage import Storage
from birdcam.models import EventSummary
from birdcam.recorder import write_clip

class Worker:
    def __init__(self, settings: Settings, detector, repo):
        self.s=settings; self.detector=detector; self.repo=repo
        self.events=EventManager(settings.event_merge_seconds, settings.cooldown_seconds, settings.detection_persistence_frames)
        self.storage=Storage(settings.storage_root)
        self.repo.upsert_camera({"camera_id":settings.camera_id,"enabled":True,"stream_type":"rtsp_or_file","stream_url_redacted":"redacted"})

    def run_file(self, input_path:str, max_frames:int|None=None):
        stream=VideoStream(input_path, buffer_seconds=self.s.pre_roll_seconds+self.s.post_roll_seconds)
        i=0
        for frame, buf in stream.frames():
            now=datetime.utcnow(); detections=self.detector.detect(frame)
            state, evt=self.events.update(now, detections)
            if state=="active" and evt and len(evt.detections)==len(detections):
                self.repo.create_detection_event(EventSummary(evt.id, self.s.camera_id, evt.start, evt.last_seen,0,0,0,"","",""))
            if state=="active" and evt:
                for idx,d in enumerate(detections):
                    self.repo.add_detection({"detection_id":f"{evt.id}-{i}-{idx}","event_id":evt.id,"camera_id":self.s.camera_id,"timestamp":now.isoformat(),"epoch_ms":int(now.timestamp()*1000),"frame_index":i,"class_name":d.label,"class_id":-1,"confidence":d.confidence,"bbox_xyxy":list(d.bbox),"bbox_xywh":[d.bbox[0],d.bbox[1],d.bbox[2]-d.bbox[0],d.bbox[3]-d.bbox[1]],"roi_applied":False,"model_name":self.s.model_name_or_path,"created_at":now.isoformat()})
            if state=="finalized" and evt: self._finalize(evt,buf)
            i+=1
            if max_frames and i>=max_frames: break

    def _finalize(self, evt, buf):
        clip,thumb,meta=self.storage.event_paths(evt.id, evt.start)
        write_clip(str(clip), buf); self.storage.write_thumbnail(thumb, buf[-1])
        self.repo.attach_clip(evt.id,{"clip_id":evt.id,"event_id":evt.id,"camera_id":self.s.camera_id,"path":str(clip),"relative_path":str(clip),"storage_root":self.s.storage_root,"codec":"h264","container":"mp4","width":buf[-1].shape[1],"height":buf[-1].shape[0],"fps":10,"duration_seconds":len(buf)/10.0,"size_bytes":clip.stat().st_size if clip.exists() else 0,"sha256":"","created_at":datetime.utcnow().isoformat(),"protected":False})
        self.repo.attach_thumbnail(evt.id,{"thumbnail_id":evt.id,"event_id":evt.id,"camera_id":self.s.camera_id,"path":str(thumb),"relative_path":str(thumb),"width":buf[-1].shape[1],"height":buf[-1].shape[0],"size_bytes":thumb.stat().st_size if thumb.exists() else 0,"sha256":"","created_at":datetime.utcnow().isoformat()})
