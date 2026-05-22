from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import os, yaml

@dataclass
class ROI: x:int=0; y:int=0; width:int=0; height:int=0
@dataclass
class RetentionConfig: max_gb:float=20; max_age_days:int=30
@dataclass
class DebugConfig: save_annotated_frames:bool=False
@dataclass
class Neo4jConfig:
    enabled: bool = True
    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = "change-me"
    database: str = "neo4j"
    create_schema: bool = True
    batch_size: int = 100
    retry_attempts: int = 5
    retry_backoff_seconds: int = 2
    use_outbox_on_failure: bool = True
@dataclass
class Settings:
    camera_id:str; stream_url:str; storage_root:str
    model_name_or_path:str="yolov8n.pt"; detection_class:str="bird"; confidence_threshold:float=0.4
    inference_interval_ms:int=400; pre_roll_seconds:int=3; post_roll_seconds:int=3
    event_merge_seconds:int=8; cooldown_seconds:int=10; detection_persistence_frames:int=2
    min_clip_duration_seconds:int=2; max_clip_duration_seconds:int=60
    roi: ROI = field(default_factory=ROI)
    sqlite_path:str="birdcam.sqlite3"; retention:RetentionConfig=field(default_factory=RetentionConfig)
    debug:DebugConfig=field(default_factory=DebugConfig); neo4j:Neo4jConfig=field(default_factory=Neo4jConfig)

def load_settings(path:str)->Settings:
    p=yaml.safe_load(Path(path).read_text()) or {}
    if os.getenv("NEO4J_URI"): p.setdefault("neo4j",{})["uri"]=os.environ["NEO4J_URI"]
    if os.getenv("NEO4J_USERNAME"): p.setdefault("neo4j",{})["username"]=os.environ["NEO4J_USERNAME"]
    if os.getenv("NEO4J_PASSWORD"): p.setdefault("neo4j",{})["password"]=os.environ["NEO4J_PASSWORD"]
    p["roi"] = ROI(**p.get("roi", {})); p["retention"] = RetentionConfig(**p.get("retention", {}))
    p["debug"] = DebugConfig(**p.get("debug", {})); p["neo4j"] = Neo4jConfig(**p.get("neo4j", {}))
    if not p.get("sqlite_path"): p["sqlite_path"]=str(Path(p["storage_root"])/"birdcam-state.sqlite3")
    return Settings(**p)
