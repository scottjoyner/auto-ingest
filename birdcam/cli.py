import argparse, uvicorn
from birdcam.config import load_settings
from birdcam.detector.yolo import YoloDetector
from birdcam.api import create_app
from birdcam.worker import Worker
from birdcam.graph import GraphDriver, DetectionGraphRepository, GraphOutbox, init_schema


def build_repo(s):
    driver=GraphDriver(s.neo4j.uri,s.neo4j.username,s.neo4j.password,s.neo4j.database)
    outbox=GraphOutbox(s.sqlite_path)
    repo=DetectionGraphRepository(driver, outbox, s.neo4j.use_outbox_on_failure)
    return driver, repo

def main():
    p=argparse.ArgumentParser(prog="birdcam"); sp=p.add_subparsers(dest="cmd", required=True)
    run=sp.add_parser("run"); run.add_argument("--config",required=True)
    det=sp.add_parser("detect-file"); det.add_argument("--input",required=True); det.add_argument("--config",required=True)
    api=sp.add_parser("api"); api.add_argument("--config",required=True); api.add_argument("--host",default="0.0.0.0"); api.add_argument("--port",type=int,default=8000)
    graph=sp.add_parser("graph"); gsp=graph.add_subparsers(dest="graph_cmd", required=True)
    gi=gsp.add_parser("init-schema"); gi.add_argument("--config",required=True)
    gc=gsp.add_parser("check"); gc.add_argument("--config",required=True)
    gr=gsp.add_parser("replay-outbox"); gr.add_argument("--config",required=True)
    gl=gsp.add_parser("list-events"); gl.add_argument("--config",required=True); gl.add_argument("--camera-id")
    ge=gsp.add_parser("event"); ge.add_argument("--config",required=True); ge.add_argument("--event-id",required=True)
    a=p.parse_args(); s=load_settings(getattr(a,'config'))
    driver, repo = build_repo(s)
    if a.cmd=="graph":
        if a.graph_cmd=="init-schema": init_schema(driver)
        elif a.graph_cmd=="check": print(driver.execute("RETURN 1 AS ok"))
        elif a.graph_cmd=="replay-outbox": print(repo.replay_outbox())
        elif a.graph_cmd=="list-events": print(repo.list_events(camera_id=a.camera_id))
        elif a.graph_cmd=="event": print(repo.get_event(a.event_id))
    elif a.cmd=="detect-file": Worker(s, YoloDetector(s.model_name_or_path,s.detection_class,s.confidence_threshold), repo).run_file(a.input)
    elif a.cmd=="api": uvicorn.run(create_app(repo), host=a.host, port=a.port)
    else: Worker(s, YoloDetector(s.model_name_or_path,s.detection_class,s.confidence_threshold), repo).run_file(s.stream_url)

if __name__ == '__main__': main()
