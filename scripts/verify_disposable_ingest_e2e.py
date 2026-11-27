"""Disposable local acceptance: generated media, deterministic transcript, real Neo4j.

Never connects to a supplied URI. Requires already-cached Neo4j image, ffmpeg,
Python driver and a locally admitted Docker host. No network inference.
"""
import ast
import datetime
import hashlib
import json
import os
import pathlib
import re
import secrets
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
import uuid
from typing import Any, Dict, List, Optional

from neo4j import GraphDatabase

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from auto_ingest.ingest_write import write_intermediate
from auto_ingest.ingest_import import import_all
from auto_ingest.executor_evidence import inspect_local_output
from auto_ingest.executor_reconcile import inspect_tasks
from auto_ingest.executor_cutover import inspect_cutover

IMAGE = "sha256:468584de930e2da70f06a23c9faa19fe85c14b18cca4ac6b2c888fe9d7d7578e"


DOCKER_SOCKET = "/var/run/docker.sock"


def cmd(*args, timeout=15):
    if args and args[0] == "docker":
        # Always address the local daemon explicitly, including cleanup. Reject
        # environment overrides rather than silently accepting a remote context.
        if any(os.environ.get(key) for key in ("DOCKER_HOST", "DOCKER_CONTEXT")):
            raise RuntimeError("Docker endpoint overrides are not admitted")
        if not stat.S_ISSOCK(os.stat(DOCKER_SOCKET).st_mode):
            raise RuntimeError("local Docker socket unavailable")
        args = ("docker", "--host", "unix://" + DOCKER_SOCKET, *args[1:])
    return subprocess.check_output(args, text=True, stderr=subprocess.PIPE, timeout=timeout).strip()


def fixture_present(cid):
    """Confirm absence only after a successful local daemon inventory call."""
    if not re.fullmatch(r"[a-f0-9]{64}", cid):
        raise ValueError("invalid fixture container identity")
    result = cmd("docker", "ps", "--all", "--no-trunc", "--filter", "id=" + cid,
                 "--format", "{{.ID}}", timeout=5)
    if result not in ("", cid):
        raise RuntimeError("unexpected fixture inventory response")
    return result == cid


def guard():
    if not __debug__:
        raise RuntimeError("fixture acceptance requires Python assertions enabled")
    m = {x.split(':')[0]: int(x.split()[1]) for x in open('/proc/meminfo')
         if x.startswith(('MemAvailable:', 'SwapFree:'))}
    assert m['MemAvailable'] >= 8*1024*1024 and m['SwapFree'] >= 1024*1024
    assert shutil.disk_usage('/tmp').free >= 10*1024**3
    for p in ('memory', 'io'):
        full = next(x for x in open('/proc/pressure/'+p) if x.startswith('full '))
        assert float(full.split()[2].split('=')[1]) < 5.0
    assert cmd('docker','image','inspect',IMAGE,'--format','{{.Id}}') == IMAGE
    print('ADMISSION_PASS', cmd('hostname'), m, flush=True)


def execute_fixture(driver, root):
    def q(cy, **params):
        with driver.session() as s:
            return s.run(cy, **params).data()
    assert q('MATCH (n) RETURN count(n) AS n') == [{'n': 0}]
    media = root / 'clip.wav'
    cmd('ffmpeg','-nostdin','-hide_banner','-loglevel','error','-threads','1',
        '-f','lavfi','-i','sine=frequency=440:duration=1','-ar','16000',str(media))
    duration = float(cmd('ffprobe','-v','error','-show_entries','format=duration',
                         '-of','default=noprint_wrappers=1:nokey=1',str(media)))
    assert 0.99 <= duration <= 1.01
    digest = hashlib.sha256(media.read_bytes()).hexdigest()
    key = 'fixture/clip'
    intermediate = root / 'intermediates'
    write_intermediate(str(intermediate),key,'segments',[
        dict(id='s1',parent='clip',start=0.0,end=0.5,text='Synthetic check one'),
        dict(id='s2',parent='clip',start=0.5,end=1.0,text='Synthetic check two')])
    write_intermediate(str(intermediate),key,'transcription',[
        dict(id='clip',source_sha256=digest,duration=duration)])
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in intermediate.iterdir()}
    expected = {key+'/transcription':1,key+'/segments':2}
    for _ in range(2):
        assert import_all(str(intermediate),driver,key=key,batch=1) == expected
    assert q("MATCH (t:Transcription)-[:HAS_SEGMENT]->(s:Segment) RETURN t.source_sha256 AS sha,s.id AS sid ORDER BY sid") == [dict(sha=digest,sid='s1'),dict(sha=digest,sid='s2')]
    assert q("MATCH (t:Transcription) RETURN count(t) AS n") == [{'n':1}]
    assert q("MATCH (s:Segment) RETURN count(s) AS n") == [{'n':2}]
    print('PASS media provenance + repeated real graph import',digest,duration,flush=True)

    # AST selects production functions without importing model/network configuration.
    source = ROOT/'summarize_from_segments.py'
    names = {'_serialize_tasks_for_db','write_back','fetch_segments_for','build_transcript',
             'q_get_task','q_approve_all_review','approve_all','q_pick_ready','q_attach_run',
             'q_finish_run','ensure_artifacts','execute_task_minimal','execute_ready',
             'replace_placeholders','acc_file_exists','acc_contains','acc_regex','run_acceptance'}
    nodes = [n for n in ast.parse(source.read_text()).body
             if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == names
    ns = dict(globals(),ARTIFACTS_DIR=str(root/'artifacts'),neo_driver=lambda:driver)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),ns)
    ns['ACCEPTANCE_FUNCS'] = {k:ns['acc_'+k] for k in ('file_exists','contains','regex')}
    with driver.session() as s:
        transcript = ns['build_transcript'](**{k:v for k,v in ns['fetch_segments_for'](s,{'id':'clip'}).items() if k in ('segments','utterances')})
        assert transcript == 'Synthetic check one\nSynthetic check two'
        # Deterministic stand-in for model output. No ASR or LLM called.
        final = {'summary':transcript,'bullets':['fixture only'],'tasks':[
          {'title':'success fixture','description':'local artifact','acceptance':[
           {'type':'file_exists','args':{'path':'artifacts/{TASK_ID}/output.txt'}},
           {'type':'contains','args':{'path':'artifacts/{TASK_ID}/output.txt','text':'success fixture'}}]},
          {'title':'failure fixture','description':'local artifact','acceptance':[
           {'type':'contains','args':{'path':'artifacts/{TASK_ID}/output.txt','text':'not present in artifact'}}]}]}
        sid = ns['write_back'](s,'clip',final,write_notes=True)
        assert sid
    rows=q("MATCH (t:Transcription {id:'clip'})-[:HAS_SUMMARY]->(s:Summary)-[:GENERATED_TASK]->(k:Task) RETURN k.id AS id,k.status AS status,k.title AS title ORDER BY title")
    assert len(rows)==2 and all(x['status']=='REVIEW' for x in rows)
    print('PASS transcript -> actual summary + two REVIEW tasks',flush=True)
    with driver.session() as s:
        assert ns['approve_all'](s,2)==2
    ns['execute_ready'](2)
    actual=q("MATCH (t:Transcription {id:'clip'})-[:HAS_SUMMARY]->(:Summary)-[:GENERATED_TASK]->(k:Task)-[:HAS_RUN]->(r:Run) RETURN k.id AS id,k.status AS status,k.title AS title,k.active_run_id AS owner,r.id AS rid,r.status AS rs,r.manifest_json AS manifest ORDER BY title")
    assert len(actual)==2
    assert [x['status'] for x in actual]==['FAILED','DONE']
    for x in actual:
        assert x['owner'] is None and x['status']==x['rs']
        d=root/'artifacts'/x['id']
        data=(d/('run-'+x['rid']+'.txt')).read_bytes()
        receipt=(d/('run-'+x['rid']+'.json')).read_bytes()
        assert data==(d/'output.txt').read_bytes()
        assert inspect_local_output(receipt,data,task_id=x['id'],run_id=x['rid'])['classification']=='LOCAL_OUTPUT_BYTES_MATCH'
        results=json.loads(x['manifest'])['acceptance_results']
        assert results and all(r['passed'] for r in results)==(x['status']=='DONE')
    print('PASS actual REVIEW -> READY -> RUNNING -> DONE/FAILED + 2 verified local receipts',flush=True)
    ns['execute_ready'](2)
    assert q("MATCH (r:Run) RETURN count(r) AS n")==[{'n':2}]
    assert {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in intermediate.iterdir()}==before
    report=inspect_tasks(driver,[x['id'] for x in actual],now=datetime.datetime.now(datetime.timezone.utc),stale_after=datetime.timedelta(hours=1))
    assert all(x['classification']=='NOT_RUNNING' and not x['replay_allowed'] for x in report)
    for cy in ('CREATE INDEX FOR (t:Task) ON (t.status)','CREATE INDEX FOR (t:Task) ON (t.active_run_id)','CREATE INDEX FOR (r:Run) ON (r.status)'):q(cy)
    q('CALL db.awaitIndexes(20)')
    report=inspect_cutover(driver)
    assert report['graph_status']=='NO_BLOCKERS_OBSERVED' and not report['activation_allowed']
    print('PASS no duplicate execution, immutable intermediates, reconciled terminal states, cutover denies activation',flush=True)


def main():
    guard()
    def interrupted(*args):
        raise TimeoutError('fixture deadline')
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGALRM,interrupted)
    signal.alarm(135)
    cid=None
    driver=None
    pw=secrets.token_hex(16)
    try:
        cid=cmd('docker','run','--detach','--rm','--pull=never','--name','muse-e2e-'+uuid.uuid4().hex[:12],
                '--memory=2g','--memory-swap=2g','--cpus=1.5','--pids-limit=384','--publish','127.0.0.1::7687',
                '--env','NEO4J_AUTH=neo4j/'+pw,'--env','NEO4J_server_memory_heap_initial__size=256m',
                '--env','NEO4J_server_memory_heap_max__size=384m','--env','NEO4J_server_memory_pagecache_size=128m',IMAGE,timeout=25)
        info=json.loads(cmd('docker','inspect',cid))[0]
        assert info['HostConfig']['Memory']==2*1024**3 and info['HostConfig']['NanoCpus']==1500000000
        assert not any(x['Type']=='bind' for x in info['Mounts'])
        binds=info['NetworkSettings']['Ports']['7687/tcp']
        assert len(binds)==1 and binds[0]['HostIp']=='127.0.0.1'
        print('ISOLATED',cid[:12],flush=True)
        driver=GraphDatabase.driver('bolt://127.0.0.1:'+binds[0]['HostPort'],auth=('neo4j',pw),
            connection_timeout=2,connection_acquisition_timeout=5,max_transaction_retry_time=0)
        deadline=time.monotonic()+65
        while True:
            try: driver.verify_connectivity();break
            except Exception:
                if time.monotonic()>=deadline:raise
                time.sleep(2)
        with tempfile.TemporaryDirectory(prefix='muse-media-e2e-') as tmp:
            execute_fixture(driver,pathlib.Path(tmp))
        print('END_TO_END_PASS',flush=True)
    finally:
        if driver:driver.close()
        if cid:
            cmd('docker','stop','--time','5',cid,timeout=15)
            deadline=time.monotonic()+10
            while fixture_present(cid):
                if time.monotonic()>deadline:raise RuntimeError('fixture remains')
                time.sleep(.5)
            print('FIXTURE_ABSENT',cid[:12],flush=True)
        signal.alarm(0)


if __name__=='__main__':
    main()
