"""Actual Bolt COMMIT transport-loss acceptance, isolated fixture only."""
import ast
import datetime
import json
import pathlib
import secrets
import select
import signal
import socket
import subprocess
import tempfile
import threading
import time
import uuid
from typing import Any, Dict, List

from neo4j import GraphDatabase
from neo4j.exceptions import ResultNotSingleError
import verify_disposable_ingest_e2e as fixture


class CommitProxy:
    """Single connection relay. Arm only after prior results are consumed.

    Before mode discards a verified COMMIT frame. After mode forwards that
    frame and discards the actual server SUCCESS response. No credentials,
    handshake bytes or other payloads are recorded.
    """
    def __init__(self, target, mode):
        self.target, self.mode = target, mode
        self.armed = threading.Event()
        self.finished = threading.Event()
        self.evidence = []
        self.error = None
        self.listener = socket.socket()
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen(1)
        self.listener.settimeout(10)
        self.port = self.listener.getsockname()[1]
        self.thread = threading.Thread(target=self.relay, daemon=True)
        self.thread.start()

    def relay(self):
        try:
            with self.listener:
                client, _ = self.listener.accept()
                with client, socket.create_connection(("127.0.0.1", self.target), timeout=3) as server:
                    client.settimeout(5)
                    server.settimeout(5)
                    while True:
                        ready, _, _ = select.select([client,server],[],[],10)
                        if not ready:
                            raise TimeoutError("relay idle timeout")
                        for src in ready:
                            data = src.recv(65536)
                            if not data:
                                return
                            if self.armed.is_set() and src is client:
                                # Bolt chunk of length two, tiny struct COMMIT (signature 0x12).
                                while len(data)<6:
                                    part=src.recv(6-len(data))
                                    if not part: raise RuntimeError("incomplete commit frame")
                                    data+=part
                                if data != bytes.fromhex("0002b0120000"):
                                    raise RuntimeError("expected exact Bolt COMMIT frame")
                                self.evidence.append("COMMIT_FRAME_VERIFIED")
                                if self.mode=="before":
                                    self.evidence.append("COMMIT_DISCARDED")
                                    return
                                server.sendall(data)
                                self.evidence.append("COMMIT_FORWARDED")
                                # A real Bolt SUCCESS message has tiny-struct header b1 70.
                                # Read its complete chunked message, discarding it to client.
                                def exact(n):
                                    buf=b""
                                    while len(buf)<n:
                                        block=server.recv(n-len(buf))
                                        if not block:raise RuntimeError("server closed before response")
                                        buf+=block
                                    return buf
                                msg=b""
                                while True:
                                    n=int.from_bytes(exact(2),"big")
                                    if n==0:break
                                    msg+=exact(n)
                                    if len(msg)>65536:raise RuntimeError("oversize commit response")
                                if not msg.startswith(bytes.fromhex("b170")):
                                    raise RuntimeError("expected Bolt SUCCESS response")
                                self.evidence.append("SERVER_SUCCESS_DISCARDED")
                                return
                            (server if src is client else client).sendall(data)
        except Exception as exc:
            self.error=type(exc).__name__+": "+str(exc)
        finally:
            self.finished.set()

    def close(self):
        self.listener.close()
        self.thread.join(timeout=12)
        assert self.finished.is_set(), "relay did not terminate"
        assert self.error is None, self.error


class ArmTransaction:
    def __init__(self, tx, proxy):
        self.tx, self.proxy=tx,proxy
    def __enter__(self):
        self.tx.__enter__()
        return self
    def __exit__(self,*exc):
        return self.tx.__exit__(*exc)
    def run(self,*args,**kwargs):
        return self.tx.run(*args,**kwargs)
    def commit(self):
        self.proxy.armed.set()
        return self.tx.commit()


class ArmSession:
    def __init__(self, session, proxy):
        self.session,self.proxy=session,proxy
    def begin_transaction(self):
        return ArmTransaction(self.session.begin_transaction(),self.proxy)


def run_acceptance(driver, target, pw):
    def q(cy,**params):
        with driver.session() as s:
            return s.run(cy,**params).data()
    assert q("MATCH (n) RETURN count(n) AS n")==[{"n":0}]
    source=fixture.ROOT/"summarize_from_segments.py"
    names={"q_pick_ready","q_attach_run","q_finish_run"}
    nodes=[n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes}==names
    ns=dict(datetime=datetime,json=json,Any=Any,Dict=Dict,List=List)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),"exec"),ns)
    def runobj(rid):
        return dict(id=rid,started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),status="RUNNING",manifest={})
    def state(tid):
        return q("MATCH (t:Task {id:$tid}) OPTIONAL MATCH (t)-[:HAS_RUN]->(r:Run) RETURN t.status AS status,t.active_run_id AS owner,r.id AS rid,r.status AS rs",tid=tid)
    def attach(tid,rid):
        with driver.session() as s:return ns["q_attach_run"](s,tid,runobj(rid))
    def finish(tid,rid):
        with driver.session() as s:return ns["q_finish_run"](s,tid,rid,True,{})
    def lose(mode,operation):
        proxy=CommitProxy(target,mode)
        proxied=GraphDatabase.driver("bolt://127.0.0.1:"+str(proxy.port),auth=("neo4j",pw),
                  connection_timeout=3,connection_acquisition_timeout=5,max_transaction_retry_time=0)
        err=None
        try:
            with proxied.session() as s:
                try:operation(ArmSession(s,proxy))
                except Exception as exc:err=type(exc).__name__
            assert err, "client unexpectedly received commit acknowledgement"
        finally:
            proxied.close()
            proxy.close()
        expected=["COMMIT_FRAME_VERIFIED","COMMIT_DISCARDED"] if mode=="before" else ["COMMIT_FRAME_VERIFIED","COMMIT_FORWARDED","SERVER_SUCCESS_DISCARDED"]
        assert proxy.evidence==expected,proxy.evidence
        print("TRANSPORT",mode,err,proxy.evidence,flush=True)
    for mode in ("before","after"):
        tid="claim-"+mode;rid="run-"+tid
        q("CREATE (:Task {id:$tid,status:'READY',updated_at:datetime()})",tid=tid)
        lose(mode,lambda s:ns["q_attach_run"](s,tid,runobj(rid)))
        if mode=="before":
            assert state(tid)==[dict(status="READY",owner=None,rid=None,rs=None)]
            attach(tid,rid)
        else:
            assert state(tid)==[dict(status="RUNNING",owner=rid,rid=rid,rs="RUNNING")]
            try:attach(tid,"foreign-"+rid)
            except ResultNotSingleError:pass
            else:raise AssertionError("duplicate claim accepted")
        finish(tid,rid)
        assert state(tid)==[dict(status="DONE",owner=None,rid=rid,rs="DONE")]
        print("PASS claim",mode,"commit loss",flush=True)
    for mode in ("before","after"):
        tid="finish-"+mode;rid="run-"+tid
        q("CREATE (:Task {id:$tid,status:'READY',updated_at:datetime()})",tid=tid)
        attach(tid,rid)
        lose(mode,lambda s:ns["q_finish_run"](s,tid,rid,True,{}))
        if mode=="before":
            assert state(tid)==[dict(status="RUNNING",owner=rid,rid=rid,rs="RUNNING")]
            finish(tid,rid)
        else:
            assert state(tid)==[dict(status="DONE",owner=None,rid=rid,rs="DONE")]
            try:finish(tid,rid)
            except ResultNotSingleError:pass
            else:raise AssertionError("duplicate finish accepted")
        assert state(tid)==[dict(status="DONE",owner=None,rid=rid,rs="DONE")]
        print("PASS finalization",mode,"commit loss",flush=True)
    assert q("MATCH (r:Run) RETURN count(r) AS n")==[{"n":4}]
    print("ACTUAL_BOLT_TRANSPORT_PASS 4/4",flush=True)


def main():
    fixture.guard()
    def interrupted(*args):raise TimeoutError("fixture deadline")
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGALRM,interrupted)
    signal.alarm(135)
    cid=None;driver=None;pw=secrets.token_hex(16)
    cmd=fixture.cmd
    try:
        cid=cmd("docker","run","--detach","--rm","--pull=never","--name","muse-transport-"+uuid.uuid4().hex[:12],
                "--memory=2g","--memory-swap=2g","--cpus=1.5","--pids-limit=384","--publish","127.0.0.1::7687",
                "--env","NEO4J_AUTH=neo4j/"+pw,"--env","NEO4J_server_memory_heap_initial__size=256m",
                "--env","NEO4J_server_memory_heap_max__size=384m","--env","NEO4J_server_memory_pagecache_size=128m",fixture.IMAGE,timeout=25)
        info=json.loads(cmd("docker","inspect",cid))[0]
        assert info["HostConfig"]["Memory"]==2*1024**3 and info["HostConfig"]["NanoCpus"]==1500000000
        assert not any(x["Type"]=="bind" for x in info["Mounts"])
        binds=info["NetworkSettings"]["Ports"]["7687/tcp"]
        assert len(binds)==1 and binds[0]["HostIp"]=="127.0.0.1"
        port=int(binds[0]["HostPort"])
        print("ISOLATED",cid[:12],flush=True)
        driver=GraphDatabase.driver("bolt://127.0.0.1:"+str(port),auth=("neo4j",pw),connection_timeout=3,connection_acquisition_timeout=5,max_transaction_retry_time=0)
        deadline=time.monotonic()+65
        while True:
            try:driver.verify_connectivity();break
            except Exception:
                if time.monotonic()>=deadline:raise
                time.sleep(2)
        run_acceptance(driver,port,pw)
    finally:
        if driver:driver.close()
        if cid:
            cmd("docker","stop","--time","5",cid,timeout=15)
            deadline=time.monotonic()+10
            while subprocess.run(["docker","container","inspect",cid],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=3).returncode==0:
                if time.monotonic()>deadline:raise RuntimeError("fixture remains")
                time.sleep(.5)
            print("FIXTURE_ABSENT",cid[:12],flush=True)
        signal.alarm(0)


if __name__=="__main__":
    main()
