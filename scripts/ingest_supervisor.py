#!/usr/bin/env python3
"""Dashcam ingestion supervisor: discovers corpus day-dirs lacking catalog
coverage, drives bulk_ingest_dashcam per month, verifies results, keeps a
resumable ledger, alerts after repeated failure, and rebuilds the docker
image when the repo advances. Designed for systemd timer every 6h."""
import fcntl, os, sys, json, glob, time, subprocess, concurrent.futures
from datetime import datetime, timezone

HOME='/home/deathstar'
STATE_DIR=f'{HOME}/recover_state'
LEDGER=f'{STATE_DIR}/ingest_ledger.json'
LOG=f'{STATE_DIR}/ingest_supervisor.log'
ALERTS=f'{STATE_DIR}/ingest_alerts.log'
CREDS=f'{STATE_DIR}/credentials.env'
REPO=f'{HOME}/git/auto-ingest'
ROOT=os.environ.get('DASHCAM_ROOT','/mnt/8TB_2025/fileserver/dashcam')
MAX_ATTEMPTS=3
MIN_AGE_S=1800          # skip day-dirs modified too recently (import in flight)

def log(msg):
    line=f"{datetime.now().isoformat(timespec='seconds')} {msg}"
    print(line); open(LOG,'a').write(line+"\n")

def alert(msg):
    log(f"ALERT {msg}")
    open(ALERTS,'a').write(f"{datetime.now().isoformat(timespec='seconds')} {msg}\n")

if os.path.exists(CREDS):
    for ln in open(CREDS):
        ln=ln.strip()
        if ln.startswith('export ') and '=' in ln:
            k,v=ln[7:].split('=',1); os.environ.setdefault(k,v.strip('"'))
os.environ.setdefault('NEO4J_PASSWORD',os.environ.get('NEO4J_PASS',''))

def load_ledger():
    if os.path.exists(LEDGER): return json.load(open(LEDGER))
    return {}
def save_ledger(l): json.dump(l,open(LEDGER,'w'),indent=1)

def discover_days():
    days={}
    for yd in sorted(glob.glob(f'{ROOT}/[0-9]'*1+'[0-9][0-9][0-9]')):
        year=os.path.basename(yd)
        if not year.isdigit(): continue
        for md in sorted(glob.glob(f'{yd}/[0-9][0-9]')):
            for dd in sorted(glob.glob(f'{md}/[0-9][0-9]')):
                has=any(glob.glob(f'{dd}/{pat}') for pat in ('*_metadata.csv','*.MP4','*.mp4'))
                if not has: continue
                age=time.time()-os.path.getmtime(dd)
                days[dd.replace(ROOT+'/','')]= {'age_ok':age>=MIN_AGE_S}
    return days

def neo4j_day_count(daykey):
    from neo4j import GraphDatabase
    d=GraphDatabase.driver('bolt://localhost:7687',auth=('neo4j',os.environ['NEO4J_PASSWORD']))
    with d.session() as s:
        n=s.run("MATCH (c:DashcamClip) WHERE c.key STARTS WITH $p RETURN count(c) AS n",p=daykey).single()['n']
    d.close(); return n

def image_fresh():
    try:
        sha=subprocess.check_output(['git','-C',REPO,'rev-parse','HEAD']).decode().strip()[:12]
        sha_file=f'{STATE_DIR}/.ingest_img_sha'
        cur=open(sha_file).read().strip() if os.path.exists(sha_file) else ''
        if sha!=cur:
            log(f"repo advanced ({cur or 'none'} -> {sha}); rebuilding image")
            rc=subprocess.run(['docker','compose','build','-q','ingest-service'],cwd=REPO).returncode
            if rc==0: open(sha_file,'w').write(sha); return True
            alert(f"image build failed rc={rc}"); return False
        return True
    except Exception as e:
        alert(f"image freshness check error: {e}"); return False

def main():
    _lf = open("/tmp/ingest_supervisor.lock", "w")
    try:
        fcntl.flock(_lf, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        log("another supervisor holds the lock; exiting")
        return 0
    ledger=load_ledger()
    days=discover_days()
    log(f"discovered {len(days)} day-dirs with content")
    if not days: log("nothing to do"); return 0
    if not image_fresh(): return 1

    pending={}
    for day,meta in days.items():
        st=ledger.get(day,{})
        if st.get('status')=='ok': continue
        if not meta['age_ok']: log(f"defer {day} (modified recently)"); continue
        ym=day.replace('/','_')[:7]
        pending.setdefault(ym,[]).append(day)
    if not pending: log("no pending days"); save_ledger(ledger); return 0

    env=dict(os.environ)
    PROVEN_CMD=[
        'nice','-n','10',
        'docker','run','--rm','--network','host',
        '-v','/mnt/8TB_2025/fileserver/dashcam:/dashcam',
        '-e','PYTHONPATH=/app',
        '-e','NEO4J_PASS',
        'auto-ingest:latest',
        'bash','-c',
        "python /app/auto_ingest/dashcam/yolo_embeddings.py "
        "--bases /dashcam/{DAY} "
        "--neo4j-uri bolt://127.0.0.1:7687 --neo4j-user neo4j "
        "--neo4j-pass '$NEO4J_PASS' --resume"
    ]
    par=max(1,int(os.environ.get("INGEST_MAX_PARALLEL","3")))
    log(f"ingesting {len(sum(pending.values(),[]))} pending day(s), parallelism={par}")
    tasks=[(d,dstr) for day,day_list in sorted(pending.items()) for dstr in day_list]
    REMOTE=os.environ.get("INGEST_REMOTE","").strip()
    def run_day(dstr):
        if REMOTE and hash(dstr)%2==1:
            r=subprocess.run(["sshpass","-p",os.environ.get("X1_370_PASS",""),
                              "ssh","-o","StrictHostKeyChecking=no",REMOTE,
                              "bash /opt/auto-ingest/run_day.sh "+dstr],
                             env=env,capture_output=True,text=True)
            return dstr,r.returncode
        cmd=[x.replace('{DAY}',dstr) for x in PROVEN_CMD]
        r=subprocess.run(cmd,env=env,capture_output=True,text=True)
        if 'AuthenticationRateLimit' in (r.stdout+r.stderr):
            log(f"{dstr}: auth rate-limited, cooling down 16min")
            time.sleep(960)
            r=subprocess.run(cmd,env=env,capture_output=True,text=True)
        return dstr,r.returncode
    with concurrent.futures.ThreadPoolExecutor(max_workers=par) as ex:
        futs={ex.submit(run_day,dstr): dstr for _,dstr in tasks}
        for fut in concurrent.futures.as_completed(futs):
            dstr,rc=fut.result(); dk=dstr.replace('/','_')
            st=ledger.setdefault(dstr,{'attempts':0})
            st['attempts']+=1; st['last_rc']=rc
            st['last_ts']=datetime.now(timezone.utc).isoformat()
            try: nodes=neo4j_day_count(dk)
            except Exception as e:
                nodes=-1; log(f"verify error {dstr}: {e}")
            st['nodes']=nodes
            if rc==0:
                st['status']='ok'; log(f"OK {dstr}: nodes={nodes}")
            else:
                st['status']='fail'
                if st['attempts']>=MAX_ATTEMPTS:
                    alert(f"day {dstr} failed {st['attempts']}x (rc={rc}, nodes={nodes})")
            save_ledger(ledger)
    fails=[d for d,s in ledger.items() if s.get('status')=='fail']
    log(f"done. fail-streak days: {len(fails)}")
    return 0

if __name__=='__main__': sys.exit(main())
