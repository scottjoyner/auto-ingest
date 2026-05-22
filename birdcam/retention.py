from pathlib import Path
from datetime import datetime, timedelta

def apply_retention(root:str, max_gb:float, max_age_days:int, repo=None):
    rootp=Path(root); now=datetime.utcnow(); max_bytes=int(max_gb*(1024**3))
    files=sorted([p for p in rootp.glob("clips/**/*.mp4") if p.is_file()], key=lambda p:p.stat().st_mtime)
    removed=[]
    for p in list(files):
        if datetime.utcfromtimestamp(p.stat().st_mtime) < now - timedelta(days=max_age_days):
            p.unlink(missing_ok=True); removed.append(str(p))
            if repo: repo.driver.execute("MATCH (c:Clip {path:$path}) SET c.deleted=true,c.deleted_at=$ts,c.deletion_reason='max_age'", path=str(p), ts=now.isoformat())
    files=sorted([p for p in rootp.glob("clips/**/*.mp4") if p.is_file()], key=lambda p:p.stat().st_mtime)
    total=sum(p.stat().st_size for p in files)
    for p in files:
        if total<=max_bytes: break
        sz=p.stat().st_size; p.unlink(missing_ok=True); total-=sz; removed.append(str(p))
        if repo: repo.driver.execute("MATCH (c:Clip {path:$path}) SET c.deleted=true,c.deleted_at=$ts,c.deletion_reason='max_size'", path=str(p), ts=now.isoformat())
    return removed
