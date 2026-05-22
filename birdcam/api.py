from fastapi import FastAPI, HTTPException

def create_app(repo):
    app=FastAPI(title="birdcam")
    @app.get('/health')
    def health(): return {"status":"ok"}
    @app.get('/cameras')
    def cameras(): return []
    @app.get('/cameras/{camera_id}')
    def camera(camera_id:str): return {"camera_id":camera_id}
    @app.get('/cameras/{camera_id}/health')
    def camera_health(camera_id:str): return []
    @app.get('/events')
    def events(limit:int=100): return repo.list_events(limit=limit)
    @app.get('/cameras/{camera_id}/events')
    def camera_events(camera_id:str, limit:int=100): return repo.list_events(camera_id=camera_id, limit=limit)
    @app.get('/events/{event_id}')
    def event(event_id:str):
        e=repo.get_event(event_id)
        if not e: raise HTTPException(404)
        return e
    @app.post('/events/{event_id}/protect')
    def protect(event_id:str): repo.set_event_protected(event_id, True); return {"ok":True}
    @app.post('/events/{event_id}/unprotect')
    def unprotect(event_id:str): repo.set_event_protected(event_id, False); return {"ok":True}
    return app
