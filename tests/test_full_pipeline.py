from auto_ingest.full_pipeline import build_tasks, default_job_key


def test_full_pipeline_is_split_into_resumable_heavy_stages():
    tasks = build_tasks()
    names = [task.name for task in tasks]
    assert names == [
        "copy-audio",
        "copy-dashcam",
        "copy-bodycam",
        "diarize",
        "transcript-ingest",
        "speaker-reconcile",
        "music-segments",
        "lyrics-classification",
        "speaker-link",
        "yolo-embeddings",
    ]
    assert len(set(names)) == len(names)


def test_full_pipeline_commands_do_not_expose_database_passwords():
    tasks = build_tasks()
    argv = [str(arg) for task in tasks for arg in task.command]
    assert "--neo4j-password" not in argv
    assert "--neo4j-pass" not in argv
    assert all("knowledge_graph_2026" not in arg for arg in argv)
    assert all("100.64." not in arg for arg in argv)
    yolo = next(task for task in tasks if task.name == "yolo-embeddings")
    assert "auto_ingest.yolo_entrypoint" in yolo.command


def test_full_pipeline_job_key_is_window_idempotent(monkeypatch):
    monkeypatch.setenv("FULL_PIPELINE_WINDOW_SEC", "1800")
    assert default_job_key(3600) == default_job_key(5399)
    assert default_job_key(5399) != default_job_key(5400)
