from auto_ingest.worker_loop import build_worker_tasks


def test_worker_tasks_do_not_put_passwords_on_argv(monkeypatch):
    monkeypatch.setenv("CONTENT", "1")
    tasks = build_worker_tasks()
    flat = [arg for task in tasks for arg in task.command]
    assert "--nextcloud-pass" not in flat
    assert all("password" not in str(arg).lower() for arg in flat)


def test_worker_tasks_include_canonical_resumable_stages(monkeypatch):
    monkeypatch.setenv("CONTENT", "1")
    monkeypatch.setenv("LINK_CHUNK", "123")
    tasks = build_worker_tasks()
    names = [task.name for task in tasks]
    assert names[:3] == ["speaker-link", "dashcam-compress", "content"]
    link = tasks[0].command
    assert "--state-file" in link
    assert "--max-speakers" in link
    assert "123" in link
