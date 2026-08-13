from auto_ingest.orchestration import _fingerprint, default_job_key


def test_default_job_key_is_idempotent_within_profile_window():
    assert default_job_key("sync", now=900) == default_job_key("sync", now=1_199)
    assert default_job_key("full", now=3_600) == default_job_key("full", now=5_399)
    assert default_job_key("dashcam", now=86_400) == default_job_key("dashcam", now=172_799)


def test_default_job_key_advances_at_next_window():
    assert default_job_key("sync", now=1_499) != default_job_key("sync", now=1_500)
    assert default_job_key("full", now=3_599) != default_job_key("full", now=3_600)
    assert default_job_key("dashcam", now=86_399) != default_job_key("dashcam", now=86_400)


def test_failure_fingerprint_is_stable_and_task_sensitive():
    a = _fingerprint("RuntimeError", "boom", "copy")
    b = _fingerprint("RuntimeError", "boom", "copy")
    c = _fingerprint("RuntimeError", "boom", "transcribe")
    assert a == b
    assert a != c
    assert len(a) == 24
