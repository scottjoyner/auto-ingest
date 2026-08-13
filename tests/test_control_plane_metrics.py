from auto_ingest.metrics import render_prometheus


def test_prometheus_render_includes_zero_states_and_core_gauges():
    text = render_prometheus(
        {
            "jobs_by_state": {"RUNNING": 2, "QUARANTINED": 1},
            "active_leases": 2,
            "stale_jobs": 1,
            "artifacts": 9,
        }
    )
    assert 'auto_ingest_jobs{state="RUNNING"} 2' in text
    assert 'auto_ingest_jobs{state="READY"} 0' in text
    assert 'auto_ingest_jobs{state="QUARANTINED"} 1' in text
    assert "auto_ingest_active_leases 2" in text
    assert "auto_ingest_stale_jobs 1" in text
    assert "auto_ingest_artifacts 9" in text
