from auto_ingest.resources import ResourcePolicy, ResourceSnapshot, admission


def test_resource_admission_accepts_healthy_host():
    snap = ResourceSnapshot(cpu_count=8, load1=2.0, memory_available_mb=8192, disk_free_gb=200)
    allowed, reasons = admission(snap, ResourcePolicy())
    assert allowed is True
    assert reasons == []


def test_resource_admission_reports_all_pressure_dimensions():
    snap = ResourceSnapshot(cpu_count=4, load1=3.0, memory_available_mb=512, disk_free_gb=2)
    allowed, reasons = admission(snap, ResourcePolicy())
    assert allowed is False
    assert len(reasons) == 3
    assert any("load_per_cpu" in r for r in reasons)
    assert any("memory_available_mb" in r for r in reasons)
    assert any("disk_free_gb" in r for r in reasons)
