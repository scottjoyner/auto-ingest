from pathlib import Path

import pytest

from content_os import core
from content_os.models import ContentObject


def make_root(tmp_path: Path) -> Path:
    root = tmp_path / "content-os"
    core.init(root)
    return root


def test_init_creates_expected_structure(tmp_path):
    root = make_root(tmp_path)
    for rel in [
        "strategy/positioning.md",
        "voice/master-avoid-slop.md",
        "stores/inbox.md",
        "runs/active",
        "modules/writer/SKILL.md",
        "workflows/feedback-loop.md",
        "config.yaml",
    ]:
        assert (root / rel).exists()


def test_init_does_not_overwrite_files(tmp_path):
    root = tmp_path / "content-os"
    core.init(root)
    target = root / "strategy" / "positioning.md"
    target.write_text("custom", encoding="utf-8")
    core.init(root)
    assert target.read_text(encoding="utf-8") == "custom"


def test_capture_can_append_to_inbox_without_run(tmp_path):
    root = make_root(tmp_path)
    path = core.capture(
        root, "Inbox only", "raw note", None, "auto", create_run_folder=False
    )
    assert path is None
    assert "Inbox only" in (root / "stores" / "inbox.md").read_text(encoding="utf-8")
    assert not list((root / "runs" / "active").glob("2026-05-inbox-only*"))


def test_research_route_feeds_candidate_ideas_store(tmp_path):
    root = make_root(tmp_path)
    path = core.create_run(
        root, "Research ideas", text="research candidate ideas about onboarding"
    )
    core.route_run(root, path.name)
    assert (root / "stores" / "ideas" / f"{path.name}.md").exists()


def test_forced_failed_approval_requires_reason(tmp_path):
    root = make_root(tmp_path)
    path = core.create_run(root, "Fail without reason", "ORIGINAL")
    (path / "draft-package.md").write_text("TODO groundbreaking", encoding="utf-8")
    core.verify(root, path.name)
    with pytest.raises(ValueError):
        core.approve(root, path.name, force=True)


def test_new_run_creates_valid_content_object(tmp_path):
    root = make_root(tmp_path)
    path = core.create_run(root, "A Test Idea", "ORIGINAL", "x_thread")
    obj = core.load_object(path)
    assert obj.title == "A Test Idea"
    assert obj.state == "idea_review"
    assert (path / "idea.md").exists()


def test_state_transition_validation():
    obj = ContentObject(id="x", title="X", slug="x")
    obj.transition("brief_ready")
    with pytest.raises(ValueError):
        obj.transition("idea_review")


def test_route_heuristic_behavior(tmp_path):
    root = make_root(tmp_path)
    path = core.create_run(
        root, "Research ideas", text="research candidate ideas about onboarding"
    )
    reason = core.route_run(root, path.name)
    assert core.load_object(path).route == "RESEARCH_IDEATE"
    assert "Heuristic" in reason


def test_brief_generation_includes_writer_context_packet_sections(tmp_path):
    root = make_root(tmp_path)
    path = core.create_run(root, "Brief me", "ORIGINAL")
    core.make_brief(root, path.name)
    text = (path / "brief.md").read_text(encoding="utf-8")
    for section in [
        "## thesis",
        "## reader",
        "## proof",
        "## angle",
        "## constraints",
        "## voice anchors",
        "## risks",
        "## open loops",
    ]:
        assert section in text


def test_verify_scorecard_calculates_correct_totals(tmp_path):
    root = make_root(tmp_path)
    path = core.create_run(root, "Score", "ORIGINAL")
    (path / "draft-package.md").write_text(
        "# Draft\n\nA checklist workflow template for a founder reader to apply without help. It uses proof, an example, screenshot visual, artifact, and metric.",
        encoding="utf-8",
    )
    core.verify(root, path.name)
    assert core.load_object(path).score == 12
    assert "12/12" in (path / "verification.md").read_text(encoding="utf-8")


def test_avoid_slop_catches_seeded_banned_phrases():
    issues = core.avoid_slop(
        "This is groundbreaking and experts believe it is a pivotal moment."
    )
    assert any("groundbreaking" in i for i in issues)
    assert any("experts believe" in i for i in issues)


def test_approve_refuses_failed_verification_unless_force(tmp_path):
    root = make_root(tmp_path)
    path = core.create_run(root, "Fail", "ORIGINAL")
    (path / "draft-package.md").write_text("TODO groundbreaking", encoding="utf-8")
    core.verify(root, path.name)
    with pytest.raises(ValueError):
        core.approve(root, path.name)
    core.approve(root, path.name, force=True, reason="manual exception")
    assert core.load_object(path).human_approved is True


def test_scheduler_handoff_refuses_unapproved_runs(tmp_path):
    root = make_root(tmp_path)
    path = core.create_run(root, "No approval", "ORIGINAL")
    with pytest.raises(ValueError):
        core.scheduler_handoff(root, path.name, "x")


def test_feedback_calculates_bookmark_rate_correctly(tmp_path):
    root = make_root(tmp_path)
    path = core.create_run(root, "Feedback", "ORIGINAL")
    rates = core.feedback(
        root, path.name, views=1000, bookmarks=50, likes=10, replies=5, reposts=5
    )
    assert rates["bookmark_rate"] == 0.05
    assert (root / "stores" / "winners" / f"{path.name}.md").exists()


def test_archive_moves_run_folder_correctly(tmp_path):
    root = make_root(tmp_path)
    path = core.create_run(root, "Archive", "ORIGINAL")
    obj = core.load_object(path)
    obj.state = "approved"
    obj.human_approved = True
    core.save_object(path, obj)
    dest = core.archive(root, path.name)
    assert dest.exists()
    assert not path.exists()


def test_doctor_detects_missing_files(tmp_path):
    root = make_root(tmp_path)
    (root / "voice" / "master-avoid-slop.md").unlink()
    issues = core.doctor(root)
    assert any("master-avoid-slop" in i for i in issues)


def test_scan_inputs_detects_transcript_json(tmp_path):
    root = make_root(tmp_path)
    source = tmp_path / "segments.json"
    source.write_text(
        '{"title":"Interview","segments":[{"start":0,"text":"First lesson"},{"start":5,"text":"Second lesson"}]}',
        encoding="utf-8",
    )
    docs, _ = core.scan_inputs(root, source)
    assert docs[0].source_type == "transcript_json"
    assert "First lesson" in docs[0].text


def test_ingest_source_creates_proof_and_run(tmp_path):
    root = make_root(tmp_path)
    source = tmp_path / "summary.md"
    source.write_text(
        "# Summary\n\nA local autoingest lesson with reusable proof.", encoding="utf-8"
    )
    run_path, proof_path, source_doc = core.import_source(
        root, source, title="Autoingest lesson"
    )
    assert source_doc.source_type == "summary"
    assert proof_path is not None and proof_path.exists()
    assert run_path is not None and (run_path / "content-object.yaml").exists()
    obj = core.load_object(run_path)
    assert obj.route == "REPURPOSE"
    assert obj.attribution["source_path"] == str(source)


def test_scan_inputs_writes_report(tmp_path):
    root = make_root(tmp_path)
    source = tmp_path / "note.txt"
    source.write_text("plain text note", encoding="utf-8")
    out = tmp_path / "scan.md"
    docs, report_path = core.scan_inputs(root, tmp_path, out=out, limit=5)
    assert docs
    assert report_path == out
    report = out.read_text(encoding="utf-8")
    assert "Input Scan Report" in report
    assert "Source ID" in report
    assert "Adapter" in report


def test_proof_path_for_source_does_not_overwrite_existing(tmp_path):
    root = make_root(tmp_path)
    from content_os.adapters import SourceDocument

    doc = SourceDocument(
        path=tmp_path / "note.md",
        title="Same Proof",
        source_type="text_note",
        text="one",
    )
    first = core.proof_path_for_source(root, doc)
    first.parent.mkdir(parents=True, exist_ok=True)
    first.write_text("existing", encoding="utf-8")
    second = core.proof_path_for_source(root, doc)
    assert second != first
    assert first.read_text(encoding="utf-8") == "existing"


def test_scheduler_handoff_can_be_regenerated_after_scheduler_ready(tmp_path):
    root = make_root(tmp_path)
    path = core.create_run(root, "Ready again", "ORIGINAL")
    (path / "draft-package.md").write_text(
        "# Draft\n\nA checklist workflow template for a founder reader to apply without help. It uses proof, an example, screenshot visual, artifact, and metric.",
        encoding="utf-8",
    )
    core.verify(root, path.name)
    core.approve(root, path.name)
    first = core.scheduler_handoff(root, path.name, "x")
    second = core.scheduler_handoff(root, path.name, "linkedin")
    assert first == second
    assert "linkedin" in second.read_text(encoding="utf-8")


def test_postiz_export_backs_up_existing_payload(tmp_path):
    root = make_root(tmp_path)
    path = core.create_run(root, "Export backup", "ORIGINAL")
    obj = core.load_object(path)
    obj.state = "approved"
    obj.human_approved = True
    core.save_object(path, obj)
    (path / "draft-package.md").write_text("approved copy", encoding="utf-8")
    out = tmp_path / "payload.json"
    out.write_text("old", encoding="utf-8")
    core.postiz_export(root, path.name, out)
    assert out.with_suffix(".json.bak").read_text(encoding="utf-8") == "old"


def test_init_writes_dynamic_source_adapter_rules(tmp_path):
    root = make_root(tmp_path)
    rules_path = root / "integrations" / "source-adapters.json"
    assert rules_path.exists()
    rules = core.load_adapter_rules(root)
    assert any(rule.name == "autoingest-cypher-note" for rule in rules)


def test_import_source_records_manifest_and_hash_attribution(tmp_path):
    root = make_root(tmp_path)
    source = tmp_path / "lesson.notejson"
    source.write_text(
        '{"title":"Lesson","body":"Dynamic source body"}', encoding="utf-8"
    )

    run_path, proof_path, source_doc = core.import_source(root, source)

    assert run_path is not None
    assert proof_path is not None
    obj = core.load_object(run_path)
    assert obj.attribution["source_id"] == source_doc.source_id
    assert obj.attribution["sha256"] == source_doc.sha256
    manifest = core.load_yaml(root / "stores" / "proof" / "source-manifest.json")
    assert manifest["sources"][0]["source_id"] == source_doc.source_id


def test_custom_rule_supports_autoingest_cypher_files(tmp_path):
    root = make_root(tmp_path)
    source = tmp_path / "location.cy"
    source.write_text("MATCH (n) RETURN n LIMIT 5", encoding="utf-8")

    docs, _ = core.scan_inputs(root, source)

    assert docs[0].adapter == "rule:autoingest-cypher-note"
    assert docs[0].source_type == "graph_query_note"
    assert docs[0].suggested_route == "RESEARCH_IDEATE"


def test_doctor_reports_invalid_adapter_registry(tmp_path):
    root = make_root(tmp_path)
    (root / "integrations" / "source-adapters.json").write_text(
        '[{"name":"bad","extensions":["notejson"],"source_type":"bad","route":"SPAM"}]',
        encoding="utf-8",
    )

    issues = core.doctor(root)

    assert any("invalid extensions" in issue for issue in issues)
    assert any("invalid route" in issue for issue in issues)


def test_malformed_adapter_registry_does_not_break_builtin_scan(tmp_path):
    root = make_root(tmp_path)
    (root / "integrations" / "source-adapters.json").write_text(
        "{not-json", encoding="utf-8"
    )
    source = tmp_path / "note.txt"
    source.write_text("still works", encoding="utf-8")

    docs, _ = core.scan_inputs(root, source)

    assert docs[0].source_type == "text_note"
    assert any("Malformed adapter registry" in issue for issue in core.doctor(root))
