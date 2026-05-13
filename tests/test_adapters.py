from pathlib import Path

from content_os.adapters import extract_source, scan_path


def test_subtitle_adapter_strips_timing_noise(tmp_path: Path):
    subtitle = tmp_path / "clip.srt"
    subtitle.write_text(
        "1\n00:00:00,000 --> 00:00:02,000\nThis is the usable line.\n\n2\n00:00:02,000 --> 00:00:04,000\nAnother line.",
        encoding="utf-8",
    )

    doc = extract_source(subtitle)

    assert doc.source_type == "transcript"
    assert "-->" not in doc.text
    assert "This is the usable line." in doc.text
    assert "Another line." in doc.text


def test_csv_adapter_extracts_text_like_column(tmp_path: Path):
    csv_file = tmp_path / "metadata.csv"
    csv_file.write_text("id,notes\n1,Useful proof\n2,Second proof\n", encoding="utf-8")

    doc = extract_source(csv_file)

    assert doc.source_type == "tabular_export"
    assert "Useful proof" in doc.text
    assert doc.metadata["columns"] == ["id", "notes"]


def test_media_adapter_creates_manifest_without_transcription(tmp_path: Path):
    media = tmp_path / "dashcam.mp4"
    media.write_bytes(b"not a real video")

    doc = extract_source(media)

    assert doc.source_type == "media_asset"
    assert "not transcribed automatically" in " ".join(doc.warnings)
    assert "Media asset captured as a manifest" in doc.text


def test_scan_path_reports_malformed_json_without_stopping(tmp_path: Path):
    bad = tmp_path / "bad.json"
    good = tmp_path / "note.txt"
    bad.write_text("{not json", encoding="utf-8")
    good.write_text("usable note", encoding="utf-8")

    docs = scan_path(tmp_path)

    by_name = {doc.path.name: doc for doc in docs}
    assert by_name["bad.json"].source_type == "unreadable"
    assert by_name["note.txt"].source_type == "text_note"


def test_rule_based_adapter_handles_custom_json_extension(tmp_path: Path):
    from content_os.adapters import AdapterRule, extract_source

    source = tmp_path / "custom.notejson"
    source.write_text(
        '{"title":"Custom Title","body":"Custom body proof"}', encoding="utf-8"
    )
    rule = AdapterRule(
        name="notejson",
        extensions={".notejson"},
        source_type="custom_json_note",
        text_keys=["body"],
        title_keys=["title"],
        route="REPURPOSE",
    )

    doc = extract_source(source, rules=[rule])

    assert doc.adapter == "rule:notejson"
    assert doc.source_type == "custom_json_note"
    assert doc.title == "Custom Title"
    assert doc.text == "Custom body proof"
    assert doc.sha256
