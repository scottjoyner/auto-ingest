"""
Regression tests for dashcam clip-key handling.

Two functions turn a dashcam filename into a "base key" (the stem with the
_F/_R/_FR camera suffix removed), and they have to agree:

  * yolo_vehicle_detction.list_files()  -- which clips still need detections
  * link_global_speakers._norm_stem()   -- which audio stem a file_key is
  * yolo_embeddings.clip_base_key()    -- the convention both of the above
                                          implement (auto_ingest/dashcam)

These tests exist because the regexes that do this work were wrong in ways that
fail silently: list_files() dropped every *_R.MP4 (rear camera) file, so half
the dashcam corpus was never handed to vehicle detection, and it returned a
plausible sorted list instead of raising.

Technique is copied from _mock_import_module() in test_ml_pure.py: stub the
heavy ML stack in sys.modules so the modules import without torch /
ultralytics / pyannote / moviepy, then call only the pure helpers. No inference
ever runs.

NB: importing yolo_vehicle_detction executes its module body, which calls
get_fileserver_path("dashcam") and list_directories() at import time. On a
machine where the dashcam share is not mounted that is a no-op (os.walk over a
missing path); test_ml_pure.py already imports the module the same way.
"""
import importlib
import sys
import unittest.mock as mock

import numpy as np  # imported before the mock-imports so pandas loads exactly once
import pytest

pd = pytest.importorskip("pandas")


def _mock_import_module(module_name, extra_stubs=None):
    """Import `module_name` while stubbing heavy ML deps in sys.modules.

    A MagicMock covers every attribute access, so submodule imports like
    `from torch.nn.functional import normalize` resolve to a dummy callable.
    """
    stubs = {
        "torch": mock.MagicMock(),
        "torchaudio": mock.MagicMock(),
        "torch.nn": mock.MagicMock(),
        "torch.nn.functional": mock.MagicMock(),
        "transformers": mock.MagicMock(),
        "soundfile": mock.MagicMock(),
        "faiss": mock.MagicMock(),
        "speechbrain": mock.MagicMock(),
        "speechbrain.pretrained": mock.MagicMock(),
        "pyannote": mock.MagicMock(),
        "pyannote.audio": mock.MagicMock(),
        "ultralytics": mock.MagicMock(),
        "ultralytics.utils": mock.MagicMock(),
        "ultralytics.utils.plotting": mock.MagicMock(),
        "cv2": mock.MagicMock(),
        "moviepy": mock.MagicMock(),
        "moviepy.editor": mock.MagicMock(),
        "PIL": mock.MagicMock(),
        "matplotlib": mock.MagicMock(),
    }
    if extra_stubs:
        stubs.update(extra_stubs)
    with mock.patch.dict(sys.modules, stubs):
        return importlib.import_module(module_name)


def _dashcam_dir(tmp_path, *names):
    """Create a directory containing `names` (empty files) and return it."""
    d = tmp_path / "2025" / "02" / "02"
    d.mkdir(parents=True, exist_ok=True)
    for name in names:
        (d / name).write_bytes(b"")
    return str(d)


# ---------------------------------------------------------------------------
# yolo_vehicle_detction.list_files
# ---------------------------------------------------------------------------
def test_list_files_discovers_rear_camera_keys(tmp_path):
    """The regression that matters: *_R.MP4 (rear camera) clips were dropped.

    The old pattern only accepted a *digit* before the camera letter
    (`_\\d+R\\.mp4`), so no real dashcam name -- CLIP_A_R.MP4,
    2025_0202_171732_R.MP4 -- ever matched, and every rear-camera clip was
    silently omitted from vehicle detection.
    """
    V = _mock_import_module("yolo_vehicle_detction")  # sic: module name is misspelled in the repo
    d = _dashcam_dir(tmp_path, "CLIP_A_R.MP4", "2025_0202_171732_R.MP4")
    assert V.list_files(d) == ["2025_0202_171732", "CLIP_A"]


def test_list_files_still_discovers_front_camera_keys(tmp_path):
    V = _mock_import_module("yolo_vehicle_detction")  # sic: module name is misspelled in the repo
    d = _dashcam_dir(tmp_path, "CLIP_A_F.MP4", "2025_0202_171732_F.mp4")
    assert V.list_files(d) == ["2025_0202_171732", "CLIP_A"]


def test_list_files_handles_stacked_front_plus_rear_key(tmp_path):
    """*_FR.MP4 (front+rear stacked, dashcam_merge_FR.py) is a camera key too."""
    V = _mock_import_module("yolo_vehicle_detction")  # sic: module name is misspelled in the repo
    d = _dashcam_dir(tmp_path, "2025_0202_171732_FR.MP4")
    assert V.list_files(d) == ["2025_0202_171732"]


def test_list_files_collapses_front_and_rear_onto_one_base_key(tmp_path):
    """Both cameras of a clip collapse onto ONE base key, sorted.

    The base key is what the rest of the chain is named after: the sidecar we
    write/skip ({key}_YOLOv8n.csv), the video we open ({key}.MP4) and
    yolo_heatmap.list_files(), which rebuilds keys from {key}_YOLOv8n.csv and
    then requires {key}.MP4 to exist.
    """
    V = _mock_import_module("yolo_vehicle_detction")  # sic: module name is misspelled in the repo
    d = _dashcam_dir(
        tmp_path,
        "CLIP_A_R.MP4", "CLIP_A_F.MP4",
        "CLIP_B_R.MP4", "CLIP_B_F.MP4",
        "IGNORE.MP4",
    )
    keys = V.list_files(d)
    assert keys == ["CLIP_A", "CLIP_B"]
    assert keys == sorted(keys)
    # No key may still carry a camera suffix or an extension.
    assert not any(k.endswith(("_F", "_R", "_FR", ".MP4", ".mp4")) for k in keys)


def test_list_files_skips_keys_that_already_have_a_sidecar(tmp_path):
    V = _mock_import_module("yolo_vehicle_detction")  # sic: module name is misspelled in the repo
    d = _dashcam_dir(
        tmp_path,
        "CLIP_A_R.MP4", "CLIP_A_F.MP4",
        "CLIP_B_R.MP4", "CLIP_B_F.MP4", "CLIP_B_YOLOv8n.csv",
    )
    keys = V.list_files(d)
    assert keys == ["CLIP_A"]
    assert "CLIP_B" not in keys


def test_list_files_ignores_names_that_are_not_camera_clips(tmp_path):
    """No camera suffix, or no .mp4 -> not a key.

    Guards the unescaped "." in the old `_F.MP4` pattern, where "." matched any
    character, so CLIP_FXMP4 was accepted as a front-camera clip and CLIP_AXMP4
    style names came within one character of being accepted.
    """
    V = _mock_import_module("yolo_vehicle_detction")  # sic: module name is misspelled in the repo
    d = _dashcam_dir(
        tmp_path,
        "IGNORE.MP4",
        "CLIP_FXMP4",       # old `_F.MP4` matched this ("." ate the X)
        "CLIP_AXMP4",
        "CLIP_A_R.avi",
        "notes.txt",
        "CLIP_A_YOLOv8n.csv",
    )
    assert V.list_files(d) == []


def test_list_files_accepts_a_digit_run_before_the_camera_suffix(tmp_path):
    """Legacy 12_F.mp4 shape (digit, no separator) still resolves."""
    V = _mock_import_module("yolo_vehicle_detction")  # sic: module name is misspelled in the repo
    d = _dashcam_dir(tmp_path, "CLIP_1_F.mp4", "CLIP_2F.MP4")
    assert V.list_files(d) == ["CLIP_1", "CLIP_2"]


# ---------------------------------------------------------------------------
# link_global_speakers._norm_stem
# ---------------------------------------------------------------------------
def test_norm_stem_strips_every_camera_suffix():
    L = _mock_import_module("auto_ingest.diarize.link_global_speakers")
    assert L._norm_stem("2025_0202_171732_F") == "2025_0202_171732"
    assert L._norm_stem("2025_0202_171732_R") == "2025_0202_171732"
    # _FR must strip as a UNIT: in "..._FR" the R is not preceded by "_", so a
    # "_(F|R)$" alternation (or a bare "_R$") cannot reach it and leaves the
    # suffix fully intact.
    assert L._norm_stem("2025_0202_171732_FR") == "2025_0202_171732"
    assert L._norm_stem("CLIP_A_F") == "CLIP_A"
    assert L._norm_stem("CLIP_A_R") == "CLIP_A"
    assert L._norm_stem("CLIP_A_FR") == "CLIP_A"


def test_norm_stem_leaves_a_key_without_a_camera_suffix_untouched():
    L = _mock_import_module("auto_ingest.diarize.link_global_speakers")
    assert L._norm_stem("2025_0202_171732") == "2025_0202_171732"
    assert L._norm_stem("interview_recording") == "interview_recording"
    assert L._norm_stem("clip.wav") == "clip"


def test_norm_stem_still_strips_sidecar_markers():
    L = _mock_import_module("auto_ingest.diarize.link_global_speakers")
    assert L._norm_stem("2025_0202_171732_medium_transcription_speakers") == "2025_0202_171732"
    assert L._norm_stem("2025_0202_171732_large_transcription") == "2025_0202_171732"
    assert L._norm_stem("a_large_transcription") == "a"
    # sidecar markers are stripped BEFORE the camera suffix, so a key carrying
    # both still collapses all the way down.
    assert L._norm_stem("2025_0202_171732_F_speakers") == "2025_0202_171732"
    assert L._norm_stem("2025_0202_171732_FR_speakers") == "2025_0202_171732"
    assert L._norm_stem("2025_0202_171732_F.wav") == "2025_0202_171732"


def test_norm_stem_agrees_with_clip_base_key():
    """_norm_stem and clip_base_key implement ONE convention; keep them equal.

    clip_base_key() in auto_ingest/dashcam/yolo_embeddings.py is the documented
    owner of the _F/_R/_FR stripping rule. _norm_stem cannot import it (that
    module pulls in moviepy/neo4j at import time), so the two regexes are pinned
    together here instead -- if either drifts, this fails.
    """
    L = _mock_import_module("auto_ingest.diarize.link_global_speakers")
    stubs = {
        "moviepy": mock.MagicMock(),
        "moviepy.editor": mock.MagicMock(),
        "neo4j": mock.MagicMock(),
        "neo4j.exceptions": mock.MagicMock(),
    }
    Y = _mock_import_module("auto_ingest.dashcam.yolo_embeddings", extra_stubs=stubs)
    for key in [
        "2025_0202_171732",
        "2025_0202_171732_F",
        "2025_0202_171732_R",
        "2025_0202_171732_FR",
        "CLIP_A",
        "CLIP_A_F",
        "CLIP_A_R",
        "CLIP_A_FR",
        "a_b_c_F",
        "interview_recording",
    ]:
        assert L._norm_stem(key) == Y.clip_base_key(key), key


# ---------------------------------------------------------------------------
# yolo_embeddings.parse_yolo_csv: the Key/classification half of the same
# contract. The legacy header written by yolo_vehicle_detection.py and
# yolo_batch_worker.py is
#   Key,vehicle_id,confidence,classification,xywh,xyxy,Frame
# so `confidence` is always present; if the name-resolution ladder stops at
# `confidence`, every row parses with name == "" and keep_detection() then drops
# every single detection.
# ---------------------------------------------------------------------------
def test_legacy_yolo_csv_header_yields_detection_names(tmp_path):
    stubs = {
        "moviepy": mock.MagicMock(),
        "moviepy.editor": mock.MagicMock(),
        "neo4j": mock.MagicMock(),
        "neo4j.exceptions": mock.MagicMock(),
    }
    Y = _mock_import_module("auto_ingest.dashcam.yolo_embeddings", extra_stubs=stubs)
    csv_path = tmp_path / "2025_0202_171732_YOLOv8n.csv"
    csv_path.write_text(
        "Key,vehicle_id,confidence,classification,xywh,xyxy,Frame\n"
        "2025_0202_171732_1_car_a1, id1, 95.0%, car, [10 20 30 40], [10, 20, 40, 60], 1\n"
        "2025_0202_171732_1_truck_b2, id2, 80%, truck, [50 50 20 20], [50, 50, 70, 70], 2\n",
        encoding="utf-8",
    )
    df = Y.parse_yolo_csv(str(csv_path))
    assert df["name"].tolist() == ["car", "truck"]
    assert np.isclose(df["confidence"].tolist()[0], 0.95)
    assert df["frame"].tolist() == [1, 2]
    keep = {"car", "truck"}
    id_map = {2: "car", 7: "truck"}
    assert all(Y.keep_detection(row, keep, id_map) for row in df.to_dict("records"))
    # the Key column's clip prefix is a base key: strippable by the shared rule
    assert Y.clip_base_key("2025_0202_171732_F") == "2025_0202_171732"
