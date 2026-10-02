"""End-to-end check on the bundled pilot recording (~45 s: scans the scene video)."""
import json
from pathlib import Path

import pandas as pd
import pytest

from fixpoint.layout import DEFAULT_SLIDES, load_calibration_slides
from fixpoint.pipeline import run_recording
from fixpoint.recording import find_recordings

EXAMPLE = Path(__file__).parents[1] / "examples" / "pilot" / "neon_export"


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    folders = find_recordings(EXAMPLE)
    if not folders or not any(folders[0].glob("*.mp4")):
        pytest.skip("example recording not available")
    out = tmp_path_factory.mktemp("out")
    run_recording(folders[0], load_calibration_slides(DEFAULT_SLIDES), out, progress=lambda *_: None)
    return out / folders[0].name


def test_slides():
    layout = load_calibration_slides(DEFAULT_SLIDES)
    assert sorted(layout.tags) == [0, 1, 2, 3]
    assert layout.targets[["x", "y"]].round().values.tolist() == [
        [384, 216], [960, 216], [1536, 216], [384, 540], [960, 540], [1536, 540],
        [384, 864], [960, 864], [1536, 864]]


def test_finds_both_blocks(result):
    pres = pd.read_csv(result / "presentations.csv")
    assert sorted(pres.block_name.unique()) == ["end", "start"]
    assert len(pres) == 18


def test_known_accuracy(result):
    blocks = pd.read_csv(result / "blocks.csv").set_index("block")
    assert blocks.loc["start", "mean_accuracy_deg"] == pytest.approx(1.15, abs=0.1)
    assert blocks.loc["end", "mean_accuracy_deg"] == pytest.approx(0.70, abs=0.1)


def test_session_offset_up_and_right(result):
    offset = json.loads((result / "session_offset.json").read_text())
    assert offset["shift_h_deg"] == pytest.approx(0.35, abs=0.1)
    assert offset["shift_v_deg"] == pytest.approx(0.33, abs=0.1)
    assert offset["loo_corrected_mean_accuracy_deg"] < offset["raw_mean_accuracy_deg"]


def test_corrected_files(result):
    gaze = pd.read_csv(result / "corrected" / "gaze.csv")
    original = pd.read_csv(next(EXAMPLE.glob("*/gaze.csv")))
    offset = json.loads((result / "session_offset.json").read_text())
    dx = (original["gaze x [px]"] - gaze["gaze x [px]"]).median()
    assert dx == pytest.approx(offset["shift_x_px"], abs=1e-6)
    assert list(gaze.columns) == list(original.columns)
    assert (result / "report.html").stat().st_size > 100_000
