from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import pytest

from scoresight.capture.base import FramePacket
from scoresight.core.models import CellSpan, NormalizedRect, RegionConfig
from scoresight.ocr.base import Recognition
from scoresight.ocr.cells import (
    DARK,
    MIX,
    UNREADABLE,
    CellActivity,
    CellEngine,
    CellModel,
    assemble_text,
    detect_cells,
)
from scoresight.ocr.pipeline import RecognitionPipeline

TESSDATA = Path(__file__).resolve().parents[2] / "tesseract" / "tessdata"


@pytest.mark.parametrize(
    ("chars", "field_type", "text"),
    [
        (["1", "3", "0", "6"], "time", "13:06"),
        ([DARK, "9", "3", "4"], "time", "9:34"),
        ([DARK, "0", "4", "7"], "time", "0:47"),
        (["2", "7", "5", DARK], "time", "27.5"),
        ([DARK, "7", "5", DARK], "time", "7.5"),
        ([DARK, "0", "0", DARK], "time", "0.0"),
        (["1", "2", "9"], "time", "1:29"),
        ([DARK, DARK, DARK, DARK], "time", ""),
        ([DARK, "1"], "number", "1"),
        (["1", "0"], "number", "10"),
        (["E"], "text", "E"),
        ([DARK, DARK], "number", ""),
    ],
)
def test_cells_spell_what_the_board_shows(chars, field_type, text) -> None:
    assert assemble_text(chars, field_type) == text


@pytest.mark.parametrize(
    ("chars", "field_type"),
    [
        (["1", MIX, "0", "6"], "time"),
        (["1", DARK, "0"], "number"),
        ([DARK, DARK, "0", "6"], "time"),
    ],
)
def test_cells_that_cannot_be_read_together_are_unreadable(chars, field_type) -> None:
    assert assemble_text(chars, field_type) == UNREADABLE


def digit_map(spans: list[tuple[int, int]], width: int = 200, height: int = 80) -> np.ndarray:
    """A lit map as CellActivity keeps it, with full-height digits at spans."""
    lit = np.zeros((height, width), np.float32)
    for x1, x2 in spans:
        lit[5 : height - 5, x1:x2] = 1.0
    return lit


def test_cell_detection_finds_lit_cells_and_places_dark_ones_by_pitch() -> None:
    # Units cell lit as an "8" (full width), tens cell never lit.
    cells = detect_cells(digit_map([(120, 170)]), 2)
    assert len(cells) == 2
    assert cells[1].observed and not cells[0].observed
    assert cells[1].x1 == pytest.approx(120 / 200)
    assert cells[0].x2 <= cells[1].x1


def test_cell_detection_widens_a_lone_one_to_a_digit_cell() -> None:
    # A "1" lights a thin bar at the right of its cell only.
    lit = digit_map([(160, 168)])
    cells = detect_cells(lit, 1)
    assert len(cells) == 1
    width = (cells[0].x2 - cells[0].x1) * 200
    assert width > 30
    assert cells[0].x2 == pytest.approx(168 / 200)


def test_cell_detection_ignores_a_colon_and_frame_edges() -> None:
    lit = digit_map([(0, 2), (10, 50), (60, 100), (198, 200)])
    # A colon: two small dots, not the digit height.
    lit[20:26, 108:114] = 1.0
    lit[50:56, 108:114] = 1.0
    lit = np.maximum(lit, digit_map([(120, 160), (170, 196)]))
    cells = detect_cells(lit, 4)
    assert [round(cell.x1 * 200) for cell in cells] == [10, 60, 120, 170]


def test_cell_activity_ignores_frame_lines_above_and_below() -> None:
    gray = np.full((100, 120), 30, np.uint8)
    gray[:4, :] = 255  # frame line on top
    gray[96:, :] = 255  # and at the bottom
    gray[20:80, 70:110] = 255  # a digit
    activity = CellActivity()
    activity.add(gray)
    columns = activity.map().max(axis=0) > 0.5
    assert not columns[:60].any()
    assert columns[75:105].all()


class RawRecorder:
    raw_regions = {"clock"}

    def __init__(self) -> None:
        self.seen: dict[str, tuple[int, ...]] = {}

    def recognize_many(self, requests):
        for image, region_id in requests:
            self.seen[region_id] = image.shape
        return [
            Recognition(UNREADABLE if region_id == "clock" else "1", 0.9)
            for _, region_id in requests
        ]

    def close(self) -> None:
        pass


def test_pipeline_hands_raw_patches_to_cell_regions_and_rejects_unreadable() -> None:
    image = np.zeros((100, 200, 3), np.uint8)
    cv2.putText(image, "12", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    cv2.putText(image, "3", (120, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    regions = [
        RegionConfig(
            id="clock",
            name="Clock",
            rect=NormalizedRect(x=0, y=0, width=0.5, height=0.5),
            field_type="time",
            cells=[CellSpan(x1=0.0, x2=0.5), CellSpan(x1=0.5, x2=1.0)],
        ),
        RegionConfig(
            id="score",
            name="Score",
            rect=NormalizedRect(x=0.5, y=0, width=0.5, height=0.5),
            field_type="number",
        ),
    ]
    engine = RawRecorder()
    activity: dict = {}
    pipeline = RecognitionPipeline(engine, regions, cell_activity=activity)
    batch = pipeline.process(
        FramePacket(sequence=1, image=image, width=200, height=100, captured_at=datetime.now(UTC))
    )
    assert engine.seen["clock"] == (50, 100, 3)  # colour patch, not binarised
    assert len(engine.seen["score"]) == 2  # prepared image for Tesseract
    fields = {field.id: field for field in batch.fields}
    assert fields["clock"].state == "rejected"
    assert set(activity) == {"clock", "score"}


@pytest.mark.skipif(
    not (TESSDATA / "nautronic.cells.onnx").exists(), reason="cell model not bundled"
)
def test_bundled_cell_model_reads_a_rendered_digit_cell() -> None:
    model = CellModel.load(TESSDATA, "nautronic")
    assert model.classes[-1] == MIX
    engine = CellEngine(model, {"score": [(0.0, 1.0)]}, {"score": "number"})
    blank = np.full((100, 60), 40, np.uint8)
    result = engine.recognize(blank, region_id="score")
    assert result.text in {"", UNREADABLE}
    assert 0.0 <= (result.confidence or 0.0) <= 1.0
