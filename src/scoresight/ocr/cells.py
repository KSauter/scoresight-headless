"""Read regions digit cell by digit cell with a small neural network.

Tesseract reads a region as a line of text. On a seven-segment style board that
invites a whole class of mistakes no per-character reader can make: a lone "1"
read as "11", a frame line read into a "7". With a fixed camera every digit sits
in a fixed cell, so each cell can be classified on its own into a digit, a dark
cell, or "mix" - a digit caught while changing, which is reported as unreadable
instead of guessed.

The network runs through OpenCV's DNN module, so the service needs no runtime
beyond what it already ships. ``scripts/training/train_cells.py`` trains it and
writes ``<model>.cells.onnx`` with a ``<model>.cells.json`` sidecar next to the
Tesseract models.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

from scoresight.ocr.base import Recognition

INPUT_WIDTH = 36
INPUT_HEIGHT = 64
DARK = " "
MIX = "mix"
# Reported for a region whose cells cannot be put together into a reading. It
# fails every field type and format check, so the pipeline rejects the frame
# and keeps the last accepted value.
UNREADABLE = "?"


def cell_input(gray: Any, x1: int, x2: int, y1: int = 0, y2: int | None = None) -> Any:
    """The network input for the cell spanning columns x1..x2 of a region.

    The whole height of the region is used: the digit fills it, and the frame
    lines above and below are part of what the network has learned to ignore.
    Brightness is standardised per cell, so exposure changes do not matter.
    """
    import cv2
    import numpy as np

    height, width = gray.shape[:2]
    x1, x2 = max(0, x1), min(width, max(x1 + 1, x2))
    y1, y2 = max(0, y1), min(height, height if y2 is None else y2)
    patch = cv2.resize(
        gray[y1:y2, x1:x2], (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_AREA
    ).astype(np.float32)
    return (patch - patch.mean()) / (patch.std() + 8.0)


def assemble_text(chars: Sequence[str], field_type: str) -> str:
    """Put cell characters together the way the board means them.

    A clock with four cells shows ``MM:SS`` and, below a minute, ``SS:t`` with
    the last cell dark; a clock with three cells is a penalty clock ``M:SS``.
    The colon is not a cell: it blinks and carries no information beyond the
    layout, which the cells already give.
    """
    if any(char == MIX for char in chars):
        return UNREADABLE
    text = "".join(chars)
    if field_type == "time":
        if len(chars) == 4:
            if chars[3] == DARK and chars[2] != DARK:
                seconds = "".join(chars[:2]).strip()
                return f"{seconds}.{chars[2]}" if seconds.isdigit() else UNREADABLE
            minutes = "".join(chars[:2]).strip()
            seconds = "".join(chars[2:])
            if minutes.isdigit() and seconds.isdigit():
                return f"{minutes}:{seconds}"
            return "" if not text.strip() else UNREADABLE
        if len(chars) == 3:
            if all(char.isdigit() for char in chars):
                return f"{chars[0]}:{chars[1]}{chars[2]}"
            return "" if not text.strip() else UNREADABLE
    stripped = text.strip()
    # Dark cells may only lead or trail: a gap inside a number is a misread.
    if DARK in stripped:
        return UNREADABLE
    return stripped


@dataclass(frozen=True, slots=True)
class CellModel:
    classes: tuple[str, ...]
    path: Path

    @classmethod
    def load(cls, directory: Path, model: str) -> CellModel:
        path = directory / f"{model}.cells.onnx"
        sidecar = directory / f"{model}.cells.json"
        if not path.exists() or not sidecar.exists():
            raise FileNotFoundError(
                f"cell model {model!r} not found: expected {path.name} and "
                f"{sidecar.name} in {directory}"
            )
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        if meta.get("input") != [INPUT_WIDTH, INPUT_HEIGHT]:
            raise ValueError(
                f"{sidecar.name}: unexpected input size {meta.get('input')}"
            )
        return cls(classes=tuple(meta["classes"]), path=path)


class CellEngine:
    """Classify the digit cells of every region that has cells configured.

    Regions without cells are handed to ``fallback`` (Tesseract), so a
    configuration can switch region by region. Cell regions receive the raw
    patch rather than the binarised one the pipeline prepares for Tesseract:
    the network was trained on grey levels, and a digit fading in or out is
    exactly the grey it needs to see.
    """

    def __init__(
        self,
        model: CellModel,
        cells: dict[str, list[tuple[float, float]]],
        field_types: dict[str, str],
        fallback: Any | None = None,
    ) -> None:
        import cv2

        self.model = model
        self.cells = {key: value for key, value in cells.items() if value}
        self.field_types = field_types
        self.fallback = fallback
        self._net = cv2.dnn.readNetFromONNX(str(model.path))
        self._lock = Lock()

    @property
    def raw_regions(self) -> set[str]:
        return set(self.cells)

    def recognize(self, image: Any, *, region_id: str) -> Recognition:
        return self.recognize_many([(image, region_id)])[0]

    def recognize_many(self, requests: list[tuple[Any, str]]) -> list[Recognition]:
        results: list[Recognition | None] = [None] * len(requests)
        cell_requests = [
            (index, image, region_id)
            for index, (image, region_id) in enumerate(requests)
            if region_id in self.cells
        ]
        other = [
            (index, image, region_id)
            for index, (image, region_id) in enumerate(requests)
            if region_id not in self.cells
        ]
        if other:
            if self.fallback is None:
                for index, _, _ in other:
                    results[index] = Recognition(UNREADABLE, 0.0)
            else:
                pairs = [(image, region_id) for _, image, region_id in other]
                many = getattr(self.fallback, "recognize_many", None)
                answers = (
                    many(pairs)
                    if callable(many)
                    else [self.fallback.recognize(i, region_id=r) for i, r in pairs]
                )
                for (index, _, _), answer in zip(other, answers, strict=True):
                    results[index] = answer
        if cell_requests:
            for (index, _, _), answer in zip(
                cell_requests, self._classify(cell_requests), strict=True
            ):
                results[index] = answer
        return [
            result if result is not None else Recognition(UNREADABLE, 0.0)
            for result in results
        ]

    def _classify(self, requests: list[tuple[int, Any, str]]) -> list[Recognition]:
        import cv2
        import numpy as np

        inputs: list[Any] = []
        spans: list[tuple[int, int]] = []
        for _, image, region_id in requests:
            gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            width = gray.shape[1]
            start = len(inputs)
            for x1, x2 in self.cells[region_id]:
                inputs.append(cell_input(gray, round(x1 * width), round(x2 * width)))
            spans.append((start, len(inputs)))
        batch = np.stack(inputs)[:, None, :, :].astype(np.float32)
        with self._lock:
            self._net.setInput(batch)
            logits = self._net.forward()
        logits = logits - logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        best = probabilities.argmax(axis=1)
        answers = []
        for (_, _, region_id), (start, end) in zip(requests, spans, strict=True):
            chars = [self.model.classes[int(k)] for k in best[start:end]]
            confidence = float(
                probabilities[np.arange(start, end), best[start:end]].min()
            )
            text = assemble_text(chars, self.field_types.get(region_id, "text"))
            answers.append(Recognition(text, 0.0 if text == UNREADABLE else confidence))
        return answers

    def close(self) -> None:
        if self.fallback is not None:
            self.fallback.close()


# --- Finding the cells -------------------------------------------------------

# Cell width over the height of the digits, used until the region has shown a
# digit wider than a "1". Measured on the Nautronic board: 0.72 for the clock
# and the scores alike.
DEFAULT_ASPECT = 0.72
# Pitch (start of one cell to the start of the next) over cell width.
DEFAULT_PITCH = 1.25
# Rows of a region that are searched for digits.
BAND = (0.12, 0.88)


@dataclass
class CellActivity:
    """Which pixels of a region have been lit recently, decayed over time.

    A dark cell cannot be seen in a single frame, and a "1" lights only the
    right edge of its cell, so the cells are inferred from what has lit up over
    the last minutes rather than from one image.
    """

    decay: float = 0.998
    # What the region looked like when collection started; a moved or
    # resized region starts over.
    key: Any = None
    lit: Any = None
    weight: float = 0.0
    shape: tuple[int, int] = (0, 0)
    samples: int = field(default=0)

    def add(self, gray: Any) -> None:
        import cv2
        import numpy as np

        # Frame lines above and below the digits would light every column.
        height = gray.shape[0]
        band = gray[int(BAND[0] * height) : int(BAND[1] * height)]
        _, binary = cv2.threshold(band, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        core = cv2.erode(binary, np.ones((3, 3), np.uint8)) > 0
        if self.lit is None or core.shape != self.shape:
            self.lit = np.zeros(core.shape, np.float32)
            self.weight = 0.0
            self.shape = core.shape
        self.lit *= self.decay
        self.lit += core
        self.weight = self.weight * self.decay + 1.0
        self.samples += 1

    def map(self) -> Any:
        return None if self.lit is None else self.lit / max(self.weight, 1e-6)


@dataclass(frozen=True, slots=True)
class DetectedCell:
    x1: float
    x2: float
    observed: bool


def detect_cells(
    lit_map: Any, count: int, *, threshold: float = 0.03
) -> list[DetectedCell]:
    """Cells of a region from how often each pixel was lit.

    Digits are right-aligned and share one width and pitch, as on any
    seven-segment style board. Columns lit often enough form runs. A run that
    is narrow but spans the digit height is a "1": its cell ends at the run's
    right edge and gets the width of a digit. A narrow run that does not span
    the height is a colon and is ignored. Cells that never lit are added to the
    left with the pitch of the ones that did.
    """
    import numpy as np

    if lit_map is None or count < 1:
        return []
    height, width = lit_map.shape
    lit = lit_map > threshold
    rows = np.flatnonzero(lit.any(axis=1))
    if not len(rows):
        return []
    digit_height = rows.max() - rows.min() + 1
    columns = lit.any(axis=0)
    runs: list[tuple[int, int]] = []
    start = None
    for x in range(width + 1):
        on = x < width and columns[x]
        if on and start is None:
            start = x
        if not on and start is not None:
            runs.append((start, x))
            start = None

    def coverage(run: tuple[int, int]) -> float:
        column_rows = np.flatnonzero(lit[:, run[0] : run[1]].any(axis=1))
        return len(column_rows) / digit_height

    expected = DEFAULT_ASPECT * digit_height
    single = [x2 - x1 for x1, x2 in runs if 0.6 * expected <= x2 - x1 <= 1.4 * expected]
    cell_width = float(np.median(single)) if single else expected
    cells: list[tuple[float, float]] = []
    for x1, x2 in runs:
        run_width = x2 - x1
        if run_width < 0.35 * cell_width and (x1 <= 2 or x2 >= width - 2):
            continue  # a vertical frame edge at the border of the region
        if run_width > 1.6 * cell_width:
            # Neighbouring digits lit together ("10") merge into one run.
            split_pitch = DEFAULT_PITCH * cell_width
            parts = max(2, round((run_width - cell_width) / split_pitch) + 1)
            for part in range(parts):
                right = x2 - part * split_pitch
                cells.append((right - cell_width, right))
        elif run_width >= 0.6 * cell_width:
            cells.append((float(x1), float(x2)))
        elif coverage((x1, x2)) >= 0.7:
            cells.append((x2 - cell_width, float(x2)))
        # otherwise a colon or noise
    # Overlapping guesses for the same cell collapse into one.
    merged: list[tuple[float, float]] = []
    for cell in sorted(cells):
        if merged and cell[0] < merged[-1][1] - 0.3 * cell_width:
            merged[-1] = (min(merged[-1][0], cell[0]), max(merged[-1][1], cell[1]))
        else:
            merged.append(cell)
    merged = merged[-count:]
    if not merged:
        return []
    starts = [x1 for x1, _ in merged]
    gaps = [b - a for a, b in zip(starts, starts[1:], strict=False)]
    pitch = min(gaps) if gaps else DEFAULT_PITCH * cell_width
    found = [DetectedCell(x1, x2, True) for x1, x2 in merged]
    while len(found) < count:
        left = found[0].x1 - pitch
        found.insert(0, DetectedCell(left, left + cell_width, False))
    return [
        DetectedCell(
            max(0.0, cell.x1) / width, min(float(width), cell.x2) / width, cell.observed
        )
        for cell in found
    ]
