"""Label clock crops written by extract_regions.py.

The clock is the only cell whose text cannot be read off a few frames by eye:
it changes ten times a second below a minute. Three sources are combined:

* minutes:seconds frames are accepted when the engine's reading agrees with
  its neighbours along a monotonic countdown,
* tenths frames are labelled from time alone - a start time and a rate are
  fitted so that the engine agrees with the derived labels as often as
  possible - and every frame is checked against digit templates learned from
  the majority, which drops the frames where the board still shows the old
  digit fading out,
* frames after the clock expired are labelled "0:0".

Whether the colon is lit is read from the pixels, because this kind of board
blinks it. Every frame the script is not sure about is kept for a manual pass.

    python scripts/training/label_clock.py config.json --inputs corpus/clock_tenths/time.jsonl \\
        corpus/cells/time.jsonl --out corpus/labels_time.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import re
from pathlib import Path

import cv2
import numpy as np

from scoresight.core.models import ServiceConfig

DIGITS = re.compile(r"\D")


def digits_of(raw: str | None) -> str:
    return DIGITS.sub("", raw or "")


def mmss_value(raw: str | None, max_minutes: int) -> tuple[int, str] | None:
    text = digits_of(raw)
    if len(text) in (3, 4):
        minutes, seconds = int(text[:-2]), int(text[-2:])
        if seconds < 60 and minutes <= max_minutes:
            return minutes * 60 + seconds, text
    return None


def separator_kind(raw: str | None) -> str | None:
    """ "tenths" for "59:8" (one digit after the separator), "mmss" for "1:09"."""
    match = re.fullmatch(r"\d{1,2}[:.]+(\d+)", (raw or "").strip())
    if match is None:
        return None
    trailing = len(match.group(1))
    return "tenths" if trailing == 1 else "mmss" if trailing == 2 else None


def detect_tenths_window(records: list[dict]) -> tuple[float | None, float | None]:
    """First stretch where separator readings turn into "SS:t", and when it reaches zero."""
    reads = [(r["t"], separator_kind(r.get("raw"))) for r in records]
    reads = [(t, kind) for t, kind in reads if kind]
    start = None
    for index, (t, kind) in enumerate(reads):
        if kind != "tenths":
            continue
        ahead = [k for tt, k in reads[index : index + 40] if tt <= t + 3.0]
        if len(ahead) >= 5 and ahead.count("tenths") / len(ahead) >= 0.6:
            start = t - 0.05
            break
    if start is None:
        return None, None
    zeros = [
        r["t"]
        for r in records
        if r["t"] > start and digits_of(r.get("raw")) in {"00", "000"}
    ]
    return start, (min(zeros) if zeros else None)


def tenths_value(raw: str | None) -> float | None:
    text = digits_of(raw)
    if len(text) == 3:
        return int(text[:2]) + int(text[2]) / 10
    if len(text) == 2 and re.search(r"[:.]", raw or ""):
        return int(text[0]) + int(text[1]) / 10
    return None


def tenths_digits(value: float) -> str:
    tenths = int(round(value * 10))
    seconds, tenth = divmod(max(0, tenths), 10)
    return f"{seconds}{tenth}" if seconds < 10 else f"{seconds:02d}{tenth}"


def with_colon(text: str, colon_on: bool, tenths: bool) -> str:
    if not colon_on:
        return text
    return f"{text[:-1]}:{text[-1]}" if tenths else f"{text[:-2]}:{text[-2:]}"


def binarize(gray: np.ndarray) -> np.ndarray:
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return binary


def find_colon_zone(
    grays: list[np.ndarray],
) -> tuple[float, float, float, float] | None:
    """The colon is the only pair of small blobs stacked vertically near the middle."""
    boxes = []
    for gray in grays:
        height, width = gray.shape
        count, _, stats, centroids = cv2.connectedComponentsWithStats(binarize(gray))
        small = [
            (
                stats[k][0],
                stats[k][1],
                stats[k][0] + stats[k][2],
                stats[k][1] + stats[k][3],
                centroids[k],
            )
            for k in range(1, count)
            if 15 <= stats[k][4] <= 250
            and stats[k][3] < 0.18 * height
            and stats[k][2] < 0.1 * width
            and 0.3 * width < centroids[k][0] < 0.7 * width
        ]
        for i, a in enumerate(small):
            for b in small[i + 1 :]:
                if (
                    abs(a[4][0] - b[4][0]) < 6
                    and 0.2 * height < abs(a[4][1] - b[4][1]) < 0.7 * height
                ):
                    boxes.append(
                        (
                            min(a[0], b[0]),
                            min(a[1], b[1]),
                            max(a[2], b[2]),
                            max(a[3], b[3]),
                        )
                    )
    if len(boxes) < 5:
        return None
    box = np.median(np.array(boxes), axis=0)
    height, width = grays[0].shape
    return (
        max(0.0, float(box[0] - 4) / width),
        max(0.0, float(box[1] - 4) / height),
        min(1.0, float(box[2] + 4) / width),
        min(1.0, float(box[3] + 4) / height),
    )


def zone_brightness(gray: np.ndarray, zone: tuple[float, float, float, float]) -> float:
    height, width = gray.shape
    patch = gray[
        int(zone[1] * height) : int(zone[3] * height),
        int(zone[0] * width) : int(zone[2] * width),
    ]
    return float(np.percentile(patch, 95)) if patch.size else 0.0


def consensus(records: list[dict], key: str) -> None:
    """Keep a reading when the neighbours agree it sits on a countdown."""

    def consistent(current: dict, other: dict | None) -> bool | None:
        if other is None or other[key] is None:
            return None
        elapsed = abs(other["t"] - current["t"])
        delta = other[key] - current[key]
        return (
            (0 <= delta <= elapsed + 1.0)
            if other["t"] < current["t"]
            else (0 <= -delta <= elapsed + 1.0)
        )

    for index, record in enumerate(records):
        record["consensus"] = False
        if record[key] is None:
            continue
        before = consistent(record, records[index - 1] if index else None)
        after = consistent(
            record, records[index + 1] if index + 1 < len(records) else None
        )
        if before is True and after is True:
            record["consensus"] = True
        elif before is True or after is True:
            other = None
            if before is None and index >= 2:
                other = records[index - 2]
            elif after is None and index + 2 < len(records):
                other = records[index + 2]
            record["consensus"] = (
                other is not None and consistent(record, other) is True
            )


def fit_tenths(records: list[dict], start: float, end: float) -> tuple[float, float]:
    """Start time and rate that make the engine agree with time-derived labels most often."""
    observed = [
        (r["t"], digits_of(r.get("raw"))) for r in records if start <= r["t"] <= end
    ]
    observed = [(t, d) for t, d in observed if len(d) in (2, 3)]
    best = (-1, start, 1.0)
    for rate in np.arange(0.997, 1.0031, 0.0005):
        for t0 in np.arange(start - 0.5, start + 0.5, 0.004):
            hits = sum(
                1
                for t, d in observed
                if tenths_digits(math.floor((60.0 - (t - t0) * rate) * 10 + 1e-9) / 10)
                == d
            )
            if hits > best[0]:
                best = (hits, float(t0), float(rate))
    print(
        f"tenths fit: t0={best[1]:.3f} rate={best[2]:.4f} agreement {best[0]}/{len(observed)}"
    )
    return best[1], best[2]


def template_mismatch(records: list[dict], grays: dict[str, np.ndarray], zone) -> None:
    """How far each frame is from the majority image of its own label, per digit cell."""
    kernel = np.ones((3, 3), np.uint8)
    cores = {}
    for record in records:
        gray = grays[record["file"]]
        cores[record["file"]] = cv2.erode(binarize(gray), kernel, iterations=1) > 0
    height, width = next(iter(cores.values())).shape
    stable = [r for r in records if 0.2 <= r["phase"] <= 0.8]
    occupancy = np.mean([cores[r["file"]].any(axis=0) for r in stable], axis=0)
    occupancy[int(zone[0] * width) - 4 : int(zone[2] * width) + 4] = 0
    cells, start = [], None
    for x in range(width + 1):
        lit = x < width and occupancy[x] > 0.05
        if lit and start is None:
            start = x
        if not lit and start is not None:
            if x - start > 8:
                cells.append((start, x))
            start = None
    cells = cells[:3]
    if len(cells) != 3:
        raise SystemExit(f"expected three digit cells below a minute, found {cells}")

    def cell_digits(label: str) -> list[str]:
        padded = label.rjust(3, " ")
        return [padded[0].strip(), padded[1], padded[2]]

    groups = collections.defaultdict(list)
    for record in stable:
        for cell, digit in enumerate(cell_digits(record["digits"])):
            x1, x2 = cells[cell]
            groups[(cell, digit)].append(cores[record["file"]][:, x1:x2])
    templates = {k: np.mean(v, axis=0) > 0.5 for k, v in groups.items() if len(v) >= 10}
    for record in records:
        worst = 0.0
        for cell, digit in enumerate(cell_digits(record["digits"])):
            template = templates.get((cell, digit))
            if template is None:
                worst = 1.0
                break
            x1, x2 = cells[cell]
            core = cores[record["file"]][:, x1:x2]
            worst = max(
                worst, float((core ^ template).sum()) / max(1, int(template.sum()))
            )
        record["mismatch"] = round(worst, 3)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("config", type=Path)
    parser.add_argument("--region", default="time")
    parser.add_argument(
        "--inputs", type=Path, nargs="+", required=True, help="<region>.jsonl files"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--tenths-start", type=float, help="video second the clock drops below a minute"
    )
    parser.add_argument(
        "--tenths-end", type=float, help="video second the clock reaches zero"
    )
    parser.add_argument("--max-minutes", type=int, default=20)
    parser.add_argument("--max-mismatch", type=float, default=0.2)
    args = parser.parse_args()

    config = ServiceConfig.model_validate_json(args.config.read_text(encoding="utf-8"))
    if args.region not in {r.id for r in config.regions}:
        raise SystemExit(f"unknown region {args.region}")
    records: dict[int, dict] = {}
    for path in args.inputs:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                record["file"] = str((path.parent / record["file"]).resolve())
                records.setdefault(record["frame"], record)
    ordered = sorted(records.values(), key=lambda r: r["t"])
    grays = {r["file"]: cv2.imread(r["file"], cv2.IMREAD_GRAYSCALE) for r in ordered}
    print(f"{len(ordered)} frames")

    zone = find_colon_zone(
        [grays[r["file"]] for r in ordered[:: max(1, len(ordered) // 300)]]
    )
    if zone is None:
        raise SystemExit("could not locate the colon")
    brightness = {r["file"]: zone_brightness(grays[r["file"]], zone) for r in ordered}
    low, high = np.percentile(list(brightness.values()), (10, 90))
    threshold = (low + high) / 2 if high - low > 40 else 0.0
    for record in ordered:
        record["colon_on"] = brightness[record["file"]] > threshold
    lit = sum(r["colon_on"] for r in ordered)
    print(f"colon zone {tuple(round(v, 3) for v in zone)}, lit in {lit} frames")

    for record in ordered:
        parsed = mmss_value(record.get("raw"), args.max_minutes)
        record["mmss"] = parsed[0] if parsed else None
        record["mmss_digits"] = parsed[1] if parsed else None
        record["tenths"] = tenths_value(record.get("raw"))

    detected_start, detected_end = detect_tenths_window(ordered)
    start = args.tenths_start if args.tenths_start is not None else detected_start
    end = args.tenths_end if args.tenths_end is not None else detected_end
    print(
        f"tenths window: {start} .. {end} (detected {detected_start} .. {detected_end})"
    )

    labelled = []
    minutes = [r for r in ordered if start is None or r["t"] < start]
    consensus(minutes, "mmss")
    for record in minutes:
        if record["mmss_digits"] is None:
            continue
        text = with_colon(record["mmss_digits"], record["colon_on"], tenths=False)
        labelled.append(
            dict(record, text=text, group="mmss", review=not record["consensus"])
        )

    if start is not None and end is not None:
        window = [r for r in ordered if start <= r["t"] <= end]
        t0, rate = fit_tenths(window, start, end)
        for record in window:
            continuous = max(0.0, 60.0 - (record["t"] - t0) * rate)
            tenths = math.floor(continuous * 10 + 1e-9)
            record["digits"] = tenths_digits(tenths / 10)
            record["phase"] = continuous * 10 - tenths
        template_mismatch(window, grays, zone)
        for record in window:
            text = with_colon(record["digits"], record["colon_on"], tenths=True)
            doubtful = record["mismatch"] > args.max_mismatch or not (
                0.1 <= record["phase"] <= 0.9
            )
            labelled.append(dict(record, text=text, group="tenths", review=doubtful))
        for record in (r for r in ordered if r["t"] > end):
            zeros = digits_of(record.get("raw")) in {"00", "000", "0"}
            text = with_colon("00", record["colon_on"], tenths=True)
            labelled.append(dict(record, text=text, group="expired", review=not zeros))

    with open(args.out, "w", encoding="utf-8") as handle:
        for record in labelled:
            handle.write(
                json.dumps(
                    {
                        "file": record["file"],
                        "region": args.region,
                        "t": record["t"],
                        "text": record["text"],
                        "group": record["group"],
                        "review": bool(record["review"]),
                        "ocr": record.get("raw"),
                    }
                )
                + "\n"
            )
    summary = collections.Counter(
        (r["group"], "review" if r["review"] else "ok") for r in labelled
    )
    print("labels:", dict(sorted(summary.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
