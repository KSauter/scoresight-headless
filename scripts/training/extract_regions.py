"""Cut scoreboard regions out of a video for OCR training.

One decoding pass, any number of regions. For every sampled frame and every
region that is not blank, the raw crop is written as PNG and one JSON line is
appended to ``<out>/<region>.jsonl`` with the frame time, the current engine's
raw reading and a few image features the labelling scripts rely on.

Example (whole game at 2 Hz for every region, then the last minute of the
clock at full frame rate)::

    python scripts/training/extract_regions.py game.mp4 config.json --step 15 --out corpus/cells
    python scripts/training/extract_regions.py game.mp4 config.json --regions time \\
        --start 1445 --end 1507 --step 1 --out corpus/clock_tenths
"""

from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path

import cv2
import numpy as np

from scoresight.core.models import ServiceConfig
from scoresight.core.runtime import RuntimeController
from scoresight.ocr.preprocess import crop_region, is_blank, preprocess
from scoresight.ocr.tesseract_engine import TesseractEngine

WHITELISTS = {"number": "0123456789", "time": "0123456789:."}


def colon_pair(binary: np.ndarray) -> bool:
    """Two small blobs above each other: the colon of a time cell is lit."""
    height, width = binary.shape
    count, _, stats, centroids = cv2.connectedComponentsWithStats(binary)
    small = [
        (centroids[k][0], centroids[k][1])
        for k in range(1, count)
        if 10 <= stats[k][4] <= 0.004 * height * width + 60
        and stats[k][3] < 0.2 * height
        and stats[k][2] < 0.15 * width
    ]
    for i, (ax, ay) in enumerate(small):
        for bx, by in small[i + 1 :]:
            if (
                abs(ax - bx) < 0.03 * width + 3
                and 0.15 * height < abs(ay - by) < 0.7 * height
            ):
                return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("video", type=Path)
    parser.add_argument("config", type=Path, help="ScoreSight service configuration")
    parser.add_argument("--regions", default="all", help="comma separated region ids")
    parser.add_argument("--start", type=float, default=0.0, help="seconds")
    parser.add_argument("--end", type=float, default=float("inf"), help="seconds")
    parser.add_argument("--step", type=int, default=15, help="use every n-th frame")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--model", help="engine used for the raw reading (default: config)"
    )
    parser.add_argument(
        "--tessdata", type=Path, default=RuntimeController.tessdata_path()
    )
    parser.add_argument(
        "--keep-blank", action="store_true", help="also write blank cells"
    )
    args = parser.parse_args()

    config = ServiceConfig.model_validate_json(args.config.read_text(encoding="utf-8"))
    wanted = None if args.regions == "all" else set(args.regions.split(","))
    regions = [r for r in config.regions if wanted is None or r.id in wanted]
    engine = TesseractEngine(
        args.model or config.ocr.model,
        args.tessdata,
        {r.id: WHITELISTS.get(r.field_type, "") for r in regions},
    )
    with contextlib.ExitStack() as stack:
        indexes = {}
        for region in regions:
            (args.out / region.id).mkdir(parents=True, exist_ok=True)
            indexes[region.id] = stack.enter_context(
                open(args.out / f"{region.id}.jsonl", "a", encoding="utf-8")
            )
        written = extract(args, config, regions, engine, indexes)
    engine.close()
    print(f"{written} crops from {len(regions)} regions written to {args.out}")
    return 0


def extract(args, config, regions, engine, indexes) -> int:
    capture = cv2.VideoCapture(str(args.video))
    capture.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000)
    seen = written = 0
    while capture.grab():
        timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
        if timestamp > args.end:
            break
        frame_index = int(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        seen += 1
        if (seen - 1) % args.step:
            continue
        ok, image = capture.retrieve()
        if not ok:
            break
        frame_height, frame_width = image.shape[:2]
        for region in regions:
            patch = crop_region(image, region.rect.pixels(frame_width, frame_height))
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            blank = is_blank(gray, region.preprocess)
            if blank and not args.keep_blank:
                continue
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            record = {
                "t": round(timestamp, 3),
                "frame": frame_index,
                "file": f"{region.id}/{region.id}_{frame_index:06d}.png",
                "blank": bool(blank),
                "bright": round(float(np.percentile(gray, 99.5)), 1),
                "colon_pair": colon_pair(binary),
            }
            if not blank:
                reading = engine.recognize(
                    preprocess(patch, region.preprocess), region_id=region.id
                )
                record["raw"] = reading.text
                record["conf"] = round(reading.confidence, 2)
            cv2.imwrite(str(args.out / record["file"]), patch)
            indexes[region.id].write(json.dumps(record) + "\n")
            written += 1
    return written


if __name__ == "__main__":
    raise SystemExit(main())
