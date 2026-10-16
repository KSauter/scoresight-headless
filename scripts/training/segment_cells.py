"""Split number cells into stretches of identical display, for labelling by eye.

Scores, the period and penalty numbers change a few times per game. Reading
them with the engine that is being replaced would copy its mistakes into the
labels, so this script never looks at the engine: it compares the pixels.
Consecutive frames whose binarised cell looks the same form one segment. For
every segment a contact sheet shows the first, middle and last frame; a person
writes the text per segment into a ranges file for ``label_cells.py``::

    python scripts/training/segment_cells.py config.json corpus/cells \\
        --regions homeScore,awayScore,period --out corpus/segments

writes ``<out>/<region>.json`` (segments with start, end, frame count and the
engine's reading for reference) and ``<out>/<region>.png``. Frames that differ
from both neighbours' segments (a display changing mid-frame) form segments of
their own that are too short to label and are left out by ``--min-frames``.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import cv2
import numpy as np

from scoresight.core.models import ServiceConfig
from scoresight.ocr.preprocess import is_blank

SIGNATURE = (24, 16)  # width, height


def signature(gray: np.ndarray) -> np.ndarray:
    height = gray.shape[0]
    band = gray[int(0.1 * height) : int(0.9 * height)]
    _, binary = cv2.threshold(band, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    small = cv2.resize(binary, SIGNATURE, interpolation=cv2.INTER_AREA)
    return small > 127


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("config", type=Path)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--regions", required=True, help="comma separated region ids")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--threshold", type=float, default=0.06, help="fraction of differing pixels"
    )
    parser.add_argument("--min-frames", type=int, default=3)
    parser.add_argument("--gap", type=float, default=30.0, help="seconds dark that split")
    args = parser.parse_args()

    config = ServiceConfig.model_validate_json(args.config.read_text(encoding="utf-8"))
    regions = {r.id: r for r in config.regions}
    args.out.mkdir(parents=True, exist_ok=True)
    for region_id in args.regions.split(","):
        region = regions[region_id]
        path = args.corpus / f"{region_id}.jsonl"
        records = sorted(
            (json.loads(line) for line in path.read_text().splitlines() if line.strip()),
            key=lambda r: r["t"],
        )
        segments: list[dict] = []
        for record in records:
            gray = cv2.imread(str(args.corpus / record["file"]), cv2.IMREAD_GRAYSCALE)
            if is_blank(gray, region.preprocess):
                continue
            sig = signature(gray)
            current = segments[-1] if segments else None
            if (
                current is not None
                and record["t"] - current["end"] <= args.gap
                and float(np.mean(sig != current["reference"])) <= args.threshold
            ):
                current["end"] = record["t"]
                current["files"].append(record["file"])
                current["ocr"][record.get("raw") or ""] += 1
            else:
                segments.append(
                    {
                        "start": record["t"],
                        "end": record["t"],
                        "reference": sig,
                        "files": [record["file"]],
                        "ocr": collections.Counter([record.get("raw") or ""]),
                    }
                )
        kept = [s for s in segments if len(s["files"]) >= args.min_frames]
        short = sum(len(s["files"]) for s in segments if len(s["files"]) < args.min_frames)
        summary = [
            {
                "index": i,
                "start": s["start"],
                "end": s["end"],
                "frames": len(s["files"]),
                "engine": s["ocr"].most_common(3),
            }
            for i, s in enumerate(kept)
        ]
        (args.out / f"{region_id}.json").write_text(json.dumps(summary, indent=1))
        print(
            f"{region_id}: {len(kept)} segments, {short} frames in segments shorter "
            f"than {args.min_frames}"
        )
        tiles = []
        for i, segment in enumerate(kept):
            files = segment["files"]
            picks = [files[0], files[len(files) // 2], files[-1]]
            images = [cv2.imread(str(args.corpus / f)) for f in picks]
            row = np.hstack(
                [
                    cv2.copyMakeBorder(im, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(60, 60, 60))
                    for im in images
                ]
            )
            label = np.full((row.shape[0], 260, 3), 30, np.uint8)
            cv2.putText(label, f"#{i}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(
                label,
                f"{segment['start']:.0f}-{segment['end']:.0f}s n={len(files)}",
                (6, 46),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            tiles.append(np.hstack([label, row]))
        if tiles:
            width = max(t.shape[1] for t in tiles)
            tiles = [
                cv2.copyMakeBorder(
                    t, 0, 0, 0, width - t.shape[1], cv2.BORDER_CONSTANT, value=(30, 30, 30)
                )
                for t in tiles
            ]
            cv2.imwrite(str(args.out / f"{region_id}.png"), np.vstack(tiles))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
