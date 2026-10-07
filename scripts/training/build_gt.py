"""Turn label files into a tesstrain ground-truth set.

Every labelled crop is run through the same preprocessing the service applies
at runtime, so the model trains on exactly the images it will later see. The
result is ``<out>/train`` and ``<out>/holdout`` with ``*.png`` + ``*.gt.txt``
pairs and a manifest per split; frames marked for review are copied to
``<out>/review`` instead and can be labelled with ``review_tool.py``, after
which ``--reviewed <out>/review`` folds them in.

    python scripts/training/build_gt.py config.json --labels corpus/labels_time.jsonl \\
        corpus/labels_cells.jsonl --out corpus/gt --holdout 600-700,1300-1350,1458-1466,1550-1560
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

from scoresight.core.models import ServiceConfig
from scoresight.ocr.preprocess import preprocess

# Static displays would otherwise dominate the set: at most this many images
# per (group, region, text). Frames of a running clock are all kept.
DEFAULT_CAPS = {
    "tenths": 1_000_000,
    "mmss": 30,
    "expired": 50,
    "pentime": 20,
    "number": 80,
    "text": 60,
}


def parse_ranges(text: str) -> list[tuple[float, float]]:
    ranges = []
    for part in filter(None, text.split(",")):
        start, end = part.split("-")
        ranges.append((float(start), float(end)))
    return ranges


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("config", type=Path)
    parser.add_argument("--labels", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--holdout",
        default="",
        help="video seconds kept out of training, e.g. 600-700,1300-1350",
    )
    parser.add_argument(
        "--cap", nargs="*", default=[], metavar="GROUP=N", help="override a default cap"
    )
    parser.add_argument(
        "--reviewed",
        type=Path,
        help="review directory with *.gt.txt written by review_tool.py",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = ServiceConfig.model_validate_json(args.config.read_text(encoding="utf-8"))
    regions = {r.id: r for r in config.regions}
    caps = dict(DEFAULT_CAPS)
    caps.update({k: int(v) for k, v in (item.split("=", 1) for item in args.cap)})
    holdout = parse_ranges(args.holdout)
    random.seed(args.seed)

    records = []
    for path in args.labels:
        records += [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    reviewed = {}
    if args.reviewed:
        for gt in args.reviewed.rglob("*.gt.txt"):
            reviewed[gt.name[: -len(".gt.txt")]] = gt.read_text(
                encoding="utf-8"
            ).strip()
    for record in records:
        stem = Path(record["file"]).stem
        if stem in reviewed:
            record["text"] = reviewed[stem]
            record["review"] = False

    for split in ("train", "holdout", "review"):
        if (args.out / split).exists():
            shutil.rmtree(args.out / split)
        (args.out / split).mkdir(parents=True)

    review_index = collections.defaultdict(list)
    for record in records:
        if not record["review"]:
            continue
        group_dir = args.out / "review" / record["group"]
        group_dir.mkdir(exist_ok=True)
        shutil.copy(record["file"], group_dir / Path(record["file"]).name)
        review_index[record["group"]].append(
            {
                "file": Path(record["file"]).name,
                "t": record["t"],
                "suggested": record["text"],
                "ocr": record.get("ocr"),
            }
        )
    for group, items in review_index.items():
        with open(
            args.out / "review" / f"{group}.jsonl", "w", encoding="utf-8"
        ) as handle:
            for item in items:
                handle.write(json.dumps(item) + "\n")

    by_key = collections.defaultdict(list)
    for record in records:
        if not record["review"] and record["text"]:
            by_key[(record["group"], record["region"], record["text"])].append(record)
    chosen = []
    for key, items in by_key.items():
        random.shuffle(items)
        chosen += items[: caps.get(key[0], 80)]
    chosen.sort(key=lambda r: (r["region"], r["t"]))

    manifests = {"train": [], "holdout": []}
    written = collections.Counter()
    for record in chosen:
        patch = cv2.imread(record["file"])
        image = preprocess(patch, regions[record["region"]].preprocess)
        if image.max() == 0 or min(image.shape) < 8:
            continue
        split = "holdout" if any(a <= record["t"] < b for a, b in holdout) else "train"
        name = Path(record["file"]).stem
        cv2.imwrite(str(args.out / split / f"{name}.png"), image)
        (args.out / split / f"{name}.gt.txt").write_text(
            record["text"] + "\n", encoding="utf-8"
        )
        manifests[split].append(
            {
                "image": f"{split}/{name}.png",
                "text": record["text"],
                "region": record["region"],
                "group": record["group"],
                "t": record["t"],
            }
        )
        written[(split, record["group"])] += 1
    for split, items in manifests.items():
        with open(args.out / f"{split}.jsonl", "w", encoding="utf-8") as handle:
            for item in items:
                handle.write(json.dumps(item) + "\n")

    print("written:", dict(sorted(written.items())))
    print("for review:", {g: len(v) for g, v in review_index.items()})
    characters = collections.Counter(
        ch for item in manifests["train"] for ch in item["text"]
    )
    print("characters in training:", dict(sorted(characters.items())))

    sample = random.sample(manifests["train"], min(40, len(manifests["train"])))
    tiles = []
    for item in sorted(sample, key=lambda m: (m["region"], m["t"])):
        tile = cv2.imread(str(args.out / item["image"]))
        tile = cv2.copyMakeBorder(
            tile, 22, 4, 4, 4, cv2.BORDER_CONSTANT, value=(40, 40, 40)
        )
        cv2.putText(
            tile,
            f'{item["region"]} t={item["t"]:.1f} GT={item["text"]!r}',
            (4, 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
        )
        tiles.append(tile)
    width = max(t.shape[1] for t in tiles)
    tiles = [
        cv2.copyMakeBorder(
            t, 0, 0, 0, width - t.shape[1], cv2.BORDER_CONSTANT, value=(40, 40, 40)
        )
        for t in tiles
    ]
    cv2.imwrite(
        str(args.out / "check_gt.jpg"), np.vstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 90]
    )
    print(f"sample sheet: {args.out / 'check_gt.jpg'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
