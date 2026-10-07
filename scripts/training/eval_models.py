"""Compare traineddata models on a ground-truth split.

Reads ``<gt_dir>/<split>.jsonl`` as written by ``build_gt.py`` and reports, per
group, how many images each model reads exactly and how many it reads right
except for separators, plus the most frequent mistakes::

    python scripts/training/eval_models.py corpus/gt holdout --models scoreboard_general,nautronic
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

import cv2

from scoresight.core.runtime import RuntimeController
from scoresight.ocr.tesseract_engine import TesseractEngine

WHITELISTS = {"time": "0123456789:.", "number": "0123456789", "text": ""}


def field_kind(region: str) -> str:
    if region == "time" or "Time" in region:
        return "time"
    return "text" if region == "period" else "number"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("gt_dir", type=Path)
    parser.add_argument("split", help="train or holdout")
    parser.add_argument("--models", required=True, help="comma separated model names")
    parser.add_argument(
        "--tessdata", type=Path, default=RuntimeController.tessdata_path()
    )
    parser.add_argument(
        "--mistakes", type=int, default=6, help="mistakes listed per group"
    )
    args = parser.parse_args()

    items = [
        json.loads(line)
        for line in (args.gt_dir / f"{args.split}.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    for model in args.models.split(","):
        engine = TesseractEngine(model, args.tessdata, dict(WHITELISTS))
        total: collections.Counter[str] = collections.Counter()
        exact: collections.Counter[str] = collections.Counter()
        digits: collections.Counter[str] = collections.Counter()
        mistakes: dict[str, collections.Counter[tuple[str, str]]] = (
            collections.defaultdict(collections.Counter)
        )
        for item in items:
            image = cv2.imread(str(args.gt_dir / item["image"]), cv2.IMREAD_GRAYSCALE)
            reading = engine.recognize(image, region_id=field_kind(item["region"]))
            predicted = re.sub(r"\s+", "", reading.text)
            group = item["group"]
            total[group] += 1
            if predicted == item["text"]:
                exact[group] += 1
            if re.sub(r"\D", "", predicted) == re.sub(r"\D", "", item["text"]):
                digits[group] += 1
            else:
                mistakes[group][(item["text"], predicted)] += 1
        engine.close()
        print(f"=== {model} ===")
        for group in sorted(total):
            print(
                f"  {group:8} n={total[group]:4}  exact={exact[group] / total[group]:6.1%}"
                f"  digits={digits[group] / total[group]:6.1%}"
                f"  mistakes={mistakes[group].most_common(args.mistakes)}"
            )
        count = sum(total.values())
        print(
            f"  total    n={count:4}  exact={sum(exact.values()) / count:6.1%}"
            f"  digits={sum(digits.values()) / count:6.1%}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
