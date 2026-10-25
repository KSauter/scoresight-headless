"""Cut labelled crops into digit cells for the cell network (train_cells.py).

Two kinds of sources, both from earlier labelling, nothing is labelled anew:

* clocks (``--clock``): the reading jsonl of ``cluster_cells.py`` with its saved
  group centres and hand-given group names. Every cell gets the character of
  its group, ``mix`` for groups named ``?`` (a digit caught while changing).
  Where the frame also has an accepted label from ``label_game_clock.py`` the
  cell must agree with it.
* numbers and text (``--cells``): label files of ``label_cells.py``. The cells
  are found with the service's own detection over all frames of the region,
  the text is right-aligned into them.

Each item keeps the cell with a few pixels of context on every side, so
training can shift it. The result is ``<out>/cells.npy`` plus
``<out>/items.json``::

    C=output/training-nautronic/corpus2
    python scripts/training/build_cells.py config.json --out cells-nn \\
        --clock 2=time:4:$C/clusters/time_all.jsonl:$C/clusters/time:$C/labels_time.jsonl \\
        --cells 2=$C/labels_cells.jsonl

``VIDEO`` before ``=`` names the recording, so train_cells.py can hold out
stretches per recording.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
from pathlib import Path

import cv2
import numpy as np
from cluster_cells import SHAPE, core_of, find_colon_zone
from label_game_clock import cell_chars, cell_columns

from scoresight.core.models import ServiceConfig
from scoresight.ocr.cells import DARK, MIX, CellActivity, detect_cells
from scoresight.ocr.preprocess import is_blank

CTX = 6
CLASSES = [*"0123456789", DARK, "E", MIX]


def digits(text: str | None) -> str:
    return re.sub(r"\D", "", text or "")


def load_gray(file: str) -> tuple[np.ndarray, np.ndarray] | None:
    """The crop and the same crop with CTX pixels of context."""
    gray = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None
    padded = Path(file).parent / "padded" / Path(file).name
    if padded.exists():
        big = cv2.imread(str(padded), cv2.IMREAD_GRAYSCALE)
        if big is not None and big.shape[0] - gray.shape[0] == 2 * CTX:
            return gray, big
    return gray, cv2.copyMakeBorder(gray, CTX, CTX, CTX, CTX, cv2.BORDER_REPLICATE)


def window(big: np.ndarray, x1: int, x2: int) -> np.ndarray:
    return big[:, max(0, x1) : x2 + 2 * CTX].copy()


def clock_items(spec: str, video: str) -> tuple[list[dict], list[np.ndarray]]:
    region, cells_count, readings, groups, labels = spec.split(":")
    count = int(cells_count)
    groups_path = Path(groups)
    centers = np.load(f"{groups_path}_centers.npz")
    names = json.loads(Path(f"{groups_path}_labels.json").read_text(encoding="utf-8"))
    accepted = {}
    for line in Path(labels).read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            accepted[record["file"]] = record
    records = [
        json.loads(line)
        for line in Path(readings).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    grays = [
        cv2.imread(r["file"], cv2.IMREAD_GRAYSCALE) for r in records[:: max(1, len(records) // 400)]
    ]
    grays = [g for g in grays if g is not None]
    zone = find_colon_zone(grays)
    columns = cell_columns([core_of(g) for g in grays], zone, count)
    items, windows = [], []
    stats = collections.Counter()
    for record in records:
        loaded = load_gray(record["file"])
        if loaded is None:
            continue
        gray, big = loaded
        core = core_of(gray)
        label = accepted.get(record["file"])
        want = None
        if label and not label["review"]:
            kind = "tenths" if label["group"].endswith(("tenths", "expired")) else "mmss"
            want = cell_chars(digits(label["text"]), kind, count)
        for index, (x1, x2) in enumerate(columns):
            sig = cv2.resize(core[:, x1:x2].astype(np.float32), SHAPE, interpolation=cv2.INTER_AREA)
            means = centers[f"cell{index}"]
            distance = np.abs(means.reshape(len(means), -1) - sig.reshape(1, -1)).mean(axis=1)
            group = int(distance.argmin())
            char = names.get(str(index), {}).get(str(group))
            if distance[group] > 0.06 or char is None:
                stats["unknown"] += 1
                continue
            char = MIX if char == "?" else char
            if want is not None and char != MIX and char != want[index]:
                stats["conflict"] += 1
                continue
            stats[char] += 1
            items.append(
                {"region": region, "video": video, "t": record["t"], "cls": CLASSES.index(char)}
            )
            windows.append(window(big, x1, x2))
    print(f"clock {region} ({video}): cells {columns}, {dict(stats)}")
    return items, windows


def number_items(
    path: Path, config: ServiceConfig, video: str, cap: int
) -> tuple[list[dict], list[np.ndarray]]:
    regions = {r.id: r for r in config.regions}
    by_region = collections.defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            region = regions.get(record["region"])
            if region is None or region.field_type == "time" or record["review"]:
                continue
            by_region[record["region"]].append(record)
    items, windows = [], []
    for region_id, records in sorted(by_region.items()):
        region = regions[region_id]
        count = 2 if region.field_type == "number" else 1
        activity = CellActivity(decay=1.0)
        for record in records[::3]:
            gray = cv2.imread(record["file"], cv2.IMREAD_GRAYSCALE)
            if gray is not None and not is_blank(gray, region.preprocess):
                activity.add(gray)
        cells = detect_cells(activity.map(), count)
        if len(cells) != count:
            print(f"{region_id} ({video}): no cells found, skipped")
            continue
        widths = [cell.x2 - cell.x1 for cell in cells]
        if max(widths) > 1.25 * min(widths):
            # Cells of one region share a width; a frame line touching a digit
            # widens one of them, and its crops would teach the wrong cell.
            print(f"{region_id} ({video}): cell widths {widths} disagree, skipped")
            continue
        by_text = collections.defaultdict(list)
        for record in records:
            by_text[record["text"]].append(record)
        stats = collections.Counter()
        for text, group in by_text.items():
            chars = list(text.rjust(count, DARK)[-count:])
            if any(c not in CLASSES for c in chars):
                continue
            random.shuffle(group)
            for record in group[:cap]:
                loaded = load_gray(record["file"])
                if loaded is None:
                    continue
                gray, big = loaded
                width = gray.shape[1]
                for cell, char in zip(cells, chars, strict=True):
                    items.append(
                        {
                            "region": region_id,
                            "video": video,
                            "t": record["t"],
                            "cls": CLASSES.index(char),
                        }
                    )
                    windows.append(window(big, round(cell.x1 * width), round(cell.x2 * width)))
                stats[text] += 1
        print(
            f"{region_id} ({video}): cells "
            f"{[(round(float(c.x1), 3), round(float(c.x2), 3)) for c in cells]}, "
            f"frames per text {dict(stats)}"
        )
    return items, windows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("config", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--clock",
        nargs="*",
        default=[],
        metavar="VIDEO=REGION:CELLS:READINGS:GROUPS:LABELS",
        help="clock source; GROUPS is the cluster_cells.py prefix, e.g. clusters/time",
    )
    parser.add_argument(
        "--cells", nargs="*", default=[], metavar="VIDEO=LABELS", help="label_cells.py output"
    )
    parser.add_argument("--cap", type=int, default=300, help="frames per region and text")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    random.seed(args.seed)
    config = ServiceConfig.model_validate_json(args.config.read_text(encoding="utf-8"))
    items: list[dict] = []
    windows: list[np.ndarray] = []
    for spec in args.clock:
        video, rest = spec.split("=", 1)
        more, wins = clock_items(rest, video)
        items += more
        windows += wins
    for spec in args.cells:
        video, path = spec.split("=", 1)
        more, wins = number_items(Path(path), config, video, args.cap)
        items += more
        windows += wins
    args.out.mkdir(parents=True, exist_ok=True)
    stored = np.empty(len(windows), dtype=object)
    stored[:] = windows
    np.save(args.out / "cells.npy", stored, allow_pickle=True)
    (args.out / "items.json").write_text(
        json.dumps({"classes": CLASSES, "context": CTX, "items": items}), encoding="utf-8"
    )
    print(len(items), "cells:", dict(collections.Counter(CLASSES[i["cls"]] for i in items)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
