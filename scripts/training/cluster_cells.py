"""Read a clock without any OCR engine: cluster each digit cell, label clusters by hand.

The engine being replaced makes systematic mistakes (a 1 read as 7, a 4 read
as 9 for whole minutes), and anything built on its readings inherits them.
With a fixed camera every digit cell of the clock sits at the same pixels, and
a seven-segment digit has only a handful of possible looks. So:

1. ``--sheet``: every frame's cells are grouped by pixel similarity. One image
   per cell shows each group's average and a few members; a person writes the
   character of every group into a JSON file, ``{"0": {"3": "1", ...}, ...}``
   (cell -> group -> character, ``" "`` for a dark cell, ``"?"`` to drop).
2. ``--labels clusters.json --out reading.jsonl``: every frame gets the reading
   of its groups, written like ``extract_regions.py`` output with ``raw``
   replaced, so ``label_game_clock.py`` checks it against the timeline.

    python scripts/training/cluster_cells.py config.json corpus/cells/time.jsonl \\
        --work corpus/clusters --sheet
    python scripts/training/cluster_cells.py config.json corpus/cells/time.jsonl \\
        --work corpus/clusters --labels corpus/clusters/time_labels.json \\
        --out corpus/clusters/time.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from label_clock import binarize, find_colon_zone
from label_game_clock import cell_columns

from scoresight.core.models import ServiceConfig
from scoresight.ocr.preprocess import is_blank

SHAPE = (16, 24)  # width, height of a cell signature


def core_of(gray: np.ndarray) -> np.ndarray:
    height = gray.shape[0]
    band = gray[int(0.12 * height) : int(0.88 * height)]
    return cv2.erode(binarize(band), np.ones((3, 3), np.uint8), iterations=1) > 0


def cluster(signatures: np.ndarray, threshold: float, passes: int = 3) -> np.ndarray:
    """Greedy leader clustering, then a few rounds of re-centring."""
    centers: list[np.ndarray] = []
    assignment = np.zeros(len(signatures), dtype=int)
    for index, sig in enumerate(signatures):
        if centers:
            distances = np.abs(np.asarray(centers) - sig).mean(axis=(1, 2))
            best = int(distances.argmin())
            if distances[best] <= threshold:
                assignment[index] = best
                continue
        centers.append(sig.astype(np.float32))
        assignment[index] = len(centers) - 1
    for _ in range(passes):
        means = np.asarray(
            [
                signatures[assignment == k].mean(axis=0)
                if (assignment == k).any()
                else np.full(signatures.shape[1:], 9.0)
                for k in range(len(centers))
            ]
        )
        flat = signatures.reshape(len(signatures), -1)
        distances = np.stack([np.abs(flat - m.reshape(1, -1)).mean(axis=1) for m in means], axis=1)
        assignment = distances.argmin(axis=1)
    # Renumber by size, largest first, so the sheet is stable to read.
    order = np.argsort(-np.bincount(assignment, minlength=len(centers)))
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    return rank[assignment]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("config", type=Path)
    parser.add_argument("inputs", type=Path, nargs="+", help="<region>.jsonl files")
    parser.add_argument("--region", default="time")
    parser.add_argument("--cells", type=int, default=4)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.07)
    parser.add_argument("--min-size", type=int, default=15, help="smaller groups are dropped")
    parser.add_argument("--sheet", action="store_true")
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--centers",
        type=Path,
        help="assign frames to the groups saved by an earlier --sheet run instead of "
        "clustering anew, so labelled group numbers stay valid for new frames",
    )
    args = parser.parse_args()

    config = ServiceConfig.model_validate_json(args.config.read_text(encoding="utf-8"))
    region = next(r for r in config.regions if r.id == args.region)
    records = {}
    for path in args.inputs:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                record["file"] = str((path.parent / record["file"]).resolve())
                records.setdefault(record["frame"], record)
    ordered = sorted(records.values(), key=lambda r: r["t"])
    grays = [cv2.imread(r["file"], cv2.IMREAD_GRAYSCALE) for r in ordered]
    keep = [i for i, g in enumerate(grays) if not is_blank(g, region.preprocess)]
    ordered = [ordered[i] for i in keep]
    grays = [grays[i] for i in keep]
    zone = find_colon_zone(grays[:: max(1, len(grays) // 400)])
    if zone is None:
        raise SystemExit("could not locate the colon")
    cores = [core_of(g) for g in grays]
    cells = cell_columns(cores[::5], zone, args.cells)
    print(f"{len(ordered)} frames, digit cells {cells}")

    args.work.mkdir(parents=True, exist_ok=True)
    assignments = {}
    saved = dict(np.load(args.centers)) if args.centers else {}
    centers_out = {}
    for cell, (x1, x2) in enumerate(cells):
        sigs = np.asarray(
            [
                cv2.resize(core[:, x1:x2].astype(np.float32), SHAPE, interpolation=cv2.INTER_AREA)
                for core in cores
            ]
        )
        if saved:
            means = saved[f"cell{cell}"]
            flat = sigs.reshape(len(sigs), -1)
            distances = np.stack(
                [np.abs(flat - m.reshape(1, -1)).mean(axis=1) for m in means], axis=1
            )
            nearest = distances.argmin(axis=1)
            # Too far from every labelled group: a look not seen before.
            nearest[distances.min(axis=1) > 2 * args.threshold] = -1
            assignments[cell] = nearest
            print(f"cell {cell}: {int((nearest < 0).sum())} frames far from every group")
            continue
        assignments[cell] = cluster(sigs, args.threshold)
        centers_out[f"cell{cell}"] = np.asarray(
            [sigs[assignments[cell] == k].mean(axis=0) for k in range(assignments[cell].max() + 1)]
        )
        sizes = np.bincount(assignments[cell])
        big = int((sizes >= args.min_size).sum())
        print(f"cell {cell}: {len(sizes)} groups, {big} with at least {args.min_size} frames")
        if args.sheet:
            tiles = []
            for k in range(big):
                members = np.flatnonzero(assignments[cell] == k)
                mean = np.mean([cores[i][:, x1:x2] for i in members], axis=0)
                picks = members[np.linspace(0, len(members) - 1, 3).astype(int)]
                images = [(mean * 255).astype(np.uint8)] + [
                    grays[i][int(0.12 * grays[i].shape[0]) : int(0.88 * grays[i].shape[0]), x1:x2]
                    for i in picks
                ]
                height = max(im.shape[0] for im in images)
                row = np.hstack(
                    [
                        cv2.copyMakeBorder(
                            im, 0, height - im.shape[0], 2, 2, cv2.BORDER_CONSTANT, value=60
                        )
                        for im in images
                    ]
                )
                row = cv2.resize(row, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_NEAREST)
                label = np.full((row.shape[0], 120), 30, np.uint8)
                cv2.putText(label, f"#{k}", (6, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2)
                cv2.putText(
                    label, f"n={len(members)}", (6, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 200, 1
                )
                tiles.append(np.hstack([label, row]))
            width = max(t.shape[1] for t in tiles)
            tiles = [
                cv2.copyMakeBorder(t, 0, 4, 0, width - t.shape[1], cv2.BORDER_CONSTANT, value=30)
                for t in tiles
            ]
            columns = 3
            while len(tiles) % columns:
                tiles.append(np.full_like(tiles[0], 30))
            per = len(tiles) // columns
            sheet = np.hstack([np.vstack(tiles[c * per : (c + 1) * per]) for c in range(columns)])
            cv2.imwrite(str(args.work / f"{args.region}_cell{cell}.png"), sheet)

    if centers_out:
        np.savez(args.work / f"{args.region}_centers.npz", **centers_out)

    if args.labels:
        names = json.loads(args.labels.read_text(encoding="utf-8"))
        if args.out is None:
            raise SystemExit("--out is required with --labels")
        written = dropped = 0
        with open(args.out, "w", encoding="utf-8") as handle:
            for index, record in enumerate(ordered):
                chars = [
                    names.get(str(cell), {}).get(str(int(assignments[cell][index])))
                    for cell in range(args.cells)
                ]
                if any(c is None or c == "?" for c in chars):
                    reading = None
                    dropped += 1
                else:
                    if chars[-1] == " ":
                        digits = "".join(chars[:-1]).strip()
                        reading = f"{digits[:-1]}:{digits[-1]}" if len(digits) >= 2 else None
                    else:
                        digits = "".join(chars).strip()
                        reading = f"{digits[:-2]}:{digits[-2:]}" if len(digits) >= 3 else None
                    if reading is None or " " in reading:
                        dropped += 1
                        reading = None
                out = dict(record, raw=reading, raw_engine=record.get("raw"))
                out["file"] = str(Path(record["file"]))
                handle.write(json.dumps(out) + "\n")
                written += 1
        print(f"{written} frames written, {dropped} without a clean reading")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
