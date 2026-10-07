"""Label score, period and penalty crops written by extract_regions.py.

Penalty clocks are labelled like the game clock: a reading counts when its
neighbours agree it sits on a countdown. Numbers and the period do not change
often enough for that, so they are labelled by hand - once per value, not per
frame: ``--list`` prints when each cell was lit and what the engine made of
it, and ``--ranges`` takes a JSON file mapping region ids to
``[[from_second, to_second, "text"], ...]``. Cells that never change get
``--fixed region=text``. Everything else goes to the manual pass.

    python scripts/training/label_cells.py config.json corpus/cells --list
    python scripts/training/label_cells.py config.json corpus/cells \\
        --out corpus/labels_cells.jsonl --fixed homeScore=0 awayScore=0 period=1 \\
        --ranges ranges.json
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import cv2
from label_clock import consensus, digits_of, with_colon

from scoresight.core.models import ServiceConfig
from scoresight.ocr.preprocess import is_blank


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "corpus", type=Path, help="directory written by extract_regions.py"
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument("--skip", default="time", help="comma separated region ids")
    parser.add_argument("--fixed", nargs="*", default=[], metavar="REGION=TEXT")
    parser.add_argument(
        "--ranges", type=Path, help='JSON: {"region": [[from, to, "text"], ...]}'
    )
    parser.add_argument(
        "--list", action="store_true", help="print lit periods per cell and exit"
    )
    parser.add_argument("--max-minutes", type=int, default=5)
    args = parser.parse_args()

    config = ServiceConfig.model_validate_json(args.config.read_text(encoding="utf-8"))
    skip = set(args.skip.split(","))
    fixed = dict(item.split("=", 1) for item in args.fixed)
    ranges = json.loads(args.ranges.read_text(encoding="utf-8")) if args.ranges else {}
    labelled = []
    for region in config.regions:
        path = args.corpus / f"{region.id}.jsonl"
        if region.id in skip or not path.exists():
            continue
        records = sorted(
            (
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ),
            key=lambda r: r["t"],
        )
        for record in records:
            record["file"] = str((args.corpus / record["file"]).resolve())
        # The configuration may have changed since extraction (e.g. min_contrast).
        records = [
            r
            for r in records
            if not is_blank(
                cv2.imread(r["file"], cv2.IMREAD_GRAYSCALE), region.preprocess
            )
        ]
        if args.list:
            blocks: list[dict] = []
            for record in records:
                if blocks and record["t"] - blocks[-1]["end"] <= 3.0:
                    blocks[-1]["end"] = record["t"]
                    blocks[-1]["ocr"][digits_of(record.get("raw"))] += 1
                else:
                    blocks.append(
                        {
                            "start": record["t"],
                            "end": record["t"],
                            "ocr": collections.Counter([digits_of(record.get("raw"))]),
                        }
                    )
            print(f"{region.id} ({region.field_type}): {len(records)} lit frames")
            for block in blocks:
                read = block["ocr"].most_common(4)
                print(
                    f"   {block['start']:7.1f} - {block['end']:7.1f} s  engine read {read}"
                )
            continue
        if region.field_type == "time":
            for record in records:
                text = digits_of(record.get("raw"))
                valid = (
                    len(text) == 3
                    and int(text[1:]) < 60
                    and int(text[0]) <= args.max_minutes
                )
                record["value"] = int(text[0]) * 60 + int(text[1:]) if valid else None
            consensus(records, "value")
            for record in records:
                text = digits_of(record.get("raw"))
                if not text:
                    continue
                if record["value"] is None and len(text) not in (2, 3):
                    continue
                labelled.append(
                    dict(
                        record,
                        region=region.id,
                        group="pentime",
                        review=not record["consensus"],
                        text=(
                            with_colon(text, record["colon_pair"], tenths=False)
                            if len(text) == 3
                            else text
                        ),
                    )
                )
        elif region.id in fixed:
            for record in records:
                labelled.append(
                    dict(
                        record,
                        region=region.id,
                        group=region.field_type,
                        review=False,
                        text=fixed[region.id],
                    )
                )
        else:
            spans = ranges.get(region.id, [])
            for record in records:
                text = next((t for a, b, t in spans if a <= record["t"] <= b), None)
                labelled.append(
                    dict(
                        record,
                        region=region.id,
                        group=region.field_type,
                        review=text is None,
                        text=text or digits_of(record.get("raw")),
                    )
                )
    if args.list:
        return 0
    if args.out is None:
        raise SystemExit("--out is required unless --list is given")
    with open(args.out, "w", encoding="utf-8") as handle:
        for record in labelled:
            if not record["text"]:
                continue
            handle.write(
                json.dumps(
                    {
                        "file": record["file"],
                        "region": record["region"],
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
        (r["region"], "review" if r["review"] else "ok") for r in labelled if r["text"]
    )
    print("labels:", dict(sorted(summary.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
