"""Label a clock over a whole game, without trusting single readings.

``label_clock.py`` accepts a reading when its neighbours agree it sits on a
countdown. A digit the engine misreads consistently passes that test: ``7:58,
7:57, 7:56`` is a perfect countdown even when the board shows ``1:58``. This
script uses two checks that do not depend on the engine being right:

* **Timeline.** Around every frame, the readings of the surrounding seconds
  vote on whether the clock runs (value + video time is constant) or stands
  (value is constant). The label comes from that majority; a frame whose own
  reading disagrees is labelled from the timeline when the second is not about
  to change, otherwise it goes to review.
* **Digit templates.** Every digit cell of the clock is compared against the
  average image of all frames that carry the same character in that cell. A
  frame labelled ``1`` that looks like the ``7`` of that cell fails and goes to
  review. ``--sheet`` writes the templates as an image so a person can confirm
  in one look that every template shows the digit it is named after.

Several stretches of the game (periods, timeouts, the tenths at every period
end) are handled alike, so the input can be a whole game at 2 Hz plus the
period ends at full frame rate::

    python scripts/training/label_game_clock.py config.json \\
        --inputs corpus/cells/time.jsonl corpus/tenths/time.jsonl \\
        --out corpus/labels_time.jsonl --sheet corpus/clock_templates.png
"""

from __future__ import annotations

import argparse
import bisect
import collections
import json
import math
import re
import statistics
from pathlib import Path

import cv2
import numpy as np
from label_clock import binarize, digits_of, find_colon_zone, zone_brightness

from scoresight.core.models import ServiceConfig


def candidates(raw: str | None, max_minutes: int) -> list[tuple[float, str, str]]:
    """Every (value in seconds, kind, digits) a raw reading can stand for.

    Without a lit colon "158" is either 1:58 or 15.8 s; both are offered and
    the timeline decides.
    """
    text = digits_of(raw)
    raw = (raw or "").strip()
    found = []
    separator = re.fullmatch(r"(\d+)[:.]+(\d+)", raw)
    if len(text) in (3, 4):
        minutes, seconds = int(text[:-2]), int(text[-2:])
        if (
            seconds < 60
            and minutes <= max_minutes
            and (separator is None or len(separator.group(2)) == 2)
        ):
            found.append((minutes * 60 + seconds, "mmss", text))
    if len(text) in (2, 3):
        seconds, tenth = int(text[:-1]), int(text[-1])
        if seconds < 60 and (separator is None or len(separator.group(2)) == 1):
            found.append((seconds + tenth / 10, "tenths", text))
    return found


def display(value: float, kind: str) -> str:
    if kind == "tenths":
        tenths = max(0, math.floor(value * 10 + 1e-9))
        seconds, tenth = divmod(tenths, 10)
        return f"{seconds}{tenth}"
    whole = max(0, math.floor(value + 1e-9))
    minutes, seconds = divmod(whole, 60)
    return f"{minutes}{seconds:02d}"


def vote(records: list[dict], index: int, window: float, minimum: int) -> None:
    """Decide from the neighbourhood whether the clock runs or stands here."""
    times = [r["t"] for r in records]
    t = records[index]["t"]
    lo = bisect.bisect_left(times, t - window)
    hi = bisect.bisect_right(times, t + window)
    near = [r for r in records[lo:hi] if r["cands"]]
    record = records[index]
    # At full frame rate a window holds hundreds of frames; a few dozen spread
    # over it vote just as well and keep this fast.
    if len(near) > 32:
        step = len(near) / 32
        near = [near[int(i * step)] for i in range(32)]
    record["mode"] = None
    if len(near) < minimum:
        return
    running = collections.Counter()
    standing = collections.Counter()
    for other in near:
        keys = {round(value + other["t"] - t, 1) for value, _, _ in other["cands"]}
        for key in keys:
            running[key] += 1
        for value in {value for value, _, _ in other["cands"]}:
            standing[value] += 1

    def support_run(key: float) -> int:
        tolerance = 1.05
        return sum(
            1
            for other in near
            if any(abs(value + other["t"] - t - key) <= tolerance for value, _, _ in other["cands"])
        )

    best_run = max(running, key=support_run, default=None)
    run_support = support_run(best_run) if best_run is not None else 0
    best_stand, stand_support = standing.most_common(1)[0] if standing else (None, 0)
    # A standing clock also fits "running" within the tolerance for about a
    # second, so standing wins ties.
    if stand_support >= run_support and stand_support >= 0.6 * len(near):
        record["mode"] = "stand"
        record["value"] = float(best_stand)
        kinds = collections.Counter(
            k for other in near for v, k, _ in other["cands"] if abs(v - best_stand) < 1e-6
        )
        record["kind"] = kinds.most_common(1)[0][0]
    elif run_support >= 0.6 * len(near):
        matched = [
            (value + other["t"] - t, kind)
            for other in near
            for value, kind, _ in other["cands"]
            if abs(value + other["t"] - t - best_run) <= 1.05
        ]
        record["mode"] = "run"
        record["value"] = statistics.median(v for v, _ in matched)
        # 0:47 and 4.7 s are different values, so the format follows from the
        # readings that fit, not from the value.
        record["kind"] = collections.Counter(k for _, k in matched).most_common(1)[0][0]
    record["support"] = max(run_support, stand_support) / len(near)


def calibrate(records: list[dict]) -> None:
    """Refine running stretches so that the displayed second is known exactly.

    The median gives the value to within half a second. Consecutive running
    frames on the same stretch share value + t; the offset that makes the most
    readings agree with the displayed digits is taken, and how close each frame
    is to a change of the display decides whether the timeline alone may label
    it.
    """
    stretches: list[list[dict]] = []
    for record in records:
        if record.get("mode") != "run":
            continue
        key = record["value"] + record["t"]
        if (
            stretches
            and abs(stretches[-1][-1]["key"] - key) <= 1.05
            and (record["t"] - stretches[-1][-1]["t"] <= 3.0)
        ):
            stretches[-1].append(record)
        else:
            stretches.append([record])
        record["key"] = key
    for stretch in stretches:
        center = statistics.median(r["key"] for r in stretch)
        best = (-1, center)
        for offset in np.arange(center - 1.0, center + 1.0, 0.01):
            hits = 0
            for r in stretch:
                value = offset - r["t"]
                kind = r["kind"]
                if any(d == display(value, kind) for _, k, d in r["cands"] if k == kind):
                    hits += 1
            if hits > best[0]:
                best = (hits, float(offset))
        for r in stretch:
            value = best[1] - r["t"]
            r["value"] = value
            step = 0.1 if r["kind"] == "tenths" else 1.0
            r["phase"] = (value / step) % 1.0
        stretch[0]["stretch_agreement"] = (best[0], len(stretch))


# Digits the engine confuses consistently on this kind of board. A whole minute
# read as 7:59 ... 7:00 instead of 1:59 ... 1:00 is a perfect countdown, so only
# the transition into it gives it away.
CONFUSABLE = ({"1", "7"},)


def continuity(records: list[dict], max_gap: float = 5.0) -> int:
    """Repair stretches that cannot follow the stretch before them.

    The clock cannot jump from 2:00 to 7:59. When a stretch starts a whole
    number of minutes away from where the previous one ended, and its minute
    digits differ from the expected ones only by a confusable pair, it is
    shifted onto the expected value. Anything else - an operator setting the
    clock - is left alone.
    """
    stretches: list[list[dict]] = []
    for record in records:
        mode = record.get("mode")
        if mode is None:
            continue
        key = record["value"] + record["t"] if mode == "run" else record["value"]
        last = stretches[-1][-1] if stretches else None
        if (
            last is not None
            and last["mode"] == mode
            and abs(last["_key"] - key) <= 1.05
            and record["t"] - last["t"] <= 3.0
        ):
            stretches[-1].append(record)
        else:
            stretches.append([record])
        record["_key"] = key
    repaired = 0
    for previous, current in zip(stretches, stretches[1:], strict=False):
        start = current[0]
        if start["t"] - previous[-1]["t"] > max_gap:
            continue
        end = previous[-1]
        expected = end["value"] - (start["t"] - end["t"]) if end["mode"] == "run" else end["value"]
        actual = start["value"]
        diff = actual - expected
        minutes = round(diff / 60)
        if (
            minutes == 0
            or abs(diff - minutes * 60) > 2.0
            or start["kind"] != "mmss"
            or end["kind"] != "mmss"
        ):
            continue
        got = display(actual, "mmss")[:-2]
        want = display(expected - 0.0, "mmss")[:-2]
        got, want = got.rjust(len(want)), want.rjust(len(got))
        if len(got) != len(want) or not all(
            a == b or {a, b} in CONFUSABLE for a, b in zip(got, want, strict=True)
        ):
            continue
        for record in current:
            record["value"] -= minutes * 60
            record["_key"] -= minutes * 60
            record["repaired"] = True
        repaired += 1
    return repaired


def cell_columns(cores: list[np.ndarray], zone, expected: int) -> list[tuple[int, int]]:
    width = cores[0].shape[1]
    occupancy = np.mean([core.any(axis=0) for core in cores], axis=0)
    occupancy[int(zone[0] * width) - 3 : int(zone[2] * width) + 3] = 0
    runs, start = [], None
    for x in range(width + 1):
        lit = x < width and occupancy[x] > 0.03
        if lit and start is None:
            start = x
        if not lit and start is not None:
            if x - start > 6:
                runs.append((start, x))
            start = None
    if len(runs) != expected:
        raise SystemExit(f"expected {expected} digit cells, found {runs}")
    return runs


def cell_chars(digits: str, kind: str, cells: int) -> list[str]:
    """Characters per cell: minutes:seconds right-aligned, tenths leave the last cell dark."""
    used = cells - 1 if kind == "tenths" else cells
    padded = digits.rjust(used, " ")[-used:]
    return list(padded) + [" "] * (cells - used)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("config", type=Path)
    parser.add_argument("--region", default="time")
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--group-prefix", default="", help='e.g. "pen" -> "penmmss"')
    parser.add_argument("--max-minutes", type=int, default=20)
    parser.add_argument("--cells", type=int, default=4, help="digit cells of the clock")
    parser.add_argument("--window", type=float, default=6.0, help="seconds either side")
    parser.add_argument("--min-neighbours", type=int, default=6)
    parser.add_argument("--max-mismatch", type=float, default=0.25)
    parser.add_argument(
        "--confirm-mismatch",
        type=float,
        default=0.08,
        help="templates alone confirm a label this close to them",
    )
    parser.add_argument(
        "--contradiction",
        type=float,
        default=0.7,
        help="review a frame when another template scores below this share of the label's",
    )
    parser.add_argument(
        "--trust-reading",
        action="store_true",
        help="readings come from cluster_cells.py: accept only labels the reading confirms",
    )
    parser.add_argument("--sheet", type=Path, help="write the digit templates here")
    args = parser.parse_args()

    config = ServiceConfig.model_validate_json(args.config.read_text(encoding="utf-8"))
    if args.region not in {r.id for r in config.regions}:
        raise SystemExit(f"unknown region {args.region}")
    by_frame: dict[int, dict] = {}
    for path in args.inputs:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                record["file"] = str((path.parent / record["file"]).resolve())
                by_frame.setdefault(record["frame"], record)
    records = sorted(by_frame.values(), key=lambda r: r["t"])
    for record in records:
        record["cands"] = candidates(record.get("raw"), args.max_minutes)
    print(f"{len(records)} frames, {sum(bool(r['cands']) for r in records)} readable")

    for index in range(len(records)):
        vote(records, index, args.window, args.min_neighbours)
    calibrate(records)
    print(f"stretches repaired by continuity: {continuity(records)}")

    grays = {r["file"]: cv2.imread(r["file"], cv2.IMREAD_GRAYSCALE) for r in records}
    zone = find_colon_zone([grays[r["file"]] for r in records[:: max(1, len(records) // 400)]])
    if zone is None:
        raise SystemExit("could not locate the colon")
    brightness = {f: zone_brightness(g, zone) for f, g in grays.items()}
    low, high = np.percentile(list(brightness.values()), (10, 90))
    colon_threshold = (low + high) / 2 if high - low > 40 else 0.0

    labelled = []
    for record in records:
        record["colon_on"] = brightness[record["file"]] > colon_threshold
        if record.get("mode") is None:
            continue
        value = record["value"]
        kind = record["kind"]
        expected = display(value, kind)
        own = any(d == expected for _, k, d in record["cands"] if k == kind)
        stable = record["mode"] == "stand" or 0.15 <= record.get("phase", 0.5) <= 0.85
        if record["mode"] == "stand":
            # Standing: the reading must agree with the majority exactly.
            own = any(abs(v - value) < 1e-6 for v, _, _ in record["cands"])
        record["kind"] = kind
        record["digits"] = expected
        record["timeline_ok"] = own or stable
        record["own"] = own

    kernel = np.ones((3, 3), np.uint8)

    def core_of(gray: np.ndarray) -> np.ndarray:
        # The white frame lines above and below the digits would light every
        # column; only the band the digits live in is compared.
        height = gray.shape[0]
        band = gray[int(0.12 * height) : int(0.88 * height)]
        return cv2.erode(binarize(band), kernel, iterations=1) > 0

    cores = {r["file"]: core_of(grays[r["file"]]) for r in records}
    # Templates come from the timeline, not from the engine: the engine hardly
    # ever reads some digits right (a lone "1" becomes "17"), the timeline does.
    trusted = [
        r
        for r in records
        if r.get("timeline_ok")
        and r.get("support", 0) >= 0.8
        and (r["mode"] == "stand" or 0.2 <= r.get("phase", 0.5) <= 0.8)
    ]
    cells = cell_columns([cores[r["file"]] for r in trusted[::5]], zone, args.cells)
    print(f"colon zone {tuple(round(v, 3) for v in zone)}, digit cells {cells}")

    def build_templates() -> tuple[dict, dict]:
        groups = collections.defaultdict(list)
        for record in trusted:
            for cell, char in enumerate(cell_chars(record["digits"], record["kind"], args.cells)):
                x1, x2 = cells[cell]
                groups[(cell, char)].append(cores[record["file"]][:, x1:x2])
        built = {k: np.mean(v, axis=0) > 0.5 for k, v in groups.items() if len(v) >= 8}
        return built, groups

    # The engine reads a lone "1" as "7" in every cell, so whole stretches can
    # carry it consistently (a standing 11:21 read as 17:27). Where a cell's
    # 1 and 7 templates disagree clearly with the label, the template wins.
    # Only this pair is touched; every other digit stays as the timeline says.
    swapped = 0
    for _ in range(3):
        templates, groups = build_templates()
        changed = 0
        for record in records:
            if "digits" not in record:
                continue
            chars = cell_chars(record["digits"], record["kind"], args.cells)
            fixed = list(chars)
            for cell, char in enumerate(chars):
                pair = {"1": "7", "7": "1"}.get(char)
                if pair is None or (cell, pair) not in templates or (cell, char) not in templates:
                    continue
                x1, x2 = cells[cell]
                core = cores[record["file"]][:, x1:x2]
                own_score = float((core ^ templates[(cell, char)]).sum())
                other_score = float((core ^ templates[(cell, pair)]).sum())
                if other_score < 0.5 * own_score:
                    fixed[cell] = pair
            if fixed != chars:
                used = args.cells - 1 if record["kind"] == "tenths" else args.cells
                record["digits"] = "".join(fixed[:used]).strip()
                record["swapped"] = True
                changed += 1
        swapped += changed
        if not changed:
            break
    print(f"labels corrected by the 1/7 templates: {swapped}")
    templates, groups = build_templates()
    digit_mass = np.median([t.sum() for (c, ch), t in templates.items() if ch != " "])

    def mismatch(record: dict) -> float:
        worst = 0.0
        for cell, char in enumerate(cell_chars(record["digits"], record["kind"], args.cells)):
            x1, x2 = cells[cell]
            core = cores[record["file"]][:, x1:x2]
            template = templates.get((cell, char))
            if template is None:
                return 1.0
            norm = max(1.0, float(template.sum()) if char != " " else digit_mass)
            worst = max(worst, float((core ^ template).sum()) / norm)
        return worst

    def contradicted(record: dict) -> bool:
        """Another character's template fits some cell clearly better than the label's.

        The board's LED segments flicker against the camera: a 9 can lose its
        top bar and show as a 4 in a single frame. Such a frame must not teach
        the model to see a bar that is not there, whatever the true value is.
        """
        for cell, char in enumerate(cell_chars(record["digits"], record["kind"], args.cells)):
            x1, x2 = cells[cell]
            core = cores[record["file"]][:, x1:x2]
            own = templates.get((cell, char))
            if own is None:
                return True
            own_score = float((core ^ own).sum())
            for (c, other), template in templates.items():
                if (
                    c == cell
                    and other != char
                    and float((core ^ template).sum()) < args.contradiction * own_score
                ):
                    return True
        return False

    confusions = collections.Counter()
    for record in records:
        if "digits" not in record:
            continue
        record["mismatch"] = round(mismatch(record), 3)
        record["contradicted"] = contradicted(record)
        if record["mismatch"] > args.max_mismatch or record["contradicted"]:
            # Which character would fit this frame better, per cell?
            for cell, char in enumerate(cell_chars(record["digits"], record["kind"], args.cells)):
                x1, x2 = cells[cell]
                core = cores[record["file"]][:, x1:x2]
                scores = {
                    ch: float((core ^ t).sum()) for (c, ch), t in templates.items() if c == cell
                }
                if scores:
                    best = min(scores, key=scores.get)
                    if best != char:
                        confusions[(cell, char, best)] += 1
        text = record["digits"]
        if record["colon_on"]:
            text = (
                f"{text[:-1]}:{text[-1]}"
                if record["kind"] == "tenths"
                else f"{text[:-2]}:{text[-2:]}"
            )
        # Close to a change of the display the timeline may be a second off;
        # then every cell has to match its template closely instead.
        confirmed = record["mismatch"] <= args.confirm_mismatch
        if args.trust_reading:
            # The reading comes from hand-labelled pixel groups: it must agree
            # with the timeline, and without one the templates must confirm.
            accepted = record["own"] or (not record["cands"] and confirmed)
        else:
            accepted = record["timeline_ok"] or confirmed
        review = record["mismatch"] > args.max_mismatch or record["contradicted"] or not accepted
        labelled.append(
            {
                "file": record["file"],
                "region": args.region,
                "t": record["t"],
                "text": text,
                "group": args.group_prefix + record["kind"],
                "review": bool(review),
                "ocr": record.get("raw"),
                "mismatch": record["mismatch"],
                "mode": record["mode"],
            }
        )

    with open(args.out, "w", encoding="utf-8") as handle:
        for item in labelled:
            handle.write(json.dumps(item) + "\n")
    summary = collections.Counter((r["group"], "review" if r["review"] else "ok") for r in labelled)
    disagreed = sum(
        1 for r in labelled if not r["review"] and digits_of(r["ocr"]) != digits_of(r["text"])
    )
    print("labels:", dict(sorted(summary.items())))
    print(f"accepted labels that differ from the engine's reading: {disagreed}")
    print(f"unlabelled (no majority around them): {len(records) - len(labelled)}")
    if confusions:
        print("template says otherwise (cell, label, looks like):", confusions.most_common(12))

    if args.sheet:
        tiles = []
        for cell, char in sorted(templates):
            t = (templates[(cell, char)] * 255).astype(np.uint8)
            t = cv2.resize(t, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
            t = cv2.copyMakeBorder(t, 24, 4, 4, 4, cv2.BORDER_CONSTANT, value=90)
            label = f"c{cell} '{char}' n={len(groups[(cell, char)])}"
            cv2.putText(t, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 255, 1)
            tiles.append(t)
        height = max(t.shape[0] for t in tiles)
        width = max(t.shape[1] for t in tiles)
        tiles = [
            cv2.copyMakeBorder(
                t, 0, height - t.shape[0], 0, width - t.shape[1], cv2.BORDER_CONSTANT, value=90
            )
            for t in tiles
        ]
        per_row = 11
        while len(tiles) % per_row:
            tiles.append(np.full((height, width), 90, np.uint8))
        rows = [np.hstack(tiles[i : i + per_row]) for i in range(0, len(tiles), per_row)]
        cv2.imwrite(str(args.sheet), np.vstack(rows))
        print(f"template sheet: {args.sheet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
