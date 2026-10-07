"""Label the frames the automatic pass was not sure about.

Walks ``<review_dir>/<group>/`` as written by ``build_gt.py``, shows each
image that has no ``.gt.txt`` yet with the suggested text filled in, and
writes ``<image>.gt.txt`` on Enter. Type the correct text to override the
suggestion, press Delete on an empty field to skip the image for good (a
``.skip`` file is written), Escape quits. ``--count`` only reports how many
images are left.

    python scripts/training/review_tool.py corpus/gt/review
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def pending(review_dir: Path) -> list[tuple[Path, str]]:
    items = []
    for index in sorted(review_dir.glob("*.jsonl")):
        group = review_dir / index.stem
        for line in index.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            image = group / entry["file"]
            stem = image.with_suffix("")
            if (
                image.exists()
                and not Path(f"{stem}.gt.txt").exists()
                and not Path(f"{stem}.skip").exists()
            ):
                items.append((image, entry.get("suggested") or ""))
    return items


def run(items: list[tuple[Path, str]]) -> None:
    import tkinter as tk

    from PIL import Image, ImageTk

    root = tk.Tk()
    root.title("ScoreSight review")
    label = tk.Label(root)
    label.pack(padx=8, pady=8)
    info = tk.Label(root, text="")
    info.pack()
    entry = tk.Entry(root, width=20, font=("TkDefaultFont", 18), justify="center")
    entry.pack(padx=8, pady=8)
    state = {"index": 0, "photo": None}

    def show() -> None:
        if state["index"] >= len(items):
            root.quit()
            return
        image, suggested = items[state["index"]]
        picture = Image.open(image)
        scale = max(1, int(600 / max(picture.width, 1)))
        picture = picture.resize(
            (picture.width * scale, picture.height * scale), Image.NEAREST
        )
        state["photo"] = ImageTk.PhotoImage(picture)
        label.configure(image=state["photo"])
        info.configure(
            text=f"{state['index'] + 1}/{len(items)}  {image.parent.name}/{image.name}"
        )
        entry.delete(0, tk.END)
        entry.insert(0, suggested)
        entry.focus_set()

    def save(_event=None) -> None:
        image, _ = items[state["index"]]
        text = entry.get().strip()
        if text:
            Path(f"{image.with_suffix('')}.gt.txt").write_text(
                text + "\n", encoding="utf-8"
            )
            state["index"] += 1
            show()

    def skip(_event=None) -> None:
        if entry.get().strip():
            return
        image, _ = items[state["index"]]
        Path(f"{image.with_suffix('')}.skip").write_text("", encoding="utf-8")
        state["index"] += 1
        show()

    entry.bind("<Return>", save)
    entry.bind("<Delete>", skip)
    root.bind("<Escape>", lambda _event: root.quit())
    show()
    root.mainloop()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("review_dir", type=Path)
    parser.add_argument(
        "--count", action="store_true", help="only report how many images are left"
    )
    args = parser.parse_args()
    items = pending(args.review_dir)
    print(f"{len(items)} images to review")
    if items and not args.count:
        run(items)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
