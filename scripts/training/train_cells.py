"""Train the digit cell network and export it for the service.

Reads what ``build_cells.py`` wrote, trains a small convolutional network on the
cells of everything outside the held-out stretches, reports how it does on
those stretches - per cell, per whole reading, with the cells shifted by up to
3 px, and how confident it was when it was wrong - and writes
``<model>.cells.onnx`` and ``<model>.cells.json`` for the service's ``cnn``
engine. The export is checked against OpenCV's DNN module, which the service
uses to run it.

PyTorch is not a dependency of the service; run this from an environment that
has it, with the repository's ``src`` and ``scripts/training`` on the path::

    uv venv /tmp/cells && uv pip install --python /tmp/cells/bin/python \\
        torch onnx onnxscript opencv-python-headless pydantic
    PYTHONPATH=src:scripts/training /tmp/cells/bin/python scripts/training/train_cells.py \\
        corpus/cells-nn --model nautronic --out tesseract/tessdata \\
        --holdout 2:300-400,600-700 1:600-700
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from scoresight.ocr.cells import INPUT_HEIGHT, INPUT_WIDTH, MIX, cell_input


def network(classes: int) -> nn.Module:
    def block(inputs: int, outputs: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(inputs, outputs, 3, padding=1),
            nn.BatchNorm2d(outputs),
            nn.ReLU(),
            nn.Conv2d(outputs, outputs, 3, padding=1),
            nn.BatchNorm2d(outputs),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

    return nn.Sequential(
        block(1, 24),
        block(24, 48),
        block(48, 96),
        nn.Flatten(),
        nn.Dropout(0.3),
        nn.Linear(96 * (INPUT_HEIGHT // 8) * (INPUT_WIDTH // 8), 128),
        nn.ReLU(),
        nn.Linear(128, classes),
    )


class Cells(Dataset):
    """Cells cut from their context window; shifted at random when training."""

    def __init__(self, windows, items, indices, context, *, augment=False, shift=0):
        self.windows, self.items, self.indices = windows, items, indices
        self.context, self.augment, self.shift = context, augment, shift

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, k):
        index = self.indices[k]
        win = self.windows[index]
        height, width = win.shape
        c = self.context
        if self.augment:
            dx, dy, dw = random.randint(-4, 4), random.randint(-4, 4), random.randint(0, 3)
        elif self.shift:
            rng = np.random.default_rng(index)
            dx = int(rng.integers(-self.shift, self.shift + 1))
            dy = int(rng.integers(-self.shift, self.shift + 1))
            dw = 0
        else:
            dx = dy = dw = 0
        image = cell_input(win, c + dx - dw, width - c + dx + dw, c + dy, height - c + dy)
        if self.augment:
            image = image * random.uniform(0.8, 1.2) + random.uniform(-0.2, 0.2)
            if random.random() < 0.3:
                image = cv2.GaussianBlur(image, (3, 3), 0)
        return torch.from_numpy(np.ascontiguousarray(image)[None]), self.items[index]["cls"]


def parse_holdout(specs: list[str]) -> dict[str, list[tuple[float, float]]]:
    ranges: dict[str, list[tuple[float, float]]] = {}
    for spec in specs:
        video, spans = spec.split(":", 1)
        for span in spans.split(","):
            start, end = span.split("-")
            ranges.setdefault(video, []).append((float(start), float(end)))
    return ranges


def predict(model, dataset, device) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for x, _ in DataLoader(dataset, batch_size=1024, num_workers=8):
            out.append(torch.softmax(model(x.to(device)).float(), 1).cpu())
    model.train()
    return torch.cat(out).numpy()


def report(name, probs, truth, classes) -> None:
    mix = classes.index(MIX)
    pred = probs.argmax(1)
    conf = probs.max(1)
    wrong = (pred != truth) & (pred != mix) & (truth != mix)
    print(f"=== {name}")
    print(f"  cells correct {np.mean(pred == truth):.3%} of {len(truth)}")
    print(
        "  read as a different character (not mix):",
        int(wrong.sum()),
        "- confidences:",
        sorted(round(float(c), 3) for c in conf[wrong])[-10:],
    )
    pairs = collections.Counter(
        (classes[t], classes[p]) for t, p in zip(truth, pred, strict=True) if t != p
    )
    print("  confusions (true, read):", pairs.most_common(8))
    good = conf[(pred == truth) & (truth != mix)]
    for threshold in (0.9, 0.99):
        print(f"  correct cells below confidence {threshold}: {np.mean(good < threshold):.2%}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("data", type=Path, help="directory written by build_cells.py")
    parser.add_argument("--model", required=True, help="name, e.g. nautronic")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--holdout", nargs="*", default=[], metavar="VIDEO:FROM-TO,...")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    meta = json.loads((args.data / "items.json").read_text(encoding="utf-8"))
    classes, context, items = meta["classes"], meta["context"], meta["items"]
    windows = np.load(args.data / "cells.npy", allow_pickle=True)
    holdout = parse_holdout(args.holdout)
    held = np.array(
        [any(a <= it["t"] < b for a, b in holdout.get(it["video"], [])) for it in items]
    )
    train_idx, hold_idx = np.flatnonzero(~held), np.flatnonzero(held)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = network(len(classes)).to(device)
    print(f"{len(train_idx)} training cells, {len(hold_idx)} held out, device {device}")

    weights = torch.ones(len(classes))
    weights[classes.index(MIX)] = 4.0
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))
    optimiser = torch.optim.AdamW(model.parameters(), 2e-3, weight_decay=1e-4)
    loader = DataLoader(
        Cells(windows, items, train_idx, context, augment=True),
        batch_size=256,
        shuffle=True,
        num_workers=8,
        persistent_workers=True,
    )
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, 3e-3, total_steps=args.epochs * len(loader)
    )
    truth = np.array([items[i]["cls"] for i in hold_idx])
    for epoch in range(args.epochs):
        started, total, seen = time.time(), 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            schedule.step()
            total += loss.item() * len(y)
            seen += len(y)
        line = f"epoch {epoch + 1}: loss {total / seen:.4f}"
        if len(hold_idx):
            probs = predict(model, Cells(windows, items, hold_idx, context), device)
            line += f", held-out cells correct {np.mean(probs.argmax(1) == truth):.3%}"
        print(f"{line} ({time.time() - started:.0f}s)")

    if len(hold_idx):
        report(
            "held out",
            predict(model, Cells(windows, items, hold_idx, context), device),
            truth,
            classes,
        )
        report(
            "held out, shifted up to 3 px",
            predict(model, Cells(windows, items, hold_idx, context, shift=3), device),
            truth,
            classes,
        )

    args.out.mkdir(parents=True, exist_ok=True)
    onnx_path = args.out / f"{args.model}.cells.onnx"
    model.eval().cpu()
    sample = torch.zeros(1, 1, INPUT_HEIGHT, INPUT_WIDTH)
    torch.onnx.export(
        model,
        (sample,),
        str(onnx_path),
        input_names=["cells"],
        output_names=["logits"],
        dynamic_axes={"cells": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    (args.out / f"{args.model}.cells.json").write_text(
        json.dumps({"classes": classes, "input": [INPUT_WIDTH, INPUT_HEIGHT]}, indent=1) + "\n",
        encoding="utf-8",
    )
    # The service runs the network through OpenCV: it must agree with PyTorch.
    check = np.stack(
        [
            Cells(windows, items, hold_idx if len(hold_idx) else train_idx, context)[k][0].numpy()
            for k in range(64)
        ]
    )
    net = cv2.dnn.readNetFromONNX(str(onnx_path))
    net.setInput(check)
    opencv = net.forward()
    with torch.no_grad():
        torch_out = model(torch.from_numpy(check)).numpy()
    print(
        f"wrote {onnx_path} ({onnx_path.stat().st_size // 1024} KiB); "
        f"OpenCV vs PyTorch max difference {np.abs(opencv - torch_out).max():.2e}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
