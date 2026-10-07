#!/usr/bin/env bash
# Fine-tune a Tesseract model on the ground truth written by build_gt.py.
#
#   scripts/training/train.sh <work_dir> [max_iterations] [learning_rate]
#
# <work_dir>/ground-truth holds the *.png + *.gt.txt pairs (copy gt/train there).
# The result is <work_dir>/data/<MODEL_NAME>.traineddata; copy it to
# tesseract/tessdata and select it with "ocr.model" in the service configuration.
#
# Environment: MODEL_NAME (default nautronic), START_MODEL (default
# scoreboard_general), IMAGE (docker image, built from the Dockerfile next to
# this script when missing).
set -euo pipefail

WORK=$(realpath "${1:?work dir}")
ITERATIONS=${2:-20000}
LEARNING_RATE=${3:-0.0001}
MODEL_NAME=${MODEL_NAME:-nautronic}
START_MODEL=${START_MODEL:-scoreboard_general}
IMAGE=${IMAGE:-scoresight-tesstrain}
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  docker build -t "$IMAGE" "$HERE"
fi

mkdir -p "$WORK/data/$MODEL_NAME-ground-truth" "$WORK/tessdata"
cp "$REPO/tesseract/tessdata/$START_MODEL.traineddata" "$WORK/tessdata/"
rsync -a --delete --include='*.png' --include='*.gt.txt' --exclude='*' \
  "$WORK/ground-truth/" "$WORK/data/$MODEL_NAME-ground-truth/"

docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$WORK/data:/tesstrain/data" -v "$WORK/tessdata:/tessdata:ro" \
  "$IMAGE" \
  make -j"$(nproc)" training MODEL_NAME="$MODEL_NAME" START_MODEL="$START_MODEL" \
    TESSDATA=/tessdata MAX_ITERATIONS="$ITERATIONS" LEARNING_RATE="$LEARNING_RATE" \
    RATIO_TRAIN=0.95 DEBUG_INTERVAL=0 PY_CMD=/venv/bin/python3

echo "model: $WORK/data/$MODEL_NAME.traineddata"
