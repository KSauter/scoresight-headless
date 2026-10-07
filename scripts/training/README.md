# Training a board-specific OCR model

The bundled `scoreboard_general` model reads most boards, but a particular
display can defeat it: the Nautronic board in `output/StadionUhr.mp4` shows
tenths as `SS:t` with the fourth cell dark, blinks its colon at 1 Hz, and
uses glyphs the generic model confuses (`7`→`1`, `8`→`0`, `66`→`22`, a lone
`0`→`00`). Fine-tuning Tesseract on a few thousand crops of the actual board
fixed all of that: on a held-out part of the recording the exact-match rate
went from 70.7 % to 100 %.

The scripts in this directory turn a recording plus the service configuration
into a trained `.traineddata`. Run them from the repository root with the
project virtualenv and `PYTHONPATH=src`; training needs Docker.

## 1. Extract crops

One decoding pass cuts every configured region out of the video, skips blank
cells and records the current engine's reading next to each crop.

```bash
python scripts/training/extract_regions.py output/StadionUhr.mp4 scoresight-config-vorschlag.json \
    --step 15 --out output/training-nautronic/corpus/cells
# the last minute of the clock at full frame rate: every tenth is a distinct image
python scripts/training/extract_regions.py output/StadionUhr.mp4 scoresight-config-vorschlag.json \
    --regions time --start 1445 --end 1507 --step 1 --out output/training-nautronic/corpus/clock_tenths
```

## 2. Label the clock

```bash
python scripts/training/label_clock.py scoresight-config-vorschlag.json \
    --inputs output/training-nautronic/corpus/clock_tenths/time.jsonl \
             output/training-nautronic/corpus/cells/time.jsonl \
    --out output/training-nautronic/corpus/labels_time.jsonl
```

Minutes:seconds frames are labelled from the engine's reading when the
neighbours agree it sits on a countdown. Below a minute the engine is too
unreliable for that, so the script fits a start time and rate to the frames
(the agreement it prints should be well above 50 %) and labels every frame
from time alone, then checks each frame against digit templates learned from
the majority: frames where the board still shows the old digit fading out
fail that check and go to review. Whether the colon is lit is read from the
pixels; the label contains a `:` only when it is. If the automatic detection
of the tenths window is off, pass `--tenths-start` / `--tenths-end`.

## 3. Label the other cells

```bash
python scripts/training/label_cells.py scoresight-config-vorschlag.json output/training-nautronic/corpus/cells --list
```

prints when each cell was lit and what the engine made of it. Look at a few
crops per block, write the truth into a ranges file and label:

```json
{"awayPenNr1": [[263, 401, "2"], [403, 414, "3"], [419, 633, "66"]],
 "homePenNr1": [[117, 238, "11"], [536, 569, "75"], [613, 905, "5"]]}
```

```bash
python scripts/training/label_cells.py scoresight-config-vorschlag.json output/training-nautronic/corpus/cells \
    --out output/training-nautronic/corpus/labels_cells.jsonl \
    --fixed homeScore=0 awayScore=0 period=1 --ranges output/training-nautronic/ranges.json
```

Penalty clocks are labelled like the game clock. Frames outside any range and
readings without consensus go to review.

## 4. Build the ground truth

```bash
python scripts/training/build_gt.py scoresight-config-vorschlag.json \
    --labels output/training-nautronic/corpus/labels_time.jsonl output/training-nautronic/corpus/labels_cells.jsonl \
    --out output/training-nautronic/gt --holdout 600-700,1300-1350,1458-1466,1550-1560
```

Every crop is run through the service's own preprocessing, so the model
trains on exactly the images it will see at runtime. `--holdout` keeps whole
stretches of the recording out of training; neighbouring frames are near
duplicates, so a random split would flatter the result. Static displays are
capped per text so the clock dominates. Check `gt/check_gt.jpg` before
training.

Frames the scripts were not sure about land in `gt/review/`. They are not
needed for a good model, but if you want them:

```bash
python scripts/training/review_tool.py output/training-nautronic/gt/review   # Enter accepts, type to correct
python scripts/training/build_gt.py ... --reviewed output/training-nautronic/gt/review
```

## 5. Train

```bash
mkdir -p output/training-nautronic/train && cp -r output/training-nautronic/gt/train output/training-nautronic/train/ground-truth
scripts/training/train.sh output/training-nautronic/train 20000
```

builds the Docker image on first use, fine-tunes `scoreboard_general` with
tesstrain and stops early once the training error reaches zero. Fewer than
ten minutes on a desktop CPU for ~4600 images. `MODEL_NAME` and `START_MODEL`
can be overridden through the environment.

## 6. Evaluate and deploy

```bash
cp output/training-nautronic/train/data/nautronic.traineddata tesseract/tessdata/
python scripts/training/eval_models.py output/training-nautronic/gt holdout --models scoreboard_general,nautronic
```

Then set `"model": "nautronic"` under `ocr` in the service configuration. The
service bundle and the container image ship everything in
`tesseract/tessdata`, so the new model travels with them.

With a model that is confident on this board the `time` region no longer
needs `confidence_threshold: 0.0`; `0.5` now rejects what a switched-off
board produces.

## Adding footage

More recordings of the same board widen what the model has seen: goals (the
scores here never left `0`), other penalty numbers, a different camera
position. Extract and label each recording into its own corpus directory,
pass all label files to `build_gt.py` and train again - the labels of earlier
recordings stay valid.
