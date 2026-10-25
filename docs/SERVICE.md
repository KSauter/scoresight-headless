# ScoreSight service

For the hardened Docker, Cloudflare Access, Nginx Proxy Manager and DeckLink bridge deployment,
see [Production deployment](PRODUCTION.md).

## Local development

Use an isolated Python 3.11 or 3.12 environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check src/scoresight tests
python -m mypy -p scoresight
```

Start the service with an explicit configuration file:

```powershell
scoresight-service --config .\local-config.json
```

The first start creates a validated configuration and a random administrator token.
Open `http://127.0.0.1:18099`, then sign in with `security.admin_token` from the file.
The token is deliberately omitted from API configuration responses.

Binding to `0.0.0.0` is an explicit deployment decision. Create a read-only token with
`POST /api/v1/read-tokens` and use it for the Lightning TV consumer or HTML preview.
Do not expose the service directly to the public internet.

## Runtime dependencies

The web/configuration service installs from the base project dependencies. Actual OCR
also needs the `ocr` extra:

```powershell
uv sync --extra ocr
$env:SCORESIGHT_TESSDATA = "C:\path\to\tessdata"
```

On 64-bit Windows with Python 3.11 through 3.14, the locked dependency uses a
hash-pinned, self-contained `tesserocr` wheel, so a separate system Tesseract install
is not required. Stop a running `scoresight-service` before syncing because Windows
locks its generated executable. Linux and macOS use the normal PyPI package and need
the platform Tesseract development libraries described by `tesserocr`.

Windows DeckLink capture additionally requires Blackmagic Desktop Video and the
`scoresight_decklink` native extension described in `DECKLINK.md`. Without those
components the service stays available and reports capture health as degraded.

## API

- `GET /api/v1/health` and `GET /api/v1/results` provide snapshots.
- `WS /api/v1/events` publishes an initial snapshot followed by latest-only batches.
- `WS /api/v1/preview` carries metadata and JPEG preview frames. The dashboard uses
  `?flow_control=ack` and sends `next` after consuming each image; only the latest
  pending frame is retained until then. Existing streaming clients remain supported.
- `/api/v1/config`, `/sources`, `/profiles`, and `/outputs` support the operator UI.
- `/preview/default?token=...` is the transparent HTML scoreboard view.
- `/metrics` exposes Prometheus text metrics.
- `GET /api/v1/regions/{id}/cells?count=N` proposes `N` digit cells for a saved
  region from what it showed over the last minutes (see below).

Configuration writes use revisions. A stale write returns HTTP 409 rather than
silently overwriting a newer operator change.

## OCR setup workflow

Apply four-corner perspective correction before drawing OCR regions. The rectified
preview keeps the selected quadrilateral's natural aspect ratio, and the browser
canvas follows that ratio so overlay coordinates match the image sent to OCR. A
crop or perspective change clears existing regions after confirmation because it
creates a new coordinate space; save the transform and then redraw the regions.

For each region, choose `Clock / time`, `Number / score`, or `Free text`. Clock and
number fields apply a Tesseract character whitelist and built-in validation in
addition to the optional regular expression. `Confirm frames` controls how many
consecutive matching candidates are required before a new value becomes accepted.
For clock fields, confirmation follows whole readings that agree with the elapsed
capture time, so moving tenths can confirm without repeating identical text.
Clock fields bypass character smoothing, which can invent times at digit rollovers.
Large jumps require consistent readings over at least 1.5 seconds (and at least three
frames), allowing clock resets while holding the last accepted time through short OCR
glitches. Paused and count-up clocks are supported too; this does not extrapolate times
when OCR is missing.
Clock fields support both `mm:ss` and seconds with tenths below one minute, such as
`01:00` → `59.9` → `09.9` → `0.0`, preserving the decimal point in the output.
Any custom regular expression must allow both formats (or use the default `^.*$`).
The selected-region panel shows the exact filtered OCR input, the current candidate,
and the stable last accepted value.
The dashboard keeps one filtered-image request in flight, releases decoded images
after drawing, and pauses its live connections while hidden. Returning to the tab
reconnects to the latest state; background OCR and outputs continue on the service.

### Digit cell engine

Under `Recognition`, `Engine` switches between Tesseract and `Digit cells (neural
network)` (`"engine": "cnn"` under `ocr`). The cell engine classifies every digit cell of
a region on its own - a digit, a dark cell, or a digit caught while changing, which is
rejected instead of guessed - so a lone `1` cannot become `11`. It loads
`<model>.cells.onnx` and `<model>.cells.json` from the same directory as the Tesseract
models and runs on OpenCV, no GPU needed. Regions without cells stay with Tesseract,
so `model` must name both.

Cells are proposed, not drawn: save the region, let the board run for a minute, select
the region, set `Digit cells` (4 for a clock `MM:SS`, 3 for a penalty clock `M:SS`, 2 for
a score, 1 for a period) and press `Detect cells`. Dark cells cannot be seen, so the
proposal places cells that never lit by the spacing of those that did and draws them
dashed; a lone `1` gets a full digit width. Drag a cell edge to adjust, then save. A
region that has never shown anything lit returns no proposal yet.

Below a minute a four-cell clock shows `SS:t` with the last cell dark; the engine reports
that as `27.5`. A clock counts in one direction: once its direction is known, a reading
against it - a changing digit whose old and new segments together look like a 9 - needs
the same 1.5 seconds as any other jump before it is believed.

## Live fan-site WebSocket output

Configure ScoreSight as the OCR client for the fan site's `/ws/ocr/<stream_id>/`
endpoint with an output entry like this:

```json
{
  "kind": "fan_site",
  "enabled": true,
  "settings": {
    "endpoint": "wss://fan.example/ws/ocr",
    "stream_id": "stream1",
    "token": "OCR_INGEST_TOKEN"
  },
  "field_mapping": {
    "Clock": "Clock.Text",
    "Home Score": "score_a",
    "Away Score": "score_b",
    "Period": "period"
  },
  "send_unchanged": false
}
```

The adapter authenticates with `Authorization: Bearer`, sends the optional OCR
registration frame, and then sends accepted values as
`{"type":"ocr","values":{...}}`. It keeps the connection open, checks every
registration and update acknowledgement, reconnects with exponential backoff, and
resends the full stable state after reconnecting. `GET /api/v1/outputs` exposes the
latest saved, ignored, conflict, mode, and game information returned by the fan site.

## Packaging

Build a Windows service bundle with:

```powershell
uv sync --extra ocr
uv pip install pyinstaller
uv run pyinstaller --clean --noconfirm scoresight-service.spec
```

The DeckLink extension and Blackmagic Desktop Video runtime remain external native
prerequisites. For Linux, install the wheel into `/opt/scoresight/.venv`, copy the
example `packaging/systemd/scoresight.service`, and keep configuration/tessdata under
`/var/lib/scoresight`.
