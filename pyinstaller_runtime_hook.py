"""Point the service at the tessdata bundled by PyInstaller.

RuntimeController._build_pipeline resolves the traineddata relative to the
source tree (parents[3] of core/runtime.py). In a frozen build that lands
beside the executable, while PyInstaller collects data files into _internal,
so the models are never found.

SCORESIGHT_TESSDATA is the override the code already honours, and setdefault
keeps an explicitly configured path winning.
"""

import os
import sys

if hasattr(sys, "_MEIPASS"):
    os.environ.setdefault(
        "SCORESIGHT_TESSDATA", os.path.join(sys._MEIPASS, "tesseract", "tessdata")
    )
