from __future__ import annotations

import importlib
import threading
import time
from datetime import UTC, datetime
from typing import Any

from scoresight.capture.base import CaptureDevice, FramePacket

# NDI capture, restoring the source the Qt application offered (src/ndi.py).
# Many venues feed the scoreboard camera through an NDI encoder rather than a
# capture card, so without it the service cannot see the board at all.
#
# The legacy implementation used cyndilib. This one uses ndi-python, which
# tracks the NDI SDK release line, ships a single 30 MB runtime DLL instead of
# three totalling 95 MB, and is not marked alpha. Both bundle the native
# runtime, so no separate NDI SDK installation is required.


class NDIUnavailable(RuntimeError):
    pass


_lock = threading.Lock()
_ndi: Any | None = None
_finder: Any | None = None


def _load_ndi() -> Any:
    """Import NDIlib and initialise the runtime once per process.

    Deliberately not imported at module load: the module must stay importable
    on machines that never use an NDI source.
    """
    global _ndi
    with _lock:
        if _ndi is not None:
            return _ndi
        try:
            module = importlib.import_module("NDIlib")
        except ImportError as exc:
            raise NDIUnavailable(
                "NDI support requires the optional ndi-python package"
            ) from exc
        if not module.initialize():
            # Returns false when the CPU lacks SSE4.2, or on Linux when Avahi
            # is missing.
            raise NDIUnavailable(
                "NDI runtime failed to initialise; on Linux this usually means "
                "Avahi is not running"
            )
        _ndi = module
        return _ndi


def _shared_finder(ndi: Any) -> Any:
    """Return the process-wide finder.

    The finder runs a discovery thread and accumulates the sources announced on
    the network. A freshly created one reports an incomplete list for the first
    few hundred milliseconds, which would make a receiver intermittently fail
    to locate a source that is in fact present.
    """
    global _finder
    with _lock:
        if _finder is None:
            _finder = ndi.find_create_v2(ndi.FindCreate())
            if _finder is None:
                raise NDIUnavailable("could not create NDI finder")
        return _finder


class NDICapture:
    """Receives one NDI source as single frames.

    Unlike DeckLink, NDI has no fixed mode list: the sender decides resolution
    and frame rate, so CaptureDevice.modes stays empty.
    """

    def __init__(self, source_name: str, *, connect_timeout: float = 5.0) -> None:
        self.source_name = source_name
        self.connect_timeout = connect_timeout
        self._receiver: Any | None = None
        self._sequence = 0

    @classmethod
    def discover(cls, timeout: float = 1.0) -> list[CaptureDevice]:
        ndi = _load_ndi()
        finder = _shared_finder(ndi)
        # find_wait_for_sources reports whether the list *changed*. A steady
        # network returns false while the known sources remain valid, so the
        # result is discarded and the current list read either way.
        ndi.find_wait_for_sources(finder, int(timeout * 1000))
        return [
            CaptureDevice(id=source.ndi_name, name=source.ndi_name)
            for source in ndi.find_get_current_sources(finder)
        ]

    def _locate(self, ndi: Any) -> Any:
        finder = _shared_finder(ndi)
        deadline = time.monotonic() + self.connect_timeout
        while True:
            for source in ndi.find_get_current_sources(finder):
                if source.ndi_name == self.source_name:
                    return source
            if time.monotonic() >= deadline:
                break
            # A source can be missing briefly because the encoder restarted.
            # Waiting through one discovery round is cheaper than failing and
            # retrying at the next reconnect.
            ndi.find_wait_for_sources(finder, 500)

        available = ", ".join(
            source.ndi_name for source in ndi.find_get_current_sources(finder)
        )
        raise RuntimeError(
            f"NDI source {self.source_name!r} not found "
            f"(available: {available or 'none'})"
        )

    def open(self) -> None:
        ndi = _load_ndi()
        source = self._locate(ndi)

        settings = ndi.RecvCreateV3()
        # BGRX_BGRA keeps the conversion on the SDK side and yields the layout
        # preprocess() expects after dropping alpha.
        settings.color_format = ndi.RECV_COLOR_FORMAT_BGRX_BGRA
        settings.bandwidth = ndi.RECV_BANDWIDTH_HIGHEST
        settings.allow_video_fields = False
        settings.source_to_connect_to = source
        settings.ndi_recv_name = "ScoreSight"

        receiver = ndi.recv_create_v3(settings)
        if receiver is None:
            raise RuntimeError(
                f"could not create NDI receiver for {self.source_name!r}"
            )
        ndi.recv_connect(receiver, source)
        self._receiver = receiver

    def read_latest(self, timeout: float = 1.0) -> FramePacket | None:
        if self._receiver is None:
            raise RuntimeError("capture is not open")
        ndi = _load_ndi()

        deadline = time.monotonic() + timeout
        while True:
            remaining_ms = max(1, int((deadline - time.monotonic()) * 1000))
            # Audio and metadata are declined at the source rather than
            # received and discarded, which is what the Qt implementation did.
            frame_type, video, _audio, _metadata = ndi.recv_capture_v3(
                self._receiver, remaining_ms, True, False, False
            )

            if frame_type == ndi.FRAME_TYPE_VIDEO:
                try:
                    return self._build_packet(video)
                finally:
                    # Every captured video frame is owned by the SDK until it is
                    # handed back; omitting this leaks the full frame buffer per
                    # frame, which at 50 fps exhausts memory within minutes.
                    ndi.recv_free_video_v2(self._receiver, video)

            if frame_type == ndi.FRAME_TYPE_ERROR:
                return None
            if time.monotonic() >= deadline:
                return None

    def _build_packet(self, video: Any) -> FramePacket | None:
        import cv2
        import numpy as np

        width = int(video.xres)
        height = int(video.yres)
        if width <= 0 or height <= 0:
            # Occurs while the sender switches resolution.
            return None

        # The buffer is padded to line_stride_in_bytes, which is not necessarily
        # width * 4, so the rows are addressed through the stride and the
        # padding is sliced off afterwards.
        stride = int(video.line_stride_in_bytes) or width * 4
        buffer = np.asarray(video.data, dtype=np.uint8).reshape(-1)
        expected = stride * height
        if buffer.size < expected:
            return None
        padded = buffer[:expected].reshape(height, stride // 4, 4)[:, :width, :]

        # Downstream expects three channels: preprocess() tests for ndim == 3
        # and then calls COLOR_BGR2GRAY, which fails on four channels.
        # np.ascontiguousarray because the stride slice above leaves a view.
        image = cv2.cvtColor(np.ascontiguousarray(padded), cv2.COLOR_BGRA2BGR)

        self._sequence += 1
        return FramePacket(
            sequence=self._sequence,
            image=image,
            width=width,
            height=height,
            captured_at=datetime.now(UTC),
            monotonic_ns=time.perf_counter_ns(),
            pixel_format="bgr24",
        )

    def close(self) -> None:
        if self._receiver is not None:
            _load_ndi().recv_destroy(self._receiver)
            self._receiver = None
