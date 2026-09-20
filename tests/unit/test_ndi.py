from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from scoresight.capture import ndi as ndi_module
from scoresight.capture.ndi import NDICapture, NDIUnavailable

FRAME_TYPE_VIDEO = 1
FRAME_TYPE_ERROR = 2
FRAME_TYPE_NONE = 3


class FakeSource:
    def __init__(self, name: str) -> None:
        self.ndi_name = name
        self.url_address = "10.0.0.1:5961"


class FakeVideoFrame:
    """Mimics NDIlib.VideoFrameV2 including the stride padding."""

    def __init__(self, width: int, height: int, stride: int | None = None) -> None:
        self.xres = width
        self.yres = height
        self.line_stride_in_bytes = stride if stride is not None else width * 4
        rows = []
        for row in range(height):
            pixels = np.tile(
                np.array([10, 20, 30, 255], dtype=np.uint8), (self.line_stride_in_bytes // 4, 1)
            )
            # Mark the padding so a wrong slice shows up as a colour shift.
            pixels[width:] = [99, 99, 99, 255]
            rows.append(pixels.reshape(-1))
        self.data = np.concatenate(rows) if rows else np.empty(0, dtype=np.uint8)


class FakeNDIlib(types.ModuleType):
    FRAME_TYPE_VIDEO = FRAME_TYPE_VIDEO
    FRAME_TYPE_ERROR = FRAME_TYPE_ERROR
    RECV_COLOR_FORMAT_BGRX_BGRA = 0
    RECV_BANDWIDTH_HIGHEST = 100

    def __init__(self, sources: list[str], frames: list[tuple[int, object]]) -> None:
        super().__init__("NDIlib")
        self._sources = [FakeSource(name) for name in sources]
        self._frames = list(frames)
        self.freed: list[object] = []
        self.destroyed = False

    def initialize(self) -> bool:
        return True

    class RecvCreateV3:
        color_format = None
        bandwidth = None
        allow_video_fields = None
        source_to_connect_to = None
        ndi_recv_name = None

    class FindCreate:
        pass

    def find_create_v2(self, settings=None):
        return object()

    def find_wait_for_sources(self, finder, timeout_ms):
        return True

    def find_get_current_sources(self, finder):
        return self._sources

    def recv_create_v3(self, settings):
        return object()

    def recv_connect(self, receiver, source):
        return None

    def recv_capture_v3(self, receiver, timeout_ms, want_video, want_audio, want_metadata):
        assert want_video is True
        assert want_audio is False, "audio must be declined at the source"
        assert want_metadata is False, "metadata must be declined at the source"
        if not self._frames:
            return FRAME_TYPE_NONE, None, None, None
        frame_type, video = self._frames.pop(0)
        return frame_type, video, None, None

    def recv_free_video_v2(self, receiver, video):
        self.freed.append(video)

    def recv_destroy(self, receiver):
        self.destroyed = True


@pytest.fixture
def fake_ndi(monkeypatch):
    def install(sources: list[str], frames: list[tuple[int, object]]) -> FakeNDIlib:
        module = FakeNDIlib(sources, frames)
        monkeypatch.setitem(sys.modules, "NDIlib", module)
        monkeypatch.setattr(ndi_module, "_ndi", None)
        monkeypatch.setattr(ndi_module, "_finder", None)
        return module

    yield install
    ndi_module._ndi = None
    ndi_module._finder = None


def test_discover_maps_announced_names(fake_ndi):
    fake_ndi(["STUDIO (Camera 1)", "OB-VAN (Programme)"], [])
    devices = NDICapture.discover(0.1)
    assert [device.id for device in devices] == ["STUDIO (Camera 1)", "OB-VAN (Programme)"]
    assert devices[0].name == "STUDIO (Camera 1)"


def test_missing_package_raises_unavailable(monkeypatch):
    monkeypatch.setattr(ndi_module, "_ndi", None)
    monkeypatch.setitem(sys.modules, "NDIlib", None)
    with pytest.raises(NDIUnavailable):
        NDICapture.discover(0.1)


def test_unknown_source_names_the_alternatives(fake_ndi):
    fake_ndi(["STUDIO (Camera 1)"], [])
    capture = NDICapture("STUDIO (Camera 2)", connect_timeout=0.0)
    with pytest.raises(RuntimeError, match="STUDIO \\(Camera 1\\)"):
        capture.open()


def test_frame_drops_alpha_and_stride_padding(fake_ndi):
    # A 1920-wide frame padded to 2048 pixels per row is the layout the SDK
    # hands over on several encoders.
    frame = FakeVideoFrame(width=8, height=4, stride=2048 * 4 // 256)
    fake_ndi(["CAM"], [(FRAME_TYPE_VIDEO, frame)])

    capture = NDICapture("CAM")
    capture.open()
    packet = capture.read_latest(1.0)

    assert packet is not None
    assert packet.width == 8
    assert packet.height == 4
    assert packet.pixel_format == "bgr24"
    assert packet.image.shape == (4, 8, 3), "alpha channel must be dropped"
    assert packet.image.flags["C_CONTIGUOUS"], "preprocess needs a contiguous buffer"
    # Padding must not bleed into the image.
    assert np.array_equal(np.unique(packet.image.reshape(-1, 3), axis=0), np.array([[10, 20, 30]]))


def test_video_frame_is_always_returned_to_the_sdk(fake_ndi):
    frame = FakeVideoFrame(width=4, height=2)
    module = fake_ndi(["CAM"], [(FRAME_TYPE_VIDEO, frame)])

    capture = NDICapture("CAM")
    capture.open()
    capture.read_latest(1.0)

    # Leaking one buffer per frame exhausts memory within minutes at 50 fps.
    assert module.freed == [frame]


def test_zero_sized_frame_is_still_returned_to_the_sdk(fake_ndi):
    frame = FakeVideoFrame(width=0, height=0)
    module = fake_ndi(["CAM"], [(FRAME_TYPE_VIDEO, frame)])

    capture = NDICapture("CAM")
    capture.open()
    assert capture.read_latest(1.0) is None
    assert module.freed == [frame], "a discarded frame must be freed as well"


def test_error_frame_yields_no_packet(fake_ndi):
    fake_ndi(["CAM"], [(FRAME_TYPE_ERROR, None)])
    capture = NDICapture("CAM")
    capture.open()
    assert capture.read_latest(0.5) is None


def test_read_before_open_is_rejected(fake_ndi):
    fake_ndi(["CAM"], [])
    with pytest.raises(RuntimeError, match="not open"):
        NDICapture("CAM").read_latest(0.1)


def test_close_releases_the_receiver(fake_ndi):
    module = fake_ndi(["CAM"], [])
    capture = NDICapture("CAM")
    capture.open()
    capture.close()
    assert module.destroyed
    capture.close()  # idempotent
