from __future__ import annotations

from datetime import UTC, datetime, timedelta

import cv2
import numpy as np
import pytest

from scoresight.capture.base import FramePacket
from scoresight.core.models import (
    NormalizedRect,
    Point,
    PreprocessConfig,
    RegionConfig,
    ResultState,
)
from scoresight.ocr.base import Recognition
from scoresight.ocr.pipeline import RecognitionPipeline
from scoresight.ocr.preprocess import is_blank, preprocess, transform_frame
from scoresight.ocr.smoothing import ClockTracker


class FakeEngine:
    def __init__(self, recognitions: list[Recognition]) -> None:
        self.recognitions = iter(recognitions)
        self.closed = False

    def recognize(self, image, *, region_id: str) -> Recognition:
        assert image.size > 0
        return next(self.recognitions)

    def close(self) -> None:
        self.closed = True


class FakeBatchEngine(FakeEngine):
    def __init__(self, recognitions: list[Recognition]) -> None:
        super().__init__(recognitions)
        self.batch_called = False

    def recognize_many(self, requests) -> list[Recognition]:
        self.batch_called = True
        return [next(self.recognitions) for _ in requests]


@pytest.mark.parametrize(
    "field_type,value", [("number", "27"), ("time", "1:49"), ("text", "A")]
)
@pytest.mark.parametrize("smoothing_window", [1, 5])
def test_confirmed_blank_clears_value_and_smoothing(
    field_type, value, smoothing_window, monkeypatch
):
    class Clock:
        tick = datetime(2026, 1, 1, tzinfo=UTC)

        @classmethod
        def now(cls, tz):
            cls.tick += timedelta(seconds=1)
            return cls.tick

    monkeypatch.setattr("scoresight.ocr.pipeline.datetime", Clock)
    region = RegionConfig(
        id="field",
        name="Field",
        rect=NormalizedRect(x=0, y=0, width=0.5, height=0.5),
        field_type=field_type,
        format_regex=r".+",
        confirmation_frames=2,
        smoothing_window=smoothing_window,
    )
    pipeline = RecognitionPipeline(
        FakeEngine(
            [
                Recognition(value, 0.99),
                Recognition(value, 0.99),
                Recognition("", 0),
                Recognition(value, 0.99),
                Recognition(" ", 0),
                Recognition("", 0),
                Recognition("", 0),
                Recognition(value, 0.99),
                Recognition(value, 0.99),
            ]
        ),
        [region],
    )
    results = [pipeline.process(frame(i)).fields[0] for i in range(9)]
    assert results[1].value == value
    assert results[2].state == ResultState.PENDING
    assert results[2].candidate_value == ""
    assert results[2].value == value
    assert results[3].state == ResultState.UNCHANGED
    assert results[4].state == ResultState.PENDING
    assert results[5].state == ResultState.OK
    assert results[5].value == ""
    assert results[5].changed_at > results[1].changed_at
    assert results[6].state == ResultState.EMPTY
    assert results[6].value == ""
    assert results[6].changed_at == results[5].changed_at
    assert results[7].state == ResultState.PENDING
    assert results[7].value == ""
    assert results[8].value == value


def test_blank_initial_read_is_confirmed_without_turning_into_zero():
    region = RegionConfig(
        name="Score",
        rect=NormalizedRect(x=0, y=0, width=0.5, height=0.5),
        field_type="number",
        remove_leading_zeros=True,
        confirmation_frames=1,
    )
    pipeline = RecognitionPipeline(
        FakeEngine(
            [
                Recognition("", 0),
                Recognition("0", 0.99),
                Recognition("", 0),
            ]
        ),
        [region],
    )
    results = [pipeline.process(frame(i)).fields[0] for i in range(3)]
    assert [result.value for result in results] == ["", "0", ""]
    assert all(result.state == ResultState.OK for result in results)


def frame(sequence: int = 1) -> FramePacket:
    # Not an all-zero image: the pipeline skips regions without contrast, so a
    # blank frame would never reach the engine and these tests would assert
    # against results nobody produced.
    image = np.zeros((100, 200), dtype=np.uint8)
    image[10:90, 10:190] = 200
    return FramePacket(
        sequence=sequence,
        image=image,
        width=200,
        height=100,
        captured_at=datetime.now(UTC),
    )


def test_pipeline_validates_and_marks_unchanged_results() -> None:
    region = RegionConfig(
        id="clock",
        name="Clock",
        rect=NormalizedRect(x=0, y=0, width=0.5, height=0.5),
        format_regex=r"\d{2}:\d{2}",
        confirmation_frames=1,
    )
    engine = FakeEngine([Recognition("12:34", 0.99), Recognition("12:34", 0.99)])
    pipeline = RecognitionPipeline(engine, [region])

    first = pipeline.process(frame(1))
    second = pipeline.process(frame(2))

    assert first.sequence == 1
    assert first.fields[0].state == ResultState.OK
    assert second.fields[0].state == ResultState.UNCHANGED
    assert second.fields[0].changed_at == first.fields[0].changed_at


def test_pipeline_rejects_low_confidence_and_invalid_format() -> None:
    regions = [
        RegionConfig(
            id="a",
            name="A",
            rect=NormalizedRect(x=0, y=0, width=0.4, height=0.4),
            confidence_threshold=0.8,
            confirmation_frames=1,
        ),
        RegionConfig(
            id="b",
            name="B",
            rect=NormalizedRect(x=0.5, y=0, width=0.4, height=0.4),
            format_regex=r"\d+",
            confirmation_frames=1,
        ),
    ]
    pipeline = RecognitionPipeline(
        FakeEngine([Recognition("1", 0.2), Recognition("BAD", 0.99)]), regions
    )
    result = pipeline.process(frame())
    assert [field.state for field in result.fields] == [
        ResultState.REJECTED,
        ResultState.REJECTED,
    ]


def test_rejected_candidate_does_not_replace_last_accepted_value() -> None:
    region = RegionConfig(
        id="score",
        name="Score",
        rect=NormalizedRect(x=0, y=0, width=0.5, height=0.5),
        format_regex=r"\d+",
        confidence_threshold=0.8,
        confirmation_frames=1,
    )
    pipeline = RecognitionPipeline(
        FakeEngine([Recognition("42", 0.99), Recognition("noise", 0.2)]), [region]
    )

    accepted = pipeline.process(frame(1)).fields[0]
    rejected = pipeline.process(frame(2)).fields[0]

    assert accepted.value == "42"
    assert accepted.candidate_value == "42"
    assert rejected.state == ResultState.REJECTED
    assert rejected.value == "42"
    assert rejected.candidate_value == "noise"
    assert rejected.changed_at == accepted.changed_at


def test_pipeline_exposes_exact_filtered_ocr_input_as_png() -> None:
    region = RegionConfig(
        id="clock",
        name="Clock",
        rect=NormalizedRect(x=0, y=0, width=0.5, height=0.5),
        confirmation_frames=1,
    )
    pipeline = RecognitionPipeline(FakeEngine([Recognition("1:23", 0.99)]), [region])
    pipeline.process(frame())
    assert pipeline.latest_previews["clock"].startswith(b"\x89PNG\r\n\x1a\n")


def test_frame_crop_uses_normalized_coordinates() -> None:
    image = np.arange(100 * 200, dtype=np.uint16).reshape((100, 200))
    cropped = transform_frame(
        image,
        crop=NormalizedRect(x=0.25, y=0.2, width=0.5, height=0.5),
    )
    assert cropped.shape == (50, 100)
    assert cropped[0, 0] == image[20, 50]


def test_autocrop_filter_removes_uniform_border() -> None:
    from scoresight.core.models import PreprocessConfig
    from scoresight.ocr.preprocess import preprocess

    image = np.zeros((20, 30), dtype=np.uint8)
    image[5:15, 8:22] = 255
    filtered = preprocess(
        image,
        PreprocessConfig(threshold_method="none", autocrop=True),
    )
    assert filtered.shape == (12, 16)


def test_perspective_transform_preserves_selected_quadrilateral_aspect() -> None:
    image = np.zeros((100, 200), dtype=np.uint8)
    transformed = transform_frame(
        image,
        perspective=[
            Point(x=0.05, y=0.05),
            Point(x=0.95, y=0.05),
            Point(x=0.95, y=0.95),
            Point(x=0.05, y=0.95),
        ],
    )
    assert transformed.shape == (90, 180)


def test_crop_after_perspective_uses_rectified_dimensions() -> None:
    image = np.zeros((100, 200), dtype=np.uint8)
    transformed = transform_frame(
        image,
        perspective=[
            Point(x=0.05, y=0.05),
            Point(x=0.95, y=0.05),
            Point(x=0.95, y=0.95),
            Point(x=0.05, y=0.95),
        ],
        crop=NormalizedRect(x=0.25, y=0.25, width=0.5, height=0.5),
    )
    assert transformed.shape == (46, 90)


def test_pipeline_requires_consecutive_confirmations() -> None:
    region = RegionConfig(
        id="score",
        name="Score",
        rect=NormalizedRect(x=0, y=0, width=0.5, height=0.5),
        field_type="number",
        confirmation_frames=2,
    )
    engine = FakeEngine(
        [
            Recognition("12", 0.99),
            Recognition("junk", 0.99),
            Recognition("12", 0.99),
            Recognition("12", 0.99),
        ]
    )
    pipeline = RecognitionPipeline(engine, [region])

    first = pipeline.process(frame(1)).fields[0]
    rejected = pipeline.process(frame(2)).fields[0]
    restarted = pipeline.process(frame(3)).fields[0]
    accepted = pipeline.process(frame(4)).fields[0]

    assert first.state == ResultState.PENDING
    assert first.value == ""
    assert rejected.state == ResultState.REJECTED
    assert restarted.state == ResultState.PENDING
    assert accepted.state == ResultState.OK
    assert accepted.value == "12"


def test_unreadable_frame_does_not_restart_clock_confirmation() -> None:
    """A frame the engine cannot read says nothing about the clock. Below a
    minute such frames come every second; restarting on each would keep a
    correction pending indefinitely.
    """
    region = RegionConfig(
        id="clock",
        name="Clock",
        rect=NormalizedRect(x=0, y=0, width=0.5, height=0.5),
        field_type="time",
        confirmation_frames=2,
    )
    engine = FakeEngine(
        [
            Recognition("12.34", 0.99),
            Recognition("junk", 0.99),
            Recognition("1234", 0.99),
        ]
    )
    pipeline = RecognitionPipeline(engine, [region])

    first = pipeline.process(frame(1)).fields[0]
    rejected = pipeline.process(frame(2)).fields[0]
    accepted = pipeline.process(frame(3)).fields[0]

    assert first.state == ResultState.PENDING
    assert first.candidate_value == "12:34"
    assert rejected.state == ResultState.REJECTED
    assert accepted.state == ResultState.OK
    assert accepted.value == "12:34"


def test_time_field_rejects_impossible_seconds() -> None:
    region = RegionConfig(
        id="clock",
        name="Clock",
        rect=NormalizedRect(x=0, y=0, width=0.5, height=0.5),
        field_type="time",
        confirmation_frames=1,
    )
    pipeline = RecognitionPipeline(FakeEngine([Recognition("12:79", 0.99)]), [region])

    result = pipeline.process(frame()).fields[0]

    assert result.state == ResultState.REJECTED
    assert result.value == ""


def test_pipeline_uses_batch_engine_when_available() -> None:
    engine = FakeBatchEngine([Recognition("8", 0.99)])
    region = RegionConfig(
        id="score",
        name="Score",
        rect=NormalizedRect(x=0, y=0, width=0.2, height=0.2),
        confirmation_frames=1,
    )
    pipeline = RecognitionPipeline(engine, [region])
    assert pipeline.process(frame()).fields[0].value == "8"
    assert engine.batch_called


@pytest.mark.parametrize("smoothing_window", [1, 3])
def test_clock_switches_between_minutes_and_tenths(smoothing_window: int) -> None:
    region = RegionConfig(
        id="clock",
        name="Clock",
        rect=NormalizedRect(x=0, y=0, width=0.5, height=0.5),
        field_type="time",
        confirmation_frames=2,
        smoothing_window=smoothing_window,
    )
    values = ["01:00", "59.9", "09.9", "9.8", "0.0", "10:00"]
    pipeline = RecognitionPipeline(
        FakeEngine([Recognition(value, 0.99) for value in values for _ in range(4)]),
        [region],
    )

    # Half a second per frame: the jumps in this sequence are resets, and a
    # reset needs to stay on the board for a while before it is believed.
    sequence = 0
    for value in values:
        results = []
        for _ in range(4):
            packet = frame(sequence)
            packet.monotonic_ns = int(sequence * 500_000_000)
            sequence += 1
            results.append(pipeline.process(packet).fields[0])
        assert all(result.state != ResultState.REJECTED for result in results)
        assert results[-1].value == value


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("59.9", "59.9"),
        (" 5 9 . 8 ", "59.8"),
        ("09.9", "09.9"),
        ("9.8", "9.8"),
        ("00.0", "00.0"),
        ("0.0", "0.0"),
        ("59,9", "59.9"),
        ("12.34", "12:34"),
        ("12,34", "12:34"),
        ("1234", "12:34"),
    ],
)
def test_clock_preserves_tenths_and_normalizes_minutes(
    text: str, expected: str
) -> None:
    region = RegionConfig(
        name="Clock",
        rect=NormalizedRect(x=0, y=0, width=0.5, height=0.5),
        field_type="time",
        confirmation_frames=1,
    )
    pipeline = RecognitionPipeline(FakeEngine([Recognition(text, 0.99)]), [region])
    result = pipeline.process(frame()).fields[0]
    assert result.state == ResultState.OK
    assert result.value == expected


# "59:9" moved out of this list: a separator followed by a single digit is now
# read as tenths, because no clock shows "MM:S" and the engine reports the dot
# of "SS.t" as a colon often enough to lose the whole final minute otherwise.
# Values that are out of range for tenths, such as "60.0", stay rejected.
@pytest.mark.parametrize("text", ["60.0", "99.9", "123.4", "59.", ".9", "-1.0"])
def test_clock_rejects_invalid_tenths_without_replacing_accepted_value(
    text: str,
) -> None:
    region = RegionConfig(
        name="Clock",
        rect=NormalizedRect(x=0, y=0, width=0.5, height=0.5),
        field_type="time",
        confirmation_frames=1,
    )
    pipeline = RecognitionPipeline(
        FakeEngine([Recognition("01:00", 0.99), Recognition(text, 0.99)]), [region]
    )
    pipeline.process(frame())
    result = pipeline.process(frame()).fields[0]
    assert result.state == ResultState.REJECTED
    assert result.value == "01:00"


def test_tenths_are_recognised_through_a_misread_separator() -> None:
    """Below a minute the board shows "SS.t" and the engine misreads the dot.

    No clock displays "MM:S", so a separator followed by a single digit can
    only be tenths. Without this the whole final minute is rejected.
    """
    normalize = RecognitionPipeline._normalize_candidate
    assert normalize("58:7", "time") == "58.7"
    assert normalize("56:.5", "time") == "56.5"
    assert normalize("1:2", "time") == "1.2"
    # Already correct spellings stay untouched.
    assert normalize("58.7", "time") == "58.7"
    # Two digits after the separator remain minutes and seconds.
    assert normalize("12:34", "time") == "12:34"
    assert normalize("12:.34", "time") == "12:34"
    assert normalize("5:07", "time") == "5:07"


def test_blank_check_sees_both_polarities_and_thin_glyphs() -> None:
    """The measure has to span both ends of the histogram.

    Comparing against the median only detects bright-on-dark: a cell with dark
    digits on a light background has its median at the background and scores
    zero. Narrowing the percentiles instead misses a thin glyph that covers a
    few percent of the area - which is what a "1" in a period cell looks like.
    """
    config = PreprocessConfig()

    bright_on_dark = np.full((60, 160), 18, dtype=np.uint8)
    cv2.putText(bright_on_dark, "1:18", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 230, 3)
    assert not is_blank(bright_on_dark, config)

    dark_on_bright = np.full((60, 160), 220, dtype=np.uint8)
    cv2.putText(dark_on_bright, "1:18", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 20, 3)
    assert not is_blank(dark_on_bright, config)

    thin_glyph = np.full((60, 160), 18, dtype=np.uint8)
    cv2.putText(thin_glyph, "1", (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 230, 3)
    assert not is_blank(thin_glyph, config)

    rng = np.random.default_rng(7)
    noise = np.clip(rng.normal(18, 3, (60, 160)), 0, 255).astype(np.uint8)
    assert is_blank(noise, config)


def test_blank_region_is_never_handed_to_the_engine() -> None:
    """Tesseract answers an all-black image with a digit and a real confidence.

    That value is then rejected on format, and a rejection leaves the previous
    value in place - so a penalty slot that emptied would keep showing its
    last time indefinitely.
    """

    class Loud:
        def __init__(self) -> None:
            self.calls = 0

        def recognize(self, image, *, region_id: str) -> Recognition:
            self.calls += 1
            return Recognition("1", 0.52)

        def close(self) -> None:
            pass

    region = RegionConfig(
        id="pen",
        name="pen",
        rect=NormalizedRect(x=0, y=0, width=1.0, height=1.0),
        field_type="number",
        confirmation_frames=1,
    )
    engine = Loud()
    pipeline = RecognitionPipeline(engine, [region])

    rng = np.random.default_rng(7)
    noise = np.clip(rng.normal(18, 3, (100, 200)), 0, 255).astype(np.uint8)
    packet = FramePacket(
        sequence=1,
        image=noise,
        width=200,
        height=100,
        captured_at=datetime.now(UTC),
    )
    result = pipeline.process(packet).fields[0]

    assert engine.calls == 0, "a blank cell must not reach the engine"
    assert result.candidate_value == ""
    assert result.value == ""


def test_blank_region_does_not_become_noise() -> None:
    """An unlit scoreboard cell must not be thresholded into digits.

    Otsu assumes two brightness classes. Given only sensor noise it splits
    that into a high-frequency pattern, and the engine reads digits from it -
    which is how empty penalty slots produced values like "3:10".
    """
    rng = np.random.default_rng(7)
    noise = np.clip(rng.normal(18, 3, (60, 160)), 0, 255).astype(np.uint8)

    result = preprocess(noise, PreprocessConfig())
    assert result.max() == 0, "a blank cell must yield no foreground at all"

    # Disabling the check restores the previous behaviour.
    unguarded = preprocess(noise, PreprocessConfig(min_contrast=0.0))
    assert unguarded.max() == 255


def test_occupied_region_survives_the_blank_check() -> None:
    patch = np.full((60, 160), 18, dtype=np.uint8)
    cv2.putText(patch, "1:18", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 230, 3)

    result = preprocess(patch, PreprocessConfig())
    assert result.max() == 255, "digits must still come through"


def clock_results(values, *, confirmations=2, smoothing=1, interval=0.1):
    region = RegionConfig(
        id="clock",
        name="Clock",
        rect=NormalizedRect(x=0, y=0, width=1, height=1),
        field_type="time",
        confirmation_frames=confirmations,
        smoothing_window=smoothing,
    )
    pipeline = RecognitionPipeline(
        FakeEngine([Recognition(value, 0.99) for value in values]), [region]
    )
    results = []
    for index in range(len(values)):
        packet = frame(index)
        packet.monotonic_ns = int((1 + index * interval) * 1_000_000_000)
        results.append(pipeline.process(packet).fields[0])
    return results


@pytest.mark.parametrize("smoothing", [1, 5, 15])
def test_moving_tenths_confirm_without_repeated_text(smoothing):
    values = ["01:00", "59.9", "59.8", "59.7", "59.6", "59.5"]
    results = clock_results(values, smoothing=smoothing)
    assert [result.value for result in results] == [
        "",
        "59.9",
        "59.9",
        "59.7",
        "59.7",
        "59.5",
    ]
    assert [result.candidate_value for result in results] == values


def test_clock_does_not_invent_digits_at_rollover():
    values = ["10:00", "10:00", "09:59", "09:59"]
    results = clock_results(values, smoothing=5)
    assert [result.candidate_value for result in results] == values
    assert results[-1].value == "09:59"


@pytest.mark.parametrize("confirmations", [1, 2])
def test_clock_holds_through_bad_bursts_but_allows_confirmed_reset(confirmations):
    values = ["12:34", "12:34", "72:34", "72:34", "12:34", "12:33", "12:33"]
    values += ["20:00"] * 16
    results = clock_results(values, confirmations=confirmations)
    assert results[2].value == results[3].value == "12:34"
    assert results[6].value == "12:33"
    # A reset stays on the board for seconds; three frames are not enough to
    # tell it from misreads that happen to agree with each other.
    assert {result.value for result in results[7:10]} == {"12:33"}
    assert results[-2].value == "12:33"
    assert results[-1].value == "20:00"


def test_clock_tracker_needs_time_not_frames_for_a_jump() -> None:
    """Below a minute the board's colon blinks, so "21.2", "21.1", "21.0" can
    arrive as "212", "211", "210" - which spell a consistent clock at 2:12.
    """
    tracker = ClockTracker(2)
    assert tracker.add("21.5", 10.0, unchanged=False) is False
    assert tracker.add("21.4", 10.1, unchanged=False) is True
    misreads = [("2:12", 10.2), ("2:11", 10.3), ("2:10", 10.4)]
    assert [tracker.add(v, t, unchanged=False) for v, t in misreads] == [False] * 3
    # Back on the trajectory: the misread run is dropped after two tolerated
    # strays, then the usual confirmation applies.
    recovery = [("20.9", 10.5), ("20.8", 10.6), ("20.7", 10.7), ("20.6", 10.8)]
    assert [tracker.add(v, t, unchanged=False) for v, t in recovery] == [
        False,
        False,
        False,
        True,
    ]
    # A reset that stays on the board is accepted once it spans reset_seconds.
    reset = [tracker.add("20:00", 11.0 + i * 0.1, unchanged=False) for i in range(16)]
    assert reset == [False] * 15 + [True]


def test_clock_correction_survives_unreadable_and_stray_frames() -> None:
    """Below a minute every second brings frames the engine cannot read and
    one it misreads. If those restarted the confirmation of a correction, a
    wrong value, once accepted, would stand for the rest of the period.
    """
    tracker = ClockTracker(2)
    tracker.add("35.9", -0.1, unchanged=False)
    tracker.add("35.8", 0.0, unchanged=False)  # accepted; the board shows 26.8
    readings: list[tuple[str, float] | None] = [
        ("26.7", 0.1),
        ("26.6", 0.2),
        None,
        None,
        ("26.2", 0.5),
        ("26.8", 0.6),
        ("25.9", 0.7),
        ("25.8", 0.8),
        None,
        ("25.5", 1.0),
        ("25.4", 1.1),
        None,
        ("25.8", 1.3),
        ("25.1", 1.4),
        ("25.0", 1.5),
        ("24.9", 1.6),
    ]
    outcome = []
    for item in readings:
        if item is None:
            tracker.reject()
        else:
            outcome.append(tracker.add(item[0], item[1], unchanged=False))
    assert outcome == [False] * (len(outcome) - 1) + [True]


def test_single_repeat_of_accepted_text_does_not_reanchor_the_clock() -> None:
    """A stray frame can repeat the accepted text ("358" for a board at 35:3).
    Anchoring on it would turn the next correct reading into a jump that
    needs seconds to be believed; a real pause shows itself again anyway.
    """
    tracker = ClockTracker(2)
    assert tracker.add("35.9", 0.0, unchanged=False) is False
    assert tracker.add("35.8", 0.1, unchanged=False) is True
    assert tracker.add("35.7", 0.2, unchanged=False) is False
    tracker.reject()
    tracker.reject()
    assert tracker.add("35.8", 0.5, unchanged=True) is True
    assert tracker.accepted == (35.8, 0.1, 0.1)
    assert tracker.add("35.3", 0.6, unchanged=False) is True
    # A pause re-anchors once it has shown itself twice.
    assert tracker.add("35.3", 0.7, unchanged=True) is True
    assert tracker.accepted == (35.3, 0.6, 0.1)
    assert tracker.add("35.3", 0.8, unchanged=True) is True
    assert tracker.accepted == (35.3, 0.8, 0.1)


def test_repeat_of_accepted_text_never_outranks_an_established_trajectory() -> None:
    tracker = ClockTracker(4)
    for value, timestamp in [("36.0", 0.0), ("35.9", 0.1), ("35.8", 0.2), ("35.7", 0.3)]:
        confirmed = tracker.add(value, timestamp, unchanged=False)
    assert confirmed is True
    assert tracker.add("35.6", 0.4, unchanged=False) is False
    assert tracker.add("35.5", 0.5, unchanged=False) is False
    assert tracker.add("35.4", 0.6, unchanged=False) is False
    assert tracker.add("35.7", 0.7, unchanged=True) is True
    assert tracker.add("35.7", 0.8, unchanged=True) is True
    assert tracker.accepted == (35.7, 0.3, 0.1)
    assert tracker.add("35.2", 0.9, unchanged=False) is True


def test_dropped_colon_below_a_minute_reads_as_tenths() -> None:
    """This board shows "SS:t" below a minute and its colon blinks, so "21.0"
    arrives as "210" now and then - "2:10" to the rule for minutes.
    """
    values = ["21.5", "21.5", "21.4", "21.3", "212", "211", "210"]
    results = clock_results(values)
    assert [r.candidate_value for r in results[4:]] == ["21.2", "21.1", "21.0"]
    assert results[-1].value == "21.1"
    assert all(":" not in r.value for r in results)
    normalize = RecognitionPipeline._normalize_candidate
    # The same digits are minutes and seconds while the clock is above a minute.
    assert normalize("210", "time") == "2:10"
    assert normalize("210", "time", tenths=True) == "21.0"
    assert normalize("53", "time", tenths=True) == "5.3"


def test_expired_clock_keeps_one_spelling() -> None:
    """The expired board shows "0:0"; the engine reads it as "0:.0", "0:0" or
    "0:00" in turn. Those are one instant and must not flip the output.
    """
    values = ["0.2", "0.2", "0.1", "0:0", "0:00", "0:.0", "0:00"]
    results = clock_results(values)
    assert [r.candidate_value for r in results[3:]] == ["0.0"] * 4
    assert [r.value for r in results] == ["", "0.2", "0.2", "0.0", "0.0", "0.0", "0.0"]
    assert RecognitionPipeline._normalize_candidate("0:00", "time") == "0:00"


def test_clock_accounts_for_elapsed_capture_time_and_pauses():
    results = clock_results(["01:00", "01:00", "00:55", "00:50"], interval=5)
    assert results[-1].value == "00:50"
    paused = clock_results(["9.8"] * 30 + ["9.7", "9.6"])
    assert paused[-1].value == "9.6"


def test_bad_reads_do_not_contaminate_smoothing_history():
    region = RegionConfig(
        name="Score",
        rect=NormalizedRect(x=0, y=0, width=1, height=1),
        field_type="number",
        confirmation_frames=1,
        smoothing_window=5,
    )
    pipeline = RecognitionPipeline(
        FakeEngine(
            [
                Recognition("12", 0.99),
                Recognition("99", 0.1),
                Recognition("99", 0.1),
                Recognition("12", 0.99),
            ]
        ),
        [region],
    )
    results = [pipeline.process(frame(i)).fields[0] for i in range(4)]
    assert [result.value for result in results] == ["12"] * 4
    assert results[-1].candidate_value == "12"
