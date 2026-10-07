from __future__ import annotations

from collections import Counter, deque


class ClockTracker:
    """Confirm whole readings along a plausible clock trajectory, including tenths.

    A running clock need not repeat the same text to confirm a new reading.
    Large corrections and resets need consistent observations spanning at
    least ``reset_seconds`` - and at least three frames - because a few
    consecutive misreads can be consistent with each other as well.

    A trajectory under confirmation survives frames the engine could not read
    and up to ``tolerated_misses`` stray readings in a row. Without that, a
    correction would restart its confirmation on every garbled frame and a
    wrong value, once accepted, could stand for the rest of the period.
    """

    def __init__(
        self,
        confirmation_frames: int,
        *,
        reset_seconds: float = 1.5,
        tolerated_misses: int = 2,
    ) -> None:
        self.confirmation_frames = confirmation_frames
        self.reset_seconds = reset_seconds
        self.tolerated_misses = tolerated_misses
        self.accepted: tuple[float, float, float] | None = None
        self.pending: tuple[float, float, float] | None = None
        self.pending_count = 0
        self.pending_since = 0.0
        self.misses = 0
        self.repeats = 0

    @property
    def tenths(self) -> bool:
        """Whether the last accepted reading showed tenths, i.e. the board is
        below a minute."""
        return self.accepted is not None and self.accepted[2] == 0.1

    @staticmethod
    def _reading(value: str, timestamp: float) -> tuple[float, float, float]:
        if ":" in value:
            minutes, seconds = value.split(":")
            return int(minutes) * 60 + int(seconds), timestamp, 1.0
        return float(value), timestamp, 0.1

    @staticmethod
    def _consistent(
        previous: tuple[float, float, float], current: tuple[float, float, float]
    ) -> bool:
        value, timestamp, resolution = current
        old_value, old_timestamp, old_resolution = previous
        elapsed = max(0.0, timestamp - old_timestamp)
        # Quantized displays can cross a second boundary between adjacent frames.
        return (
            abs(value - old_value) <= elapsed + max(resolution, old_resolution) + 0.02
        )

    def reset_pending(self) -> None:
        self.pending = None
        self.pending_count = 0
        self.pending_since = 0.0
        self.misses = 0

    def clear(self) -> None:
        self.accepted = None
        self.repeats = 0
        self.reset_pending()

    def reject(self) -> None:
        """Note a frame without a usable reading."""
        self._miss()

    def _miss(self) -> None:
        self.misses += 1
        if self.misses > self.tolerated_misses:
            self.reset_pending()

    def _established(self) -> bool:
        return self.pending_count >= 3

    def _start(self, current: tuple[float, float, float]) -> None:
        self.pending = current
        self.pending_count = 1
        self.pending_since = current[1]
        self.misses = 0

    def add(self, value: str, timestamp: float, *, unchanged: bool) -> bool:
        current = self._reading(value, timestamp)
        if unchanged:
            # A single frame repeating the accepted text is as likely a stray
            # ("358" for a board at 35:3) as the start of a pause. Anchoring
            # the clock on a stray turns the next correct reading into a jump
            # that then needs seconds to be believed - so a pause has to show
            # itself twice, and never outranks a trajectory already under way.
            self.repeats += 1
            if self._established():
                self._miss()
                if self.pending is not None:
                    return True
            elif self.repeats < 2:
                return True
            self.accepted = current
            self.reset_pending()
            return True
        self.repeats = 0
        plausible = self.accepted is None or self._consistent(self.accepted, current)
        if self.pending is None:
            self._start(current)
        elif self._consistent(self.pending, current):
            self.pending = current
            self.pending_count += 1
            self.misses = 0
        elif not self._established():
            # Until a trajectory has some standing the newest reading wins.
            self._start(current)
        else:
            # An established trajectory outranks a stray reading: this board
            # shows "x.8" and "x.7" as "x.0" and "x.1" for a frame every second.
            self._miss()
            if self.pending is None:
                self._start(current)
            return False
        if plausible:
            confirmed = self.pending_count >= self.confirmation_frames
        else:
            # A jump off the trajectory is a reset - or a run of misreads that
            # agree with each other: "21.2", "21.1", "21.0" read without their
            # colon arrive as "2:12", "2:11", "2:10", a perfectly consistent
            # clock. Frames cannot tell the two apart, time can: a reset stays
            # on the board for seconds, a misread pattern does not.
            confirmed = (
                self.pending_count >= max(3, self.confirmation_frames)
                and timestamp - self.pending_since + 1e-6 >= self.reset_seconds
            )
        if not confirmed:
            return False
        self.accepted = current
        self.reset_pending()
        return True


class CharacterSmoother:
    def __init__(self, max_history: int = 5) -> None:
        if max_history < 1:
            raise ValueError("max_history must be positive")
        self.history: deque[str] = deque(maxlen=max_history)

    def add(self, value: str) -> str:
        self.history.append(value)
        output: list[str] = []
        width = max((len(item) for item in self.history), default=0)
        for index in range(width):
            values = [item[index] for item in self.history if index < len(item)]
            if values:
                counts = Counter(values)
                # Prefer the newest value when frequencies tie.
                output.append(max(reversed(values), key=lambda item: counts[item]))
        return "".join(output)

    def clear(self) -> None:
        self.history.clear()
