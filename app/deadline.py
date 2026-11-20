"""Request-level deadline (D20). Every step asks how much time is left; nothing waits past it."""

import time
from collections.abc import Callable

Clock = Callable[[], float]


class Deadline:
    def __init__(self, budget_s: float, clock: Clock = time.monotonic) -> None:
        self._clock = clock
        self._start = clock()
        self._end = self._start + budget_s

    def remaining(self) -> float:
        return max(0.0, self._end - self._clock())

    def elapsed_ms(self) -> int:
        return int((self._clock() - self._start) * 1000)
