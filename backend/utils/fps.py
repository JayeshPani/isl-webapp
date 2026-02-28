from __future__ import annotations

import time
from collections import deque
from typing import Deque


class FPSMeter:
    """Rolling FPS meter based on recent frame timestamps."""

    def __init__(self, window_size: int = 30) -> None:
        self.window_size = max(2, window_size)
        self.timestamps: Deque[float] = deque(maxlen=self.window_size)

    def tick(self) -> float:
        now = time.perf_counter()
        self.timestamps.append(now)
        if len(self.timestamps) < 2:
            return 0.0

        elapsed = self.timestamps[-1] - self.timestamps[0]
        if elapsed <= 0:
            return 0.0

        return (len(self.timestamps) - 1) / elapsed
