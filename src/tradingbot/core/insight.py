# insight.py
# Mirrors QuantConnect's Insight: an alpha's prediction about one symbol.
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import IntEnum


class Direction(IntEnum):
    DOWN = -1
    FLAT = 0
    UP = 1


@dataclass
class Insight:
    symbol: str
    direction: Direction
    generated_at: datetime
    period: timedelta
    confidence: float = 1.0
    magnitude: float | None = None
    source: str = ""

    @property
    def expires_at(self):
        return self.generated_at + self.period

    def is_active(self, now):
        return now < self.expires_at
