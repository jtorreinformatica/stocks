"""
Base classes for the pattern detection engine.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class PatternAnnotation:
    """
    A visual annotation to draw on the chart for a detected pattern.

    Supported types:
        - 'line': a trend line defined by (x0, y0) -> (x1, y1)
        - 'region': a shaded rectangular region
        - 'marker': a point marker on the chart
    """
    type: str  # 'line', 'region', 'marker'
    coords: dict[str, Any] = field(default_factory=dict)
    style: dict[str, Any] = field(default_factory=dict)


@dataclass
class PatternMatch:
    """
    Represents a single detected pattern instance.
    """
    pattern_name: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    confidence: float  # 0.0 to 1.0
    description: str = ""
    annotations: list[PatternAnnotation] = field(default_factory=list)

    @property
    def is_active_today(self) -> bool:
        """Check if this pattern ends on the most recent trading day."""
        today = pd.Timestamp.now().normalize()
        return self.end_date.normalize() >= today - pd.Timedelta(days=3)


class PatternDetector(ABC):
    """
    Abstract base class for all pattern detectors.

    To create a new pattern detector:
    1. Create a new file in the patterns/ directory.
    2. Subclass PatternDetector.
    3. Set the `name` attribute.
    4. Implement the `detect()` method.
    5. That's it! Auto-discovery handles registration.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the pattern."""
        ...

    @property
    def description(self) -> str:
        """Optional description of the pattern."""
        return ""

    @abstractmethod
    def detect(self, df: pd.DataFrame) -> list[PatternMatch]:
        """
        Detect pattern occurrences in the given OHLCV DataFrame.

        Args:
            df: DataFrame with columns Open, High, Low, Close, Volume
                and a DatetimeIndex.

        Returns:
            List of PatternMatch instances found in the data.
        """
        ...
