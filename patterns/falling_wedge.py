"""
Falling Wedge pattern detector.

A falling wedge is a bullish reversal pattern where:
- Both highs and lows make lower peaks/troughs (downtrend)
- The slope of the highs is steeper than the slope of the lows (converging)
- Volume typically decreases during the pattern
"""

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

from .base import PatternAnnotation, PatternDetector, PatternMatch


class FallingWedgeDetector(PatternDetector):

    @property
    def name(self) -> str:
        return "Falling Wedge"

    @property
    def description(self) -> str:
        return (
            "Patrón de reversión alcista donde máximos y mínimos decrecientes "
            "convergen formando una cuña descendente."
        )

    def detect(self, df: pd.DataFrame) -> list[PatternMatch]:
        if len(df) < 30:
            return []

        matches = []
        highs = df["High"].values
        lows = df["Low"].values
        dates = df.index

        # Find local maxima and minima
        order = 5
        local_max_idx = argrelextrema(highs, np.greater_equal, order=order)[0]
        local_min_idx = argrelextrema(lows, np.less_equal, order=order)[0]

        if len(local_max_idx) < 3 or len(local_min_idx) < 3:
            return []

        # Sliding window — look for wedge patterns in groups of pivots
        for i in range(len(local_max_idx) - 2):
            for j in range(len(local_min_idx) - 2):
                max_indices = local_max_idx[i:i + 3]
                min_indices = local_min_idx[j:j + 3]

                # The pivots should overlap in time
                start = min(max_indices[0], min_indices[0])
                end = max(max_indices[-1], min_indices[-1])

                if end - start < 15 or end - start > 120:
                    continue

                max_values = highs[max_indices]
                min_values = lows[min_indices]

                # Check: both sequences should be decreasing
                if not (np.all(np.diff(max_values) < 0) and np.all(np.diff(min_values) < 0)):
                    continue

                # Fit trend lines
                max_slope = np.polyfit(max_indices.astype(float), max_values, 1)[0]
                min_slope = np.polyfit(min_indices.astype(float), min_values, 1)[0]

                # Both slopes should be negative
                if max_slope >= 0 or min_slope >= 0:
                    continue

                # Upper line should be steeper (more negative) than lower line -> convergence
                if max_slope >= min_slope:
                    continue

                # Convergence ratio
                convergence = abs(max_slope) / abs(min_slope) if abs(min_slope) > 1e-9 else 0
                if convergence < 1.2:
                    continue

                # Confidence based on convergence and number of touches
                confidence = min(0.9, 0.5 + (convergence - 1.0) * 0.2)

                # Build trend lines for visualization
                max_fit = np.poly1d(np.polyfit(max_indices.astype(float), max_values, 1))
                min_fit = np.poly1d(np.polyfit(min_indices.astype(float), min_values, 1))

                annotations = [
                    PatternAnnotation(
                        type="line",
                        coords={
                            "x0": dates[start], "y0": float(max_fit(start)),
                            "x1": dates[end], "y1": float(max_fit(end)),
                        },
                        style={"color": "rgba(255, 82, 82, 0.8)", "width": 2, "dash": "dash"},
                    ),
                    PatternAnnotation(
                        type="line",
                        coords={
                            "x0": dates[start], "y0": float(min_fit(start)),
                            "x1": dates[end], "y1": float(min_fit(end)),
                        },
                        style={"color": "rgba(76, 175, 80, 0.8)", "width": 2, "dash": "dash"},
                    ),
                    PatternAnnotation(
                        type="region",
                        coords={
                            "x0": dates[start], "x1": dates[end],
                            "y0": float(min_fit(start)), "y1": float(max_fit(start)),
                        },
                        style={"color": "rgba(33, 150, 243, 0.08)"},
                    ),
                ]

                matches.append(PatternMatch(
                    pattern_name=self.name,
                    start_date=dates[start],
                    end_date=dates[end],
                    confidence=round(confidence, 2),
                    description=f"Falling Wedge detectado ({end - start} barras)",
                    annotations=annotations,
                ))

        # Deduplicate overlapping matches — keep the one with highest confidence
        matches = self._deduplicate(matches)
        return matches

    @staticmethod
    def _deduplicate(matches: list[PatternMatch]) -> list[PatternMatch]:
        if not matches:
            return []

        matches.sort(key=lambda m: m.confidence, reverse=True)
        kept = []
        for m in matches:
            overlaps = False
            for k in kept:
                overlap_start = max(m.start_date, k.start_date)
                overlap_end = min(m.end_date, k.end_date)
                if overlap_start < overlap_end:
                    overlaps = True
                    break
            if not overlaps:
                kept.append(m)
        return kept
