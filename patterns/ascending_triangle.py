"""
Ascending Triangle pattern detector.

An Ascending Triangle is a bullish continuation pattern characterized by:
- A flat upper resistance line (horizontal)
- A rising lower support line (higher lows)
- Price converging towards the apex
"""

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

from .base import PatternAnnotation, PatternDetector, PatternMatch


class AscendingTriangleDetector(PatternDetector):

    @property
    def name(self) -> str:
        return "Triángulo Ascendente"

    @property
    def description(self) -> str:
        return (
            "Patrón de continuación alcista con una zona de resistencia plana "
            "y mínimos crecientes que convergen hacia el ápice."
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

        # Sliding window approach
        for i in range(len(local_max_idx) - 2):
            for j in range(len(local_min_idx) - 2):
                max_indices = local_max_idx[i:i + 3]
                min_indices = local_min_idx[j:j + 3]

                start = min(max_indices[0], min_indices[0])
                end = max(max_indices[-1], min_indices[-1])

                if end - start < 15 or end - start > 120:
                    continue

                max_vals = highs[max_indices]
                min_vals = lows[min_indices]

                # Check for flat resistance (max values should be roughly equal)
                max_std_pct = np.std(max_vals) / np.mean(max_vals)
                if max_std_pct > 0.015:  # Tolerance for "flat"
                    continue
                
                # Check for rising support (min values should be increasing)
                if not np.all(np.diff(min_vals) > 0):
                    continue

                # Regression to find slopes
                min_slope, min_intercept = np.polyfit(min_indices.astype(float), min_vals, 1)
                max_slope, max_intercept = np.polyfit(max_indices.astype(float), max_vals, 1)

                # Slope of resistance should be near zero, support should be positive
                if abs(max_slope) > abs(min_slope) * 0.4: # Resistance must be much flatter than support
                    continue
                
                if min_slope <= 0:
                    continue

                # Calculate apex (where lines meet)
                # min_slope * x + min_intercept = max_slope * x + max_intercept
                # x * (min_slope - max_slope) = max_intercept - min_intercept
                if abs(min_slope - max_slope) > 1e-9:
                    apex_x = (max_intercept - min_intercept) / (min_slope - max_slope)
                    # Apex should be after the current 'end' but not too far
                    if apex_x < end or apex_x > end + (end - start):
                         pass # Not a strict requirement but good for "convergence"

                # Confidence
                confidence = 0.5
                confidence += 0.2 * (1 - max_std_pct / 0.015)
                # Check if price is near the apex (tighter range)
                current_range = highs[end] - lows[end]
                start_range = highs[start] - lows[start]
                if start_range > 0:
                    tightness = current_range / start_range
                    confidence += 0.2 * (1 - tightness)
                
                confidence = min(0.95, max(0.4, confidence))

                # Annotations
                avg_res = np.mean(max_vals)
                min_fit = np.poly1d([min_slope, min_intercept])

                annotations = [
                    # Resistance line
                    PatternAnnotation(
                        type="line",
                        coords={
                            "x0": dates[start], "y0": float(avg_res),
                            "x1": dates[end], "y1": float(avg_res),
                        },
                        style={"color": "rgba(255, 82, 82, 0.8)", "width": 2},
                    ),
                    # Support line
                    PatternAnnotation(
                        type="line",
                        coords={
                            "x0": dates[start], "y0": float(min_fit(start)),
                            "x1": dates[end], "y1": float(min_fit(end)),
                        },
                        style={"color": "rgba(76, 175, 80, 0.8)", "width": 2},
                    ),
                    # Triangle area
                    PatternAnnotation(
                        type="region",
                        coords={
                            "x0": dates[start], "x1": dates[end],
                            "y0": float(min_fit(start)), "y1": float(avg_res),
                        },
                        style={"color": "rgba(33, 150, 243, 0.05)"},
                    )
                ]

                matches.append(PatternMatch(
                    pattern_name=self.name,
                    start_date=dates[start],
                    end_date=dates[end],
                    confidence=round(confidence, 2),
                    description=f"Triángulo Ascendente detectado ({end - start} barras)",
                    annotations=annotations,
                ))

        return self._deduplicate(matches)

    @staticmethod
    def _deduplicate(matches: list[PatternMatch]) -> list[PatternMatch]:
        if not matches:
            return []
        matches.sort(key=lambda m: m.confidence, reverse=True)
        kept = []
        for m in matches:
            overlaps = False
            for k in kept:
                # Simple check: if start/end overlap significantly
                overlap_start = max(m.start_date, k.start_date)
                overlap_end = min(m.end_date, k.end_date)
                if overlap_start < overlap_end:
                    overlaps = True
                    break
            if not overlaps:
                kept.append(m)
        return kept
