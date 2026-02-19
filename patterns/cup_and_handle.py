"""
Cup and Handle pattern detector.

A Cup and Handle is a bullish continuation pattern where:
- Price forms a rounded bottom (the 'cup') — a U-shape
- Followed by a small downward drift (the 'handle')
- The cup should have roughly equal rim levels
- Volume typically decreases during cup formation and increases at breakout
"""

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

from .base import PatternAnnotation, PatternDetector, PatternMatch


class CupAndHandleDetector(PatternDetector):

    @property
    def name(self) -> str:
        return "Cup and Handle"

    @property
    def description(self) -> str:
        return (
            "Patrón de continuación alcista con formación en U (copa) "
            "seguida de un pequeño retroceso (asa)."
        )

    def detect(self, df: pd.DataFrame) -> list[PatternMatch]:
        if len(df) < 40:
            return []

        matches = []
        highs = df["High"].values
        lows = df["Low"].values
        closes = df["Close"].values
        dates = df.index

        # Find prominent local maxima (potential rim levels)
        order = 8
        local_max_idx = argrelextrema(highs, np.greater_equal, order=order)[0]
        local_min_idx = argrelextrema(lows, np.less_equal, order=order)[0]

        if len(local_max_idx) < 2 or len(local_min_idx) < 1:
            return []

        # Try pairs of highs as the left rim and right rim of the cup
        for i in range(len(local_max_idx)):
            for j in range(i + 1, len(local_max_idx)):
                left_rim_idx = local_max_idx[i]
                right_rim_idx = local_max_idx[j]

                left_rim = highs[left_rim_idx]
                right_rim = highs[right_rim_idx]

                cup_width = right_rim_idx - left_rim_idx

                # Cup should be between 20 and 150 bars
                if cup_width < 20 or cup_width > 150:
                    continue

                # Rim levels should be roughly equal (within 8%)
                rim_diff = abs(left_rim - right_rim) / max(left_rim, right_rim)
                if rim_diff > 0.08:
                    continue

                # Find the lowest point between the two rims (cup bottom)
                cup_section = lows[left_rim_idx:right_rim_idx + 1]
                cup_bottom_rel = np.argmin(cup_section)
                cup_bottom_idx = left_rim_idx + cup_bottom_rel
                cup_bottom = lows[cup_bottom_idx]

                # Cup depth should be between 12% and 50%
                rim_avg = (left_rim + right_rim) / 2
                cup_depth = (rim_avg - cup_bottom) / rim_avg
                if cup_depth < 0.12 or cup_depth > 0.50:
                    continue

                # The cup bottom should be roughly in the middle (U-shape check)
                center_position = cup_bottom_rel / cup_width
                if center_position < 0.25 or center_position > 0.75:
                    continue

                # Check for U-shape (not V-shape): the bottom should be rounded
                # Compare left and right halves of the cup
                left_half = lows[left_rim_idx:cup_bottom_idx + 1]
                right_half = lows[cup_bottom_idx:right_rim_idx + 1]

                if len(left_half) > 3 and len(right_half) > 3:
                    # The descent and ascent should be gradual
                    left_std = np.std(np.diff(left_half))
                    right_std = np.std(np.diff(right_half))
                    smoothness = (left_std + right_std) / 2

                    # A V-shape would have very high std in the diffs
                    price_range = rim_avg - cup_bottom
                    if price_range > 0:
                        normalized_smoothness = smoothness / price_range
                        if normalized_smoothness > 0.5:
                            continue  # Too V-shaped

                # Look for the handle: a small pullback after the right rim
                handle_search_end = min(right_rim_idx + cup_width // 3, len(df) - 1)
                if handle_search_end <= right_rim_idx + 3:
                    # No room for a handle, but the cup itself is valid
                    handle_end_idx = right_rim_idx
                    handle_depth_pct = 0
                    has_handle = False
                else:
                    handle_section = lows[right_rim_idx:handle_search_end + 1]
                    handle_low_rel = np.argmin(handle_section)
                    handle_low_idx = right_rim_idx + handle_low_rel
                    handle_low = lows[handle_low_idx]

                    handle_depth_pct = (right_rim - handle_low) / right_rim

                    # Handle should retrace 5-15% of cup depth, max 38% of cup
                    has_handle = 0.02 < handle_depth_pct < cup_depth * 0.5
                    handle_end_idx = handle_search_end if has_handle else right_rim_idx

                # Confidence scoring
                confidence = 0.4
                confidence += 0.15 * (1 - rim_diff / 0.08)  # More equal rims = better
                confidence += 0.1 * (1 - abs(center_position - 0.5) / 0.25)  # More centered bottom
                confidence += 0.15 if has_handle else 0
                confidence += 0.1 if cup_depth > 0.15 else 0
                confidence = min(0.95, max(0.3, confidence))

                # Build annotations
                annotations = []

                # Cup curve — approximate with a set of line segments
                cup_points_idx = np.linspace(left_rim_idx, right_rim_idx, min(20, cup_width)).astype(int)
                for k in range(len(cup_points_idx) - 1):
                    idx0 = cup_points_idx[k]
                    idx1 = cup_points_idx[k + 1]
                    annotations.append(PatternAnnotation(
                        type="line",
                        coords={
                            "x0": dates[idx0], "y0": float(lows[idx0]),
                            "x1": dates[idx1], "y1": float(lows[idx1]),
                        },
                        style={"color": "rgba(33, 150, 243, 0.6)", "width": 2},
                    ))

                # Rim level line
                annotations.append(PatternAnnotation(
                    type="line",
                    coords={
                        "x0": dates[left_rim_idx], "y0": float(rim_avg),
                        "x1": dates[handle_end_idx], "y1": float(rim_avg),
                    },
                    style={"color": "rgba(76, 175, 80, 0.7)", "width": 2, "dash": "dash"},
                ))

                # Cup region
                annotations.append(PatternAnnotation(
                    type="region",
                    coords={
                        "x0": dates[left_rim_idx], "x1": dates[right_rim_idx],
                        "y0": float(cup_bottom), "y1": float(rim_avg),
                    },
                    style={"color": "rgba(33, 150, 243, 0.06)"},
                ))

                # Handle region
                if has_handle:
                    annotations.append(PatternAnnotation(
                        type="region",
                        coords={
                            "x0": dates[right_rim_idx], "x1": dates[handle_end_idx],
                            "y0": float(lows[handle_low_idx]), "y1": float(right_rim),
                        },
                        style={"color": "rgba(255, 152, 0, 0.1)"},
                    ))

                desc = f"Cup and Handle ({cup_width} barras, profundidad {cup_depth * 100:.1f}%)"
                if has_handle:
                    desc += f" con asa ({handle_depth_pct * 100:.1f}%)"

                matches.append(PatternMatch(
                    pattern_name=self.name,
                    start_date=dates[left_rim_idx],
                    end_date=dates[handle_end_idx],
                    confidence=round(confidence, 2),
                    description=desc,
                    annotations=annotations,
                ))

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
