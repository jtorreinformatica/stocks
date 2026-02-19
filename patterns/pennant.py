"""
Pennant pattern detector.

A Pennant is a continuation pattern that forms after a strong price move (the flagpole).
It consists of a brief period of consolidation with converging trendlines,
resembling a small symmetrical triangle.
"""

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

from .base import PatternAnnotation, PatternDetector, PatternMatch


class PennantDetector(PatternDetector):

    @property
    def name(self) -> str:
        return "Banderín (Pennant)"

    @property
    def description(self) -> str:
        return (
            "Patrón de continuación a corto plazo que consiste en un movimiento "
            "fuerte (mástil) seguido de una pequeña consolidación simétrica."
        )

    def detect(self, df: pd.DataFrame) -> list[PatternMatch]:
        if len(df) < 25:
            return []

        matches = []
        highs = df["High"].values
        lows = df["Low"].values
        closes = df["Close"].values
        dates = df.index

        # Parameters
        FLAGPOLE_MAX_BARS = 10
        PENNANT_MAX_BARS = 20
        MIN_FLAGPOLE_MOVE = 0.05  # 5%

        for end_idx in range(len(df) - 1, 15, -1):
            # Look for a flagpole ending before the pennant
            # We'll search backwards
            for p_len in range(5, PENNANT_MAX_BARS + 1):
                pennant_start = end_idx - p_len
                if pennant_start < FLAGPOLE_MAX_BARS:
                    continue
                
                # Check for flagpole before pennant_start
                # The flagpole is a sharp move up or down
                found_flagpole = False
                flag_start = 0
                flag_move = 0
                
                for f_len in range(3, FLAGPOLE_MAX_BARS + 1):
                    f_start = pennant_start - f_len
                    move = (closes[pennant_start] - closes[f_start]) / closes[f_start]
                    
                    if abs(move) >= MIN_FLAGPOLE_MOVE:
                        found_flagpole = True
                        flag_start = f_start
                        flag_move = move
                        break
                
                if not found_flagpole:
                    continue

                # Now check the pennant consolidation (converging)
                p_highs = highs[pennant_start:end_idx + 1]
                p_lows = lows[pennant_start:end_idx + 1]
                
                # Symmetrical triangle check: highs decreasing, lows increasing
                # We'll use a simpler check: max of pennant is at the start, min is at the start
                # and range is contracting.
                
                first_half = p_len // 2
                if first_half < 2: continue
                
                max_1 = np.max(highs[pennant_start:pennant_start + first_half])
                max_2 = np.max(highs[pennant_start + first_half:end_idx + 1])
                
                min_1 = np.min(lows[pennant_start:pennant_start + first_half])
                min_2 = np.min(lows[pennant_start + first_half:end_idx + 1])
                
                # Highs should be roughly decreasing, lows roughly increasing
                if max_2 > max_1 * 1.01 or min_2 < min_1 * 0.99:
                    continue
                
                # Range must contract
                range_1 = max_1 - min_1
                range_2 = max_2 - min_2
                if range_2 > range_1 * 0.8:
                    continue

                # Confidence
                confidence = 0.5
                confidence += 0.2 * (abs(flag_move) / 0.1) # Stronger flagpole
                confidence += 0.2 * (1 - range_2 / range_1) # Better contraction
                confidence = min(0.95, max(0.4, confidence))

                # Annotations
                annotations = [
                    # Flagpole
                    PatternAnnotation(
                        type="line",
                        coords={
                            "x0": dates[flag_start], "y0": float(closes[flag_start]),
                            "x1": dates[pennant_start], "y1": float(closes[pennant_start]),
                        },
                        style={"color": "rgba(156, 39, 176, 0.8)", "width": 3},
                    ),
                    # Upper pennant line
                    PatternAnnotation(
                        type="line",
                        coords={
                            "x0": dates[pennant_start], "y0": float(max_1),
                            "x1": dates[end_idx], "y1": float(max_2),
                        },
                        style={"color": "rgba(255, 82, 82, 0.8)", "width": 2},
                    ),
                    # Lower pennant line
                    PatternAnnotation(
                        type="line",
                        coords={
                            "x0": dates[pennant_start], "y0": float(min_1),
                            "x1": dates[end_idx], "y1": float(min_2),
                        },
                        style={"color": "rgba(76, 175, 80, 0.8)", "width": 2},
                    ),
                ]

                matches.append(PatternMatch(
                    pattern_name=self.name,
                    start_date=dates[flag_start],
                    end_date=dates[end_idx],
                    confidence=round(confidence, 2),
                    description=f"Banderín detectado (Mástil {flag_move*100:.1f}%)",
                    annotations=annotations,
                ))
                break # Found one for this end_idx

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
                if max(m.start_date, k.start_date) < min(m.end_date, k.end_date):
                    overlaps = True
                    break
            if not overlaps:
                kept.append(m)
        return kept
