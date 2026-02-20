"""
VCP (Volatility Contraction Pattern) detector.

The VCP, popularized by Mark Minervini, is characterized by:
- A series of price contractions (bases)
- Each contraction has a smaller range than the previous one
- Volume decreases during each contraction
- Price stays above a key moving average (e.g. 200-day MA)
"""

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

from .base import PatternAnnotation, PatternDetector, PatternMatch


class VCPDetector(PatternDetector):

    @property
    def name(self) -> str:
        return "VCP"

    @property
    def description(self) -> str:
        return (
            "Volatility Contraction Pattern: contracciones sucesivas del rango "
            "de precios, cada una menor que la anterior, señal de acumulación."
        )

    def detect(self, df: pd.DataFrame) -> list[PatternMatch]:
        # Adaptive minimum length: at least 60 for daily, 24 for weekly/monthly
        is_daily = len(df) > 100 or (df.index[1] - df.index[0]).days == 1
        min_len = 60 if is_daily else 24
        
        if len(df) < min_len:
            return []

        matches = []
        highs = df["High"].values
        lows = df["Low"].values
        closes = df["Close"].values
        volumes = df["Volume"].values
        dates = df.index

        # Adaptive peak-finding order
        order = max(2, len(df) // 20)
        local_max_idx = argrelextrema(highs, np.greater_equal, order=order)[0]
        local_min_idx = argrelextrema(lows, np.less_equal, order=order)[0]

        if len(local_max_idx) < 2 or len(local_min_idx) < 2:
            return []

        # Look for the initial high (pivot high) followed by contractions
        for hi_start in range(len(local_max_idx) - 1):
            pivot_idx = local_max_idx[hi_start]
            pivot_high = highs[pivot_idx]

            # Find subsequent contractions after the pivot
            contractions = []
            search_start = pivot_idx

            for _ in range(5):  # Max 5 contractions
                # Find the next low after search_start
                future_lows = local_min_idx[local_min_idx > search_start]
                if len(future_lows) == 0:
                    break
                next_low_idx = future_lows[0]

                # Find the next high after the low
                future_highs = local_max_idx[local_max_idx > next_low_idx]
                if len(future_highs) == 0:
                    break
                next_high_idx = future_highs[0]

                contraction_range = highs[next_high_idx] - lows[next_low_idx]
                contraction_pct = contraction_range / pivot_high * 100

                contractions.append({
                    "low_idx": next_low_idx,
                    "high_idx": next_high_idx,
                    "low_val": lows[next_low_idx],
                    "high_val": highs[next_high_idx],
                    "range_pct": contraction_pct,
                })

                search_start = next_high_idx

            # Require at least 2 contractions (or just 1 if monthly)
            min_contractions = 2 if is_daily else 1
            if len(contractions) < min_contractions:
                continue

            # Check that each contraction is smaller than the previous
            ranges = [c["range_pct"] for c in contractions]
            if len(ranges) > 1 and not all(ranges[i] > ranges[i + 1] for i in range(len(ranges) - 1)):
                continue

            # Check if contraction highs stay close to the resistance line
            high_vals = [c["high_val"] for c in contractions]
            ceiling_check = all(abs(hv - pivot_high) / pivot_high < 0.08 for hv in high_vals)
            if not ceiling_check:
                continue

            # Last contraction should be tight
            if len(ranges) > 1:
                tightness_check = ranges[-1] <= ranges[0] * 0.8 or ranges[-1] < 10
            else:
                tightness_check = ranges[0] < 15
            
            if not tightness_check:
                continue

            start_idx = pivot_idx
            end_idx = contractions[-1]["high_idx"]

            # Check volume contraction
            if end_idx > start_idx + 5:
                first_half_vol = np.mean(volumes[start_idx:start_idx + (end_idx - start_idx) // 2])
                second_half_vol = np.mean(volumes[start_idx + (end_idx - start_idx) // 2:end_idx])
                vol_contracting = second_half_vol < first_half_vol
            else:
                vol_contracting = True

            # Confidence
            confidence = 0.4
            confidence += 0.15 * min(len(contractions), 4) / 4  # More contractions = better
            confidence += 0.15 if vol_contracting else 0
            if len(ranges) > 1:
                confidence += 0.1 * (1 - ranges[-1] / ranges[0])  # Tighter last contraction = better
            else:
                confidence += 0.05
            confidence = min(0.95, confidence)

            # Annotations: draw the contraction ranges
            annotations = []
            colors = [
                "rgba(255, 152, 0, 0.3)",
                "rgba(255, 193, 7, 0.25)",
                "rgba(255, 235, 59, 0.2)",
                "rgba(205, 220, 57, 0.15)",
                "rgba(139, 195, 74, 0.1)",
            ]

            # Pivot high line
            annotations.append(PatternAnnotation(
                type="line",
                coords={
                    "x0": dates[start_idx], "y0": float(pivot_high),
                    "x1": dates[end_idx], "y1": float(pivot_high),
                },
                style={"color": "rgba(156, 39, 176, 0.7)", "width": 2, "dash": "dot"},
            ))

            for ci, c in enumerate(contractions):
                color = colors[ci % len(colors)]
                annotations.append(PatternAnnotation(
                    type="region",
                    coords={
                        "x0": dates[c["low_idx"]], "x1": dates[c["high_idx"]],
                        "y0": float(c["low_val"]), "y1": float(c["high_val"]),
                    },
                    style={"color": color},
                ))

            matches.append(PatternMatch(
                pattern_name=self.name,
                start_date=dates[start_idx],
                end_date=dates[end_idx],
                confidence=round(confidence, 2),
                description=(
                    f"VCP con {len(contractions)} contracciones "
                    f"({ranges[0]:.1f}% → {ranges[-1]:.1f}%)"
                ),
                annotations=annotations,
            ))

        # Deduplicate
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
