"""
Inverse Head and Shoulders pattern detector.

A bullish reversal pattern consisting of:
- A left shoulder (low)
- A head (lower low)
- A right shoulder (higher low than head)
- A neckline connecting the intermediate highs
"""

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

from .base import PatternAnnotation, PatternDetector, PatternMatch


class InverseHeadAndShouldersDetector(PatternDetector):

    @property
    def name(self) -> str:
        return "Hombro Cabeza Hombro Invertido"

    @property
    def description(self) -> str:
        return (
            "Patrón de reversión alcista con tres mínimos sucesivos, "
            "siendo el central (cabeza) el más profundo."
        )

    def detect(self, df: pd.DataFrame) -> list[PatternMatch]:
        if len(df) < 40:
            return []

        matches = []
        lows = df["Low"].values
        highs = df["High"].values
        dates = df.index

        # Find local minima (shoulders and head)
        order = 5
        min_idx = argrelextrema(lows, np.less_equal, order=order)[0]
        max_idx = argrelextrema(highs, np.greater_equal, order=order)[0]

        if len(min_idx) < 3:
            return []

        # Iterate through triplets of local minima
        for i in range(len(min_idx) - 2):
            ls_idx = min_idx[i]
            h_idx = min_idx[i + 1]
            rs_idx = min_idx[i + 2]

            ls_val = lows[ls_idx]
            h_val = lows[h_idx]
            rs_val = lows[rs_idx]

            # Condition 1: Head must be deeper than both shoulders
            if h_val >= ls_val or h_val >= rs_val:
                continue

            # Condition 2: Shoulders should be roughly at the same level (within 10% of head depth)
            head_depth = max(ls_val, rs_val) - h_val
            if head_depth <= 0: continue
            
            shoulder_diff = abs(ls_val - rs_val)
            if shoulder_diff > head_depth * 0.6:
                continue

            # Find peaks (neckline points)
            # Peak 1 between LS and H
            between_ls_h = highs[ls_idx:h_idx + 1]
            if len(between_ls_h) == 0: continue
            p1_rel = np.argmax(between_ls_h)
            p1_idx = ls_idx + p1_rel
            p1_val = highs[p1_idx]

            # Peak 2 between H and RS
            between_h_rs = highs[h_idx:rs_idx + 1]
            if len(between_h_rs) == 0: continue
            p2_rel = np.argmax(between_h_rs)
            p2_idx = h_idx + p2_rel
            p2_val = highs[p2_idx]

            # Confidence
            confidence = 0.5
            # Symmetry check
            symmetry = 1 - abs((h_idx - ls_idx) - (rs_idx - h_idx)) / (rs_idx - ls_idx)
            confidence += 0.2 * max(0, symmetry)
            # Depth check
            confidence += 0.15 if head_depth / h_val > 0.05 else 0
            
            confidence = min(0.95, max(0.4, confidence))

            # Neckline extrapolation
            slope = (p2_val - p1_val) / (p2_idx - p1_idx) if p2_idx != p1_idx else 0
            intercept = p1_val - slope * p1_idx
            
            def neckline(x): return slope * x + intercept

            annotations = [
                # Neckline
                PatternAnnotation(
                    type="line",
                    coords={
                        "x0": dates[ls_idx], "y0": float(neckline(ls_idx)),
                        "x1": dates[rs_idx], "y1": float(neckline(rs_idx)),
                    },
                    style={"color": "rgba(156, 39, 176, 0.8)", "width": 2, "dash": "dash"},
                ),
                # Shoulders and Head markers
                PatternAnnotation(
                    type="marker",
                    coords={"x": dates[ls_idx], "y": float(ls_val)},
                    style={"symbol": "triangle-up", "color": "blue", "size": 10},
                ),
                PatternAnnotation(
                    type="marker",
                    coords={"x": dates[h_idx], "y": float(h_val)},
                    style={"symbol": "triangle-up", "color": "red", "size": 12},
                ),
                PatternAnnotation(
                    type="marker",
                    coords={"x": dates[rs_idx], "y": float(rs_val)},
                    style={"symbol": "triangle-up", "color": "blue", "size": 10},
                ),
            ]

            matches.append(PatternMatch(
                pattern_name=self.name,
                start_date=dates[ls_idx],
                end_date=dates[rs_idx],
                confidence=round(confidence, 2),
                description=f"HCH Invertido detectado (Cabeza en {dates[h_idx].strftime('%d/%m/%Y')})",
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
                if max(m.start_date, k.start_date) < min(m.end_date, k.end_date):
                    overlaps = True
                    break
            if not overlaps:
                kept.append(m)
        return kept
