"""
Alert system — filters for patterns detected today and generates notifications.
"""

import pandas as pd

from patterns.base import PatternMatch


def filter_today_patterns(all_matches: dict[str, list[PatternMatch]]) -> dict[str, list[PatternMatch]]:
    """
    Filter pattern matches to only those active today (or very recent).

    Args:
        all_matches: Dict mapping symbol -> list of PatternMatch.

    Returns:
        Dict mapping symbol -> list of PatternMatch that are active today.
    """
    today_matches = {}
    for symbol, matches in all_matches.items():
        active = [m for m in matches if m.is_active_today]
        if active:
            today_matches[symbol] = active
    return today_matches


def format_alert_message(symbol: str, match: PatternMatch) -> str:
    """Format a human-readable alert message for a detected pattern."""
    return (
        f"🚨 **{match.pattern_name}** detectado en **{symbol}** | "
        f"Confianza: {match.confidence:.0%} | "
        f"{match.description}"
    )


def get_alert_summary(today_matches: dict[str, list[PatternMatch]]) -> list[str]:
    """
    Generate a list of alert messages for all today's patterns.

    Returns:
        List of formatted alert strings.
    """
    alerts = []
    for symbol, matches in today_matches.items():
        for match in matches:
            alerts.append(format_alert_message(symbol, match))
    return alerts
