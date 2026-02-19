"""
Chart renderer — creates interactive candlestick charts with pattern overlays.
"""

import plotly.graph_objects as go
import pandas as pd

from patterns.base import PatternMatch


def render_chart(
    df: pd.DataFrame,
    symbol: str,
    matches: list[PatternMatch] | None = None,
) -> go.Figure:
    """
    Render an interactive candlestick chart with optional pattern annotations.

    Args:
        df: OHLCV DataFrame with DatetimeIndex.
        symbol: Ticker symbol for the chart title.
        matches: List of PatternMatch instances to overlay.

    Returns:
        A Plotly Figure object.
    """
    fig = go.Figure()

    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name=symbol,
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
    ))

    # Volume bars on secondary y-axis
    colors = [
        "#26a69a" if c >= o else "#ef5350"
        for o, c in zip(df["Open"], df["Close"])
    ]
    fig.add_trace(go.Bar(
        x=df.index,
        y=df["Volume"],
        marker_color=colors,
        opacity=0.3,
        name="Volumen",
        yaxis="y2",
    ))

    # Overlay pattern annotations
    if matches:
        _add_pattern_annotations(fig, matches)

    fig.update_layout(
        title=dict(
            text=f"📊 {symbol}",
            font=dict(size=20, color="#e0e0e0"),
        ),
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        xaxis=dict(
            rangeslider=dict(visible=False),
            gridcolor="rgba(255,255,255,0.05)",
        ),
        yaxis=dict(
            title="Precio",
            side="right",
            gridcolor="rgba(255,255,255,0.05)",
        ),
        yaxis2=dict(
            title="Volumen",
            overlaying="y",
            side="left",
            showgrid=False,
            range=[0, df["Volume"].max() * 4],  # Shrink volume bars
        ),
        height=550,
        margin=dict(l=60, r=60, t=60, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        hovermode="x unified",
    )

    return fig


# -- Pattern color palette for different patterns --
_PATTERN_COLORS = {
    "Falling Wedge": {"line": "#42a5f5", "fill": "rgba(66,165,245,0.08)", "label": "🔽"},
    "VCP": {"line": "#ab47bc", "fill": "rgba(171,71,188,0.08)", "label": "🔄"},
    "Cup and Handle": {"line": "#66bb6a", "fill": "rgba(102,187,106,0.08)", "label": "☕"},
}
_DEFAULT_COLOR = {"line": "#ffa726", "fill": "rgba(255,167,38,0.08)", "label": "📐"}


def _get_pattern_color(pattern_name: str) -> dict:
    return _PATTERN_COLORS.get(pattern_name, _DEFAULT_COLOR)


def _add_pattern_annotations(fig: go.Figure, matches: list[PatternMatch]):
    """Add visual overlays for detected patterns."""
    for match in matches:
        palette = _get_pattern_color(match.pattern_name)

        for ann in match.annotations:
            if ann.type == "line":
                color = ann.style.get("color", palette["line"])
                width = ann.style.get("width", 2)
                dash = ann.style.get("dash", "solid")

                fig.add_shape(
                    type="line",
                    x0=ann.coords["x0"], y0=ann.coords["y0"],
                    x1=ann.coords["x1"], y1=ann.coords["y1"],
                    line=dict(color=color, width=width, dash=dash),
                )

            elif ann.type == "region":
                color = ann.style.get("color", palette["fill"])

                fig.add_shape(
                    type="rect",
                    x0=ann.coords["x0"], x1=ann.coords["x1"],
                    y0=ann.coords["y0"], y1=ann.coords["y1"],
                    fillcolor=color,
                    line=dict(width=0),
                    layer="below",
                )

            elif ann.type == "marker":
                fig.add_trace(go.Scatter(
                    x=[ann.coords["x"]],
                    y=[ann.coords["y"]],
                    mode="markers",
                    marker=dict(
                        size=ann.style.get("size", 10),
                        color=ann.style.get("color", palette["line"]),
                        symbol=ann.style.get("symbol", "diamond"),
                    ),
                    name=match.pattern_name,
                    showlegend=False,
                ))

        # Add label annotation at the pattern start
        mid_date = match.start_date + (match.end_date - match.start_date) / 2
        fig.add_annotation(
            x=mid_date,
            y=1.05,
            yref="paper",
            text=f"{palette['label']} {match.pattern_name} ({match.confidence:.0%})",
            showarrow=False,
            font=dict(size=11, color=palette["line"]),
            bgcolor="rgba(0,0,0,0.6)",
            bordercolor=palette["line"],
            borderwidth=1,
            borderpad=4,
        )
