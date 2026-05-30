from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_rsi_macd_chart(frame: pd.DataFrame, *, rsi_window: int = 14) -> go.Figure:
    if frame.empty:
        return _empty_figure("No indicator data")

    rsi_column = f"rsi_{rsi_window}"
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.45, 0.55],
    )

    if rsi_column in frame.columns:
        fig.add_trace(
            go.Scatter(x=frame["date"], y=frame[rsi_column], name="RSI", mode="lines"),
            row=1,
            col=1,
        )
        fig.add_hline(y=70, line_dash="dash", line_color="#ef4444", row=1, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#22c55e", row=1, col=1)

    fig.add_trace(
        go.Bar(
            x=frame["date"],
            y=frame.get("macd_hist"),
            name="MACD Hist",
            marker_color="#94a3b8",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=frame["date"], y=frame.get("macd"), name="MACD", mode="lines"),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=frame["date"],
            y=frame.get("macd_signal"),
            name="Signal",
            mode="lines",
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title="Momentum",
        template="plotly_white",
        hovermode="x unified",
        margin={"l": 16, "r": 16, "t": 48, "b": 16},
        legend={"orientation": "h", "y": 1.02, "x": 0},
        height=560,
    )
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=1, col=1)
    fig.update_yaxes(title_text="MACD", row=2, col=1)
    return fig


def create_bollinger_chart(frame: pd.DataFrame) -> go.Figure:
    if frame.empty:
        return _empty_figure("No Bollinger data")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=frame["date"], y=frame["close"], name="Close", mode="lines")
    )
    for column, name in (
        ("bb_upper", "Upper"),
        ("bb_middle", "Middle"),
        ("bb_lower", "Lower"),
    ):
        if column in frame.columns:
            fig.add_trace(
                go.Scatter(x=frame["date"], y=frame[column], name=name, mode="lines")
            )

    fig.update_layout(
        title="Bollinger Bands",
        template="plotly_white",
        hovermode="x unified",
        margin={"l": 16, "r": 16, "t": 48, "b": 16},
        legend={"orientation": "h", "y": 1.02, "x": 0},
        height=460,
    )
    return fig


def _empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False)
    fig.update_layout(template="plotly_white", height=420)
    return fig

