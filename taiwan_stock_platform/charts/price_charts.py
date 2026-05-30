from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_price_chart(
    frame: pd.DataFrame,
    *,
    ma_windows: Iterable[int] = (5, 20, 60),
    show_volume: bool = True,
    title: str = "Price",
) -> go.Figure:
    if frame.empty:
        return _empty_figure("No price data")

    rows = 2 if show_volume else 1
    row_heights = [0.72, 0.28] if show_volume else [1.0]
    specs = [[{"secondary_y": False}] for _ in range(rows)]

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=row_heights,
        specs=specs,
    )

    fig.add_trace(
        go.Candlestick(
            x=frame["date"],
            open=frame["open"],
            high=frame["high"],
            low=frame["low"],
            close=frame["close"],
            name="OHLC",
            increasing_line_color="#dc2626",
            decreasing_line_color="#16a34a",
        ),
        row=1,
        col=1,
    )

    for window in ma_windows:
        column = f"sma_{window}"
        if column in frame.columns:
            fig.add_trace(
                go.Scatter(
                    x=frame["date"],
                    y=frame[column],
                    mode="lines",
                    name=f"SMA {window}",
                    line={"width": 1.6},
                ),
                row=1,
                col=1,
            )

    if show_volume and "volume" in frame.columns:
        fig.add_trace(
            go.Bar(
                x=frame["date"],
                y=frame["volume"],
                name="Volume",
                marker_color="#64748b",
                opacity=0.55,
            ),
            row=2,
            col=1,
        )

    fig.update_layout(
        title=title,
        template="plotly_white",
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        margin={"l": 16, "r": 16, "t": 48, "b": 16},
        legend={"orientation": "h", "y": 1.02, "x": 0},
        height=620,
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    if show_volume:
        fig.update_yaxes(title_text="Volume", row=2, col=1)
    return fig


def _empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False)
    fig.update_layout(template="plotly_white", height=420)
    return fig

