from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import streamlit as st

from taiwan_stock_platform.config import AppConfig
from taiwan_stock_platform.utils.date_utils import default_date_range


@dataclass(frozen=True)
class SidebarState:
    page: str
    symbol: str
    exchange: str
    data_source: str
    start_date: date
    end_date: date
    ma_windows: tuple[int, ...]
    show_volume: bool


def render_sidebar(config: AppConfig) -> SidebarState:
    start_default, end_default = default_date_range(config.default_start_days)

    with st.sidebar:
        page = st.radio("Page", ("Dashboard", "Technical", "News"))
        symbol = st.text_input("Stock ID", value=config.default_stock_id).strip()
        exchange = st.selectbox(
            "Exchange",
            options=("TW", "TWO"),
            index=0 if config.default_exchange == "TW" else 1,
            format_func=lambda value: "TWSE (.TW)" if value == "TW" else "TPEx (.TWO)",
        )
        data_source = st.selectbox(
            "Data Source",
            options=("auto", "yfinance", "finmind"),
            index=0,
        )
        date_range = st.date_input(
            "Date Range",
            value=(start_default, end_default),
            max_value=end_default,
        )
        ma_windows = st.multiselect(
            "Moving Averages",
            options=(5, 10, 20, 60, 120, 240),
            default=(5, 20, 60),
        )
        show_volume = st.checkbox("Show Volume", value=True)

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = start_default, end_default

    return SidebarState(
        page=page,
        symbol=symbol or config.default_stock_id,
        exchange=exchange,
        data_source=data_source,
        start_date=start_date,
        end_date=end_date,
        ma_windows=tuple(sorted(ma_windows)),
        show_volume=show_volume,
    )

