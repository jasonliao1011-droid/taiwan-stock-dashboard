from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from taiwan_stock_platform.data.finmind_client import FinMindClient
from taiwan_stock_platform.data.stock_repository import StockRepository
from taiwan_stock_platform.data.yahoo_client import YahooFinanceClient
from taiwan_stock_platform.utils.formatters import format_number, format_price


@st.cache_data(ttl=900, show_spinner=False)
def load_prices_cached(
    symbol: str,
    exchange: str,
    source: str,
    start_date_iso: str,
    end_date_iso: str,
    finmind_token: str | None,
    finmind_base_url: str,
) -> pd.DataFrame:
    repository = StockRepository(
        yahoo=YahooFinanceClient(),
        finmind=FinMindClient(token=finmind_token, base_url=finmind_base_url),
    )
    return repository.get_daily_prices(
        symbol=symbol,
        start_date=date.fromisoformat(start_date_iso),
        end_date=date.fromisoformat(end_date_iso),
        source=source,
        exchange=exchange,
    )


def render_latest_metrics(frame: pd.DataFrame) -> None:
    if frame.empty:
        return

    latest = frame.iloc[-1]
    previous = frame.iloc[-2] if len(frame) >= 2 else latest
    close_delta = latest["close"] - previous["close"]
    close_delta_pct = (close_delta / previous["close"] * 100) if previous["close"] else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Close", format_price(latest["close"]), f"{close_delta_pct:.2f}%")
    col2.metric("High", format_price(latest["high"]))
    col3.metric("Low", format_price(latest["low"]))
    col4.metric("Volume", format_number(latest["volume"]))

