from __future__ import annotations

import streamlit as st

from taiwan_stock_platform.charts.indicator_charts import (
    create_bollinger_chart,
    create_rsi_macd_chart,
)
from taiwan_stock_platform.config import AppConfig
from taiwan_stock_platform.indicators import enrich_technical_indicators
from taiwan_stock_platform.ui.pages.common import load_prices_cached
from taiwan_stock_platform.ui.sidebar import SidebarState


def render_technical(state: SidebarState, config: AppConfig) -> None:
    try:
        prices = load_prices_cached(
            state.symbol,
            state.exchange,
            state.data_source,
            state.start_date.isoformat(),
            state.end_date.isoformat(),
            config.finmind_token,
            config.finmind_base_url,
        )
    except Exception as exc:
        st.error(f"Data loading failed: {exc}")
        return

    enriched = enrich_technical_indicators(prices, ma_windows=state.ma_windows)
    st.plotly_chart(create_rsi_macd_chart(enriched), use_container_width=True)
    st.plotly_chart(create_bollinger_chart(enriched), use_container_width=True)

