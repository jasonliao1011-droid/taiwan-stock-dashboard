from __future__ import annotations

import streamlit as st

from taiwan_stock_platform.charts.price_charts import create_price_chart
from taiwan_stock_platform.config import AppConfig
from taiwan_stock_platform.indicators import enrich_technical_indicators
from taiwan_stock_platform.ui.pages.common import load_prices_cached, render_latest_metrics
from taiwan_stock_platform.ui.sidebar import SidebarState


def render_dashboard(state: SidebarState, config: AppConfig) -> None:
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
    render_latest_metrics(enriched)

    st.plotly_chart(
        create_price_chart(
            enriched,
            ma_windows=state.ma_windows,
            show_volume=state.show_volume,
            title=f"{state.symbol} Price",
        ),
        use_container_width=True,
    )

    st.dataframe(
        enriched.sort_values("date", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

