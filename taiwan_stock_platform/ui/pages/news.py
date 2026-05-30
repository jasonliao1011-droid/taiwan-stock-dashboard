from __future__ import annotations

import streamlit as st

from taiwan_stock_platform.crawler.news_crawler import YahooTaiwanNewsCrawler
from taiwan_stock_platform.ui.sidebar import SidebarState


@st.cache_data(ttl=900, show_spinner=False)
def load_news_cached(symbol: str, limit: int = 10) -> list[dict[str, str]]:
    crawler = YahooTaiwanNewsCrawler()
    return [item.to_dict() for item in crawler.get_stock_news(symbol, limit=limit)]


def render_news(state: SidebarState) -> None:
    try:
        news_items = load_news_cached(state.symbol)
    except Exception as exc:
        st.error(f"News loading failed: {exc}")
        return

    if not news_items:
        st.info("No news found.")
        return

    for item in news_items:
        st.markdown(f"- [{item['title']}]({item['url']})")

