from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from html import escape
from math import sqrt
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

from data_fetch import (
    build_ai_candidate_frame,
    calculate_ai_recommendation,
    fetch_monthly_revenue,
    generate_news_summary,
    get_portfolio_suggestion,
)
from taiwan_stock_platform.charts.indicator_charts import (
    create_bollinger_chart,
    create_rsi_macd_chart,
)
from taiwan_stock_platform.charts.price_charts import create_price_chart
from taiwan_stock_platform.config import AppConfig
from taiwan_stock_platform.crawler.news_crawler import YahooTaiwanNewsCrawler
from taiwan_stock_platform.data.finmind_client import FinMindClient
from taiwan_stock_platform.data.stock_repository import StockRepository
from taiwan_stock_platform.data.yahoo_client import YahooFinanceClient
from taiwan_stock_platform.indicators import enrich_technical_indicators
from taiwan_stock_platform.utils.date_utils import default_date_range
from taiwan_stock_platform.utils.formatters import format_number, format_price
from taiwan_stock_platform.utils.validators import extract_stock_id


@dataclass(frozen=True)
class StockProfile:
    symbol: str
    name: str
    exchange: str = "TW"
    aliases: tuple[str, ...] = ()

    @property
    def display_name(self) -> str:
        return f"{self.symbol} {self.name}"


@dataclass(frozen=True)
class SearchSelection:
    profile: StockProfile
    exchange: str
    data_source: str
    start_date: date
    end_date: date
    ma_windows: tuple[int, ...]
    show_volume: bool


COMMON_STOCKS: tuple[StockProfile, ...] = (
    StockProfile("2330", "台積電", aliases=("tsmc", "台積")),
    StockProfile("2317", "鴻海", aliases=("foxconn", "鴻海精密")),
    StockProfile("2454", "聯發科", aliases=("mediatek",)),
    StockProfile("2308", "台達電", aliases=("delta",)),
    StockProfile("2382", "廣達", aliases=("quanta",)),
    StockProfile("2303", "聯電", aliases=("umc",)),
    StockProfile("2412", "中華電", aliases=("中華電信",)),
    StockProfile("2881", "富邦金", aliases=("富邦",)),
    StockProfile("2882", "國泰金", aliases=("國泰",)),
    StockProfile("2891", "中信金", aliases=("中信",)),
    StockProfile("2886", "兆豐金", aliases=("兆豐",)),
    StockProfile("5871", "中租-KY", aliases=("中租",)),
    StockProfile("1301", "台塑"),
    StockProfile("1303", "南亞"),
    StockProfile("2002", "中鋼"),
    StockProfile("2207", "和泰車", aliases=("和泰",)),
    StockProfile("2357", "華碩", aliases=("asus",)),
    StockProfile("2395", "研華", aliases=("advantech",)),
    StockProfile("2603", "長榮", aliases=("長榮海運",)),
    StockProfile("2609", "陽明", aliases=("陽明海運",)),
    StockProfile("2615", "萬海"),
    StockProfile("3008", "大立光"),
    StockProfile("3045", "台灣大", aliases=("台灣大哥大",)),
    StockProfile("3711", "日月光投控", aliases=("日月光", "ase")),
    StockProfile("4904", "遠傳"),
    StockProfile("5880", "合庫金", aliases=("合庫",)),
    StockProfile("6505", "台塑化"),
    StockProfile("6669", "緯穎"),
    StockProfile("6770", "力積電"),
    StockProfile("8069", "元太", exchange="TWO", aliases=("e ink",)),
)


INVESTMENT_STYLE_ALLOCATIONS: dict[str, dict[str, int]] = {
    "保守型": {
        "高股息台股": 45,
        "金融股": 25,
        "債券 ETF": 20,
        "現金": 10,
    },
    "積極型": {
        "半導體成長股": 45,
        "AI 與伺服器供應鏈": 25,
        "中小型動能股": 20,
        "現金": 10,
    },
    "價值型": {
        "低估值大型股": 40,
        "金融與傳產龍頭": 30,
        "高股息台股": 20,
        "現金": 10,
    },
}


INVESTMENT_STYLE_RECOMMENDATIONS: dict[str, tuple[str, ...]] = {
    "保守型": ("2412", "3045", "2881", "2882", "2891"),
    "積極型": ("2330", "2454", "2382", "6669", "8069"),
    "價值型": ("2303", "1301", "2002", "5871", "6505"),
}


def main() -> None:
    config = AppConfig.from_env()
    configure_page()
    apply_dashboard_style()

    selection = render_sidebar(config)
    prices, load_error = load_market_data(selection, config)
    enriched = (
        enrich_technical_indicators(prices, ma_windows=selection.ma_windows)
        if not prices.empty
        else prices
    )

    render_header(selection, enriched)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["個股概況", "技術分析", "財務健檢", "即時新聞", "💡 AI 智投推薦"]
    )

    with tab1:
        render_overview_tab(selection, enriched, load_error)

    with tab2:
        render_technical_tab(selection, enriched, load_error)

    with tab3:
        render_financial_tab(selection, enriched, config, load_error)

    with tab4:
        render_news_tab(selection)

    with tab5:
        render_ai_tab(selection, enriched, config, load_error)


def configure_page() -> None:
    st.set_page_config(
        page_title="台股一站式數據整合分析平台",
        page_icon=":chart_with_upwards_trend:",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def apply_dashboard_style() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: #f6f8fb;
            color: #111827;
        }

        .block-container {
            max-width: 1440px;
            padding-top: 1.35rem;
            padding-bottom: 2rem;
        }

        [data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid #e5e7eb;
        }

        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 14px 16px;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }

        [data-testid="stMetricLabel"] {
            color: #64748b;
        }

        .dashboard-title {
            font-size: 1.65rem;
            font-weight: 700;
            letter-spacing: 0;
            margin-bottom: 0.15rem;
        }

        .dashboard-subtitle {
            color: #64748b;
            font-size: 0.92rem;
            margin-bottom: 1rem;
        }

        .status-pill {
            display: inline-block;
            border-radius: 999px;
            padding: 0.18rem 0.55rem;
            font-size: 0.78rem;
            font-weight: 650;
            border: 1px solid #d1d5db;
            background: #ffffff;
            color: #374151;
        }

        .status-pass {
            border-color: #bbf7d0;
            background: #f0fdf4;
            color: #166534;
        }

        .status-watch {
            border-color: #fde68a;
            background: #fffbeb;
            color: #92400e;
        }

        .status-risk {
            border-color: #fecaca;
            background: #fef2f2;
            color: #991b1b;
        }

        .news-link {
            border-bottom: 1px solid #e5e7eb;
            padding: 0.72rem 0;
        }

        .news-link a {
            color: #0f172a;
            font-weight: 650;
            text-decoration: none;
        }

        @media (max-width: 768px) {
            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }

            .dashboard-title {
                font-size: 1.25rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(config: AppConfig) -> SearchSelection:
    start_default, end_default = default_date_range(config.default_start_days)
    default_profile = find_profile_by_symbol(config.default_stock_id)

    with st.sidebar:
        st.header("股票搜尋")
        query = st.text_input(
            "輸入股票代號或名稱",
            value=default_profile.symbol,
            placeholder="例如 2330、台積電、TSMC",
        ).strip()

        candidates = find_stock_candidates(query)
        if not candidates:
            st.warning("找不到對應名稱，請輸入股票代號。")
            candidates = [default_profile]

        selected_profile = candidates[0]
        if len(candidates) > 1:
            selected_profile = st.selectbox(
                "搜尋結果",
                options=candidates,
                format_func=lambda profile: profile.display_name,
            )
        else:
            st.caption(selected_profile.display_name)

        exchange_index = 1 if selected_profile.exchange == "TWO" else 0
        exchange = st.selectbox(
            "市場",
            options=("TW", "TWO"),
            index=exchange_index,
            format_func=lambda value: "上市 TWSE" if value == "TW" else "上櫃 TPEx",
        )

        date_range = st.date_input(
            "資料區間",
            value=(start_default, end_default),
            max_value=end_default,
        )
        start_date, end_date = parse_date_range(date_range, start_default, end_default)

        data_source = st.selectbox(
            "資料來源",
            options=("auto", "yfinance", "finmind"),
            index=0,
            format_func={
                "auto": "自動",
                "yfinance": "Yahoo Finance",
                "finmind": "FinMind",
            }.get,
        )

        ma_windows = st.multiselect(
            "均線",
            options=(5, 10, 20, 60, 120, 240),
            default=(5, 20, 60),
        )
        show_volume = st.checkbox("顯示成交量", value=True)

    return SearchSelection(
        profile=selected_profile,
        exchange=exchange,
        data_source=data_source,
        start_date=start_date,
        end_date=end_date,
        ma_windows=tuple(sorted(ma_windows)),
        show_volume=show_volume,
    )


def render_header(selection: SearchSelection, frame: pd.DataFrame) -> None:
    latest_date = "-"
    source = selection.data_source
    if not frame.empty:
        latest_date = str(frame.iloc[-1]["date"])
        source = str(frame.iloc[-1].get("source", selection.data_source))

    st.markdown(
        f"""
        <div class="dashboard-title">{selection.profile.display_name}</div>
        <div class="dashboard-subtitle">
            {selection.exchange} / {source} / latest {latest_date}
        </div>
        """,
        unsafe_allow_html=True,
    )


def load_market_data(
    selection: SearchSelection,
    config: AppConfig,
) -> tuple[pd.DataFrame, str | None]:
    try:
        with st.spinner("載入市場資料中..."):
            prices = load_prices_cached(
                selection.profile.symbol,
                selection.exchange,
                selection.data_source,
                selection.start_date.isoformat(),
                selection.end_date.isoformat(),
                config.finmind_token,
                config.finmind_base_url,
            )
        return prices, None
    except Exception as exc:
        return pd.DataFrame(), str(exc)


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


def render_overview_tab(
    selection: SearchSelection,
    frame: pd.DataFrame,
    load_error: str | None,
) -> None:
    if load_error:
        st.error(f"資料載入失敗：{load_error}")
        return
    if frame.empty:
        st.info("目前沒有可顯示的市場資料。")
        return

    render_market_metrics(frame)
    st.plotly_chart(
        create_price_chart(
            frame,
            ma_windows=selection.ma_windows,
            show_volume=selection.show_volume,
            title=f"{selection.profile.display_name} 走勢",
        ),
        use_container_width=True,
    )

    st.dataframe(
        frame.sort_values("date", ascending=False).head(120),
        use_container_width=True,
        hide_index=True,
    )


def render_technical_tab(
    selection: SearchSelection,
    frame: pd.DataFrame,
    load_error: str | None,
) -> None:
    if load_error:
        st.error(f"資料載入失敗：{load_error}")
        return
    if frame.empty:
        st.info("目前沒有可顯示的技術資料。")
        return

    render_technical_snapshot(frame)
    st.plotly_chart(
        create_price_chart(
            frame,
            ma_windows=selection.ma_windows,
            show_volume=selection.show_volume,
            title="K 線與均線",
        ),
        use_container_width=True,
    )
    left, right = st.columns((1, 1), gap="large")
    with left:
        st.plotly_chart(create_rsi_macd_chart(frame), use_container_width=True)
    with right:
        st.plotly_chart(create_bollinger_chart(frame), use_container_width=True)


def render_financial_tab(
    selection: SearchSelection,
    frame: pd.DataFrame,
    config: AppConfig,
    load_error: str | None,
) -> None:
    financial_frame = load_financial_statements_cached(
        selection.profile.symbol,
        config.finmind_token,
        config.finmind_base_url,
    )

    if not financial_frame.empty:
        render_financial_statement_health(financial_frame)
        return

    if load_error:
        st.error(f"資料載入失敗：{load_error}")
        return
    if frame.empty:
        st.info("目前沒有可顯示的健檢資料。")
        return

    render_price_based_health(frame)


@st.cache_data(ttl=3600, show_spinner=False)
def load_financial_statements_cached(
    symbol: str,
    finmind_token: str | None,
    finmind_base_url: str,
) -> pd.DataFrame:
    stock_id = extract_stock_id(symbol)
    today = date.today()
    start_date = date(today.year - 5, 1, 1)
    params: dict[str, str] = {
        "dataset": "TaiwanStockFinancialStatements",
        "data_id": stock_id,
        "start_date": start_date.isoformat(),
        "end_date": today.isoformat(),
    }
    if finmind_token:
        params["token"] = finmind_token

    try:
        response = requests.get(finmind_base_url, params=params, timeout=20)
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return pd.DataFrame()

    data = payload.get("data") or []
    if not data:
        return pd.DataFrame()

    return pd.DataFrame(data)


def render_news_tab(selection: SearchSelection) -> None:
    try:
        news_items = load_news_cached(selection.profile.symbol)
    except Exception as exc:
        st.error(f"新聞載入失敗：{exc}")
        return

    if not news_items:
        st.info("目前沒有新聞資料。")
        return

    st.subheader("🤖 AI 市場輿情綜合評論")
    summary = generate_news_summary(news_items)
    st.info(f"✨ {summary}")
    if hasattr(st, "hr"):
        st.hr()
    else:
        st.divider()

    for item in news_items:
        title = escape(item["title"])
        url = escape(item["url"], quote=True)
        source = escape(item.get("source", "Yahoo Taiwan"))
        st.markdown(
            f"""
            <div class="news-link">
                <a href="{url}" target="_blank">{title}</a>
                <div class="dashboard-subtitle">{source}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_ai_tab(
    selection: SearchSelection,
    frame: pd.DataFrame,
    config: AppConfig,
    load_error: str | None,
) -> None:
    revenue_frame = pd.DataFrame()
    candidate_frame = pd.DataFrame()
    candidate: pd.Series | None = None
    signal = "觀望保守"
    ai_score = 0
    reasons: list[str] = []

    if load_error:
        reasons.append(f"市場資料載入失敗：{load_error}")
    elif frame.empty:
        reasons.append("目前沒有可用的股價資料，無法進行 AI 實時診斷。")
    else:
        revenue_frame = load_monthly_revenue_cached(
            selection.profile.symbol,
            config.finmind_token,
            config.finmind_base_url,
        )
        candidate_frame = build_ai_candidate_frame(
            symbol=selection.profile.symbol,
            name=selection.profile.name,
            price_frame=frame,
            revenue_frame=revenue_frame,
        )

        if candidate_frame.empty:
            reasons.append("目前資料不足，無法計算 5MA、20MA 或營收 YoY。")
        else:
            candidate = candidate_frame.iloc[0]
            signal, ai_score, reasons = calculate_ai_recommendation(
                candidate_frame,
                revenue_frame,
            )

    left, right = st.columns(2, gap="large")

    with left:
        st.subheader("AI 實時診斷")
        st.caption(f"{selection.profile.display_name} 技術面與基本面綜合評估")

        score_text = f"{signal}｜AI 綜合分數：{ai_score}/100"
        if signal == "強力推薦":
            st.success(score_text)
        elif signal == "中性持有":
            st.warning(score_text)
        else:
            st.error(score_text)

        reason_items = reasons or ["暫無可用評估明細。"]
        st.markdown("**評估依據**\n" + "\n".join(f"- {reason}" for reason in reason_items))

        ma5 = candidate.get("ma5") if candidate is not None else None
        ma20 = candidate.get("ma20") if candidate is not None else None
        revenue_yoy = candidate.get("revenue_yoy") if candidate is not None else None

        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("最新 5MA", format_metric_integer(ma5))
        metric_col2.metric("最新 20MA", format_metric_integer(ma20))
        metric_col3.metric("最新營收 YoY%", format_metric_percent_integer(revenue_yoy))

    with right:
        st.subheader("客製化資產配置建議")
        investor_style = st.selectbox(
            "選擇投資風格",
            options=(
                "保守型 (領息小資/退休族)",
                "積極型 (追求資本利得)",
                "價值型 (巴菲特價值投資)",
            ),
            index=0,
        )
        labels, values, strategy_text = get_portfolio_suggestion(
            investor_style,
            score=ai_score,
        )
        st.info(strategy_text)

        recommendation_map = {
            "保守型 (領息小資/退休族)": {
                "高股息 ETF": "推薦：0056, 00878, 00713",
                "大型權值股": "推薦：2330, 2317, 2454",
                "穩定金融股": "推薦：2881, 2882, 2891",
            },
            "積極型 (追求資本利得)": {
                "半導體上游": "推薦：2330, 2454",
                "AI 伺服器供應鏈": "推薦：2382, 6669, 2356",
                "高股性設備股": "推薦：3008, 6415, 6196",
            },
            "價值型 (巴菲特價值投資)": {
                "低本益比價值股": "推薦：2303, 2002, 2603",
                "傳產龍頭": "推薦：1301, 1303, 6505",
                "高淨值比潛力股": "推薦：5871, 2886, 5880",
            },
        }
        recommend_stocks = [
            recommendation_map.get(investor_style, {}).get(label, "推薦：請依最新研究名單評估")
            for label in labels
        ]
        allocation_frame = pd.DataFrame(
            {
                "資產類別": labels,
                "配置比例": values,
            }
        )
        fig_pie = px.pie(
            allocation_frame,
            names="資產類別",
            values="配置比例",
            hole=0.4,
            color_discrete_sequence=px.colors.sequential.RdBu,
            custom_data=[recommend_stocks],
        )
        fig_pie.update_traces(
            textinfo="percent+label",
            hovertemplate=(
                "<b>%{label}</b><br>"
                "配置比例: %{percent}<br>"
                "<b>%{customdata[0]}</b><extra></extra>"
            ),
            marker={"line": {"color": "#ffffff", "width": 2}},
        )
        fig_pie.update_layout(
            title="資產配置甜甜圈圖",
            template="plotly_white",
            height=380,
            margin={"l": 8, "r": 8, "t": 48, "b": 8},
            showlegend=True,
            legend={"orientation": "h", "y": -0.08, "x": 0},
        )
        st.plotly_chart(fig_pie, use_container_width=True)


@st.cache_data(ttl=3600, show_spinner=False)
def load_monthly_revenue_cached(
    symbol: str,
    finmind_token: str | None,
    finmind_base_url: str,
) -> pd.DataFrame:
    return fetch_monthly_revenue(
        symbol,
        finmind_token=finmind_token,
        finmind_base_url=finmind_base_url,
    )


def create_asset_allocation_chart(investment_style: str) -> go.Figure:
    allocation = INVESTMENT_STYLE_ALLOCATIONS[investment_style]
    fig = go.Figure(
        data=[
            go.Pie(
                labels=list(allocation.keys()),
                values=list(allocation.values()),
                hole=0.42,
                textinfo="label+percent",
                marker={
                    "colors": ["#2563eb", "#16a34a", "#f59e0b", "#64748b"],
                    "line": {"color": "#ffffff", "width": 2},
                },
            )
        ]
    )
    fig.update_layout(
        title=f"{investment_style}資產配置",
        template="plotly_white",
        height=360,
        margin={"l": 8, "r": 8, "t": 48, "b": 8},
        showlegend=True,
        legend={"orientation": "h", "y": -0.05, "x": 0},
    )
    return fig


def render_style_recommendations(investment_style: str) -> None:
    recommended_symbols = INVESTMENT_STYLE_RECOMMENDATIONS[investment_style]
    recommended = [
        find_profile_by_symbol(symbol).display_name for symbol in recommended_symbols
    ]
    st.markdown("**推薦台股代號**")
    st.dataframe(
        pd.DataFrame(
            {
                "股票代號": recommended_symbols,
                "名稱": [item.split(" ", maxsplit=1)[1] for item in recommended],
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


def format_ai_candidate_frame(
    candidate_frame: pd.DataFrame,
    high_score_frame: pd.DataFrame,
) -> pd.DataFrame:
    result = candidate_frame.copy()
    high_score_ids = set(high_score_frame.get("stock_id", pd.Series(dtype=str)))
    result["AI 標籤"] = result["stock_id"].apply(
        lambda stock_id: "AI 綜合評分高分股"
        if stock_id in high_score_ids
        else "觀察中"
    )
    result = result.rename(
        columns={
            "stock_id": "股票代號",
            "stock_name": "股票名稱",
            "close": "收盤價",
            "ma5": "5MA",
            "ma20": "20MA",
            "revenue_yoy": "最新一個月營收 YoY",
            "recommendation_index": "推薦指數",
        }
    )
    for column in ("收盤價", "5MA", "20MA"):
        result[column] = result[column].apply(format_nullable_number)
    result["最新一個月營收 YoY"] = result["最新一個月營收 YoY"].apply(format_percent)
    result["推薦指數"] = result["推薦指數"].apply(lambda value: f"{int(value)}/100")
    return result[
        [
            "股票代號",
            "股票名稱",
            "5MA",
            "20MA",
            "最新一個月營收 YoY",
            "推薦指數",
            "AI 標籤",
        ]
    ]


@st.cache_data(ttl=900, show_spinner=False)
def load_news_cached(symbol: str, limit: int = 12) -> list[dict[str, str]]:
    crawler = YahooTaiwanNewsCrawler()
    return [item.to_dict() for item in crawler.get_stock_news(symbol, limit=limit)]


def render_market_metrics(frame: pd.DataFrame) -> None:
    latest = frame.iloc[-1]
    previous = frame.iloc[-2] if len(frame) >= 2 else latest
    delta = float(latest["close"]) - float(previous["close"])
    delta_pct = safe_pct(delta, float(previous["close"]))

    period_high = float(frame["high"].max())
    period_low = float(frame["low"].min())
    avg_volume = float(frame["volume"].tail(20).mean())

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("收盤價", format_price(latest["close"]), format_percent(delta_pct))
    col2.metric("區間高點", format_price(period_high))
    col3.metric("區間低點", format_price(period_low))
    col4.metric("成交量", format_number(latest["volume"]))
    col5.metric("20 日均量", format_number(avg_volume))


def render_technical_snapshot(frame: pd.DataFrame) -> None:
    latest = frame.iloc[-1]
    rsi = latest.get("rsi_14")
    macd = latest.get("macd")
    signal = latest.get("macd_signal")
    sma_20 = latest.get("sma_20")
    sma_60 = latest.get("sma_60")
    close = latest.get("close")

    trend = "多頭" if is_number(close) and is_number(sma_60) and close >= sma_60 else "觀察"
    macd_state = "偏多" if is_number(macd) and is_number(signal) and macd >= signal else "偏弱"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("RSI 14", format_nullable_number(rsi))
    col2.metric("MACD", format_nullable_number(macd))
    col3.metric("SMA 20", format_nullable_number(sma_20))
    col4.metric("趨勢", trend, macd_state)


def render_price_based_health(frame: pd.DataFrame) -> None:
    health = calculate_price_health(frame)
    st.subheader("市場健檢")
    st.progress(health["score"] / 100, text=f"健檢分數 {health['score']}/100")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("年化波動", format_percent(health["volatility"]))
    col2.metric("最大回撤", format_percent(health["max_drawdown"]))
    col3.metric("20 日均量", format_number(health["avg_volume_20"]))
    col4.metric("60 日趨勢", health["trend"])

    st.divider()
    for check in health["checks"]:
        pill_class = {
            "通過": "status-pass",
            "觀察": "status-watch",
            "風險": "status-risk",
        }[check["status"]]
        st.markdown(
            f"""
            <span class="status-pill {pill_class}">{check["status"]}</span>
            <strong>{check["label"]}</strong>
            <div class="dashboard-subtitle">{check["detail"]}</div>
            """,
            unsafe_allow_html=True,
        )


def render_financial_statement_health(frame: pd.DataFrame) -> None:
    pivot = pivot_financial_statements(frame)
    if pivot.empty:
        st.info("目前沒有可顯示的財務資料。")
        return

    latest = pivot.iloc[-1]
    previous_year = pivot.iloc[-5] if len(pivot) >= 5 else None
    revenue_col = find_metric_column(pivot, ("revenue", "營業收入", "營收"))
    net_income_col = find_metric_column(pivot, ("netincome", "net income", "本期淨利", "稅後淨利"))
    eps_col = find_metric_column(pivot, ("eps", "每股盈餘"))
    assets_col = find_metric_column(pivot, ("totalassets", "total assets", "資產總計"))
    liabilities_col = find_metric_column(pivot, ("totalliabilities", "total liabilities", "負債總計"))

    revenue = get_metric_value(latest, revenue_col)
    net_income = get_metric_value(latest, net_income_col)
    eps = get_metric_value(latest, eps_col)
    assets = get_metric_value(latest, assets_col)
    liabilities = get_metric_value(latest, liabilities_col)
    revenue_yoy = None
    if previous_year is not None:
        previous_revenue = get_metric_value(previous_year, revenue_col)
        if revenue is not None and previous_revenue is not None:
            revenue_yoy = safe_pct(revenue - previous_revenue, previous_revenue)
    net_margin = safe_pct(net_income, revenue)
    debt_ratio = safe_pct(liabilities, assets)

    st.subheader("財務摘要")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("營收", format_large_number(revenue), format_percent(revenue_yoy))
    col2.metric("淨利率", format_percent(net_margin))
    col3.metric("EPS", format_nullable_number(eps))
    col4.metric("負債比", format_percent(debt_ratio))

    figure = create_financial_bar_chart(pivot, revenue_col, net_income_col)
    if figure is not None:
        st.plotly_chart(figure, use_container_width=True)
    else:
        st.dataframe(pivot.tail(12), use_container_width=True)


def calculate_price_health(frame: pd.DataFrame) -> dict[str, Any]:
    data = frame.copy()
    data["return"] = data["close"].pct_change()
    volatility = float(data["return"].std() * sqrt(252)) if len(data) > 2 else 0.0
    drawdown = data["close"] / data["close"].cummax() - 1
    max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0
    latest = data.iloc[-1]
    avg_volume_20 = float(data["volume"].tail(20).mean())

    checks = [
        build_health_check(
            "中期趨勢",
            bool(latest.get("close", 0) >= latest.get("sma_60", latest.get("close", 0))),
            "收盤價高於 60 日均線" if latest.get("close", 0) >= latest.get("sma_60", 0) else "收盤價低於 60 日均線",
        ),
        build_health_check(
            "成交量能",
            bool(latest.get("volume", 0) >= avg_volume_20 * 0.8),
            "成交量接近或高於 20 日均量" if latest.get("volume", 0) >= avg_volume_20 * 0.8 else "成交量低於近期均量",
        ),
        build_health_check(
            "波動風險",
            volatility <= 0.45,
            f"年化波動率 {format_percent(volatility)}",
        ),
        build_health_check(
            "回撤控制",
            max_drawdown >= -0.3,
            f"區間最大回撤 {format_percent(max_drawdown)}",
        ),
    ]
    passed = sum(1 for check in checks if check["status"] == "通過")
    watched = sum(1 for check in checks if check["status"] == "觀察")
    score = min(100, int((passed * 25) + (watched * 10)))
    trend = "多頭" if latest.get("close", 0) >= latest.get("sma_60", latest.get("close", 0)) else "整理"

    return {
        "score": score,
        "volatility": volatility,
        "max_drawdown": max_drawdown,
        "avg_volume_20": avg_volume_20,
        "trend": trend,
        "checks": checks,
    }


def build_health_check(label: str, passed: bool, detail: str) -> dict[str, str]:
    return {
        "label": label,
        "status": "通過" if passed else "觀察",
        "detail": detail,
    }


def create_financial_bar_chart(
    pivot: pd.DataFrame,
    revenue_col: str | None,
    net_income_col: str | None,
) -> go.Figure | None:
    if revenue_col is None and net_income_col is None:
        return None

    recent = pivot.tail(12)
    fig = go.Figure()
    if revenue_col:
        fig.add_trace(go.Bar(x=recent.index, y=recent[revenue_col], name="營收"))
    if net_income_col:
        fig.add_trace(go.Bar(x=recent.index, y=recent[net_income_col], name="淨利"))

    fig.update_layout(
        title="近 12 期財務趨勢",
        template="plotly_white",
        barmode="group",
        hovermode="x unified",
        margin={"l": 16, "r": 16, "t": 48, "b": 16},
        legend={"orientation": "h", "y": 1.02, "x": 0},
        height=430,
    )
    return fig


def pivot_financial_statements(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "date" not in frame.columns or "value" not in frame.columns:
        return pd.DataFrame()

    metric_column = first_existing_column(frame, ("type", "origin_name", "name"))
    if metric_column is None:
        return pd.DataFrame()

    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["value"] = pd.to_numeric(data["value"], errors="coerce")
    data = data.dropna(subset=["date", "value", metric_column])
    if data.empty:
        return pd.DataFrame()

    pivot = data.pivot_table(
        index="date",
        columns=metric_column,
        values="value",
        aggfunc="last",
    )
    return pivot.sort_index()


def find_metric_column(frame: pd.DataFrame, keywords: tuple[str, ...]) -> str | None:
    normalized_keywords = [normalize_search_text(keyword) for keyword in keywords]
    for column in frame.columns:
        normalized = normalize_search_text(str(column))
        if any(keyword in normalized for keyword in normalized_keywords):
            return str(column)
    return None


def first_existing_column(frame: pd.DataFrame, columns: tuple[str, ...]) -> str | None:
    for column in columns:
        if column in frame.columns:
            return column
    return None


def get_metric_value(row: pd.Series, column: str | None) -> float | None:
    if column is None or column not in row:
        return None
    value = row[column]
    return float(value) if pd.notna(value) else None


def find_stock_candidates(query: str) -> list[StockProfile]:
    normalized = normalize_search_text(query)
    if not normalized:
        return [COMMON_STOCKS[0]]

    code_match = re.search(r"\d{4,6}", query)
    if code_match:
        symbol = code_match.group(0)
        return [find_profile_by_symbol(symbol)]

    matches = [
        profile
        for profile in COMMON_STOCKS
        if normalized in normalize_search_text(profile.name)
        or any(normalized in normalize_search_text(alias) for alias in profile.aliases)
    ]
    return matches[:8]


def find_profile_by_symbol(symbol: str) -> StockProfile:
    try:
        stock_id = extract_stock_id(symbol)
    except ValueError:
        stock_id = "2330"

    for profile in COMMON_STOCKS:
        if profile.symbol == stock_id:
            return profile
    return StockProfile(symbol=stock_id, name="自訂股票")


def parse_date_range(
    value: Any,
    start_default: date,
    end_default: date,
) -> tuple[date, date]:
    if isinstance(value, tuple) and len(value) == 2:
        start_date, end_date = value
        return start_date, end_date
    return start_default, end_default


def normalize_search_text(value: str) -> str:
    return re.sub(r"\s+", "", value.strip().lower())


def format_percent(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{value * 100:,.2f}%"


def format_nullable_number(value: Any) -> str:
    if not is_number(value):
        return "-"
    return format_price(float(value))


def format_metric_integer(value: Any) -> str:
    number = to_float_or_none(value)
    if number is None:
        return "-"
    return f"{int(round(number)):,}"


def format_metric_percent_integer(value: Any) -> str:
    number = to_float_or_none(value)
    if number is None:
        return "-"
    percent = number * 100 if abs(number) <= 1 else number
    return f"{int(round(percent))}%"


def format_large_number(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    abs_value = abs(value)
    if abs_value >= 100_000_000:
        return f"{value / 100_000_000:,.2f} 億"
    if abs_value >= 10_000:
        return f"{value / 10_000:,.2f} 萬"
    return format_number(value)


def safe_pct(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0) or pd.isna(numerator) or pd.isna(denominator):
        return None
    return numerator / denominator


def is_number(value: Any) -> bool:
    return value is not None and not pd.isna(value)


def to_float_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(number):
        return None
    return number


if __name__ == "__main__":
    main()
