from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
import requests


AI_SCORE_THRESHOLD = 80
REVENUE_YOY_THRESHOLD = 0.20


def fetch_monthly_revenue(
    symbol: str,
    *,
    finmind_base_url: str,
    finmind_token: str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    timeout: int = 20,
) -> pd.DataFrame:
    """Fetch monthly revenue data from FinMind and normalize it for screening."""
    today = end_date or date.today()
    start = start_date or date(today.year - 2, 1, 1)
    stock_id = _extract_stock_id(symbol)

    params: dict[str, str] = {
        "dataset": "TaiwanStockMonthRevenue",
        "data_id": stock_id,
        "start_date": start.isoformat(),
        "end_date": today.isoformat(),
    }
    if finmind_token:
        params["token"] = finmind_token

    try:
        response = requests.get(finmind_base_url, params=params, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return pd.DataFrame(columns=["date", "revenue", "revenue_yoy"])

    data = payload.get("data") or []
    if not data:
        return pd.DataFrame(columns=["date", "revenue", "revenue_yoy"])

    return normalize_monthly_revenue(pd.DataFrame(data))


def normalize_monthly_revenue(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a frame with date, revenue, and revenue_yoy columns."""
    if frame.empty:
        return pd.DataFrame(columns=["date", "revenue", "revenue_yoy"])

    result = frame.copy()
    result.columns = [_normalize_column_name(column) for column in result.columns]

    if "date" not in result.columns:
        return pd.DataFrame(columns=["date", "revenue", "revenue_yoy"])

    revenue_column = _first_existing_column(
        result,
        ("revenue", "month_revenue", "monthly_revenue"),
    )
    if revenue_column is None:
        return pd.DataFrame(columns=["date", "revenue", "revenue_yoy"])

    yoy_column = _first_existing_column(
        result,
        ("revenue_yoy", "yoy", "growth_rate", "revenue_growth_rate"),
    )

    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    result["revenue"] = pd.to_numeric(result[revenue_column], errors="coerce")

    if yoy_column:
        result["revenue_yoy"] = pd.to_numeric(result[yoy_column], errors="coerce")
        result["revenue_yoy"] = result["revenue_yoy"].where(
            result["revenue_yoy"].abs() <= 1,
            result["revenue_yoy"] / 100,
        )
    else:
        result["revenue_yoy"] = _calculate_yoy_from_revenue(result)

    result = result.dropna(subset=["date", "revenue"]).sort_values("date")
    return result[["date", "revenue", "revenue_yoy"]].reset_index(drop=True)


def build_ai_candidate_frame(
    *,
    symbol: str,
    name: str,
    price_frame: pd.DataFrame,
    revenue_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Build a one-row candidate frame from price and revenue data."""
    if price_frame.empty or "close" not in price_frame.columns:
        return _empty_candidate_frame()

    prices = price_frame.copy()
    prices["close"] = pd.to_numeric(prices["close"], errors="coerce")
    prices = prices.dropna(subset=["close"]).sort_values("date")
    if prices.empty:
        return _empty_candidate_frame()

    prices["ma5"] = prices["close"].rolling(window=5).mean()
    prices["ma20"] = prices["close"].rolling(window=20).mean()
    latest = prices.iloc[-1]

    revenue_yoy = latest_revenue_yoy(revenue_frame)
    ma5 = _to_float_or_none(latest.get("ma5"))
    ma20 = _to_float_or_none(latest.get("ma20"))
    close = _to_float_or_none(latest.get("close"))

    candidate = {
        "stock_id": _extract_stock_id(symbol),
        "stock_name": name,
        "close": close,
        "ma5": ma5,
        "ma20": ma20,
        "revenue_yoy": revenue_yoy,
    }
    candidate["recommendation_index"] = calculate_recommendation_index(candidate)
    return pd.DataFrame([candidate])


def screen_ai_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    """Filter stocks where 5MA > 20MA and latest monthly revenue YoY > 20%."""
    if candidates.empty:
        return candidates.copy()

    required_columns = {"ma5", "ma20", "revenue_yoy"}
    missing = required_columns.difference(candidates.columns)
    if missing:
        raise ValueError(f"Missing AI screening columns: {sorted(missing)}")

    data = candidates.copy()
    condition = (data["ma5"] > data["ma20"]) & (
        data["revenue_yoy"] > REVENUE_YOY_THRESHOLD
    )
    return data.loc[condition].assign(ai_label="AI 綜合評分高分股").reset_index(drop=True)


def calculate_recommendation_index(candidate: dict[str, Any]) -> int:
    """Calculate a simple 0-100 recommendation index for st.metric."""
    score = 40

    ma5 = candidate.get("ma5")
    ma20 = candidate.get("ma20")
    revenue_yoy = candidate.get("revenue_yoy")

    if ma5 is not None and ma20 is not None:
        if ma5 > ma20:
            score += 30
        elif ma20:
            score += max(0, int((ma5 / ma20) * 15))

    if revenue_yoy is not None:
        if revenue_yoy > REVENUE_YOY_THRESHOLD:
            score += 30
        else:
            score += max(0, int((revenue_yoy / REVENUE_YOY_THRESHOLD) * 15))

    return max(0, min(100, score))


def latest_revenue_yoy(revenue_frame: pd.DataFrame) -> float | None:
    if revenue_frame.empty or "revenue_yoy" not in revenue_frame.columns:
        return None

    data = revenue_frame.dropna(subset=["revenue_yoy"]).sort_values("date")
    if data.empty:
        return None
    return _to_float_or_none(data.iloc[-1]["revenue_yoy"])


def _calculate_yoy_from_revenue(frame: pd.DataFrame) -> pd.Series:
    data = frame[["date", "revenue"]].copy()
    data["year_month"] = data["date"].dt.strftime("%m")
    data["previous_year_revenue"] = data.groupby("year_month")["revenue"].shift(1)
    return (data["revenue"] - data["previous_year_revenue"]) / data[
        "previous_year_revenue"
    ]


def _extract_stock_id(symbol: str) -> str:
    return symbol.strip().upper().removesuffix(".TW").removesuffix(".TWO")


def _first_existing_column(frame: pd.DataFrame, columns: tuple[str, ...]) -> str | None:
    for column in columns:
        if column in frame.columns:
            return column
    return None


def _normalize_column_name(column: object) -> str:
    return str(column).strip().lower().replace(" ", "_")


def _to_float_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(number):
        return None
    return number


def _empty_candidate_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "stock_id",
            "stock_name",
            "close",
            "ma5",
            "ma20",
            "revenue_yoy",
            "recommendation_index",
        ]
    )


def calculate_ai_recommendation(
    df_stock: pd.DataFrame,
    df_revenue: pd.DataFrame,
) -> tuple[str, int, list[str]]:
    """Combine technical and revenue signals into a 0-100 AI recommendation."""
    score = 60
    reasons: list[str] = []

    if df_stock.empty:
        reasons.append("技術面資料不足，維持基準分。")
    else:
        stock_data = df_stock.copy()
        stock_columns = {
            _normalize_column_name(column): column for column in stock_data.columns
        }
        date_column = stock_columns.get("date")
        if date_column:
            stock_data["_sort_date"] = pd.to_datetime(
                stock_data[date_column],
                errors="coerce",
            )
            stock_data = stock_data.sort_values("_sort_date")

        latest_stock = stock_data.iloc[-1]
        ma5_column = next(
            (
                stock_columns[column]
                for column in ("ma5", "5ma", "5_ma", "sma5", "sma_5")
                if column in stock_columns
            ),
            None,
        )
        ma20_column = next(
            (
                stock_columns[column]
                for column in ("ma20", "20ma", "20_ma", "sma20", "sma_20")
                if column in stock_columns
            ),
            None,
        )

        ma5 = _to_float_or_none(latest_stock.get(ma5_column)) if ma5_column else None
        ma20 = (
            _to_float_or_none(latest_stock.get(ma20_column)) if ma20_column else None
        )

        if ma5 is None or ma20 is None:
            reasons.append("技術面缺少 5MA 或 20MA，分數不調整。")
        elif ma5 > ma20:
            score += 15
            reasons.append("技術面：5MA 高於 20MA，呈現多頭排列，分數 +15。")
        else:
            score -= 10
            reasons.append("技術面：5MA 未高於 20MA，短線動能偏弱，分數 -10。")

    if df_revenue.empty:
        reasons.append("基本面營收資料不足，分數不調整。")
    else:
        revenue_data = df_revenue.copy()
        revenue_columns = {
            _normalize_column_name(column): column for column in revenue_data.columns
        }
        date_column = revenue_columns.get("date")
        if date_column:
            revenue_data["_sort_date"] = pd.to_datetime(
                revenue_data[date_column],
                errors="coerce",
            )
            revenue_data = revenue_data.sort_values("_sort_date")

        latest_revenue = revenue_data.iloc[-1]
        yoy_column = next(
            (
                revenue_columns[column]
                for column in (
                    "revenue_yoy",
                    "yoy",
                    "yoy(%)",
                    "yoy_%",
                    "yoy_percent",
                    "yoy_pct",
                    "growth_rate",
                    "revenue_growth_rate",
                )
                if column in revenue_columns
            ),
            None,
        )
        yoy = _to_float_or_none(latest_revenue.get(yoy_column)) if yoy_column else None

        if yoy is None:
            reasons.append("基本面缺少最新 YoY(%)，分數不調整。")
        else:
            yoy_percent = yoy * 100 if abs(yoy) <= 1 else yoy
            if yoy_percent > 20:
                score += 25
                reasons.append(
                    f"基本面：最新營收 YoY {yoy_percent:.1f}%，營收成長強勁，分數 +25。"
                )
            elif yoy_percent < 0:
                score -= 15
                reasons.append(
                    f"基本面：最新營收 YoY {yoy_percent:.1f}%，營收年減，分數 -15。"
                )
            else:
                reasons.append(
                    f"基本面：最新營收 YoY {yoy_percent:.1f}%，未達強勁成長門檻，分數不調整。"
                )

    score = max(0, min(100, score))
    if score >= 80:
        signal = "強力推薦"
    elif score >= 60:
        signal = "中性持有"
    else:
        signal = "觀望保守"

    return signal, score, reasons


def get_portfolio_suggestion(
    style: str,
    score: int | float = 80,
) -> tuple[list[str], list[float], str]:
    """Return portfolio pie chart labels, weights, and strategy text."""
    allocations: dict[str, dict[str, float]] = {
        "保守型 (領息小資/退休族)": {
            "高股息 ETF": 50,
            "大型權值股": 20,
            "穩定金融股": 30,
        },
        "積極型 (追求資本利得)": {
            "半導體上游": 40,
            "AI 伺服器供應鏈": 40,
            "高股性設備股": 20,
        },
        "價值型 (巴菲特價值投資)": {
            "低本益比價值股": 40,
            "傳產龍頭": 30,
            "高淨值比潛力股": 30,
        },
    }
    style_aliases = {
        "保守型": "保守型 (領息小資/退休族)",
        "積極型": "積極型 (追求資本利得)",
        "價值型": "價值型 (巴菲特價值投資)",
    }
    normalized_style = style_aliases.get(style, style)
    if normalized_style not in allocations:
        valid_styles = "、".join(allocations)
        raise ValueError(f"未知的投資風格：{style}。可用風格：{valid_styles}")

    allocation = allocations[normalized_style].copy()
    score_value = _to_float_or_none(score) or 0
    core_label = next(iter(allocation))

    if score_value > AI_SCORE_THRESHOLD:
        boost = min(10.0, 100.0 - allocation[core_label])
        other_labels = [label for label in allocation if label != core_label]
        other_total = sum(allocation[label] for label in other_labels)
        if boost > 0 and other_total > 0:
            allocation[core_label] += boost
            reduction_ratio = (other_total - boost) / other_total
            for label in other_labels:
                allocation[label] = round(allocation[label] * reduction_ratio, 2)

            rounding_gap = round(100.0 - sum(allocation.values()), 2)
            allocation[other_labels[-1]] = round(
                allocation[other_labels[-1]] + rounding_gap,
                2,
            )

        strategy_text = (
            f"AI 評分 {score_value:.0f} 分高於 {AI_SCORE_THRESHOLD} 分，"
            f"將「{core_label}」核心部位調高 10 個百分點，其他資產等比例下修。"
        )
    else:
        strategy_text = (
            f"AI 評分 {score_value:.0f} 分未高於 {AI_SCORE_THRESHOLD} 分，"
            f"維持「{normalized_style}」標準配置。"
        )

    labels = list(allocation.keys())
    values = [allocation[label] for label in labels]
    return labels, values, strategy_text


def generate_news_summary(news_list: list[Any]) -> str:
    """
    傳入一個包含多篇新聞標題與摘要的 list，
    將所有文字組合成一個 Prompt，調用免費的 AI 模型
    （優先使用 google-generativeai 或 g4f，若未安裝則使用預設的簡單規則提取），
    返回一篇 200 字以內的台股個股實時綜合輿情評論。
    """
    empty_message = "目前尚無相關新聞，無法生成 AI 綜合評論。"
    if not news_list:
        return empty_message

    def clean_text(value: Any) -> str:
        return " ".join(str(value).strip().split()) if value is not None else ""

    def extract_news_text(item: Any) -> str:
        if isinstance(item, str):
            return clean_text(item)
        if isinstance(item, dict):
            title = clean_text(item.get("title") or item.get("headline"))
            summary = clean_text(
                item.get("summary")
                or item.get("description")
                or item.get("content")
                or item.get("snippet")
            )
            if title and summary:
                return f"{title}：{summary}"
            return title or summary
        return clean_text(item)

    def trim_summary(text: Any) -> str:
        summary = clean_text(text)
        if len(summary) <= 200:
            return summary
        return f"{summary[:197].rstrip()}..."

    news_texts = [
        extract_news_text(item)
        for item in news_list[:5]
        if extract_news_text(item)
    ]
    if not news_texts:
        return empty_message

    news_text = "\n".join(f"{index + 1}. {text}" for index, text in enumerate(news_texts))
    system_prompt = (
        "你是一位專業的台股分析師。請根據以下最新的市場新聞標題，"
        "為這檔股票整理出一篇簡短、客觀的綜合市場輿情評論，"
        "指出目前的市場關注焦點（如 AI 需求、營收表現或產能動態）。"
        "內容請控制在 150-200 字內，並用繁體中文回答。"
    )
    prompt = f"{system_prompt}\n\n最新新聞：\n{news_text}"

    try:
        import importlib
        import os

        genai = importlib.import_module("google.generativeai")
        api_key = (
            os.getenv("GOOGLE_API_KEY")
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_GENERATIVEAI_API_KEY")
        )
        if api_key:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
            ai_text = getattr(response, "text", "")
            if clean_text(ai_text):
                return trim_summary(ai_text)
    except Exception:
        pass

    try:
        import importlib

        try:
            client_module = importlib.import_module("g4f.client")
            client = client_module.Client()
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": news_text},
                ],
            )
            ai_text = response.choices[0].message.content
        except Exception:
            g4f = importlib.import_module("g4f")
            ai_text = g4f.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": news_text},
                ],
            )
        if clean_text(ai_text):
            return trim_summary(ai_text)
    except Exception:
        pass

    focus_keywords = {
        "AI 需求與伺服器供應鏈": ("AI", "伺服器", "GPU", "HPC", "CoWoS", "資料中心"),
        "營收與獲利表現": ("營收", "獲利", "EPS", "毛利", "財報", "法說"),
        "產能、訂單與出貨動態": ("產能", "擴產", "訂單", "出貨", "庫存", "供應鏈"),
        "籌碼與市場評價": ("股價", "外資", "投信", "買超", "賣超", "目標價"),
    }
    focuses = [
        label
        for label, keywords in focus_keywords.items()
        if any(keyword.lower() in news_text.lower() for keyword in keywords)
    ]
    focus_text = "、".join(focuses[:3]) if focuses else "公司營運與市場評價"
    fallback = (
        f"近期新聞顯示，市場對該股的關注主要集中在{focus_text}。"
        "相關標題反映投資人正觀察題材能否轉化為營收與獲利動能，"
        "以及供應鏈或產能變化對後續股價表現的影響。"
        "整體輿情仍需搭配最新月營收、財報與技術面趨勢一併評估。"
    )
    return trim_summary(fallback)
