from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


OHLCV_COLUMNS = ["date", "open", "high", "low", "close", "volume"]


def normalize_price_frame(
    frame: pd.DataFrame,
    *,
    symbol: str,
    source: str,
    required_columns: Iterable[str] = OHLCV_COLUMNS,
) -> pd.DataFrame:
    if frame.empty:
        return _empty_price_frame()

    normalized = frame.copy()
    normalized.columns = [_normalize_column_name(column) for column in normalized.columns]

    if "datetime" in normalized.columns and "date" not in normalized.columns:
        normalized = normalized.rename(columns={"datetime": "date"})

    missing = [column for column in required_columns if column not in normalized.columns]
    if missing:
        raise ValueError(f"Price frame missing required columns: {missing}")

    normalized = normalized[list(required_columns)].copy()
    normalized["date"] = pd.to_datetime(normalized["date"]).dt.date

    numeric_columns = [column for column in required_columns if column != "date"]
    for column in numeric_columns:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    normalized = normalized.dropna(subset=["date", "open", "high", "low", "close"])
    normalized = normalized.sort_values("date").reset_index(drop=True)
    normalized["symbol"] = symbol
    normalized["source"] = source
    return normalized


def _empty_price_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=[*OHLCV_COLUMNS, "symbol", "source"])


def _normalize_column_name(column: object) -> str:
    if isinstance(column, tuple):
        column = column[0]
    return str(column).strip().lower().replace(" ", "_")

