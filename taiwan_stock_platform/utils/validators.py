from __future__ import annotations

import re
from datetime import date


STOCK_ID_PATTERN = re.compile(r"^\d{4,6}$")
EXCHANGE_SUFFIXES = {
    "TW": ".TW",
    "TWO": ".TWO",
}


def extract_stock_id(symbol: str) -> str:
    cleaned = symbol.strip().upper()
    cleaned = cleaned.removesuffix(".TW").removesuffix(".TWO")
    if not STOCK_ID_PATTERN.match(cleaned):
        raise ValueError(f"Invalid Taiwan stock id: {symbol}")
    return cleaned


def normalize_yahoo_ticker(symbol: str, *, exchange: str = "TW") -> str:
    cleaned = symbol.strip().upper()
    if cleaned.endswith(".TW") or cleaned.endswith(".TWO"):
        return cleaned

    stock_id = extract_stock_id(cleaned)
    suffix = EXCHANGE_SUFFIXES.get(exchange)
    if suffix is None:
        raise ValueError(f"Unsupported exchange: {exchange}")
    return f"{stock_id}{suffix}"


def validate_date_range(start_date: date, end_date: date) -> None:
    if start_date > end_date:
        raise ValueError("start_date must be earlier than or equal to end_date")

