from __future__ import annotations

from numbers import Number


def format_price(value: Number | None) -> str:
    if value is None:
        return "-"
    return f"{float(value):,.2f}"


def format_number(value: Number | None) -> str:
    if value is None:
        return "-"
    return f"{float(value):,.0f}"

