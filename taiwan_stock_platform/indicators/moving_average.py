from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def add_sma(frame: pd.DataFrame, windows: Iterable[int]) -> pd.DataFrame:
    result = frame.copy()
    for window in windows:
        result[f"sma_{window}"] = result["close"].rolling(window=window).mean()
    return result


def add_ema(frame: pd.DataFrame, windows: Iterable[int]) -> pd.DataFrame:
    result = frame.copy()
    for window in windows:
        result[f"ema_{window}"] = result["close"].ewm(span=window, adjust=False).mean()
    return result

