from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from taiwan_stock_platform.indicators.momentum import add_macd, add_rsi
from taiwan_stock_platform.indicators.moving_average import add_ema, add_sma
from taiwan_stock_platform.indicators.volatility import add_bollinger_bands


def enrich_technical_indicators(
    frame: pd.DataFrame,
    *,
    ma_windows: Iterable[int] = (5, 20, 60),
    rsi_window: int = 14,
) -> pd.DataFrame:
    result = frame.copy()
    result = add_sma(result, ma_windows)
    result = add_ema(result, (12, 26))
    result = add_rsi(result, window=rsi_window)
    result = add_macd(result)
    result = add_bollinger_bands(result)
    return result

