from __future__ import annotations

import pandas as pd


def add_rsi(frame: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    result = frame.copy()
    delta = result["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    relative_strength = avg_gain / avg_loss
    result[f"rsi_{window}"] = 100 - (100 / (1 + relative_strength))
    return result


def add_macd(
    frame: pd.DataFrame,
    *,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    result = frame.copy()
    ema_fast = result["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = result["close"].ewm(span=slow, adjust=False).mean()
    result["macd"] = ema_fast - ema_slow
    result["macd_signal"] = result["macd"].ewm(span=signal, adjust=False).mean()
    result["macd_hist"] = result["macd"] - result["macd_signal"]
    return result

