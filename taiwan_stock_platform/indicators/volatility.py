from __future__ import annotations

import pandas as pd


def add_bollinger_bands(
    frame: pd.DataFrame,
    *,
    window: int = 20,
    std_multiplier: float = 2.0,
) -> pd.DataFrame:
    result = frame.copy()
    middle = result["close"].rolling(window=window).mean()
    std = result["close"].rolling(window=window).std()
    result["bb_middle"] = middle
    result["bb_upper"] = middle + (std_multiplier * std)
    result["bb_lower"] = middle - (std_multiplier * std)
    return result

