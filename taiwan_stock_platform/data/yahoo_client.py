from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd
import yfinance as yf

from taiwan_stock_platform.data.exceptions import DataNotFoundError, DataSourceError
from taiwan_stock_platform.data.schema import normalize_price_frame
from taiwan_stock_platform.utils.validators import normalize_yahoo_ticker


@dataclass(frozen=True)
class YahooFinanceClient:
    auto_adjust: bool = False

    def fetch_daily_prices(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        *,
        exchange: str = "TW",
    ) -> pd.DataFrame:
        ticker = normalize_yahoo_ticker(symbol, exchange=exchange)
        end_exclusive = end_date + timedelta(days=1)

        try:
            raw = yf.download(
                ticker,
                start=start_date.isoformat(),
                end=end_exclusive.isoformat(),
                interval="1d",
                auto_adjust=self.auto_adjust,
                progress=False,
                group_by="column",
                threads=False,
            )
        except Exception as exc:  # pragma: no cover - upstream network behavior
            raise DataSourceError(f"yfinance request failed for {ticker}") from exc

        if raw.empty:
            raise DataNotFoundError(f"No yfinance rows found for {ticker}")

        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        raw = raw.reset_index()
        raw = raw.rename(
            columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        return normalize_price_frame(raw, symbol=ticker, source="yfinance")

