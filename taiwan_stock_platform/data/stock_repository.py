from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from taiwan_stock_platform.data.exceptions import DataSourceError
from taiwan_stock_platform.data.finmind_client import FinMindClient
from taiwan_stock_platform.data.yahoo_client import YahooFinanceClient
from taiwan_stock_platform.utils.validators import validate_date_range


@dataclass
class StockRepository:
    yahoo: YahooFinanceClient
    finmind: FinMindClient

    def get_daily_prices(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        *,
        source: str = "auto",
        exchange: str = "TW",
    ) -> pd.DataFrame:
        validate_date_range(start_date, end_date)

        if source == "yfinance":
            return self.yahoo.fetch_daily_prices(
                symbol,
                start_date,
                end_date,
                exchange=exchange,
            )

        if source == "finmind":
            return self.finmind.fetch_daily_prices(symbol, start_date, end_date)

        if source != "auto":
            raise ValueError(f"Unsupported data source: {source}")

        errors: list[str] = []
        for loader in (
            lambda: self.yahoo.fetch_daily_prices(
                symbol,
                start_date,
                end_date,
                exchange=exchange,
            ),
            lambda: self.finmind.fetch_daily_prices(symbol, start_date, end_date),
        ):
            try:
                return loader()
            except DataSourceError as exc:
                errors.append(str(exc))

        raise DataSourceError("All data sources failed: " + " | ".join(errors))

