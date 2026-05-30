from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from datetime import date
from typing import Any, Callable, TypeVar

import pandas as pd
import yfinance as yf


T = TypeVar("T")


STANDARD_COLUMNS = [
    "date",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "current_price",
    "company_name",
    "ticker",
    "symbol",
    "currency",
    "source",
]


class StockDataError(RuntimeError):
    """Base exception for stock data loading errors."""


class StockNotFoundError(StockDataError):
    """Raised when Yahoo Finance returns no usable data for a stock."""


class StockDataTimeoutError(StockDataError):
    """Raised when a Yahoo Finance request exceeds the configured timeout."""


@dataclass(frozen=True)
class StockMetadata:
    ticker: str
    symbol: str
    company_name: str
    current_price: float | None
    currency: str | None


@dataclass(frozen=True)
class TaiwanStockDataClient:
    """Fetch Taiwan stock price data from Yahoo Finance via yfinance."""

    default_exchange_suffix: str = ".TW"
    timeout_seconds: int = 10
    auto_adjust: bool = False
    source_name: str = "yfinance"

    def fetch_stock_data(
        self,
        symbol: str,
        *,
        period: str = "1y",
        interval: str = "1d",
        start: str | date | None = None,
        end: str | date | None = None,
    ) -> pd.DataFrame:
        """
        Fetch historical prices, volume, current price, and company name.

        Examples:
            2330 -> 2330.TW
            2330.TW -> 2330.TW

        Returns:
            A normalized pandas DataFrame with one row per trading date.
        """
        ticker = normalize_taiwan_ticker(
            symbol,
            default_exchange_suffix=self.default_exchange_suffix,
        )

        history = self._download_history(
            ticker=ticker,
            period=period,
            interval=interval,
            start=start,
            end=end,
        )
        if history.empty:
            raise StockNotFoundError(f"No stock data found for {ticker}.")

        metadata = self._load_metadata(ticker=ticker, history=history)
        result = history.copy()
        result["current_price"] = metadata.current_price
        result["company_name"] = metadata.company_name
        result["ticker"] = metadata.ticker
        result["symbol"] = metadata.symbol
        result["currency"] = metadata.currency
        result["source"] = self.source_name

        return result[STANDARD_COLUMNS]

    def _download_history(
        self,
        *,
        ticker: str,
        period: str,
        interval: str,
        start: str | date | None,
        end: str | date | None,
    ) -> pd.DataFrame:
        def request() -> pd.DataFrame:
            kwargs: dict[str, Any] = {
                "tickers": ticker,
                "period": period if start is None else None,
                "start": _to_date_string(start),
                "end": _to_date_string(end),
                "interval": interval,
                "auto_adjust": self.auto_adjust,
                "progress": False,
                "threads": False,
                "timeout": self.timeout_seconds,
                "group_by": "column",
                "multi_level_index": False,
            }
            kwargs = {key: value for key, value in kwargs.items() if value is not None}

            try:
                return yf.download(**kwargs)
            except TypeError:
                kwargs.pop("multi_level_index", None)
                return yf.download(**kwargs)

        raw = _run_with_timeout(
            request,
            timeout_seconds=self.timeout_seconds + 2,
            timeout_message=f"Yahoo Finance history request timed out for {ticker}.",
        )
        return normalize_history_frame(raw, ticker=ticker)

    def _load_metadata(self, *, ticker: str, history: pd.DataFrame) -> StockMetadata:
        fallback_price = _to_optional_float(history["close"].iloc[-1])
        stock = yf.Ticker(ticker)

        current_price = self._load_current_price(stock) or fallback_price
        company_name, currency = self._load_company_info(stock, ticker=ticker)

        return StockMetadata(
            ticker=ticker,
            symbol=ticker.split(".", maxsplit=1)[0],
            company_name=company_name,
            current_price=current_price,
            currency=currency,
        )

    def _load_current_price(self, stock: yf.Ticker) -> float | None:
        def request() -> float | None:
            fast_info = stock.fast_info
            for key in (
                "lastPrice",
                "last_price",
                "regularMarketPrice",
                "regular_market_price",
                "currentPrice",
                "current_price",
            ):
                value = _read_mapping_or_attr(fast_info, key)
                price = _to_optional_float(value)
                if price is not None:
                    return price
            return None

        try:
            return _run_with_timeout(
                request,
                timeout_seconds=self.timeout_seconds,
                timeout_message="Yahoo Finance current price request timed out.",
            )
        except StockDataError:
            return None

    def _load_company_info(self, stock: yf.Ticker, *, ticker: str) -> tuple[str, str | None]:
        def request() -> dict[str, Any]:
            if hasattr(stock, "get_info"):
                return stock.get_info()
            return stock.info

        try:
            info = _run_with_timeout(
                request,
                timeout_seconds=self.timeout_seconds,
                timeout_message=f"Yahoo Finance company info request timed out for {ticker}.",
            )
        except StockDataError:
            info = {}

        company_name = (
            info.get("longName")
            or info.get("shortName")
            or info.get("displayName")
            or ticker
        )
        currency = info.get("currency") or info.get("financialCurrency")
        return str(company_name), str(currency) if currency else None


def fetch_stock_data(
    symbol: str,
    *,
    period: str = "1y",
    interval: str = "1d",
    start: str | date | None = None,
    end: str | date | None = None,
    timeout_seconds: int = 10,
) -> pd.DataFrame:
    """Convenience function for fetching Taiwan stock data."""
    client = TaiwanStockDataClient(timeout_seconds=timeout_seconds)
    return client.fetch_stock_data(
        symbol,
        period=period,
        interval=interval,
        start=start,
        end=end,
    )


def normalize_taiwan_ticker(
    symbol: str,
    *,
    default_exchange_suffix: str = ".TW",
) -> str:
    cleaned = symbol.strip().upper()
    if not cleaned:
        raise ValueError("Stock symbol cannot be empty.")

    if cleaned.endswith(".TW") or cleaned.endswith(".TWO"):
        return cleaned

    if cleaned.isdigit():
        return f"{cleaned}{default_exchange_suffix}"

    raise ValueError(
        "Invalid Taiwan stock symbol. Use a numeric code like 2330 or a ticker like 2330.TW."
    )


def normalize_history_frame(raw: pd.DataFrame | None, *, ticker: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        return _empty_stock_frame()

    frame = raw.copy()
    frame = _flatten_yfinance_columns(frame)
    frame = frame.reset_index()
    frame = frame.rename(columns={column: _normalize_column_name(column) for column in frame.columns})

    if "datetime" in frame.columns and "date" not in frame.columns:
        frame = frame.rename(columns={"datetime": "date"})

    if "adj_close" not in frame.columns:
        frame["adj_close"] = frame.get("close")

    required = {"date", "open", "high", "low", "close", "volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise StockDataError(f"Yahoo Finance response for {ticker} is missing: {sorted(missing)}")

    frame = frame[["date", "open", "high", "low", "close", "adj_close", "volume"]].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")

    numeric_columns = ["open", "high", "low", "close", "adj_close", "volume"]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.dropna(subset=["date", "close"])
    frame["volume"] = frame["volume"].fillna(0)
    frame = frame.sort_values("date").reset_index(drop=True)

    if frame.empty:
        return _empty_stock_frame()

    return frame


def _run_with_timeout(
    func: Callable[[], T],
    *,
    timeout_seconds: int,
    timeout_message: str,
) -> T:
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(func)
    try:
        return future.result(timeout=timeout_seconds)
    except FutureTimeoutError as exc:
        future.cancel()
        raise StockDataTimeoutError(timeout_message) from exc
    except Exception as exc:
        raise StockDataError(str(exc)) from exc
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def _flatten_yfinance_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame.columns, pd.MultiIndex):
        return frame

    flattened = frame.copy()
    flattened.columns = [
        next((str(part) for part in column if str(part).lower() not in {"", "none"}), "")
        for column in flattened.columns
    ]
    return flattened


def _normalize_column_name(column: object) -> str:
    return str(column).strip().lower().replace(" ", "_")


def _to_date_string(value: str | date | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value.isoformat()
    return value


def _to_optional_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if pd.isna(number):
        return None
    return number


def _read_mapping_or_attr(target: Any, key: str) -> Any:
    if hasattr(target, "get"):
        value = target.get(key)
        if value is not None:
            return value

    try:
        return target[key]
    except (KeyError, TypeError, AttributeError):
        return getattr(target, key, None)


def _empty_stock_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=STANDARD_COLUMNS)


if __name__ == "__main__":
    data = fetch_stock_data("2330", period="5d")
    print(data.tail())
