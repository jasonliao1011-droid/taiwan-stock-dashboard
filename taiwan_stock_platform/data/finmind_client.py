from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd
import requests

from taiwan_stock_platform.data.exceptions import DataNotFoundError, DataSourceError
from taiwan_stock_platform.data.schema import normalize_price_frame
from taiwan_stock_platform.utils.validators import extract_stock_id


@dataclass
class FinMindClient:
    token: str | None = None
    base_url: str = "https://api.finmindtrade.com/api/v4/data"
    timeout: int = 20
    session: requests.Session = field(default_factory=requests.Session)

    def fetch_daily_prices(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        stock_id = extract_stock_id(symbol)
        params: dict[str, str] = {
            "dataset": "TaiwanStockPrice",
            "data_id": stock_id,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
        }
        if self.token:
            params["token"] = self.token

        try:
            response = self.session.get(self.base_url, params=params, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:  # pragma: no cover - upstream network behavior
            raise DataSourceError(f"FinMind request failed for {stock_id}") from exc

        status = payload.get("status")
        if status not in (None, 200):
            message = payload.get("msg") or payload.get("message") or "Unknown FinMind error"
            raise DataSourceError(f"FinMind returned status {status}: {message}")

        data = payload.get("data") or []
        if not data:
            raise DataNotFoundError(f"No FinMind rows found for {stock_id}")

        raw = pd.DataFrame(data)
        raw = raw.rename(
            columns={
                "max": "high",
                "min": "low",
                "Trading_Volume": "volume",
            }
        )
        return normalize_price_frame(raw, symbol=stock_id, source="finmind")

