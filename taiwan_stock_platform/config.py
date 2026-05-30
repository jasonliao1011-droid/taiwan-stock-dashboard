from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class AppConfig:
    finmind_token: str | None
    finmind_base_url: str
    default_stock_id: str
    default_exchange: str
    cache_ttl_seconds: int
    default_start_days: int

    @classmethod
    def from_env(cls) -> "AppConfig":
        load_dotenv()
        token = os.getenv("FINMIND_TOKEN") or None
        return cls(
            finmind_token=token,
            finmind_base_url=os.getenv(
                "FINMIND_BASE_URL",
                "https://api.finmindtrade.com/api/v4/data",
            ),
            default_stock_id=os.getenv("DEFAULT_STOCK_ID", "2330"),
            default_exchange=os.getenv("DEFAULT_EXCHANGE", "TW"),
            cache_ttl_seconds=_get_int_env("CACHE_TTL_SECONDS", 900),
            default_start_days=_get_int_env("DEFAULT_START_DAYS", 365),
        )


def _get_int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError:
        return default
