from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo


TAIPEI_TZ = ZoneInfo("Asia/Taipei")


def today_taipei() -> date:
    return datetime.now(TAIPEI_TZ).date()


def default_date_range(days: int = 365) -> tuple[date, date]:
    end_date = today_taipei()
    start_date = end_date - timedelta(days=days)
    return start_date, end_date

