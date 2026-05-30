from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


@dataclass
class HttpClient:
    timeout: int = 20
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0 Safari/537.36"
    )
    session: requests.Session = field(default_factory=requests.Session)

    def __post_init__(self) -> None:
        retry = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        self.session.headers.update({"User-Agent": self.user_agent})

    def get_text(self, url: str, **params: Any) -> str:
        response = self.session.get(url, params=params or None, timeout=self.timeout)
        response.raise_for_status()
        return response.text

    def get_json(self, url: str, **params: Any) -> dict[str, Any]:
        response = self.session.get(url, params=params or None, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

