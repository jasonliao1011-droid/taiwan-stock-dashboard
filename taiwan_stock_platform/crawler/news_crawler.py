from __future__ import annotations

from dataclasses import asdict, dataclass, field
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from taiwan_stock_platform.crawler.base import HttpClient
from taiwan_stock_platform.utils.validators import extract_stock_id


@dataclass(frozen=True)
class NewsItem:
    title: str
    url: str
    source: str = "Yahoo Taiwan"

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass
class YahooTaiwanNewsCrawler:
    http: HttpClient = field(default_factory=HttpClient)
    base_url: str = "https://tw.stock.yahoo.com"

    def get_stock_news(self, symbol: str, *, limit: int = 10) -> list[NewsItem]:
        stock_id = extract_stock_id(symbol)
        url = f"{self.base_url}/quote/{stock_id}/news"
        html = self.http.get_text(url)
        return self.parse_news(html, limit=limit)

    def parse_news(self, html: str, *, limit: int = 10) -> list[NewsItem]:
        soup = BeautifulSoup(html, "html.parser")
        items: list[NewsItem] = []
        seen_titles: set[str] = set()

        for anchor in soup.select("a[href]"):
            title = " ".join(anchor.get_text(" ", strip=True).split())
            href = anchor.get("href", "")
            if len(title) < 8 or "/news/" not in href or title in seen_titles:
                continue

            seen_titles.add(title)
            items.append(
                NewsItem(
                    title=title,
                    url=urljoin(self.base_url, href),
                )
            )
            if len(items) >= limit:
                break

        return items
