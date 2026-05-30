from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from bs4 import BeautifulSoup

from taiwan_stock_platform.crawler.base import HttpClient


@dataclass
class HtmlTableCrawler:
    http: HttpClient = field(default_factory=HttpClient)

    def fetch_tables(self, url: str, *, table_selector: str = "table") -> list[pd.DataFrame]:
        html = self.http.get_text(url)
        return self.parse_tables(html, table_selector=table_selector)

    def parse_tables(self, html: str, *, table_selector: str = "table") -> list[pd.DataFrame]:
        soup = BeautifulSoup(html, "html.parser")
        tables = soup.select(table_selector)
        frames: list[pd.DataFrame] = []

        for table in tables:
            rows: list[list[str]] = []
            for tr in table.select("tr"):
                cells = [cell.get_text(" ", strip=True) for cell in tr.select("th,td")]
                if cells:
                    rows.append(cells)

            if not rows:
                continue

            header, *body = rows
            if body and len(header) == len(body[0]):
                frames.append(pd.DataFrame(body, columns=header))
            else:
                frames.append(pd.DataFrame(rows))

        return frames
