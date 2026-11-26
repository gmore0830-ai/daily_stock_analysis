# -*- coding: utf-8 -*-
"""Process-wide registry of news items retrieved during one run.

Every news item shown to an LLM gets a stable citation id (``N1``, ``N2`` ...).
The id lets post-processing attach the item's real source URL and its own
publication date (instead of the report date), and lets the market review see
the same news the per-stock analyses already used, so the two reports never
disagree about whether news exists.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

CITATION_PATTERN = re.compile(r"\[(N\d{1,4})\]")
_DATE_PATTERN = re.compile(r"(\d{4})[-/.年](\d{1,2})[-/.月](\d{1,2})")


@dataclass(frozen=True)
class NewsItem:
    news_id: str
    title: str
    url: str
    published_date: str
    source: str


def normalize_news_date(value: Any) -> str:
    """Return ``YYYY-MM-DD`` when ``value`` contains a full date, else ``''``."""
    match = _DATE_PATTERN.search(str(value or ""))
    if not match:
        return ""
    year, month, day = (int(part) for part in match.groups())
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return ""
    return f"{year:04d}-{month:02d}-{day:02d}"


def _field(item: Any, name: str) -> str:
    if isinstance(item, dict):
        value = item.get(name)
    else:
        value = getattr(item, name, None)
    return " ".join(str(value or "").split())


class NewsRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_key: Dict[str, NewsItem] = {}
        self._by_id: Dict[str, NewsItem] = {}

    def register(self, item: Any) -> Optional[str]:
        """Register a SearchResult-like object or dict; return its citation id."""
        title = _field(item, "title")
        url = _field(item, "url")
        if not title and not url:
            return None
        key = url or title
        with self._lock:
            existing = self._by_key.get(key)
            if existing is not None:
                return existing.news_id
            news_id = f"N{len(self._by_id) + 1}"
            record = NewsItem(
                news_id=news_id,
                title=title,
                url=url,
                published_date=normalize_news_date(_field(item, "published_date")),
                source=_field(item, "source"),
            )
            self._by_key[key] = record
            self._by_id[news_id] = record
            return news_id

    def get(self, news_id: str) -> Optional[NewsItem]:
        with self._lock:
            return self._by_id.get(news_id)

    def items(self) -> List[NewsItem]:
        with self._lock:
            return list(self._by_id.values())

    def reset(self) -> None:
        with self._lock:
            self._by_key.clear()
            self._by_id.clear()


_REGISTRY = NewsRegistry()


def get_news_registry() -> NewsRegistry:
    return _REGISTRY


def render_inline_citations(text: str, language: str = "zh") -> str:
    """Turn ``[N3]`` / ``[N3][N5]`` markers in prose into source links; drop unknown ids."""
    if not isinstance(text, str) or "[N" not in text:
        return text
    registry = get_news_registry()
    label = "来源" if language == "zh" else "Source"

    def _group(match: "re.Match[str]") -> str:
        ids = list(dict.fromkeys(CITATION_PATTERN.findall(match.group(0))))
        urls = [item.url for item in (registry.get(i) for i in ids) if item is not None and item.url]
        if not urls:
            return ""
        parts = [f"[{label}{idx if len(urls) > 1 else ''}]({url})" for idx, url in enumerate(urls, 1)]
        return f"（{' '.join(parts)}）" if language == "zh" else f" ({' '.join(parts)})"

    return re.sub(r"(?:\s*\[N\d{1,4}\])+", _group, text)
