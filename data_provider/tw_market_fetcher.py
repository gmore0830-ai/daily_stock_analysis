# -*- coding: utf-8 -*-
"""Taiwan whole-market data from official TWSE / TPEx endpoints (fail-open).

Fills what yfinance does not provide for the Taiwan market review:

* TPEx 櫃買指數 (TWOII)       -> https://www.tpex.org.tw/openapi/v1/tpex_index
* TWSE 加權指數成交金額         -> TWSE MI_INDEX type=MS「大盤統計資訊」總計
* 上市漲跌家數 / 漲跌停家數      -> TWSE MI_INDEX type=MS「漲跌證券數合計」股票欄
* 上市類股漲跌幅排行            -> TWSE MI_INDEX type=IND「價格指數(臺灣證券交易所)」類指數

Every public method returns ``None`` / empty on any error so the review never fails
because an exchange endpoint is slow or changed (verified live 2026-11-26).
"""

from __future__ import annotations

import logging
import re
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

_TWSE_MI_INDEX = "https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX"
_TPEX_INDEX = "https://www.tpex.org.tw/openapi/v1/tpex_index"
_HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

# Aggregate groupings in the TWSE sector table; ranking uses the leaf sectors only.
_AGGREGATE_SECTORS = {
    "水泥窯製類指數",
    "塑膠化工類指數",
    "機電類指數",
    "化學生技醫療類指數",
    "電子工業類指數",
}
_COUNT_WITH_LIMIT = re.compile(r"^\s*([\d,]+)\s*(?:\(\s*([\d,]+)\s*\))?\s*$")


def _num(value: Any) -> Optional[float]:
    text = re.sub(r"<[^>]+>", "", str(value or "")).replace(",", "").strip()
    if text in {"", "--", "-", "X"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _count_pair(value: Any) -> Tuple[Optional[int], Optional[int]]:
    """'971(57)' -> (971, 57)."""
    match = _COUNT_WITH_LIMIT.match(str(value or ""))
    if not match:
        return None, None
    total = int(match.group(1).replace(",", ""))
    limit = int(match.group(2).replace(",", "")) if match.group(2) else 0
    return total, limit


class TwMarketFetcher:
    def __init__(self, timeout: float = 30.0, cache_ttl: float = 600.0) -> None:
        self._timeout = timeout
        self._cache_ttl = cache_ttl
        self._cache: Dict[Tuple[str, str], Tuple[float, Any]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ http
    def _get_json(self, url: str, params: Optional[dict] = None) -> Any:
        key = (url, repr(sorted((params or {}).items())))
        with self._lock:
            hit = self._cache.get(key)
            if hit and time.time() - hit[0] < self._cache_ttl:
                return hit[1]
        resp = requests.get(url, params=params, headers=_HEADERS, timeout=self._timeout)
        resp.raise_for_status()
        payload = resp.json()
        with self._lock:
            self._cache[key] = (time.time(), payload)
        return payload

    def _mi_index_tables(self, date: str, kind: str) -> List[dict]:
        payload = self._get_json(_TWSE_MI_INDEX, {"date": date, "type": kind, "response": "json"})
        if not isinstance(payload, dict) or payload.get("stat") != "OK":
            return []
        tables = payload.get("tables")
        return [t for t in tables if isinstance(t, dict)] if isinstance(tables, list) else []

    @staticmethod
    def _norm_date(date: Optional[str]) -> str:
        if date:
            text = str(date).replace("-", "").replace("/", "").strip()
            if len(text) == 8 and text.isdigit():
                return text
        return datetime.now().strftime("%Y%m%d")

    # ------------------------------------------------------------ TPEx index
    def get_tpex_index(self) -> Optional[Dict[str, Any]]:
        """Latest 櫃買指數 bar in the get_main_indices item shape, or None."""
        try:
            rows = self._get_json(_TPEX_INDEX)
            if not isinstance(rows, list) or not rows:
                return None
            row = max((r for r in rows if isinstance(r, dict)), key=lambda r: str(r.get("Date", "")))
            close, change = _num(row.get("Close")), _num(row.get("Change"))
            if close is None:
                return None
            change = change or 0.0
            prev = close - change
            open_, high, low = _num(row.get("Open")), _num(row.get("High")), _num(row.get("Low"))
            return {
                "code": "TWOII",
                "name": "台湾柜买指数",
                "current": close,
                "change": round(change, 2),
                "change_pct": round(change / prev * 100, 2) if prev else 0.0,
                "open": open_ or 0.0,
                "high": high or 0.0,
                "low": low or 0.0,
                "prev_close": round(prev, 2),
                "volume": 0.0,
                "amount": 0.0,
                "amplitude": round((high - low) / prev * 100, 2) if (high and low and prev) else 0.0,
                "date": str(row.get("Date", "")),
                "source": "TPEx-OpenAPI",
            }
        except Exception as exc:  # noqa: BLE001 - fail-open
            logger.info("[tw-market] TPEx index fetch failed: %s", exc)
            return None

    # --------------------------------------------------------- TWSE MS table
    def _ms_tables(self, date: Optional[str]) -> Dict[str, dict]:
        out: Dict[str, dict] = {}
        for table in self._mi_index_tables(self._norm_date(date), "MS"):
            title = str(table.get("title") or "")
            if "大盤統計資訊" in title:
                out["summary"] = table
            elif "漲跌證券數" in title:
                out["breadth"] = table
        return out

    def get_twse_total_amount(self, date: Optional[str] = None) -> Optional[float]:
        """上市總成交金額（元）, or None."""
        try:
            summary = self._ms_tables(date).get("summary")
            for row in (summary or {}).get("data", []):
                if row and str(row[0]).startswith("總計"):
                    return _num(row[1])
        except Exception as exc:  # noqa: BLE001
            logger.info("[tw-market] TWSE turnover fetch failed: %s", exc)
        return None

    def get_twse_market_stats(self, date: Optional[str] = None) -> Dict[str, Any]:
        """上市股票漲跌家數 / 漲跌停家數 / 成交金額(億元); {} on failure."""
        try:
            tables = self._ms_tables(date)
            breadth = tables.get("breadth")
            if not breadth:
                return {}
            fields = [str(f) for f in breadth.get("fields") or []]
            col = fields.index("股票") if "股票" in fields else 2
            stats: Dict[str, Any] = {}
            for row in breadth.get("data", []):
                if not row or len(row) <= col:
                    continue
                label = str(row[0])
                total, limit = _count_pair(row[col])
                if total is None:
                    continue
                if label.startswith("上漲"):
                    stats["up_count"], stats["limit_up_count"] = total, limit
                elif label.startswith("下跌"):
                    stats["down_count"], stats["limit_down_count"] = total, limit
                elif label.startswith("持平"):
                    stats["flat_count"] = total
            if "up_count" not in stats or "down_count" not in stats:
                return {}
            stats.setdefault("flat_count", 0)
            amount = self.get_twse_total_amount(date)
            stats["total_amount"] = round(amount / 1e8, 2) if amount else 0.0
            stats["scope"] = "上市股票"
            stats["source"] = "TWSE-MI_INDEX"
            return stats
        except Exception as exc:  # noqa: BLE001
            logger.info("[tw-market] TWSE breadth fetch failed: %s", exc)
            return {}

    # ------------------------------------------------------- TWSE sectors
    def get_twse_sector_rankings(self, n: int = 5, date: Optional[str] = None) -> Tuple[List[Dict], List[Dict]]:
        """(top n, bottom n) 上市類股 by change %, each {'name', 'change_pct'}."""
        try:
            sectors: List[Dict[str, Any]] = []
            for table in self._mi_index_tables(self._norm_date(date), "IND"):
                if "價格指數(臺灣證券交易所)" not in str(table.get("title") or ""):
                    continue
                fields = [str(f) for f in table.get("fields") or []]
                pct_col = next((i for i, f in enumerate(fields) if "百分比" in f), 4)
                for row in table.get("data", []):
                    name = str(row[0]).strip() if row else ""
                    if not name.endswith("類指數") or name in _AGGREGATE_SECTORS:
                        continue
                    pct = _num(row[pct_col]) if len(row) > pct_col else None
                    if pct is None:
                        continue
                    sectors.append({"name": name.replace("類指數", ""), "change_pct": pct})
            if not sectors:
                return [], []
            ordered = sorted(sectors, key=lambda s: s["change_pct"], reverse=True)
            return ordered[:n], list(reversed(ordered[-n:]))
        except Exception as exc:  # noqa: BLE001
            logger.info("[tw-market] TWSE sector fetch failed: %s", exc)
            return [], []


_SHARED: Optional[TwMarketFetcher] = None


def get_tw_market_fetcher() -> TwMarketFetcher:
    global _SHARED
    if _SHARED is None:
        _SHARED = TwMarketFetcher()
    return _SHARED
