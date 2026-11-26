# -*- coding: utf-8 -*-
"""Taiwan stock daily price-limit helpers (漲跌停價依升降單位計算).

TWSE/TPEx limit = previous close ±10%, then snapped to the tick size of the
limit price band: limit-up rounds DOWN, limit-down rounds UP. A close equal to
the computed limit-up price is a genuine 漲停 (not "接近漲停").
"""

from __future__ import annotations

from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal, InvalidOperation
from typing import Any, Dict, Optional

_LIMIT_RATIO = Decimal("0.10")

# (upper bound exclusive, tick) for common stocks.
_STOCK_TICKS = (
    (Decimal("10"), Decimal("0.01")),
    (Decimal("50"), Decimal("0.05")),
    (Decimal("100"), Decimal("0.1")),
    (Decimal("500"), Decimal("0.5")),
    (Decimal("1000"), Decimal("1")),
)
_STOCK_TOP_TICK = Decimal("5")

# ETFs / ETNs (codes starting with 00): 0.01 below 50, 0.05 at or above 50.
_ETF_SPLIT = Decimal("50")


def is_tw_code(code: Any) -> bool:
    upper = str(code or "").strip().upper()
    return upper.endswith(".TW") or upper.endswith(".TWO")


def _is_etf(code: Any) -> bool:
    base = str(code or "").strip().upper().rsplit(".", 1)[0]
    return base.startswith("00")


def tick_size(price: Decimal, *, etf: bool = False) -> Decimal:
    if etf:
        return Decimal("0.01") if price < _ETF_SPLIT else Decimal("0.05")
    for upper, tick in _STOCK_TICKS:
        if price < upper:
            return tick
    return _STOCK_TOP_TICK


def _to_decimal(value: Any) -> Optional[Decimal]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = Decimal(str(value).replace(",", "").strip())
    except (InvalidOperation, ValueError):
        return None
    if not number.is_finite() or number <= 0:
        return None
    return number


def _snap(raw: Decimal, *, etf: bool, rounding: str) -> Decimal:
    tick = tick_size(raw, etf=etf)
    snapped = (raw / tick).to_integral_value(rounding=rounding) * tick
    # Snapping may cross into a lower band with a finer tick; re-snap there once.
    finer = tick_size(snapped, etf=etf)
    if finer != tick:
        snapped = (raw / finer).to_integral_value(rounding=rounding) * finer
    return snapped


def limit_prices(prev_close: Any, code: Any = "") -> Optional[Dict[str, Decimal]]:
    """Return {'limit_up', 'limit_down'} for a TW stock, or None when not computable."""
    prev = _to_decimal(prev_close)
    if prev is None:
        return None
    etf = _is_etf(code)
    up = _snap(prev * (1 + _LIMIT_RATIO), etf=etf, rounding=ROUND_FLOOR)
    down = _snap(prev * (1 - _LIMIT_RATIO), etf=etf, rounding=ROUND_CEILING)
    return {"limit_up": up, "limit_down": down}


def _same_price(a: Optional[Decimal], b: Decimal) -> bool:
    return a is not None and abs(a - b) < Decimal("0.000001")


def limit_status(
    code: Any,
    *,
    prev_close: Any,
    close: Any,
    open_price: Any = None,
    high: Any = None,
    low: Any = None,
) -> Optional[Dict[str, Any]]:
    """Classify today's bar against the daily price limits.

    Returns None for non-TW codes or missing data. ``status`` is one of
    ``limit_up`` / ``one_price_limit_up`` / ``limit_down`` / ``one_price_limit_down`` / ``none``.
    """
    if not is_tw_code(code):
        return None
    limits = limit_prices(prev_close, code)
    close_d = _to_decimal(close)
    if limits is None or close_d is None:
        return None
    open_d, high_d, low_d = _to_decimal(open_price), _to_decimal(high), _to_decimal(low)
    up, down = limits["limit_up"], limits["limit_down"]

    def _one_price(target: Decimal) -> bool:
        return all(_same_price(v, target) for v in (open_d, high_d, low_d, close_d))

    if _same_price(close_d, up):
        status = "one_price_limit_up" if _one_price(up) else "limit_up"
    elif _same_price(close_d, down):
        status = "one_price_limit_down" if _one_price(down) else "limit_down"
    else:
        status = "none"
    return {
        "status": status,
        "prev_close": float(_to_decimal(prev_close)),
        "limit_up": float(up),
        "limit_down": float(down),
        "close": float(close_d),
    }


def compute_tw_limit_status_from_context(context: Any) -> Optional[Dict[str, Any]]:
    """Limit status from an analysis context (today bar + yesterday close / realtime pre_close)."""
    if not isinstance(context, dict):
        return None
    code = context.get("code")
    if not is_tw_code(code):
        return None
    today = context.get("today") if isinstance(context.get("today"), dict) else {}
    yesterday = context.get("yesterday") if isinstance(context.get("yesterday"), dict) else {}
    realtime = context.get("realtime") if isinstance(context.get("realtime"), dict) else {}
    prev_close = realtime.get("pre_close") or yesterday.get("close")
    close = today.get("close") or realtime.get("price")
    return limit_status(
        code,
        prev_close=prev_close,
        close=close,
        open_price=today.get("open"),
        high=today.get("high"),
        low=today.get("low"),
    )


def format_limit_prompt_section(info: Optional[Dict[str, Any]], *, english: bool = False) -> str:
    """Prompt block telling the LLM the computed limit state and how to read volume."""
    if not info:
        return ""
    status = info["status"]
    if english:
        state_text = {
            "one_price_limit_up": "ONE-PRICE LIMIT-UP (open = high = low = close = limit-up price; locked all day)",
            "limit_up": "LIMIT-UP (close equals the limit-up price)",
            "one_price_limit_down": "ONE-PRICE LIMIT-DOWN (locked at the limit-down price all day)",
            "limit_down": "LIMIT-DOWN (close equals the limit-down price)",
            "none": "not at a price limit",
        }[status]
        lines = [
            "",
            "### Daily Price Limit (Taiwan, computed from tick sizes)",
            f"- Previous close {info['prev_close']}, limit-up {info['limit_up']}, limit-down {info['limit_down']}",
            f"- Today's close {info['close']}: **{state_text}**",
        ]
        if status != "none":
            lines.append("- When the close equals a limit price, call it limit-up/limit-down; never write \"near limit-up\".")
        if status in {"one_price_limit_up", "limit_up"}:
            lines.append(
                "- Low volume on a locked limit-up means holders refuse to sell (very light selling pressure); "
                "do NOT describe it as weak momentum or \"rising on shrinking volume\"."
            )
        if status in {"one_price_limit_down", "limit_down"}:
            lines.append(
                "- Low volume on a locked limit-down means buyers are absent; it is not a sign that selling is exhausted."
            )
        return "\n".join(lines) + "\n"

    state_text = {
        "one_price_limit_up": "一字涨停（开盘＝最高＝最低＝收盘＝涨停价，全日锁死）",
        "limit_up": "涨停（收盘价等于涨停价）",
        "one_price_limit_down": "一字跌停（全日锁死在跌停价）",
        "limit_down": "跌停（收盘价等于跌停价）",
        "none": "未触及涨跌停",
    }[status]
    lines = [
        "",
        "### 涨跌停状态（台股，依升降单位计算）",
        f"- 昨收 {info['prev_close']}，涨停价 {info['limit_up']}，跌停价 {info['limit_down']}",
        f"- 今日收盘 {info['close']}：**{state_text}**",
    ]
    if status != "none":
        lines.append("- 收盘价等于涨/跌停价时必须直接写“涨停”/“跌停”，不得写“接近涨停”“逼近涨停”。")
    if status in {"one_price_limit_up", "limit_up"}:
        lines.append("- 涨停锁死时成交量萎缩代表筹码惜售、卖压极轻，不得解读为“缩量上涨”“上攻动能不足”。")
    if status in {"one_price_limit_down", "limit_down"}:
        lines.append("- 跌停锁死时成交量萎缩代表承接缺席，不得解读为“卖压衰竭”。")
    return "\n".join(lines) + "\n"
