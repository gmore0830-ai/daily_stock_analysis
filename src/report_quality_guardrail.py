# -*- coding: utf-8 -*-
"""Post-LLM report-quality guardrails.

Deterministic fixes applied after the model's dashboard JSON is parsed, because
prompt rules alone are not reliably followed:

* post-close / non-trading output never uses same-day timing ("今日内");
* low-confidence output never carries an immediate / same-day action;
* stop-loss closer than 2% to the entry, or reward:risk below 1:1, is flagged;
* a Taiwan close equal to the computed limit-up price is called 涨停, not 接近涨停.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from src.report_language import normalize_report_language
from src.tw_price_limits import compute_tw_limit_status_from_context

if TYPE_CHECKING:
    from src.analyzer import AnalysisResult

POST_CLOSE_PHASES = {"postmarket", "non_trading"}

_IMMEDIATE_TIMING = {
    "立即行动", "今日内", "立即行動", "今日內", "立即", "今日", "今天", "当日", "當日",
    "act now", "today", "immediately",
}
_LOW_CONFIDENCE = {"低", "low", "낮음"}

_NEXT_SESSION = {"zh": "下一交易日", "en": "Next trading day"}
_THIS_WEEK = {"zh": "本周内", "en": "This week"}
_WAIT_NEXT_SESSION = {"zh": "等待下一交易日确认", "en": "Wait for next-session confirmation"}
_WAIT_CONFIRM = {"zh": "等待确认", "en": "Wait for confirmation"}

# Same-day wording inside free text when output is produced after the close.
_ZH_SAME_DAY_TEXT = (("今日内", "下一交易日"), ("今天内", "下一交易日"), ("今日盘中", "下一交易日盘中"))

# Strong immediate wording softened when confidence is low.
_ZH_LOW_CONFIDENCE_SOFTEN = (
    ("立即买入", "确认后再买入"),
    ("立即加仓", "确认后再加仓"),
    ("立即卖出", "确认后再卖出"),
    ("立即清仓", "确认后再减仓"),
    ("立即减仓", "确认后再减仓"),
    ("果断买入", "确认后再买入"),
    ("果断卖出", "确认后再卖出"),
    ("果断清仓", "确认后再减仓"),
    ("立即行动", "等待确认"),
)

_NEAR_LIMIT_UP = ("接近涨停", "逼近涨停", "接近漲停", "逼近漲停", "近乎涨停", "几近涨停")
_NEAR_LIMIT_DOWN = ("接近跌停", "逼近跌停", "接近跌停", "逼近跌停")

_PRICE_WITH_UNIT = re.compile(r"(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:元|新台币|新臺幣|NT\$?)")
_BARE_NUMBER = re.compile(r"(?<![A-Za-z])(\d+(?:,\d{3})*(?:\.\d+)?)(?!\s*[%％日])")

MIN_STOP_DISTANCE_PCT = 2.0
MIN_REWARD_RISK = 1.0


def _lang(language: Any) -> str:
    return "zh" if normalize_report_language(language) == "zh" else "en"


def _text(value: Any) -> str:
    return str(value).strip() if isinstance(value, (str, int, float)) else ""


def _dict(parent: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        value = {}
        parent[key] = value
    return value


def _map_strings(value: Any, fn: Callable[[str], str]) -> Any:
    if isinstance(value, str):
        return fn(value)
    if isinstance(value, list):
        return [_map_strings(item, fn) for item in value]
    if isinstance(value, dict):
        return {key: _map_strings(item, fn) for key, item in value.items()}
    return value


def _replace_all(text: str, pairs) -> str:
    for old, new in pairs:
        text = text.replace(old, new)
    return text


def parse_price(value: Any) -> Optional[float]:
    """First price in a sniper-point string such as '166.60元（回踩MA5）'."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value) if value > 0 else None
    text = _text(value)
    if not text:
        return None
    match = _PRICE_WITH_UNIT.search(text)
    if match is None:
        # Strip indicator names (MA5, MA20...) before falling back to a bare number.
        match = _BARE_NUMBER.search(re.sub(r"[A-Za-z]+\d+", " ", text))
    if match is None:
        return None
    try:
        number = float(match.group(1).replace(",", ""))
    except ValueError:
        return None
    return number if number > 0 else None


def _immediate(value: Any) -> bool:
    return _text(value).lower() in _IMMEDIATE_TIMING


def _phase_of(market_phase_summary: Any) -> str:
    if isinstance(market_phase_summary, dict):
        return _text(market_phase_summary.get("phase")).lower()
    return ""


def apply_report_quality_guardrails(
    result: "AnalysisResult",
    *,
    context: Optional[Dict[str, Any]] = None,
    market_phase_summary: Optional[Dict[str, Any]] = None,
    report_language: str = "zh",
) -> List[str]:
    """Apply deterministic report-quality fixes to ``result`` in place; return adjustment tags."""
    if result is None:
        return []
    lang = _lang(report_language or getattr(result, "report_language", "zh"))
    dashboard = getattr(result, "dashboard", None)
    if not isinstance(dashboard, dict):
        dashboard = {}
        result.dashboard = dashboard
    adjustments: List[str] = []

    phase = _phase_of(market_phase_summary)
    if phase in POST_CLOSE_PHASES and _fix_post_close_timing(dashboard, lang):
        adjustments.append("post_close_timing_to_next_session")

    if _text(getattr(result, "confidence_level", "")).lower() in _LOW_CONFIDENCE:
        if _fix_low_confidence_actions(dashboard, lang):
            adjustments.append("low_confidence_action_softened")

    tag = _flag_stop_and_reward_risk(dashboard, lang)
    if tag:
        adjustments.append(tag)

    if _fix_limit_wording(result, context):
        adjustments.append("limit_wording_corrected")

    if apply_news_citations(result, lang):
        adjustments.append("news_citations_resolved")
    return adjustments


# ---------------------------------------------------------------- timing rules
def _fix_post_close_timing(dashboard: Dict[str, Any], lang: str) -> bool:
    changed = False
    core = _dict(dashboard, "core_conclusion")
    if _immediate(core.get("time_sensitivity")):
        core["time_sensitivity"] = _NEXT_SESSION[lang]
        changed = True
    phase_decision = dashboard.get("phase_decision")
    if isinstance(phase_decision, dict) and _text(phase_decision.get("immediate_action")) in {"立即行动", "立即行動", "Act now"}:
        phase_decision["immediate_action"] = _WAIT_NEXT_SESSION[lang]
        changed = True
    if lang == "zh":
        before = {k: core.get(k) for k in ("one_sentence", "position_advice")}
        for key in before:
            if key in core:
                core[key] = _map_strings(core[key], lambda s: _replace_all(s, _ZH_SAME_DAY_TEXT))
        changed = changed or any(core.get(k) != v for k, v in before.items())
    return changed


def _fix_low_confidence_actions(dashboard: Dict[str, Any], lang: str) -> bool:
    changed = False
    core = _dict(dashboard, "core_conclusion")
    timing = _text(core.get("time_sensitivity"))
    if _immediate(timing) or timing in {_NEXT_SESSION["zh"], _NEXT_SESSION["en"]}:
        core["time_sensitivity"] = _THIS_WEEK[lang]
        changed = True
    phase_decision = dashboard.get("phase_decision")
    if isinstance(phase_decision, dict) and _text(phase_decision.get("immediate_action")) in {"立即行动", "立即行動", "Act now"}:
        phase_decision["immediate_action"] = _WAIT_CONFIRM[lang]
        changed = True
    if lang == "zh":
        for key in ("one_sentence", "position_advice"):
            if key in core:
                updated = _map_strings(core[key], lambda s: _replace_all(s, _ZH_LOW_CONFIDENCE_SOFTEN))
                if updated != core[key]:
                    core[key] = updated
                    changed = True
    return changed


# ------------------------------------------------------------ stop / reward-risk
def _flag_stop_and_reward_risk(dashboard: Dict[str, Any], lang: str) -> Optional[str]:
    battle = dashboard.get("battle_plan")
    if not isinstance(battle, dict):
        return None
    points = battle.get("sniper_points")
    if not isinstance(points, dict):
        return None
    entry = parse_price(points.get("ideal_buy")) or parse_price(points.get("secondary_buy"))
    stop = parse_price(points.get("stop_loss"))
    target = parse_price(points.get("take_profit"))
    if entry is None or stop is None or stop >= entry:
        return None  # no long setup to evaluate (or sell-side plan)

    warnings: List[str] = []
    stop_pct = (entry - stop) / entry * 100
    if stop_pct < MIN_STOP_DISTANCE_PCT:
        warnings.append(
            f"⚠️ 停损距买点仅 {stop_pct:.2f}%（低于 {MIN_STOP_DISTANCE_PCT:.0f}%），正常波动就可能触发停损"
            if lang == "zh"
            else f"⚠️ Stop is only {stop_pct:.2f}% below entry (< {MIN_STOP_DISTANCE_PCT:.0f}%); normal noise may trigger it"
        )
    if target is not None and target > entry:
        reward_risk = (target - entry) / (entry - stop)
        if reward_risk < MIN_REWARD_RISK:
            warnings.append(
                f"⚠️ 风报比约 1:{reward_risk:.2f}（低于 1:1），潜在获利小于可能亏损"
                if lang == "zh"
                else f"⚠️ Reward:risk is about 1:{reward_risk:.2f} (below 1:1)"
            )
    elif target is not None and target <= entry:
        warnings.append(
            "⚠️ 目标价不高于买点，此买入计划没有获利空间"
            if lang == "zh"
            else "⚠️ Target is not above the entry; this buy plan has no upside"
        )
    if not warnings:
        return None

    intelligence = _dict(dashboard, "intelligence")
    alerts = intelligence.get("risk_alerts")
    alerts = list(alerts) if isinstance(alerts, list) else ([alerts] if _text(alerts) else [])
    for warning in reversed(warnings):
        if warning not in alerts:
            alerts.insert(0, warning)
    intelligence["risk_alerts"] = alerts

    strategy = _dict(battle, "position_strategy")
    risk_control = _text(strategy.get("risk_control")).rstrip("。；;. ")
    joined = "；".join(w.lstrip("⚠️ ").strip() for w in warnings) if lang == "zh" else "; ".join(warnings)
    if joined not in risk_control:
        strategy["risk_control"] = f"{risk_control}{'；' if risk_control else ''}⚠️ {joined}" if lang == "zh" else (
            f"{risk_control}{'; ' if risk_control else ''}{joined}"
        )
    return "stop_or_reward_risk_flagged"


# ------------------------------------------------------------ news citations
_DATA_TAGS = ("[数据]", "[數據]", "[资料]", "[資料]", "[DATA]", "[Data]", "[data]")
_LEADING_DATE = re.compile(r"^\s*[\[【(（]?\s*(\d{4}[-/.]\d{1,2}[-/.]\d{1,2})\s*[\]】)）]?\s*[:：]?\s*")
_OWN_WARNING_PREFIXES = ("⚠️ 停损距", "⚠️ 风报比", "⚠️ 目标价", "⚠️ Stop is only", "⚠️ Reward:risk", "⚠️ Target is not")


def annotate_news_citation(text: str, lang: str = "zh", registry: Any = None) -> str:
    """Resolve [Nx] ids into source links + event dates; mark unsourced items unverified."""
    from src.news_registry import CITATION_PATTERN, citation_ids, get_news_registry, normalize_news_date

    if not isinstance(text, str) or not text.strip() or text.startswith(_OWN_WARNING_PREFIXES):
        return text
    registry = registry or get_news_registry()
    body = text
    data_backed = any(tag in body for tag in _DATA_TAGS)
    for tag in _DATA_TAGS:
        body = body.replace(tag, "")
    ids = citation_ids(body)
    body = CITATION_PATTERN.sub("", body)
    body = re.sub(r"\s+([，。；,.;)）])", r"\1", body).rstrip(" ，,;；")
    items = [item for item in (registry.get(i) for i in ids) if item is not None]

    # Event date = the cited item's own publication date, not the report date.
    dated = next((item.published_date for item in items if item.published_date), "")
    if dated:
        match = _LEADING_DATE.match(body)
        if match:
            if normalize_news_date(match.group(1)) != dated:
                body = f"[{dated}] " + body[match.end():]
        else:
            body = f"[{dated}] {body}"

    links = [item for item in items if item.url]
    if links:
        label = "来源" if lang == "zh" else "Source"
        parts = [
            f"[{label}{idx if len(links) > 1 else ''}]({item.url})" for idx, item in enumerate(links, 1)
        ]
        return f"{body}（{' '.join(parts)}）" if lang == "zh" else f"{body} ({' '.join(parts)})"
    if data_backed:
        return f"{body}（资料面）" if lang == "zh" else f"{body} (data)"
    return f"{body}（未证实）" if lang == "zh" else f"{body} (unverified)"


def apply_news_citations(result: "AnalysisResult", report_language: str = "zh") -> bool:
    lang = _lang(report_language or getattr(result, "report_language", "zh"))
    dashboard = getattr(result, "dashboard", None)
    if not isinstance(dashboard, dict):
        return False
    intelligence = dashboard.get("intelligence")
    if not isinstance(intelligence, dict):
        return False
    changed = False
    for key in ("risk_alerts", "positive_catalysts"):
        entries = intelligence.get(key)
        if isinstance(entries, list):
            updated = [annotate_news_citation(e, lang) if isinstance(e, str) else e for e in entries]
            if updated != entries:
                intelligence[key] = updated
                changed = True
    latest = intelligence.get("latest_news")
    if isinstance(latest, str) and latest.strip() and latest.strip() not in {"无", "無", "N/A", "None"}:
        updated = annotate_news_citation(latest, lang)
        if updated != latest:
            intelligence["latest_news"] = updated
            changed = True
    return changed


# ------------------------------------------------------------- limit wording
_RESULT_TEXT_FIELDS = (
    "analysis_summary", "key_points", "risk_warning", "buy_reason", "trend_analysis",
    "short_term_outlook", "medium_term_outlook", "technical_analysis", "volume_analysis",
    "pattern_analysis", "news_summary", "market_sentiment",
)


def _fix_limit_wording(result: "AnalysisResult", context: Optional[Dict[str, Any]]) -> bool:
    info = compute_tw_limit_status_from_context(context)
    if not info or info["status"] == "none":
        return False
    if info["status"] in {"limit_up", "one_price_limit_up"}:
        pairs = [(phrase, "涨停" if "涨" in phrase else "漲停") for phrase in _NEAR_LIMIT_UP]
    else:
        pairs = [(phrase, "跌停") for phrase in _NEAR_LIMIT_DOWN]

    def fix(text: str) -> str:
        return _replace_all(text, pairs)

    changed = False
    dashboard = getattr(result, "dashboard", None)
    if isinstance(dashboard, dict):
        updated = _map_strings(dashboard, fix)
        if updated != dashboard:
            dashboard.clear()
            dashboard.update(updated)
            changed = True
    for field in _RESULT_TEXT_FIELDS:
        value = getattr(result, field, None)
        if isinstance(value, str):
            fixed = fix(value)
            if fixed != value:
                setattr(result, field, fixed)
                changed = True
    return changed
