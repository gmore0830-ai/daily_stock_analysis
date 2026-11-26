# -*- coding: utf-8 -*-
"""Tests for TW price limits, report-quality guardrails and news citations."""

import unittest
from types import SimpleNamespace

from src.news_registry import NewsRegistry, get_news_registry, render_inline_citations
from src.report_quality_guardrail import (
    annotate_news_citation,
    apply_report_quality_guardrails,
    parse_price,
)
from src.tw_price_limits import (
    compute_tw_limit_status_from_context,
    format_limit_prompt_section,
    limit_prices,
    limit_status,
)


class TwPriceLimitTest(unittest.TestCase):
    def test_limit_prices_follow_tick_bands(self):
        cases = {
            154.5: (169.5, 139.5),   # 169.95 floors to 0.5 tick
            9.5: (10.45, 8.55),
            95: (104.5, 85.5),
            455: (500, 409.5),       # 500.5 floors to 1 tick -> 500
            910: (1000, 819),        # 1001 floors to 5 tick -> 1000
        }
        for prev, (up, down) in cases.items():
            limits = limit_prices(prev, "1234.TW")
            self.assertEqual(float(limits["limit_up"]), up, prev)
            self.assertEqual(float(limits["limit_down"]), down, prev)

    def test_etf_uses_etf_ticks(self):
        limits = limit_prices(48, "0050.TW")
        self.assertEqual(float(limits["limit_up"]), 52.8)

    def test_one_price_limit_up_detected(self):
        info = limit_status("2313.TW", prev_close=154.5, close=169.5, open_price=169.5, high=169.5, low=169.5)
        self.assertEqual(info["status"], "one_price_limit_up")
        section = format_limit_prompt_section(info)
        self.assertIn("一字涨停", section)
        self.assertIn("不得解读为“缩量上涨”", section)

    def test_close_at_limit_up_not_one_price(self):
        info = limit_status("3147.TWO", prev_close=100, close=110, open_price=102, high=110, low=101)
        self.assertEqual(info["status"], "limit_up")

    def test_non_tw_and_missing_data(self):
        self.assertIsNone(limit_status("600519", prev_close=100, close=110))
        self.assertIsNone(limit_status("2330.TW", prev_close=None, close=110))
        self.assertIsNone(compute_tw_limit_status_from_context({"code": "AAPL"}))

    def test_context_prefers_realtime_pre_close(self):
        ctx = {
            "code": "2313.TW",
            "today": {"open": 169.5, "high": 169.5, "low": 169.5, "close": 169.5},
            "yesterday": {"close": 1},
            "realtime": {"pre_close": 154.5},
        }
        self.assertEqual(compute_tw_limit_status_from_context(ctx)["status"], "one_price_limit_up")


def _result(**dashboard):
    return SimpleNamespace(confidence_level="中", report_language="zh", analysis_summary="", dashboard=dashboard)


class ReportQualityGuardrailTest(unittest.TestCase):
    def setUp(self):
        get_news_registry().reset()

    def test_parse_price(self):
        self.assertEqual(parse_price("166.60元（回踩MA5获支撑）"), 166.6)
        self.assertEqual(parse_price("在MA10附近 180.45"), 180.45)
        self.assertEqual(parse_price("1,050元"), 1050.0)
        self.assertIsNone(parse_price("理想买入点：XX元（在MA5附近）"))

    def test_post_close_never_uses_same_day_timing(self):
        result = _result(core_conclusion={"time_sensitivity": "今日内", "one_sentence": "今日内逢低布局"})
        tags = apply_report_quality_guardrails(result, market_phase_summary={"phase": "postmarket"})
        self.assertIn("post_close_timing_to_next_session", tags)
        core = result.dashboard["core_conclusion"]
        self.assertEqual(core["time_sensitivity"], "下一交易日")
        self.assertNotIn("今日内", core["one_sentence"])

    def test_intraday_keeps_same_day_timing(self):
        result = _result(core_conclusion={"time_sensitivity": "今日内"})
        apply_report_quality_guardrails(result, market_phase_summary={"phase": "intraday"})
        self.assertEqual(result.dashboard["core_conclusion"]["time_sensitivity"], "今日内")

    def test_low_confidence_softens_immediate_action(self):
        result = _result(
            core_conclusion={"time_sensitivity": "立即行动", "one_sentence": "立即买入"},
            phase_decision={"immediate_action": "立即行动"},
        )
        result.confidence_level = "低"
        apply_report_quality_guardrails(result, market_phase_summary={"phase": "intraday"})
        self.assertEqual(result.dashboard["core_conclusion"]["time_sensitivity"], "本周内")
        self.assertEqual(result.dashboard["core_conclusion"]["one_sentence"], "确认后再买入")
        self.assertEqual(result.dashboard["phase_decision"]["immediate_action"], "等待确认")

    def test_tight_stop_and_poor_reward_risk_flagged(self):
        result = _result(
            battle_plan={"sniper_points": {"ideal_buy": "100元", "stop_loss": "99元", "take_profit": "100.5元"}},
            intelligence={"risk_alerts": ["外资卖超 [数据]"]},
        )
        apply_report_quality_guardrails(result)
        alerts = result.dashboard["intelligence"]["risk_alerts"]
        self.assertTrue(alerts[0].startswith("⚠️ 停损距买点仅 1.00%"))
        self.assertTrue(alerts[1].startswith("⚠️ 风报比约 1:0.50"))
        self.assertIn("风报比", result.dashboard["battle_plan"]["position_strategy"]["risk_control"])

    def test_healthy_plan_not_flagged(self):
        result = _result(
            battle_plan={"sniper_points": {"ideal_buy": "166.60元", "stop_loss": "160.00元", "take_profit": "180.45元"}}
        )
        self.assertNotIn("stop_or_reward_risk_flagged", apply_report_quality_guardrails(result))

    def test_near_limit_up_wording_corrected(self):
        result = _result(core_conclusion={"one_sentence": "今日接近涨停，量能萎缩"})
        result.analysis_summary = "股价逼近涨停"
        ctx = {"code": "2313.TW", "today": {"close": 169.5}, "yesterday": {"close": 154.5}}
        apply_report_quality_guardrails(result, context=ctx)
        self.assertEqual(result.dashboard["core_conclusion"]["one_sentence"], "今日涨停，量能萎缩")
        self.assertEqual(result.analysis_summary, "股价涨停")


class NewsCitationTest(unittest.TestCase):
    def setUp(self):
        self.registry = NewsRegistry()
        self.registry.register({"title": "台积电重申AI需求", "url": "https://ex.com/a", "published_date": "2026-11-23 08:00"})
        self.registry.register({"title": "无链接新闻", "url": "", "published_date": "2026-11-20"})

    def test_registry_dedupes_by_url(self):
        self.assertEqual(self.registry.register({"title": "dup", "url": "https://ex.com/a"}), "N1")

    def test_citation_uses_event_date_and_link(self):
        out = annotate_news_citation("[2026-11-24] 台积电重申AI需求强劲 [N1]", registry=self.registry)
        self.assertEqual(out, "[2026-11-23] 台积电重申AI需求强劲（[来源](https://ex.com/a)）")

    def test_missing_source_marked_unverified(self):
        self.assertTrue(annotate_news_citation("传闻将有大单", registry=self.registry).endswith("（未证实）"))
        self.assertTrue(annotate_news_citation("消息 [N2]", registry=self.registry).endswith("（未证实）"))

    def test_data_backed_entry(self):
        self.assertEqual(
            annotate_news_citation("外资卖超83万股 [数据]", registry=self.registry), "外资卖超83万股（资料面）"
        )

    def test_inline_citations_in_market_review(self):
        registry = get_news_registry()
        registry.reset()
        registry.register({"title": "a", "url": "https://x/a"})
        self.assertEqual(
            render_inline_citations("台积电带动大盘 [N1]，另有传闻 [N9]。"),
            "台积电带动大盘（[来源](https://x/a)），另有传闻。",
        )


if __name__ == "__main__":
    unittest.main()


class MultiIdCitationTest(unittest.TestCase):
    def test_comma_separated_ids_resolve(self):
        registry = get_news_registry()
        registry.reset()
        registry.register({"title": "a", "url": "https://x/a"})
        registry.register({"title": "b", "url": "https://x/b"})
        self.assertEqual(
            render_inline_citations("大跌后收复 [N1, N2]。另见 [N2]、[N9]。"),
            "大跌后收复（[来源1](https://x/a) [来源2](https://x/b)）。另见（[来源](https://x/b)）。",
        )
