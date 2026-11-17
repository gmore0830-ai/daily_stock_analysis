# -*- coding: utf-8 -*-
"""Multi-region final review reuses per-region context reviews from the same run."""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from tests.litellm_stub import ensure_litellm_stub

ensure_litellm_stub()

from tests.test_market_review import _build_optional_module_stubs  # noqa: E402

sys.modules.update(_build_optional_module_stubs())
import src.core.market_review as market_review_module  # noqa: E402

run_market_review = market_review_module.run_market_review


def _review(region: str, body: str, *, template_fallback: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        report=body,
        market_light_snapshot={"region": region, "trade_date": "2026-11-17", "score": 50},
        template_fallback=template_fallback,
    )


class RunScopedMarketReviewReuseTestCase(unittest.TestCase):
    def setUp(self) -> None:
        market_review_module._clear_run_scoped_reviews()
        self.addCleanup(market_review_module._clear_run_scoped_reviews)
        self.notifier = MagicMock()
        self.notifier.save_report_to_file.return_value = "/tmp/market_review.md"
        self.notifier.is_available.return_value = False

    def _run(self, *, region: str, query_id: str, analyzers: dict) -> str:
        def _factory(**kwargs):
            return analyzers[kwargs["region"]]

        config = SimpleNamespace(report_language="zh", market_review_region=region)
        with patch.object(market_review_module, "get_config", return_value=config), patch.object(
            market_review_module, "MarketAnalyzer", side_effect=_factory
        ), patch.object(market_review_module, "_persist_market_review_history"):
            return run_market_review(
                self.notifier,
                send_notification=False,
                override_region=region,
                query_id=query_id,
                save_report_file=False,
            )

    @staticmethod
    def _analyzer(review: SimpleNamespace) -> MagicMock:
        analyzer = MagicMock()
        analyzer.run_daily_review_with_snapshot.return_value = review
        return analyzer

    def test_final_multi_region_review_reuses_context_reviews(self) -> None:
        context_tw = self._analyzer(_review("tw", "TW context body"))
        context_us = self._analyzer(_review("us", "US context body"))
        self._run(region="tw", query_id="market_context_run1_tw", analyzers={"tw": context_tw})
        self._run(region="us", query_id="market_context_run1_us", analyzers={"us": context_us})

        final_tw = self._analyzer(_review("tw", "TW regenerated"))
        final_us = self._analyzer(_review("us", "US regenerated"))
        report = self._run(region="tw,us", query_id="run1", analyzers={"tw": final_tw, "us": final_us})

        final_tw.run_daily_review_with_snapshot.assert_not_called()
        final_us.run_daily_review_with_snapshot.assert_not_called()
        self.assertIn("TW context body", report)
        self.assertIn("US context body", report)
        self.assertNotIn("regenerated", report)

    def test_other_runs_and_template_fallbacks_are_not_reused(self) -> None:
        self._run(
            region="tw",
            query_id="market_context_run1_tw",
            analyzers={"tw": self._analyzer(_review("tw", "TW run1 body"))},
        )
        self._run(
            region="us",
            query_id="market_context_run2_us",
            analyzers={"us": self._analyzer(_review("us", "US template", template_fallback=True))},
        )

        final_tw = self._analyzer(_review("tw", "TW regenerated"))
        final_us = self._analyzer(_review("us", "US regenerated"))
        report = self._run(region="tw,us", query_id="run2", analyzers={"tw": final_tw, "us": final_us})

        final_tw.run_daily_review_with_snapshot.assert_called_once()
        final_us.run_daily_review_with_snapshot.assert_called_once()
        self.assertIn("TW regenerated", report)
        self.assertIn("US regenerated", report)

    def test_context_review_is_consumed_once(self) -> None:
        self._run(
            region="tw",
            query_id="market_context_run3_tw",
            analyzers={"tw": self._analyzer(_review("tw", "TW context body"))},
        )
        us = self._analyzer(_review("us", "US body"))
        self._run(region="tw,us", query_id="run3", analyzers={"tw": MagicMock(), "us": us})

        second_tw = self._analyzer(_review("tw", "TW second run"))
        report = self._run(region="tw,us", query_id="run3", analyzers={"tw": second_tw, "us": us})
        second_tw.run_daily_review_with_snapshot.assert_called_once()
        self.assertIn("TW second run", report)

    def test_cache_is_bounded(self) -> None:
        limit = market_review_module._RUN_SCOPED_REVIEW_MAX_ENTRIES
        for index in range(limit + 5):
            market_review_module._remember_run_scoped_review(
                f"market_context_r{index}_tw", "tw", _review("tw", f"body {index}")
            )
        self.assertEqual(len(market_review_module._RUN_SCOPED_REVIEWS), limit)
        self.assertIsNone(market_review_module._take_run_scoped_review("r0", "tw"))
        self.assertIsNotNone(market_review_module._take_run_scoped_review(f"r{limit + 4}", "tw"))


if __name__ == "__main__":
    unittest.main()
