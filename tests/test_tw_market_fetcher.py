# -*- coding: utf-8 -*-
"""Offline tests for the TWSE/TPEx market fetcher (payload shapes verified live 2026-11-26)."""

import unittest
from unittest import mock

from data_provider.tw_market_fetcher import TwMarketFetcher, _count_pair

_MS_PAYLOAD = {
    "stat": "OK",
    "tables": [
        {},
        {
            "title": "115年11月26日 大盤統計資訊",
            "fields": ["成交統計", "成交金額(元)", "成交股數(股)", "成交筆數"],
            "data": [
                ["1.一般股票", "734,503,796,285", "4,057,496,026", "2,733,186"],
                ["總計(1~15)", "845,336,632,675", "11,760,283,963", "3,717,271"],
            ],
        },
        {
            "title": "漲跌證券數合計",
            "fields": ["類型", "整體市場", "股票"],
            "data": [
                ["上漲(漲停)", "7,814(262)", "971(57)"],
                ["下跌(跌停)", "1,871(99)", "83(3)"],
                ["持平", "413", "32"],
            ],
        },
    ],
}

_IND_PAYLOAD = {
    "stat": "OK",
    "tables": [
        {
            "title": "115年11月26日 價格指數(臺灣證券交易所)",
            "fields": ["指數", "收盤指數", "漲跌(+/-)", "漲跌點數", "漲跌百分比(%)", "特殊處理註記"],
            "data": [
                ["發行量加權股價指數", "39,151.81", "<p>+</p>", "956.38", "2.50", ""],
                ["電子工業類指數", "2,419.82", "<p>+</p>", "56.48", "2.39", ""],
                ["電子零組件類指數", "619.47", "<p>+</p>", "47.53", "8.31", ""],
                ["半導體類指數", "1,329.89", "<p>+</p>", "9.26", "0.70", ""],
                ["玻璃陶瓷類指數", "138.38", "<p style ='color:green'>-</p>", "0.03", "-0.02", ""],
            ],
        }
    ],
}

_TPEX_INDEX = [
    {"Date": "20261125", "Open": "270", "High": "277", "Low": "269", "Close": "275.97", "Change": "3.1"},
    {"Date": "20261126", "Open": "278.31", "High": "290.64", "Low": "278.31", "Close": "289.84", "Change": "13.87"},
]


def _fake_get(url, params=None):
    if "tpex_index" in url:
        return _TPEX_INDEX
    return _MS_PAYLOAD if (params or {}).get("type") == "MS" else _IND_PAYLOAD


class TwMarketFetcherTest(unittest.TestCase):
    def setUp(self):
        self.fetcher = TwMarketFetcher()
        patcher = mock.patch.object(self.fetcher, "_get_json", side_effect=_fake_get)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_count_pair(self):
        self.assertEqual(_count_pair("971(57)"), (971, 57))
        self.assertEqual(_count_pair("1,871(99)"), (1871, 99))
        self.assertEqual(_count_pair("32"), (32, 0))

    def test_market_stats_use_stock_column(self):
        stats = self.fetcher.get_twse_market_stats("20261126")
        self.assertEqual(
            (stats["up_count"], stats["limit_up_count"], stats["down_count"], stats["limit_down_count"], stats["flat_count"]),
            (971, 57, 83, 3, 32),
        )
        self.assertEqual(stats["total_amount"], 8453.37)

    def test_total_amount(self):
        self.assertEqual(self.fetcher.get_twse_total_amount("20261126"), 845336632675.0)

    def test_sector_rankings_skip_aggregates_and_composites(self):
        top, bottom = self.fetcher.get_twse_sector_rankings(2, "20261126")
        self.assertEqual([s["name"] for s in top], ["電子零組件", "半導體"])
        self.assertEqual(bottom[0], {"name": "玻璃陶瓷", "change_pct": -0.02})
        self.assertNotIn("電子工業", [s["name"] for s in top + bottom])

    def test_tpex_index_uses_latest_row(self):
        item = self.fetcher.get_tpex_index()
        self.assertEqual(item["code"], "TWOII")
        self.assertEqual(item["current"], 289.84)
        self.assertEqual(item["prev_close"], 275.97)
        self.assertEqual(item["change_pct"], 5.03)

    def test_fail_open_on_error(self):
        with mock.patch.object(self.fetcher, "_get_json", side_effect=TimeoutError("slow")):
            self.assertIsNone(self.fetcher.get_tpex_index())
            self.assertEqual(self.fetcher.get_twse_market_stats("20261126"), {})
            self.assertEqual(self.fetcher.get_twse_sector_rankings(5, "20261126"), ([], []))


class SupplementTwIndicesTest(unittest.TestCase):
    def test_fills_twoii_and_twii_amount(self):
        from data_provider.base import DataFetcherManager

        fake = mock.Mock()
        fake.get_tpex_index.return_value = {"code": "TWOII", "current": 289.84}
        fake.get_twse_total_amount.return_value = 845336632675.0
        manager = DataFetcherManager.__new__(DataFetcherManager)
        with mock.patch("data_provider.tw_market_fetcher.get_tw_market_fetcher", return_value=fake):
            items = manager._supplement_tw_indices([{"code": "TWII", "amount": 0}])
        self.assertEqual([i["code"] for i in items], ["TWII", "TWOII"])
        self.assertEqual(items[0]["amount"], 845336632675.0)


if __name__ == "__main__":
    unittest.main()
