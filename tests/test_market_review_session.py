"""Market review data-date contracts: holiday / stale detection, prompt dates,
next-session labels, excluded news sources and Taiwan institutional totals."""

from datetime import datetime
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from src.config import Config, parse_news_excluded_sources
from src.core import trading_calendar
from src.core.trading_calendar import (
    build_review_session_context,
    format_session_label,
    normalize_bar_date,
)
from src.market_analyzer import MarketAnalyzer, MarketIndex, MarketOverview
from src.search_service import is_excluded_news_source

TAIPEI = ZoneInfo("Asia/Taipei")
needs_calendar = pytest.mark.skipif(
    not trading_calendar._XCALS_AVAILABLE, reason="exchange-calendars not installed"
)


def _us_index(code, data_date, change_pct=-0.5):
    return MarketIndex(
        code=code, name=code, current=100.0, change=-0.5, change_pct=change_pct,
        prev_close=100.5, data_date=data_date,
    )


def _analyzer(region="us", llm=None, language="zh"):
    with patch("src.market_analyzer.DataFetcherManager"):
        return MarketAnalyzer(region=region, config=Config(report_language=language), analyzer=llm)


def _closed_overview():
    return MarketOverview(
        date="2026-11-27",
        indices=[_us_index(c, "2026-11-25") for c in ("SPX", "IXIC", "DJI", "VIX")],
        data_date="2026-11-25",
        session={
            "market": "us",
            "data_date": "2026-11-25",
            "expected_date": "2026-11-25",
            "status": "closed",
            "closed_days": [{"date": "2026-11-26", "weekend": False, "name": "感恩节"}],
            "next_session": "2026-11-27",
            "next_session_early_close": True,
        },
    )


# --------------------------------------------------------------- calendar


def test_normalize_bar_date_accepts_iso_compact_and_minguo():
    assert normalize_bar_date("2026-11-25") == "2026-11-25"
    assert normalize_bar_date("20261125") == "2026-11-25"
    assert normalize_bar_date("1151125") == "2026-11-25"
    assert normalize_bar_date("") == ""
    assert normalize_bar_date("bad") == ""
    assert format_session_label("2026-11-27") == "11/27（五）"


@needs_calendar
@pytest.mark.parametrize("run_at", ["2026-11-27 08:10", "2026-11-27 16:45"])
def test_thanksgiving_is_closed_with_next_half_day(run_at):
    now = datetime.fromisoformat(run_at).replace(tzinfo=TAIPEI)
    ctx = build_review_session_context("us", "2026-11-25", now)
    assert ctx["status"] == "closed"
    assert ctx["expected_date"] == "2026-11-25"
    assert ctx["closed_days"] == [{"date": "2026-11-26", "weekend": False, "name": "感恩节"}]
    assert ctx["next_session"] == "2026-11-27"
    assert ctx["next_session_early_close"] is True


@needs_calendar
def test_regular_day_is_normal_and_old_data_is_stale():
    now = datetime(2026, 11, 25, 16, 45, tzinfo=TAIPEI)
    assert build_review_session_context("us", "2026-11-24", now)["status"] == "normal"
    late = datetime(2026, 11, 27, 16, 45, tzinfo=TAIPEI)
    assert build_review_session_context("us", "2026-11-24", late)["status"] == "stale"


@needs_calendar
def test_weekend_is_closed_and_unknown_without_data_date():
    monday = datetime(2026, 11, 30, 8, 10, tzinfo=TAIPEI)
    ctx = build_review_session_context("us", "2026-11-27", monday)
    assert ctx["status"] == "closed"
    assert all(day["weekend"] for day in ctx["closed_days"])
    assert build_review_session_context("us", "", monday)["status"] == "unknown"


# --------------------------------------------------------------- analyzer


def test_overview_data_date_ignores_lagging_vix():
    indices = [_us_index("SPX", "2026-11-25"), _us_index("VIX", "2026-11-24")]
    assert MarketAnalyzer._resolve_overview_data_date(indices) == "2026-11-25"
    assert MarketAnalyzer._resolve_overview_data_date([_us_index("VIX", "2026-11-24")]) == "2026-11-24"


def test_closed_prompt_states_data_date_holiday_and_next_session():
    prompt = _analyzer()._build_review_prompt(_closed_overview(), [])
    assert "资料日期：2026-11-25" in prompt
    assert "11/26（四）感恩节休市，以下为 11/25（三）收盘回顾" in prompt
    assert "11/27（五）美股半日盘" in prompt
    assert "# 今日市场数据" not in prompt
    assert "明日交易计划" not in prompt


def test_closed_review_has_no_today_wording_and_starts_with_note():
    llm_text = (
        "## 2026-11-27 美股大盘复盘\n\n### 一、盘面总览\n今日美股小幅回落，今天科技股承压。\n\n"
        "### 二、指数结构\n三大指数同步回落。\n"
    )
    llm = MagicMock(generate_text=MagicMock(return_value=llm_text))
    analyzer = _analyzer(llm=llm)
    report = analyzer.generate_market_review(_closed_overview(), [])
    assert "今日" not in report and "今天" not in report
    assert report.startswith("## 美股收盘回顾｜11/25（三）收盘，11/26（四）感恩节休市")
    assert "> 11/26（四）感恩节休市，以下为 11/25（三）收盘回顾" in report
    assert "11/25（三）美股小幅回落" in report
    assert "资料日期：2026-11-25" in report


def test_closed_template_title_never_says_today():
    report = _analyzer()._apply_session_wording(
        _analyzer()._generate_template_review(_closed_overview(), []), _closed_overview()
    )
    first_line = report.splitlines()[0]
    assert first_line.startswith("## 美股收盘回顾｜11/25（三）收盘，11/26（四）感恩节休市")
    assert "今日" not in report


def test_stale_data_skips_the_llm():
    llm = MagicMock(generate_text=MagicMock(return_value="should not be used"))
    analyzer = _analyzer(llm=llm)
    overview = _closed_overview()
    overview.session = {**overview.session, "status": "stale", "data_date": "2026-11-24"}
    report = analyzer.generate_market_review(overview, [])
    assert "资料未更新" in report
    assert "2026-11-24" in report and "2026-11-25" in report
    llm.generate_text.assert_not_called()


def test_normal_session_keeps_legacy_headings():
    overview = _closed_overview()
    overview.session = {**overview.session, "status": "normal", "closed_days": [], "next_session_early_close": False}
    prompt = _analyzer()._build_review_prompt(overview, [])
    assert "休市" not in prompt
    assert "交易计划｜11/27（五）美股" in prompt


# --------------------------------------------------------------- news


def test_excluded_news_sources_match_host_and_path():
    patterns = parse_news_excluded_sources(None)
    assert is_excluded_news_source("https://www.cmoney.tw/forum/stock/2330", patterns)
    assert not is_excluded_news_source("https://www.cmoney.tw/notes/123", patterns)
    assert is_excluded_news_source("https://m.youtube.com/watch?v=x", patterns)
    assert is_excluded_news_source("https://youtu.be/x", patterns)
    assert not is_excluded_news_source("https://news.cnyes.com/news/id/1", patterns)
    assert parse_news_excluded_sources("") == []
    assert parse_news_excluded_sources(" https://www.Example.com/Forum/ ,") == ["example.com/forum"]


# --------------------------------------------------------------- institutional


def test_tw_institutional_summary_parses_twse_and_tpex():
    from data_provider.tw_market_fetcher import TwMarketFetcher

    twse = {
        "stat": "OK", "date": "20261126",
        "data": [
            ["自營商(自行買賣)", "1", "1", "4,003,444,121"],
            ["自營商(避險)", "1", "1", "8,861,705,660"],
            ["投信", "1", "1", "21,213,076,619"],
            ["外資及陸資(不含外資自營商)", "1", "1", "-49,181,133,625"],
            ["外資自營商", "0", "0", "0"],
            ["合計", "1", "1", "-15,102,907,225"],
        ],
    }
    tpex = {"tables": [{"date": "115/11/26", "data": [
        ["外資及陸資合計", "1", "1", "-3,451,569,107"],
        ["投信", "1", "1", "1,132,267,886"],
        ["自營商合計", "1", "1", "349,446,034"],
        ["三大法人合計*", "1", "1", "-1,969,855,187"],
    ]}]}
    fetcher = TwMarketFetcher()
    with patch.object(fetcher, "_get_json", side_effect=lambda url, params=None: twse if "BFI82U" in url else tpex):
        summary = fetcher.get_institutional_summary("20261126")
    assert summary["date"] == "2026-11-26"
    assert summary["twse"]["total"] == -15102907225.0
    assert summary["twse"]["dealer"] == 4003444121.0 + 8861705660.0
    assert summary["tpex"]["foreign"] == -3451569107.0

    overview = MarketOverview(date="2026-11-26", institutional=summary)
    block = _analyzer(region="tw")._build_institutional_block(overview)
    assert "| 上市 | -491.81 | +212.13 | +128.65 | -151.03 |" in block
    assert "| 上柜 | -34.52 | +11.32 | +3.49 | -19.70 |" in block


def test_tw_institutional_rejects_other_dates():
    from data_provider.tw_market_fetcher import TwMarketFetcher

    fetcher = TwMarketFetcher()
    stale = {"stat": "OK", "date": "20261125", "data": [["合計", "1", "1", "5"]]}
    with patch.object(fetcher, "_get_json", return_value=stale):
        assert fetcher._twse_institutional("20261126") == {}


@needs_calendar
def test_partial_bar_before_close_is_intraday():
    # 10:17 ET on the 11/27 half-day: the 11/27 bar is not a completed session yet.
    now = datetime(2026, 11, 27, 23, 17, tzinfo=TAIPEI)
    ctx = build_review_session_context("us", "2026-11-27", now)
    assert ctx["expected_date"] == "2026-11-25"
    assert ctx["status"] == "intraday"


def test_intraday_prompt_forbids_closing_recap():
    overview = _closed_overview()
    overview.session = {**overview.session, "status": "intraday", "data_date": "2026-11-27", "closed_days": []}
    overview.data_date = "2026-11-27"
    prompt = _analyzer()._build_review_prompt(overview, [])
    assert "盘中数据" in prompt
    assert "休市" not in prompt


# --------------------------------------------------------------- TAIEX fallback


def _taiex_payload(rows):
    return {"stat": "OK", "data": rows}


def test_official_taiex_bar_from_twse_history():
    from data_provider.tw_market_fetcher import TwMarketFetcher

    fetcher = TwMarketFetcher()
    payload = _taiex_payload([
        ["115/11/26", "38,435.99", "39,459.95", "38,435.99", "39,151.81"],
        ["115/11/27", "39,168.57", "39,495.41", "38,890.53", "39,308.08"],
    ])
    with patch.object(fetcher, "_get_json", return_value=payload):
        bar = fetcher.get_twse_taiex("20261127")
    assert bar["date"] == "2026-11-27"
    assert bar["current"] == 39308.08 and bar["prev_close"] == 39151.81
    assert bar["change_pct"] == 0.4


def test_official_taiex_uses_previous_month_for_first_session():
    from data_provider.tw_market_fetcher import TwMarketFetcher

    fetcher = TwMarketFetcher()
    october = _taiex_payload([["115/10/30", "1", "1", "1", "45,211.83"]])
    november = _taiex_payload([["115/11/02", "45,034.53", "45,156.17", "44,399.10", "44,920.14"]])
    with patch.object(fetcher, "_get_json", side_effect=lambda url, params: november if params["date"] == "20261102" else october):
        bar = fetcher.get_twse_taiex("20261102")
    assert bar["prev_close"] == 45211.83
    assert bar["change_pct"] == -0.65


@pytest.mark.parametrize(
    "yf_date,expect_official",
    [("2026-11-26", True), ("2026-11-27", False), (None, True)],
)
def test_lagging_yfinance_taiex_is_replaced(yf_date, expect_official):
    from data_provider.base import DataFetcherManager

    official = {"code": "TWII", "current": 39308.08, "date": "2026-11-27"}
    tw = MagicMock(get_twse_taiex=MagicMock(return_value=official))
    items = [{"code": "TWOII", "current": 285.77, "date": "1151127"}]
    if yf_date:
        items.insert(0, {"code": "TWII", "current": 39151.81, "date": yf_date})
    result = DataFetcherManager._prefer_official_taiex(tw, items)
    twii = next(i for i in result if i["code"] == "TWII")
    assert (twii is official) == expect_official
    assert [i["code"] for i in result].count("TWII") == 1


def test_index_table_labels_lagging_index():
    overview = MarketOverview(
        date="2026-11-28",
        indices=[_us_index("TWII", "2026-11-26"), _us_index("TWOII", "2026-11-27")],
        data_date="2026-11-27",
        session={"data_date": "2026-11-27", "status": "closed"},
    )
    block = _analyzer(region="tw")._build_indices_block(overview)
    assert "TWII（资料日期 2026-11-26）" in block
    assert "TWOII（资料日期" not in block


# --------------------------------------------------------------- closed-today markets


@needs_calendar
@pytest.mark.real_review_freshness
@pytest.mark.parametrize(
    "run_at,expect_tw,expect_us",
    [
        ("2026-11-26 16:45", True, True),    # US Thanksgiving: 11/25 close still owed
        ("2026-11-28 08:20", False, True),   # Saturday: Friday TW close already reviewed
        ("2026-10-09 16:45", False, True),   # TW National Day: TW closed today
        ("2026-12-25 16:45", False, True),   # Christmas: US 12/24 half-day close still owed
    ],
)
def test_review_markets_keep_unreviewed_recent_sessions(run_at, expect_tw, expect_us):
    from src.core.trading_calendar import get_market_review_markets_today

    now = datetime.fromisoformat(run_at).replace(tzinfo=TAIPEI)
    markets = get_market_review_markets_today(now)
    assert ("tw" in markets) == expect_tw
    assert ("us" in markets) == expect_us


@needs_calendar
def test_closed_today_note_names_holiday_or_weekend():
    from src.core.trading_calendar import build_market_closed_today_note

    assert build_market_closed_today_note("tw", datetime(2026, 10, 9, 16, 45, tzinfo=TAIPEI)) == (
        "台股 10/9（五）国庆日休市，本次无新交易资料"
    )
    assert build_market_closed_today_note("tw", datetime(2026, 11, 28, 8, 20, tzinfo=TAIPEI)) == (
        "台股 11/28（六）周末休市，本次无新交易资料"
    )


def test_main_records_notes_for_dropped_markets():
    import main as main_module
    from src.core import market_review

    config = Config(market_review_region="tw,us")
    with patch("src.core.trading_calendar.get_market_review_markets_today", return_value={"us"}), patch(
        "src.core.trading_calendar.build_market_closed_today_note", side_effect=lambda m: f"{m} closed"
    ):
        region = main_module._resolve_review_region_with_closed_notes(config)
    assert region == "us"
    assert market_review._CLOSED_MARKET_NOTES == {"tw": "tw closed"}


def test_single_market_review_still_states_closed_market():
    from types import SimpleNamespace

    from src.core import market_review

    notifier = MagicMock()
    notifier.save_report_to_file.return_value = "/tmp/market_review.md"
    us = MagicMock()
    us.run_daily_review_with_snapshot.return_value = SimpleNamespace(report="US body", market_light_snapshot=None)
    market_review.set_closed_market_notes({"tw": "台股 10/9（五）国庆日休市，本次无新交易资料"})
    with patch.object(
        market_review, "get_config", return_value=SimpleNamespace(report_language="zh", market_review_region="tw,us")
    ), patch.object(market_review, "MarketAnalyzer", return_value=us), patch.object(
        market_review, "_persist_market_review_history"
    ):
        result = market_review.run_market_review(notifier, send_notification=False, override_region="us")
    assert "# 美股大盘复盘\n\nUS body" in result
    assert "# 台股大盘复盘\n\n> 📌 台股 10/9（五）国庆日休市，本次无新交易资料" in result
    saved = notifier.save_report_to_file.call_args.args[0]
    assert "国庆日休市" in saved


# --------------------------------------------------------------- titles & sources


def test_normal_and_intraday_titles_use_data_session():
    overview = _closed_overview()
    overview.session = {**overview.session, "status": "normal", "closed_days": [], "data_date": "2026-11-27"}
    analyzer = _analyzer()
    assert analyzer._get_session_review_title(overview) == "## 11/27（五）美股大盘复盘"
    fixed = analyzer._apply_session_wording("## 📊 2026-11-28 大盘复盘\n\n正文", overview)
    assert fixed.startswith("## 11/27（五）美股大盘复盘\n")
    overview.session["status"] = "intraday"
    assert analyzer._get_session_review_title(overview) == "## 11/27（五）美股盘中复盘"
    overview.session["status"] = "stale"
    assert analyzer._get_session_review_title(overview) == analyzer._get_review_title(overview.date)


def test_google_finance_quote_pages_are_excluded_by_default():
    patterns = parse_news_excluded_sources(None)
    assert is_excluded_news_source("https://www.google.com/finance/beta?hl=ZH-cn", patterns)
    assert not is_excluded_news_source("https://www.google.com/search?q=stocks", patterns)


@pytest.mark.parametrize(
    "url,excluded",
    [
        ("https://www.instagram.com/p/DgQQhWlGxVa", True),
        ("https://m.facebook.com/story.php?id=1", True),
        ("https://www.threads.net/@x/post/1", True),
        ("https://www.moomoo.com/hans/community/feed/pre-market-117688766103957", True),
        ("https://www.moomoo.com/community/feed/dow-117689705299973", True),
        ("https://www.moomoo.com/news/post/123", False),
        ("https://tw.stock.yahoo.com/class-quote?sectorId=40&exchange=TAI", True),
        ("https://tw.stock.yahoo.com/quote/2330.TW", True),
        ("https://tw.stock.yahoo.com/news/台積電-210530975.html", False),
        ("https://www.cmoney.tw/forum/stock/2330", True),
        ("https://www.cmoney.tw/forumx/1", False),
    ],
)
def test_default_excluded_sources_cover_social_and_quote_pages(url, excluded):
    assert is_excluded_news_source(url, parse_news_excluded_sources(None)) == excluded
