"""Tests for LLM_ANALYSIS_MAX_OUTPUT_TOKENS and the output-budget warning."""

import logging
from types import SimpleNamespace

import pytest

from src.analyzer import (
    _resolve_analysis_max_output_tokens,
    _warn_if_output_budget_exhausted,
)
from src.config import Config


@pytest.fixture(autouse=True)
def _reset_config():
    Config.reset_instance()
    yield
    Config.reset_instance()


def _load_config(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("LLM_ANALYSIS_MAX_OUTPUT_TOKENS", raising=False)
    else:
        monkeypatch.setenv("LLM_ANALYSIS_MAX_OUTPUT_TOKENS", value)
    Config.reset_instance()
    return Config.get_instance()


def test_default_is_8192(monkeypatch):
    assert _load_config(monkeypatch, None).llm_analysis_max_output_tokens == 8192


def test_env_override(monkeypatch):
    assert _load_config(monkeypatch, "16384").llm_analysis_max_output_tokens == 16384


@pytest.mark.parametrize("raw", ["abc", "100", "999999"])
def test_invalid_values_fall_back_or_clamp(monkeypatch, raw):
    value = _load_config(monkeypatch, raw).llm_analysis_max_output_tokens
    assert 1024 <= value <= 131072


def test_resolver_uses_config_value():
    assert _resolve_analysis_max_output_tokens(SimpleNamespace(llm_analysis_max_output_tokens=16384)) == 16384


@pytest.mark.parametrize("value", [None, 0, -1, "16384", True])
def test_resolver_falls_back_for_bad_values(value):
    cfg = SimpleNamespace(llm_analysis_max_output_tokens=value)
    assert _resolve_analysis_max_output_tokens(cfg) == 8192


def test_resolver_handles_config_without_field():
    assert _resolve_analysis_max_output_tokens(SimpleNamespace()) == 8192


def test_warns_when_budget_exhausted(caplog):
    usage = {"completion_tokens": 8192, "completion_tokens_details": {"reasoning_tokens": 6974}}
    with caplog.at_level(logging.WARNING, logger="src.analyzer"):
        assert _warn_if_output_budget_exhausted(usage, 8192, stock_label="X(600016)") is True
    assert "LLM_ANALYSIS_MAX_OUTPUT_TOKENS" in caplog.text
    assert "6974" in caplog.text


@pytest.mark.parametrize(
    "usage",
    [None, {}, {"completion_tokens": 7000}, {"completion_tokens": None}, {"completion_tokens": True}],
)
def test_no_warning_when_under_budget_or_unknown(usage, caplog):
    with caplog.at_level(logging.WARNING, logger="src.analyzer"):
        assert _warn_if_output_budget_exhausted(usage, 8192, stock_label="X") is False
    assert "LLM输出上限" not in caplog.text


# ---------------------------------------------------------------------------
# LLM_ANALYSIS_TIMEOUT_SEC
# ---------------------------------------------------------------------------

import json
from unittest.mock import patch

from src.analyzer import _resolve_analysis_timeout_sec


def test_timeout_default_is_disabled(monkeypatch):
    monkeypatch.delenv("LLM_ANALYSIS_TIMEOUT_SEC", raising=False)
    Config.reset_instance()
    assert Config.get_instance().llm_analysis_timeout_sec == 0.0


def test_timeout_env_override(monkeypatch):
    monkeypatch.setenv("LLM_ANALYSIS_TIMEOUT_SEC", "420")
    Config.reset_instance()
    assert Config.get_instance().llm_analysis_timeout_sec == 420.0


@pytest.mark.parametrize("value,expected", [(420, 420.0), (90.5, 90.5), (0, None), (-5, None), (None, None), ("420", None), (True, None)])
def test_timeout_resolver(value, expected):
    assert _resolve_analysis_timeout_sec(SimpleNamespace(llm_analysis_timeout_sec=value)) == expected


def _make_analyzer(config):
    from src.analyzer import GeminiAnalyzer

    analyzer = GeminiAnalyzer.__new__(GeminiAnalyzer)
    analyzer._router = None
    analyzer._litellm_available = True
    analyzer._config_override = config
    return analyzer


_RESPONSE_TEXT = json.dumps({
    "sentiment_score": 70, "trend_prediction": "看多",
    "operation_advice": "持有", "analysis_summary": "fixture",
})


def _run_analyze(config, dispatch):
    analyzer = _make_analyzer(config)
    with patch.object(analyzer, "_dispatch_litellm_completion", side_effect=dispatch), \
         patch.object(analyzer, "_get_analysis_system_prompt", return_value="system"), \
         patch.object(analyzer, "_get_skill_prompt_sections", return_value=(None, None, True)), \
         patch.object(analyzer, "_format_prompt", return_value="prompt"):
        return analyzer.analyze({"code": "600519", "stock_name": "fixture"})


def _ok_response(kwargs):
    if kwargs.get("stream"):
        return iter([SimpleNamespace(
            model="openai/m", usage=None,
            choices=[SimpleNamespace(delta=SimpleNamespace(content=_RESPONSE_TEXT))],
        )])
    return SimpleNamespace(
        model="openai/m", usage=None,
        choices=[SimpleNamespace(message=SimpleNamespace(content=_RESPONSE_TEXT))],
    )


@pytest.mark.parametrize("timeout_sec,expected_timeout", [(420.0, 420.0), (0.0, None)])
def test_analyze_passes_timeout_and_output_budget(timeout_sec, expected_timeout):
    seen = []

    def dispatch(model, kwargs, **_):
        seen.append(dict(kwargs))
        return _ok_response(kwargs)

    config = Config(
        litellm_model="primary-route",
        litellm_fallback_models=[],
        gemini_request_delay=0,
        report_integrity_enabled=False,
        llm_analysis_max_output_tokens=16384,
        llm_analysis_timeout_sec=timeout_sec,
    )
    result = _run_analyze(config, dispatch)

    assert result.success is True
    assert seen
    for kwargs in seen:
        assert kwargs["max_tokens"] == 16384
        assert kwargs.get("timeout") == expected_timeout


def test_analyze_falls_back_to_next_model_on_timeout():
    import litellm

    attempted = []

    def dispatch(model, kwargs, **_):
        attempted.append(model)
        if model == "primary-route":
            raise litellm.Timeout(message="timed out", model=model, llm_provider="openai")
        return _ok_response(kwargs)

    config = Config(
        litellm_model="primary-route",
        litellm_fallback_models=["backup-route"],
        gemini_request_delay=0,
        report_integrity_enabled=False,
        llm_analysis_timeout_sec=420.0,
    )
    result = _run_analyze(config, dispatch)

    assert result.success is True
    assert "backup-route" in attempted


# ---------------------------------------------------------------------------
# Market review / free-text generation also honors LLM_ANALYSIS_TIMEOUT_SEC
# ---------------------------------------------------------------------------

def _text_response(kwargs):
    if kwargs.get("stream"):
        return iter([SimpleNamespace(
            model="openai/m", usage=None,
            choices=[SimpleNamespace(delta=SimpleNamespace(content="复盘内容"))],
        )])
    return SimpleNamespace(
        model="openai/m", usage=None,
        choices=[SimpleNamespace(message=SimpleNamespace(content="复盘内容"))],
    )


@pytest.mark.parametrize("method", ["generate_text_with_metadata", "generate_text"])
@pytest.mark.parametrize("timeout_sec,expected_timeout", [(420.0, 420.0), (0.0, None)])
def test_text_generation_passes_timeout(method, timeout_sec, expected_timeout):
    seen = []

    def dispatch(model, kwargs, **_):
        seen.append(dict(kwargs))
        return _text_response(kwargs)

    analyzer = _make_analyzer(Config(
        litellm_model="primary-route",
        litellm_fallback_models=[],
        gemini_request_delay=0,
        llm_analysis_timeout_sec=timeout_sec,
    ))
    with patch.object(analyzer, "_dispatch_litellm_completion", side_effect=dispatch), \
         patch("src.analyzer.persist_llm_usage"):
        result = getattr(analyzer, method)("prompt", max_tokens=8192, temperature=0.7)

    assert result
    assert seen
    for kwargs in seen:
        assert kwargs["max_tokens"] == 8192
        assert kwargs.get("timeout") == expected_timeout


# ---------------------------------------------------------------------------
# Hard wall-clock deadline per model attempt
# ---------------------------------------------------------------------------

import contextvars
import threading
import time

from src.analyzer import _ModelAttemptDeadlineExceeded, _run_with_deadline


def test_run_with_deadline_inline_when_disabled():
    caller = threading.current_thread()
    assert _run_with_deadline(lambda: threading.current_thread() is caller, None, label="m") is True


def test_run_with_deadline_returns_value_and_propagates_errors():
    assert _run_with_deadline(lambda: 42, 5, label="m") == 42

    def boom():
        raise ValueError("bad")

    with pytest.raises(ValueError, match="bad"):
        _run_with_deadline(boom, 5, label="m")


def test_run_with_deadline_gives_up_on_hang():
    release = threading.Event()
    start = time.monotonic()
    with pytest.raises(_ModelAttemptDeadlineExceeded):
        _run_with_deadline(lambda: release.wait(10), 0.2, label="m")
    assert time.monotonic() - start < 2
    release.set()


def test_run_with_deadline_no_budget_left():
    called = []
    with pytest.raises(_ModelAttemptDeadlineExceeded):
        _run_with_deadline(lambda: called.append(1), 0, label="m")
    assert called == []


def test_run_with_deadline_keeps_context_vars():
    var = contextvars.ContextVar("v", default="unset")
    var.set("caller")
    assert _run_with_deadline(var.get, 5, label="m") == "caller"


def _deadline_analyzer(timeout_sec):
    return _make_analyzer(Config(
        litellm_model="primary-route",
        litellm_fallback_models=["backup-route"],
        gemini_request_delay=0,
        llm_analysis_timeout_sec=timeout_sec,
    ))


def test_hung_model_is_abandoned_and_next_model_used():
    release = threading.Event()
    calls = []

    def dispatch(model, kwargs, **_):
        calls.append((model, bool(kwargs.get("stream"))))
        if model == "primary-route":
            release.wait(10)  # simulates a connection that never times out on its own
            raise RuntimeError("released")
        return _text_response(kwargs)

    analyzer = _deadline_analyzer(0.3)
    start = time.monotonic()
    try:
        with patch.object(analyzer, "_dispatch_litellm_completion", side_effect=dispatch), \
             patch("src.analyzer.persist_llm_usage"):
            result = analyzer.generate_text_with_metadata("prompt", max_tokens=8192)
    finally:
        release.set()

    assert result.text == "复盘内容"
    assert time.monotonic() - start < 3
    assert [c[0] for c in calls].count("primary-route") == 1
    assert any(c[0] == "backup-route" for c in calls)


def test_hung_stream_skips_non_stream_retry_and_falls_back():
    release = threading.Event()
    calls = []

    def dispatch(model, kwargs, **_):
        calls.append((model, bool(kwargs.get("stream"))))
        if model == "primary-route":
            release.wait(10)
            raise RuntimeError("released")
        return _ok_response(kwargs)

    config = Config(
        litellm_model="primary-route",
        litellm_fallback_models=["backup-route"],
        gemini_request_delay=0,
        report_integrity_enabled=False,
        llm_analysis_timeout_sec=0.3,
    )
    start = time.monotonic()
    try:
        result = _run_analyze(config, dispatch)
    finally:
        release.set()

    assert result.success is True
    assert time.monotonic() - start < 5
    # The stream attempt used the whole budget, so the hung model gets no non-stream retry.
    assert [c for c in calls if c[0] == "primary-route"] == [("primary-route", True)]
    assert any(c[0] == "backup-route" for c in calls)


def test_fast_stream_failure_still_retries_non_stream_within_budget():
    calls = []

    def dispatch(model, kwargs, **_):
        calls.append((model, bool(kwargs.get("stream"))))
        if kwargs.get("stream"):
            raise RuntimeError("stream unsupported")
        return _text_response(kwargs)

    analyzer = _deadline_analyzer(5)
    with patch.object(analyzer, "_dispatch_litellm_completion", side_effect=dispatch), \
         patch("src.analyzer.persist_llm_usage"):
        result = analyzer.generate_text_with_metadata("prompt", max_tokens=8192)

    assert result.text == "复盘内容"
    assert ("primary-route", False) in calls
    assert all(c[0] == "primary-route" for c in calls)
