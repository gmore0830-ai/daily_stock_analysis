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
