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
