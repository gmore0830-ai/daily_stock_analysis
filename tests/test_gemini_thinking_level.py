# -*- coding: utf-8 -*-
"""GEMINI_THINKING_LEVEL -> LiteLLM reasoning_effort for Gemini 3+ models."""

import os
import unittest
from unittest import mock

from src.llm.errors import classify_litellm_generation_param_error
from src.llm.generation_params import (
    apply_litellm_generation_params,
    clear_litellm_generation_param_recovery_cache,
    is_gemini_3_or_newer_wire_model,
)


class GeminiThinkingLevelTestCase(unittest.TestCase):
    def setUp(self) -> None:
        clear_litellm_generation_param_recovery_cache()

    def _apply(self, model: str, **extra):
        kwargs = {"model": model, "messages": [], **extra}
        return apply_litellm_generation_params(kwargs, model, 0.7)

    def test_gemini_3_plus_detection(self) -> None:
        for model in (
            "gemini/gemini-3.8-flash",
            "gemini-3.6-flash",
            "vertex_ai/gemini-3-pro",
            "gemini/gemini-4-hydrogen",
            "gemini/gemini-10.0-pro",
        ):
            self.assertTrue(is_gemini_3_or_newer_wire_model(model), model)
        for model in (
            "gemini/gemini-2.5-flash",
            "openai/gemini-3.8-flash",
            "deepseek/deepseek-v4-flash",
            "",
        ):
            self.assertFalse(is_gemini_3_or_newer_wire_model(model), model)

    def test_unset_keeps_provider_default(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GEMINI_THINKING_LEVEL", None)
            self.assertNotIn("reasoning_effort", self._apply("gemini/gemini-3.8-flash"))

    def test_level_applied_to_gemini_3_only(self) -> None:
        with mock.patch.dict(os.environ, {"GEMINI_THINKING_LEVEL": " LOW "}):
            self.assertEqual(self._apply("gemini/gemini-3.8-flash")["reasoning_effort"], "low")
            self.assertNotIn("reasoning_effort", self._apply("gemini/gemini-2.5-flash"))
            self.assertNotIn("reasoning_effort", self._apply("deepseek/deepseek-v4-flash"))

    def test_invalid_level_ignored_and_explicit_value_kept(self) -> None:
        with mock.patch.dict(os.environ, {"GEMINI_THINKING_LEVEL": "turbo"}):
            self.assertNotIn("reasoning_effort", self._apply("gemini/gemini-3.8-flash"))
        with mock.patch.dict(os.environ, {"GEMINI_THINKING_LEVEL": "low"}):
            out = self._apply("gemini/gemini-3.8-flash", reasoning_effort="high")
            self.assertEqual(out["reasoning_effort"], "high")

    def test_rejected_reasoning_effort_is_recoverable(self) -> None:
        recovery = classify_litellm_generation_param_error(
            Exception("Unsupported parameter: reasoning_effort is not supported for this model")
        )
        self.assertIsNotNone(recovery)
        self.assertEqual(recovery.omit_params, ("reasoning_effort",))


if __name__ == "__main__":
    unittest.main()
