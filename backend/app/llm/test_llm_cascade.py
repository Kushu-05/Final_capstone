import unittest
from unittest.mock import patch, MagicMock
import requests
import json
from backend.app.config import config
from backend.app.llm.engine import UnifiedLLMEngine

class TestLLMProviderCascade(unittest.TestCase):
    """
    Comprehensive verification of LLM provider routing, fallback cascade,
    bounded retries, and local dynamic generation across the exact chain:
    Claude -> NVIDIA -> OpenRouter -> Gemini -> OpenAI -> Local.
    """

    def setUp(self):
        self.engine = UnifiedLLMEngine()
        # Save original config
        self.orig_anthropic_key = config.anthropic_api_key
        self.orig_nvidia_key = config.nvidia_api_key
        self.orig_openrouter_key = config.openrouter_api_key
        self.orig_gemini_key = config.gemini_api_key
        self.orig_openai_key = config.openai_api_key
        self.orig_preferred = config.preferred_provider
        self.orig_retries = config.llm_max_retries
        self.orig_anthropic_model = config.anthropic_model

    def tearDown(self):
        config.anthropic_api_key = self.orig_anthropic_key
        config.nvidia_api_key = self.orig_nvidia_key
        config.openrouter_api_key = self.orig_openrouter_key
        config.gemini_api_key = self.orig_gemini_key
        config.openai_api_key = self.orig_openai_key
        config.preferred_provider = self.orig_preferred
        config.llm_max_retries = self.orig_retries
        config.anthropic_model = self.orig_anthropic_model

    def test_provider_cascade_order_in_auto_mode(self):
        """Validates exact priority order: Claude -> NVIDIA -> OpenRouter -> Gemini -> OpenAI -> Local."""
        config.preferred_provider = "auto"
        config.anthropic_api_key = "test_anthropic_key"
        config.nvidia_api_key = "test_nvidia_key"
        config.openrouter_api_key = "test_openrouter_key"
        config.gemini_api_key = "test_gemini_key"
        config.openai_api_key = "test_openai_key"

        cascade = self.engine.get_provider_cascade()
        expected = ["anthropic", "nvidia", "openrouter", "gemini", "openai", "local"]
        self.assertEqual(cascade, expected)

    def test_skip_providers_without_credentials(self):
        """Providers without credentials should be excluded from auto cascade."""
        config.preferred_provider = "auto"
        config.anthropic_api_key = None
        config.nvidia_api_key = "test_nvidia_key"
        config.openrouter_api_key = None
        config.gemini_api_key = None
        config.openai_api_key = "test_openai_key"

        cascade = self.engine.get_provider_cascade()
        self.assertEqual(cascade, ["nvidia", "openai", "local"])

    def test_provider_override(self):
        """Explicit override should select the requested provider."""
        config.anthropic_api_key = "test_claude_key"
        cascade = self.engine.get_provider_cascade(override_provider="nvidia")
        self.assertEqual(cascade, ["nvidia"])

        cascade_local = self.engine.get_provider_cascade(override_provider="local")
        self.assertEqual(cascade_local, ["local"])

    @patch("requests.post")
    def test_claude_model_and_primary_success(self, mock_post):
        """Verifies configured Claude model is used first and requested with claude-sonnet-5-5."""
        config.preferred_provider = "auto"
        config.anthropic_api_key = "test_claude_key"
        config.anthropic_model = "claude-sonnet-5-5"

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "content": [{"type": "text", "text": "Claude 5.5 operational diagnosis."}]
        }
        mock_post.return_value = mock_resp

        result = self.engine.generate_with_metadata("How to fix cold food?")
        self.assertEqual(result["provider"], "anthropic")
        self.assertEqual(result["model"], "claude-sonnet-5-5")
        self.assertEqual(result["text"], "Claude 5.5 operational diagnosis.")
        self.assertEqual(len(result["fallback_history"]), 0)

        # Inspect request payload sent to Anthropic
        called_kwargs = mock_post.call_args.kwargs
        self.assertEqual(called_kwargs["json"]["model"], "claude-sonnet-5-5")

    @patch("requests.post")
    def test_stepwise_cascade_claude_to_nvidia(self, mock_post):
        """Transition 1: Claude fails -> NVIDIA succeeds."""
        config.preferred_provider = "auto"
        config.anthropic_api_key = "test_claude_key"
        config.nvidia_api_key = "test_nvidia_key"
        config.llm_max_retries = 0

        claude_resp = MagicMock(status_code=503, text="Service Unavailable")
        nvidia_resp = MagicMock(status_code=200)
        nvidia_resp.json.return_value = {
            "choices": [{"message": {"content": "NVIDIA NIM response"}}]
        }
        mock_post.side_effect = [claude_resp, nvidia_resp]

        result = self.engine.generate_with_metadata("Test query")
        self.assertEqual(result["provider"], "nvidia")
        self.assertEqual(result["text"], "NVIDIA NIM response")
        self.assertEqual(len(result["fallback_history"]), 1)
        self.assertEqual(result["fallback_history"][0]["provider"], "anthropic")

    @patch("requests.post")
    def test_stepwise_cascade_nvidia_to_openrouter(self, mock_post):
        """Transition 2: Claude fails + NVIDIA fails -> OpenRouter succeeds."""
        config.preferred_provider = "auto"
        config.anthropic_api_key = "test_claude_key"
        config.nvidia_api_key = "test_nvidia_key"
        config.openrouter_api_key = "test_openrouter_key"
        config.llm_max_retries = 0

        claude_resp = MagicMock(status_code=401, text="Unauthorized")
        nvidia_resp = MagicMock(status_code=503, text="NVIDIA busy")
        openrouter_resp = MagicMock(status_code=200)
        openrouter_resp.json.return_value = {
            "choices": [{"message": {"content": "OpenRouter response"}}]
        }
        mock_post.side_effect = [claude_resp, nvidia_resp, openrouter_resp]

        result = self.engine.generate_with_metadata("Test query")
        self.assertEqual(result["provider"], "openrouter")
        self.assertEqual(result["text"], "OpenRouter response")
        self.assertEqual([h["provider"] for h in result["fallback_history"]], ["anthropic", "nvidia"])

    @patch("requests.post")
    def test_stepwise_cascade_openrouter_to_gemini(self, mock_post):
        """Transition 3: Claude, NVIDIA, OpenRouter all fail -> Gemini succeeds."""
        config.preferred_provider = "auto"
        config.anthropic_api_key = "k1"
        config.nvidia_api_key = "k2"
        config.openrouter_api_key = "k3"
        config.gemini_api_key = "k4"
        config.llm_max_retries = 0

        r1 = MagicMock(status_code=500, text="Claude err")
        r2 = MagicMock(status_code=500, text="Nvidia err")
        r3 = MagicMock(status_code=500, text="Openrouter err")
        r_gemini = MagicMock(status_code=200)
        r_gemini.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "Gemini response"}]}}]
        }
        mock_post.side_effect = [r1, r2, r3, r_gemini]

        result = self.engine.generate_with_metadata("Test query")
        self.assertEqual(result["provider"], "gemini")
        self.assertEqual(result["text"], "Gemini response")
        self.assertEqual([h["provider"] for h in result["fallback_history"]], ["anthropic", "nvidia", "openrouter"])

    @patch("requests.post")
    def test_stepwise_cascade_gemini_to_openai(self, mock_post):
        """Transition 4: Claude, NVIDIA, OpenRouter, Gemini all fail -> OpenAI succeeds."""
        config.preferred_provider = "auto"
        config.anthropic_api_key = "k1"
        config.nvidia_api_key = "k2"
        config.openrouter_api_key = "k3"
        config.gemini_api_key = "k4"
        config.openai_api_key = "k5"
        config.llm_max_retries = 0

        r1 = MagicMock(status_code=500, text="Claude err")
        r2 = MagicMock(status_code=500, text="Nvidia err")
        r3 = MagicMock(status_code=500, text="Openrouter err")
        r4 = MagicMock(status_code=500, text="Gemini err")
        r_openai = MagicMock(status_code=200)
        r_openai.json.return_value = {
            "choices": [{"message": {"content": "OpenAI response"}}]
        }
        mock_post.side_effect = [r1, r2, r3, r4, r_openai]

        result = self.engine.generate_with_metadata("Test query")
        self.assertEqual(result["provider"], "openai")
        self.assertEqual(result["text"], "OpenAI response")
        self.assertEqual([h["provider"] for h in result["fallback_history"]], ["anthropic", "nvidia", "openrouter", "gemini"])

    @patch("requests.post")
    def test_stepwise_cascade_openai_to_local(self, mock_post):
        """Transition 5 & Complete Chain: All 5 cloud providers fail -> Local succeeds."""
        config.preferred_provider = "auto"
        config.anthropic_api_key = "k1"
        config.nvidia_api_key = "k2"
        config.openrouter_api_key = "k3"
        config.gemini_api_key = "k4"
        config.openai_api_key = "k5"
        config.llm_max_retries = 0

        # All 5 return HTTP 503
        mock_post.side_effect = [
            MagicMock(status_code=503, text="Claude 503"),
            MagicMock(status_code=503, text="Nvidia 503"),
            MagicMock(status_code=503, text="OpenRouter 503"),
            MagicMock(status_code=503, text="Gemini 503"),
            MagicMock(status_code=503, text="OpenAI 503"),
        ]

        result = self.engine.generate_with_metadata("What are the wait time complaints?", json_mode=True)
        self.assertEqual(result["provider"], "local")
        self.assertEqual(
            [h["provider"] for h in result["fallback_history"]],
            ["anthropic", "nvidia", "openrouter", "gemini", "openai"]
        )
        parsed = json.loads(result["text"])
        self.assertIn("Wait Times & Ticket Bottlenecks", parsed["detected_domains"])

    @patch("requests.post")
    def test_fail_fast_on_permanent_errors(self, mock_post):
        """Permanent errors (401, 403, 404) should not be retried with exponential backoff."""
        config.anthropic_api_key = "test_claude_key"
        config.llm_max_retries = 2
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.text = "Invalid API Key"
        mock_post.return_value = mock_resp

        content, error = self.engine._call_anthropic("query", "system", 0.2, False)
        self.assertIsNone(content)
        self.assertIn("Permanent Error", error)
        self.assertEqual(mock_post.call_count, 1)

    def test_full_fallback_when_no_credentials_configured(self):
        """When no credentials are in environment, local dynamic generator executes directly."""
        config.preferred_provider = "auto"
        config.anthropic_api_key = None
        config.nvidia_api_key = None
        config.openrouter_api_key = None
        config.gemini_api_key = None
        config.openai_api_key = None

        result = self.engine.generate_with_metadata("Explain delay causes", json_mode=True)
        self.assertEqual(result["provider"], "local")
        parsed = json.loads(result["text"])
        self.assertIn("Wait Times & Ticket Bottlenecks", parsed["detected_domains"])

if __name__ == "__main__":
    unittest.main()
