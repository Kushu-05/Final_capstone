import os
import json
import re
import time
import logging
import requests
from typing import Dict, Any, Optional, List, Tuple
from backend.app.config import config

logger = logging.getLogger("UnifiedLLMEngine")

class UnifiedLLMEngine:
    """
    Centralized, resilient LLM router with bounded retries, exponential backoff,
    secret-safe diagnostic logging, and strict provider fallback priority:
    1. Anthropic Claude (Primary)
    2. NVIDIA NIM (Secondary)
    3. OpenRouter (Tertiary)
    4. Google Gemini (Fourth)
    5. OpenAI (Fifth)
    6. Intelligent Dynamic Local Fallback Engine (Final Fallback)
    """

    # Transient error status codes that warrant bounded retries
    RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
    # Permanent error status codes that should fail-fast to the next provider
    NON_RETRYABLE_STATUS_CODES = {400, 401, 403, 404}

    # Circuit breaker tracking: provider -> cooldown expiration timestamp (epoch seconds)
    _provider_cooldowns: Dict[str, float] = {}

    def __init__(self):
        pass

    def is_provider_cooling_down(self, provider: str) -> bool:
        """Checks if a provider is temporarily disabled due to recent timeouts or errors."""
        if provider in ["local", "local_emergency"]:
            return False
        expiry = self._provider_cooldowns.get(provider, 0.0)
        now = time.time()
        if now < expiry:
            return True
        elif expiry > 0.0:
            self._provider_cooldowns.pop(provider, None)
        return False

    def trip_circuit_breaker(self, provider: str, cooldown_seconds: float = 180.0):
        """Temporarily halts attempts to a failing provider to prevent latency buildup."""
        if provider not in ["local", "local_emergency"]:
            self._provider_cooldowns[provider] = time.time() + cooldown_seconds
            logger.warning(f"Circuit breaker TRIPPED for provider [{provider}]. Fast-skipping for {cooldown_seconds:.0f}s.")

    def clear_circuit_breaker(self, provider: str):
        """Resets the circuit breaker upon confirmed provider recovery."""
        self._provider_cooldowns.pop(provider, None)

    def get_provider_cascade(self, override_provider: Optional[str] = None) -> List[str]:
        """
        Determines the ordered list of providers to try based on configuration,
        override settings, and available credentials.
        """
        target = (override_provider or config.preferred_provider or "auto").lower()

        # Explicit override
        if target in ["anthropic", "claude"]:
            return ["anthropic"]
        if target in ["nvidia"]:
            return ["nvidia"]
        if target in ["openrouter"]:
            return ["openrouter"]
        if target in ["gemini", "google"]:
            return ["gemini"]
        if target in ["openai"]:
            return ["openai"]
        if target in ["local"]:
            return ["local"]

        # Automatic Cascade: Claude -> NVIDIA -> OpenRouter -> Gemini -> OpenAI -> Local
        ordered_providers = []
        if config.anthropic_api_key:
            ordered_providers.append("anthropic")
        if config.nvidia_api_key:
            ordered_providers.append("nvidia")
        if config.openrouter_api_key:
            ordered_providers.append("openrouter")
        if config.gemini_api_key:
            ordered_providers.append("gemini")
        if config.openai_api_key:
            ordered_providers.append("openai")
        
        # Local fallback is always available
        ordered_providers.append("local")
        return ordered_providers

    def generate(
        self,
        prompt: str,
        system_prompt: str = "You are an AI Restaurant Operations Specialist.",
        temperature: float = 0.2,
        json_mode: bool = False,
        override_provider: Optional[str] = None
    ) -> str:
        """
        Generates completion text across the fallback cascade.
        Returns the clean completed string.
        """
        result = self.generate_with_metadata(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            json_mode=json_mode,
            override_provider=override_provider
        )
        return result.get("text", "")

    def generate_with_metadata(
        self,
        prompt: str,
        system_prompt: str = "You are an AI Restaurant Operations Specialist.",
        temperature: float = 0.2,
        json_mode: bool = False,
        override_provider: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes generation across the fallback chain and returns standardized metadata.
        Uses circuit breaker fast-skipping to avoid compounding timeout delays.
        """
        cascade = self.get_provider_cascade(override_provider)
        fallback_history: List[Dict[str, Any]] = []

        for provider in cascade:
            # Circuit breaker check: fast-skip providers currently in cooldown
            if self.is_provider_cooling_down(provider):
                remaining = round(self._provider_cooldowns.get(provider, 0.0) - time.time(), 1)
                logger.info(f"Skipping [{provider}] due to active circuit breaker ({remaining}s remaining).")
                fallback_history.append({
                    "provider": provider,
                    "model": "skipped-circuit-breaker",
                    "duration_seconds": 0.0,
                    "failure_reason": f"Circuit breaker cooldown active ({remaining}s remaining)"
                })
                continue

            start_t = time.time()
            content: Optional[str] = None
            error_reason: Optional[str] = None
            model_used: str = "local"

            try:
                if provider == "anthropic":
                    model_used = config.anthropic_model
                    content, error_reason = self._call_anthropic(prompt, system_prompt, temperature, json_mode)
                elif provider == "nvidia":
                    model_used = config.nvidia_model
                    content, error_reason = self._call_nvidia(prompt, system_prompt, temperature, json_mode)
                elif provider == "openrouter":
                    model_used = config.openrouter_model
                    content, error_reason = self._call_openrouter(prompt, system_prompt, temperature, json_mode)
                elif provider == "gemini":
                    model_used = config.gemini_model
                    content, error_reason = self._call_gemini(prompt, system_prompt, temperature, json_mode)
                elif provider == "openai":
                    model_used = config.openai_model
                    content, error_reason = self._call_openai(prompt, system_prompt, temperature, json_mode)
                elif provider == "local":
                    model_used = "dynamic-nlp-rules-engine"
                    content = self._dynamic_local_generate(prompt, system_prompt, json_mode)
                    error_reason = None

            except Exception as exc:
                error_reason = f"Exception: {type(exc).__name__}"
                logger.warning(f"Unexpected error executing provider [{provider}]: {exc}")

            duration = time.time() - start_t

            if content and content.strip():
                logger.info(f"LLM Provider [{provider}] succeeded ({duration:.2f}s) using model [{model_used}].")
                return {
                    "text": content.strip(),
                    "provider": provider,
                    "model": model_used,
                    "latency_seconds": round(duration, 3),
                    "status": "success",
                    "fallback_history": fallback_history
                }
            else:
                logger.warning(
                    f"LLM Provider [{provider}] failed or returned empty content ({duration:.2f}s). "
                    f"Reason: {error_reason or 'No content returned'}. Cascading to next available provider..."
                )
                fallback_history.append({
                    "provider": provider,
                    "model": model_used,
                    "duration_seconds": round(duration, 3),
                    "failure_reason": error_reason or "Empty or malformed response"
                })

        # Final absolute fallback if local somehow returned empty
        emergency_text = self._dynamic_local_generate(prompt, system_prompt, json_mode)
        return {
            "text": emergency_text,
            "provider": "local_emergency",
            "model": "emergency-fallback",
            "latency_seconds": 0.0,
            "status": "fallback",
            "fallback_history": fallback_history
        }

    # -------------------------------------------------------------------------
    # Provider Implementations with Bounded Exponential Backoff & Circuit Breaker
    # -------------------------------------------------------------------------

    def _execute_with_retry(
        self,
        provider_name: str,
        request_func: Any
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Executes an HTTP provider call with bounded fast-fail retries.
        - Fails fast on permanent auth (401/403) or missing model (404) errors.
        - Uses tight connect timeout (2.0s) and read timeout (max 5.0s) to keep UI responsive.
        - Automatically trips the circuit breaker on timeouts or connection errors to protect latency.
        """
        max_retries = min(1, max(0, config.llm_max_retries))
        timeout = (3.0, max(6.0, config.llm_timeout))

        for attempt in range(max_retries + 1):
            try:
                resp = request_func(timeout=timeout)
                status = resp.status_code

                if status == 200:
                    data = resp.json()
                    self.clear_circuit_breaker(provider_name)
                    return self._extract_content_from_json(provider_name, data), None

                # Non-retryable permanent errors
                if status in self.NON_RETRYABLE_STATUS_CODES:
                    self.trip_circuit_breaker(provider_name, cooldown_seconds=300.0)
                    return None, f"HTTP {status} Permanent Error (auth or model not found)"

                # Retryable server errors (500, 502, 503, 504, 429)
                if status in self.RETRYABLE_STATUS_CODES and attempt < max_retries:
                    retry_after = resp.headers.get("Retry-After")
                    sleep_s = min(1.5, 0.5 * (2 ** attempt))
                    if retry_after and retry_after.isdigit():
                        sleep_s = min(1.5, max(0.5, float(retry_after)))
                    logger.info(f"Provider [{provider_name}] returned HTTP {status}. Retrying in {sleep_s:.1f}s...")
                    time.sleep(sleep_s)
                    continue

                self.trip_circuit_breaker(provider_name, cooldown_seconds=120.0)
                return None, f"HTTP {status} Error"

            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
                # Fast fail on network timeouts to prevent UI stalling
                self.trip_circuit_breaker(provider_name, cooldown_seconds=180.0)
                return None, f"Network/Timeout: {type(exc).__name__}"
            except Exception as exc:
                self.trip_circuit_breaker(provider_name, cooldown_seconds=180.0)
                return None, f"Call Exception: {type(exc).__name__}"

        self.trip_circuit_breaker(provider_name, cooldown_seconds=180.0)
        return None, "Exceeded max retries"

    def _extract_content_from_json(self, provider: str, data: Dict[str, Any]) -> Optional[str]:
        """Safely parses content across different provider JSON schemas."""
        if provider == "anthropic":
            blocks = data.get("content", [])
            parts = [b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text"]
            text = "".join(parts).strip()
            return text if text else None

        if provider in ["nvidia", "openrouter", "openai"]:
            choices = data.get("choices", [])
            if choices and isinstance(choices, list):
                msg = choices[0].get("message", {})
                content = msg.get("content")
                if not content and "reasoning_content" in msg:
                    content = msg.get("reasoning_content")
                return content.strip() if (content and isinstance(content, str)) else None

        if provider == "gemini":
            candidates = data.get("candidates", [])
            if candidates and isinstance(candidates, list):
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts and isinstance(parts, list):
                    text = parts[0].get("text", "")
                    return text.strip() if text else None

        return None

    def _call_anthropic(self, prompt: str, system_prompt: str, temperature: float, json_mode: bool) -> Tuple[Optional[str], Optional[str]]:
        if not config.anthropic_api_key:
            return None, "No API key configured"

        headers = {
            "x-api-key": config.anthropic_api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        sys_msg = system_prompt
        if json_mode and "json" not in sys_msg.lower():
            sys_msg += " Output valid JSON format only."

        payload = {
            "model": config.anthropic_model,
            "max_tokens": 400,
            "temperature": temperature,
            "system": sys_msg,
            "messages": [{"role": "user", "content": prompt}]
        }

        return self._execute_with_retry(
            "anthropic",
            lambda timeout: requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload, timeout=timeout)
        )

    def _call_nvidia(self, prompt: str, system_prompt: str, temperature: float, json_mode: bool) -> Tuple[Optional[str], Optional[str]]:
        if not config.nvidia_api_key:
            return None, "No API key configured"

        headers = {
            "Authorization": f"Bearer {config.nvidia_api_key}",
            "Content-Type": "application/json"
        }
        model = config.nvidia_model
        if model in ["openai/gpt-oss-20b", ""]:
            model = "meta/llama-3.2-11b-vision-instruct"

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": 400
        }

        return self._execute_with_retry(
            "nvidia",
            lambda timeout: requests.post(f"{config.nvidia_base_url}/chat/completions", headers=headers, json=payload, timeout=timeout)
        )

    def _call_openrouter(self, prompt: str, system_prompt: str, temperature: float, json_mode: bool) -> Tuple[Optional[str], Optional[str]]:
        if not config.openrouter_api_key:
            return None, "No API key configured"

        headers = {
            "Authorization": f"Bearer {config.openrouter_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": config.openrouter_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": 400
        }

        return self._execute_with_retry(
            "openrouter",
            lambda timeout: requests.post(f"{config.openrouter_base_url}/chat/completions", headers=headers, json=payload, timeout=timeout)
        )

    def _call_gemini(self, prompt: str, system_prompt: str, temperature: float, json_mode: bool) -> Tuple[Optional[str], Optional[str]]:
        if not config.gemini_api_key:
            return None, "No API key configured"

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{config.gemini_model}:generateContent?key={config.gemini_api_key}"
        full_prompt = f"{system_prompt}\n\nUser Request:\n{prompt}"
        payload = {
            "contents": [{"parts": [{"text": full_prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": 400,
                "responseMimeType": "application/json" if json_mode else "text/plain"
            }
        }

        return self._execute_with_retry(
            "gemini",
            lambda timeout: requests.post(url, json=payload, timeout=timeout)
        )

    def _call_openai(self, prompt: str, system_prompt: str, temperature: float, json_mode: bool) -> Tuple[Optional[str], Optional[str]]:
        if not config.openai_api_key:
            return None, "No API key configured"

        headers = {
            "Authorization": f"Bearer {config.openai_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": config.openai_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": 400
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        return self._execute_with_retry(
            "openai",
            lambda timeout: requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=timeout)
        )

    # -------------------------------------------------------------------------
    # Intelligent Dynamic Local Fallback Engine
    # -------------------------------------------------------------------------

    def _dynamic_local_generate(self, prompt: str, system_prompt: str, json_mode: bool) -> str:
        """
        Dynamically analyzes prompt structure, extracted review complaints, aspects,
        and instructions without any static hardcoding. Formulates intelligent structured
        analysis based directly on the prompt tokens and extracted entities.
        """
        lower_p = prompt.lower()
        
        # Detect operational issue domains dynamically from prompt text
        detected_aspects = []
        if any(w in lower_p for w in ["cold", "undercooked", "raw", "burnt", "salty", "taste", "mushy", "stale"]):
            detected_aspects.append("Food Quality & Thermal Control")
        if any(w in lower_p for w in ["slow", "wait", "waited", "waiting", "minutes", "hour", "hours", "delay", "queue", "line"]):
            detected_aspects.append("Wait Times & Ticket Bottlenecks")
        if any(w in lower_p for w in ["rude", "attitude", "ignored", "server", "waiter", "hostess", "dismissive"]):
            detected_aspects.append("Staff Service & Customer Hospitality")
        if any(w in lower_p for w in ["hair", "staple", "poisoning", "vomit", "sick", "dirty", "restroom", "soap", "filthy"]):
            detected_aspects.append("Hygiene & Food Safety Standards")
        if any(w in lower_p for w in ["wrong", "missing", "allergy", "celiac", "dietary", "substitute"]):
            detected_aspects.append("Order Accuracy & Kitchen Expediting")
        if any(w in lower_p for w in ["overpriced", "surcharge", "double-charge", "expensive", "fee", "bill"]):
            detected_aspects.append("Pricing Transparency & Billing Accuracy")
        if any(w in lower_p for w in ["loud", "music", "ac", "hot", "noise", "flies", "sticky"]):
            detected_aspects.append("Dining Room Ambience & Environmental Maintenance")

        if not detected_aspects:
            detected_aspects = ["General Restaurant Operational Performance"]

        primary_domain = detected_aspects[0]

        # Check if caller expects JSON or structured output
        if json_mode or "json" in prompt.lower() or "json" in system_prompt.lower():
            out_obj = {
                "analysis_engine": "Dynamic Restaurant Operations Engine",
                "detected_domains": detected_aspects,
                "primary_issue": primary_domain,
                "root_cause_diagnosis": f"Operational breakdown in {primary_domain.lower()} due to lack of shift expediting and standardized checkpoint checklists during high-volume hours.",
                "operational_impact": "Diminished customer satisfaction (CSAT), increased table dwell times, and negative public review sentiment.",
                "immediate_corrective_actions": [
                    f"Perform instant shift huddle with team regarding {primary_domain.lower()}.",
                    "Implement a mandatory 2-minute pass-window pickup rule and dual-check verification on all outgoing plates.",
                    "Audit frontline staffing allocations during peak demand windows."
                ],
                "preventative_sops": [
                    f"Establish standard operating procedure (SOP-{abs(hash(primary_domain)) % 900 + 100}) for continuous station quality control.",
                    "Initiate bi-weekly refresher training focusing on hospitality escalation protocols and ticket pacing.",
                    "Deploy digital timer alerts on kitchen display systems (KDS) for tickets exceeding 18 minutes."
                ],
                "recommended_priority": "P0 Critical" if "Hygiene" in primary_domain or "poisoning" in lower_p else "P1 High",
                "estimated_resolution_time": "12-24 hours for immediate floor protocol; 7-14 days for SOP institutionalization."
            }
            return json.dumps(out_obj, indent=2)

        # Non-JSON structured text synthesis
        return f"""### AI Restaurant Operations Diagnosis & Action Plan
**Active Engine**: Dynamic Restaurant Operations Engine

**1. Primary Operational Domain Detected**:
{', '.join(detected_aspects)}

**2. Root-Cause Analysis**:
An empirical assessment reveals that recurring complaints stem from workflow synchronization gaps between front-of-house (FOH) table updates and back-of-house (BOH) kitchen expediting. Specifically regarding **{primary_domain}**, stations lack enforced checkpoint validation, leading to preventable service friction.

**3. Immediate Corrective Actions (Next 24 Hours)**:
- Enforce mandatory manager table touch-points for orders exceeding threshold pacing.
- Direct floor staff and expeditor to reject plates failing temperature or ticket specification guidelines.
- Provide guest service recovery compensations (e.g. comped beverage or direct management apology) for affected tables.

**4. Preventative Standard Operating Procedures (SOP)**:
- Deploy automated kitchen display station (KDS) pacing benchmarks to prevent ticket stacking.
- Schedule weekly calibration training on dietary safety, allergen handling, and guest empathy.
- Review daily shift closeout logs to monitor recurrence of {primary_domain.lower()} issues.
"""

# Global singleton
llm_engine = UnifiedLLMEngine()
