"""
llm_advisor.py
---------------
Optional, swappable LLM-generated commentary layer on top of the
deterministic recommendation.

Design principle: the LLM NEVER decides BUY/HOLD/SELL. That decision is
always made by the deterministic rule-based scoring in `recommendation.py`.
When enabled, an LLM only writes a short narrative explanation of a
recommendation that has *already been made* -- it receives the finished
signal_details/recommendation/reasons as input and is explicitly
instructed not to change the call, only to explain it. This keeps the
agent's core behavior auditable and reproducible even when the optional
LLM layer is turned on, and means a caller who never sets `llm_provider`
gets identical behavior to before this module existed.

Supported providers
--------------------
    "openai" -> OpenAI Chat Completions API (requires OPENAI_API_KEY)
    "gemini" -> Google Gemini API via the `google-genai` SDK (requires
                GEMINI_API_KEY or GOOGLE_API_KEY)

Both providers are imported lazily (only when actually used), so neither
SDK is a hard dependency of the package -- someone who never sets
`llm_provider` doesn't need `openai` or `google-genai` installed at all.

Adding a new provider
-----------------------
Implement the `LLMProvider` protocol (a single `generate(prompt) -> str`
method) and add it to `_PROVIDERS` in `get_provider()`. Nothing else in
the agent needs to change -- `nodes.py` only ever talks to the
`LLMProvider` interface, never to a specific vendor SDK.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

# Default model per provider, used when the caller doesn't specify one.
# Kept as simple string constants (not fetched dynamically) so behavior is
# stable and doesn't silently change when a vendor updates their "default"
# model server-side.
_DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.0-flash",
}

SUPPORTED_PROVIDERS = ("openai", "gemini")


class LLMAdvisorError(Exception):
    """Raised when LLM commentary cannot be generated.

    Deliberately a distinct exception type from DataFetchError etc. so
    nodes.py can catch it specifically and degrade gracefully (the
    deterministic report is always produced regardless of whether this
    succeeds).
    """


class LLMProvider(Protocol):
    """Minimal interface every LLM provider implementation must satisfy.

    Keeping this to a single method is deliberate: it's the smallest
    surface that lets nodes.py stay completely vendor-agnostic, and makes
    writing a fake/mock provider for tests trivial (see tests/test_llm_advisor.py).
    """

    def generate(self, prompt: str) -> str:
        """Send `prompt` to the LLM and return its text response.

        Raises:
            LLMAdvisorError: on any failure (missing API key, network
                error, API error, empty response, etc). Implementations
                should catch vendor-specific exceptions and re-raise as
                LLMAdvisorError so callers only need to handle one type.
        """
        ...


@dataclass
class OpenAIProvider:
    """LLMProvider backed by OpenAI's Chat Completions API."""

    model: str = _DEFAULT_MODELS["openai"]
    api_key: str | None = None  # falls back to OPENAI_API_KEY env var if None
    temperature: float = 0.3    # low temperature: this is explanatory text, not creative writing
    max_tokens: int = 300

    def generate(self, prompt: str) -> str:
        try:
            import openai
        except ImportError as exc:
            raise LLMAdvisorError(
                "The 'openai' package is not installed. Run `pip install openai`."
            ) from exc

        key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise LLMAdvisorError(
                "No OpenAI API key found. Set the OPENAI_API_KEY environment "
                "variable, or pass api_key=... when constructing OpenAIProvider."
            )

        try:
            client = openai.OpenAI(api_key=key)
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            text = response.choices[0].message.content
        except Exception as exc:
            raise LLMAdvisorError(f"OpenAI API call failed: {exc}") from exc

        if not text or not text.strip():
            raise LLMAdvisorError("OpenAI returned an empty response.")
        return text.strip()


@dataclass
class GeminiProvider:
    """LLMProvider backed by Google's Gemini API via the google-genai SDK."""

    model: str = _DEFAULT_MODELS["gemini"]
    api_key: str | None = None  # falls back to GEMINI_API_KEY / GOOGLE_API_KEY env vars if None
    temperature: float = 0.3
    max_tokens: int = 300

    def generate(self, prompt: str) -> str:
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise LLMAdvisorError(
                "The 'google-genai' package is not installed. Run `pip install google-genai`."
            ) from exc

        key = self.api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise LLMAdvisorError(
                "No Gemini API key found. Set the GEMINI_API_KEY (or GOOGLE_API_KEY) "
                "environment variable, or pass api_key=... when constructing GeminiProvider."
            )

        try:
            client = genai.Client(api_key=key)
            response = client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                ),
            )
            text = response.text
        except Exception as exc:
            raise LLMAdvisorError(f"Gemini API call failed: {exc}") from exc

        if not text or not text.strip():
            raise LLMAdvisorError("Gemini returned an empty response.")
        return text.strip()


def get_provider(name: str, model: str | None = None, **kwargs) -> LLMProvider:
    """Factory: construct an LLMProvider by name.

    Args:
        name: "openai" or "gemini".
        model: Optional model override; falls back to a sensible per-provider default.
        **kwargs: Passed through to the provider's constructor (e.g. api_key, temperature).

    Returns:
        A constructed LLMProvider instance.

    Raises:
        ValueError: if `name` isn't a supported provider.
    """
    name = (name or "").strip().lower()
    if name == "openai":
        return OpenAIProvider(model=model or _DEFAULT_MODELS["openai"], **kwargs)
    if name == "gemini":
        return GeminiProvider(model=model or _DEFAULT_MODELS["gemini"], **kwargs)
    raise ValueError(
        f"Unsupported LLM provider '{name}'. Supported providers: {SUPPORTED_PROVIDERS}"
    )


def build_commentary_prompt(
    ticker: str,
    company_name: str | None,
    recommendation: str,
    reasons: list[str],
    signal_details: dict,
) -> str:
    """Build the prompt sent to the LLM for narrative commentary.

    The prompt is deliberately explicit that the recommendation is already
    final and the LLM's job is only to explain it in plain language --
    this is what keeps the LLM from being able to silently override the
    deterministic decision, even if a model tries to "helpfully" second-guess it.
    """
    name = f"{company_name} ({ticker})" if company_name else ticker
    reasons_block = "\n".join(f"- {r}" for r in reasons)

    return f"""You are a financial analysis assistant. A deterministic rule-based \
system has ALREADY made the following stock recommendation. Your only job is \
to explain it in clear, plain-English language for a retail investor -- \
you must NOT change, second-guess, or contradict the recommendation itself.

Stock: {name}
Recommendation (final, not to be changed): {recommendation}
Current price: {signal_details.get('currency', 'USD')} {signal_details.get('close', 'N/A')}
10-day SMA: {signal_details.get('sma_10', 'N/A')}
20-day SMA: {signal_details.get('sma_20', 'N/A')}
14-day RSI: {signal_details.get('rsi_14', 'N/A')}

Rule-based reasoning that produced this recommendation:
{reasons_block}

Write a short (3-5 sentence) plain-English explanation of why these \
indicators led to a {recommendation} recommendation, in a neutral, \
educational tone suitable for someone learning technical analysis. Do not \
suggest a different action than {recommendation}. Do not give financial \
advice beyond explaining this specific recommendation. Do not add \
disclaimers -- those are added separately by the application."""


def generate_llm_commentary(
    provider: LLMProvider,
    ticker: str,
    company_name: str | None,
    recommendation: str,
    reasons: list[str],
    signal_details: dict,
) -> str:
    """Generate narrative commentary explaining an already-final recommendation.

    Args:
        provider: A constructed LLMProvider (see `get_provider`).
        ticker, company_name, recommendation, reasons, signal_details: The
            already-computed deterministic recommendation output to explain.

    Returns:
        The LLM's commentary text.

    Raises:
        LLMAdvisorError: if the provider fails for any reason. Callers
            (see nodes.py) should catch this and continue without
            commentary rather than failing the whole analysis -- the
            deterministic report never depends on this succeeding.
    """
    prompt = build_commentary_prompt(ticker, company_name, recommendation, reasons, signal_details)
    return provider.generate(prompt)
