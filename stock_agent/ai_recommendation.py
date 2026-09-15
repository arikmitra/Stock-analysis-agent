"""
ai_recommendation.py
---------------------
Optional, swappable AI-generated BUY/HOLD/SELL recommendation -- an
alternative *decision-making* path to the deterministic scorer in
recommendation.py, not a narrative add-on like llm_advisor.py.

This is a different feature from llm_advisor.py's commentary generation.
llm_advisor.py explains a decision that has already been made and can
never change it. This module lets an LLM MAKE its own BUY/HOLD/SELL call
from the same indicator data the deterministic scorer sees -- but only
when the caller explicitly opts into AI-driven recommendations, and even
then the deterministic call is still always computed and always
available (see nodes.py / graph.py / state.py: `deterministic_recommendation`
is populated whenever `recommendation_mode="ai"`, so the rule-based
baseline is never hidden or discarded).

Design principle: deterministic stays the default, in two senses:
    1. state["recommendation_mode"] defaults to "deterministic" -- an
       agent run with no extra arguments never calls this module at all.
    2. Even when "ai" mode is requested, if the AI call fails for any
       reason (missing key, network error, unparseable response, an
       out-of-vocabulary answer, etc), the agent FALLS BACK to the
       deterministic recommendation rather than failing the whole
       analysis or returning something unvalidated. This fallback is
       recorded in `ai_recommendation_error` / `ai_recommendation_fallback`
       so the caller can see it happened, but the user-facing report
       still shows a valid, safe recommendation either way.

Why a separate module from llm_advisor.py, given both call an LLM
--------------------------------------------------------------------
They share the underlying `LLMProvider` transport (see get_provider() in
llm_advisor.py, reused here) but are functionally distinct: one produces
free-form explanatory prose that can never affect the outcome, the other
produces a decision that -- when this optional path is enabled -- BECOMES
the outcome. Keeping them in separate modules keeps that distinction
explicit in the codebase rather than blurring "explain" and "decide"
behind a single "llm stuff" module.

Output parsing / safety
-------------------------
LLMs are not reliable structured-output engines by default: they can
wrap their answer in markdown, add caveats, misspell "BUY", or ignore
formatting instructions outright. `parse_ai_response()` is deliberately
strict and defensive -- it looks for an unambiguous BUY/HOLD/SELL token
using a clear priority order, and raises AIRecommendationError (rather
than guessing) if the response can't be confidently parsed. An
unparseable response is treated as a FAILURE of the AI path, which
triggers the deterministic fallback described above -- never as a
silent "guess and hope."
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .llm_advisor import LLMAdvisorError, LLMProvider

VALID_CALLS = ("BUY", "HOLD", "SELL")


class AIRecommendationError(Exception):
    """Raised when an AI-generated recommendation cannot be produced or
    parsed with confidence.

    Distinct from LLMAdvisorError (which signals a transport-level
    failure -- no API key, network error, etc) so callers can log/inspect
    *why* the AI path failed: transport failure vs. an unparseable
    response are different problems worth distinguishing, even though
    both currently result in the same fallback behavior in nodes.py.
    """


@dataclass
class AIRecommendationResult:
    """Structured output of the AI recommendation engine."""

    recommendation: str                    # "BUY" | "HOLD" | "SELL"
    reasoning: str                         # The model's explanation, verbatim (lightly cleaned)
    raw_response: str = field(repr=False)  # Full untouched response, kept for debugging/audit


def build_ai_recommendation_prompt(
    ticker: str,
    company_name: str | None,
    signal_details: dict,
    deterministic_recommendation: str | None = None,
    deterministic_reasons: list[str] | None = None,
) -> str:
    """Build the prompt sent to the LLM for an AI-generated recommendation.

    Unlike llm_advisor.build_commentary_prompt, this prompt does NOT tell
    the model what the "correct" answer is -- the whole point of this
    mode is to get the model's own independent read of the indicators.
    The deterministic result is included only as context (clearly labeled
    as a separate, rule-based signal the model may agree or disagree
    with), never as an instruction to match it.

    The prompt enforces a strict, greppable output format so
    `parse_ai_response` has the best possible chance of extracting a
    clean answer without guessing.
    """
    name = f"{company_name} ({ticker})" if company_name else ticker

    context_block = ""
    if deterministic_recommendation:
        reasons_str = "; ".join(deterministic_reasons or [])
        context_block = f"""
For reference, a separate deterministic rule-based system (SMA crossover \
+ RSI thresholds) independently arrived at: {deterministic_recommendation} \
({reasons_str}). You may agree or disagree with this -- form your own \
judgment from the raw numbers below; do not simply defer to it."""

    return f"""You are a technical-analysis assistant. Based ONLY on the \
indicator values below, decide whether a trader following a short-term \
technical strategy should BUY, HOLD, or SELL {name}.

Current price: {signal_details.get('currency', 'USD')} {signal_details.get('close', 'N/A')}
10-day SMA: {signal_details.get('sma_10', 'N/A')}
20-day SMA: {signal_details.get('sma_20', 'N/A')}
14-day RSI: {signal_details.get('rsi_14', 'N/A')}
{context_block}

Respond in EXACTLY this format, with no markdown, no extra commentary \
before or after:

RECOMMENDATION: <BUY, HOLD, or SELL -- exactly one word>
REASONING: <2-4 sentences explaining your call from the indicator values above>

Your RECOMMENDATION line must contain exactly one of the words BUY, HOLD, \
or SELL and nothing else on that line. This is for an educational \
technical-analysis tool, not personalized financial advice -- do not add \
disclaimers, they are handled separately by the application."""


def parse_ai_response(raw_response: str) -> AIRecommendationResult:
    """Extract a validated BUY/HOLD/SELL call and reasoning from raw LLM text.

    Parsing strategy, in priority order:
        1. Look for a line matching `RECOMMENDATION: <WORD>` (the format
           requested in the prompt) and validate <WORD> is exactly one of
           BUY/HOLD/SELL.
        2. If that exact format isn't found, fall back to scanning the
           whole response for a standalone BUY/HOLD/SELL token -- but
           ONLY if exactly one of the three appears (if the model
           mentions multiple, e.g. "not a SELL, more of a HOLD", we
           cannot confidently disambiguate and must fail rather than guess).
        3. If neither succeeds, raise AIRecommendationError.

    A REASONING line is extracted on a best-effort basis; its absence
    does not by itself cause a parse failure, since the recommendation
    token is the safety-critical part.

    Args:
        raw_response: The LLM's raw text output.

    Returns:
        An AIRecommendationResult with a validated `recommendation` in
        VALID_CALLS.

    Raises:
        AIRecommendationError: if no unambiguous BUY/HOLD/SELL call can
            be extracted.
    """
    if not raw_response or not raw_response.strip():
        raise AIRecommendationError("AI response was empty.")

    text = raw_response.strip()

    # Strategy 1: strict "RECOMMENDATION: <WORD>" line match.
    match = re.search(r"RECOMMENDATION\s*:\s*([A-Za-z]+)", text, re.IGNORECASE)
    if match:
        candidate = match.group(1).strip().upper()
        if candidate in VALID_CALLS:
            reasoning = _extract_reasoning(text)
            return AIRecommendationResult(
                recommendation=candidate, reasoning=reasoning, raw_response=raw_response
            )
        # Found the label but the value wasn't a valid call (e.g. model
        # wrote "RECOMMENDATION: Buy, but cautiously") -- don't guess,
        # fall through to strategy 2 in case a clean token appears elsewhere.

    # Strategy 2: exactly one unambiguous BUY/HOLD/SELL token anywhere,
    # matched as a whole word so "SELLING" or "HOLDER" don't false-match.
    found = {
        call for call in VALID_CALLS
        if re.search(rf"\b{call}\b", text, re.IGNORECASE)
    }
    if len(found) == 1:
        recommendation = found.pop()
        reasoning = _extract_reasoning(text)
        return AIRecommendationResult(
            recommendation=recommendation, reasoning=reasoning, raw_response=raw_response
        )
    if len(found) > 1:
        raise AIRecommendationError(
            f"AI response mentioned multiple possible calls {sorted(found)} "
            f"without a clear RECOMMENDATION: line -- cannot determine a "
            f"single unambiguous recommendation."
        )

    raise AIRecommendationError(
        "Could not find a valid BUY/HOLD/SELL recommendation in the AI response."
    )


def _extract_reasoning(text: str) -> str:
    """Best-effort extraction of the REASONING line/section, if present."""
    match = re.search(r"REASONING\s*:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return text  # fall back to the full response if no explicit REASONING label


def generate_ai_recommendation(
    provider: LLMProvider,
    ticker: str,
    company_name: str | None,
    signal_details: dict,
    deterministic_recommendation: str | None = None,
    deterministic_reasons: list[str] | None = None,
) -> AIRecommendationResult:
    """Generate and parse an AI-driven BUY/HOLD/SELL recommendation.

    Args:
        provider: A constructed LLMProvider (see llm_advisor.get_provider).
        ticker, company_name, signal_details: Context describing the stock
            and its current indicator values.
        deterministic_recommendation, deterministic_reasons: Optional --
            the already-computed deterministic call, included in the
            prompt as reference context only (see build_ai_recommendation_prompt).

    Returns:
        A validated AIRecommendationResult.

    Raises:
        LLMAdvisorError: if the underlying provider call fails (missing
            API key, network error, etc).
        AIRecommendationError: if the provider responds but the response
            can't be confidently parsed into a BUY/HOLD/SELL call.

    Callers (see nodes.py) should catch both exception types and fall
    back to the deterministic recommendation -- this function deliberately
    never returns a low-confidence guess in place of raising.
    """
    prompt = build_ai_recommendation_prompt(
        ticker, company_name, signal_details,
        deterministic_recommendation, deterministic_reasons,
    )
    raw = provider.generate(prompt)  # may raise LLMAdvisorError
    return parse_ai_response(raw)     # may raise AIRecommendationError
