"""
nodes.py
--------
LangGraph node functions for the stock analysis agent.

Each function has the signature `def node(state: AgentState) -> dict`,
which is the standard LangGraph pattern: a node receives the current
state and returns a partial dict of updates to merge into it.

Graph flow:

    START
      │
      ▼
    validate_ticker ──(invalid)──► handle_error ──► END
      │ (valid)
      ▼
    fetch_data ──(fetch failed)──► handle_error ──► END
      │ (success)
      ▼
    calculate_indicators ──(insufficient data)──► handle_error ──► END
      │ (success)
      ▼
    generate_recommendation  (ALWAYS deterministic -- see recommendation.py)
      │
      ▼
    generate_ai_recommendation  (SKIPPED unless state["recommendation_mode"] == "ai")
      │
      ▼
    generate_llm_commentary  (SKIPPED unless state["llm_provider"] is set)
      │
      ▼
    build_report
      │
      ▼
     END

Two independent, both-optional LLM-touching nodes exist and serve
different purposes:

  - `generate_ai_recommendation` (this file) / `ai_recommendation.py`:
    can produce its OWN BUY/HOLD/SELL call, as an opt-in alternative
    DECISION-MAKING path to the deterministic scorer. Only reached when
    `recommendation_mode == "ai"`. If it fails or returns something
    unparseable, the node falls back to the deterministic call rather
    than propagating the failure -- `recommendation` is guaranteed to
    always be a valid BUY/HOLD/SELL either way.

  - `generate_llm_commentary` / `llm_advisor.py`: can only EXPLAIN a
    call that's already been finalized (whichever one ended up in
    `recommendation` -- deterministic or AI). It never changes the
    outcome. Only reached when `llm_provider` is set.

`generate_recommendation` (the deterministic scorer) always runs, on
every single request, regardless of either of the above -- this is what
"deterministic path is the default" means concretely: the rule-based
call is computed unconditionally, and it is also what `recommendation`
resolves to unless AI mode was both requested AND successful.
"""

from __future__ import annotations

from .ai_recommendation import (
    AIRecommendationError,
    generate_ai_recommendation as _generate_ai_recommendation,
)
from .data_source import DataFetchError, fetch_history
from .feedback import FeedbackError, record_feedback
from .indicators import add_all_indicators
from .llm_advisor import LLMAdvisorError, generate_llm_commentary, get_provider
from .recommendation import generate_recommendation as _generate_recommendation
from .report import format_report
from .state import AgentState

# Toggle for demo/offline environments. See data_source.py for details.
# In a normal environment with internet access this has no effect, since
# yfinance is always tried first.
_ALLOW_SYNTHETIC_FALLBACK = True

# Default lookback window (calendar days) when the caller doesn't specify
# one via state["lookback_days"]. 60 calendar days comfortably yields 40+
# trading days, enough for a 20-day SMA and 14-day RSI to fully populate.
DEFAULT_LOOKBACK_DAYS = 60

# Minimum lookback that can still possibly satisfy a 20-day SMA (needs
# ~20 trading days -> ~28-30 calendar days accounting for weekends). Below
# this, we fail fast with a clear message instead of a confusing empty result.
MIN_LOOKBACK_DAYS = 30

VALID_RECOMMENDATION_MODES = ("deterministic", "ai")


def validate_ticker(state: AgentState) -> dict:
    """Validate and normalize the input ticker symbol.

    Populates `normalized_ticker` and `is_valid`. Does not hit the network --
    this is a fast, local sanity check so obviously-bad input fails quickly
    with a clear message before we spend a network round trip on it.

    Also validates `lookback_days` and `recommendation_mode` if the caller
    provided them, since a bad value here is just as much a "fix your
    input" error as a bad ticker, and should fail before any network call too.
    """
    ticker = state.get("ticker", "")

    if not ticker or not isinstance(ticker, str) or not ticker.strip():
        return {
            "is_valid": False,
            "error": "No ticker symbol was provided. Please supply a stock ticker, e.g. 'AAPL'.",
            "error_stage": "validate_ticker",
        }

    normalized = ticker.strip().upper()
    allowed_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-")

    if not set(normalized) <= allowed_chars or len(normalized) > 10:
        return {
            "is_valid": False,
            "normalized_ticker": normalized,
            "error": (
                f"'{ticker}' does not look like a valid ticker symbol "
                f"(expected 1-10 letters/digits, e.g. 'AAPL', 'BRK-B')."
            ),
            "error_stage": "validate_ticker",
        }

    lookback_days = state.get("lookback_days")
    if lookback_days is not None:
        if not isinstance(lookback_days, int) or isinstance(lookback_days, bool):
            return {
                "is_valid": False,
                "normalized_ticker": normalized,
                "error": f"lookback_days must be an integer number of days, got {lookback_days!r}.",
                "error_stage": "validate_ticker",
            }
        if lookback_days < MIN_LOOKBACK_DAYS:
            return {
                "is_valid": False,
                "normalized_ticker": normalized,
                "error": (
                    f"lookback_days={lookback_days} is too small to compute a 20-day SMA "
                    f"and 14-day RSI. Use at least {MIN_LOOKBACK_DAYS} calendar days "
                    f"(roughly {MIN_LOOKBACK_DAYS // 7 * 5} trading days)."
                ),
                "error_stage": "validate_ticker",
            }

    recommendation_mode = state.get("recommendation_mode")
    if recommendation_mode is not None and recommendation_mode not in VALID_RECOMMENDATION_MODES:
        return {
            "is_valid": False,
            "normalized_ticker": normalized,
            "error": (
                f"recommendation_mode must be one of {VALID_RECOMMENDATION_MODES}, "
                f"got {recommendation_mode!r}."
            ),
            "error_stage": "validate_ticker",
        }

    return {"normalized_ticker": normalized, "is_valid": True}


def fetch_data(state: AgentState) -> dict:
    """Fetch historical OHLCV data for the validated ticker.

    Uses `state["lookback_days"]` if provided, otherwise falls back to
    `DEFAULT_LOOKBACK_DAYS` (60 calendar days) -- the same default the
    agent always used before this became configurable.
    """
    ticker = state["normalized_ticker"]
    lookback_days = state.get("lookback_days") or DEFAULT_LOOKBACK_DAYS

    try:
        df, meta = fetch_history(
            ticker,
            period_days=lookback_days,
            use_synthetic_fallback=_ALLOW_SYNTHETIC_FALLBACK,
        )
    except DataFetchError as exc:
        return {
            "error": str(exc),
            "error_stage": "fetch_data",
        }

    return {
        "raw_data": df,
        "company_name": meta.get("company_name"),
        "currency": meta.get("currency", "USD"),
        "_data_source": meta.get("source", "yfinance"),  # internal bookkeeping
    }


def calculate_indicators(state: AgentState) -> dict:
    """Compute SMA10, SMA20, RSI14, and any registered custom indicators
    on the fetched price history.
    """
    df = state.get("raw_data")

    if df is None or df.empty:
        return {
            "error": "No price data available to calculate indicators.",
            "error_stage": "calculate_indicators",
        }

    try:
        enriched = add_all_indicators(df)
    except ValueError as exc:
        return {"error": str(exc), "error_stage": "calculate_indicators"}

    # Confirm we have at least one fully-populated row (needs >= 20 trading days).
    valid_rows = enriched.dropna(subset=["SMA_10", "SMA_20", "RSI_14"])
    if valid_rows.empty:
        lookback_days = state.get("lookback_days") or DEFAULT_LOOKBACK_DAYS
        return {
            "error": (
                f"Only {len(enriched)} trading days of data were available "
                f"(requested {lookback_days} calendar days of history), which "
                f"isn't enough to compute a 20-day SMA and 14-day RSI. Try a "
                f"larger lookback_days value."
            ),
            "error_stage": "calculate_indicators",
        }

    return {"indicators": enriched}


def generate_recommendation(state: AgentState) -> dict:
    """Run the rule-based BUY/HOLD/SELL scoring logic on the latest indicators.

    This ALWAYS runs and ALWAYS produces `deterministic_recommendation` --
    it never consults an LLM, regardless of state["recommendation_mode"]
    or state["llm_provider"]. It also seeds `recommendation` /
    `recommendation_reasons` / `recommendation_source` with the
    deterministic result as the default "active" call; the optional
    `generate_ai_recommendation` node (running strictly after this one)
    may override those three fields if AI mode was requested and
    succeeded, but `deterministic_recommendation` itself is never
    touched by that -- the rule-based baseline is always present in the
    final state for comparison, regardless of which mode is active.
    """
    indicators = state.get("indicators")

    try:
        result = _generate_recommendation(indicators)
    except ValueError as exc:
        return {"error": str(exc), "error_stage": "generate_recommendation"}

    return {
        "deterministic_recommendation": result.recommendation,
        "deterministic_reasons": result.reasons,
        "signal_details": result.signal_details,
        # Seed the "active" call with the deterministic result. This is
        # the ONLY value `recommendation` will ever hold unless
        # generate_ai_recommendation_node below both runs and succeeds.
        "recommendation": result.recommendation,
        "recommendation_reasons": result.reasons,
        "recommendation_source": "deterministic",
    }


def generate_ai_recommendation_node(state: AgentState) -> dict:
    """Optionally let an LLM produce its OWN BUY/HOLD/SELL call, as an
    alternative to the deterministic scorer's result.

    Only reached if `state["recommendation_mode"] == "ai"` (see graph.py's
    conditional edge) -- if unset or "deterministic" (the default), this
    node is skipped entirely and `recommendation` stays exactly what
    `generate_recommendation` set it to, so there is zero behavior change
    for callers who don't opt into this feature.

    On success: overwrites `recommendation` / `recommendation_reasons` /
    `recommendation_source` with the AI's call, and records it separately
    in `ai_recommendation` / `ai_recommendation_reasoning` /
    `ai_recommendation_used=True`. `deterministic_recommendation` is left
    untouched throughout, so the rule-based baseline remains visible for
    comparison in the final report.

    On failure (missing API key, network error, unparseable response,
    unsupported provider name, etc): does NOT raise and does NOT route to
    handle_error. Instead it records the failure in
    `ai_recommendation_error` / `ai_recommendation_used=False` and leaves
    `recommendation` exactly as the deterministic node set it -- this is
    the fallback-to-deterministic guarantee described in the module
    docstring. A broken or misconfigured AI provider can never leave the
    user without a valid, safe recommendation.
    """
    mode = state.get("recommendation_mode")
    if mode != "ai":
        # Defensive fallback; graph.py's routing should prevent reaching
        # here at all when mode isn't "ai", but keep this safe regardless
        # in case this node is ever wired up differently.
        return {}

    provider_name = state.get("llm_provider")
    if not provider_name:
        return {
            "ai_recommendation_used": False,
            "ai_recommendation_error": (
                "recommendation_mode='ai' was set but no llm_provider was "
                "specified; falling back to the deterministic recommendation."
            ),
        }

    try:
        provider = get_provider(provider_name, model=state.get("llm_model"))
        result = _generate_ai_recommendation(
            provider=provider,
            ticker=state["normalized_ticker"],
            company_name=state.get("company_name"),
            signal_details=state["signal_details"],
            deterministic_recommendation=state.get("deterministic_recommendation"),
            deterministic_reasons=state.get("deterministic_reasons"),
        )
        return {
            "ai_recommendation": result.recommendation,
            "ai_recommendation_reasoning": result.reasoning,
            "ai_recommendation_used": True,
            # Override the "active" call -- deterministic_recommendation
            # remains untouched in state, so it's still visible/comparable.
            "recommendation": result.recommendation,
            "recommendation_reasons": [result.reasoning],
            "recommendation_source": "ai",
        }
    except (LLMAdvisorError, AIRecommendationError, ValueError) as exc:
        # ValueError covers get_provider() rejecting an unsupported name.
        # Deliberately no "recommendation"/"recommendation_reasons"/
        # "recommendation_source" keys in this return -- leaving them
        # unset means the deterministic values set by generate_recommendation
        # remain in effect, which is exactly the fallback behavior we want.
        return {
            "ai_recommendation_used": False,
            "ai_recommendation_error": str(exc),
        }


def generate_llm_commentary_node(state: AgentState) -> dict:
    """Optionally generate narrative commentary via an LLM, explaining the
    already-final ACTIVE recommendation (deterministic or AI, whichever
    `recommendation` currently holds).

    Only reached if `state["llm_provider"]` is set (see graph.py's
    conditional edge) -- if unset, this node is skipped entirely and the
    graph proceeds straight to build_report, so there is zero behavior
    change for callers who don't opt in.

    Failures here (missing API key, network error, etc) are caught and
    recorded in `llm_error` rather than raised -- a broken or
    misconfigured LLM provider must never block the report from being
    produced, regardless of which recommendation path is active.
    """
    provider_name = state.get("llm_provider")
    if not provider_name:
        # Defensive fallback; graph.py's routing should prevent reaching
        # here at all when llm_provider is unset, but keep this safe
        # regardless in case this node is ever wired up differently.
        return {}

    try:
        provider = get_provider(provider_name, model=state.get("llm_model"))
        commentary = generate_llm_commentary(
            provider=provider,
            ticker=state["normalized_ticker"],
            company_name=state.get("company_name"),
            recommendation=state["recommendation"],
            reasons=state["recommendation_reasons"],
            signal_details=state["signal_details"],
        )
        return {"llm_commentary": commentary}
    except (LLMAdvisorError, ValueError) as exc:
        # ValueError covers get_provider() rejecting an unsupported name.
        return {"llm_error": str(exc)}


def build_report(state: AgentState) -> dict:
    """Assemble the final formatted text report from the completed state."""
    report_text = format_report(
        ticker=state["normalized_ticker"],
        company_name=state.get("company_name"),
        currency=state.get("currency"),
        indicators_df=state["indicators"],
        recommendation=state["recommendation"],
        reasons=state["recommendation_reasons"],
        signal_details=state["signal_details"],
        data_source=state.get("_data_source", "yfinance"),
        llm_commentary=state.get("llm_commentary"),
        llm_error=state.get("llm_error"),
        history_rows=state.get("history_rows") or 10,
        recommendation_source=state.get("recommendation_source", "deterministic"),
        deterministic_recommendation=state.get("deterministic_recommendation"),
        deterministic_reasons=state.get("deterministic_reasons"),
        ai_recommendation_error=state.get("ai_recommendation_error"),
    )
    return {"report": report_text}


def handle_error(state: AgentState) -> dict:
    """Terminal node reached whenever any prior node reports an error.

    Produces a short, user-facing error report so the graph always ends
    with a printable `report`, regardless of where it failed.
    """
    stage = state.get("error_stage", "unknown")
    message = state.get("error", "An unknown error occurred.")

    report_text = (
        "=" * 62 + "\n"
        f"  STOCK ANALYSIS FAILED\n"
        + "=" * 62 + "\n"
        f"  Stage:   {stage}\n"
        f"  Reason:  {message}\n"
        + "=" * 62
    )
    return {"report": report_text}


def record_feedback_node(state: AgentState) -> dict:
    """Record user feedback on a completed recommendation, if requested.

    This node is NOT part of the main analyze_stock() graph flow -- it's
    provided so applications that want feedback-recording wired into a
    LangGraph pipeline (e.g. a chat agent that re-invokes the graph on a
    "was this helpful?" follow-up) can add it as an extra node without
    reimplementing the storage logic. For simple scripts/notebooks, calling
    `stock_agent.feedback.record_feedback()` directly (see the convenience
    wrapper `submit_feedback` in graph.py) is usually more direct.

    Reads `feedback_rating` (required: "helpful" | "not_helpful") and
    `feedback_comment` (optional) from state, alongside the already-computed
    recommendation fields, and writes a FeedbackRecord to local storage.

    Never raises: a failure to persist feedback (e.g. disk permissions) is
    recorded in `error`/`error_stage` like other node failures, but
    feedback is inherently a side-channel -- callers may choose to ignore
    a feedback-recording failure rather than treat the whole analysis as failed.
    """
    rating = state.get("feedback_rating")
    if not rating:
        return {"feedback_recorded": False}

    try:
        record_feedback(
            ticker=state.get("normalized_ticker", state.get("ticker", "UNKNOWN")),
            recommendation=state.get("recommendation", "UNKNOWN"),
            rating=rating,
            signal_details=state.get("signal_details") or {},
            score=(state.get("signal_details") or {}).get("score"),
            comment=state.get("feedback_comment"),
        )
        return {"feedback_recorded": True}
    except FeedbackError as exc:
        return {
            "feedback_recorded": False,
            "error": str(exc),
            "error_stage": "record_feedback",
        }
