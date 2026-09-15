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
    generate_llm_commentary  (SKIPPED unless state["llm_provider"] is set)
      │
      ▼
    build_report
      │
      ▼
     END

The LLM commentary node is opt-in and best-effort: if `llm_provider` is
unset, it's skipped entirely via a conditional edge (see graph.py) and
never even runs. If it IS set but the call fails (missing API key,
network error, etc), the node catches the error, records it in
`llm_error`, and the graph proceeds to `build_report` exactly as if the
LLM had never been requested -- a failed or misconfigured LLM provider
never blocks or changes the deterministic recommendation.
"""

from __future__ import annotations

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


def validate_ticker(state: AgentState) -> dict:
    """Validate and normalize the input ticker symbol.

    Populates `normalized_ticker` and `is_valid`. Does not hit the network --
    this is a fast, local sanity check so obviously-bad input fails quickly
    with a clear message before we spend a network round trip on it.

    Also validates `lookback_days` if the caller provided one, since a bad
    value here (too small, non-numeric) is just as much a "fix your input"
    error as a bad ticker, and should fail before any network call too.
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

    This is ALWAYS the deterministic path -- see recommendation.py. It
    never consults an LLM, regardless of state["llm_provider"]. The
    optional LLM commentary node runs strictly after this one and can only
    explain this result, never change it.
    """
    indicators = state.get("indicators")

    try:
        result = _generate_recommendation(indicators)
    except ValueError as exc:
        return {"error": str(exc), "error_stage": "generate_recommendation"}

    return {
        "recommendation": result.recommendation,
        "recommendation_reasons": result.reasons,
        "signal_details": result.signal_details,
    }


def generate_llm_commentary_node(state: AgentState) -> dict:
    """Optionally generate narrative commentary via an LLM, explaining the
    already-final deterministic recommendation.

    Only reached if `state["llm_provider"]` is set (see graph.py's
    conditional edge) -- if unset, this node is skipped entirely and the
    graph proceeds straight to build_report, so there is zero behavior
    change for callers who don't opt in.

    Failures here (missing API key, network error, bad model name, etc)
    are caught and recorded in `llm_error` rather than raised -- a broken
    or misconfigured LLM provider must never block the deterministic
    report from being produced.
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
