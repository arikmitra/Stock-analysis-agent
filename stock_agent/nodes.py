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
    generate_recommendation
      │
      ▼
    build_report
      │
      ▼
     END
"""

from __future__ import annotations

from .data_source import DataFetchError, fetch_history
from .indicators import add_all_indicators
from .recommendation import generate_recommendation as _generate_recommendation
from .report import format_report
from .state import AgentState

# Toggle for demo/offline environments. See data_source.py for details.
# In a normal environment with internet access this has no effect, since
# yfinance is always tried first.
_ALLOW_SYNTHETIC_FALLBACK = True


def validate_ticker(state: AgentState) -> dict:
    """Validate and normalize the input ticker symbol.

    Populates `normalized_ticker` and `is_valid`. Does not hit the network --
    this is a fast, local sanity check so obviously-bad input fails quickly
    with a clear message before we spend a network round trip on it.
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

    return {"normalized_ticker": normalized, "is_valid": True}


def fetch_data(state: AgentState) -> dict:
    """Fetch 60 days of historical OHLCV data for the validated ticker."""
    ticker = state["normalized_ticker"]

    try:
        df, meta = fetch_history(
            ticker,
            period_days=60,
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
    """Compute SMA10, SMA20, and RSI14 on the fetched price history."""
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
        return {
            "error": (
                f"Only {len(enriched)} trading days of data were available, "
                f"which isn't enough to compute a 20-day SMA and 14-day RSI. "
                f"Try again later once more history has accumulated."
            ),
            "error_stage": "calculate_indicators",
        }

    return {"indicators": enriched}


def generate_recommendation(state: AgentState) -> dict:
    """Run the rule-based BUY/HOLD/SELL scoring logic on the latest indicators."""
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
