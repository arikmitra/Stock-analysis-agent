"""
graph.py
--------
Wires the node functions in `nodes.py` into a LangGraph `StateGraph`.

Architecture
============

    START
      │
      ▼
┌─────────────────┐
│ validate_ticker  │  Local, fast sanity check on ticker + lookback_days.
└────────┬─────────┘
         │
   ┌─────┴─────┐
   │  is_valid? │
   └─────┬─────┘
    valid│  invalid
         │     └───────────────┐
         ▼                     ▼
┌─────────────────┐   ┌────────────────┐
│   fetch_data     │   │  handle_error   │
└────────┬─────────┘   └───────┬────────┘
         │                     │
   ┌─────┴──────┐              │
   │ fetch ok?   │              │
   └─────┬──────┘              │
    ok   │  error               │
         │     └────────────────┤
         ▼                      │
┌────────────────────┐          │
│ calculate_indicators│          │
└────────┬────────────┘          │
         │                       │
   ┌─────┴──────┐                │
   │ enough data?│                │
   └─────┬──────┘                │
    ok   │  error                 │
         │     └──────────────────┤
         ▼                        │
┌──────────────────────┐          │
│generate_recommendation│  ALWAYS deterministic (recommendation.py)
└────────┬───────────────┘          │
         │                          │
   ┌─────┴──────────┐               │
   │ llm_provider set?│               │
   └─────┬──────────┘               │
   yes   │   no                     │
         │    └──────────┐          │
         ▼                │          │
┌──────────────────────┐  │          │
│generate_llm_commentary│  │          │
│  (best-effort; never  │  │          │
│  blocks the report)   │  │          │
└────────┬───────────────┘  │          │
         │                  │          │
         ▼                  ▼          │
        ┌──────────────────┐           │
        │   build_report    │           │
        └────────┬───────────┘           │
                 │                       │
                 ▼                       ▼
                END ◄──────────────────────

Every stage that can fail (validation, fetch, indicator calculation) has
its own conditional edge that routes straight to `handle_error` on failure,
so a bad ticker or a network hiccup never crashes the graph -- it always
terminates with a `report` string in the final state.

The LLM commentary node is opt-in via `state["llm_provider"]`. When unset
(the default), the conditional edge after `generate_recommendation` skips
straight to `build_report` -- the LLM node never even runs, so there is
zero behavior or performance difference for callers who don't use this
feature. When the LLM node does run and fails, it degrades gracefully
(see nodes.py) rather than routing to handle_error, since a broken LLM
provider should never prevent the deterministic report from being shown.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from . import nodes
from .feedback import record_feedback
from .state import AgentState


def _route_after_validation(state: AgentState) -> str:
    """Conditional edge: go to fetch_data if valid, else handle_error."""
    return "fetch_data" if state.get("is_valid") else "handle_error"


def _route_after_fetch(state: AgentState) -> str:
    """Conditional edge: go to calculate_indicators if fetch succeeded, else handle_error."""
    return "handle_error" if state.get("error") else "calculate_indicators"


def _route_after_indicators(state: AgentState) -> str:
    """Conditional edge: proceed to recommendation if indicators computed, else handle_error."""
    return "handle_error" if state.get("error") else "generate_recommendation"


def _route_after_recommendation(state: AgentState) -> str:
    """Conditional edge: only visit the LLM commentary node if the caller
    opted in via state["llm_provider"]. This is the switch that keeps the
    deterministic path as the default -- an unset llm_provider means the
    LLM node is skipped entirely, not just a no-op call.
    """
    return "generate_llm_commentary" if state.get("llm_provider") else "build_report"


def build_graph():
    """Construct and compile the stock analysis LangGraph.

    Returns:
        A compiled LangGraph graph ready to `.invoke({"ticker": "AAPL"})`.
    """
    workflow = StateGraph(AgentState)

    # Register nodes
    workflow.add_node("validate_ticker", nodes.validate_ticker)
    workflow.add_node("fetch_data", nodes.fetch_data)
    workflow.add_node("calculate_indicators", nodes.calculate_indicators)
    workflow.add_node("generate_recommendation", nodes.generate_recommendation)
    workflow.add_node("generate_llm_commentary", nodes.generate_llm_commentary_node)
    workflow.add_node("build_report", nodes.build_report)
    workflow.add_node("handle_error", nodes.handle_error)

    # Entry point
    workflow.add_edge(START, "validate_ticker")

    # Conditional routing after each stage that can fail
    workflow.add_conditional_edges(
        "validate_ticker",
        _route_after_validation,
        {"fetch_data": "fetch_data", "handle_error": "handle_error"},
    )
    workflow.add_conditional_edges(
        "fetch_data",
        _route_after_fetch,
        {"calculate_indicators": "calculate_indicators", "handle_error": "handle_error"},
    )
    workflow.add_conditional_edges(
        "calculate_indicators",
        _route_after_indicators,
        {"generate_recommendation": "generate_recommendation", "handle_error": "handle_error"},
    )

    # After the deterministic recommendation, optionally detour through
    # LLM commentary before reaching build_report.
    workflow.add_conditional_edges(
        "generate_recommendation",
        _route_after_recommendation,
        {"generate_llm_commentary": "generate_llm_commentary", "build_report": "build_report"},
    )
    workflow.add_edge("generate_llm_commentary", "build_report")

    workflow.add_edge("build_report", END)
    workflow.add_edge("handle_error", END)

    return workflow.compile()


def submit_feedback(
    result: AgentState,
    rating: str,
    comment: str | None = None,
) -> AgentState:
    """Record user feedback on a completed `analyze_stock()` result.

    This is the simple, direct way to record feedback from a script or
    notebook -- it doesn't require building/invoking the graph again. It
    reads the ticker/recommendation/signal_details straight out of the
    result dict `analyze_stock()` returned, and calls
    `stock_agent.feedback.record_feedback` on your behalf.

    Args:
        result: The dict returned by `analyze_stock()`. Must be a
            successful result (i.e. `result["recommendation"]` is set) --
            feedback on a failed analysis isn't meaningful.
        rating: "helpful" or "not_helpful".
        comment: Optional free-text comment.

    Returns:
        The result dict, with `feedback_rating`, `feedback_comment`, and
        `feedback_recorded` merged in -- so callers can chain/inspect the
        outcome without a separate variable.

    Raises:
        ValueError: if `result` has no recommendation to attach feedback to.
        FeedbackError: if the feedback can't be persisted (re-raised from
            `stock_agent.feedback.record_feedback`, since this is a direct
            user action, not a background pipeline step -- the caller should
            know immediately if their feedback wasn't saved).
    """
    if not result.get("recommendation"):
        raise ValueError(
            "Cannot record feedback on a result with no recommendation "
            "(the analysis may have failed -- check result['error'])."
        )

    record = record_feedback(
        ticker=result.get("normalized_ticker", result.get("ticker", "UNKNOWN")),
        recommendation=result["recommendation"],
        rating=rating,
        signal_details=result.get("signal_details") or {},
        comment=comment,
    )

    updated = dict(result)
    updated["feedback_rating"] = rating
    updated["feedback_comment"] = comment
    updated["feedback_recorded"] = True
    updated["_feedback_id"] = record.feedback_id
    return updated


def analyze_stock(
    ticker: str,
    lookback_days: int | None = None,
    history_rows: int | None = None,
    llm_provider: str | None = None,
    llm_model: str | None = None,
) -> AgentState:
    """Convenience wrapper: build the graph, run it once, and return final state.

    Args:
        ticker: Stock ticker symbol, e.g. "AAPL".
        lookback_days: Calendar days of history to fetch. Defaults to 60
            if not provided (the agent's original behavior). Must be at
            least ~30 days to leave room for a 20-day SMA and 14-day RSI.
        history_rows: How many recent trading days to display in the
            report's history table. Defaults to 10 if not provided (the
            agent's original behavior). Independent of `lookback_days` --
            e.g. fetch 90 days for stable indicators but only display the
            most recent 15 rows.
        llm_provider: Optional -- "openai" or "gemini" to enable narrative
            AI commentary alongside the (always deterministic)
            recommendation. Defaults to None, meaning no LLM is used at
            all and behavior is identical to the pre-LLM version of this
            agent.
        llm_model: Optional model name override for the chosen provider
            (e.g. "gpt-4o", "gemini-1.5-pro"). Ignored if llm_provider is None.

    Returns:
        The final AgentState dict after the graph has run to completion.
        Always contains a "report" key -- either a success report or an
        error report -- so callers can always safely print(result["report"]).
    """
    graph = build_graph()
    inputs: dict = {"ticker": ticker}
    if lookback_days is not None:
        inputs["lookback_days"] = lookback_days
    if history_rows is not None:
        inputs["history_rows"] = history_rows
    if llm_provider is not None:
        inputs["llm_provider"] = llm_provider
    if llm_model is not None:
        inputs["llm_model"] = llm_model

    result = graph.invoke(inputs)
    return result
