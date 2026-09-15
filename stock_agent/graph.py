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
│ validate_ticker  │  Local, fast sanity check on ticker + lookback_days
│                  │  + recommendation_mode.
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
│generate_recommendation│  ALWAYS deterministic (recommendation.py);
└────────┬───────────────┘  seeds `recommendation` with the rule-based call
         │                          │
   ┌─────┴────────────────┐          │
   │ recommendation_mode  │          │
   │      == "ai"?         │          │
   └─────┬────────────────┘          │
   yes   │   no                     │
         │    └──────────┐          │
         ▼                │          │
┌──────────────────────┐  │          │
│generate_ai_          │  │          │
│  recommendation       │  │          │
│  (may OVERRIDE        │  │          │
│  `recommendation`;    │  │          │
│  falls back to the    │  │          │
│  deterministic call   │  │          │
│  on any failure --    │  │          │
│  never blocks report) │  │          │
└────────┬───────────────┘  │          │
         │                  │          │
         ▼                  ▼          │
   ┌─────┴──────────┐               │
   │ llm_provider set?│               │
   └─────┬──────────┘               │
   yes   │   no                     │
         │    └──────────┐          │
         ▼                │          │
┌──────────────────────┐  │          │
│generate_llm_commentary│  │          │
│  (best-effort; never  │  │          │
│  blocks the report;   │  │          │
│  explains whichever   │  │          │
│  call is active)      │  │          │
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

Two independent opt-in nodes follow `generate_recommendation`, each
skipped entirely by default:

  - `generate_ai_recommendation` is reached only when
    `state["recommendation_mode"] == "ai"`. It can OVERRIDE
    `state["recommendation"]` with an AI-generated call -- this is the
    one node in the whole graph allowed to change the "active"
    recommendation after the deterministic scorer set it. If it fails
    for any reason (missing key, network error, unparseable response),
    it does NOT route to handle_error -- it leaves `recommendation`
    exactly as the deterministic node set it, which is the
    fallback-to-deterministic guarantee. `deterministic_recommendation`
    itself is never touched, so the rule-based baseline remains visible
    in the final report regardless of which mode is active.

  - `generate_llm_commentary` is reached only when `state["llm_provider"]`
    is set. It can only EXPLAIN whichever call is currently active
    (deterministic or AI) -- it never changes `recommendation`. Failures
    here degrade the same way (recorded in `llm_error`, never routes to
    handle_error).

Both nodes read from the same `llm_provider`/`llm_model` fields and reuse
the same `LLMProvider` transport (see llm_advisor.py), but serve
different purposes: one can decide, the other can only explain.
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


def _route_after_recommendation_common(state: AgentState) -> str:
    """Conditional edge shared by both the deterministic-only and
    AI-recommendation paths: only visit the LLM commentary node if the
    caller opted in via state["llm_provider"]. This runs after whichever
    recommendation path was active, so commentary always explains the
    final `recommendation` value -- deterministic or AI.
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
    workflow.add_node("generate_ai_recommendation", nodes.generate_ai_recommendation_node)
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
    # AI recommendation generation (which may override `recommendation`),
    # then optionally through LLM commentary (which never overrides
    # anything), before reaching build_report.
    #
    # This single conditional edge on generate_recommendation covers all
    # three possible next steps (AI node / commentary node / straight to
    # build_report) in one routing function, rather than chaining through
    # an intermediate pass-through node -- LangGraph conditional edges can
    # map to any number of named destinations, so there's no need for one.
    workflow.add_conditional_edges(
        "generate_recommendation",
        lambda state: (
            "generate_ai_recommendation"
            if state.get("recommendation_mode") == "ai"
            else _route_after_recommendation_common(state)
        ),
        {
            "generate_ai_recommendation": "generate_ai_recommendation",
            "generate_llm_commentary": "generate_llm_commentary",
            "build_report": "build_report",
        },
    )
    workflow.add_conditional_edges(
        "generate_ai_recommendation",
        _route_after_recommendation_common,
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
    recommendation_mode: str | None = None,
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
        llm_provider: Optional -- "openai" or "gemini". Used by both
            optional LLM-touching features below: narrative commentary
            (always) and AI-driven recommendations (only if
            recommendation_mode="ai"). Defaults to None, meaning no LLM
            is used at all and behavior is identical to the pre-LLM
            version of this agent.
        llm_model: Optional model name override for the chosen provider
            (e.g. "gpt-4o", "gemini-1.5-pro"). Ignored if llm_provider is None.
        recommendation_mode: "deterministic" (default, whether passed
            explicitly or left as None) or "ai". Controls which call
            becomes the ACTIVE `result["recommendation"]`:
              - "deterministic" (default): the rule-based scorer's call,
                exactly as before this feature existed.
              - "ai": requires `llm_provider` to also be set. An LLM
                independently evaluates the same indicator values and its
                call becomes `result["recommendation"]` -- but only if
                the call succeeds and parses cleanly; any failure falls
                back to the deterministic call automatically (see
                `result["ai_recommendation_error"]` /
                `result["ai_recommendation_used"]`). The deterministic
                result remains available in
                `result["deterministic_recommendation"]` regardless of
                which mode was requested or which one ultimately won.

    Returns:
        The final AgentState dict after the graph has run to completion.
        Always contains a "report" key -- either a success report or an
        error report -- so callers can always safely print(result["report"]).
        Always contains "deterministic_recommendation" on a successful
        run, regardless of `recommendation_mode`.
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
    if recommendation_mode is not None:
        inputs["recommendation_mode"] = recommendation_mode

    result = graph.invoke(inputs)
    return result
