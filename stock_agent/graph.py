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
│ validate_ticker  │  Local, fast sanity check on the ticker string.
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
│generate_recommendation│          │
└────────┬───────────────┘          │
         │                          │
         ▼                          │
┌─────────────────┐                 │
│  build_report    │                 │
└────────┬─────────┘                 │
         │                           │
         ▼                           ▼
        END ◄──────────────────────────

Every stage that can fail (validation, fetch, indicator calculation) has
its own conditional edge that routes straight to `handle_error` on failure,
so a bad ticker or a network hiccup never crashes the graph -- it always
terminates with a `report` string in the final state.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from . import nodes
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

    # Linear tail: recommendation -> report -> END
    workflow.add_edge("generate_recommendation", "build_report")
    workflow.add_edge("build_report", END)
    workflow.add_edge("handle_error", END)

    return workflow.compile()


def analyze_stock(ticker: str) -> AgentState:
    """Convenience wrapper: build the graph, run it once, and return final state.

    Args:
        ticker: Stock ticker symbol, e.g. "AAPL".

    Returns:
        The final AgentState dict after the graph has run to completion.
        Always contains a "report" key -- either a success report or an
        error report -- so callers can always safely print(result["report"]).
    """
    graph = build_graph()
    result = graph.invoke({"ticker": ticker})
    return result
