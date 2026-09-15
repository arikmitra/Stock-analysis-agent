"""
state.py
--------
Defines the shared state object that flows through every node of the
LangGraph stock analysis agent.

LangGraph agents are built around a single state object (a TypedDict, dataclass,
or pydantic model) that each node reads from and writes back to. Keeping the
schema in one place makes the graph easy to reason about: every node's
signature is `def node(state: AgentState) -> AgentState`.

Design note on the deterministic-by-default guarantee
-------------------------------------------------------
`deterministic_recommendation`, `deterministic_reasons`, and
`signal_details` are always produced by the rule-based scoring in
`recommendation.py` -- this is computed unconditionally on every run,
regardless of `recommendation_mode` or `llm_provider`.

`recommendation` / `recommendation_reasons` are the "active" call shown
as the headline result: by default (`recommendation_mode="deterministic"`,
the default) these are simply copies of the deterministic fields. Only
if the caller explicitly sets `recommendation_mode="ai"` AND a
successful, parseable AI response comes back does `recommendation`
switch to the AI-generated call instead -- and even then,
`deterministic_recommendation` remains populated alongside it, so the
rule-based baseline is never hidden. If AI mode is requested but fails
for any reason, `recommendation` silently falls back to the
deterministic call (recorded via `ai_recommendation_used=False` and
`ai_recommendation_error`), so a broken/misconfigured AI provider can
never leave the user without a valid, safe recommendation.

`llm_commentary` (from llm_advisor.py) is a separate, purely narrative
feature: it explains whichever recommendation is active and can never
change it, in either mode.
"""

from __future__ import annotations

from typing import Optional, TypedDict

import pandas as pd


class AgentState(TypedDict, total=False):
    """Shared state passed between nodes in the stock analysis graph.

    Using `total=False` means no key is required up front -- each node
    populates the fields it is responsible for as the graph executes.
    This lets us start a run with just a `ticker` and let the graph fill
    in everything else.
    """

    # ---- Input -----------------------------------------------------
    ticker: str                     # Raw ticker symbol as provided by the user (e.g. "aapl")
    lookback_days: Optional[int]    # Calendar days of history to fetch (default: 60 if unset)
    history_rows: Optional[int]     # Trading days shown in the report's history table (default: 10)
    llm_provider: Optional[str]     # "openai" | "gemini" | None (None = no LLM calls at all, the default)
    llm_model: Optional[str]        # Provider-specific model name override, e.g. "gpt-4o-mini"
    recommendation_mode: Optional[str]  # "deterministic" (default) | "ai" -- which call becomes `recommendation`

    # ---- Validation --------------------------------------------------
    normalized_ticker: str          # Cleaned/upper-cased ticker (e.g. "AAPL")
    is_valid: bool                  # Whether the ticker passed validation

    # ---- Data fetch --------------------------------------------------
    raw_data: Optional[pd.DataFrame]  # OHLCV data from yfinance, `lookback_days` calendar days
    company_name: Optional[str]       # Long name of the company, if available
    currency: Optional[str]           # Trading currency, if available

    # ---- Technical indicators ----------------------------------------
    indicators: Optional[pd.DataFrame]  # raw_data + SMA10/SMA20/RSI14 + any custom indicator columns

    # ---- Deterministic recommendation (ALWAYS computed, every run) -------
    deterministic_recommendation: Optional[str]        # "BUY" | "HOLD" | "SELL" -- rule-based, always present
    deterministic_reasons: Optional[list[str]]          # Human-readable justifications for the rule-based call
    signal_details: Optional[dict]                      # Raw numeric signal values used in both scoring paths

    # ---- Active recommendation (what the report headlines) ----------------
    recommendation: Optional[str]               # "BUY" | "HOLD" | "SELL" -- deterministic, unless AI mode succeeded
    recommendation_reasons: Optional[list[str]]  # Reasons for whichever call is active
    recommendation_source: Optional[str]         # "deterministic" | "ai" -- which path actually produced `recommendation`

    # ---- AI-generated recommendation (optional, opt-in only) ----------------
    ai_recommendation: Optional[str]            # The AI's raw BUY/HOLD/SELL call, if recommendation_mode="ai" was requested
    ai_recommendation_reasoning: Optional[str]  # The AI's own reasoning text
    ai_recommendation_used: Optional[bool]      # True if `recommendation` above actually came from the AI path
    ai_recommendation_error: Optional[str]      # Populated if AI mode was requested but failed/fell back

    # ---- LLM commentary (optional, opt-in only; narrative-only, never decides) ---
    llm_commentary: Optional[str]       # Narrative commentary from the LLM, if llm_provider was set
    llm_error: Optional[str]            # Populated if the LLM call failed; never blocks the deterministic report

    # ---- User feedback ----------------------------------------------------
    feedback_rating: Optional[str]      # "helpful" | "not_helpful" | None
    feedback_comment: Optional[str]     # Optional free-text comment from the user
    feedback_recorded: Optional[bool]   # Whether feedback was successfully persisted

    # ---- Reporting -----------------------------------------------------
    report: Optional[str]               # Final formatted text report

    # ---- Error handling --------------------------------------------------
    error: Optional[str]                # Populated whenever a node fails
    error_stage: Optional[str]          # Which node raised the error
