"""
state.py
--------
Defines the shared state object that flows through every node of the
LangGraph stock analysis agent.

LangGraph agents are built around a single state object (a TypedDict, dataclass,
or pydantic model) that each node reads from and writes back to. Keeping the
schema in one place makes the graph easy to reason about: every node's
signature is `def node(state: AgentState) -> AgentState`.
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

    # ---- Validation --------------------------------------------------
    normalized_ticker: str          # Cleaned/upper-cased ticker (e.g. "AAPL")
    is_valid: bool                  # Whether the ticker passed validation

    # ---- Data fetch --------------------------------------------------
    raw_data: Optional[pd.DataFrame]  # 60 days of OHLCV data from yfinance
    company_name: Optional[str]       # Long name of the company, if available
    currency: Optional[str]           # Trading currency, if available

    # ---- Technical indicators ----------------------------------------
    indicators: Optional[pd.DataFrame]  # raw_data + SMA10/SMA20/RSI14 columns

    # ---- Recommendation ------------------------------------------------
    recommendation: Optional[str]       # "BUY" | "HOLD" | "SELL"
    recommendation_reasons: Optional[list[str]]  # Human-readable justifications
    signal_details: Optional[dict]      # Raw numeric signal values used in the decision

    # ---- Reporting -----------------------------------------------------
    report: Optional[str]               # Final formatted text report

    # ---- Error handling --------------------------------------------------
    error: Optional[str]                # Populated whenever a node fails
    error_stage: Optional[str]          # Which node raised the error
