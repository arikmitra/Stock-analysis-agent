"""
stock_agent
===========
A LangGraph-based agent that analyzes a stock ticker using simple
technical indicators (SMA10, SMA20, RSI14) and produces a BUY/HOLD/SELL
recommendation with a formatted report.

Quick start:

    from stock_agent import analyze_stock

    result = analyze_stock("AAPL")
    print(result["report"])
"""

from .graph import analyze_stock, build_graph
from .state import AgentState

__all__ = ["analyze_stock", "build_graph", "AgentState"]
__version__ = "1.0.0"
