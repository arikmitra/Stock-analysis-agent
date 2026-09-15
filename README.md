# Stock Analysis Agent (LangGraph)

A from-scratch [LangGraph](https://github.com/langchain-ai/langgraph) agent that
takes a stock ticker, fetches 60 days of historical price data via `yfinance`,
computes technical indicators (10/20-day SMA, 14-day RSI), and produces a
rule-based **BUY / HOLD / SELL** recommendation with a formatted report.

This agent contains **no LLM calls** — every node is deterministic Python.
LangGraph is used purely as the orchestration/state-machine layer, which is a
valid and common use case: it gives you a typed shared state, explicit
control flow, conditional routing, and (if desired later) easy points to
swap in an LLM node for narrative commentary.

## Project Structure

```
stock_analysis_agent/
├── stock_agent/               # Core package
│   ├── __init__.py            # Public API: analyze_stock(), build_graph()
│   ├── state.py                # AgentState TypedDict (shared graph state)
│   ├── data_source.py          # yfinance wrapper + error handling + demo fallback
│   ├── indicators.py           # SMA / RSI math (pure functions, no LangGraph dep)
│   ├── recommendation.py       # BUY/HOLD/SELL scoring logic
│   ├── report.py               # Formats final state into a text report
│   ├── nodes.py                # LangGraph node functions
│   └── graph.py                # Wires nodes into a StateGraph
├── notebooks/
│   └── stock_analysis_demo.ipynb   # End-to-end interactive demonstration
├── tests/
│   └── test_indicators_and_recommendation.py   # Unit tests (pytest-compatible)
├── requirements.txt
└── README.md
```

## Architecture

```
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
build_report ──► END
```

Every node that can fail has a **conditional edge** that routes to a
dedicated `handle_error` node instead of raising an exception up through
the graph. This means `analyze_stock(...)` always returns cleanly with a
`report` key in the final state — whether analysis succeeded or failed —
so calling code never needs a try/except around the graph invocation
itself.

### Nodes

| Node | Responsibility |
|---|---|
| `validate_ticker` | Local, network-free sanity check on the ticker string (non-empty, valid characters, length). |
| `fetch_data` | Calls `yfinance` for 60 days of OHLCV history; captures company name/currency metadata. |
| `calculate_indicators` | Computes SMA-10, SMA-20, RSI-14 and confirms enough history exists. |
| `generate_recommendation` | Rule-based scoring model combining trend (SMA crossover) and momentum (RSI) signals. |
| `build_report` | Formats the final state into a readable text report. |
| `handle_error` | Terminal node for any failure; always produces a short, clear error report. |

## Technical Indicators

- **SMA (Simple Moving Average)**: `close.rolling(window).mean()`, computed for 10-day and 20-day windows.
- **RSI (Relative Strength Index, 14-day)**: Computed with Wilder's smoothing (an EMA with `alpha = 1/14`) on gains and losses, per the standard formula:
  `RSI = 100 - 100 / (1 + avg_gain / avg_loss)`.

Both are implemented from scratch in `indicators.py` on top of `pandas` — no external TA library — so the math is fully inspectable.

## Recommendation Logic

A simple, transparent scoring model (see `recommendation.py` for full detail):

| Signal | Condition | Score |
|---|---|---|
| Trend (SMA crossover) | SMA10 > SMA20 | +2 |
| Trend (SMA crossover) | SMA10 < SMA20 | −2 |
| Momentum (RSI) | RSI < 30 (oversold) | +1 |
| Momentum (RSI) | RSI > 70 (overbought) | −1 |

**Decision:** score ≥ 2 → `BUY`, score ≤ −2 → `SELL`, otherwise → `HOLD`.

This mirrors a common introductory technical-analysis approach: use a
moving-average crossover for trend direction, and RSI as a
mean-reversion check for entries/exits at extremes.

## Error Handling

The agent gracefully handles:
- **Invalid ticker format** (empty string, disallowed characters, too long) — caught before any network call.
- **Network/API failures** (timeouts, connectivity issues) — wrapped as `DataFetchError` and routed to `handle_error`.
- **Delisted / nonexistent tickers** — detected when `yfinance` returns an empty DataFrame.
- **Insufficient history** — if fewer than 20 trading days come back, the agent reports this explicitly rather than computing indicators on partial windows.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from stock_agent import analyze_stock

result = analyze_stock("AAPL")
print(result["report"])

print(result["recommendation"])          # "BUY" | "HOLD" | "SELL"
print(result["recommendation_reasons"])  # list of human-readable reasons
print(result["signal_details"])          # raw numeric values behind the decision
```

Or run the graph manually for more control:

```python
from stock_agent.graph import build_graph

graph = build_graph()
result = graph.invoke({"ticker": "MSFT"})
print(result["report"])
```

See `notebooks/stock_analysis_demo.ipynb` for a full interactive walkthrough,
including multiple tickers, error-handling demos, and indicator charts.

## Testing

```bash
python3 -m pytest tests/ -v
```

## Note on Demo/Offline Environments

`data_source.py` will always attempt a real `yfinance` fetch first. Only if
that fails **and** the caller has explicitly enabled
`use_synthetic_fallback=True` will it fall back to a seeded synthetic price
series, purely so the notebook demo can run in network-restricted sandboxes.
The synthetic data is clearly labeled ("SYNTHETIC DEMO DATA") in the company
name field and flagged with a warning banner in the report — it is never
silently substituted for real data in a normal environment with internet
access.

## Disclaimer

This tool is for educational purposes only. It is not financial advice.
Simple moving averages and RSI are basic, well-known indicators with known
limitations (lag, false signals in choppy markets, no consideration of
fundamentals, news, or macro conditions). Always do your own research and
consult a licensed financial advisor before making investment decisions.
