# Stock Analysis Agent (LangGraph)

A from-scratch [LangGraph](https://github.com/langchain-ai/langgraph) agent that
takes a stock ticker, fetches historical price data via `yfinance`,
computes technical indicators (10/20-day SMA, 14-day RSI, plus any
user-registered custom indicators), and produces a
**BUY / HOLD / SELL** recommendation with a formatted report.

**The deterministic path is always the default.** Every recommendation
starts from plain, auditable Python scoring logic in `recommendation.py`,
computed on every single run with no exceptions. Two separate,
independent LLM-touching features can optionally be layered on top —
both opt-in, both OpenAI/Gemini-swappable, both never breaking the
deterministic report if they fail:

- **Narrative commentary** (`llm_provider=...`): an LLM explains an
  already-final recommendation in plain language. It can never change
  the call — see `llm_advisor.py`.
- **AI-generated recommendations** (`recommendation_mode="ai"`,
  requires `llm_provider` too): an LLM independently evaluates the same
  indicator data and can produce its *own* BUY/HOLD/SELL call, which
  becomes the active recommendation shown to the user — but the
  deterministic call is always computed too and stays visible in the
  report for comparison, and any AI failure falls back to the
  deterministic call automatically. See `ai_recommendation.py`.

Leave both off (the default) and the agent behaves exactly as a pure
rule-based system, with zero LLM calls anywhere in the pipeline. See
[Design Principles](#design-principles) below for how this and the
agent's other extension points fit together.

## Project Structure

```
stock_analysis_agent/
├── stock_agent/               # Core package
│   ├── __init__.py            # Public API: analyze_stock(), build_graph(), run_backtest(), etc.
│   ├── state.py                # AgentState TypedDict (shared graph state)
│   ├── data_source.py          # yfinance wrapper + error handling + demo fallback
│   ├── indicators.py           # SMA / RSI math + custom indicator registry
│   ├── recommendation.py       # BUY/HOLD/SELL scoring logic (score_row is the single source of truth, always deterministic, always computed)
│   ├── ai_recommendation.py    # Optional, swappable OpenAI/Gemini recommendation DECISION path (opt-in alternative to recommendation.py)
│   ├── llm_advisor.py          # Optional, swappable OpenAI/Gemini narrative commentary (explains a decision, never makes one)
│   ├── feedback.py             # User feedback recording (JSONL) + aggregate summaries
│   ├── backtest.py             # Historical backtest of the strategy vs. buy-and-hold
│   ├── report.py               # Formats final state into a text report
│   ├── nodes.py                # LangGraph node functions
│   └── graph.py                # Wires nodes into a StateGraph
├── notebooks/
│   ├── stock_analysis_demo.ipynb   # End-to-end interactive demonstration
│   └── stock_analysis_demo.py      # Same walkthrough as a standalone script
├── tests/
│   ├── test_indicators_and_recommendation.py   # SMA/RSI + scoring logic on synthetic series
│   ├── test_backtest.py                        # Backtesting engine correctness on synthetic series
│   ├── test_graph.py                            # The compiled LangGraph itself: routing & node traversal
│   ├── test_real_data.py                        # Ground-truth validation against real historical data
│   ├── test_llm_advisor.py                      # LLMProvider abstraction, prompts, error paths (FakeProvider, no network)
│   ├── test_graph_llm_routing.py                # Graph-level: LLM commentary node skipped/invoked correctly, failure isolation
│   ├── test_ai_recommendation.py                # AI recommendation module: strict response parsing, prompt construction (FakeProvider, no network)
│   ├── test_graph_ai_recommendation_routing.py  # Graph-level: AI recommendation node skipped/invoked correctly, fallback-to-deterministic guarantee
│   ├── test_custom_indicators.py                # Custom indicator registry + report integration
│   ├── test_feedback.py                         # Feedback recording/loading/summarizing
│   └── test_configurable_window.py              # lookback_days / history_rows validation and threading
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
generate_recommendation  (ALWAYS deterministic, ALWAYS computed —
  │                        seeds the active `recommendation` field)
  │
  ├──(recommendation_mode != "ai", the default)───────────────┐
  │                                                             │
  └──(recommendation_mode == "ai")──► generate_ai_recommendation│
                                        (may OVERRIDE the active  │
                                        `recommendation`; falls    │
                                        back to the deterministic   │
                                        call on ANY failure — never  │
                                        routes to handle_error)       │
                                              │                        │
                                              ▼                        ▼
                                       ┌────────────────────────────────┐
                                       │  llm_provider set?               │
                                       └──────┬─────────────────┬────────┘
                                         no    │             yes │
                                              ▼                 ▼
                                        build_report ◄── generate_llm_commentary
                                              │            (explains whichever call
                                              ▼             is active; best-effort)
                                             END
```

Every node that can fail has a **conditional edge** that routes to a
dedicated `handle_error` node instead of raising an exception up through
the graph. This means `analyze_stock(...)` always returns cleanly with a
`report` key in the final state — whether analysis succeeded or failed —
so calling code never needs a try/except around the graph invocation
itself.

Two independent, both-optional nodes follow `generate_recommendation`,
each skipped entirely unless explicitly requested, and neither one
following the fail-to-`handle_error` pattern:

- **`generate_ai_recommendation`** — reached only when
  `recommendation_mode == "ai"`. It is the *one* node in the graph
  allowed to override the active `recommendation` after the
  deterministic scorer set it. If it fails for any reason (missing key,
  network error, an LLM response that can't be confidently parsed into
  BUY/HOLD/SELL), it does not route to `handle_error` — it leaves
  `recommendation` exactly as the deterministic node set it. The
  deterministic call (`deterministic_recommendation`) is never touched
  by this node either way, so the rule-based baseline stays visible in
  the final report regardless of which mode is active or whether it
  succeeded.
- **`generate_llm_commentary`** — reached only when `llm_provider` is
  set. It can only *explain* whichever call ended up active
  (deterministic or AI) — it never changes `recommendation`. A failure
  here is recorded in `llm_error` but, again, never blocks the report.

Both nodes reuse the same swappable `LLMProvider` transport (see
`llm_advisor.py`), but serve different purposes: one can decide, the
other can only explain.

### Nodes

| Node | Responsibility |
|---|---|
| `validate_ticker` | Local, network-free sanity check on the ticker string and, if provided, `lookback_days` / `recommendation_mode`. |
| `fetch_data` | Calls `yfinance` for `lookback_days` (default 60) of OHLCV history; captures company name/currency metadata. |
| `calculate_indicators` | Computes SMA-10, SMA-20, RSI-14, any registered custom indicators, and confirms enough history exists. |
| `generate_recommendation` | Rule-based scoring model combining trend (SMA crossover) and momentum (RSI) signals. **Always runs, on every request**, and seeds the active recommendation — regardless of LLM/AI configuration. |
| `generate_ai_recommendation` | *Opt-in* (`recommendation_mode="ai"`). Lets an LLM independently evaluate the same indicators and produce its own call, which becomes the active recommendation on success. Falls back to the deterministic call on any failure; skipped entirely when not requested. |
| `generate_llm_commentary` | *Opt-in* (`llm_provider=...`). Explains the active recommendation (deterministic or AI) in plain language. Skipped entirely when `llm_provider` is unset; failures degrade gracefully rather than blocking the report. |
| `build_report` | Formats the final state (active recommendation, deterministic baseline when AI mode is used, LLM commentary if any, and any custom indicator columns) into a readable text report. |
| `handle_error` | Terminal node for any *analysis* failure; always produces a short, clear error report. |

## Design Principles

The extension points added on top of the original agent (swappable LLM
commentary, an optional AI-driven recommendation path, custom
indicators, user feedback, and a configurable history window) share a
common set of design rules that keep the core deterministic agent
unchanged and unaffected by any of them:

1. **The deterministic path is always the default, and always complete
   on its own.** `analyze_stock("AAPL")` with no extra arguments behaves
   identically to the pre-extension version of this agent: same
   recommendation logic, same 60-day/10-row defaults, no LLM calls, no
   custom indicators, no feedback prompts. Every new parameter
   (`llm_provider`, `recommendation_mode`, `lookback_days`,
   `history_rows`, custom indicator registration) is additive and
   opt-in — nothing you must configure to get a working agent.

2. **The deterministic scorer is computed unconditionally, every run,
   no exceptions.** Unlike the optional features layered on top,
   `generate_recommendation` (the rule-based scorer) is not behind any
   conditional edge — it always executes and always populates
   `deterministic_recommendation`. This is true even when
   `recommendation_mode="ai"` is requested and succeeds: the AI's call
   becomes the *active* `recommendation`, but the rule-based result is
   computed regardless and remains available in
   `deterministic_recommendation` for comparison. The rule-based
   baseline is never skipped, and never hidden.

3. **Optional features degrade gracefully, never destructively.** A
   missing LLM API key, an unsupported provider name, an AI response
   that can't be confidently parsed into BUY/HOLD/SELL, or a broken
   custom indicator function all produce a clear, localized error
   (`llm_error`, `ai_recommendation_error`, a wrapped `ValueError`
   naming the offending indicator) — they do not prevent a valid report
   from being generated, and they do not route to the graph's
   `handle_error` terminal node the way a genuine analysis failure (bad
   ticker, no data) does. Those are two different kinds of failure and
   the graph treats them differently on purpose: one means "the
   analysis itself failed," the other means "an optional add-on failed,
   but the analysis you asked for still succeeded" (for AI
   recommendations specifically, this means falling back to the
   deterministic call rather than showing nothing or guessing).

4. **An LLM can only ever affect the ACTIVE recommendation, and only
   the AI-recommendation path can do even that.** Narrative commentary
   (`llm_provider` alone) can never change `recommendation` under any
   circumstances — its node runs strictly *after* the active call is
   finalized, receives it as fixed input, and its prompt explicitly
   instructs it not to contradict that call. The AI-recommendation path
   (`recommendation_mode="ai"`) is the sole, explicit exception: it is
   allowed to override the active `recommendation` field, but only when
   explicitly requested, only with its own independently-parsed and
   validated output, and never by modifying or feeding back into
   `recommendation.py`'s scoring logic itself — the deterministic
   scorer has no knowledge that an AI path exists.

5. **New capabilities are additive to the shared state, not replacements
   for it.** `AgentState` (see `state.py`) gained new optional fields —
   `lookback_days`, `llm_provider`, `recommendation_mode`,
   `ai_recommendation`, `feedback_rating`, etc. — rather than new
   required fields or restructured existing ones. A caller or test
   written against the pre-extension `AgentState` schema still works
   unmodified; it simply never sees the new keys populated.

6. **Every extension point is designed to be extended again.** Adding a
   third LLM provider means implementing one method (`LLMProvider.generate`)
   and registering it in `get_provider()` — no changes to `nodes.py` or
   `graph.py`, and the change applies to both `llm_advisor.py` and
   `ai_recommendation.py` simultaneously, since they share the same
   provider transport. Adding a new built-in indicator or a new report
   section follows the same "isolated module, thin integration point"
   pattern used throughout the codebase.

## Technical Indicators

- **SMA (Simple Moving Average)**: `close.rolling(window).mean()`, computed for 10-day and 20-day windows.
- **RSI (Relative Strength Index, 14-day)**: Computed with Wilder's smoothing (an EMA with `alpha = 1/14`) on gains and losses, per the standard formula:
  `RSI = 100 - 100 / (1 + avg_gain / avg_loss)`.

Both are implemented from scratch in `indicators.py` on top of `pandas` — no external TA library — so the math is fully inspectable.

Beyond these two built-ins, any number of custom indicators can be
registered at runtime via `register_custom_indicator()` — see
[Adding a custom indicator](#adding-a-custom-indicator) below. They are
computed alongside SMA/RSI and rendered in the report automatically, but
(as of this version) do not yet feed into the recommendation scoring
described next — that logic only consults SMA10/SMA20/RSI14.

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

This is the logic behind `deterministic_recommendation`, which is always
computed. An optional AI-generated alternative can independently
evaluate the same signals and become the *active* recommendation instead
— see [AI-generated recommendations](#ai-generated-recommendations-openai--gemini--deterministic-path-stays-default) below.

## Backtesting

The recommendation logic is deliberately simple and rule-based, which
means it can (and should) be validated against history rather than just
trusted. `stock_agent/backtest.py` simulates the strategy's BUY/HOLD/SELL
signals as a long/flat position over a historical price series and
compares it to buy-and-hold on the same window:

```python
from stock_agent import run_backtest, format_backtest_report
from stock_agent.data_source import fetch_history

df, meta = fetch_history("AAPL", period_days=730)  # ~2 years, needed for a meaningful backtest
result = run_backtest(df, ticker="AAPL")
print(format_backtest_report(result))
```

Reports total/annualized return, max drawdown, Sharpe ratio, win rate,
and trade count, alongside a day-by-day equity curve for charting. The
backtester calls the exact same `score_row()` function the live agent
uses, so the two can never silently drift apart — this is enforced by a
dedicated test in `tests/test_backtest.py`.

**What this backtest does NOT model** (stated here deliberately, not
buried in fine print): transaction costs, slippage, taxes, or bid/ask
spread; position sizing (it's all-in or all-out); realistic fill timing
(it assumes same-day-close fills); portfolio effects across multiple
holdings. Treat backtest results as a sanity check on the strategy's
logic, not as evidence it will be profitable going forward.

## Error Handling

The agent distinguishes two kinds of failure, and handles them differently:

**Analysis failures** (the core pipeline couldn't produce a
recommendation) route to the graph's `handle_error` terminal node and
produce a `STOCK ANALYSIS FAILED` report instead of a normal one:
- **Invalid ticker format** (empty string, disallowed characters, too long) — caught before any network call.
- **Invalid `lookback_days`** (non-integer, or too small to populate a 20-day SMA) — also caught before any network call.
- **Invalid `recommendation_mode`** (anything other than `"deterministic"` or `"ai"`) — also caught before any network call.
- **Network/API failures** (timeouts, connectivity issues) — wrapped as `DataFetchError` and routed to `handle_error`.
- **Delisted / nonexistent tickers** — detected when `yfinance` returns an empty DataFrame.
- **Insufficient history** — if fewer than 20 trading days come back, the agent reports this explicitly rather than computing indicators on partial windows.
- **A broken custom indicator** — a registered indicator function that raises, or returns something other than a `pd.Series`, fails with a message naming the specific indicator.

**Optional-feature failures** (an add-on couldn't complete, but the
analysis itself succeeded) are captured in a dedicated state field and
never prevent the normal report from being produced:
- **LLM commentary unavailable** (missing API key, unsupported provider name, SDK not installed, network/API error) — recorded in `result["llm_error"]`; `result["report"]` still contains the full recommendation, with a short "(AI commentary was requested but unavailable: ...)" notice in place of the commentary section.
- **AI recommendation unavailable** (missing API key, unsupported provider name, network/API error, or a response that can't be confidently parsed into BUY/HOLD/SELL) — recorded in `result["ai_recommendation_error"]`; `result["recommendation"]` automatically falls back to the deterministic call (`result["recommendation_source"]` reads `"deterministic"` in this case), and the report includes a short notice explaining that the fallback happened and why.
- **Feedback recording failure** (e.g. disk permissions) — recorded in `result["error"]`/`result["error_stage"]` only if you're using `record_feedback_node` inside a custom graph; the standalone `submit_feedback()` convenience wrapper raises `FeedbackError` directly instead, since recording feedback is a direct user action where silent failure would be worse than an exception.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from stock_agent import analyze_stock

result = analyze_stock("AAPL")
print(result["report"])

print(result["recommendation"])                # "BUY" | "HOLD" | "SELL" -- the ACTIVE call
print(result["recommendation_source"])          # "deterministic" (always, unless AI mode is used and succeeds)
print(result["deterministic_recommendation"])   # the rule-based call -- always computed, always present
print(result["recommendation_reasons"])         # human-readable reasons for the active call
print(result["signal_details"])                 # raw numeric values behind the decision
```

Or run the graph manually for more control:

```python
from stock_agent.graph import build_graph

graph = build_graph()
result = graph.invoke({"ticker": "MSFT"})
print(result["report"])
```

### Configuring the history window

Two independent knobs, both optional (defaults match the agent's original
behavior — 60 calendar days fetched, last 10 trading days displayed):

```python
result = analyze_stock(
    "AAPL",
    lookback_days=120,   # calendar days of history to fetch (default: 60)
    history_rows=15,     # rows shown in the report's history table (default: 10)
)
```

`lookback_days` must be at least ~30 (enough calendar days to populate a
20-day SMA and 14-day RSI); smaller values fail fast with a clear
validation error before any network call is made.

### Optional LLM commentary (OpenAI / Gemini) — deterministic path stays default

The recommendation itself is **always** produced by the deterministic
rule-based scoring in `recommendation.py` — nothing below changes that.
Setting `llm_provider` adds a supplementary narrative explanation of the
already-final call; leaving it unset (the default) means the LLM node
never runs at all, so existing callers see zero behavior change.

```python
result = analyze_stock("AAPL", llm_provider="openai")   # or "gemini"
print(result["recommendation"])    # still fully deterministic
print(result["llm_commentary"])    # narrative explanation, or None if unavailable/unset
print(result["llm_error"])         # populated if the LLM call failed; never blocks the report
```

Requires `OPENAI_API_KEY` (for `"openai"`) or `GEMINI_API_KEY` /
`GOOGLE_API_KEY` (for `"gemini"`) to be set in the environment, and the
corresponding SDK installed (`pip install openai` or
`pip install google-genai` — see `requirements.txt`). If the key is
missing, the SDK isn't installed, or the API call fails for any reason,
the deterministic report is still produced in full; only `llm_commentary`
stays empty and `llm_error` explains why. Override the model with
`llm_model="gpt-4o"` / `llm_model="gemini-1.5-pro"`, etc.

Adding a third provider means implementing the single-method
`LLMProvider` protocol in `stock_agent/llm_advisor.py` and registering it
in `get_provider()` — nothing else in the agent needs to change.

### AI-generated recommendations (OpenAI / Gemini) — deterministic path stays default

A separate, independent feature from narrative commentary above: instead
of just *explaining* the deterministic call, an LLM can *make its own*
BUY/HOLD/SELL call from the same indicator data. This requires opting
into both `recommendation_mode="ai"` **and** `llm_provider` — leaving
either unset means this path is skipped entirely and behavior is
identical to not having this feature at all.

```python
result = analyze_stock("AAPL", recommendation_mode="ai", llm_provider="openai")

print(result["recommendation"])                # the AI's call, if it succeeded and parsed cleanly
print(result["recommendation_source"])          # "ai" on success, or "deterministic" on any fallback
print(result["deterministic_recommendation"])   # the rule-based call -- ALWAYS present, for comparison
print(result["ai_recommendation_used"])         # True/False -- whether the AI call actually won
print(result["ai_recommendation_error"])        # populated on fallback, explains why
```

The deterministic scorer always runs first and is never skipped — the
AI is prompted with the same SMA/RSI values (plus the deterministic
call as optional reference context it's explicitly told it may
disagree with) and asked to form its own independent judgment, not
asked to rubber-stamp the rule-based result.

**The fallback guarantee:** if the AI call fails for any reason —
missing API key, network error, or (just as importantly) a response
that can't be confidently parsed into an unambiguous BUY/HOLD/SELL —
the agent automatically falls back to the deterministic call rather
than failing the analysis or guessing at a low-confidence answer.
`ai_recommendation.py`'s response parser is deliberately strict: it
looks for a clearly-labeled `RECOMMENDATION: <call>` line, falls back to
scanning for a single unambiguous BUY/HOLD/SELL token if that's absent,
and raises rather than guessing if the model's response mentions
multiple possible calls with no way to disambiguate (e.g. "this could
be a BUY or maybe a SELL"). A fallback is never silent — check
`ai_recommendation_error` / `recommendation_source` to see whether it
happened.

**The report always shows the rule-based baseline.** Whenever AI mode
successfully overrides the active call, the formatted report includes a
"DETERMINISTIC BASELINE" section stating explicitly whether the
rule-based scorer agreed or differed with the AI's call — the deterministic
result is never hidden just because AI mode is in use. When narrative
commentary is also enabled (`llm_provider` alone, alongside
`recommendation_mode="ai"`), it explains whichever call ended up active
(the AI's, if it won).

### Adding a custom indicator

```python
from stock_agent.indicators import register_custom_indicator

def sma_50(df):
    return df["Close"].rolling(50, min_periods=50).mean()

register_custom_indicator("SMA_50", sma_50)

result = analyze_stock("AAPL", lookback_days=120)  # need 50+ trading days for SMA_50 to populate
print(result["report"])  # SMA_50 now appears in the indicator summary and history table
```

A custom indicator function receives the OHLCV DataFrame and must return
a `pd.Series` aligned to the same dates. Registered indicators apply to
every subsequent `analyze_stock()` call in the process — use
`unregister_custom_indicator("SMA_50")` or `clear_custom_indicators()` to
remove them. Note: custom indicators currently appear in the report for
visibility, but do not (yet) feed into the BUY/HOLD/SELL scoring itself —
that logic still only consults SMA10/SMA20/RSI14 (see
`recommendation.py` if you want to extend the scoring to use them).

### Recording user feedback

```python
from stock_agent import analyze_stock, submit_feedback, summarize_feedback

result = analyze_stock("AAPL")
submit_feedback(result, rating="helpful", comment="Matched my own read of the chart")

print(summarize_feedback())  # {"total": 1, "helpful": 1, "not_helpful": 0, "helpful_rate_pct": 100.0, ...}
```

Feedback is stored as append-only JSONL at `~/.stock_agent/feedback.jsonl`
by default (override via the `path=` argument on `record_feedback` /
`load_feedback` / `summarize_feedback`), and each record captures the
exact `signal_details` the recommendation was based on — so feedback can
later be analyzed against the specific indicator values that produced it
(e.g. "does feedback get worse when RSI is near the 70 boundary?").
`submit_feedback` only accepts feedback on a successful result — attaching
feedback to a failed analysis raises `ValueError`, since there's no
recommendation to rate.

See `notebooks/stock_analysis_demo.ipynb` for a full interactive walkthrough,
including multiple tickers, error-handling demos, and indicator charts.

### Running the demo without Jupyter

The same walkthrough is also available as a plain script, for environments
without Jupyter or when you just want to run it from the command line:

```bash
python3 notebooks/stock_analysis_demo.py
```

Charts that would normally render inline in the notebook are instead saved
as PNG files to `notebooks/demo_output/`. Useful flags:

```bash
python3 notebooks/stock_analysis_demo.py --ticker MSFT   # ticker for the "try it yourself" section (default: NVDA)
python3 notebooks/stock_analysis_demo.py --skip-tests     # skip running the pytest suite at the end
python3 notebooks/stock_analysis_demo.py --no-charts       # skip generating/saving charts entirely
```

## Testing

```bash
python3 -m pytest tests/ -v
```

The suite spans eleven files, each testing a different layer:

| File | What it tests |
|---|---|
| `test_indicators_and_recommendation.py` | SMA/RSI math and recommendation scoring on synthetic series (known boundary cases: all-gains, all-losses, flat price) |
| `test_backtest.py` | Backtesting engine correctness — equity curve arithmetic, max drawdown, and an explicit check that returns aren't computed with look-ahead bias |
| `test_graph.py` | The compiled LangGraph `StateGraph` itself — asserts on actual node traversal order via `.stream()`, not just the routing functions in isolation. Confirms an invalid ticker never triggers a network call and a failed fetch never reaches the indicator/recommendation nodes |
| `test_real_data.py` | **Ground-truth validation against real historical AAPL data**, fetched live over the network from a public dataset, cross-checked against independently re-implemented (non-pandas) reference SMA/RSI calculations, plus a pinned-value regression fixture on a specific historical date |
| `test_llm_advisor.py` | The `LLMProvider` abstraction — provider factory, prompt construction (including the explicit "don't change the recommendation" instruction), and error paths (missing API key, unsupported provider), all using a `FakeProvider` with no real network/API calls |
| `test_graph_llm_routing.py` | The graph's conditional LLM-commentary edge — confirms the commentary node is **never invoked** (not just absent from logs) when `llm_provider` is unset, confirms it runs when set, and confirms an LLM failure still reaches `build_report` rather than `handle_error` |
| `test_ai_recommendation.py` | The AI recommendation module's defensive response parser — strict-format happy path, markdown tolerance, word-boundary correctness (rejecting false matches like "SELLING"), and refusing to guess on genuinely ambiguous multi-token responses; plus prompt construction, all using a `FakeProvider` with no real network/API calls |
| `test_graph_ai_recommendation_routing.py` | The graph's conditional AI-recommendation edge — confirms the node is **never invoked** when `recommendation_mode` isn't `"ai"`, confirms it can override the active call on success while leaving `deterministic_recommendation` untouched, and confirms transport failures / unparseable responses / missing-provider misconfiguration all fall back to the deterministic call **without** reaching `handle_error` |
| `test_custom_indicators.py` | The custom indicator registry — registration/collision/removal, correct values flowing through `add_all_indicators`, clear error wrapping when a user's indicator function is broken or misshapen |
| `test_feedback.py` | Feedback recording/loading/summarizing (JSONL round-trips, validation, aggregate stats), plus the `submit_feedback()` and `record_feedback_node` integration points |
| `test_configurable_window.py` | `lookback_days` and `history_rows` validation and threading through `fetch_data`/`calculate_indicators`/`format_report`, including the property that the two are independent of each other |

`test_real_data.py` requires network access (skips gracefully if
unavailable) and includes an optional yfinance cross-check that also
skips gracefully rather than failing if Yahoo Finance isn't reachable in
your environment. `test_llm_advisor.py`, `test_graph_llm_routing.py`,
`test_ai_recommendation.py`, and `test_graph_ai_recommendation_routing.py`
never make real network/API calls — they use a `FakeProvider` and
monkeypatched fixtures throughout, so they run the same with or without
`OPENAI_API_KEY`/`GEMINI_API_KEY` set.

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
fundamentals, news, or macro conditions). When `recommendation_mode="ai"`
is used, the active recommendation may instead come from an LLM's own
judgment rather than the rule-based scorer — LLM output can be
inconsistent, non-reproducible, or wrong even when it parses cleanly into
a valid BUY/HOLD/SELL call; the deterministic baseline shown alongside it
is there specifically so you can sanity-check it. Always do your own
research and consult a licensed financial advisor before making
investment decisions.
