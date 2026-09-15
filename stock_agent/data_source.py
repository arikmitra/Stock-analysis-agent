"""
data_source.py
---------------
Wraps yfinance so the rest of the agent never talks to the network directly.

This module exposes a single function, `fetch_history`, along with a custom
`DataFetchError` so the LangGraph node layer can catch one well-defined
exception type instead of guessing at yfinance's various failure modes
(empty DataFrame, network exception, delisted ticker, etc).

A `synthetic` fallback generator is also included. It is NOT meant to be
used for real trading decisions -- it exists purely so this project's demo
notebook can run end-to-end in environments without outbound internet
access (e.g. sandboxes, CI). In any environment with normal internet
access, `fetch_history` will always attempt yfinance first.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

# yfinance logs network/parsing failures at ERROR level by default, which is
# redundant noise here since we already surface failures via DataFetchError.
logging.getLogger("yfinance").setLevel(logging.CRITICAL)


class DataFetchError(Exception):
    """Raised when historical stock data cannot be retrieved for a ticker."""


def _validate_ticker_format(ticker: str) -> str:
    """Lightweight sanity check on ticker formatting before hitting the network.

    Args:
        ticker: Raw ticker string.

    Returns:
        The normalized (stripped, upper-cased) ticker.

    Raises:
        DataFetchError: if the ticker is empty or contains characters that
            no real exchange ticker would contain.
    """
    if ticker is None:
        raise DataFetchError("No ticker symbol was provided.")

    normalized = ticker.strip().upper()

    if not normalized:
        raise DataFetchError("Ticker symbol cannot be empty.")

    # Real tickers are short and alphanumeric, sometimes with '.' or '-'
    # (e.g. "BRK-B", "RY.TO"). Reject anything wildly outside that shape.
    allowed_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-")
    if not set(normalized) <= allowed_chars or len(normalized) > 10:
        raise DataFetchError(
            f"'{ticker}' does not look like a valid ticker symbol."
        )

    return normalized


def fetch_history(
    ticker: str,
    period_days: int = 60,
    use_synthetic_fallback: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Fetch historical OHLCV data for a ticker.

    Args:
        ticker: Stock ticker symbol, e.g. "AAPL".
        period_days: Approximate number of calendar days of history to request.
            60 calendar days comfortably yields 40+ trading days, which is
            enough for a 20-day SMA and 14-day RSI to fully populate.
        use_synthetic_fallback: If True, and the real fetch fails or returns
            no data, generate a plausible synthetic price series instead of
            raising. Intended ONLY for demos/tests in network-restricted
            environments -- never enable this for real investment decisions.

    Returns:
        A tuple of (DataFrame, metadata_dict).
        DataFrame is indexed by date with columns:
            Open, High, Low, Close, Volume
        metadata_dict contains best-effort "company_name", "currency",
        and "source" ("yfinance" or "synthetic").

    Raises:
        DataFetchError: if data cannot be obtained and no fallback is used
            (or the fallback is also disabled).
    """
    normalized = _validate_ticker_format(ticker)

    try:
        df, meta = _fetch_from_yfinance(normalized, period_days)
        return df, meta
    except DataFetchError:
        if use_synthetic_fallback:
            df, meta = _generate_synthetic_history(normalized, period_days)
            return df, meta
        raise


def _fetch_from_yfinance(ticker: str, period_days: int) -> tuple[pd.DataFrame, dict]:
    """Attempt to fetch real data via yfinance. Raises DataFetchError on any failure."""
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - environment issue
        raise DataFetchError(
            "The 'yfinance' package is not installed. Run `pip install yfinance`."
        ) from exc

    try:
        ticker_obj = yf.Ticker(ticker)
        df = ticker_obj.history(period=f"{period_days}d", auto_adjust=True)
    except Exception as exc:
        raise DataFetchError(
            f"Network or API error while fetching data for '{ticker}': {exc}"
        ) from exc

    if df is None or df.empty:
        raise DataFetchError(
            f"No historical data returned for '{ticker}'. It may be an "
            f"invalid or delisted ticker."
        )

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.sort_index()

    meta = {"source": "yfinance"}
    try:
        info = ticker_obj.info
        meta["company_name"] = info.get("longName") or info.get("shortName")
        meta["currency"] = info.get("currency", "USD")
    except Exception:
        # `.info` is a separate, sometimes-flaky network call. If it fails,
        # we still have the price history, so degrade gracefully.
        meta["company_name"] = None
        meta["currency"] = "USD"

    return df, meta


def _generate_synthetic_history(ticker: str, period_days: int) -> tuple[pd.DataFrame, dict]:
    """Generate a plausible, deterministic-per-ticker synthetic price series.

    Uses a seeded random walk so the same ticker always produces the same
    demo data (useful for reproducible notebook output), while different
    tickers produce visibly different price paths.

    NOT real market data. For demonstration purposes only.
    """
    approx_days = max(45, int(period_days * (5 / 7)))  # approx weekdays only
    seed = abs(hash(ticker)) % (2**32)
    rng = np.random.default_rng(seed)

    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=approx_days)
    trading_days = len(dates)  # bdate_range's actual length may differ slightly

    start_price = rng.uniform(50, 400)
    daily_returns = rng.normal(loc=0.0004, scale=0.018, size=trading_days)
    close = start_price * np.cumprod(1 + daily_returns)

    high = close * (1 + rng.uniform(0.001, 0.02, size=trading_days))
    low = close * (1 - rng.uniform(0.001, 0.02, size=trading_days))
    open_ = low + (high - low) * rng.uniform(0.2, 0.8, size=trading_days)
    volume = rng.integers(1_000_000, 50_000_000, size=trading_days)

    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )

    meta = {
        "source": "synthetic",
        "company_name": f"{ticker} (SYNTHETIC DEMO DATA)",
        "currency": "USD",
    }
    return df, meta
