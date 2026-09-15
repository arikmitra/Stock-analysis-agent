"""
indicators.py
-------------
Pure, dependency-light technical indicator calculations.

Kept separate from the LangGraph nodes so they can be unit tested in
isolation and reused outside the agent (e.g. in a backtester or a
notebook) without importing LangGraph at all.
"""

from __future__ import annotations

import pandas as pd


def simple_moving_average(close: pd.Series, window: int) -> pd.Series:
    """Compute a Simple Moving Average (SMA).

    Args:
        close: Series of closing prices, indexed by date.
        window: Number of periods to average over (e.g. 10 or 20).

    Returns:
        A Series of the same length as `close`. The first `window - 1`
        entries are NaN because there isn't enough history yet to form
        a full window.
    """
    return close.rolling(window=window, min_periods=window).mean()


def relative_strength_index(close: pd.Series, window: int = 14) -> pd.Series:
    """Compute the Relative Strength Index (RSI) using Wilder's smoothing.

    RSI oscillates between 0 and 100:
        RSI < 30  -> commonly considered "oversold" (potential BUY signal)
        RSI > 70  -> commonly considered "overbought" (potential SELL signal)

    Formula:
        delta      = close.diff()
        gain       = positive deltas, 0 elsewhere
        loss       = absolute value of negative deltas, 0 elsewhere
        avg_gain   = Wilder-smoothed rolling average of gain over `window`
        avg_loss   = Wilder-smoothed rolling average of loss over `window`
        RS         = avg_gain / avg_loss
        RSI        = 100 - (100 / (1 + RS))

    Args:
        close: Series of closing prices, indexed by date.
        window: Lookback period, 14 by default (the industry-standard value).

    Returns:
        A Series of RSI values. The first `window` entries will be NaN.
    """
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's smoothing is an exponential moving average with alpha = 1/window.
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Where avg_loss is 0 (no down days in the window), RSI should be 100,
    # not NaN/inf from a division by zero.
    rsi = rsi.where(avg_loss != 0, 100.0)
    # Where avg_gain is also 0 (flat price), RSI is conventionally 50.
    rsi = rsi.where(~((avg_gain == 0) & (avg_loss == 0)), 50.0)

    return rsi


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Attach SMA10, SMA20, and RSI14 columns to an OHLCV DataFrame.

    Args:
        df: DataFrame with at least a "Close" column, sorted ascending by date.

    Returns:
        A copy of `df` with three new columns: "SMA_10", "SMA_20", "RSI_14".
    """
    if "Close" not in df.columns:
        raise ValueError("DataFrame must contain a 'Close' column")

    out = df.copy()
    out["SMA_10"] = simple_moving_average(out["Close"], window=10)
    out["SMA_20"] = simple_moving_average(out["Close"], window=20)
    out["RSI_14"] = relative_strength_index(out["Close"], window=14)
    return out
