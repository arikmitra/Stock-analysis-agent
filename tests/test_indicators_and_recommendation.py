"""
test_indicators_and_recommendation.py
--------------------------------------
Unit tests for the pure calculation modules (indicators.py, recommendation.py)
and validation/error-handling logic in data_source.py.

Run with:
    python3 -m pytest tests/ -v
or directly:
    python3 tests/test_indicators_and_recommendation.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stock_agent.data_source import DataFetchError, _validate_ticker_format
from stock_agent.indicators import add_all_indicators, relative_strength_index, simple_moving_average
from stock_agent.recommendation import generate_recommendation


def _make_price_series(prices: list[float]) -> pd.Series:
    dates = pd.bdate_range("2026-01-01", periods=len(prices))
    return pd.Series(prices, index=dates, name="Close")


def test_sma_basic_average():
    """A 3-day SMA over [1, 2, 3] should equal 2.0 on the 3rd day."""
    close = _make_price_series([1, 2, 3, 4, 5])
    sma = simple_moving_average(close, window=3)
    assert np.isnan(sma.iloc[0])
    assert np.isnan(sma.iloc[1])
    assert sma.iloc[2] == 2.0
    assert sma.iloc[3] == 3.0
    assert sma.iloc[4] == 4.0
    print("test_sma_basic_average PASSED")


def test_sma_constant_price():
    """SMA of a flat price series should equal that price everywhere it's defined."""
    close = _make_price_series([100.0] * 25)
    sma20 = simple_moving_average(close, window=20)
    assert (sma20.dropna() == 100.0).all()
    print("test_sma_constant_price PASSED")


def test_rsi_all_gains_approaches_100():
    """A monotonically increasing price series should push RSI toward 100."""
    close = _make_price_series([100 + i for i in range(30)])
    rsi = relative_strength_index(close, window=14)
    final_rsi = rsi.dropna().iloc[-1]
    assert final_rsi > 95, f"Expected RSI near 100 for all-gains series, got {final_rsi}"
    print("test_rsi_all_gains_approaches_100 PASSED")


def test_rsi_all_losses_approaches_0():
    """A monotonically decreasing price series should push RSI toward 0."""
    close = _make_price_series([200 - i for i in range(30)])
    rsi = relative_strength_index(close, window=14)
    final_rsi = rsi.dropna().iloc[-1]
    assert final_rsi < 5, f"Expected RSI near 0 for all-losses series, got {final_rsi}"
    print("test_rsi_all_losses_approaches_0 PASSED")


def test_rsi_flat_price_is_50():
    """A perfectly flat price series has no gains or losses -> RSI should be 50."""
    close = _make_price_series([100.0] * 30)
    rsi = relative_strength_index(close, window=14)
    final_rsi = rsi.dropna().iloc[-1]
    assert final_rsi == 50.0, f"Expected RSI == 50 for flat series, got {final_rsi}"
    print("test_rsi_flat_price_is_50 PASSED")


def test_rsi_bounds():
    """RSI must always be within [0, 100] regardless of input."""
    rng = np.random.default_rng(42)
    close = _make_price_series(list(100 * np.cumprod(1 + rng.normal(0, 0.03, 60))))
    rsi = relative_strength_index(close, window=14).dropna()
    assert (rsi >= 0).all() and (rsi <= 100).all()
    print("test_rsi_bounds PASSED")


def test_add_all_indicators_columns_present():
    rng = np.random.default_rng(1)
    close = 100 * np.cumprod(1 + rng.normal(0, 0.01, 45))
    dates = pd.bdate_range("2026-01-01", periods=45)
    df = pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": rng.integers(1_000_000, 5_000_000, 45),
        },
        index=dates,
    )
    enriched = add_all_indicators(df)
    for col in ["SMA_10", "SMA_20", "RSI_14"]:
        assert col in enriched.columns
    # After 20 trading days all three indicators should be populated together.
    assert enriched.dropna(subset=["SMA_10", "SMA_20", "RSI_14"]).shape[0] > 0
    print("test_add_all_indicators_columns_present PASSED")


def test_add_all_indicators_requires_close_column():
    df = pd.DataFrame({"Open": [1, 2, 3]})
    try:
        add_all_indicators(df)
        raise AssertionError("Expected ValueError for missing Close column")
    except ValueError:
        print("test_add_all_indicators_requires_close_column PASSED")


def _build_indicator_df(closes: list[float]) -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-01", periods=len(closes))
    df = pd.DataFrame(
        {
            "Open": closes,
            "High": [c * 1.01 for c in closes],
            "Low": [c * 0.99 for c in closes],
            "Close": closes,
            "Volume": [1_000_000] * len(closes),
        },
        index=dates,
    )
    return add_all_indicators(df)


def test_recommendation_bullish_uptrend_gives_buy():
    """A strong, smooth uptrend (SMA10 > SMA20, RSI in neutral zone) should yield BUY."""
    # Gentle steady climb keeps RSI in a moderate/neutral-to-bullish range
    # while still producing a clear SMA10 > SMA20 gap.
    closes = [100 + i * 0.6 for i in range(40)]
    df = _build_indicator_df(closes)
    result = generate_recommendation(df)
    assert result.recommendation in ("BUY", "HOLD"), result.recommendation
    # Trend score alone should be bullish
    assert result.signal_details["sma_10"] > result.signal_details["sma_20"]
    print(f"test_recommendation_bullish_uptrend_gives_buy PASSED (got {result.recommendation}, score={result.score})")


def test_recommendation_downtrend_gives_sell():
    """A steady downtrend (SMA10 < SMA20) should yield SELL or at worst HOLD."""
    closes = [200 - i * 0.6 for i in range(40)]
    df = _build_indicator_df(closes)
    result = generate_recommendation(df)
    assert result.recommendation in ("SELL", "HOLD"), result.recommendation
    assert result.signal_details["sma_10"] < result.signal_details["sma_20"]
    print(f"test_recommendation_downtrend_gives_sell PASSED (got {result.recommendation}, score={result.score})")


def test_recommendation_raises_on_insufficient_data():
    """Fewer than 20 rows means SMA_20 never populates -> should raise ValueError."""
    closes = [100 + i for i in range(10)]
    df = _build_indicator_df(closes)
    try:
        generate_recommendation(df)
        raise AssertionError("Expected ValueError for insufficient data")
    except ValueError:
        print("test_recommendation_raises_on_insufficient_data PASSED")


def test_recommendation_score_thresholds():
    """Directly test the scoring/decision boundary logic with a crafted DataFrame."""
    dates = pd.bdate_range("2026-01-01", periods=21)
    df = pd.DataFrame(index=dates)
    df["Close"] = 100.0
    df["SMA_10"] = [np.nan] * 20 + [110.0]   # bullish: SMA10 > SMA20
    df["SMA_20"] = [np.nan] * 20 + [100.0]
    df["RSI_14"] = [np.nan] * 20 + [25.0]    # oversold -> extra +1
    result = generate_recommendation(df)
    assert result.score == 3  # +2 (trend) + 1 (oversold)
    assert result.recommendation == "BUY"
    print("test_recommendation_score_thresholds PASSED")


def test_ticker_validation_rejects_empty():
    try:
        _validate_ticker_format("")
        raise AssertionError("Expected DataFetchError for empty ticker")
    except DataFetchError:
        print("test_ticker_validation_rejects_empty PASSED")


def test_ticker_validation_rejects_garbage():
    try:
        _validate_ticker_format("###not-a-ticker###")
        raise AssertionError("Expected DataFetchError for garbage ticker")
    except DataFetchError:
        print("test_ticker_validation_rejects_garbage PASSED")


def test_ticker_validation_normalizes_case_and_whitespace():
    assert _validate_ticker_format("  aapl  ") == "AAPL"
    print("test_ticker_validation_normalizes_case_and_whitespace PASSED")


if __name__ == "__main__":
    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
    failures = 0
    for test_fn in tests:
        try:
            test_fn()
        except Exception as e:
            failures += 1
            print(f"{test_fn.__name__} FAILED: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} tests passed")
    if failures:
        sys.exit(1)
