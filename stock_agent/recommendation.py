"""
recommendation.py
------------------
Translates the latest technical indicator readings into a BUY / HOLD / SELL
recommendation with human-readable justification.

The logic is intentionally simple, transparent, and rule-based (no ML model)
so that every recommendation can be fully explained. It combines two
classic signals:

  1. Moving Average Crossover (trend-following)
     - SMA10 > SMA20  -> short-term trend is up  -> bullish tilt
     - SMA10 < SMA20  -> short-term trend is down -> bearish tilt

  2. RSI Overbought / Oversold (mean-reversion)
     - RSI < 30 -> oversold  -> bullish tilt (potential bounce)
     - RSI > 70 -> overbought -> bearish tilt (potential pullback)

A scoring system combines both signals into one of three recommendations.
This mirrors how many retail technical-analysis dashboards blend a trend
indicator with a momentum oscillator, and is a common introductory strategy
taught in technical analysis courses.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

# Thresholds -- centralized here so they're easy to tune / unit test.
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
BUY_SCORE_THRESHOLD = 2
SELL_SCORE_THRESHOLD = -2


@dataclass
class RecommendationResult:
    """Structured output of the recommendation engine."""

    recommendation: str                    # "BUY" | "HOLD" | "SELL"
    score: int                             # Combined signal score
    reasons: list[str] = field(default_factory=list)
    signal_details: dict = field(default_factory=dict)


def generate_recommendation(df: pd.DataFrame) -> RecommendationResult:
    """Generate a BUY/HOLD/SELL recommendation from the most recent indicator row.

    Scoring model (each signal contributes -2..+2 to a combined score):
        SMA crossover:
            SMA10 > SMA20            -> +2 (bullish trend)
            SMA10 < SMA20            -> -2 (bearish trend)
            SMA10 == SMA20           ->  0 (neutral)
        RSI:
            RSI < 30 (oversold)      -> +1 (potential upward reversal)
            RSI > 70 (overbought)    -> -1 (potential downward reversal)
            30 <= RSI <= 70          ->  0 (neutral momentum)

    Final decision:
        score >=  2  -> BUY
        score <= -2  -> SELL
        otherwise    -> HOLD

    Args:
        df: Indicator DataFrame (output of `indicators.add_all_indicators`),
            sorted ascending by date, with at least one fully-populated
            (non-NaN) row for SMA_10, SMA_20, and RSI_14.

    Returns:
        A RecommendationResult with the final call, score, and reasons.

    Raises:
        ValueError: if there isn't at least one row with all indicators
            populated (i.e. not enough history was fetched).
    """
    required_cols = {"SMA_10", "SMA_20", "RSI_14", "Close"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required indicator columns: {missing}")

    valid_rows = df.dropna(subset=["SMA_10", "SMA_20", "RSI_14"])
    if valid_rows.empty:
        raise ValueError(
            "Not enough historical data to compute a recommendation. "
            "Need at least 20 trading days for SMA_20 and 14 for RSI_14."
        )

    latest = valid_rows.iloc[-1]
    latest_date = valid_rows.index[-1]

    close = float(latest["Close"])
    sma10 = float(latest["SMA_10"])
    sma20 = float(latest["SMA_20"])
    rsi = float(latest["RSI_14"])

    score = 0
    reasons: list[str] = []

    # --- Signal 1: SMA crossover (trend) ---
    if sma10 > sma20:
        score += 2
        pct_gap = (sma10 - sma20) / sma20 * 100
        reasons.append(
            f"Bullish trend: 10-day SMA (${sma10:.2f}) is above the 20-day SMA "
            f"(${sma20:.2f}), a {pct_gap:.2f}% gap, indicating short-term "
            f"momentum is stronger than the medium-term trend."
        )
    elif sma10 < sma20:
        score -= 2
        pct_gap = (sma20 - sma10) / sma20 * 100
        reasons.append(
            f"Bearish trend: 10-day SMA (${sma10:.2f}) is below the 20-day SMA "
            f"(${sma20:.2f}), a {pct_gap:.2f}% gap, indicating short-term "
            f"momentum is weaker than the medium-term trend."
        )
    else:
        reasons.append(
            f"Neutral trend: 10-day and 20-day SMAs are essentially equal "
            f"(${sma10:.2f})."
        )

    # --- Signal 2: RSI overbought / oversold (momentum) ---
    if rsi < RSI_OVERSOLD:
        score += 1
        reasons.append(
            f"Oversold condition: RSI(14) = {rsi:.1f}, below the {RSI_OVERSOLD} "
            f"threshold, suggesting the stock may be due for a rebound."
        )
    elif rsi > RSI_OVERBOUGHT:
        score -= 1
        reasons.append(
            f"Overbought condition: RSI(14) = {rsi:.1f}, above the {RSI_OVERBOUGHT} "
            f"threshold, suggesting the stock may be due for a pullback."
        )
    else:
        reasons.append(
            f"Neutral momentum: RSI(14) = {rsi:.1f}, within the normal "
            f"{RSI_OVERSOLD}-{RSI_OVERBOUGHT} range."
        )

    # --- Final decision ---
    if score >= BUY_SCORE_THRESHOLD:
        decision = "BUY"
    elif score <= SELL_SCORE_THRESHOLD:
        decision = "SELL"
    else:
        decision = "HOLD"

    reasons.append(f"Combined signal score: {score:+d} -> {decision}")

    return RecommendationResult(
        recommendation=decision,
        score=score,
        reasons=reasons,
        signal_details={
            "date": str(latest_date.date()) if hasattr(latest_date, "date") else str(latest_date),
            "close": close,
            "sma_10": sma10,
            "sma_20": sma20,
            "rsi_14": rsi,
        },
    )
