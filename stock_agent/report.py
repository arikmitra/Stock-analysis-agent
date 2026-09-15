"""
report.py
---------
Formats the final agent state into a clean, human-readable text report.
"""

from __future__ import annotations

import pandas as pd

_BAR = "=" * 62


def format_report(
    ticker: str,
    company_name: str | None,
    currency: str | None,
    indicators_df: pd.DataFrame,
    recommendation: str,
    reasons: list[str],
    signal_details: dict,
    data_source: str = "yfinance",
) -> str:
    """Build the final formatted analysis report as a plain-text string.

    Args:
        ticker: Normalized ticker symbol.
        company_name: Company long name, if known.
        currency: Trading currency code, e.g. "USD".
        indicators_df: Full indicator DataFrame (used to show recent history).
        recommendation: "BUY" | "HOLD" | "SELL".
        reasons: List of human-readable justification strings.
        signal_details: Dict of the raw numeric values behind the decision.
        data_source: "yfinance" or "synthetic" -- shown as a disclaimer when
            synthetic, so nobody mistakes demo data for real market data.

    Returns:
        A formatted multi-line string ready to print or save.
    """
    name_line = f"{company_name} ({ticker})" if company_name else ticker
    currency = currency or "USD"

    latest = indicators_df.dropna(subset=["SMA_10", "SMA_20", "RSI_14"]).iloc[-1]
    latest_date = indicators_df.dropna(subset=["SMA_10", "SMA_20", "RSI_14"]).index[-1]

    recommendation_emoji = {"BUY": "🟢", "HOLD": "🟡", "SELL": "🔴"}.get(recommendation, "")

    lines = []
    lines.append(_BAR)
    lines.append(f"  STOCK ANALYSIS REPORT — {name_line}")
    lines.append(_BAR)
    lines.append(f"  As of: {latest_date.strftime('%Y-%m-%d')}")
    lines.append(f"  Data points analyzed: {len(indicators_df)} trading days")
    if data_source == "synthetic":
        lines.append("  ⚠️  DEMO MODE: using synthetic data, NOT real market data")
    lines.append("")

    lines.append("  PRICE & INDICATORS")
    lines.append("  " + "-" * 40)
    lines.append(f"  {'Close Price:':<22}{currency} {latest['Close']:.2f}")
    lines.append(f"  {'10-Day SMA:':<22}{currency} {signal_details['sma_10']:.2f}")
    lines.append(f"  {'20-Day SMA:':<22}{currency} {signal_details['sma_20']:.2f}")
    lines.append(f"  {'14-Day RSI:':<22}{signal_details['rsi_14']:.2f}")
    lines.append("")

    lines.append("  RECOMMENDATION")
    lines.append("  " + "-" * 40)
    lines.append(f"  {recommendation_emoji}  {recommendation}")
    lines.append("")

    lines.append("  RATIONALE")
    lines.append("  " + "-" * 40)
    for reason in reasons:
        lines.append(f"  • {reason}")
    lines.append("")

    lines.append("  RECENT PRICE HISTORY (last 10 trading days)")
    lines.append("  " + "-" * 40)
    recent = indicators_df.tail(10)
    header = f"  {'Date':<12}{'Close':>10}{'SMA10':>10}{'SMA20':>10}{'RSI14':>9}"
    lines.append(header)
    for idx, row in recent.iterrows():
        date_str = idx.strftime("%Y-%m-%d")
        sma10 = f"{row['SMA_10']:.2f}" if pd.notna(row["SMA_10"]) else "  n/a"
        sma20 = f"{row['SMA_20']:.2f}" if pd.notna(row["SMA_20"]) else "  n/a"
        rsi14 = f"{row['RSI_14']:.2f}" if pd.notna(row["RSI_14"]) else " n/a"
        lines.append(
            f"  {date_str:<12}{row['Close']:>10.2f}{sma10:>10}{sma20:>10}{rsi14:>9}"
        )
    lines.append("")

    lines.append("  DISCLAIMER")
    lines.append("  " + "-" * 40)
    lines.append(
        "  This report is generated from simple technical indicators for"
    )
    lines.append(
        "  educational purposes only and is NOT financial advice. Always"
    )
    lines.append(
        "  do your own research and consult a licensed financial advisor"
    )
    lines.append("  before making investment decisions.")
    lines.append(_BAR)

    return "\n".join(lines)
