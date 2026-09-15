"""
feedback.py
-----------
Records user feedback on recommendations (helpful / not helpful, plus an
optional comment) to local, append-only JSONL storage.

This closes a real gap in a recommendation system: without any feedback
loop, there's no way to know whether real users actually found a given
call useful, and no data trail to eventually evaluate or retrain the
scoring logic against. This module is deliberately simple -- a flat
JSONL file, no database dependency -- so it works the same way in a
notebook, a script, or (with the storage path swapped out) a small
service, without adding infrastructure requirements to the project.

Each feedback record captures the recommendation it's reacting to
(ticker, recommendation, score, date, signal snapshot) alongside the
user's rating, so feedback can later be joined back against the exact
signal state that produced the recommendation -- useful for asking
"does the strategy get worse feedback specifically when RSI is near the
oversold/overbought boundary?" or similar analysis.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

VALID_RATINGS = ("helpful", "not_helpful")

# Default location for the feedback log. Kept as a plain JSONL file next
# to the package rather than a database -- see module docstring.
DEFAULT_FEEDBACK_PATH = Path.home() / ".stock_agent" / "feedback.jsonl"


class FeedbackError(Exception):
    """Raised when feedback cannot be recorded or read."""


@dataclass
class FeedbackRecord:
    """A single piece of user feedback on a recommendation."""

    feedback_id: str
    ticker: str
    recommendation: str
    score: int | None
    rating: str                      # "helpful" | "not_helpful"
    comment: str | None
    signal_details: dict
    recorded_at: str                 # ISO 8601 UTC timestamp

    @staticmethod
    def new(
        ticker: str,
        recommendation: str,
        rating: str,
        signal_details: dict,
        score: int | None = None,
        comment: str | None = None,
    ) -> "FeedbackRecord":
        """Construct a new record with a generated id and timestamp."""
        if rating not in VALID_RATINGS:
            raise FeedbackError(f"rating must be one of {VALID_RATINGS}, got '{rating}'")
        return FeedbackRecord(
            feedback_id=str(uuid.uuid4()),
            ticker=ticker,
            recommendation=recommendation,
            score=score,
            rating=rating,
            comment=comment,
            signal_details=signal_details or {},
            recorded_at=datetime.now(timezone.utc).isoformat(),
        )


def record_feedback(
    ticker: str,
    recommendation: str,
    rating: str,
    signal_details: dict,
    score: int | None = None,
    comment: str | None = None,
    path: Path | str | None = None,
) -> FeedbackRecord:
    """Append a feedback record to the local JSONL feedback log.

    Args:
        ticker: Normalized ticker the recommendation was for.
        recommendation: "BUY" | "HOLD" | "SELL" -- the call being rated.
        rating: "helpful" or "not_helpful".
        signal_details: The signal_details dict from the recommendation
            (close, sma_10, sma_20, rsi_14, date) -- stored alongside the
            rating so feedback can be analyzed against the exact inputs.
        score: The combined signal score, if available.
        comment: Optional free-text comment from the user.
        path: Override the feedback log location (mainly for testing).
            Defaults to `DEFAULT_FEEDBACK_PATH`.

    Returns:
        The FeedbackRecord that was written.

    Raises:
        FeedbackError: if `rating` is invalid or the record can't be written
            (e.g. permissions issue).
    """
    record = FeedbackRecord.new(
        ticker=ticker,
        recommendation=recommendation,
        rating=rating,
        signal_details=signal_details,
        score=score,
        comment=comment,
    )

    target = Path(path) if path is not None else DEFAULT_FEEDBACK_PATH
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record)) + "\n")
    except OSError as exc:
        raise FeedbackError(f"Could not write feedback to {target}: {exc}") from exc

    return record


def load_feedback(path: Path | str | None = None) -> list[FeedbackRecord]:
    """Load all feedback records from the JSONL log.

    Args:
        path: Override the feedback log location. Defaults to `DEFAULT_FEEDBACK_PATH`.

    Returns:
        A list of FeedbackRecord, in the order they were recorded. Returns
        an empty list if the log file doesn't exist yet (not an error --
        "no feedback recorded yet" is a normal, expected state).

    Raises:
        FeedbackError: if the file exists but contains malformed JSON.
    """
    target = Path(path) if path is not None else DEFAULT_FEEDBACK_PATH
    if not target.exists():
        return []

    records = []
    try:
        with open(target, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                records.append(FeedbackRecord(**data))
    except (json.JSONDecodeError, TypeError) as exc:
        raise FeedbackError(f"Malformed feedback log at {target} (line {line_num}): {exc}") from exc

    return records


def summarize_feedback(path: Path | str | None = None) -> dict:
    """Compute simple aggregate stats over all recorded feedback.

    Returns:
        A dict with:
            total: total number of feedback records
            helpful: count rated "helpful"
            not_helpful: count rated "not_helpful"
            helpful_rate_pct: helpful / total * 100 (0.0 if no records)
            by_ticker: {ticker: {"helpful": n, "not_helpful": n}}
            by_recommendation: {"BUY"/"HOLD"/"SELL": {"helpful": n, "not_helpful": n}}
    """
    records = load_feedback(path)

    summary = {
        "total": len(records),
        "helpful": sum(1 for r in records if r.rating == "helpful"),
        "not_helpful": sum(1 for r in records if r.rating == "not_helpful"),
        "by_ticker": {},
        "by_recommendation": {},
    }
    summary["helpful_rate_pct"] = (
        summary["helpful"] / summary["total"] * 100 if summary["total"] else 0.0
    )

    for r in records:
        t = summary["by_ticker"].setdefault(r.ticker, {"helpful": 0, "not_helpful": 0})
        t[r.rating] += 1

        rec = summary["by_recommendation"].setdefault(r.recommendation, {"helpful": 0, "not_helpful": 0})
        rec[r.rating] += 1

    return summary
