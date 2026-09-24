"""WhatsApp .txt export parser - moved here from detective.py so it can sit
in the Phase 1 parser registry (parsers/__init__.py) alongside Instagram and
Telegram instead of being the one hardcoded parse path.

Phase 5 adds `sent_at` to every parsed message. The one real difficulty is
that a WhatsApp export's date order depends on the exporting phone's locale:
"12/01/23" is 1 Dec in a month-first locale and 12 Jan in a day-first one,
and nothing in the line says which. `_infer_date_order` settles it from the
file itself where the data can (a day above 12 rules an order out; a
chronological export rules out an order that would run backwards). Where it
genuinely can't - a short export whose every date is <= 12 - it reads the
file month-first (the layout this regex accepts, `H:MM AM/PM` with a
2-digit year, is WhatsApp's en-US export style) and says so through `notes`
rather than silently picking one.
"""
import re
from datetime import datetime
from typing import List, Optional, Tuple

# The line-matching set is unchanged from Phase 1 (the capture groups are
# new): backfill_timestamps() in ingestion.py depends on a re-parse of an old
# file producing exactly the same messages in the same order.
MSG_RE = re.compile(
    r"^(\d{1,2})/(\d{1,2})/(\d{2,4}), (\d{1,2}):(\d{2}) ([AP]M) - ([^:]+): (.*)$"
)

AMBIGUOUS_DATE_NOTE = (
    "Dates in this export are ambiguous between DD/MM and MM/DD (no day above 12 "
    "and nothing else in the file settles it) - read as MM/DD."
)


def _to_datetime(n1: int, n2: int, year_str: str, hour: int, minute: int,
                 meridiem: str, month_first: bool) -> Optional[datetime]:
    """One export line's date fields -> a datetime under the given date order,
    or None if they aren't a valid date under it (e.g. month 13)."""
    if len(year_str) == 2:
        year = 2000 + int(year_str)
    elif len(year_str) == 4:
        year = int(year_str)
    else:
        return None
    month, day = (n1, n2) if month_first else (n2, n1)
    if not 1 <= hour <= 12:
        return None
    hour24 = hour % 12 + (12 if meridiem == "PM" else 0)
    try:
        return datetime(year, month, day, hour24, minute)
    except ValueError:
        return None


def _infer_date_order(rows: List[tuple]) -> Tuple[Optional[List[datetime]], bool]:
    """Implements the Phase 5 date-order inference described in the module
    docstring. `rows` is each parsed line's (n1, n2, year, hour, minute,
    meridiem) in file order. Returns (datetimes, assumed): datetimes is None
    when no order yields valid dates for every line; assumed is True only
    when both orders were fully valid and equally chronological.
    """
    candidates = {}
    for month_first in (True, False):
        stamps = []
        for n1, n2, year, hour, minute, meridiem in rows:
            dt = _to_datetime(n1, n2, year, hour, minute, meridiem, month_first)
            if dt is None:
                stamps = None
                break
            stamps.append(dt)
        if stamps is not None:
            candidates[month_first] = stamps

    if not candidates:
        return None, False
    if len(candidates) == 1:
        return next(iter(candidates.values())), False

    chronological = [
        month_first for month_first, stamps in candidates.items()
        if all(a <= b for a, b in zip(stamps, stamps[1:]))
    ]
    if len(chronological) == 1:
        return candidates[chronological[0]], False
    return candidates[True], True


def parse_whatsapp(path, notes: Optional[list] = None):
    """Returns [{"sender", "text", "sent_at"}] in file order; sent_at is a
    naive datetime (the export's own local time) or None if no consistent
    date order could be read. Caveats about how the dates were read are
    appended to `notes` if given (see parsers/__init__.py).
    """
    parsed = []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = MSG_RE.match(line.strip())
            if m:
                n1, n2, year, hour, minute, meridiem, sender, text = m.groups()
                rows.append((int(n1), int(n2), year, int(hour), int(minute), meridiem))
                parsed.append({"sender": sender, "text": text})

    stamps, assumed = _infer_date_order(rows)
    for i, message in enumerate(parsed):
        message["sent_at"] = stamps[i] if stamps else None
    if assumed and notes is not None:
        notes.append(AMBIGUOUS_DATE_NOTE)
    return parsed
