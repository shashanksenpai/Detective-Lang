"""WhatsApp .txt export parser - moved here from detective.py so it can sit
in the Phase 1 parser registry (parsers/__init__.py) alongside Instagram and
Telegram instead of being the one hardcoded parse path.

Phase 5 adds `sent_at` to every parsed message. The one real difficulty is
that a WhatsApp export's date order depends on the exporting phone's locale:
"12/01/23" is 1 Dec in a month-first locale and 12 Jan in a day-first one,
and nothing in the line says which. `_infer_date_order` settles it from the
file itself where the data can (a day above 12 rules an order out; a
chronological export rules out an order that would run backwards). Where it
genuinely can't - a short export whose every date is <= 12 - it falls back to
a default and says so through `notes` rather than silently picking one.

BACKLOG F-01/F-02: the export layouts and message shapes real phones produce.
Recognised header layouts (`_ANDROID_RE`, `_IOS_RE`):
  Android   `25/12/2023, 21:14 - Name: text`   24-hour, or `9:14 pm` / `9:14 PM`,
            `/` `.` or `-` separators, 2- or 4-digit year, year-first (`2023-12-25`),
            a narrow no-break space (U+202F) before AM/PM as newer exports write it
  iOS       `[25/12/23, 21:14:05] Name: text`   seconds, optional AM/PM, and the
            invisible left-to-right mark iOS puts on some lines
A header with no `Name: ` after it is a system event (joined, left, encryption
notice ...) and is skipped. A line with no header at all continues the previous
message (a multi-line message); a continuation after a system event is dropped.
Message count and order for a single-line export are exactly what the original
en-US-only matcher (`MSG_RE`, kept below) produced - backfill_timestamps() in
ingestion.py depends on a re-parse of an old file yielding the same messages in
the same order, and test_timestamps.py pins it.
"""
import re
from datetime import datetime
from typing import List, NamedTuple, Optional, Tuple

# The original en-US-only line matcher, unchanged. parse_whatsapp() no longer
# uses it (it accepts the wider set below); it is kept because it defines the
# line set old sources were ingested with - the parity tests and
# test_leak_case.py check the parser against it.
MSG_RE = re.compile(
    r"^(\d{1,2})/(\d{1,2})/(\d{2,4}), (\d{1,2}):(\d{2}) ([AP]M) - ([^:]+): (.*)$"
)

_MARKS = "‎‏﻿"      # LRM / RLM / BOM: invisible, never part of a message
_STRIP = " \t\r\n" + _MARKS

_DATE = r"(?P<n1>\d{1,4})(?P<sep>[/.\-])(?P<n2>\d{1,2})(?P=sep)(?P<n3>\d{1,4})"
_CLOCK = (
    r"(?P<hour>\d{1,2}):(?P<minute>\d{2})(?::(?P<second>\d{2}))?"
    r"(?:[\s  ]*(?P<mer>[AaPp]\.?[Mm]\.?))?"
)
_ANDROID_RE = re.compile(r"^" + _DATE + r",?\s+" + _CLOCK + r"\s*[-–—]\s+(?P<rest>.*)$")
_IOS_RE = re.compile(r"^\[" + _DATE + r",?\s+" + _CLOCK + r"\s*\]\s*(?P<rest>.*)$")
_SENDER_RE = re.compile(r"^([^:]+): (.*)$")

# BACKLOG F-03: lines WhatsApp writes in place of a message's content. Only a
# message that is *entirely* one of these is flagged; a real caption next to a
# placeholder ("look at this <Media omitted>") stays ordinary text.
_MEDIA_PLACEHOLDER_RE = re.compile(
    r"<Media omitted>"                                   # Android
    r"|<attached: [^>]+>"                                # iOS
    r"|(?:image|video|audio|sticker|GIF|document|Contact card|video note|voice message) omitted"
    r"|\S+\.\w{2,5} \(file attached\)",                  # Android, media included
    re.IGNORECASE,
)
_DELETED_RE = re.compile(
    r"(?:\U0001F6AB\s*)?(?:this message was deleted|you deleted this message)\.?",
    re.IGNORECASE,
)


def classify_text(text: str) -> str:
    """What a message's text is: "media" (a photo/video/sticker/... placeholder),
    "deleted" (a deleted-message notice) or "text" (something a person wrote).
    Non-text messages stay in the record - they are real events, and the chat
    reader will want to show them - but are not speech, so profiles, sentiment,
    embeddings and turn-taking skip them (engine_cache.load_case_sources)."""
    body = text.strip(_STRIP)
    if _MEDIA_PLACEHOLDER_RE.fullmatch(body):
        return "media"
    if _DELETED_RE.fullmatch(body):
        return "deleted"
    return "text"


_AMBIGUOUS_DATE_NOTE_HEAD = (
    "Dates in this export are ambiguous between DD/MM and MM/DD (no day above 12 "
    "and nothing else in the file settles it) - read as "
)
AMBIGUOUS_DATE_NOTE = _AMBIGUOUS_DATE_NOTE_HEAD + "MM/DD."
AMBIGUOUS_DATE_NOTE_DAY_FIRST = _AMBIGUOUS_DATE_NOTE_HEAD + "DD/MM."


class _Fields(NamedTuple):
    """One header's date/time fields, as written. For a year-first date
    (`2023-12-25`) n1/n2/n3 are year/month/day; otherwise they are the first
    number, the second number and the year (n3_digits says 2 or 4 digits)."""
    n1: int
    n2: int
    n3: int
    n3_digits: int
    year_first: bool
    sep: str
    hour: int
    minute: int
    second: int
    mer: Optional[str]        # "AM" / "PM" (normalised) or None for a 24-hour clock
    mer_raw: Optional[str]    # as written: "PM", "pm", "p.m." ...


def _match_header(line: str) -> Optional[Tuple[_Fields, str]]:
    """(fields, text after the header) if `line` starts with a WhatsApp
    date/time header, else None."""
    m = _ANDROID_RE.match(line) or _IOS_RE.match(line)
    if m is None:
        return None
    n1, n2, n3 = m["n1"], m["n2"], m["n3"]
    year_first = len(n1) == 4
    if year_first:
        if len(n3) > 2:
            return None
    elif len(n1) > 2 or len(n3) not in (2, 4):
        return None
    mer_raw = m["mer"]
    mer = mer_raw.replace(".", "").upper() if mer_raw else None
    fields = _Fields(
        int(n1), int(n2), int(n3), len(n3), year_first, m["sep"],
        int(m["hour"]), int(m["minute"]), int(m["second"] or 0), mer, mer_raw,
    )
    return fields, m["rest"]


def _to_datetime(f: _Fields, month_first: bool) -> Optional[datetime]:
    """One header's fields -> a datetime under the given date order, or None
    if they aren't a valid date/time under it (e.g. month 13, hour 25)."""
    if f.year_first:
        year, month, day = f.n1, f.n2, f.n3
    else:
        year = 2000 + f.n3 if f.n3_digits == 2 else f.n3
        month, day = (f.n1, f.n2) if month_first else (f.n2, f.n1)
    if f.mer:
        if not 1 <= f.hour <= 12:
            return None
        hour24 = f.hour % 12 + (12 if f.mer == "PM" else 0)
    else:
        if not 0 <= f.hour <= 23:
            return None
        hour24 = f.hour
    try:
        return datetime(year, month, day, hour24, f.minute, f.second)
    except ValueError:
        return None


def _default_month_first(f: _Fields) -> bool:
    """The order to assume when the file can't settle it. Month-first only for
    the classic en-US layout - `/` separators with an upper-case AM/PM - which
    is what this parser first accepted and what old sources were read as. Every
    other layout (24-hour, lower-case am/pm as en-IN/en-GB phones write it,
    dotted dates, year-first) is day-first or unambiguous in practice."""
    return (not f.year_first) and f.sep == "/" and f.mer_raw in ("AM", "PM")


def _infer_date_order(
    rows: List[_Fields], default_month_first: bool
) -> Tuple[Optional[List[datetime]], bool]:
    """Implements the Phase 5 date-order inference described in the module
    docstring. `rows` is each header's fields in file order. Returns
    (datetimes, assumed): datetimes is None when no order yields valid dates
    for every line; assumed is True only when both orders were fully valid,
    read differently, and were equally chronological - i.e. the file itself
    could not decide, so `default_month_first` was used.
    """
    candidates = {}
    for month_first in (True, False):
        stamps = []
        for f in rows:
            dt = _to_datetime(f, month_first)
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
    if candidates[True] == candidates[False]:
        return candidates[True], False       # both orders read every line the same

    chronological = [
        month_first for month_first, stamps in candidates.items()
        if all(a <= b for a, b in zip(stamps, stamps[1:]))
    ]
    if len(chronological) == 1:
        return candidates[chronological[0]], False
    return candidates[default_month_first], True


def parse_whatsapp(path, notes: Optional[list] = None):
    """Returns [{"sender", "text", "sent_at", "kind"}] in file order; sent_at is
    a naive datetime (the export's own local time) or None if no consistent
    date order could be read, and kind is "text" / "media" / "deleted" (see
    classify_text). Caveats about how the dates were read are appended to
    `notes` if given (see parsers/__init__.py).
    """
    parsed: List[dict] = []
    rows: List[_Fields] = []
    current: Optional[dict] = None      # the message a headerless line continues
    with open(path, encoding="utf-8-sig") as f:
        for raw in f:
            header = _match_header(raw.strip(_STRIP))
            if header is None:
                if current is not None:
                    current["text"] += "\n" + raw.rstrip()
                continue
            fields, rest = header
            m = _SENDER_RE.match(rest.lstrip(_MARKS))
            if m is None:               # a system event: no sender
                current = None
                continue
            sender, text = m.groups()
            current = {"sender": sender, "text": text.lstrip(_MARKS)}
            parsed.append(current)
            rows.append(fields)

    for message in parsed:
        message["text"] = message["text"].rstrip()
        message["kind"] = classify_text(message["text"])

    default_month_first = _default_month_first(rows[0]) if rows else True
    stamps, assumed = _infer_date_order(rows, default_month_first)
    for i, message in enumerate(parsed):
        message["sent_at"] = stamps[i] if stamps else None
    if assumed and notes is not None:
        notes.append(AMBIGUOUS_DATE_NOTE if default_month_first else AMBIGUOUS_DATE_NOTE_DAY_FIRST)
    return parsed
