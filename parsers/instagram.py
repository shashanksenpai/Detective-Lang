"""Instagram JSON export parser - Phase 3 of the CLAUDE.md roadmap.

Format: Meta's "Download Your Information" data export, JSON option,
Messages category (messages/inbox/<thread>/message_1.json). Two real-world
quirks handled here:
- Messages are listed newest-first in the export; reversed to chronological
  order here since the rest of the pipeline (Message.seq, the discourse/
  turn-taking signal) assumes oldest-first, same as WhatsApp already gives.
- Meta's export has a well-known mojibake bug: non-ASCII text is UTF-8
  bytes that got decoded as Latin-1 into the JSON string escapes. Standard
  fix: re-encode as Latin-1, decode as UTF-8.

Phase 5: `timestamp_ms` is a UTC epoch, so `sent_at` here is naive UTC -
unlike WhatsApp/Telegram, whose times are the exporting device's local time.
That mismatch can't be corrected without knowing the user's timezone, so it
is reported through `notes` instead of being papered over.
"""
import json
from datetime import datetime, timezone
from typing import Optional

UTC_NOTE = (
    "Instagram exports timestamps in UTC - times for this source are UTC, "
    "not local time."
)


def _fix_mojibake(text):
    try:
        return text.encode("latin1").decode("utf8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return text


def _from_epoch_ms(value) -> Optional[datetime]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        return datetime.fromtimestamp(value / 1000, tz=timezone.utc).replace(tzinfo=None)
    except (OverflowError, OSError, ValueError):
        return None


def parse_instagram(path, notes: Optional[list] = None):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if "messages" not in data:
        raise ValueError(
            "This doesn't look like an Instagram message export (missing 'messages' key) - "
            "use the JSON export from Meta's 'Download Your Information' tool."
        )

    messages = []
    for m in reversed(data["messages"]):
        if m.get("is_unsent"):
            continue
        sender = m.get("sender_name")
        content = m.get("content")
        if not sender or not content:
            continue
        messages.append({
            "sender": _fix_mojibake(sender),
            "text": _fix_mojibake(content),
            "sent_at": _from_epoch_ms(m.get("timestamp_ms")),
        })
    if notes is not None and any(m["sent_at"] for m in messages):
        notes.append(UTC_NOTE)
    return messages
