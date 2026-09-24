"""Telegram JSON export parser - Phase 3 of the CLAUDE.md roadmap.

Format: Telegram Desktop's "Export chat history" as JSON. Already
chronological (oldest-first), unlike Instagram's export. Two real-world
quirks handled here:
- Non-message entries (joins/leaves/pins/etc, type != "message") are
  skipped - they're not authored chat text.
- The `text` field isn't always a plain string: formatted text (links,
  bold, mentions) comes back as a list mixing plain strings and
  {"type": ..., "text": ...} entity objects, which needs flattening into
  plain text.

Phase 5: `date` is an ISO-8601 string in the exporting machine's local
time, so `sent_at` is stored as-is (naive), same convention as WhatsApp.
"""
import json
from datetime import datetime, timezone
from typing import Optional


def _flatten_text(text_field):
    if isinstance(text_field, str):
        return text_field
    if isinstance(text_field, list):
        parts = []
        for piece in text_field:
            if isinstance(piece, str):
                parts.append(piece)
            elif isinstance(piece, dict):
                parts.append(piece.get("text", ""))
        return "".join(parts)
    return ""


def _parse_date(value) -> Optional[datetime]:
    if not isinstance(value, str):
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def parse_telegram(path, notes: Optional[list] = None):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if "messages" not in data:
        raise ValueError(
            "This doesn't look like a Telegram chat export (missing 'messages' key) - "
            "use Telegram Desktop's 'Export chat history' as JSON."
        )

    messages = []
    for m in data["messages"]:
        if m.get("type") != "message":
            continue
        sender = m.get("from")
        text = _flatten_text(m.get("text", ""))
        if not sender or not text.strip():
            continue
        messages.append({"sender": sender, "text": text, "sent_at": _parse_date(m.get("date"))})
    return messages
