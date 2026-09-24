"""Parser registry - maps a Source.platform value to its parser function.
Every imported source goes through one of these instead of a single
hardcoded parse_whatsapp call. Phase 1 of the CLAUDE.md roadmap.

Parser contract: `parser(path, notes=None) -> [{"sender", "text", "sent_at"}]`
in chronological order, where `sent_at` is a naive datetime or None (Phase 5).
If a `notes` list is passed, the parser appends short human-readable caveats
about how it interpreted the file (an assumed date order, a non-local
timezone); ingestion stores them on the Source so the UI can show them
instead of the assumption being invisible. A bad file raises ValueError.
"""
from .instagram import parse_instagram
from .telegram import parse_telegram
from .whatsapp import parse_whatsapp

PARSERS = {
    "whatsapp": parse_whatsapp,
    "instagram": parse_instagram,
    "telegram": parse_telegram,
}
