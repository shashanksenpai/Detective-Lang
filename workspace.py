"""Case-workspace queries - Phase 5 of the CLAUDE.md roadmap: free-text
search across a case's pooled messages, a message-in-context view, and (added
alongside their endpoints) the timeline and evidence pins. Pure query logic
over the Message/Identifier/Source tables; server.py holds the thin routes,
same split as graph_analysis.py.

Everything here is case-scoped: a message is only ever reachable through the
case its Source belongs to, matching the "cases are isolated" principle.
"""
import re
from datetime import date, datetime, time, timedelta
from typing import Dict, List, Optional

from sqlalchemy import func
from sqlmodel import Session, select

from detective import emotion_features_batch
from models import Identifier, Message, Pin, Source

# `"exact phrase"` or a bare word. Every term must appear (AND), any case.
_TERM_RE = re.compile(r'"([^"]+)"|(\S+)')


class NotFound(Exception):
    """A message/pin id that doesn't exist in the requested case."""


def compile_terms(query: str) -> List[re.Pattern]:
    """Search string -> one case-insensitive literal pattern per term.
    re.IGNORECASE (not str.lower/casefold) so a match's span indexes the
    original text one-to-one - casefolding can change a string's length.
    """
    terms = [quoted or bare for quoted, bare in _TERM_RE.findall(query or "")]
    return [re.compile(re.escape(t), re.IGNORECASE) for t in terms if t.strip()]


def highlight_parts(text: str, patterns: List[re.Pattern]) -> List[list]:
    """Splits `text` into [[segment, is_match], ...] so the frontend can mark
    matches without doing any index math of its own: Python indexes by code
    point but JS by UTF-16 unit, so raw offsets would drift after an emoji.
    """
    spans = sorted((m.start(), m.end()) for p in patterns for m in p.finditer(text))
    merged: List[list] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    parts: List[list] = []
    pos = 0
    for start, end in merged:
        if start > pos:
            parts.append([text[pos:start], False])
        parts.append([text[start:end], True])
        pos = end
    if pos < len(text):
        parts.append([text[pos:], False])
    return parts


def message_view(msg: Message, ident: Identifier, src: Source, pin_id: Optional[int] = None) -> dict:
    """The one message shape every workspace endpoint returns. `pin_id` is the
    evidence pin on this message, or None if it isn't pinned."""
    return {
        "id": msg.id,
        "pin_id": pin_id,
        "text": msg.text,
        "sender": ident.raw_sender_name,
        "source_id": src.id,
        "source_label": src.label,
        "platform": src.platform,
        "context": src.context,
        "sent_at": msg.sent_at.isoformat() if msg.sent_at else None,
        "seq": msg.seq,
    }


def _case_messages_stmt(case_id: int):
    return (
        select(Message, Identifier, Source)
        .join(Identifier, Message.identifier_id == Identifier.id)
        .join(Source, Message.source_id == Source.id)
        .where(Source.case_id == case_id)
    )


def _chronological(stmt):
    """Oldest first; undated messages last; ties (WhatsApp is minute-resolution,
    so many share a stamp) broken by source then original position."""
    return stmt.order_by(Message.sent_at.is_(None), Message.sent_at, Source.id, Message.seq)


def source_stats(session: Session, case_id: int) -> Dict[int, dict]:
    """Per-source message count and date span, keyed by source id. A source
    with no messages yet (pending/failed) simply has no entry."""
    rows = session.exec(
        select(Message.source_id, func.count(Message.id), func.min(Message.sent_at), func.max(Message.sent_at))
        .join(Source, Message.source_id == Source.id)
        .where(Source.case_id == case_id)
        .group_by(Message.source_id)
    ).all()
    return {
        source_id: {
            "message_count": count,
            "first_at": first.isoformat() if first else None,
            "last_at": last.isoformat() if last else None,
        }
        for source_id, count, first, last in rows
    }


def senders(session: Session, case_id: int) -> List[dict]:
    """Every raw sender name in the case with its message count - a light,
    SQL-only alternative to /people (which builds full ML profiles)."""
    counts: Dict[str, int] = {}
    for _msg, ident, _src in session.exec(_case_messages_stmt(case_id)).all():
        counts[ident.raw_sender_name] = counts.get(ident.raw_sender_name, 0) + 1
    return [{"name": n, "message_count": c} for n, c in sorted(counts.items())]


def list_messages(
    session: Session,
    case_id: int,
    q: str = "",
    sender: Optional[str] = None,
    source_id: Optional[int] = None,
    on_date: Optional[date] = None,
    limit: int = 50,
    offset: int = 0,
) -> dict:
    """Implements Phase 5 free-text search across pooled messages. With no `q`
    it's a plain filtered browse (the timeline's day view uses that). Matching
    is done in Python rather than SQL LIKE: LIKE is only case-insensitive for
    ASCII and needs `%`/`_` escaping, while re.IGNORECASE handles Unicode and
    emoji correctly. Fine at chat-export scale; an indexed search would come
    with the Postgres move.
    """
    stmt = _case_messages_stmt(case_id)
    if sender:
        stmt = stmt.where(Identifier.raw_sender_name == sender)
    if source_id is not None:
        stmt = stmt.where(Source.id == source_id)
    if on_date is not None:
        day_start = datetime.combine(on_date, time.min)
        stmt = stmt.where(Message.sent_at >= day_start, Message.sent_at < day_start + timedelta(days=1))
    rows = session.exec(_chronological(stmt)).all()

    patterns = compile_terms(q)
    if patterns:
        rows = [r for r in rows if all(p.search(r[0].text) for p in patterns)]

    pins = pin_ids(session, case_id)
    results = []
    for msg, ident, src in rows[offset:offset + limit]:
        view = message_view(msg, ident, src, pins.get(msg.id))
        if patterns:
            view["parts"] = highlight_parts(msg.text, patterns)
        results.append(view)
    return {"total": len(rows), "offset": offset, "limit": limit, "results": results}


def message_context(session: Session, case_id: int, message_id: int, before: int = 5, after: int = 5) -> dict:
    """A message and its neighbours in the same source, by original position
    (`seq`) - the conversation around a hit. Raises NotFound if the message
    isn't in this case, so one case can never read another's messages by id.
    """
    row = session.exec(_case_messages_stmt(case_id).where(Message.id == message_id)).first()
    if row is None:
        raise NotFound(f"No message {message_id} in this case")
    target, _ident, src = row

    neighbours = session.exec(
        _case_messages_stmt(case_id)
        .where(
            Source.id == src.id,
            Message.seq >= target.seq - before,
            Message.seq <= target.seq + after,
        )
        .order_by(Message.seq)
    ).all()
    pins = pin_ids(session, case_id)
    messages = []
    for msg, ident, s in neighbours:
        view = message_view(msg, ident, s, pins.get(msg.id))
        view["is_target"] = msg.id == target.id
        messages.append(view)
    return {"source_id": src.id, "source_label": src.label, "messages": messages}


# ---- timeline --------------------------------------------------------------

def timeline(session: Session, case_id: int) -> dict:
    """Implements Phase 5's timeline view data: for each calendar day that has
    dated messages, how many each person sent and their average sentiment that
    day (the same Hinglish-aware sentiment score the engine's emotion
    signal uses - it reads wording, not sarcasm or subtext). Only days with
    activity are returned; the client fills the gaps between them, since a
    silent day is itself worth seeing. Messages with no timestamp can't be
    placed on a day and are only counted in `undated_count`.
    """
    per_day: Dict[date, Dict[str, list]] = {}
    people = set()
    undated = 0
    dated = []
    for msg, ident, _src in session.exec(_case_messages_stmt(case_id)).all():
        if msg.sent_at is None:
            undated += 1
            continue
        dated.append((msg.sent_at.date(), ident.raw_sender_name, msg.text))
    # one batched pass: the trained scorer is far faster than message-at-a-time
    compounds = [e["compound"] for e in emotion_features_batch([text for _d, _p, text in dated])]
    for (day, person, _text), compound in zip(dated, compounds):
        people.add(person)
        per_day.setdefault(day, {}).setdefault(person, []).append(compound)

    days = []
    for day in sorted(per_day):
        by_person = per_day[day]
        days.append({
            "date": day.isoformat(),
            "total": sum(len(v) for v in by_person.values()),
            "people": {
                name: {"count": len(scores), "mood": round(sum(scores) / len(scores), 3)}
                for name, scores in sorted(by_person.items())
            },
        })
    return {"days": days, "people": sorted(people), "undated_count": undated}


# ---- evidence pins ---------------------------------------------------------

def pin_ids(session: Session, case_id: int) -> Dict[int, int]:
    """message id -> pin id for every pinned message in the case."""
    rows = session.exec(
        select(Pin.message_id, Pin.id)
        .join(Message, Pin.message_id == Message.id)
        .join(Source, Message.source_id == Source.id)
        .where(Source.case_id == case_id)
    ).all()
    return {message_id: pin_id for message_id, pin_id in rows}


def _pin_entry(pin: Pin, msg: Message, ident: Identifier, src: Source) -> dict:
    return {**message_view(msg, ident, src, pin.id), "note": pin.note, "pinned_at": pin.created_at.isoformat()}


def _pin_row(session: Session, case_id: int, pin_id: int):
    """(Pin, Message, Identifier, Source) for a pin, but only if its message
    belongs to `case_id` - a pin id from another case is treated as absent."""
    row = session.exec(
        select(Pin, Message, Identifier, Source)
        .join(Message, Pin.message_id == Message.id)
        .join(Identifier, Message.identifier_id == Identifier.id)
        .join(Source, Message.source_id == Source.id)
        .where(Source.case_id == case_id, Pin.id == pin_id)
    ).first()
    if row is None:
        raise NotFound(f"No pin {pin_id} in this case")
    return row


def list_pins(session: Session, case_id: int) -> List[dict]:
    """Implements Phase 5 evidence pinning: every pinned message in the case
    with its note, in message-chronological order (the order the evidence
    happened, not the order it was pinned)."""
    stmt = (
        select(Pin, Message, Identifier, Source)
        .join(Message, Pin.message_id == Message.id)
        .join(Identifier, Message.identifier_id == Identifier.id)
        .join(Source, Message.source_id == Source.id)
        .where(Source.case_id == case_id)
    )
    return [_pin_entry(*row) for row in session.exec(_chronological(stmt)).all()]


def create_pin(session: Session, case_id: int, message_id: int, note: str = "") -> dict:
    """Pins a message. Idempotent: pinning an already-pinned message returns
    the existing pin unchanged (notes are edited via update_pin), so a double
    click can't create a duplicate or clobber a note."""
    row = session.exec(_case_messages_stmt(case_id).where(Message.id == message_id)).first()
    if row is None:
        raise NotFound(f"No message {message_id} in this case")
    msg, ident, src = row

    pin = session.exec(select(Pin).where(Pin.message_id == message_id)).first()
    if pin is None:
        pin = Pin(message_id=message_id, note=note)
        session.add(pin)
        session.commit()
        session.refresh(pin)
    return _pin_entry(pin, msg, ident, src)


def update_pin(session: Session, case_id: int, pin_id: int, note: str) -> dict:
    pin, msg, ident, src = _pin_row(session, case_id, pin_id)
    pin.note = note
    session.add(pin)
    session.commit()
    session.refresh(pin)
    return _pin_entry(pin, msg, ident, src)


def delete_pin(session: Session, case_id: int, pin_id: int) -> None:
    pin = _pin_row(session, case_id, pin_id)[0]
    session.delete(pin)
    session.commit()
