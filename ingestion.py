"""Background ingestion: parse an uploaded Source's file and populate
Identifier/Person/PersonCase/Message rows. Phase 1 of the CLAUDE.md roadmap.
Phase 5: also persists each message's timestamp and the parser's notes about
how dates were read, and backfills both for sources ingested before that.
"""
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

from sqlmodel import Session, select

import engine_cache
import identity_resolution
from db import engine as db_engine
from models import Identifier, Message, Person, PersonCase, Source
from parsers import PARSERS

log = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=2)


def submit_ingestion(source_id: int):
    _executor.submit(ingest_source, source_id)


def _find_or_create_person(session: Session, case_id: int, raw_name: str) -> int:
    """One Person per unique raw sender name, scoped to this case only -
    never a global/cross-case match. Two different cases' same-named
    senders are never silently assumed to be the same person; that's
    Phase 2's reviewed merge flow.
    """
    existing = session.exec(
        select(Identifier)
        .join(Source, Identifier.source_id == Source.id)
        .where(Source.case_id == case_id, Identifier.raw_sender_name == raw_name)
    ).first()
    if existing:
        return existing.person_id

    person = Person(display_name=raw_name)
    session.add(person)
    session.flush()
    session.add(PersonCase(person_id=person.id, case_id=case_id))
    return person.id


def _date_note(messages: List[dict], parser_notes: List[str]) -> Optional[str]:
    """Combines the parser's own caveats with what ingestion can see for
    itself: how many messages ended up with no timestamp at all. Returns None
    when every message is dated and the parser had nothing to flag.
    """
    parts = list(parser_notes)
    undated = sum(1 for m in messages if m.get("sent_at") is None)
    if undated == len(messages):
        parts.append("No readable timestamps in this export - messages are undated.")
    elif undated:
        parts.append(f"{undated} of {len(messages)} messages have no readable timestamp.")
    return " ".join(parts) or None


def ingest_source(source_id: int):
    with Session(db_engine) as session:
        source = session.get(Source, source_id)
        if source is None:
            return
        case_id = source.case_id
        source.status = "processing"
        session.add(source)
        session.commit()

        try:
            parser = PARSERS[source.platform]
            parser_notes: List[str] = []
            messages = parser(source.file_path, parser_notes)
            if not messages:
                raise ValueError("No messages parsed from this file - check the export format.")

            identifier_id_by_name = {}
            for raw_name in {m["sender"] for m in messages}:
                person_id = _find_or_create_person(session, case_id, raw_name)
                identifier = Identifier(
                    source_id=source.id, raw_sender_name=raw_name, person_id=person_id
                )
                session.add(identifier)
                session.flush()
                identifier_id_by_name[raw_name] = identifier.id

            for seq, m in enumerate(messages):
                session.add(Message(
                    source_id=source.id,
                    identifier_id=identifier_id_by_name[m["sender"]],
                    text=m["text"],
                    sent_at=m.get("sent_at"),
                    seq=seq,
                ))

            source.status = "ready"
            source.error_message = None
            source.date_note = _date_note(messages, parser_notes)
        except Exception as exc:
            source.status = "failed"
            source.error_message = str(exc)

        session.add(source)
        session.commit()
        final_status = source.status

    engine_cache.invalidate(case_id)

    if final_status == "ready":
        try:
            identity_resolution.scan_for_matches(case_id)
        except Exception:
            # Best-effort: a matching failure shouldn't undo an otherwise
            # successful ingestion.
            pass


def backfill_timestamps() -> int:
    """Implements Phase 5's timestamp backfill: sources ingested before
    timestamps were captured have Message.sent_at NULL. Re-parse each such
    source's stored file and fill the dates in by `seq`, leaving every
    Message/Identifier id untouched (pins reference message ids).

    Only runs for a ready source whose `date_note` is still NULL and which has
    an undated message; a source that couldn't be dated gets a note from
    _date_note, so it isn't re-parsed on every startup. A source whose file is
    gone or no longer reproduces the stored messages exactly is skipped with a
    warning rather than half-updated. Returns how many sources were filled.
    """
    filled = 0
    with Session(db_engine) as session:
        sources = session.exec(
            select(Source).where(Source.status == "ready", Source.date_note.is_(None))
        ).all()
        for source in sources:
            rows = session.exec(
                select(Message).where(Message.source_id == source.id).order_by(Message.seq)
            ).all()
            if not rows or all(r.sent_at is not None for r in rows):
                continue

            try:
                parser_notes: List[str] = []
                parsed = PARSERS[source.platform](source.file_path, parser_notes)
            except (OSError, ValueError, KeyError) as exc:
                log.warning("timestamp backfill skipped source %s: cannot re-parse (%s)", source.id, exc)
                continue

            if len(parsed) != len(rows) or any(p["text"] != r.text for p, r in zip(parsed, rows)):
                log.warning(
                    "timestamp backfill skipped source %s: file no longer matches stored messages",
                    source.id,
                )
                continue

            for parsed_msg, row in zip(parsed, rows):
                row.sent_at = parsed_msg.get("sent_at")
                session.add(row)
            source.date_note = _date_note(parsed, parser_notes)
            session.add(source)
            session.commit()
            filled += 1
    return filled


def recover_stranded_sources() -> int:
    """Implements Phase 5 ingestion-status polish. Ingestion runs on an
    in-process thread pool, so a server restart silently drops any queued or
    running job and leaves its Source 'pending'/'processing' forever (and the
    UI polling it forever). Called at startup: each such source is re-queued
    if its uploaded file is still there, and failed with a clear message if
    not. A running import writes all its rows in one transaction that only
    commits together with status='ready', so a stranded source has no rows to
    clean up - if it somehow does, re-running would duplicate them, so it is
    failed instead. Returns how many sources were touched.
    """
    touched = 0
    with Session(db_engine) as session:
        stranded = session.exec(
            select(Source).where(Source.status.in_(["pending", "processing"]))
        ).all()
        for source in stranded:
            has_rows = session.exec(select(Message.id).where(Message.source_id == source.id)).first()
            if has_rows is not None:
                source.status = "failed"
                source.error_message = "Import was interrupted by a server restart and left partial data."
            elif not source.file_path or not os.path.exists(source.file_path):
                source.status = "failed"
                source.error_message = "Upload was interrupted before the file was saved - please upload it again."
            else:
                source.status = "pending"
                session.add(source)
                session.commit()
                submit_ingestion(source.id)
                touched += 1
                continue
            session.add(source)
            session.commit()
            touched += 1
    return touched
