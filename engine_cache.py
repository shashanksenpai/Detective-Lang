"""Per-case DetectiveEngine / person-profile caching. The engine used to be
built once from a hardcoded file at import time; now it's built lazily per
case from that case's Message rows and dropped whenever a source finishes
ingesting. Phase 1 of the CLAUDE.md roadmap.
"""
from sqlmodel import Session, select

from db import engine as db_engine
from detective import DetectiveEngine
from models import Identifier, Message, Source
import workspace
from person_profile import build_all_profiles

_engine_cache = {}
_profiles_cache = {}
_timeline_cache = {}


def invalidate(case_id: int):
    _engine_cache.pop(case_id, None)
    _profiles_cache.pop(case_id, None)
    _timeline_cache.pop(case_id, None)


def load_case_sources(session: Session, case_id: int):
    """Returns [{"label", "context", "messages": [{"sender", "text"}]}] for
    every ready source in this case, messages in original per-source order.
    Public: also reused by eval_attribution.py's train/test split.
    """
    sources = session.exec(
        select(Source).where(Source.case_id == case_id, Source.status == "ready")
    ).all()

    result = []
    for source in sources:
        rows = session.exec(
            select(Message, Identifier)
            .join(Identifier, Message.identifier_id == Identifier.id)
            .where(Message.source_id == source.id)
            .order_by(Message.seq)
        ).all()
        messages = [{"sender": ident.raw_sender_name, "text": msg.text} for msg, ident in rows]
        result.append({"label": source.label, "context": source.context, "messages": messages})
    return result


def get_engine(case_id: int) -> DetectiveEngine:
    if case_id not in _engine_cache:
        with Session(db_engine) as session:
            sources = load_case_sources(session, case_id)
        det_engine = DetectiveEngine()
        all_messages = [m for src in sources for m in src["messages"]]
        if all_messages:
            det_engine.build(all_messages)
        _engine_cache[case_id] = det_engine
    return _engine_cache[case_id]


def get_profiles(case_id: int) -> dict:
    if case_id not in _profiles_cache:
        with Session(db_engine) as session:
            sources = load_case_sources(session, case_id)
        _profiles_cache[case_id] = build_all_profiles(sources) if sources else {}
    return _profiles_cache[case_id]


def get_timeline(case_id: int) -> dict:
    """Phase 5 timeline data (workspace.timeline), cached per case and dropped
    by invalidate() like the engine and profiles - scoring every message's
    sentiment isn't free on a large export."""
    if case_id not in _timeline_cache:
        with Session(db_engine) as session:
            _timeline_cache[case_id] = workspace.timeline(session, case_id)
    return _timeline_cache[case_id]
