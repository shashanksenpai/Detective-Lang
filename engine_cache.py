"""Per-case DetectiveEngine / person-profile caching. The engine used to be
built once from a hardcoded file at import time; now it's built lazily per
case from that case's Message rows and dropped whenever a source finishes
ingesting. Phase 1 of the CLAUDE.md roadmap.
"""
from sqlmodel import Session, select

from db import engine as db_engine
from detective import DetectiveEngine
from engine_kind import DEFAULT_ENGINE, ENGINE_KINDS, default_engine_kind  # noqa: F401  (re-exported for callers)
from models import Identifier, Message, Source
import workspace
from person_profile import build_all_profiles

# Which engine answers "who said this?" is engine_kind.py's business (BACKLOG A-1); get_engine() stays the
# legacy engine because the relationship graph and dossiers read its per-person profiles.
_engine_cache = {}
_attributor_cache = {}
_profiles_cache = {}
_timeline_cache = {}


def new_engine(kind: str):
    """A fresh, unbuilt attribution engine. Public: also used by eval_attribution.py."""
    if kind == "legacy":
        return DetectiveEngine()
    if kind == "learned":
        from learned_engine import LearnedEngine
        return LearnedEngine()
    raise ValueError(f"unknown engine {kind!r}; expected one of {ENGINE_KINDS}")


def invalidate(case_id: int):
    _engine_cache.pop(case_id, None)
    for key in [k for k in _attributor_cache if k[0] == case_id]:
        del _attributor_cache[key]
    _profiles_cache.pop(case_id, None)
    _timeline_cache.pop(case_id, None)


def load_case_sources(session: Session, case_id: int):
    """Returns [{"label", "context", "messages": [{"sender", "text"}]}] for
    every ready source in this case, messages in original per-source order.
    Public: also reused by eval_attribution.py's train/test split.

    Only messages a person actually wrote (kind == "text", BACKLOG F-03): media
    placeholders and deleted-message notices are not speech, so they never reach
    the engine, profiles, sentiment, relationship graph or evaluation. One rule
    at one choke point; the trade-off is that a photo sent as a reply is not
    counted as a turn or an exchange.
    """
    sources = session.exec(
        select(Source).where(Source.case_id == case_id, Source.status == "ready")
    ).all()

    result = []
    for source in sources:
        rows = session.exec(
            select(Message, Identifier)
            .join(Identifier, Message.identifier_id == Identifier.id)
            .where(Message.source_id == source.id, Message.kind == "text")
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


def get_attributor(case_id: int, kind: str = None):
    """The engine that answers /investigate for this case, built lazily and cached like
    get_engine(). Both kinds expose `is_ready` and `investigate(sentence, context=None)`."""
    kind = kind or default_engine_kind()
    if kind == "legacy":
        return get_engine(case_id)
    if (case_id, kind) not in _attributor_cache:
        with Session(db_engine) as session:
            sources = load_case_sources(session, case_id)
        engine = new_engine(kind)
        engine.build([m for src in sources for m in src["messages"]])
        _attributor_cache[(case_id, kind)] = engine
    return _attributor_cache[(case_id, kind)]


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
