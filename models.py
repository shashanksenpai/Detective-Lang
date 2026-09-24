"""Case/Source/Person/Message data model - Phase 1 of the CLAUDE.md roadmap.
Person is deliberately global (not case-scoped) so the same real person can
later be linked across cases (Phase 2); Identifier resolution stays
case-scoped for now (see ingestion.py) so two cases' same-named senders are
never silently assumed to be the same person.
"""
from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel


class Case(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    description: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Source(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    case_id: int = Field(foreign_key="case.id")
    label: str
    platform: str  # whatsapp | instagram | telegram
    context: str  # group | dm
    status: str = "pending"  # pending | processing | ready | failed
    error_message: Optional[str] = None
    # Phase 5: caveats about how this source's timestamps were read (an
    # assumed WhatsApp date order, Instagram's UTC times, or "undated") -
    # see parsers/__init__.py. None means the dates were read unambiguously.
    date_note: Optional[str] = None
    file_path: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Person(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    display_name: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class PersonCase(SQLModel, table=True):
    person_id: int = Field(foreign_key="person.id", primary_key=True)
    case_id: int = Field(foreign_key="case.id", primary_key=True)


class Identifier(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    source_id: int = Field(foreign_key="source.id")
    raw_sender_name: str
    person_id: int = Field(foreign_key="person.id")


class Message(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    source_id: int = Field(foreign_key="source.id")
    identifier_id: int = Field(foreign_key="identifier.id")
    text: str
    sent_at: Optional[datetime] = None
    seq: int
    # BACKLOG F-03: "text" (something a person wrote) | "media" (a photo/video/sticker
    # placeholder) | "deleted" (a deleted-message notice). Non-text rows stay in the
    # record but are not analysed as speech - see engine_cache.load_case_sources.
    kind: str = "text"


class MergeSuggestion(SQLModel, table=True):
    """A candidate cross-case identity match awaiting human review - Phase 2
    of the CLAUDE.md roadmap. Never applied automatically; accepting one is
    what actually merges person_b into person_a (see identity_resolution.py).
    person_a_id is always the lower id, so a pair is never suggested twice
    in reversed order.
    """
    id: Optional[int] = Field(default=None, primary_key=True)
    person_a_id: int = Field(foreign_key="person.id")
    person_b_id: int = Field(foreign_key="person.id")
    match_type: str  # hard | fuzzy | soft
    confidence: float
    evidence: str  # JSON-encoded per-tier scores/details
    status: str = "pending"  # pending | accepted | rejected
    created_at: datetime = Field(default_factory=datetime.utcnow)
    decided_at: Optional[datetime] = None


class Pin(SQLModel, table=True):
    """A message an investigator has pinned as evidence, with their own note -
    Phase 5 of the CLAUDE.md roadmap. `message_id` is unique (one pin per
    message) and is the only link: the case is derived through the message's
    Source, so there's no separate case_id to fall out of sync. Messages are
    never edited or deleted, so a pin can't dangle.
    """
    id: Optional[int] = Field(default=None, primary_key=True)
    message_id: int = Field(foreign_key="message.id", unique=True, index=True)
    note: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)


class BoardLayout(SQLModel, table=True):
    """Where an investigator dragged each person to on the investigation board
    (one row per case). `positions` is JSON {raw sender name: [x, y]} in board
    coordinates - kept so a hand-arranged board survives a reload instead of
    springing back. Names that are no longer in the case are simply ignored by
    the page.
    """
    case_id: int = Field(foreign_key="case.id", primary_key=True)
    positions: str = "{}"
    updated_at: datetime = Field(default_factory=datetime.utcnow)

