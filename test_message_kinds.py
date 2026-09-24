"""BACKLOG F-03: media placeholders and deleted-message notices are kept in the
record but are not analysed as speech. Goes through the real ingestion path
against the throwaway test database (conftest.py). Run: pytest test_message_kinds.py
"""
import pytest
from sqlalchemy import create_engine
from sqlmodel import Session, select

import db
import engine_cache
import identity_resolution
import workspace
from combined_profile import build_combined_profile
from db import create_db_and_tables, engine as db_engine
from ingestion import backfill_message_kinds, ingest_source
from models import Case, Identifier, Message, Person, PersonCase, Source
from parsers.whatsapp import classify_text, parse_whatsapp


# ---- the classifier --------------------------------------------------------

@pytest.mark.parametrize("text, expected", [
    ("<Media omitted>", "media"),
    ("<media omitted>", "media"),
    ("‎<Media omitted>", "media"),
    ("<attached: 00000012-PHOTO-2023-12-25-21-15-10.jpg>", "media"),
    ("image omitted", "media"),
    ("sticker omitted", "media"),
    ("GIF omitted", "media"),
    ("Contact card omitted", "media"),
    ("IMG-20231225-WA0001.jpg (file attached)", "media"),
    ("This message was deleted", "deleted"),
    ("You deleted this message", "deleted"),
    ("\U0001F6AB This message was deleted.", "deleted"),
    ("you deleted this message.", "deleted"),
    # a person wrote these - a placeholder next to real words is still speech
    ("look at this <Media omitted>", "text"),
    ("<Media omitted> lol", "text"),
    ("I omitted the last paragraph", "text"),
    ("this message was deleted by mistake, sorry", "text"),
    ("the attached file is wrong", "text"),
    ("kal 5 baje", "text"),
    ("", "text"),
])
def test_classify_text(text, expected):
    assert classify_text(text) == expected


def test_parser_sets_kind(tmp_path):
    p = tmp_path / "chat.txt"
    p.write_text("\n".join([
        "25/12/2023, 9:14 pm - Riya: hello",
        "25/12/2023, 9:15 pm - Riya: <Media omitted>",
        "25/12/2023, 9:16 pm - Karan: This message was deleted",
        "25/12/2023, 9:17 pm - Karan: photo <Media omitted>",
    ]) + "\n", encoding="utf-8")
    assert [m["kind"] for m in parse_whatsapp(str(p))] == ["text", "media", "deleted", "text"]


# ---- through ingestion and every consumer ----------------------------------

CHAT = [
    "25/12/2023, 9:14 pm - Riya: bhai kal ka plan kya hai",           # 0 text
    "25/12/2023, 9:15 pm - Karan: 3 cheezein karni hain:",            # 1 text (multi-line)
    "1. notes print karne hain",
    "25/12/2023, 9:16 pm - Riya: <Media omitted>",                    # 2 media
    "25/12/2023, 9:17 pm - Karan: This message was deleted",          # 3 deleted
    "25/12/2023, 9:18 pm - Riya: theek hai chalo",                    # 4 text
    "26/12/2023, 8:05 am - Meera: <Media omitted>",                   # 5 media - Meera's only message
    "26/12/2023, 8:06 am - Karan: good morning yaar",                 # 6 text
]


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    create_db_and_tables()
    path = tmp_path_factory.mktemp("kinds") / "chat.txt"
    path.write_text("\n".join(CHAT) + "\n", encoding="utf-8")
    with Session(db_engine) as s:
        c = Case(name="F-03 kinds")
        s.add(c)
        s.commit()
        s.refresh(c)
        src = Source(case_id=c.id, label="chat", platform="whatsapp", context="group",
                     status="pending", file_path=str(path))
        s.add(src)
        s.commit()
        s.refresh(src)
        case_id, source_id = c.id, src.id
    ingest_source(source_id)
    with Session(db_engine) as s:
        assert s.get(Source, source_id).status == "ready", s.get(Source, source_id).error_message
    return case_id, source_id


def test_every_message_is_stored_with_its_kind_and_a_stable_seq(case):
    _, source_id = case
    with Session(db_engine) as s:
        rows = s.exec(select(Message).where(Message.source_id == source_id).order_by(Message.seq)).all()
    assert [r.seq for r in rows] == list(range(7))
    assert [r.kind for r in rows] == ["text", "text", "media", "deleted", "text", "media", "text"]
    assert rows[1].text == "3 cheezein karni hain:\n1. notes print karne hain"


def test_the_engine_loader_returns_only_speech(case):
    case_id, _ = case
    with Session(db_engine) as s:
        sources = engine_cache.load_case_sources(s, case_id)
    msgs = [m for src in sources for m in src["messages"]]
    assert [(m["sender"], m["text"]) for m in msgs] == [
        ("Riya", "bhai kal ka plan kya hai"),
        ("Karan", "3 cheezein karni hain:\n1. notes print karne hain"),
        ("Riya", "theek hai chalo"),
        ("Karan", "good morning yaar"),
    ]


def test_engine_and_profiles_never_see_media_or_deleted_notices(case):
    case_id, _ = case
    engine = engine_cache.get_engine(case_id)
    assert set(engine.centroids) == {"Riya", "Karan"}            # Meera only ever sent media
    assert engine.messages_by_sender["Riya"] == ["bhai kal ka plan kya hai", "theek hai chalo"]
    profiles = engine_cache.get_profiles(case_id)
    assert set(profiles) == {"Riya", "Karan"}
    assert profiles["Riya"]["message_count"] == 2 and profiles["Karan"]["message_count"] == 2


def test_the_timeline_counts_every_message_but_scores_mood_from_text_only(case):
    case_id, _ = case
    with Session(db_engine) as s:
        tl = workspace.timeline(s, case_id)
    by_date = {d["date"]: d for d in tl["days"]}
    day1, day2 = by_date["2023-12-25"], by_date["2023-12-26"]
    assert day1["total"] == 5                                      # matches the day view: nothing hidden
    assert day1["people"]["Riya"]["count"] == 3 and isinstance(day1["people"]["Riya"]["mood"], float)
    assert day1["people"]["Karan"]["count"] == 2 and isinstance(day1["people"]["Karan"]["mood"], float)
    assert day2["total"] == 2
    assert day2["people"]["Meera"] == {"count": 1, "mood": None}   # only media: nothing to score, not "neutral"
    assert isinstance(day2["people"]["Karan"]["mood"], float)
    assert "Meera" in tl["people"]


def test_search_and_browse_still_show_every_message_with_its_kind(case):
    case_id, _ = case
    with Session(db_engine) as s:
        res = workspace.list_messages(s, case_id)
        senders = workspace.senders(s, case_id)
    assert res["total"] == 7
    assert [m["kind"] for m in res["results"]] == ["text", "text", "media", "deleted", "text", "media", "text"]
    assert {"name": "Meera", "message_count": 1} in senders


def test_combined_profile_and_identity_evidence_skip_non_text(case):
    case_id, source_id = case
    with Session(db_engine) as s:
        riya = s.exec(select(Identifier).where(
            Identifier.source_id == source_id, Identifier.raw_sender_name == "Riya")).one()
        assert build_combined_profile(s, riya.person_id)["message_count"] == 2
        assert identity_resolution._person_texts(s, riya.person_id) == [
            "bhai kal ka plan kya hai", "theek hai chalo"]


# ---- data that predates the column -----------------------------------------

def test_backfill_relabels_old_whatsapp_rows_once_and_leaves_the_rest(case):
    create_db_and_tables()
    with Session(db_engine) as s:
        c = Case(name="F-03 legacy")
        s.add(c)
        s.commit()
        s.refresh(c)
        wa = Source(case_id=c.id, label="old wa", platform="whatsapp", context="group",
                    status="ready", file_path="x")
        ig = Source(case_id=c.id, label="old ig", platform="instagram", context="dm",
                    status="ready", file_path="y")
        s.add(wa)
        s.add(ig)
        s.commit()
        s.refresh(wa)
        s.refresh(ig)
        person = Person(display_name="Old")
        s.add(person)
        s.commit()
        s.refresh(person)
        s.add(PersonCase(person_id=person.id, case_id=c.id))
        ident_wa = Identifier(source_id=wa.id, raw_sender_name="Old", person_id=person.id)
        ident_ig = Identifier(source_id=ig.id, raw_sender_name="Old", person_id=person.id)
        s.add(ident_wa)
        s.add(ident_ig)
        s.commit()
        s.refresh(ident_wa)
        s.refresh(ident_ig)
        # every row reads "text" - exactly what an install from before F-03 has after the column is added
        for i, (src, ident, text) in enumerate([
            (wa, ident_wa, "<Media omitted>"),
            (wa, ident_wa, "This message was deleted"),
            (wa, ident_wa, "I omitted the last paragraph"),
            (wa, ident_wa, "hello"),
            (ig, ident_ig, "<Media omitted>"),
        ]):
            s.add(Message(source_id=src.id, identifier_id=ident.id, text=text, seq=i))
        s.commit()
        wa_id, ig_id = wa.id, ig.id

    assert backfill_message_kinds() == 2
    assert backfill_message_kinds() == 0                            # idempotent

    with Session(db_engine) as s:
        wa_kinds = [m.kind for m in s.exec(select(Message).where(Message.source_id == wa_id).order_by(Message.seq))]
        ig_kinds = [m.kind for m in s.exec(select(Message).where(Message.source_id == ig_id))]
    assert wa_kinds == ["media", "deleted", "text", "text"]
    assert ig_kinds == ["text"]                                     # only WhatsApp placeholders are recognised


def test_the_column_is_added_to_an_existing_message_table(tmp_path, monkeypatch):
    old = create_engine(f"sqlite:///{(tmp_path / 'old.db').as_posix()}")
    with old.begin() as conn:
        conn.exec_driver_sql(
            "create table message (id integer primary key, source_id integer, identifier_id integer, "
            "text varchar, sent_at datetime, seq integer)")
        conn.exec_driver_sql("insert into message (source_id, identifier_id, text, seq) values (1, 1, 'hi', 0)")
    monkeypatch.setattr(db, "engine", old)
    db._add_missing_columns()
    db._add_missing_columns()                                       # idempotent
    with old.begin() as conn:
        cols = {row[1] for row in conn.exec_driver_sql("pragma table_info(message)")}
        kinds = [row[0] for row in conn.exec_driver_sql("select kind from message")]
    assert "kind" in cols
    assert kinds == ["text"]                                        # existing rows default to text
