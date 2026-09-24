"""BACKLOG A-1: POST /cases/{id}/investigate can be answered by either engine, the choice is explicit
and validated, and a typo can never silently serve the other engine. Runs against the throwaway test
database (conftest.py). Run: pytest test_investigate_route.py
"""
import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

import engine_cache
from db import create_db_and_tables, engine as db_engine
from engine_kind import DEFAULT_ENGINE, ENGINE_KINDS, default_engine_kind
from models import Case, Identifier, Message, Person, Source

ASHA = ["dont worry guys ill pay", "its on me guys", "no stress ill cover it", "dont worry about it", "ill get the bill guys", "chill ill pay"]
BHARAT = ["send the notes please", "the deadline is friday", "submit the slides by noon", "check the rubric first", "notes are in the folder", "deadline moved to monday"]


def _case(name, by_sender):
    create_db_and_tables()
    with Session(db_engine) as s:
        case = Case(name=name)
        s.add(case)
        s.commit()
        s.refresh(case)
        src = Source(case_id=case.id, label="chat", platform="whatsapp", context="group", status="ready", file_path="x")
        s.add(src)
        s.commit()
        s.refresh(src)
        seq = 0
        for sender, texts in by_sender.items():
            person = Person(display_name=sender)
            s.add(person)
            s.commit()
            s.refresh(person)
            ident = Identifier(source_id=src.id, raw_sender_name=sender, person_id=person.id)
            s.add(ident)
            s.commit()
            s.refresh(ident)
            for text in texts:
                s.add(Message(source_id=src.id, identifier_id=ident.id, text=text, seq=seq))
                seq += 1
        s.commit()
        return case.id


@pytest.fixture(scope="module")
def client():
    import server   # heavy (loads the ML stack); TestClient without `with` does not run the seeding lifespan
    return TestClient(server.app)


@pytest.fixture(scope="module")
def case_id():
    return _case("route: two people", {"Asha": ASHA, "Bharat": BHARAT})


def _ask(client, case_id, **body):
    return client.post(f"/cases/{case_id}/investigate", json={"sentence": "dont worry guys ill pay", **body})


# ---- which engine ----------------------------------------------------------------

def test_the_learned_engine_is_the_default(monkeypatch):
    monkeypatch.delenv("DETECTIVE_ENGINE", raising=False)
    assert DEFAULT_ENGINE == "learned" == default_engine_kind()
    assert set(ENGINE_KINDS) == {"learned", "legacy"}


def test_the_environment_can_switch_the_default_and_a_typo_is_an_error_not_a_silent_fallback(monkeypatch):
    monkeypatch.setenv("DETECTIVE_ENGINE", "legacy")
    assert default_engine_kind() == "legacy"
    monkeypatch.setenv("DETECTIVE_ENGINE", "lerned")
    with pytest.raises(ValueError, match="DETECTIVE_ENGINE"):
        default_engine_kind()


def test_new_engine_refuses_an_unknown_kind():
    with pytest.raises(ValueError):
        engine_cache.new_engine("magic")


def test_a_request_without_an_engine_gets_the_servers_default(client, case_id, monkeypatch):
    monkeypatch.delenv("DETECTIVE_ENGINE", raising=False)
    assert _ask(client, case_id).json()["engine"] == "learned"
    monkeypatch.setenv("DETECTIVE_ENGINE", "legacy")
    assert _ask(client, case_id).json()["engine"] == "legacy"


def test_a_request_can_ask_for_either_engine_and_gets_that_engines_shape(client, case_id):
    learned = _ask(client, case_id, engine="learned").json()
    assert learned["engine"] == "learned"
    assert {"ranking", "uncertain", "plausible_senders", "evidence", "calibration", "notes"} <= set(learned)
    assert learned["ranking"][0]["sender"] == "Asha"

    legacy = _ask(client, case_id, engine="legacy").json()
    assert legacy["engine"] == "legacy"
    assert {"ranking", "uncertain", "weights_used"} <= set(legacy)
    assert "signals" in legacy["ranking"][0]


def test_an_unknown_engine_is_rejected(client, case_id):
    assert _ask(client, case_id, engine="lerned").status_code == 422


def test_context_is_accepted_by_both_engines_and_the_learned_one_says_it_ignored_it(client, case_id):
    ctx = [{"sender": "Bharat", "text": "who is paying?"}]
    learned = _ask(client, case_id, engine="learned", context=ctx).json()
    assert learned["context_used"] == 0 and any("were not used" in n for n in learned["notes"])
    assert _ask(client, case_id, engine="legacy", context=ctx).json()["context_used"] == 1


def test_a_case_with_no_messages_is_refused_by_either_engine(client):
    empty = _case("route: empty", {})
    for engine in ("learned", "legacy"):
        resp = _ask(client, empty, engine=engine)
        assert resp.status_code == 422 and "no ingested sources" in resp.json()["detail"]


def test_an_unknown_case_is_a_404(client):
    assert _ask(client, 999999, engine="learned").status_code == 404


# ---- the cache -------------------------------------------------------------------

def test_the_learned_engine_is_built_once_per_case_and_rebuilt_after_invalidation(case_id):
    first = engine_cache.get_attributor(case_id, "learned")
    assert engine_cache.get_attributor(case_id, "learned") is first
    engine_cache.invalidate(case_id)
    assert engine_cache.get_attributor(case_id, "learned") is not first


def test_the_legacy_attributor_is_the_shared_profile_engine(case_id):
    assert engine_cache.get_attributor(case_id, "legacy") is engine_cache.get_engine(case_id)


def test_new_messages_reach_the_learned_engine_after_invalidation():
    cid = _case("route: growing", {"Asha": ASHA, "Bharat": BHARAT})
    assert set(engine_cache.get_attributor(cid, "learned").senders) == {"Asha", "Bharat"}
    with Session(db_engine) as s:
        from sqlmodel import select
        src = s.exec(select(Source).where(Source.case_id == cid)).first()
        person = Person(display_name="Chitra")
        s.add(person)
        s.commit()
        s.refresh(person)
        ident = Identifier(source_id=src.id, raw_sender_name="Chitra", person_id=person.id)
        s.add(ident)
        s.commit()
        s.refresh(ident)
        for i in range(6):
            s.add(Message(source_id=src.id, identifier_id=ident.id, text=f"lol omg haha {i}", seq=100 + i))
        s.commit()
    assert "Chitra" not in engine_cache.get_attributor(cid, "learned").senders    # cached until told otherwise
    engine_cache.invalidate(cid)
    assert "Chitra" in engine_cache.get_attributor(cid, "learned").senders


# ---- /status ---------------------------------------------------------------------

def test_status_reports_the_engine_the_server_will_use(client, monkeypatch):
    monkeypatch.delenv("DETECTIVE_ENGINE", raising=False)
    att = client.get("/status").json()["attribution"]
    assert att == {"engine": "learned", "engines": ["learned", "legacy"]}
    monkeypatch.setenv("DETECTIVE_ENGINE", "legacy")
    assert client.get("/status").json()["attribution"]["engine"] == "legacy"
