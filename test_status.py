"""GET /status feeds the workstation bar's indicators, so every value must be the real
state, not a constant (BACKLOG N-3). Also covers the message_count the case index shows.
Runs against the throwaway test database (conftest.py). Run: pytest test_status.py
"""
import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

import hinglish_sentiment as hs
from db import create_db_and_tables, engine as db_engine
from judge import GeminiJudge
from models import Case, Identifier, Message, Person, Source
from status import system_status


@pytest.fixture(autouse=True)
def tables_and_clean_env(monkeypatch):
    create_db_and_tables()
    for var in (GeminiJudge.ENV_CONSENT, GeminiJudge.ENV_KEY, GeminiJudge.ENV_MODEL):
        monkeypatch.delenv(var, raising=False)


def _status():
    with Session(db_engine) as s:
        return system_status(s)


def test_it_reports_the_sentiment_scorer_that_is_really_loaded(monkeypatch):
    expected = "vader-fallback" if hs.get_analyzer().is_fallback else "hinglish-model"
    assert _status()["sentiment"]["engine"] == expected
    monkeypatch.setattr(hs, "_analyzer", hs._VaderFallback())      # force the fallback: the status must follow
    assert _status()["sentiment"]["engine"] == "vader-fallback"


def test_the_external_llm_is_off_by_default_and_says_why():
    st = _status()["external_llm"]
    assert st["state"] == "off" and st["implemented"] is False
    assert GeminiJudge.ENV_CONSENT in st["detail"]


def test_configured_is_not_mistaken_for_working(monkeypatch):
    monkeypatch.setenv(GeminiJudge.ENV_CONSENT, "1")
    monkeypatch.setenv(GeminiJudge.ENV_KEY, "not-a-real-key")
    monkeypatch.setenv(GeminiJudge.ENV_MODEL, "some-model")
    st = _status()["external_llm"]
    assert st["state"] == "configured"
    assert st["implemented"] is False and "placeholder" in st["detail"]     # nothing is sent yet


def test_the_placeholder_is_marked_unimplemented():
    assert GeminiJudge.implemented is False   # flip only when judge() really calls the API


def _add_case_with_messages(name, texts, status="ready"):
    with Session(db_engine) as s:
        case = Case(name=name)
        s.add(case)
        s.commit()
        s.refresh(case)
        src = Source(case_id=case.id, label="chat", platform="whatsapp", context="group", status=status, file_path="x")
        s.add(src)
        person = Person(display_name="A")
        s.add(person)
        s.commit()
        s.refresh(src)
        s.refresh(person)
        ident = Identifier(source_id=src.id, raw_sender_name="A", person_id=person.id)
        s.add(ident)
        s.commit()
        s.refresh(ident)
        for i, text in enumerate(texts):
            s.add(Message(source_id=src.id, identifier_id=ident.id, text=text, seq=i))
        s.commit()
        return case.id


def test_counts_follow_the_database():
    before = _status()["counts"]
    _add_case_with_messages("status counts", ["a", "b", "c"])
    after = _status()["counts"]
    assert after["cases"] == before["cases"] + 1
    assert after["sources_ready"] == before["sources_ready"] + 1
    assert after["messages"] == before["messages"] + 3


def test_a_source_that_is_not_ready_is_not_counted_as_ready():
    before = _status()["counts"]["sources_ready"]
    _add_case_with_messages("status not ready", [], status="failed")
    assert _status()["counts"]["sources_ready"] == before


# ---- the routes -----------------------------------------------------------

@pytest.fixture()
def client():
    import server   # heavy (loads the ML stack); TestClient without `with` does not run the seeding lifespan
    return TestClient(server.app)


def test_status_route(client):
    resp = client.get("/status")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"attribution", "sentiment", "external_llm", "counts"}
    assert body["sentiment"]["engine"] in ("hinglish-model", "vader-fallback")


def test_cases_route_reports_each_cases_message_count(client):
    case_id = _add_case_with_messages("status route counts", ["x", "y"])
    empty_id = _add_case_with_messages("status route empty", [])
    by_id = {c["id"]: c for c in client.get("/cases").json()}
    assert by_id[case_id]["message_count"] == 2
    assert by_id[empty_id]["message_count"] == 0
