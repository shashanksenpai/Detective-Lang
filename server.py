"""FastAPI dev server for Detective Lang. Case-backed as of Phase 1 of the
CLAUDE.md roadmap: sources are uploaded per case instead of a hardcoded
SOURCES list, and the DetectiveEngine/person-profiles are built lazily per
case (engine_cache.py) instead of once at import time. Still not the
production backend (no auth, in-process job queue instead of Celery/Redis) -
just enough to make multi-case, multi-source investigation live locally.
"""
import json
import math
import os
import shutil
from contextlib import asynccontextmanager
from datetime import date, datetime
from typing import Dict, List, Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import or_
from sqlmodel import Session, select

import engine_cache
import graph_analysis
import identity_resolution
import workspace
from combined_profile import build_combined_profile
from db import create_db_and_tables, get_session
from ingestion import backfill_message_kinds, backfill_timestamps, recover_stranded_sources, submit_ingestion
from models import BoardLayout, Case, Identifier, MergeSuggestion, Message, Person, PersonCase, Source
from parsers import PARSERS
from seed_demo_case import seed_demo_case_if_needed
from ui_static import UIStaticFiles

UPLOAD_DIR = "uploads"


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    create_db_and_tables()
    seed_demo_case_if_needed()
    backfill_timestamps()
    backfill_message_kinds()
    recover_stranded_sources()
    yield


app = FastAPI(title="Detective Lang (dev)", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class CaseCreate(BaseModel):
    name: str
    description: str = ""


class ContextMessage(BaseModel):
    sender: Optional[str] = None
    text: str


class InvestigateRequest(BaseModel):
    sentence: str
    context: List[ContextMessage] = []


class RelationshipInferenceRequest(BaseModel):
    people: List[str]


class BoardLayoutIn(BaseModel):
    positions: Dict[str, List[float]]


class PinCreate(BaseModel):
    message_id: int
    note: str = Field("", max_length=2000)


class PinUpdate(BaseModel):
    note: str = Field(..., max_length=2000)


@app.post("/cases")
def create_case(req: CaseCreate, session: Session = Depends(get_session)):
    case = Case(name=req.name, description=req.description)
    session.add(case)
    session.commit()
    session.refresh(case)
    return case


@app.get("/cases")
def list_cases(session: Session = Depends(get_session)):
    cases = session.exec(select(Case).order_by(Case.created_at)).all()
    summaries = []
    for case in cases:
        sources = session.exec(select(Source).where(Source.case_id == case.id)).all()
        summaries.append({
            "id": case.id,
            "name": case.name,
            "description": case.description,
            "created_at": case.created_at,
            "source_count": len(sources),
            "ready_source_count": sum(1 for s in sources if s.status == "ready"),
            "importing_source_count": sum(1 for s in sources if s.status in ("pending", "processing")),
            "failed_source_count": sum(1 for s in sources if s.status == "failed"),
        })
    return summaries


@app.get("/cases/{case_id}")
def get_case(case_id: int, session: Session = Depends(get_session)):
    case = session.get(Case, case_id)
    if not case:
        raise HTTPException(status_code=404, detail=f"No case {case_id}")
    return case


@app.post("/cases/{case_id}/sources")
async def create_source(
    case_id: int,
    file: UploadFile = File(...),
    platform: str = Form(...),
    context: str = Form(...),
    label: str = Form(...),
    session: Session = Depends(get_session),
):
    case = session.get(Case, case_id)
    if not case:
        raise HTTPException(status_code=404, detail=f"No case {case_id}")
    if platform not in PARSERS:
        raise HTTPException(status_code=400, detail=f"Unknown platform '{platform}'")
    if context not in ("group", "dm"):
        raise HTTPException(status_code=400, detail="context must be 'group' or 'dm'")

    source = Source(
        case_id=case_id, label=label, platform=platform, context=context,
        status="pending", file_path="",
    )
    session.add(source)
    session.commit()
    session.refresh(source)

    dest_path = os.path.join(UPLOAD_DIR, f"{source.id}_{os.path.basename(file.filename)}")
    with open(dest_path, "wb") as out:
        shutil.copyfileobj(file.file, out)
    source.file_path = dest_path
    session.add(source)
    session.commit()
    session.refresh(source)

    submit_ingestion(source.id)
    return source


@app.get("/cases/{case_id}/sources")
def list_sources(case_id: int, session: Session = Depends(get_session)):
    case = session.get(Case, case_id)
    if not case:
        raise HTTPException(status_code=404, detail=f"No case {case_id}")
    sources = session.exec(
        select(Source).where(Source.case_id == case_id).order_by(Source.created_at)
    ).all()
    stats = workspace.source_stats(session, case_id)
    empty = {"message_count": 0, "first_at": None, "last_at": None}
    return [{**s.model_dump(), **stats.get(s.id, empty)} for s in sources]


@app.delete("/cases/{case_id}/sources/{source_id}")
def delete_failed_source(case_id: int, source_id: int, session: Session = Depends(get_session)):
    """Phase 5: remove a source whose import failed, so a bad upload doesn't
    sit in the case forever. Only failed sources: they never wrote any
    Identifier/Message rows, so there is nothing that could reference them.
    Removing a *ready* source would have to untangle people, merges and pins
    and is deliberately not offered."""
    _get_case_or_404(session, case_id)
    source = session.get(Source, source_id)
    if not source or source.case_id != case_id:
        raise HTTPException(status_code=404, detail=f"No source {source_id} in this case")
    if source.status != "failed":
        raise HTTPException(status_code=409, detail="Only failed imports can be removed.")
    if session.exec(select(Message.id).where(Message.source_id == source_id)).first() is not None:
        raise HTTPException(status_code=409, detail="This source has imported messages and can't be removed.")

    # Delete the uploaded file only if it lives in uploads/ - never the bundled sample files.
    upload_root = os.path.abspath(UPLOAD_DIR)
    path = os.path.abspath(source.file_path) if source.file_path else None
    if path and os.path.commonpath([upload_root, path]) == upload_root and os.path.isfile(path):
        os.remove(path)
    session.delete(source)
    session.commit()
    return {"status": "deleted"}


@app.post("/cases/{case_id}/investigate")
def investigate(case_id: int, req: InvestigateRequest, session: Session = Depends(get_session)):
    case = session.get(Case, case_id)
    if not case:
        raise HTTPException(status_code=404, detail=f"No case {case_id}")
    det_engine = engine_cache.get_engine(case_id)
    if not det_engine.centroids:
        raise HTTPException(
            status_code=422,
            detail="This case has no ingested sources yet - upload a chat export first.",
        )
    context = [c.model_dump() for c in req.context]
    return det_engine.investigate(req.sentence, context=context)


@app.get("/cases/{case_id}/people")
def people(case_id: int, session: Session = Depends(get_session)):
    case = session.get(Case, case_id)
    if not case:
        raise HTTPException(status_code=404, detail=f"No case {case_id}")
    profiles = engine_cache.get_profiles(case_id)
    return sorted(
        [
            {"name": p["name"], "message_count": p["message_count"], "sources": p["sources"]}
            for p in profiles.values()
        ],
        key=lambda p: p["name"],
    )


@app.get("/cases/{case_id}/person/{name}")
def person(case_id: int, name: str, session: Session = Depends(get_session)):
    case = session.get(Case, case_id)
    if not case:
        raise HTTPException(status_code=404, detail=f"No case {case_id}")
    profile = engine_cache.get_profiles(case_id).get(name)
    if not profile:
        raise HTTPException(status_code=404, detail=f"No profile for '{name}' in this case")

    # Phase 2: tell the frontend whether this person has been merged with
    # someone from another case, so it can offer the combined-dossier view.
    identifier = session.exec(
        select(Identifier)
        .join(Source, Identifier.source_id == Source.id)
        .where(Source.case_id == case_id, Identifier.raw_sender_name == name)
    ).first()
    if identifier:
        linked_case_count = len(
            session.exec(select(PersonCase).where(PersonCase.person_id == identifier.person_id)).all()
        )
        profile = {**profile, "person_id": identifier.person_id, "linked_case_count": linked_case_count}
    return profile


@app.post("/cases/{case_id}/rescan-matches")
def rescan_matches(case_id: int, session: Session = Depends(get_session)):
    case = session.get(Case, case_id)
    if not case:
        raise HTTPException(status_code=404, detail=f"No case {case_id}")
    identity_resolution.scan_for_matches(case_id)
    return {"status": "scanned"}


def _person_summary(session: Session, person_id: int):
    person_row = session.get(Person, person_id)
    cases = session.exec(
        select(Case).join(PersonCase, PersonCase.case_id == Case.id).where(PersonCase.person_id == person_id)
    ).all()
    return {
        "person_id": person_id,
        "display_name": person_row.display_name if person_row else "(deleted)",
        "cases": [{"id": c.id, "name": c.name} for c in cases],
    }


@app.get("/merge-suggestions")
def list_merge_suggestions(status: str = "pending", session: Session = Depends(get_session)):
    suggestions = session.exec(
        select(MergeSuggestion)
        .where(MergeSuggestion.status == status)
        .order_by(MergeSuggestion.created_at)
    ).all()
    return [
        {
            "id": s.id,
            "match_type": s.match_type,
            "confidence": s.confidence,
            "evidence": json.loads(s.evidence),
            "status": s.status,
            "created_at": s.created_at,
            "person_a": _person_summary(session, s.person_a_id),
            "person_b": _person_summary(session, s.person_b_id),
        }
        for s in suggestions
    ]


@app.post("/merge-suggestions/{suggestion_id}/reject")
def reject_merge(suggestion_id: int, session: Session = Depends(get_session)):
    suggestion = session.get(MergeSuggestion, suggestion_id)
    if not suggestion:
        raise HTTPException(status_code=404, detail=f"No suggestion {suggestion_id}")
    if suggestion.status != "pending":
        raise HTTPException(status_code=400, detail=f"Suggestion already {suggestion.status}")
    suggestion.status = "rejected"
    suggestion.decided_at = datetime.utcnow()
    session.add(suggestion)
    session.commit()
    return {"status": "rejected"}


@app.post("/merge-suggestions/{suggestion_id}/accept")
def accept_merge(suggestion_id: int, session: Session = Depends(get_session)):
    suggestion = session.get(MergeSuggestion, suggestion_id)
    if not suggestion:
        raise HTTPException(status_code=404, detail=f"No suggestion {suggestion_id}")
    if suggestion.status != "pending":
        raise HTTPException(status_code=400, detail=f"Suggestion already {suggestion.status}")

    # person_a_id is always the lower id (identity_resolution.py) - it survives.
    survivor_id, removed_id = suggestion.person_a_id, suggestion.person_b_id

    for ident in session.exec(select(Identifier).where(Identifier.person_id == removed_id)).all():
        ident.person_id = survivor_id
        session.add(ident)

    survivor_cases = {
        pc.case_id for pc in session.exec(select(PersonCase).where(PersonCase.person_id == survivor_id)).all()
    }
    for pc in session.exec(select(PersonCase).where(PersonCase.person_id == removed_id)).all():
        if pc.case_id not in survivor_cases:
            session.add(PersonCase(person_id=survivor_id, case_id=pc.case_id))
        session.delete(pc)

    removed_person = session.get(Person, removed_id)
    if removed_person:
        session.delete(removed_person)

    suggestion.status = "accepted"
    suggestion.decided_at = datetime.utcnow()
    session.add(suggestion)
    session.commit()

    # Any other pending suggestion that referenced the now-removed person
    # needs to point at the survivor instead, or be dropped if that would
    # make it reference the same person on both sides (already resolved by
    # this merge) or duplicate another pending suggestion.
    others = session.exec(
        select(MergeSuggestion).where(
            MergeSuggestion.status == "pending",
            MergeSuggestion.id != suggestion_id,
            or_(MergeSuggestion.person_a_id == removed_id, MergeSuggestion.person_b_id == removed_id),
        )
    ).all()
    for other in others:
        a = survivor_id if other.person_a_id == removed_id else other.person_a_id
        b = survivor_id if other.person_b_id == removed_id else other.person_b_id
        note = None
        if a == b:
            note = "resolved by another merge"
        else:
            lo, hi = (a, b) if a < b else (b, a)
            dup = session.exec(
                select(MergeSuggestion).where(
                    MergeSuggestion.id != other.id,
                    MergeSuggestion.person_a_id == lo,
                    MergeSuggestion.person_b_id == hi,
                )
            ).first()
            if dup:
                note = "duplicate of another suggestion after merge"
            else:
                other.person_a_id, other.person_b_id = lo, hi

        if note:
            other.status = "rejected"
            other.decided_at = datetime.utcnow()
            evidence = json.loads(other.evidence)
            evidence["note"] = note
            other.evidence = json.dumps(evidence)
        session.add(other)
    session.commit()

    return {"status": "accepted", "survivor_person_id": survivor_id}


@app.get("/people/{person_id}")
def combined_person(person_id: int, session: Session = Depends(get_session)):
    profile = build_combined_profile(session, person_id)
    if not profile:
        raise HTTPException(status_code=404, detail=f"No person {person_id}")
    return profile


@app.get("/cases/{case_id}/graph")
def graph(case_id: int, session: Session = Depends(get_session)):
    case = session.get(Case, case_id)
    if not case:
        raise HTTPException(status_code=404, detail=f"No case {case_id}")
    if not engine_cache.get_engine(case_id).centroids:
        raise HTTPException(
            status_code=422,
            detail="This case has no ingested sources yet - upload a chat export first.",
        )
    return graph_analysis.build_graph(case_id)


@app.post("/cases/{case_id}/relationship-inference")
def relationship_inference(
    case_id: int, req: RelationshipInferenceRequest, session: Session = Depends(get_session)
):
    case = session.get(Case, case_id)
    if not case:
        raise HTTPException(status_code=404, detail=f"No case {case_id}")
    if len(req.people) < 2:
        raise HTTPException(status_code=400, detail="Select at least 2 people to infer a relationship.")
    try:
        return graph_analysis.infer_relationship(case_id, req.people)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# ---- Phase 5: case workspace (search, message context, evidence pins) --------------------

def _get_case_or_404(session: Session, case_id: int) -> Case:
    case = session.get(Case, case_id)
    if not case:
        raise HTTPException(status_code=404, detail=f"No case {case_id}")
    return case


@app.get("/cases/{case_id}/senders")
def case_senders(case_id: int, session: Session = Depends(get_session)):
    _get_case_or_404(session, case_id)
    return workspace.senders(session, case_id)


@app.get("/cases/{case_id}/messages")
def case_messages(
    case_id: int,
    q: str = "",
    sender: Optional[str] = None,
    source_id: Optional[int] = None,
    on_date: Optional[date] = Query(None, alias="date"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    session: Session = Depends(get_session),
):
    _get_case_or_404(session, case_id)
    return workspace.list_messages(
        session, case_id, q=q, sender=sender, source_id=source_id,
        on_date=on_date, limit=limit, offset=offset,
    )


@app.get("/cases/{case_id}/messages/{message_id}/context")
def case_message_context(
    case_id: int,
    message_id: int,
    before: int = Query(5, ge=0, le=25),
    after: int = Query(5, ge=0, le=25),
    session: Session = Depends(get_session),
):
    _get_case_or_404(session, case_id)
    try:
        return workspace.message_context(session, case_id, message_id, before=before, after=after)
    except workspace.NotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.get("/cases/{case_id}/timeline")
def case_timeline(case_id: int, session: Session = Depends(get_session)):
    _get_case_or_404(session, case_id)
    return engine_cache.get_timeline(case_id)


@app.get("/cases/{case_id}/board-layout")
def get_board_layout(case_id: int, session: Session = Depends(get_session)):
    _get_case_or_404(session, case_id)
    row = session.get(BoardLayout, case_id)
    return {"positions": json.loads(row.positions) if row else {}}


@app.put("/cases/{case_id}/board-layout")
def save_board_layout(case_id: int, req: BoardLayoutIn, session: Session = Depends(get_session)):
    """Phase 5+ investigation board: remember where each person was dragged to."""
    _get_case_or_404(session, case_id)
    if len(req.positions) > 500:
        raise HTTPException(status_code=400, detail="Too many positions.")
    for name, xy in req.positions.items():
        if not 1 <= len(name) <= 200:
            raise HTTPException(status_code=400, detail="Bad person name in layout.")
        if len(xy) != 2 or not all(math.isfinite(v) and abs(v) <= 1_000_000 for v in xy):
            raise HTTPException(status_code=400, detail=f"Bad position for '{name}'.")
    row = session.get(BoardLayout, case_id) or BoardLayout(case_id=case_id)
    row.positions = json.dumps(req.positions)
    row.updated_at = datetime.utcnow()
    session.add(row)
    session.commit()
    return {"status": "saved", "count": len(req.positions)}


@app.delete("/cases/{case_id}/board-layout")
def reset_board_layout(case_id: int, session: Session = Depends(get_session)):
    _get_case_or_404(session, case_id)
    row = session.get(BoardLayout, case_id)
    if row:
        session.delete(row)
        session.commit()
    return {"status": "reset"}


@app.get("/cases/{case_id}/pins")
def case_pins(case_id: int, session: Session = Depends(get_session)):
    _get_case_or_404(session, case_id)
    return workspace.list_pins(session, case_id)


@app.post("/cases/{case_id}/pins")
def create_pin(case_id: int, req: PinCreate, session: Session = Depends(get_session)):
    _get_case_or_404(session, case_id)
    try:
        return workspace.create_pin(session, case_id, req.message_id, req.note)
    except workspace.NotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.patch("/cases/{case_id}/pins/{pin_id}")
def update_pin(case_id: int, pin_id: int, req: PinUpdate, session: Session = Depends(get_session)):
    _get_case_or_404(session, case_id)
    try:
        return workspace.update_pin(session, case_id, pin_id, req.note)
    except workspace.NotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.delete("/cases/{case_id}/pins/{pin_id}")
def delete_pin(case_id: int, pin_id: int, session: Session = Depends(get_session)):
    _get_case_or_404(session, case_id)
    try:
        workspace.delete_pin(session, case_id, pin_id)
    except workspace.NotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"status": "deleted"}


# Only the top-level .html pages are served (ui_static.py) - never the database,
# uploads, source or .git that live in the same directory.
app.mount("/", UIStaticFiles(directory=".", html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
