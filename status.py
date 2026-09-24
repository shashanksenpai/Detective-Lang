"""System status for the workstation bar (GET /status) - BACKLOG N-3 / Phase 7.

Every value is read from real state at request time: which sentiment scorer is
actually loaded, whether the external-LLM switch is on, and what is in the
database. The UI shows these as indicators, so nothing here may be a guess or a
constant that could silently go stale. Kept light (no engine imports) so the
route and its test are cheap.
"""
from sqlalchemy import func
from sqlmodel import Session, select

from engine_kind import ENGINE_KINDS, default_engine_kind
from hinglish_sentiment import get_analyzer
from judge import GeminiJudge
from models import Case, Message, Source


def system_status(session: Session) -> dict:
    """Implements the /status contract the shell reads:
    sentiment.engine  "hinglish-model" | "vader-fallback"  (which scorer is really loaded)
    external_llm.state "off" | "configured"                 (configured != working: see `implemented`)
    attribution.engine "learned" | "legacy"                 (the engine /investigate uses unless a request picks one)
    counts            cases, ready sources and stored messages.
    """
    judge = GeminiJudge()
    reason = judge.unavailable_reason()
    if reason is None:
        state = "configured"
        detail = ("Configured, but the Gemini judge is still a placeholder: nothing is sent."
                  if not judge.implemented else "Configured: compared statements are sent to Google.")
    else:
        state, detail = "off", reason
    return {
        "attribution": {"engine": default_engine_kind(), "engines": list(ENGINE_KINDS)},
        "sentiment": {"engine": "vader-fallback" if get_analyzer().is_fallback else "hinglish-model"},
        "external_llm": {"state": state, "implemented": judge.implemented, "detail": detail},
        "counts": {
            "cases": session.exec(select(func.count(Case.id))).one(),
            "sources_ready": session.exec(select(func.count(Source.id)).where(Source.status == "ready")).one(),
            "messages": session.exec(select(func.count(Message.id))).one(),
        },
    }
