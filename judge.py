"""The judge interface for contradiction detection - BACKLOG N-1 (roadmap Phase 6).

A judge looks at two statements from a chat and says whether they conflict.
Candidate pairs are found locally; a judge is the one stage that has to
understand language, so it is a plug-in point. The first semantic provider is
Gemini (GeminiJudge below) - and today it is a PLACEHOLDER: it checks its
configuration and then raises NotImplementedError. It never returns a made-up
verdict (project rule: no fake logic in a real code path).

Everything here is provisional - the shape of a pair, the prompt, the
structured-output schema, error handling and the record/replay cache that keeps
the eval reproducible are still being designed with the owner (BACKLOG N-1).
This module is deliberately light: no ML imports, and the google-genai SDK is
not imported at all yet, so nothing else in the app depends on it.

Facts checked against Google's own pages on 2026-09-24 (re-check before building):
- SDK: `pip install google-genai`; `from google import genai; genai.Client()` reads
  GEMINI_API_KEY. The docs' current call is `client.interactions.create(model=...,
  input=..., response_format={"type": "text", "mime_type": "application/json",
  "schema": <JSON schema>})` with the answer in `interaction.output_text`.
- Structured JSON output needs a Gemini 3-series model. Model IDs change often (the
  docs list e.g. gemini-3.8-flash, gemini-3.5-flash-lite, gemini-3.1-flash-lite), so
  the model is required configuration with no default.
- Free-tier rate limits are not published in the docs (only visible in AI Studio, "not
  guaranteed"); how safety-blocked responses and quota errors surface is not
  documented on the pages checked - both must be discovered against the real API.
- PRIVACY: on the unpaid tier Google "uses the content you submit ... and any generated
  responses to provide, improve, and develop Google products and services and machine
  learning technologies", "human reviewers may read, annotate, and process your API
  input and output", and it says "Do not submit sensitive, confidential, or personal
  information to the Unpaid Services". The paid tier does not use prompts or responses
  to improve products. Chat exports are personal information, so this judge is OFF
  unless explicitly switched on (see ENV_CONSENT), and the free tier should only ever
  see synthetic or consented data.
"""
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Protocol, Tuple

# What a judge may answer. "uncertain" is a first-class answer: a judge that
# cannot tell (or could not run) must say so rather than pick a side.
RELATIONS = ("conflict", "consistent", "unrelated", "uncertain")


class JudgeUnavailable(RuntimeError):
    """The judge cannot run right now (switched off, not configured). The message
    says what to change; it is meant to be shown to the user as is."""


@dataclass(frozen=True)
class Statement:
    """One message being compared, with what a finding must cite: which message,
    who said it, where (source + group/DM) and when."""
    message_id: int
    sender: str
    source_label: str
    context: str                    # "group" | "dm"
    sent_at: Optional[datetime]
    text: str


@dataclass(frozen=True)
class StatementPair:
    """Two statements plus the few messages around each, so a judge can read them
    the way a person would (a reply, a retraction, an explanation just after)."""
    a: Statement
    b: Statement
    context_a: Tuple[str, ...] = ()
    context_b: Tuple[str, ...] = ()


@dataclass(frozen=True)
class Verdict:
    relation: str                   # one of RELATIONS
    reason: str                     # one line the UI shows: what conflicts with what
    judge: str                      # who decided, incl. model id - for audit and cache keys

    def __post_init__(self):
        if self.relation not in RELATIONS:
            raise ValueError(f"relation must be one of {RELATIONS}, got {self.relation!r}")


class Judge(Protocol):
    name: str

    def unavailable_reason(self) -> Optional[str]:
        """None if the judge can run, else what is missing or switched off."""

    def judge(self, pair: StatementPair) -> Verdict:
        """Decide one pair. Raises JudgeUnavailable if it cannot run."""


class GeminiJudge:
    """Placeholder for the Gemini-backed semantic judge (see the module docstring)."""

    name = "gemini"
    ENV_KEY = "GEMINI_API_KEY"                  # the variable the google-genai SDK reads by default
    ENV_MODEL = "DETECTIVE_GEMINI_MODEL"        # no default: model IDs change, pick one deliberately
    ENV_CONSENT = "DETECTIVE_ALLOW_EXTERNAL_LLM"  # must be "1": the compared statements go to Google

    def unavailable_reason(self) -> Optional[str]:
        if os.environ.get(self.ENV_CONSENT) != "1":
            return (
                f"External LLM use is off. Set {self.ENV_CONSENT}=1 to allow the compared statements "
                "to be sent to Google (read the privacy note in the README first: on Google's free tier "
                "the content may be used to improve its products and read by reviewers)."
            )
        if not os.environ.get(self.ENV_KEY):
            return f"{self.ENV_KEY} is not set."
        if not os.environ.get(self.ENV_MODEL):
            return (
                f"{self.ENV_MODEL} is not set (there is deliberately no default; see "
                "https://ai.google.dev/gemini-api/docs/models)."
            )
        return None

    def judge(self, pair: StatementPair) -> Verdict:
        reason = self.unavailable_reason()
        if reason:
            raise JudgeUnavailable(reason)
        # Placeholder: no request is made and no verdict is invented.
        raise NotImplementedError(
            "GeminiJudge.judge is a placeholder (BACKLOG N-1): the prompt, structured-output schema, "
            "error handling and the record/replay cache are still being designed."
        )
