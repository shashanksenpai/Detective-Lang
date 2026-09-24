"""Which attribution engine answers "who said this?" (BACKLOG A-1). Light on purpose - no ML
imports - so /status and its test can read it without loading either engine.

"legacy" is DetectiveEngine (six hand-weighted signals); "learned" is learned_engine.LearnedEngine.
The relationship graph and the dossiers always use the legacy engine's per-person profiles; only
attribution is switchable.
"""
import os

ENGINE_KINDS = ("learned", "legacy")
DEFAULT_ENGINE = "learned"   # measured better on all three demo cases (BACKLOG A-1); "legacy" stays selectable


def default_engine_kind() -> str:
    """DETECTIVE_ENGINE=legacy|learned overrides DEFAULT_ENGINE. An unknown value is an error, not a
    silent fallback - a typo must not quietly serve the other engine."""
    kind = os.environ.get("DETECTIVE_ENGINE", DEFAULT_ENGINE)
    if kind not in ENGINE_KINDS:
        raise ValueError(f"DETECTIVE_ENGINE must be one of {ENGINE_KINDS}, not {kind!r}")
    return kind
