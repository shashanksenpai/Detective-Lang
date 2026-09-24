"""Cross-case person dossier - pools all of a person's messages across
every case they belong to (after a Phase 2 identity-resolution merge),
instead of one case's sources. Reuses person_profile.py's aggregation core
so the sentiment/traits/vocabulary math is identical to the case-scoped
dossier; the only difference is how "this person's own messages" gets
filtered - by identifier id here, since the same real person can have
different raw sender names in different cases.
"""
from collections import Counter

from sqlmodel import Session, select

from models import Case, Identifier, Message, Person, Source
from person_profile import build_person_profile_from_sources, content_words


def build_combined_profile(session: Session, person_id: int):
    person = session.get(Person, person_id)
    if person is None:
        return None

    identifiers = session.exec(select(Identifier).where(Identifier.person_id == person_id)).all()
    if not identifiers:
        return None
    identifier_id_by_source = {ident.source_id: ident.id for ident in identifiers}

    background_sources = []  # every sender's messages in these sources - the keyness baseline
    person_sources = []      # just this person's own messages
    case_labels = []

    for source_id, identifier_id in identifier_id_by_source.items():
        source = session.get(Source, source_id)
        case = session.get(Case, source.case_id)
        label = f"{case.name} — {source.label}"
        case_labels.append({"case_id": case.id, "case_name": case.name, "source_label": source.label})

        rows = session.exec(
            select(Message, Identifier)
            .join(Identifier, Message.identifier_id == Identifier.id)
            .where(Message.source_id == source_id)
            .order_by(Message.seq)
        ).all()
        all_msgs = [{"sender": ident.raw_sender_name, "text": msg.text} for msg, ident in rows]
        own_msgs = [
            {"sender": ident.raw_sender_name, "text": msg.text}
            for msg, ident in rows if ident.id == identifier_id
        ]

        background_sources.append({"label": label, "context": source.context, "messages": all_msgs})
        person_sources.append({"label": label, "context": source.context, "messages": own_msgs})

    global_freq = Counter()
    for src in background_sources:
        for m in src["messages"]:
            global_freq.update(content_words(m["text"]))
    global_total = sum(global_freq.values()) or 1

    profile = build_person_profile_from_sources(
        person.display_name, person_sources, global_freq, global_total
    )
    profile["person_id"] = person_id
    profile["cases"] = case_labels
    return profile
