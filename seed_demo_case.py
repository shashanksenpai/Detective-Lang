"""Seeds bundled demo cases from sample WhatsApp exports through the real
ingestion path (not a special-cased shortcut), so the app isn't empty on
first run. Each case is checked/seeded independently by name, so adding a
new demo case here doesn't require wiping an existing install's database -
running this again only creates whichever demo cases aren't already present.
Phase 1 of the CLAUDE.md roadmap.
"""
from sqlmodel import Session, select

from db import engine as db_engine
from ingestion import ingest_source
from models import Case, Source

DEMO_CASES = [
    {
        "name": "Demo: Study Group",
        "description": "Seeded from the bundled sample exports.",
        "sources": [
            {"path": "sample_chat.txt", "label": "WhatsApp — Study Group", "context": "group"},
            {"path": "sample_dm_riya_karan.txt", "label": "WhatsApp — DM with Karan", "context": "dm"},
        ],
    },
    {
        "name": "Demo: Housemates",
        "description": (
            "A richer synthetic case - one person's emotional arc across a stressful week "
            "(calm -> anxious -> relieved), two steady contrasting baselines, and a group-vs-DM "
            "contrast for the same two people."
        ),
        "sources": [
            {"path": "sample_housemates.txt", "label": "WhatsApp — Housemates", "context": "group"},
            {"path": "sample_dm_meera_dev.txt", "label": "WhatsApp — DM with Dev", "context": "dm"},
        ],
    },
    {
        "name": "Demo: Other Platforms",
        "description": (
            "Phase 3 multi-platform demo - Karan (Instagram) and Riya (Telegram), both also in "
            "'Demo: Study Group' on WhatsApp. Ingesting this is expected to produce two real "
            "cross-platform merge suggestions against their WhatsApp identities - see Identity "
            "Review."
        ),
        "sources": [
            {"path": "sample_instagram_karan_zoya.json", "label": "Instagram — DM with Zoya",
             "platform": "instagram", "context": "dm"},
            {"path": "sample_telegram_group.json", "label": "Telegram — Weekend Plans",
             "platform": "telegram", "context": "group"},
        ],
    },
    {
        "name": "Demo: The Paper Leak",
        "description": (
            "Investigation demo - a friendly, generous classmate (Yash) is gradually narrowed down "
            "as the person selling leaked mid-sem papers. Nine people, ~730 messages: the class "
            "group plus five DMs (two of them between other people, handed over as witnesses). "
            "Built for the contradiction-detection phase: sample_leak_case_key.json lists every "
            "planted contradiction and innocent decoy with exact timestamps and quotes."
        ),
        "sources": [
            {"path": "sample_leak_group.txt", "label": "WhatsApp — Class Group", "context": "group"},
            {"path": "sample_leak_dm_yash_kritika.txt", "label": "WhatsApp — Yash & Kritika", "context": "dm"},
            {"path": "sample_leak_dm_yash_harshit.txt", "label": "WhatsApp — Yash & Harshit", "context": "dm"},
            {"path": "sample_leak_dm_yash_parth.txt", "label": "WhatsApp — Yash & Parth", "context": "dm"},
            {"path": "sample_leak_dm_meenakshi_sana.txt", "label": "WhatsApp — Meenakshi & Sana", "context": "dm"},
            {"path": "sample_leak_dm_arjun_nikhil.txt", "label": "WhatsApp — Arjun & Nikhil", "context": "dm"},
        ],
    },
]


def seed_demo_case_if_needed():
    with Session(db_engine) as session:
        existing_names = set(session.exec(select(Case.name)).all())

    for spec in DEMO_CASES:
        if spec["name"] in existing_names:
            continue
        _seed_one(spec)


def _seed_one(spec):
    with Session(db_engine) as session:
        case = Case(name=spec["name"], description=spec["description"])
        session.add(case)
        session.commit()
        session.refresh(case)

        source_ids = []
        for s in spec["sources"]:
            source = Source(
                case_id=case.id, label=s["label"], platform=s.get("platform", "whatsapp"),
                context=s["context"], status="pending", file_path=s["path"],
            )
            session.add(source)
            session.commit()
            session.refresh(source)
            source_ids.append(source.id)

    # Ingest synchronously (not via the thread pool) so the demo case is
    # fully ready by the time the server finishes starting up.
    for source_id in source_ids:
        ingest_source(source_id)
