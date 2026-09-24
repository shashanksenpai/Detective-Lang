"""Phase 5 regression tests for parser timestamp capture, especially the
WhatsApp date-order inference (parsers/whatsapp.py). Run: pytest test_timestamps.py
"""
import json
import re
from datetime import datetime

from parsers.instagram import UTC_NOTE, parse_instagram
from parsers.telegram import parse_telegram
from parsers.whatsapp import AMBIGUOUS_DATE_NOTE, parse_whatsapp

# The pre-Phase-5 line matcher, kept here verbatim: the new capture-group
# regex must accept exactly the same lines, or backfill_timestamps() would
# misalign old Message.seq values against a re-parse.
LEGACY_RE = re.compile(r"^\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2} [AP]M - ([^:]+): (.*)$")


def _wa(tmp_path, lines):
    p = tmp_path / "chat.txt"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    notes = []
    return parse_whatsapp(str(p), notes), notes


def test_day_above_12_forces_day_first(tmp_path):
    msgs, notes = _wa(tmp_path, [
        "13/01/23, 9:14 AM - Riya: hi",
        "14/01/23, 9:15 AM - Karan: hey",
    ])
    assert [m["sent_at"] for m in msgs] == [datetime(2023, 1, 13, 9, 14), datetime(2023, 1, 14, 9, 15)]
    assert notes == []


def test_month_above_12_when_read_day_first_forces_month_first(tmp_path):
    msgs, notes = _wa(tmp_path, [
        "03/12/25, 9:00 AM - Meera: a",
        "03/13/25, 9:00 AM - Meera: b",
    ])
    assert msgs[1]["sent_at"] == datetime(2025, 3, 13, 9, 0)
    assert msgs[0]["sent_at"] == datetime(2025, 3, 12, 9, 0)
    assert notes == []


def test_chronology_breaks_the_tie_when_only_one_order_runs_forward(tmp_path):
    # Month-first would read Dec 11 -> Jan 12 (backwards); day-first reads
    # Nov 12 -> Dec 1 (forwards), so day-first wins without any assumption.
    msgs, notes = _wa(tmp_path, [
        "12/11/23, 9:00 AM - A: x",
        "01/12/23, 9:00 AM - B: y",
    ])
    assert [m["sent_at"] for m in msgs] == [datetime(2023, 11, 12, 9, 0), datetime(2023, 12, 1, 9, 0)]
    assert notes == []


def test_genuinely_ambiguous_export_reads_month_first_and_says_so(tmp_path):
    msgs, notes = _wa(tmp_path, [
        "12/01/23, 9:14 AM - Riya: a",
        "12/02/23, 9:14 AM - Riya: b",
    ])
    assert msgs[0]["sent_at"] == datetime(2023, 12, 1, 9, 14)
    assert notes == [AMBIGUOUS_DATE_NOTE]


def test_unreadable_dates_leave_messages_undated_rather_than_guessing(tmp_path):
    msgs, notes = _wa(tmp_path, ["13/13/23, 9:14 AM - Riya: a"])
    assert msgs[0]["sent_at"] is None
    assert msgs[0]["text"] == "a"


def test_twelve_hour_clock_edges(tmp_path):
    msgs, _ = _wa(tmp_path, [
        "13/01/23, 12:05 AM - A: midnight",
        "13/01/23, 12:30 PM - A: noon",
        "13/01/23, 11:59 PM - A: late",
    ])
    assert [m["sent_at"].hour for m in msgs] == [0, 12, 23]


def test_real_samples_parse_and_match_the_legacy_line_set():
    for path in ("sample_chat.txt", "sample_housemates.txt", "sample_dm_meera_dev.txt", "sample_dm_riya_karan.txt"):
        with open(path, encoding="utf-8") as f:
            legacy = [LEGACY_RE.match(line.strip()) for line in f]
        legacy = [m.groups() for m in legacy if m]
        new = parse_whatsapp(path)
        assert [(m["sender"], m["text"]) for m in new] == legacy, path
        assert all(m["sent_at"] is not None for m in new), path


def test_housemates_sample_is_decided_not_assumed_and_starts_on_a_monday():
    notes = []
    msgs = parse_whatsapp("sample_housemates.txt", notes)
    assert notes == []
    assert msgs[0]["sent_at"] == datetime(2025, 3, 10, 9, 2)
    assert msgs[0]["sent_at"].weekday() == 0
    assert msgs[-1]["sent_at"].date() == datetime(2025, 3, 15).date()


def test_study_group_sample_is_flagged_as_assumed():
    notes = []
    msgs = parse_whatsapp("sample_chat.txt", notes)
    assert notes == [AMBIGUOUS_DATE_NOTE]
    assert msgs[0]["sent_at"] == datetime(2023, 12, 1, 9, 14)


def test_instagram_is_chronological_utc_and_noted(tmp_path):
    p = tmp_path / "ig.json"
    p.write_text(json.dumps({"messages": [
        {"sender_name": "B", "timestamp_ms": 1717301520000, "content": "second"},
        {"sender_name": "A", "timestamp_ms": 1717301460000, "content": "first"},
    ]}), encoding="utf-8")
    notes = []
    msgs = parse_instagram(str(p), notes)
    assert [m["text"] for m in msgs] == ["first", "second"]
    assert msgs[0]["sent_at"] == datetime(2024, 6, 2, 4, 11)
    assert notes == [UTC_NOTE]


def test_instagram_missing_timestamp_is_none_and_unnoted(tmp_path):
    p = tmp_path / "ig.json"
    p.write_text(json.dumps({"messages": [{"sender_name": "A", "content": "hi"}]}), encoding="utf-8")
    notes = []
    msgs = parse_instagram(str(p), notes)
    assert msgs[0]["sent_at"] is None
    assert notes == []


def test_telegram_dates_and_entity_text(tmp_path):
    p = tmp_path / "tg.json"
    p.write_text(json.dumps({"messages": [
        {"type": "service", "date": "2024-06-01T18:00:00", "action": "create_group"},
        {"type": "message", "date": "2024-06-01T18:01:00", "from": "Riya",
         "text": ["see ", {"type": "link", "text": "example.com"}]},
        {"type": "message", "date": "not-a-date", "from": "Zoya", "text": "hi"},
    ]}), encoding="utf-8")
    msgs = parse_telegram(str(p))
    assert msgs[0]["text"] == "see example.com"
    assert msgs[0]["sent_at"] == datetime(2024, 6, 1, 18, 1)
    assert msgs[1]["sent_at"] is None
