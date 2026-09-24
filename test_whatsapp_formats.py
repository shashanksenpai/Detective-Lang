"""BACKLOG F-01 / F-02: the WhatsApp export layouts real phones produce, and
multi-line messages. The original en-US behaviour is pinned separately by
test_timestamps.py (legacy line-set parity) and test_leak_case.py.
Run: pytest test_whatsapp_formats.py
"""
from datetime import datetime

import pytest

from parsers.whatsapp import AMBIGUOUS_DATE_NOTE, AMBIGUOUS_DATE_NOTE_DAY_FIRST, parse_whatsapp


def _wa(tmp_path, lines, prefix=""):
    p = tmp_path / "chat.txt"
    p.write_text(prefix + "\n".join(lines) + "\n", encoding="utf-8")
    notes = []
    return parse_whatsapp(str(p), notes), notes


def _stamps(msgs):
    return [m["sent_at"] for m in msgs]


# ---- F-01: layouts ---------------------------------------------------------

def test_indian_locale_lowercase_12_hour_day_first(tmp_path):
    msgs, notes = _wa(tmp_path, [
        "25/12/2023, 9:14 pm - Riya: hi",
        "26/12/2023, 8:05 am - Karan: hey",
    ])
    assert _stamps(msgs) == [datetime(2023, 12, 25, 21, 14), datetime(2023, 12, 26, 8, 5)]
    assert [m["sender"] for m in msgs] == ["Riya", "Karan"]
    assert notes == []


def test_24_hour_clock(tmp_path):
    msgs, notes = _wa(tmp_path, [
        "25/12/2023, 21:14 - Riya: hi",
        "25/12/2023, 00:05 - Karan: past midnight",
    ])
    assert _stamps(msgs) == [datetime(2023, 12, 25, 21, 14), datetime(2023, 12, 25, 0, 5)]
    assert notes == []


def test_narrow_no_break_space_before_am_pm(tmp_path):
    msgs, notes = _wa(tmp_path, [
        "12/25/23, 9:14 PM - A: x",
        "12/26/23, 9:15 pm - B: y",
    ])
    assert _stamps(msgs) == [datetime(2023, 12, 25, 21, 14), datetime(2023, 12, 26, 21, 15)]
    assert notes == []


def test_dotted_dates(tmp_path):
    msgs, notes = _wa(tmp_path, ["25.12.23, 21:14 - A: x", "25.12.2023, 21:15 - B: y"])
    assert _stamps(msgs) == [datetime(2023, 12, 25, 21, 14), datetime(2023, 12, 25, 21, 15)]
    assert notes == []


def test_ios_brackets_seconds_and_invisible_marks(tmp_path):
    msgs, notes = _wa(tmp_path, [
        "‎[25/12/23, 21:14:05] Riya: hi",
        "[25/12/23, 21:14:40] ‎Karan: ‎hello",
    ])
    assert [(m["sender"], m["text"]) for m in msgs] == [("Riya", "hi"), ("Karan", "hello")]
    assert _stamps(msgs) == [datetime(2023, 12, 25, 21, 14, 5), datetime(2023, 12, 25, 21, 14, 40)]
    assert notes == []


def test_ios_12_hour_us_style(tmp_path):
    msgs, notes = _wa(tmp_path, ["[12/25/23, 9:14:05 PM] A: x", "[12/25/23, 9:15:00 PM] B: y"])
    assert _stamps(msgs) == [datetime(2023, 12, 25, 21, 14, 5), datetime(2023, 12, 25, 21, 15, 0)]
    assert notes == []


def test_year_first_dates_are_never_ambiguous(tmp_path):
    msgs, notes = _wa(tmp_path, ["2023-01-02, 21:14 - A: x", "2023-01-03, 08:05 - B: y"])
    assert _stamps(msgs) == [datetime(2023, 1, 2, 21, 14), datetime(2023, 1, 3, 8, 5)]
    assert notes == []


def test_byte_order_mark_does_not_eat_the_first_message(tmp_path):
    msgs, _ = _wa(tmp_path, ["25/12/2023, 9:14 pm - Riya: first"], prefix="﻿")
    assert [m["text"] for m in msgs] == ["first"]


def test_a_colon_in_the_text_and_a_phone_number_sender(tmp_path):
    msgs, _ = _wa(tmp_path, [
        "25/12/2023, 9:14 pm - Riya: meet at 5:30: ok",
        "25/12/2023, 9:15 pm - +91 98765 43210: hi",
    ])
    assert (msgs[0]["sender"], msgs[0]["text"]) == ("Riya", "meet at 5:30: ok")
    assert msgs[1]["sender"] == "+91 98765 43210"


# ---- F-01: which order to assume when the file cannot say ------------------

def test_ambiguous_indian_style_defaults_to_day_first_and_says_so(tmp_path):
    msgs, notes = _wa(tmp_path, [
        "12/01/2023, 9:14 am - A: a",
        "12/02/2023, 9:14 am - A: b",
    ])
    assert _stamps(msgs) == [datetime(2023, 1, 12, 9, 14), datetime(2023, 2, 12, 9, 14)]
    assert notes == [AMBIGUOUS_DATE_NOTE_DAY_FIRST]


@pytest.mark.parametrize("lines", [
    ["12/01/2023, 09:14 - A: a", "12/02/2023, 09:14 - A: b"],      # 24-hour
    ["12.01.23, 09:14 - A: a", "12.02.23, 09:14 - A: b"],          # dotted
])
def test_ambiguous_24_hour_and_dotted_default_to_day_first(tmp_path, lines):
    msgs, notes = _wa(tmp_path, lines)
    assert _stamps(msgs)[0].month == 1 and _stamps(msgs)[1].month == 2
    assert notes == [AMBIGUOUS_DATE_NOTE_DAY_FIRST]


def test_upper_case_am_pm_keeps_the_original_month_first_default_even_with_a_4_digit_year(tmp_path):
    msgs, notes = _wa(tmp_path, [
        "12/01/2023, 9:14 AM - A: a",
        "12/02/2023, 9:14 AM - A: b",
    ])
    assert _stamps(msgs)[0] == datetime(2023, 12, 1, 9, 14)
    assert notes == [AMBIGUOUS_DATE_NOTE]


def test_no_note_when_both_orders_read_every_line_identically(tmp_path):
    msgs, notes = _wa(tmp_path, ["05/05/23, 9:14 AM - A: x", "06/06/23, 9:14 AM - A: y"])
    assert _stamps(msgs) == [datetime(2023, 5, 5, 9, 14), datetime(2023, 6, 6, 9, 14)]
    assert notes == []


def test_impossible_clock_times_leave_the_file_undated(tmp_path):
    msgs, notes = _wa(tmp_path, ["25/12/2023, 24:00 - A: x"])
    assert msgs[0]["text"] == "x" and msgs[0]["sent_at"] is None


# ---- F-02: multi-line messages and system lines ----------------------------

def test_continuation_lines_join_the_previous_message(tmp_path):
    msgs, _ = _wa(tmp_path, [
        "25/12/2023, 9:14 pm - Riya: line one",
        "line two",
        "",
        "  indented line four",
        "25/12/2023, 9:15 pm - Karan: reply",
    ])
    assert len(msgs) == 2
    assert msgs[0]["text"] == "line one\nline two\n\n  indented line four"
    assert msgs[1]["text"] == "reply"


def test_a_system_event_ends_the_message_and_its_own_extra_lines_are_dropped(tmp_path):
    msgs, _ = _wa(tmp_path, [
        "25/12/2023, 9:00 pm - Messages and calls are end-to-end encrypted. No one outside of this chat can read them.",
        "25/12/2023, 9:14 pm - Riya: hello",
        "25/12/2023, 9:15 pm - Riya added Karan",
        "this line belongs to the system event, not to Riya's hello",
        "25/12/2023, 9:16 pm - Karan: hi",
    ])
    assert [(m["sender"], m["text"]) for m in msgs] == [("Riya", "hello"), ("Karan", "hi")]


def test_text_before_the_first_header_and_trailing_blank_lines_are_dropped(tmp_path):
    msgs, _ = _wa(tmp_path, ["a preamble line", "25/12/2023, 9:14 pm - Riya: hello", "", ""])
    assert [m["text"] for m in msgs] == ["hello"]


def test_a_date_without_the_dash_is_not_a_header(tmp_path):
    msgs, _ = _wa(tmp_path, [
        "25/12/2023, 9:14 pm - Riya: agenda",
        "25/12/2023 10:30 meeting moved",
    ])
    assert len(msgs) == 1
    assert msgs[0]["text"] == "agenda\n25/12/2023 10:30 meeting moved"


def test_message_count_and_order_do_not_depend_on_continuation_lines(tmp_path):
    single, _ = _wa(tmp_path, [
        "25/12/2023, 9:14 pm - Riya: a",
        "25/12/2023, 9:15 pm - Karan: b",
        "25/12/2023, 9:16 pm - Riya: c",
    ])
    multi, _ = _wa(tmp_path, [
        "25/12/2023, 9:14 pm - Riya: a", "more of a",
        "25/12/2023, 9:15 pm - Karan: b", "more of b", "and more",
        "25/12/2023, 9:16 pm - Riya: c",
    ])
    assert [(m["sender"], m["sent_at"]) for m in single] == [(m["sender"], m["sent_at"]) for m in multi]


def test_a_file_with_no_recognisable_lines_parses_to_nothing(tmp_path):
    msgs, _ = _wa(tmp_path, ["just some text", "and more"])
    assert msgs == []
