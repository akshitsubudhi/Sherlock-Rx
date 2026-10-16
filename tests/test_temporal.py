from __future__ import annotations

import pytest

from sherlock_rx.data.calendar import Month
from sherlock_rx.nlp.temporal import TemporalParser
from sherlock_rx.nlp.text import normalize

P = TemporalParser(ref=Month(2026, 9))

CASES = [
    ("why is cardivex down in Q2 FY27", "Q2 FY27", "2026-07", "2026-09"),
    ("q3", "Q3 FY26", "2025-10", "2025-12"),  # Q3 FY27 is not complete yet
    ("sales in FY 2025-26", "FY26", "2025-04", "2026-03"),
    ("last quarter", "Q2 FY27", "2026-07", "2026-09"),
    ("previous quarter", "Q1 FY27", "2026-04", "2026-06"),
    ("last 3 months", "Last 3 months (Jul-Sep 2026)", "2026-07", "2026-09"),
    ("past two quarters", "Last 2 quarters (Apr-Sep 2026)", "2026-04", "2026-09"),
    ("in August", "Aug 2026", "2026-08", "2026-08"),
    ("aug'25", "Aug 2025", "2025-08", "2025-08"),
    ("jul-sep 2026", "Jul-Sep 2026", "2026-07", "2026-09"),
    ("from april to june", "Apr-Jun 2026", "2026-04", "2026-06"),
    ("since june", "Since Jun 2026", "2026-06", "2026-09"),
    ("ytd", "FY27 (YTD)", "2026-04", "2026-09"),
    ("H1 FY27", "H1 FY27", "2026-04", "2026-09"),
    ("second quarter of fy26", "Q2 FY26", "2025-07", "2025-09"),
    ("fy27 q1", "Q1 FY27", "2026-04", "2026-06"),
    ("last fy", "FY26", "2025-04", "2026-03"),
]


@pytest.mark.parametrize("text,label,first,last", CASES)
def test_resolves(text: str, label: str, first: str, last: str) -> None:
    t = P.parse(normalize(text))
    assert t is not None
    assert (t.period.label, t.period.keys[0], t.period.keys[-1]) == (label, first, last)


def test_comparison_baseline_is_not_the_period() -> None:
    t = P.parse(normalize("sales in q1 vs last year"))
    assert t is not None and t.period.label == "Q1 FY27"
    assert P.parse(normalize("growth vs last quarter")) is None


def test_modal_may_is_not_a_month() -> None:
    assert P.parse("it may be low") is None
    t = P.parse("sales in may")
    assert t is not None and t.period.label == "May 2026"


def test_period_prior_year_alignment() -> None:
    t = P.parse("q2 fy27")
    assert t is not None and t.period.py_keys == ["2025-07", "2025-08", "2025-09"]
