"""Rule-based temporal expression recognition and normalisation (a mini TimeML/SUTime).

Recognises and resolves, relative to the latest month in the warehouse:

    fiscal quarters   "Q2", "q2 fy27", "Q2 FY 2026-27", "second quarter", "Q3 of FY26"
    fiscal years      "FY26", "FY 2025-26", "last financial year", "this fiscal", "YTD"
    halves            "H1 FY27", "first half"
    months            "Aug", "august 2026", "aug'26", "in June"
    month ranges      "Jul-Sep 2026", "from april to june", "since June"
    relative          "last month", "last quarter", "previous quarter", "last 3 months",
                      "past two quarters", "last year" (= trailing 12 months)
    calendar years    "2025", "in 2024"

Indian fiscal convention: FY27 = Apr 2026 - Mar 2027; Q1 = Apr-Jun.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from sherlock_rx.analytics.period import Period
from sherlock_rx.data.calendar import ALL_MONTHS, LATEST, MONTH_NAMES, Month, quarter_months

MONTHS_FULL = ("january", "february", "march", "april", "may", "june", "july", "august",
               "september", "october", "november", "december")
MONTH_LOOKUP: dict[str, int] = {}
for _i, (_full, _abbr) in enumerate(zip(MONTHS_FULL, MONTH_NAMES, strict=True), 1):
    MONTH_LOOKUP[_full] = _i
    MONTH_LOOKUP[_abbr.lower()] = _i
MONTH_LOOKUP["sept"] = 9
MON = r"(?:" + "|".join(sorted(MONTH_LOOKUP, key=len, reverse=True)) + r")"
NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "nine": 9,
             "twelve": 12, "a": 1}
ORDINALS = {"first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3, "fourth": 4,
            "4th": 4, "last": 4}
FY = r"(?:fy|f\.y\.|financial year|fiscal year|fiscal)\s*'?\s*(\d{2,4}(?:\s*-\s*\d{2,4})?)"
RELATIVE_RULES = {"trailing_year", "last_quarter", "prev_quarter", "last_fy", "last_month"}
COMPARATIVE = re.compile(r"\b(?:vs|versus|than|compared (?:to|with)|against|over|from)\s*$")


class TemporalError(ValueError):
    pass


@dataclass(frozen=True)
class TimeExpr:
    period: Period
    text: str
    start: int
    end: int
    rule: str


def _fy_num(raw: str) -> int:
    """'27' -> 27, '2027' -> 27, '2026-27' -> 27, '2025-2026' -> 26."""
    raw = raw.replace(" ", "")
    if "-" in raw:
        raw = raw.split("-")[1]
    return int(raw) % 100


def _current_fy(ref: Month) -> int:
    return ref.fy


def _latest_quarter_le(fq: int, ref: Month) -> int:
    """FY of the most recent *completed* fiscal quarter fq at or before ref."""
    fy = ref.fy
    if quarter_months(fy, fq)[-1] > ref:
        fy -= 1
    return fy


def _latest_month_le(month: int, ref: Month) -> Month:
    year = ref.year if month <= ref.month else ref.year - 1
    return Month(year, month)


def _span(first: Month, last: Month) -> tuple[Month, ...]:
    out, m = [], first
    while m <= last:
        out.append(m)
        m = m.shift(1)
    return tuple(out)


def _range_label(ms: tuple[Month, ...]) -> str:
    a, b = ms[0], ms[-1]
    if a == b:
        return a.label
    if a.year == b.year:
        return f"{MONTH_NAMES[a.month - 1]}-{MONTH_NAMES[b.month - 1]} {a.year}"
    return f"{a.label} - {b.label}"


def _quarter(fy: int, fq: int) -> Period:
    return Period(quarter_months(fy, fq), f"Q{fq} FY{fy:02d}")


def _last_complete_quarter(ref: Month) -> tuple[int, int]:
    fy, fq = ref.fy, ref.fq
    if quarter_months(fy, fq)[-1] > ref:
        fq -= 1
        if fq == 0:
            fy, fq = fy - 1, 4
    return fy, fq


def _shift_quarter(fy: int, fq: int, k: int) -> tuple[int, int]:
    idx = fy * 4 + (fq - 1) + k
    return idx // 4, idx % 4 + 1


class TemporalParser:
    def __init__(self, ref: Month = LATEST):
        self.ref = ref
        r = self.ref
        self.rules: list[tuple[str, re.Pattern[str], object]] = [
            ("quarter_fy", re.compile(rf"\bq\s?([1-4])\s*(?:of\s+|,\s*)?{FY}"), self._q_fy),
            ("fy_quarter", re.compile(rf"{FY}\s*q\s?([1-4])\b"), self._fy_q),
            ("ordinal_quarter_fy", re.compile(
                rf"\b(first|second|third|fourth|1st|2nd|3rd|4th)\s+quarter\s*(?:of\s+)?{FY}"),
             self._ordq_fy),
            ("half_fy", re.compile(rf"\bh([12])\s*(?:of\s+)?{FY}"), self._h_fy),
            ("quarter", re.compile(r"\bq\s?([1-4])\b"), self._q),
            ("ordinal_quarter", re.compile(
                r"\b(first|second|third|fourth|1st|2nd|3rd|4th)\s+quarter\b"), self._ordq),
            ("half", re.compile(r"\b(?:h([12])|(first|second) half)\b"), self._h),
            ("rel_n", re.compile(
                r"\b(?:last|past|previous|trailing|recent)\s+(\d+|one|two|three|four|five|six|"
                r"nine|twelve)\s+(months?|quarters?)\b"), self._rel_n),
            ("prev_quarter", re.compile(
                r"\b(?:previous|prior|preceding)\s+quarter\b|\bquarter before\b"),
             lambda m: _quarter(*_shift_quarter(*_last_complete_quarter(r), -1))),
            ("last_quarter", re.compile(
                r"\b(?:last|this|current|latest|recent|most recent)\s+(?:quarter|qtr)\b"),
             lambda m: _quarter(*_last_complete_quarter(r))),
            ("last_month", re.compile(
                r"\b(?:last|this|current|latest|previous|recent)\s+month\b"),
             lambda m: Period((r,), r.label)),
            ("ytd", re.compile(r"\b(?:ytd|year to date|this fiscal|this financial year|"
                               r"current fiscal|this fy|current fy)\b"), self._ytd),
            ("last_fy", re.compile(r"\b(?:last|previous|prior)\s+(?:fiscal|financial year|fy)\b"),
             lambda m: self._fy_period(_current_fy(r) - 1)),
            ("fy", re.compile(FY), lambda m: self._fy_period(_fy_num(m.group(1)))),
            ("month_range", re.compile(
                rf"\b(?:from\s+|between\s+)?({MON})\s*(?:-|to|till|until|through|and)\s*({MON})"
                rf"(?:\s*'?\s*(\d{{2,4}}))?\b"), self._month_range),
            ("since_month", re.compile(rf"\bsince\s+({MON})(?:\s*'?\s*(\d{{2,4}}))?\b"),
             self._since),
            ("month_year", re.compile(rf"\b({MON})\s*'?\s*-?\s*(\d{{4}}|\d{{2}})\b"),
             self._month_year),
            ("month", re.compile(rf"\b({MON})\b"), self._month),
            ("trailing_year", re.compile(r"\b(?:last|past)\s+(?:12 months|year)\b"),
             lambda m: self._last_n(12)),
            ("calendar_year", re.compile(r"\b(?:in\s+|for\s+|during\s+)?(20[2-3]\d)\b"),
             self._cal_year),
        ]

    # ------------------------------------------------------------------ handlers
    def _q_fy(self, m: re.Match[str]) -> Period:
        return _quarter(_fy_num(m.group(2)), int(m.group(1)))

    def _fy_q(self, m: re.Match[str]) -> Period:
        return _quarter(_fy_num(m.group(1)), int(m.group(2)))

    def _ordq_fy(self, m: re.Match[str]) -> Period:
        return _quarter(_fy_num(m.group(2)), ORDINALS[m.group(1)])

    def _q(self, m: re.Match[str]) -> Period:
        fq = int(m.group(1))
        return _quarter(_latest_quarter_le(fq, self.ref), fq)

    def _ordq(self, m: re.Match[str]) -> Period:
        fq = ORDINALS[m.group(1)]
        return _quarter(_latest_quarter_le(fq, self.ref), fq)

    def _half(self, fy: int, h: int) -> Period:
        q1 = quarter_months(fy, 2 * h - 1)
        q2 = quarter_months(fy, 2 * h)
        return Period((*q1, *q2), f"H{h} FY{fy:02d}")

    def _h_fy(self, m: re.Match[str]) -> Period:
        return self._half(_fy_num(m.group(2)), int(m.group(1)))

    def _h(self, m: re.Match[str]) -> Period:
        h = int(m.group(1)) if m.group(1) else (1 if m.group(2) == "first" else 2)
        fy = self.ref.fy
        if quarter_months(fy, 2 * h)[-1] > self.ref:
            fy -= 1
        return self._half(fy, h)

    def _last_n(self, n: int) -> Period:
        ms = _span(self.ref.shift(-(n - 1)), self.ref)
        return Period(ms, f"Last {n} months ({_range_label(ms)})")

    def _rel_n(self, m: re.Match[str]) -> Period:
        raw = m.group(1)
        n = int(raw) if raw.isdigit() else NUM_WORDS[raw]
        if m.group(2).startswith("quarter"):
            fy, fq = _last_complete_quarter(self.ref)
            first = quarter_months(*_shift_quarter(fy, fq, -(n - 1)))[0]
            last = quarter_months(fy, fq)[-1]
            ms = _span(first, last)
            return Period(ms, f"Last {n} quarters ({_range_label(ms)})")
        return self._last_n(n)

    def _fy_period(self, fy: int) -> Period:
        first = Month(2000 + fy - 1, 4)
        last = min(first.shift(11), self.ref)
        label = f"FY{fy:02d}" + (" (YTD)" if last < first.shift(11) else "")
        return Period(_span(first, last), label)

    def _ytd(self, m: re.Match[str]) -> Period:
        return self._fy_period(self.ref.fy)

    def _year(self, raw: str | None, month: int) -> int:
        if raw is None:
            return _latest_month_le(month, self.ref).year
        y = int(raw)
        return y + 2000 if y < 100 else y

    def _month_range(self, m: re.Match[str]) -> Period:
        a, b = MONTH_LOOKUP[m.group(1)], MONTH_LOOKUP[m.group(2)]
        end_year = self._year(m.group(3), b)
        last = Month(end_year, b)
        first = Month(end_year if a <= b else end_year - 1, a)
        ms = _span(first, last)
        return Period(ms, _range_label(ms))

    def _since(self, m: re.Match[str]) -> Period:
        first = Month(self._year(m.group(2), MONTH_LOOKUP[m.group(1)]), MONTH_LOOKUP[m.group(1)])
        ms = _span(first, self.ref)
        return Period(ms, f"Since {first.label}")

    def _month_year(self, m: re.Match[str]) -> Period:
        mon = Month(self._year(m.group(2), MONTH_LOOKUP[m.group(1)]), MONTH_LOOKUP[m.group(1)])
        return Period((mon,), mon.label)

    def _month(self, m: re.Match[str]) -> Period:
        mon = _latest_month_le(MONTH_LOOKUP[m.group(1)], self.ref)
        return Period((mon,), mon.label)

    def _cal_year(self, m: re.Match[str]) -> Period:
        y = int(m.group(1))
        ms = _span(max(Month(y, 1), ALL_MONTHS[0]), min(Month(y, 12), self.ref))
        if not ms:
            raise TemporalError(f"No data for {y}.")
        return Period(ms, str(y) if len(ms) == 12 else f"{y} ({_range_label(ms)})")

    # ------------------------------------------------------------------ api
    def find_all(self, text: str) -> list[TimeExpr]:
        """Leftmost-longest matching across rules; text should be normalised (lower-case)."""
        found: list[TimeExpr] = []
        taken: list[tuple[int, int]] = []
        for name, pattern, handler in self.rules:
            for m in pattern.finditer(text):
                s, e = m.span()
                if any(s < te and ts < e for ts, te in taken):
                    continue
                if name == "month" and m.group(1) in ("may", "mar") and not self._looks_temporal(
                        text, s):
                    continue  # "may" is usually a modal verb
                if name in RELATIVE_RULES and COMPARATIVE.search(text[max(0, s - 16):s]):
                    continue  # "vs last year" is a comparison baseline, not the period asked about
                try:
                    period = handler(m)  # type: ignore[operator]
                except (TemporalError, KeyError, ValueError):
                    continue
                found.append(TimeExpr(period, m.group().strip(), s, e, name))
                taken.append((s, e))
        return sorted(found, key=lambda t: t.start)

    @staticmethod
    def _looks_temporal(text: str, start: int) -> bool:
        before = text[max(0, start - 8):start]
        return bool(re.search(r"\b(in|for|of|during|since|from)\s*$", before))

    def parse(self, text: str) -> TimeExpr | None:
        found = self.find_all(text)
        return found[0] if found else None

    def validate(self, period: Period) -> Period:
        months = [m for m in period.months if ALL_MONTHS[0] <= m <= self.ref]
        if not months:
            raise TemporalError(
                f"No data for {period.label}. Data covers {ALL_MONTHS[0].label} - {self.ref.label}.")
        if len(months) < len(period.months):
            return Period(tuple(months), f"{period.label} (available months)")
        return period
