"""Indian fiscal calendar helpers (FY runs April -> March; FY27 = Apr 2026 - Mar 2027)."""

from __future__ import annotations

from dataclasses import dataclass

START_YEAR, START_MONTH = 2023, 4  # first month in the warehouse
N_MONTHS = 42  # Apr 2023 .. Sep 2026

MONTH_NAMES = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


@dataclass(frozen=True, order=True)
class Month:
    year: int
    month: int

    @property
    def key(self) -> str:
        return f"{self.year:04d}-{self.month:02d}"

    @property
    def label(self) -> str:
        return f"{MONTH_NAMES[self.month - 1]} {self.year}"

    @property
    def fy(self) -> int:
        """Fiscal year number, e.g. 27 for Apr 2026 - Mar 2027."""
        return (self.year + 1) % 100 if self.month >= 4 else self.year % 100

    @property
    def fq(self) -> int:
        return ((self.month - 4) % 12) // 3 + 1

    @property
    def quarter_label(self) -> str:
        return f"Q{self.fq} FY{self.fy:02d}"

    def shift(self, months: int) -> Month:
        idx = self.year * 12 + (self.month - 1) + months
        return Month(idx // 12, idx % 12 + 1)


def month_at(index: int) -> Month:
    return Month(START_YEAR, START_MONTH).shift(index)


def month_index(m: Month) -> int:
    return (m.year - START_YEAR) * 12 + (m.month - START_MONTH)


ALL_MONTHS: tuple[Month, ...] = tuple(month_at(i) for i in range(N_MONTHS))
LATEST: Month = ALL_MONTHS[-1]


def quarter_months(fy: int, fq: int) -> tuple[Month, Month, Month]:
    """Months of fiscal quarter `fq` of fiscal year `fy` (two-digit)."""
    start_cal_year = 2000 + fy - 1
    first = Month(start_cal_year, 4).shift(3 * (fq - 1))
    return (first, first.shift(1), first.shift(2))


def fiscal_year_months(fy: int) -> tuple[Month, ...]:
    first = Month(2000 + fy - 1, 4)
    return tuple(first.shift(i) for i in range(12))
