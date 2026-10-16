from __future__ import annotations

from dataclasses import dataclass

from sherlock_rx.data.calendar import ALL_MONTHS, Month


@dataclass(frozen=True)
class Period:
    """A set of consecutive months plus a human label ("Q2 FY27", "Aug 2026")."""

    months: tuple[Month, ...]
    label: str

    @property
    def keys(self) -> list[str]:
        return [m.key for m in self.months]

    @property
    def py_keys(self) -> list[str]:
        return [m.shift(-12).key for m in self.months]

    @property
    def has_prior_year(self) -> bool:
        return self.months[0].shift(-12) >= ALL_MONTHS[0]

    @classmethod
    def from_keys(cls, keys: list[str], label: str) -> Period:
        return cls(tuple(Month(int(k[:4]), int(k[5:7])) for k in keys), label)

    def to_dict(self) -> dict[str, object]:
        return {"label": self.label, "months": self.keys}
