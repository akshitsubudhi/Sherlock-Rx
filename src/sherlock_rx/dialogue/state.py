"""Dialogue state tracking (DST): the frame of slots that persists across turns."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sherlock_rx.analytics.period import Period
from sherlock_rx.analytics.rca import RCAReport

TASK_SLOTS = ("brands", "regions", "states", "territories", "skus", "competitors", "therapies",
              "metrics", "period", "dimension", "baseline")


@dataclass
class DialogueState:
    brands: list[str] = field(default_factory=list)
    regions: list[str] = field(default_factory=list)
    states: list[str] = field(default_factory=list)
    territories: list[str] = field(default_factory=list)
    skus: list[str] = field(default_factory=list)
    competitors: list[str] = field(default_factory=list)
    therapies: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    period: Period | None = None
    dimension: str | None = None
    baseline: str | None = None
    last_intent: str | None = None
    pending: dict[str, Any] | None = None  # {"intent": ..., "missing": "brand"}
    last_report: RCAReport | None = None
    last_scan: list[dict[str, Any]] = field(default_factory=list)
    turns: int = 0

    def update(self, slots: dict[str, Any]) -> list[str]:
        """Merge newly extracted slots; returns the names of slots that changed.

        Geography is hierarchical: a new region clears states/territories that the user
        did not repeat; a new state clears territories. Everything else carries over.
        """
        changed = []
        if "regions" in slots and "states" not in slots:
            self.states, self.territories = [], []
        if "states" in slots and "territories" not in slots:
            self.territories = []
        if "brands" in slots and "skus" not in slots:
            self.skus = []
        for k in TASK_SLOTS:
            if k in slots and slots[k] not in (None, []):
                if getattr(self, k) != slots[k]:
                    changed.append(k)
                setattr(self, k, slots[k])
        return changed

    def snapshot(self) -> dict[str, Any]:
        return {
            "brands": self.brands, "regions": self.regions, "states": self.states,
            "territories": self.territories, "skus": self.skus, "metrics": self.metrics,
            "period": self.period.label if self.period else None, "dimension": self.dimension,
            "baseline": self.baseline, "last_intent": self.last_intent,
            "pending": self.pending, "turns": self.turns,
            "has_report": self.last_report is not None,
        }
