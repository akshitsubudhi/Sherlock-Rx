"""Planted root-cause scenarios: the ground truth the RCA engine is evaluated against.

Each scenario perturbs one causal lever for one brand in one region during one fiscal
quarter. Placement rules keep scenarios from contaminating each other's baselines, so a
miss in evaluation is a miss by the engine, not an artefact of overlapping causes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from sherlock_rx.data.calendar import Month, quarter_months
from sherlock_rx.data.catalog import BRANDS, REGIONS, STATE_WEIGHT

CAUSES: tuple[str, ...] = (
    "field_force_vacancy",
    "stockout",
    "competitor_launch",
    "price_cut",
    "market_slowdown",
    "distribution_loss",
)

# Quarters eligible for planting; each has a full prior-year quarter in the warehouse.
PLANT_QUARTERS: tuple[tuple[int, int], ...] = ((26, 1), (26, 2), (26, 3), (26, 4), (27, 1), (27, 2))
STATE_LEVEL = {"field_force_vacancy", "stockout", "competitor_launch", "distribution_loss"}


@dataclass
class Scenario:
    scenario_id: str
    cause: str
    brand: str
    region: str
    fy: int
    fq: int
    severity: float
    states: list[str] = field(default_factory=list)
    sku: str | None = None
    competitor: str | None = None

    @property
    def quarter_label(self) -> str:
        return f"Q{self.fq} FY{self.fy:02d}"

    @property
    def months(self) -> tuple[Month, Month, Month]:
        return quarter_months(self.fy, self.fq)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["quarter"] = self.quarter_label
        d["months"] = [m.key for m in self.months]
        return d


@dataclass
class Control:
    """A brand x region x quarter cell with no planted cause (false-alarm check)."""

    brand: str
    region: str
    fy: int
    fq: int

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "quarter": f"Q{self.fq} FY{self.fy:02d}"}


def plan_scenarios(
    rng: np.random.Generator, per_cause: int = 6, n_controls: int = 20,
    severity_scale: float = 1.0,
) -> tuple[list[Scenario], list[Control]]:
    regions = list(REGIONS)
    brands = [b.name for b in BRANDS]
    n_q = len(PLANT_QUARTERS)
    blocked: set[tuple[str, str, int]] = set()
    occupied: set[tuple[str, str, int]] = set()
    scenarios: list[Scenario] = []

    def block(brand: str, region: str, qi: int) -> None:
        for q in (qi - 4, qi, qi + 4):
            if 0 <= q < n_q:
                blocked.add((brand, region, q))

    # Price cuts are national and permanent (NPPA ceiling prices), so they go in the last
    # quarter only and own every region of their brand for that quarter and its prior year.
    price_brands = rng.choice(brands, size=min(2, per_cause), replace=False)
    for b in price_brands:
        region = str(rng.choice(regions))
        scenarios.append(Scenario("", "price_cut", str(b), region, *PLANT_QUARTERS[-1],
                                  severity=float(rng.uniform(0.18, 0.30)) * severity_scale))
        for r in regions:
            block(str(b), r, n_q - 1)
            occupied.add((str(b), r, n_q - 1))

    others = [c for c in CAUSES if c != "price_cut"]
    for cause in others * per_cause:
        for _ in range(500):
            b = str(rng.choice(brands))
            r = str(rng.choice(regions))
            qi = int(rng.integers(0, n_q))
            if (b, r, qi) in blocked:
                continue
            # A competitor entrant persists, so nothing may already sit in later quarters.
            if cause == "competitor_launch" and any(
                    (b, r, q) in occupied for q in range(qi + 1, n_q)):
                continue
            sev = {
                "field_force_vacancy": rng.uniform(0.55, 0.85),
                "stockout": rng.uniform(0.35, 0.65),
                "competitor_launch": rng.uniform(0.20, 0.35),
                "market_slowdown": rng.uniform(0.15, 0.28),
                "distribution_loss": rng.uniform(0.30, 0.45),
            }[cause]
            sc = Scenario("", cause, b, r, *PLANT_QUARTERS[qi],
                          severity=float(sev) * severity_scale)
            if cause in STATE_LEVEL:
                # Pick states until they cover >= 25% of the region, so the planted cause is
                # material at region level (a Goa-only issue cannot explain the West).
                order = [str(s) for s in rng.permutation(list(REGIONS[r]))]
                total = sum(STATE_WEIGHT[s] for s in order)
                for s_name in order:
                    sc.states.append(s_name)
                    if sum(STATE_WEIGHT[x] for x in sc.states) >= 0.25 * total:
                        break
            scenarios.append(sc)
            block(b, r, qi)
            occupied.add((b, r, qi))
            if cause == "competitor_launch":  # entrant keeps its share afterwards
                for q in range(qi + 1, n_q):
                    blocked.add((b, r, q))
            break

    scenarios.sort(key=lambda s: (s.fy, s.fq, s.brand, s.region))
    for i, s in enumerate(scenarios, 1):
        s.scenario_id = f"S{i:02d}"

    controls: list[Control] = []
    for _ in range(5000):
        if len(controls) >= n_controls:
            break
        b, r, qi = str(rng.choice(brands)), str(rng.choice(regions)), int(rng.integers(0, n_q))
        if (b, r, qi) in blocked:
            continue
        controls.append(Control(b, r, *PLANT_QUARTERS[qi]))
        blocked.add((b, r, qi))
    return scenarios, controls
