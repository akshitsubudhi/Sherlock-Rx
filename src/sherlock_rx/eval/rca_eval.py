"""Evaluate the RCA engine against the planted ground truth.

For each planted scenario we ask the engine to diagnose (brand, region, quarter) and check:
  * cause@1  - the top material cause is the planted cause
  * cause@2  - the planted cause is in the top two
  * location - for state-level causes, the cause's top location is a planted state
  * flagged  - the cell is reported as under-performing / below plan
For each control cell (nothing planted) we check the engine does not invent a material cause
that explains a large share of the gap (false alarm).
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sherlock_rx.analytics.period import Period
from sherlock_rx.analytics.rca import RCAEngine
from sherlock_rx.data.calendar import quarter_months
from sherlock_rx.data.scenarios import STATE_LEVEL
from sherlock_rx.warehouse.engine import Warehouse


@dataclass
class RCAEvalResult:
    n_scenarios: int
    cause_at_1: float
    cause_at_2: float
    location_acc: float
    flagged_rate: float
    false_alarm_rate: float
    per_cause: dict[str, dict[str, float]]
    confusion: dict[str, dict[str, int]]
    median_latency_ms: float
    misses: list[dict[str, Any]] = field(default_factory=list)


def _period(fy: int, fq: int) -> Period:
    return Period(quarter_months(fy, fq), f"Q{fq} FY{fy:02d}")


def evaluate(data_dir: str | Path) -> RCAEvalResult:
    data_dir = Path(data_dir)
    gt = json.loads((data_dir / "_ground_truth.json").read_text())
    engine = RCAEngine(Warehouse(data_dir))
    hits1 = hits2 = loc_ok = loc_n = flagged = 0
    per: dict[str, Counter[str]] = defaultdict(Counter)
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    latencies: list[float] = []
    misses: list[dict[str, Any]] = []
    for s in gt["scenarios"]:
        rep = engine.diagnose(s["brand"], s["region"], _period(s["fy"], s["fq"]))
        latencies.append(rep.elapsed_ms)
        material = [c for c in rep.causes if c.material]
        pred = material[0].cause if material else "none"
        top2 = [c.cause for c in material[:2]]
        ok1, ok2 = pred == s["cause"], s["cause"] in top2
        hits1 += ok1
        hits2 += ok2
        flagged += rep.headline.status != "On track"
        per[s["cause"]]["n"] += 1
        per[s["cause"]]["hit"] += ok1
        confusion[s["cause"]][pred] += 1
        if s["cause"] in STATE_LEVEL and ok1:
            loc_n += 1
            loc_ok += bool(material[0].locations) and material[0].locations[0] in s["states"]
        if not ok1:
            misses.append({"scenario": s["scenario_id"], "truth": s["cause"], "pred": pred,
                           "brand": s["brand"], "region": s["region"], "quarter": s["quarter"],
                           "severity": round(s["severity"], 2),
                           "causes": [(c.cause, round(c.pct_of_py, 3)) for c in rep.causes[:3]]})
    false_alarms = 0
    for c in gt["controls"]:
        rep = engine.diagnose(c["brand"], c["region"], _period(c["fy"], c["fq"]))
        top = rep.top_cause
        false_alarms += bool(top and rep.headline.status == "Underperforming"
                             and top.pct_of_py <= -0.03)
    n = len(gt["scenarios"])
    latencies.sort()
    return RCAEvalResult(
        n_scenarios=n,
        cause_at_1=hits1 / n,
        cause_at_2=hits2 / n,
        location_acc=loc_ok / loc_n if loc_n else 1.0,
        flagged_rate=flagged / n,
        false_alarm_rate=false_alarms / max(len(gt["controls"]), 1),
        per_cause={k: {"n": v["n"], "acc": v["hit"] / v["n"]} for k, v in sorted(per.items())},
        confusion={k: dict(v) for k, v in sorted(confusion.items())},
        median_latency_ms=latencies[len(latencies) // 2],
        misses=misses,
    )
