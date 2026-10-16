from __future__ import annotations

import json
from pathlib import Path

import pytest

from sherlock_rx.analytics.period import Period
from sherlock_rx.analytics.rca import RCAEngine, RCAError
from sherlock_rx.data.calendar import quarter_months
from sherlock_rx.eval.rca_eval import evaluate


@pytest.fixture(scope="module")
def engine(warehouse):  # type: ignore[no-untyped-def]
    return RCAEngine(warehouse)


@pytest.fixture(scope="module")
def gt(warehouse_dir: Path) -> dict:  # type: ignore[type-arg]
    return json.loads((warehouse_dir / "_ground_truth.json").read_text())  # type: ignore[no-any-return]


def _q(fy: int, fq: int) -> Period:
    return Period(quarter_months(fy, fq), f"Q{fq} FY{fy}")


def test_gate_rca_accuracy(warehouse_dir: Path) -> None:
    """Acceptance gate: top-1 cause >= 90%, location >= 90%, false alarms <= 10%."""
    r = evaluate(warehouse_dir)
    assert r.cause_at_1 >= 0.90, r.misses
    assert r.location_acc >= 0.90
    assert r.false_alarm_rate <= 0.10


def test_bridge_adds_up_exactly(engine: RCAEngine, gt: dict) -> None:  # type: ignore[type-arg]
    s = gt["scenarios"][0]
    rep = engine.diagnose(s["brand"], s["region"], _q(s["fy"], s["fq"]))
    steps = [b["value"] for b in rep.bridge]
    assert steps[0] + sum(steps[1:-1]) == pytest.approx(steps[-1], rel=1e-9)


def test_every_cause_carries_evidence(engine: RCAEngine, gt: dict) -> None:  # type: ignore[type-arg]
    for s in gt["scenarios"][:10]:
        rep = engine.diagnose(s["brand"], s["region"], _q(s["fy"], s["fq"]))
        for c in rep.causes:
            assert c.evidence, c.cause
            assert 0 <= c.confidence <= 1
        assert rep.query_ids  # answers cite the queries they ran


def test_stockout_signature_prescriptions_hold(engine: RCAEngine, gt: dict) -> None:  # type: ignore[type-arg]
    s = next(x for x in gt["scenarios"] if x["cause"] == "stockout")
    rep = engine.diagnose(s["brand"], s["region"], _q(s["fy"], s["fq"]))
    top = rep.top_cause
    assert top is not None and top.cause == "stockout"
    assert top.confidence_label in ("High", "Medium")


def test_price_cut_is_recognised_as_national(engine: RCAEngine, gt: dict) -> None:  # type: ignore[type-arg]
    s = next(x for x in gt["scenarios"] if x["cause"] == "price_cut")
    rep = engine.diagnose(s["brand"], s["region"], _q(s["fy"], s["fq"]))
    assert rep.top_cause is not None and rep.top_cause.scope.startswith("national")


def test_state_scope(engine: RCAEngine, gt: dict) -> None:  # type: ignore[type-arg]
    s = next(x for x in gt["scenarios"] if x["states"])
    rep = engine.diagnose(s["brand"], s["region"], _q(s["fy"], s["fq"]), state=s["states"][0])
    assert rep.state == s["states"][0] and rep.headline.last_year > 0


def test_errors_are_explicit(engine: RCAEngine) -> None:
    with pytest.raises(RCAError, match="Unknown brand"):
        engine.diagnose("Nope", "North", _q(27, 2))
    with pytest.raises(RCAError, match="prior-year"):
        engine.diagnose("Cardivex", "North", _q(24, 1))
