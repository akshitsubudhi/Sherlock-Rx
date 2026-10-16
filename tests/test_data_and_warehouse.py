from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sherlock_rx.data.generator import generate
from sherlock_rx.warehouse.semantic import MetricQuery, SemanticError, SemanticLayer, transpile


def _hash(d: Path) -> str:
    h = hashlib.sha1()
    for f in sorted(d.rglob("*.parquet")):
        h.update(f.read_bytes())
    return h.hexdigest()


def test_generation_is_deterministic(tmp_path: Path) -> None:
    generate(tmp_path / "a", scale=0.5, seed=3)
    generate(tmp_path / "b", scale=0.5, seed=3)
    assert _hash(tmp_path / "a") == _hash(tmp_path / "b")


def test_ground_truth_is_well_formed(warehouse_dir: Path) -> None:
    gt = json.loads((warehouse_dir / "_ground_truth.json").read_text())
    assert len(gt["scenarios"]) >= 30 and len(gt["controls"]) == 20
    cells = {(s["brand"], s["region"], s["quarter"]) for s in gt["scenarios"]}
    assert len(cells) == len(gt["scenarios"])  # no two scenarios share a cell
    assert not cells & {(c["brand"], c["region"], c["quarter"]) for c in gt["controls"]}


def test_planted_stockout_hits_sales_not_prescriptions(warehouse, warehouse_dir: Path) -> None:  # type: ignore[no-untyped-def]
    gt = json.loads((warehouse_dir / "_ground_truth.json").read_text())
    s = next(x for x in gt["scenarios"] if x["cause"] == "stockout")
    ph = ", ".join("?" * len(s["states"]))
    df = warehouse.query(f"""
        SELECT (SELECT avg(i.stockout_days) FROM fact_inventory i JOIN dim_geo g USING (territory_id)
                JOIN dim_sku k USING (sku_id) JOIN dim_brand b USING (brand_id)
                WHERE b.brand = ? AND g.state IN ({ph}) AND i.month = ?) AS so""",
                         [s["brand"], *s["states"], s["months"][1]])
    assert df.so.iat[0] > 5  # the mechanism, not just the label, is in the data


def test_semantic_layer_compiles_and_binds_parameters(warehouse) -> None:  # type: ignore[no-untyped-def]
    sl = SemanticLayer.load()
    sql, params = sl.compile(MetricQuery("net_sales", ["2026-07"], {"brand": ["Cardivex"]},
                                         ["region"]))
    assert "?" in sql and "Cardivex" not in sql and params == ["2026-07", "Cardivex"]
    assert len(warehouse.query(sql, params)) == 5


def test_semantic_layer_injection_is_inert(warehouse) -> None:  # type: ignore[no-untyped-def]
    sl = SemanticLayer.load()
    sql, params = sl.compile(MetricQuery("net_sales", ["2026-07"],
                                         {"brand": ["x'; DROP TABLE dim_geo; --"]}))
    warehouse.query(sql, params)
    assert len(warehouse.query("SELECT * FROM dim_geo")) > 0


def test_semantic_layer_rejects_grain_violations() -> None:
    sl = SemanticLayer.load()
    with pytest.raises(SemanticError, match="state level"):
        sl.compile(MetricQuery("market_share", ["2026-07"], {}, ["territory"]))
    with pytest.raises(SemanticError, match="Unknown metric"):
        sl.compile(MetricQuery("ebitda", ["2026-07"]))


def test_transpiles_to_snowflake() -> None:
    sql, _ = SemanticLayer.load().compile(MetricQuery("calls", ["2026-07"], {}, ["state"]))
    assert "SELECT" in transpile(sql, "snowflake")


def test_query_cache_and_log(warehouse) -> None:  # type: ignore[no-untyped-def]
    n = len(warehouse.log.records)
    warehouse.query("SELECT count(*) FROM dim_brand")
    warehouse.query("SELECT count(*) FROM dim_brand")
    recs = warehouse.log.records[n:]
    assert [r.cached for r in recs] == [False, True]
