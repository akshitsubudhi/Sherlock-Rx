"""Semantic layer compiler: (metric, filters, group-by, months) -> parameterised SQL.

The NLU layer produces slot values, never SQL. User text only reaches the database as
bound parameters, and only after entity linking mapped it to a known dimension member,
so there is no injection surface. The generated SQL is plain ANSI and can be transpiled
to Snowflake / BigQuery / Databricks with sqlglot when the warehouse moves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

CONFIG = Path(__file__).with_name("semantic.yaml")

# Which join keys each physical table carries.
TABLE_KEYS: dict[str, set[str]] = {
    "agg_sales_tbm": {"territory_id", "brand_id", "sku_id"},
    "fact_targets": {"territory_id", "brand_id"},
    "fact_field": {"territory_id", "brand_id"},
    "fact_rx": {"territory_id", "brand_id"},
    "fact_distribution": {"territory_id", "brand_id"},
    "fact_inventory": {"territory_id", "sku_id"},
    "fact_market": {"state", "brand_id"},
}
GEO_ORDER = ("region", "state", "territory")


class SemanticError(ValueError):
    pass


@dataclass(frozen=True)
class Metric:
    name: str
    label: str
    table: str
    expr: str
    unit: str
    grain: tuple[str, ...]


@dataclass
class MetricQuery:
    metric: str
    months: list[str]
    filters: dict[str, list[str]] = field(default_factory=dict)
    group_by: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SemanticLayer:
    metrics: dict[str, Metric]
    dimensions: dict[str, dict[str, str]]

    @classmethod
    def load(cls, path: Path = CONFIG) -> SemanticLayer:
        return _load(str(path))

    def _geo_allowed(self, metric: Metric, dim: str) -> bool:
        finest = "territory" if "territory" in metric.grain else "state"
        return GEO_ORDER.index(dim) <= GEO_ORDER.index(finest)

    def compile(self, q: MetricQuery) -> tuple[str, list[Any]]:
        if q.metric not in self.metrics:
            raise SemanticError(f"Unknown metric '{q.metric}'. Known: {sorted(self.metrics)}")
        if not q.months:
            raise SemanticError("A period is required.")
        m = self.metrics[q.metric]
        keys = TABLE_KEYS[m.table]
        used = set(q.group_by) | set(q.filters)
        for d in used:
            if d not in self.dimensions:
                raise SemanticError(f"Unknown dimension '{d}'.")
            if self.dimensions[d].get("level") == "geo" and not self._geo_allowed(m, d):
                raise SemanticError(f"{m.label} is only available down to state level, not {d}.")
            if d == "sku" and "sku_id" not in keys:
                raise SemanticError(f"{m.label} is not tracked by SKU.")

        joins: list[str] = []
        col: dict[str, str] = {}
        if used & set(GEO_ORDER):
            if "territory_id" in keys:
                joins.append("JOIN dim_geo g ON g.territory_id = f.territory_id")
            else:
                joins.append("JOIN (SELECT DISTINCT state, region FROM dim_geo) g ON g.state = f.state")
            for d in GEO_ORDER:
                col[d] = f"g.{d}"
        if used & {"brand", "therapy", "sku"}:
            if "brand_id" in keys:
                joins.append("JOIN dim_brand b ON b.brand_id = f.brand_id")
            else:
                joins.append("JOIN dim_sku s0 ON s0.sku_id = f.sku_id "
                             "JOIN dim_brand b ON b.brand_id = s0.brand_id")
            col["brand"], col["therapy"] = "b.brand", "b.therapy"
            if "sku" in used:
                joins.append("JOIN dim_sku s ON s.sku_id = f.sku_id")
                col["sku"] = "s.sku"
        if used & {"quarter"}:
            joins.append("JOIN dim_month mo ON mo.month = f.month")
            col["quarter"] = "mo.quarter"
        col["month"] = "f.month"

        params: list[Any] = list(q.months)
        where = [f"f.month IN ({', '.join('?' * len(q.months))})"]
        for d, values in sorted(q.filters.items()):
            if not values:
                continue
            where.append(f"{col[d]} IN ({', '.join('?' * len(values))})")
            params.extend(values)
        select = [f"{col[d]} AS {d}" for d in q.group_by]
        sql = (f"SELECT {', '.join([*select, f'{m.expr} AS value'])} FROM {m.table} f "
               + " ".join(joins) + " WHERE " + " AND ".join(where))
        if q.group_by:
            sql += f" GROUP BY {', '.join(col[d] for d in q.group_by)}"
            sql += f" ORDER BY {', '.join(col[d] for d in q.group_by)}"
        return sql, params


@lru_cache(maxsize=4)
def _load(path: str) -> SemanticLayer:
    raw = yaml.safe_load(Path(path).read_text())
    metrics = {
        name: Metric(name, spec["label"], spec["table"], spec["expr"], spec["unit"],
                     tuple(spec["grain"]))
        for name, spec in raw["metrics"].items()
    }
    return SemanticLayer(metrics, raw["dimensions"])


def transpile(sql: str, dialect: str) -> str:
    """Render compiled SQL for another warehouse (requires the optional `sqlglot`)."""
    import sqlglot

    return str(sqlglot.transpile(sql, read="duckdb", write=dialect)[0])
