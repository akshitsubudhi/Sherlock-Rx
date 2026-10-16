"""Analytics services behind the non-RCA intents. All SQL goes through the semantic layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from sherlock_rx.analytics.period import Period
from sherlock_rx.analytics.rca import _weighted_median
from sherlock_rx.data.calendar import ALL_MONTHS, LATEST
from sherlock_rx.nlp.retrieval import BM25, EXPANSIONS, Doc, Hit
from sherlock_rx.warehouse.engine import Warehouse
from sherlock_rx.warehouse.semantic import MetricQuery, SemanticLayer

BASE_METRIC = {"achievement": "net_sales", "growth": "net_sales"}


@dataclass
class Scope:
    brands: list[str]
    regions: list[str]
    states: list[str]
    territories: list[str]
    skus: list[str]

    def filters(self) -> dict[str, list[str]]:
        f: dict[str, list[str]] = {}
        if self.brands:
            f["brand"] = self.brands
        if self.territories:
            f["territory"] = self.territories
        elif self.states:
            f["state"] = self.states
        elif self.regions:
            f["region"] = self.regions
        if self.skus:
            f["sku"] = self.skus
        return f

    @property
    def geo_label(self) -> str:
        return (", ".join(self.territories) or ", ".join(self.states)
                or ", ".join(self.regions) or "India")


class Analytics:
    def __init__(self, wh: Warehouse, semantic: SemanticLayer | None = None):
        self.wh = wh
        self.sl = semantic or SemanticLayer.load()
        self._notes: BM25 | None = None

    # ------------------------------------------------------------------ primitives
    def _value(self, metric: str, scope: Scope, months: list[str],
               group_by: list[str] | None = None) -> pd.DataFrame:
        filters = scope.filters()
        m = self.sl.metrics[metric]
        if m.table == "fact_market" and "territory" in filters:
            filters.pop("territory")
        sql, params = self.sl.compile(MetricQuery(metric, months, filters, group_by or []))
        return self.wh.query(sql, params)

    def scalar(self, metric: str, scope: Scope, months: list[str]) -> float:
        df = self._value(metric, scope, months)
        v = df["value"].iat[0] if len(df) else None
        return float(v) if v is not None and not pd.isna(v) else float("nan")

    # ------------------------------------------------------------------ intents
    def metric_summary(self, metric: str, scope: Scope, period: Period) -> dict[str, Any]:
        base = BASE_METRIC.get(metric, metric)
        cur = self.scalar(base, scope, period.keys)
        py = self.scalar(base, scope, period.py_keys) if period.has_prior_year else float("nan")
        out: dict[str, Any] = {"metric": metric, "base_metric": base,
                               "label": self.sl.metrics[base].label,
                               "unit": self.sl.metrics[base].unit, "value": cur,
                               "last_year": py, "yoy": cur / py - 1 if py and py == py else None}
        if base == "net_sales":
            tgt = self.scalar("target", scope, period.keys)
            out["target"] = tgt
            out["achievement"] = cur / tgt if tgt else None
        return out

    def trend(self, metric: str, scope: Scope, period: Period | None = None) -> pd.DataFrame:
        base = BASE_METRIC.get(metric, metric)
        # A trend needs history: at least 12 months, ending where the asked period ends.
        end = period.months[-1] if period else LATEST
        if period and len(period.months) >= 12:
            months = period.keys
        else:
            months = [m.key for m in ALL_MONTHS if end.shift(-12) < m <= end]
        cur = self._value(base, scope, months, ["month"])
        py_months = [f"{int(k[:4]) - 1:04d}{k[4:]}" for k in months]
        py = self._value(base, scope, py_months, ["month"])
        py["month"] = py["month"].map(lambda k: f"{int(k[:4]) + 1:04d}{k[4:]}")  # align to cur
        df = cur.merge(py.rename(columns={"value": "last_year"}), on="month", how="left")
        df["yoy"] = df["value"] / df["last_year"] - 1
        return df

    def compare(self, metric: str, scope: Scope, period: Period, dimension: str = "region",
                members: list[str] | None = None) -> pd.DataFrame:
        base = BASE_METRIC.get(metric, metric)
        wide = Scope(scope.brands, scope.regions if dimension != "region" else [],
                     scope.states if dimension == "territory" else [], [], scope.skus)
        cur = self._value(base, wide, period.keys, [dimension])
        py = self._value(base, wide, period.py_keys, [dimension])
        df = cur.merge(py.rename(columns={"value": "last_year"}), on=dimension, how="left")
        df["yoy"] = df["value"] / df["last_year"] - 1
        if base == "net_sales":
            tgt = self._value("target", wide, period.keys, [dimension])
            df = df.merge(tgt.rename(columns={"value": "target"}), on=dimension, how="left")
            df["achievement"] = df["value"] / df["target"]
        if members:
            df["highlight"] = df[dimension].isin(members)
        return df.sort_values("yoy").reset_index(drop=True)

    def scan(self, period: Period, regions: list[str] | None = None,
             brands: list[str] | None = None, therapies: list[str] | None = None) -> pd.DataFrame:
        """Brand x region health check: achievement, YoY, and growth gap vs rest of India."""
        cur_ph = ", ".join("?" * len(period.keys))
        all_m = period.keys + period.py_keys
        ph = ", ".join("?" * len(all_m))
        sales = self.wh.query(f"""
            SELECT b.brand, b.therapy, g.region,
                   sum(CASE WHEN f.month IN ({cur_ph}) THEN f.net_value END) AS actual,
                   sum(CASE WHEN f.month NOT IN ({cur_ph}) THEN f.net_value END) AS last_year
            FROM agg_sales_tbm f JOIN dim_geo g USING (territory_id)
            JOIN dim_brand b USING (brand_id)
            WHERE f.month IN ({ph}) GROUP BY ALL""", period.keys + period.keys + all_m)
        tgt = self.wh.query(f"""
            SELECT b.brand, g.region, sum(t.target_value) AS target
            FROM fact_targets t JOIN dim_geo g USING (territory_id) JOIN dim_brand b USING (brand_id)
            WHERE t.month IN ({cur_ph}) GROUP BY ALL""", period.keys)
        df = sales.merge(tgt, on=["brand", "region"])
        df["yoy"] = df.actual / df.last_year - 1
        # Peer growth = weighted median of the brand's other regions (robust to one region
        # bouncing back from last year's problem; same rule as the RCA engine).
        df["peer_yoy"] = [
            _weighted_median(g.yoy, g.last_year)
            for g in (df[(df.brand == b) & (df.region != r)] for b, r in zip(df.brand, df.region,
                                                                              strict=True))]
        df["growth_gap"] = df.yoy - df.peer_yoy
        df["achievement"] = df.actual / df.target
        df["gap_vs_target"] = df.actual - df.target
        df["status"] = np.select(
            [(df.achievement < 0.95) | (df.growth_gap < -0.04),
             (df.achievement < 0.98) | (df.growth_gap < -0.02)],
            ["Underperforming", "Slightly below plan"], "On track")
        # Severity blends the two signals so a big miss on a big brand ranks first.
        df["severity"] = (np.clip(0.95 - df.achievement, 0, None) * 2
                          + np.clip(-df.growth_gap, 0, None)) * np.sqrt(df.last_year / 1e7)
        if regions:
            df = df[df.region.isin(regions)]
        if brands:
            df = df[df.brand.isin(brands)]
        if therapies:
            df = df[df.therapy.isin(therapies)]
        return df.sort_values("severity", ascending=False).reset_index(drop=True)

    def drill(self, scope: Scope, period: Period, dimension: str) -> pd.DataFrame:
        if dimension == "month":
            return self.trend("net_sales", scope, period)
        return self.compare("net_sales", scope, period, dimension)

    # ------------------------------------------------------------------ notes
    def notes_index(self) -> BM25:
        if self._notes is None:
            df = self.wh.query("""
                SELECT n.note_id, n.month, n.author_role, n.text, g.territory, g.state, g.region,
                       b.brand FROM field_notes n JOIN dim_geo g USING (territory_id)
                LEFT JOIN dim_brand b USING (brand_id)""")
            docs = [Doc(r.note_id, r.text, {"month": r.month, "role": r.author_role,
                                            "territory": r.territory, "state": r.state,
                                            "region": r.region, "brand": r.brand})
                    for r in df.itertuples()]
            self._notes = BM25(docs)
        return self._notes

    def search_notes(self, query: str, scope: Scope, period: Period | None = None,
                     cause: str | None = None, k: int = 5) -> list[Hit]:
        months = set(period.keys) if period else None

        def where(d: Doc) -> bool:
            m = d.meta
            return ((not scope.brands or m["brand"] in scope.brands)
                    and (not scope.territories or m["territory"] in scope.territories)
                    and (not scope.states or m["state"] in scope.states)
                    and (not scope.regions or m["region"] in scope.regions)
                    and (months is None or m["month"] in months))

        expand = EXPANSIONS.get(cause, ()) if cause else ()
        hits = self.notes_index().search(query, k=k * 4, where=where, expand=expand)
        seen: set[str] = set()
        unique = []
        for h in hits:  # templated reports repeat verbatim across towns; show distinct ones
            if h.doc.text not in seen:
                seen.add(h.doc.text)
                unique.append(h)
        return unique[:k]
