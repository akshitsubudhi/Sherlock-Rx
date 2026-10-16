"""DuckDB warehouse over Parquet.

* Views are registered over the Parquet files (sales is hive-partitioned by fiscal year so
  period filters prune whole files).
* One cursor per thread: DuckDB connections are not thread-safe, cursors are.
* Every query is logged with an id, its SQL and latency so answers can cite their evidence.
* A small LRU result cache absorbs the repeated lookups a conversation generates.
* Queries run straight on Parquet; worker processes share the OS page cache.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

TABLES = (
    "fact_inventory", "fact_targets", "fact_field", "fact_rx", "fact_distribution",
    "fact_market", "fact_pricing", "field_notes", "dim_geo", "dim_brand", "dim_sku",
    "dim_month", "dim_competitor",
)


@dataclass
class QueryRecord:
    query_id: str
    sql: str
    params: list[Any]
    rows: int
    ms: float
    cached: bool


@dataclass
class QueryLog:
    records: list[QueryRecord] = field(default_factory=list)

    def add(self, rec: QueryRecord) -> None:
        self.records.append(rec)
        if len(self.records) > 2000:
            del self.records[:1000]


class Warehouse:
    def __init__(self, data_dir: str | Path, cache_size: int = 512, materialize: bool = False,
                 threads: int | None = None):
        self.data_dir = Path(data_dir)
        if not (self.data_dir / "dim_geo.parquet").exists():
            raise FileNotFoundError(
                f"No warehouse at {self.data_dir}. Run `sherlock generate` first.")
        self._con = duckdb.connect(":memory:")
        if threads:
            # With N worker processes, give each cores/N threads; the default (all cores per
            # process) oversubscribes the CPU and throughput collapses (see reports/BENCH.md).
            self._con.execute(f"SET threads = {int(threads)}")
        self._local = threading.local()
        self._cache: OrderedDict[str, pd.DataFrame] = OrderedDict()
        self._cache_size = cache_size
        self._lock = threading.Lock()
        self.log = QueryLog()
        self._register(materialize)

    # ------------------------------------------------------------------ setup
    def _register(self, materialize: bool) -> None:
        d = self.data_dir.as_posix()
        self._con.execute(
            f"CREATE VIEW fact_sales AS SELECT * FROM read_parquet('{d}/fact_sales/*/*.parquet', "
            "hive_partitioning = true)")
        for t in TABLES:
            self._con.execute(f"CREATE VIEW {t} AS SELECT * FROM read_parquet('{d}/{t}.parquet')")
        if materialize:
            # Optional in-memory copy of the sales fact. Measured: no faster than scanning
            # the Parquet (DuckDB prunes by partition and column), and it costs a full copy
            # per worker process, which halved multi-process throughput at 7.6M rows.
            # Off by default; kept for experiments (docs/SCALING.md).
            self._con.execute("""
                CREATE TABLE agg_sales_tbm AS
                SELECT month, territory_id, brand_id, sku_id,
                       sum(units) AS units, sum(gross_value) AS gross_value,
                       sum(net_value) AS net_value
                FROM fact_sales GROUP BY ALL""")
        else:
            self._con.execute("CREATE VIEW agg_sales_tbm AS SELECT * FROM fact_sales")

    def _cursor(self) -> duckdb.DuckDBPyConnection:
        cur = getattr(self._local, "cursor", None)
        if cur is None:
            cur = self._con.cursor()
            self._local.cursor = cur
        return cur

    # ------------------------------------------------------------------ query
    def query(self, sql: str, params: list[Any] | None = None, use_cache: bool = True) -> pd.DataFrame:
        params = params or []
        key = hashlib.sha1(f"{sql}|{params!r}".encode()).hexdigest()[:12]
        if use_cache:
            with self._lock:
                hit = self._cache.get(key)
                if hit is not None:
                    self._cache.move_to_end(key)
                    self.log.add(QueryRecord(f"Q{key}", sql, params, len(hit), 0.0, True))
                    return hit.copy()
        t0 = time.perf_counter()
        df = self._cursor().execute(sql, params).df()
        ms = (time.perf_counter() - t0) * 1000
        self.log.add(QueryRecord(f"Q{key}", sql, params, len(df), round(ms, 2), False))
        if use_cache:
            with self._lock:
                self._cache[key] = df
                if len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)
        return df.copy()

    def scalar(self, sql: str, params: list[Any] | None = None) -> Any:
        df = self.query(sql, params)
        return None if df.empty else df.iat[0, 0]

    def clear_cache(self) -> None:
        with self._lock:
            self._cache.clear()

    # ------------------------------------------------------------------ catalog
    def distinct(self, table: str, column: str) -> list[str]:
        if table not in TABLES:
            raise ValueError(table)
        df = self.query(f"SELECT DISTINCT {column} AS v FROM {table} ORDER BY 1")
        return [str(v) for v in df["v"].tolist()]

    def latest_month(self) -> str:
        return str(self.scalar("SELECT max(month) FROM dim_month"))
