# Scaling

`make bench` → [`reports/BENCH.md`](../reports/BENCH.md). Hardware: 8-core arm64 laptop, 8.6 GB RAM.

| Scale | Territories | Sales rows | All rows | Parquet | Generate | RCA cold p50 | RCA warm p50 | Chat turn p50 | RCA/s 1 thread | 8 threads | 4 processes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ×1 | 113 | 76k | 0.33M | 2.4 MB | 0.2 s | 68 ms | 55 ms | 10 ms | 17.1 | 12.0 | 51.4 |
| ×10 | 1,134 | 762k | 3.1M | 18 MB | 1.4 s | 109 ms | 62 ms | 12 ms | 13.9 | 10.7 | 24.5 |
| ×50 | 5,670 | 3.8M | 15.3M | 63 MB | 7.1 s | 203 ms | 93 ms | 30 ms | 9.0 | 8.5 | 13.9 |
| ×100 | 11,340 | 7.6M | 30.6M | 118 MB | 14.5 s | 253 ms | 118 ms | 36 ms | 6.9 | 6.3 | 8.2 |

NLU latency is flat (0.5 ms) at every scale: it does not touch the data.

## Findings that changed the code

1. **Threads vs processes.** The RCA engine spends most of its time in pandas (pivots,
   merges), which holds the GIL; DuckDB queries are a minority of the time. 8 threads were
   *slower* than 1 at every scale. 4 processes gave 3.0× at ×1. Deployment therefore scales
   with uvicorn `--workers`, not a thread pool.
2. **CPU oversubscription.** Each DuckDB connection defaults to all cores, so 4 workers ran
   32 threads on 8 cores. `SHERLOCK_DUCKDB_THREADS` (= cores ÷ workers) raised ×50
   throughput from 9.6 to 13.0 RCA/s.
3. **Removing an optimisation.** I first materialised the sales fact in memory as a
   "roll-up". Measured at ×100: single-process RCA 272 ms with it vs 264 ms without (DuckDB
   prunes Parquet by partition and column just as well). Multi-process throughput was 5.0 vs
   8.1 RCA/s, because each worker held its own copy on an 8.6 GB machine. It is now off by
   default; the flag remains for experiments.
4. **Start-up work belongs at start-up.** The BM25 index over field notes was built lazily,
   so the first diagnosis at ×50 took 1.3 s. It is now built in the API lifespan
   (`Sherlock.warmup()`), 2.2 s at ×100, before taking traffic.
5. **Generator memory.** Object-dtype string columns for 7.6M rows would need an estimated ~2 GB (not measured).
   Categorical codes keep the generator at ×100 inside the laptop's memory. Output was
   checked unchanged (same row count and the same ₹ total to the paisa).

## Production path

| Concern | Here | In production |
|---|---|---|
| Warehouse | DuckDB on Parquet | Snowflake / BigQuery / Databricks SQL; `semantic.transpile(sql, "snowflake")` already works |
| Sessions | in-memory LRU+TTL, pickled states | Redis (same interface: `get` / `put`) |
| Scale-out | uvicorn workers | N stateless replicas behind a load balancer |
| Query cache | per-process LRU keyed by (SQL, params) | shared Redis cache, invalidated on data refresh |
| Notes search | in-process BM25 | OpenSearch / Elasticsearch BM25 (same scoring) |
| Data refresh | regenerate | nightly ELT; RCA reads the latest partition |
| Observability | `/metrics` (p50/p95/p99 per route, intents, errors), `x-latency-ms` header | Prometheus + tracing; the query log already gives SQL-level timings |
| LLM cost | fallback only below the confidence threshold | same; measure the fallback rate from `/metrics` intents |

The expensive part, the RCA, is **per (brand, region, quarter)** and deterministic, so it can
be precomputed nightly for all 40 brand-region cells per quarter. Answers then become cache
reads, which is the step after horizontal scaling.
