"""Scale benchmark: how generation, loading, RCA and chat latency grow with data volume.

For each scale factor: generate the warehouse, open it (incl. building the roll-up),
then time cold and warm RCA, the portfolio scan, NLU parsing, full chat turns, and
concurrent throughput from a thread pool (DuckDB releases the GIL while executing).
"""

from __future__ import annotations

import json
import os
import resource
import statistics as st
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from sherlock_rx.analytics.period import Period
from sherlock_rx.data.calendar import quarter_months

_ENGINE: Any = None


def _init_worker(data_dir: str) -> None:
    from sherlock_rx.analytics.rca import RCAEngine
    from sherlock_rx.warehouse.engine import Warehouse

    global _ENGINE
    _ENGINE = RCAEngine(Warehouse(data_dir, threads=max(1, (os.cpu_count() or 4) // 4)))


def _diagnose_in_worker(case: tuple[str, str, list[str], str]) -> float:
    t0 = time.perf_counter()
    _ENGINE.wh.clear_cache()
    _ENGINE.diagnose(case[0], case[1], Period.from_keys(case[2], case[3]))
    return time.perf_counter() - t0


def _dir_mb(p: Path) -> float:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6


def _ms(fn: Any) -> float:
    t0 = time.perf_counter()
    fn()
    return (time.perf_counter() - t0) * 1000


def bench_scale(scale: float, models: Path, tmp: Path) -> dict[str, Any]:
    from sherlock_rx.analytics.queries import Analytics
    from sherlock_rx.analytics.rca import RCAEngine
    from sherlock_rx.cli import build_bot
    from sherlock_rx.data.generator import generate
    from sherlock_rx.warehouse.engine import Warehouse

    d = tmp / f"sf{scale:g}"
    t0 = time.perf_counter()
    summary = generate(d, scale=scale, seed=7)
    gen_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    wh = Warehouse(d)
    load_s = time.perf_counter() - t0
    eng, an = RCAEngine(wh), Analytics(wh)
    gt = json.loads((d / "_ground_truth.json").read_text())["scenarios"]
    cases = [(s["brand"], s["region"], Period(quarter_months(s["fy"], s["fq"]), s["quarter"]))
             for s in gt[:12]]

    wh.clear_cache()
    cold = [_ms(lambda c=c: eng.diagnose(*c)) for c in cases]
    warm = [_ms(lambda c=c: eng.diagnose(*c)) for c in cases]
    def cold_scan(p: Period) -> None:
        wh.clear_cache()
        an.scan(p)

    scan = [_ms(lambda c=c: cold_scan(c[2])) for c in cases[:5]]

    bot = build_bot(d, models)
    index_s = bot.warmup()
    qs = ["which brands are underperforming?", "why?", "break it down by territory",
          "Glucoryl market share in Rajasthan", "show the monthly trend",
          "why is Cardivex down in North in Q1 FY27", "what should we do?"]
    state = bot.new_state()
    wh2 = bot.wh
    wh2.clear_cache()
    turns = [bot.handle(q, state).timings_ms["total"] for q in qs]
    nlu = [_ms(lambda q=q: bot.nlu.parse(q)) for q in qs * 5]

    # Concurrency: 48 cold RCA requests, sequential vs 8 threads, fresh cache each time.
    work = (cases * 4)[:48]
    wh.clear_cache()
    t0 = time.perf_counter()
    for c in work:
        eng.diagnose(*c)
    seq_s = time.perf_counter() - t0
    wh.clear_cache()
    t0 = time.perf_counter()
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda c: eng.diagnose(*c), work))
    par_s = time.perf_counter() - t0

    # Scale-out path: separate processes (one warehouse handle each), like uvicorn workers.
    payload = [(b, r, p.keys, p.label) for b, r, p in work]
    with ProcessPoolExecutor(4, initializer=_init_worker, initargs=(str(d),)) as pool:
        list(pool.map(_diagnose_in_worker, payload[:8]))  # warm worker start-up
        t0 = time.perf_counter()
        list(pool.map(_diagnose_in_worker, payload))
        proc_s = time.perf_counter() - t0

    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1e6 if os.uname().sysname
                                                                   == "Darwin" else 1e3)
    return {
        "scale": scale,
        "territories": summary.territories,
        "sales_rows": summary.rows["fact_sales"],
        "total_rows": sum(summary.rows.values()),
        "parquet_mb": round(_dir_mb(d), 1),
        "generate_s": round(gen_s, 2),
        "open_and_rollup_s": round(load_s, 2),
        "rca_cold_ms_p50": round(st.median(cold), 1),
        "rca_cold_ms_max": round(max(cold), 1),
        "rca_warm_ms_p50": round(st.median(warm), 1),
        "scan_ms_p50": round(st.median(scan), 1),
        "nlu_ms_p50": round(st.median(nlu), 2),
        "chat_turn_ms_p50": round(st.median(turns), 1),
        "chat_turn_ms_max": round(max(turns), 1),
        "rca_throughput_seq_per_s": round(len(work) / seq_s, 1),
        "rca_throughput_8_threads_per_s": round(len(work) / par_s, 1),
        "rca_throughput_4_procs_per_s": round(len(work) / proc_s, 1),
        "notes_index_build_s": round(index_s, 2),
        "peak_rss_mb": round(rss_mb),
    }


def run(scales: list[float], models: Path, out: Path) -> list[dict[str, Any]]:
    out.mkdir(parents=True, exist_ok=True)
    results = []
    with tempfile.TemporaryDirectory() as t:
        for sc in scales:
            print(f"benchmarking scale {sc:g} ...", flush=True)
            results.append(bench_scale(sc, models, Path(t)))
            print(json.dumps(results[-1]), flush=True)
    (out / "bench.json").write_text(json.dumps(results, indent=2))
    (out / "BENCH.md").write_text(render(results))
    return results


def render(rows: list[dict[str, Any]]) -> str:
    cols = [("scale", "Scale"), ("territories", "Territories"), ("sales_rows", "Sales rows"),
            ("total_rows", "All rows"), ("parquet_mb", "Parquet MB"), ("generate_s", "Generate s"),
            ("open_and_rollup_s", "Open s"), ("rca_cold_ms_p50", "RCA cold p50 ms"),
            ("rca_warm_ms_p50", "RCA warm p50 ms"), ("scan_ms_p50", "Scan p50 ms"),
            ("chat_turn_ms_p50", "Chat turn p50 ms"), ("nlu_ms_p50", "NLU p50 ms"),
            ("rca_throughput_seq_per_s", "RCA/s (1 thread)"),
            ("rca_throughput_8_threads_per_s", "RCA/s (8 threads)"),
            ("rca_throughput_4_procs_per_s", "RCA/s (4 processes)"),
            ("notes_index_build_s", "Notes index build s"), ("peak_rss_mb", "Peak RSS MB")]
    lines = ["# Scale benchmark", "",
             f"_Measured {time.strftime('%Y-%m-%d')} on {os.uname().machine} "
             f"({os.cpu_count()} cores) with `sherlock bench`. Peak RSS is cumulative across "
             "the run._", "",
             "| " + " | ".join(c[1] for c in cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(f"{r[k]:,}" if isinstance(r[k], int) else str(r[k])
                                       for k, _ in cols) + " |")
    return "\n".join(lines) + "\n"
