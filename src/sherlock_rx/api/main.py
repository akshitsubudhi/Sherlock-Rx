"""FastAPI service.

The process is stateless apart from the session store, which sits behind a small
interface: in-memory LRU+TTL here, Redis (or any KV store) in a multi-replica deployment,
since DialogueState is plain picklable data. The warehouse is read-only and each worker
thread gets its own DuckDB cursor, so the sync endpoints run safely in FastAPI's pool.
"""

from __future__ import annotations

import math
import os
import pickle
import threading
import time
import uuid
from collections import Counter, OrderedDict, deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from sherlock_rx.analytics.period import Period
from sherlock_rx.dialogue.manager import Sherlock, latest_quarter
from sherlock_rx.dialogue.state import DialogueState
from sherlock_rx.nlp.temporal import TemporalParser
from sherlock_rx.nlp.text import normalize

ROOT = Path(__file__).resolve().parents[3]


def to_jsonable(o: Any) -> Any:
    """numpy/pandas/dataclass/NaN-safe conversion for API responses."""
    if isinstance(o, Period):
        return o.to_dict()
    if is_dataclass(o) and not isinstance(o, type):
        return to_jsonable(asdict(o))
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, list | tuple | set):
        return [to_jsonable(v) for v in o]
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    return o


# ---------------------------------------------------------------------- sessions
class SessionStore(Protocol):
    def get(self, sid: str) -> DialogueState | None: ...

    def put(self, sid: str, state: DialogueState) -> None: ...


class MemorySessionStore:
    """LRU + TTL. States are pickled so the semantics match an external KV store."""

    def __init__(self, max_sessions: int = 10_000, ttl_s: int = 3600):
        self._d: OrderedDict[str, tuple[float, bytes]] = OrderedDict()
        self._lock = threading.Lock()
        self.max, self.ttl = max_sessions, ttl_s

    def get(self, sid: str) -> DialogueState | None:
        with self._lock:
            item = self._d.get(sid)
            if item is None or time.time() - item[0] > self.ttl:
                self._d.pop(sid, None)
                return None
            self._d.move_to_end(sid)
            state: DialogueState = pickle.loads(item[1])
            return state

    def put(self, sid: str, state: DialogueState) -> None:
        with self._lock:
            self._d[sid] = (time.time(), pickle.dumps(state))
            self._d.move_to_end(sid)
            while len(self._d) > self.max:
                self._d.popitem(last=False)

    def __len__(self) -> int:
        return len(self._d)


# ---------------------------------------------------------------------- metrics
class Metrics:
    def __init__(self) -> None:
        self.requests: Counter[str] = Counter()
        self.intents: Counter[str] = Counter()
        self.errors: Counter[str] = Counter()
        self.latency: dict[str, deque[float]] = {}
        self.started = time.time()
        self._lock = threading.Lock()

    def observe(self, route: str, ms: float) -> None:
        with self._lock:
            self.requests[route] += 1
            self.latency.setdefault(route, deque(maxlen=5000)).append(ms)

    def summary(self) -> dict[str, Any]:
        with self._lock:
            lat = {}
            for r, v in self.latency.items():
                a = np.array(v)
                lat[r] = {"p50": round(float(np.percentile(a, 50)), 1),
                          "p95": round(float(np.percentile(a, 95)), 1),
                          "p99": round(float(np.percentile(a, 99)), 1), "n": len(a)}
            return {"uptime_s": round(time.time() - self.started), "requests": dict(self.requests),
                    "errors": dict(self.errors), "intents": dict(self.intents),
                    "latency_ms": lat}


# ---------------------------------------------------------------------- app
class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1000)
    session_id: str | None = None


class NLURequest(BaseModel):
    text: str = Field(min_length=1, max_length=1000)


class RCARequest(BaseModel):
    brand: str
    region: str | None = None
    state: str | None = None
    period: str | None = Field(None, description='e.g. "Q2 FY27", "last quarter", "Aug 2026"')


def _build() -> Sherlock:
    from sherlock_rx.cli import build_bot

    data = Path(os.environ.get("SHERLOCK_DATA", ROOT / "data" / "warehouse"))
    models = Path(os.environ.get("SHERLOCK_MODELS", ROOT / "models"))
    return build_bot(data, models, llm=os.environ.get("SHERLOCK_LLM_FALLBACK") == "1",
                     threads=int(os.environ.get("SHERLOCK_DUCKDB_THREADS", "0")) or None)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.bot = _build()
    app.state.sessions = MemorySessionStore()
    app.state.metrics = Metrics()
    app.state.bot.warmup()  # build the BM25 notes index before taking traffic
    app.state.bot.handle("hello", DialogueState())
    yield


app = FastAPI(title="Sherlock Rx", version="0.1.0", lifespan=lifespan,
              description="NLP chatbot for root-cause analysis of under-performing pharma brands.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def timing(request: Request, call_next):  # type: ignore[no-untyped-def]
    t0 = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - t0) * 1000
    response.headers["x-latency-ms"] = f"{ms:.1f}"
    if hasattr(request.app.state, "metrics"):
        request.app.state.metrics.observe(request.url.path, ms)
    return response


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {"status": "ok", "sessions": len(app.state.sessions)}


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    return app.state.metrics.summary()  # type: ignore[no-any-return]


@app.get("/v1/catalog")
def catalog() -> dict[str, Any]:
    bot: Sherlock = app.state.bot
    return {"brands": bot.brands, "regions": bot.regions,
            "states": sorted(bot.state_region), "latest_quarter": latest_quarter().label,
            "intents": bot.nlu.meta.get("intents", []), "model": bot.nlu.meta}


@app.post("/v1/chat")
def chat(req: ChatRequest) -> dict[str, Any]:
    bot: Sherlock = app.state.bot
    sid = req.session_id or uuid.uuid4().hex
    state = app.state.sessions.get(sid) or bot.new_state()
    try:
        resp = bot.handle(req.message, state)
    except Exception as e:  # surface as a 500 with the reason; never kill the worker
        app.state.metrics.errors[type(e).__name__] += 1
        raise HTTPException(500, f"{type(e).__name__}: {e}") from e
    app.state.sessions.put(sid, state)
    app.state.metrics.intents[resp.intent] += 1
    out = to_jsonable(resp.to_dict())
    out["session_id"] = sid
    return out  # type: ignore[no-any-return]


@app.post("/v1/nlu")
def nlu(req: NLURequest) -> dict[str, Any]:
    return to_jsonable(app.state.bot.nlu.parse(req.text, explain=True).debug())  # type: ignore[no-any-return]


@app.post("/v1/rca")
def rca(req: RCARequest) -> dict[str, Any]:
    bot: Sherlock = app.state.bot
    period = latest_quarter()
    if req.period:
        t = TemporalParser().parse(normalize(req.period))
        if t is None:
            raise HTTPException(422, f"Could not understand period '{req.period}'.")
        period = t.period
    region = req.region
    if region is None and req.state is None:
        raise HTTPException(422, "Give a region or a state.")
    if region is None:
        region = bot.state_region.get(req.state or "")
        if region is None:
            raise HTTPException(422, f"Unknown state '{req.state}'.")
    from sherlock_rx.analytics.rca import RCAError

    try:
        report = bot.rca.diagnose(req.brand, region, period, state=req.state)
    except RCAError as e:
        raise HTTPException(422, str(e)) from e
    return to_jsonable(report.to_dict())  # type: ignore[no-any-return]
