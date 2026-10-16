from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(warehouse_dir: Path, models_dir: Path):  # type: ignore[no-untyped-def]
    import os

    os.environ["SHERLOCK_DATA"] = str(warehouse_dir)
    os.environ["SHERLOCK_MODELS"] = str(models_dir)
    from sherlock_rx.api.main import app

    with TestClient(app) as c:
        yield c


def test_health_and_catalog(client: TestClient) -> None:
    assert client.get("/healthz").json()["status"] == "ok"
    cat = client.get("/v1/catalog").json()
    assert len(cat["brands"]) == 8 and "North" in cat["regions"]


def test_chat_session_carries_state(client: TestClient) -> None:
    r1 = client.post("/v1/chat", json={"message": "why is Cardivex down in North"}).json()
    r2 = client.post("/v1/chat", json={"message": "what about South?",
                                       "session_id": r1["session_id"]}).json()
    assert r2["state"]["regions"] == ["South"] and r2["state"]["brands"] == ["Cardivex"]
    assert r2["payload"]["type"] == "rca"
    json.dumps(r2)  # fully JSON-serialisable (no numpy / NaN leaks)


def test_rca_endpoint_validates(client: TestClient) -> None:
    ok = client.post("/v1/rca", json={"brand": "Glucoryl", "state": "Rajasthan",
                                      "period": "Q1 FY27"})
    assert ok.status_code == 200 and ok.json()["region"] == "North"
    assert client.post("/v1/rca", json={"brand": "Nope", "region": "North"}).status_code == 422
    assert client.post("/v1/rca", json={"brand": "Glucoryl"}).status_code == 422


def test_metrics_endpoint(client: TestClient) -> None:
    m = client.get("/metrics").json()
    assert "/v1/chat" in m["latency_ms"]


# ---------------------------------------------------------------- LLM fallback (mocked)
class FakeClient:
    def __init__(self, payload: dict[str, Any], stop: str = "end_turn"):
        self.payload, self.stop, self.calls = payload, stop, []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kw: Any) -> Any:
        self.calls.append(kw)
        block = SimpleNamespace(type="text", text=json.dumps(self.payload))
        return SimpleNamespace(stop_reason=self.stop, content=[block])


def test_fallback_links_llm_output_through_gazetteer() -> None:
    from sherlock_rx.nlp.llm_fallback import ClaudeFallback

    fake = FakeClient({"intent": "diagnose", "brands": ["glucoril", "NotABrand"],
                       "regions": ["southern"], "states": [], "metric": "",
                       "period_text": "Q1 FY27"})
    out = ClaudeFallback(client=fake)("Glucoril dakshin mein kyun gir raha")
    assert out == {"intent": "diagnose",
                   "slots": {"brands": ["Glucoryl"], "regions": ["South"],
                             "period": out["slots"]["period"]}}  # type: ignore[index]
    kw = fake.calls[0]
    assert kw["model"] == "claude-opus-5-5"
    assert kw["output_config"]["format"]["type"] == "json_schema"


def test_fallback_handles_refusal() -> None:
    from sherlock_rx.nlp.llm_fallback import ClaudeFallback

    fake = FakeClient({"intent": "diagnose"}, stop="refusal")
    assert ClaudeFallback(client=fake)("anything") is None
