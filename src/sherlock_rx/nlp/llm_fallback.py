"""Optional LLM fallback for utterances the classical NLU is unsure about.

Only called when the intent classifier's calibrated confidence is below threshold. Claude
maps the utterance to the same closed schema the classical pipeline produces (intent +
slot strings), via structured outputs. Its slot strings are then linked through the same
gazetteer and temporal parser as everything else, so the LLM can never introduce a brand,
region or number the warehouse doesn't know. It never sees or produces any figures.
"""

from __future__ import annotations

import json
from typing import Any

from sherlock_rx.nlp.gazetteer import Gazetteer
from sherlock_rx.nlp.temporal import TemporalParser
from sherlock_rx.nlp.text import normalize

INTENTS = ["diagnose", "find_underperformers", "metric_lookup", "compare", "drill_down",
           "trend", "search_notes", "recommend", "define_term", "help", "greet", "thanks",
           "goodbye", "out_of_scope"]

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": INTENTS},
        "brands": {"type": "array", "items": {"type": "string"}},
        "regions": {"type": "array", "items": {"type": "string"}},
        "states": {"type": "array", "items": {"type": "string"}},
        "metric": {"type": "string"},
        "period_text": {"type": "string"},
    },
    "required": ["intent", "brands", "regions", "states", "metric", "period_text"],
    "additionalProperties": False,
}

SYSTEM = (
    "You route messages for a pharma sales-analytics chatbot used by Indian field and sales "
    "teams. Classify the user's message into exactly one intent and copy any brand, region "
    "(North/South/East/West/Central), Indian state, metric word and time expression exactly "
    "as written (empty string or empty list when absent). Intents: diagnose = why is a brand "
    "under-performing; find_underperformers = which brands/regions are weak; metric_lookup = "
    "a number; compare = regions/states side by side; drill_down = break down; trend = over "
    "time; search_notes = field reports; recommend = what to do; define_term = explain a "
    "term; help; greet; thanks; goodbye; out_of_scope = anything unrelated to brand "
    "performance. Messages may be in English or Hinglish."
)


class ClaudeFallback:
    def __init__(self, gazetteer: Gazetteer | None = None, model: str = "claude-opus-5-5",
                 client: Any = None):
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.client = client
        self.model = model
        self.gaz = gazetteer or Gazetteer.build()
        self.temporal = TemporalParser()

    def raw(self, text: str) -> dict[str, Any] | None:
        resp = self.client.beta.messages.create(
            model=self.model,
            max_tokens=1024,
            betas=["server-side-fallback-2026-07-01"],
            extra_body={"fallbacks": "default"},
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
            system=SYSTEM,
            messages=[{"role": "user", "content": text}],
        )
        if resp.stop_reason in ("refusal", "max_tokens"):
            return None
        block = next((b for b in resp.content if b.type == "text"), None)
        if block is None:
            return None
        try:
            data: dict[str, Any] = json.loads(block.text)
        except json.JSONDecodeError:
            return None
        return data

    def __call__(self, text: str) -> dict[str, Any] | None:
        try:
            data = self.raw(text)
        except Exception:  # the fallback is best-effort; the classical path still answers
            return None
        if not data or data.get("intent") not in INTENTS:
            return None
        slots: dict[str, Any] = {}
        for key, etype in (("brands", "BRAND"), ("regions", "REGION"), ("states", "STATE")):
            vals = []
            for v in data.get(key, []):
                linked = self.gaz.link(etype, v)
                if linked and linked[0] not in vals:
                    vals.append(linked[0])
            if vals:
                slots[key] = vals
        if data.get("metric"):
            linked = self.gaz.link("METRIC", data["metric"], cutoff=85)
            if linked:
                slots["metrics"] = [linked[0]]
        if data.get("period_text"):
            t = self.temporal.parse(normalize(data["period_text"]))
            if t is not None:
                slots["period"] = t.period
        return {"intent": data["intent"], "slots": slots}
