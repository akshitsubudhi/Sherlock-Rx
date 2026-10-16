"""Hybrid entity extraction = gazetteer (precision) + perceptron tagger (recall/context)
+ entity linking (normalisation) + temporal parser (TIME semantics).

    1. Gazetteer spans (exact / fuzzy / state codes) are kept as-is.
    2. Tagger spans that do not overlap a gazetteer span are linked to a canonical value
       with the type-restricted fuzzy linker; unlinkable spans are dropped, never guessed.
    3. TIME: the rule-based parser supplies the normalised Period.
    4. The result is turned into dialogue slots, with geography resolved up the hierarchy
       (territory -> state -> region).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sherlock_rx.analytics.period import Period
from sherlock_rx.data.catalog import HQ_TOWNS, region_of_state
from sherlock_rx.nlp.gazetteer import Gazetteer, Span
from sherlock_rx.nlp.tagger import PerceptronTagger, bio_spans
from sherlock_rx.nlp.temporal import TemporalError, TemporalParser, TimeExpr
from sherlock_rx.nlp.text import normalize, tokenize


@dataclass
class Extraction:
    tokens: list[str]
    spans: list[Span]
    time: TimeExpr | None
    slots: dict[str, Any] = field(default_factory=dict)

    def masked(self) -> str:
        """Utterance with entity spans replaced by type placeholders, for intent features:
        "why is entbrand down in entregion enttime"."""
        out: list[str] = []
        i = 0
        by_start = {s.start: s for s in self.spans}
        while i < len(self.tokens):
            if i in by_start:
                sp = by_start[i]
                out.append(f"ent{sp.type.lower()}")  # one token, e.g. "entbrand"
                i = sp.end
            else:
                out.append(self.tokens[i])
                i += 1
        return " ".join(out)


class EntityExtractor:
    def __init__(self, gazetteer: Gazetteer, tagger: PerceptronTagger | None,
                 temporal: TemporalParser | None = None,
                 territory_state: dict[str, str] | None = None):
        self.gaz = gazetteer
        self.tagger = tagger
        self.temporal = temporal or TemporalParser()
        self.territory_state = territory_state or {
            t: s for s, towns in HQ_TOWNS.items() for t in towns}

    def extract(self, text: str, use_tagger: bool = True, use_gazetteer: bool = True
                ) -> Extraction:
        norm = normalize(text)
        toks = tokenize(norm)
        words = [t.text for t in toks]
        gspans = self.gaz.find(text)
        spans: list[Span] = list(gspans) if use_gazetteer else []
        if use_tagger and self.tagger is not None:
            tags = self.tagger.predict(words, gspans)
            for s, e, typ in bio_spans(tags):
                if typ in ("TIME", "TERM"):
                    continue
                if any(s < o.end and o.start < e for o in spans):
                    continue
                surface = " ".join(words[s:e])
                linked = self.gaz.link(typ, surface)
                if linked is None:
                    continue
                spans.append(Span(s, e, toks[s].start, toks[e - 1].end, typ, linked[0],
                                  linked[1], surface, "tagger"))
        time = None
        try:
            time = self.temporal.parse(norm)
            if time is not None:
                self.temporal.validate(time.period)
        except TemporalError:
            time = None
        if time is not None:
            # A time expression wins over overlapping dictionary spans ("may", "march").
            spans = [s for s in spans if not (s.char_start < time.end and time.start < s.char_end)]
            ts = [i for i, t in enumerate(toks) if t.start >= time.start and t.end <= time.end]
            if ts:
                spans.append(Span(ts[0], ts[-1] + 1, time.start, time.end, "TIME",
                                  time.period.label, 100.0, time.text, "temporal"))
        spans.sort(key=lambda s: s.start)
        ex = Extraction(words, spans, time)
        ex.slots = self.to_slots(ex)
        return ex

    def to_slots(self, ex: Extraction) -> dict[str, Any]:
        slots: dict[str, Any] = {}
        multi = {"BRAND": "brands", "REGION": "regions", "STATE": "states",
                 "TERRITORY": "territories", "SKU": "skus", "COMPETITOR": "competitors",
                 "THERAPY": "therapies", "METRIC": "metrics"}
        for sp in ex.spans:
            if sp.type in multi:
                vals = slots.setdefault(multi[sp.type], [])
                if sp.value not in vals:
                    vals.append(sp.value)
            elif sp.type == "DIM":
                slots["dimension"] = sp.value
            elif sp.type == "BASELINE":
                slots["baseline"] = sp.value
        if ex.time is not None:
            slots["period"] = ex.time.period
        # SKU implies brand; territory implies state; state implies region.
        for sku in slots.get("skus", []):
            brand = sku.split()[0]
            if brand not in slots.setdefault("brands", []):
                slots["brands"].append(brand)
        for terr in slots.get("territories", []):
            st = self.territory_state.get(terr.split("-")[0])
            if st and st not in slots.setdefault("states", []):
                slots["states"].append(st)
        for st in slots.get("states", []):
            reg = region_of_state(st)
            if reg not in slots.setdefault("regions", []):
                slots["regions"].append(reg)
        return slots


def period_of(slots: dict[str, Any]) -> Period | None:
    p = slots.get("period")
    return p if isinstance(p, Period) else None
