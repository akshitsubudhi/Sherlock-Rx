"""Dialogue manager: NLU result + dialogue state -> policy decision -> analytics -> NLG.

Policy rules worth knowing:
  * follow-ups    "what about South?" / "and Kerala?" re-run the previous task intent with
                  the new slot (ellipsis resolution); "why?" after a ranking diagnoses the
                  top-ranked brand-region.
  * carry-over    unspecified slots come from the dialogue state (DST).
  * clarification missing mandatory slots set a pending frame; the next turn that supplies
                  the slot resumes the pending intent ("Cardivex" -> continues the diagnosis).
  * defaults      a missing period becomes the latest complete quarter, and the answer
                  says so (assumptions are always surfaced, never silent).
  * low confidence  below the calibrated threshold the bot asks, offering the top-2 intents,
                  or defers to the optional LLM fallback when one is configured.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from sherlock_rx.analytics.period import Period
from sherlock_rx.analytics.queries import Analytics, Scope
from sherlock_rx.analytics.rca import RCAEngine, RCAError
from sherlock_rx.data.calendar import LATEST, quarter_months
from sherlock_rx.dialogue import nlg
from sherlock_rx.dialogue.state import DialogueState
from sherlock_rx.nlp.pipeline import NLU, NLUResult
from sherlock_rx.nlp.retrieval import Glossary
from sherlock_rx.warehouse.engine import Warehouse

TASK_INTENTS = {"diagnose", "find_underperformers", "metric_lookup", "compare", "drill_down",
                "trend", "search_notes", "recommend"}
SMALLTALK = {"greet", "thanks", "goodbye", "help", "out_of_scope", "define_term"}
FOLLOWUP_RE = re.compile(r"^\s*(and|what about|how about|same for|now for|and for|what of|"
                         r"ok and|also for|then)\b")
ANAPHORA_RE = re.compile(r"\b(there|that region|that state|that brand|same region|it|this)\b")
INTENT_DESCRIPTIONS = {
    "diagnose": "diagnose why a brand is under-performing",
    "find_underperformers": "find under-performing brands",
    "metric_lookup": "look up a number",
    "compare": "compare regions",
    "drill_down": "break the numbers down",
    "trend": "show a trend",
    "search_notes": "search field reports",
    "recommend": "suggest actions",
    "define_term": "explain a term",
    "help": "show what I can do",
}
CAUSE_FROM_METRIC = {"stockout_days": "stockout", "vacancy_rate": "field_force_vacancy",
                     "calls": "field_force_vacancy", "active_retailers": "distribution_loss",
                     "market_units": "market_slowdown"}


@dataclass
class BotResponse:
    text: str
    intent: str
    confidence: float
    suggestions: list[str] = field(default_factory=list)
    payload: dict[str, Any] | None = None
    nlu: dict[str, Any] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict)
    timings_ms: dict[str, float] = field(default_factory=dict)
    policy: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def latest_quarter() -> Period:
    fy, fq = LATEST.fy, LATEST.fq
    if quarter_months(fy, fq)[-1] > LATEST:
        fq -= 1
        if fq == 0:
            fy, fq = fy - 1, 4
    return Period(quarter_months(fy, fq), f"Q{fq} FY{fy:02d}")


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [{k: (None if isinstance(v, float) and v != v else v) for k, v in r.items()}
            for r in df.to_dict("records")]


Fallback = Callable[[str], dict[str, Any] | None]


class Sherlock:
    def __init__(self, wh: Warehouse, nlu: NLU, fallback: Fallback | None = None):
        self.wh = wh
        self.nlu = nlu
        self.analytics = Analytics(wh)
        self.rca = RCAEngine(wh)
        self.glossary = Glossary()
        self.fallback = fallback
        self.brands = self.rca.brands.brand.tolist()
        self.regions = sorted(self.rca.geo.region.unique())
        self.state_region = dict(zip(self.rca.geo.state, self.rca.geo.region, strict=True))

    def warmup(self) -> float:
        """Build lazy structures (BM25 notes index) up front; returns seconds taken."""
        t0 = time.perf_counter()
        self.analytics.notes_index()
        return time.perf_counter() - t0

    def new_state(self) -> DialogueState:
        return DialogueState()

    # ------------------------------------------------------------------ entry point
    def handle(self, text: str, state: DialogueState) -> BotResponse:
        t0 = time.perf_counter()
        res = self.nlu.parse(text, explain=True)
        t_nlu = (time.perf_counter() - t0) * 1000
        intent, policy = res.intent, "classifier"
        slots = dict(res.slots)
        has_entities = any(k in slots for k in ("brands", "regions", "states", "territories",
                                                "skus", "period", "metrics", "dimension"))
        lowered = res.corrected.lower()

        entity_only = has_entities and self._entity_only(res)
        comparing = bool(re.search(r"\b(vs|versus|compare|against)\b", lowered))
        if (has_entities and state.last_intent in TASK_INTENTS and not comparing
                and (FOLLOWUP_RE.search(lowered) or entity_only) and not state.pending):
            intent, policy = str(state.last_intent), "follow-up (ellipsis)"
        elif state.pending and intent not in {"greet", "thanks", "goodbye", "help"}:
            missing = state.pending.get("missing")
            if missing and slots.get(missing):
                intent, policy = state.pending["intent"], f"resume pending ({missing})"
        if policy == "classifier" and res.uncertain:
            alt = self.fallback(text) if self.fallback else None
            if alt and alt.get("intent") in INTENT_DESCRIPTIONS | {"greet": "", "thanks": "",
                                                                   "goodbye": "", "out_of_scope": ""}:
                intent, policy = alt["intent"], "llm fallback"
                slots.update({k: v for k, v in alt.get("slots", {}).items() if v})
            elif not has_entities:
                return self._finish(self._clarify(res), res, state, t0, t_nlu, "clarify")

        if intent in TASK_INTENTS:
            if intent != "find_underperformers":
                state.update(slots)
            if state.pending and intent == state.pending.get("intent"):
                state.pending = None
        handler = getattr(self, f"_h_{intent}", self._h_out_of_scope)
        t1 = time.perf_counter()
        try:
            out = handler(state, slots, res)
        except RCAError as e:
            out = (f"I couldn't run that analysis: {e}", None, ["Help"])
        t_an = (time.perf_counter() - t1) * 1000
        resp = BotResponse(out[0], intent, res.confidence, out[2], out[1], res.debug(), {},
                           {"nlu": round(t_nlu, 1), "analytics": round(t_an, 1)}, policy)
        if intent in TASK_INTENTS:
            state.last_intent = intent
        state.turns += 1
        resp.state = state.snapshot()
        resp.timings_ms["total"] = round((time.perf_counter() - t0) * 1000, 1)
        return resp

    def _finish(self, out: tuple[str, dict[str, Any] | None, list[str]], res: NLUResult,
                state: DialogueState, t0: float, t_nlu: float, policy: str) -> BotResponse:
        state.turns += 1
        return BotResponse(out[0], res.intent, res.confidence, out[2], out[1], res.debug(),
                           state.snapshot(), {"nlu": round(t_nlu, 1),
                                              "total": round((time.perf_counter() - t0) * 1000, 1)},
                           policy)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _entity_only(res: NLUResult) -> bool:
        """True when the utterance is only brand/geo/time mentions: "Painex", "and Kerala?".
        Metric, dimension or baseline mentions carry intent, so they never count."""
        if any(sp.type in ("METRIC", "DIM", "BASELINE", "COMPETITOR", "THERAPY")
               for sp in res.spans):
            return False
        covered = {i for sp in res.spans for i in range(sp.start, sp.end)}
        filler = {"and", "what", "about", "how", "ok", "now", "then", "for", "in", "the", "?",
                  "same", "also", "of", "please", ",", "."}
        return all(i in covered or t in filler for i, t in enumerate(res.tokens))

    @staticmethod
    def _period(state: DialogueState) -> tuple[Period, str]:
        if state.period is not None:
            return state.period, ""
        p = latest_quarter()
        return p, f"_Assumed {p.label}, the latest complete quarter._"

    @staticmethod
    def _scope(state: DialogueState) -> Scope:
        return Scope(state.brands, state.regions, state.states, state.territories, state.skus)

    def _clarify(self, res: NLUResult) -> tuple[str, dict[str, Any] | None, list[str]]:
        opts = [i for i, _ in res.top[:2] if i in INTENT_DESCRIPTIONS]
        if not opts:
            text, payload, sugg = self._h_help(DialogueState(), {}, res)
            return "Sorry, I didn't catch that.\n\n" + text, payload, sugg
        ask = " or ".join(f"*{INTENT_DESCRIPTIONS[o]}*" for o in opts)
        return (f"I'm not sure I understood. Did you want me to {ask}? You can also say "
                f"“help” for examples.", None,
                ["Which brands are underperforming?", "Why is Cardivex down in North?", "Help"])

    # ------------------------------------------------------------------ task handlers
    def _h_diagnose(self, state: DialogueState, slots: dict[str, Any], res: NLUResult
                    ) -> tuple[str, dict[str, Any] | None, list[str]]:
        new_geo = any(k in slots for k in ("regions", "states", "territories"))
        notes_prefix = []
        if not slots.get("brands") and not new_geo and state.last_scan and \
                state.last_intent == "find_underperformers":
            worst = state.last_scan[0]
            state.brands, state.regions, state.states = [worst["brand"]], [worst["region"]], []
        period, assumed = self._period(state)
        if assumed:
            notes_prefix.append(assumed)
        if not state.brands and not (state.regions or state.states):
            state.pending = {"intent": "diagnose", "missing": "brands"}
            return ("Which brand (and region) should I investigate? For example: *“Why is "
                    "Glucoryl underperforming in North?”*. Or ask *“which brands are "
                    "underperforming?”* and I'll find candidates.", None,
                    ["Which brands are underperforming?", *[f"Diagnose {b}" for b in self.brands[:3]]])
        if not state.brands:
            scan = self.analytics.scan(period, regions=state.regions)
            state.brands = [str(scan.brand.iat[0])]
            notes_prefix.append(f"_No brand given; the weakest in {', '.join(state.regions)} "
                                f"is **{state.brands[0]}**, so I diagnosed that._")
        if not (state.regions or state.states):
            scan = self.analytics.scan(period, brands=state.brands)
            state.regions = [str(scan.region.iat[0])]
            notes_prefix.append(f"_No region given; {state.brands[0]} is weakest in "
                                f"**{state.regions[0]}**, so I diagnosed that._")
        brand = state.brands[0]
        geo_state = state.states[0] if state.states else None
        report = self.rca.diagnose(brand, state.regions[0], period, state=geo_state)
        state.last_report = report
        top = report.top_cause
        notes = []
        if top is not None:
            locs = top.locations if top.cause not in ("price_cut", "market_slowdown") else []
            states = [s for s in locs if s in self.state_region] or (
                [geo_state] if geo_state else [])
            scope = Scope([brand], [] if states else [report.region], states, [], [])
            notes = self.analytics.search_notes(top.label, scope, period, cause=top.cause, k=2)
        text = nlg.rca_markdown(report, notes)
        if notes_prefix:
            text = "\n".join(notes_prefix) + "\n\n" + text
        where = geo_state or report.region
        payload = {"type": "rca", "data": report.to_dict(),
                   "notes": [{"text": n.doc.text, **n.doc.meta, "score": n.score} for n in notes]}
        sugg = ["Break it down by territory", "What should we do?",
                f"Show the monthly trend of {brand} in {where}", "Any field reports?"]
        return text, payload, sugg

    def _h_find_underperformers(self, state: DialogueState, slots: dict[str, Any],
                                res: NLUResult) -> tuple[str, dict[str, Any] | None, list[str]]:
        use_state = bool(ANAPHORA_RE.search(res.corrected.lower()))
        regions = slots.get("regions") or (state.regions if use_state else [])
        brands = slots.get("brands") or []
        therapies = slots.get("therapies") or []
        if slots.get("period"):
            state.period = slots["period"]
        period, assumed = self._period(state)
        df = self.analytics.scan(period, regions or None, brands or None, therapies or None)
        state.last_scan = _records(df.head(10))
        label = ", ".join(regions or brands or therapies) or "the portfolio"
        text = nlg.scan_markdown(df, period.label, label)
        if assumed:
            text = assumed + "\n\n" + text
        payload = {"type": "scan", "data": _records(df.head(20))}
        sugg = ["Why?"] + [f"Why is {r['brand']} underperforming in {r['region']}?"
                           for r in state.last_scan[1:3]]
        return text, payload, sugg

    def _h_metric_lookup(self, state: DialogueState, slots: dict[str, Any], res: NLUResult
                         ) -> tuple[str, dict[str, Any] | None, list[str]]:
        metric = (slots.get("metrics") or state.metrics or ["net_sales"])[0]
        if slots.get("baseline") == "target" and metric == "net_sales":
            metric = "achievement"
        period, assumed = self._period(state)
        scope = self._scope(state)
        if metric in ("market_share", "market_units") and not scope.brands:
            state.pending = {"intent": "metric_lookup", "missing": "brands"}
            return "Market share for which brand?", None, [f"{b} market share" for b in
                                                            self.brands[:3]]
        s = self.analytics.metric_summary(metric, scope, period)
        brand_label = ", ".join(scope.skus or scope.brands) or "all brands"
        text = nlg.metric_markdown(s, scope.geo_label, brand_label, period.label)
        if assumed:
            text += "\n\n" + assumed
        sugg = [f"Why is {brand_label} underperforming in {scope.geo_label}?",
                "Compare regions", "Show the trend"] if scope.brands else [
            "Which brands are underperforming?"]
        return text, {"type": "metric", "data": s}, sugg

    def _h_compare(self, state: DialogueState, slots: dict[str, Any], res: NLUResult
                   ) -> tuple[str, dict[str, Any] | None, list[str]]:
        metric = (slots.get("metrics") or ["net_sales"])[0]
        period, assumed = self._period(state)
        want = slots.get("dimension")
        if want == "region" or (slots.get("regions") and len(slots["regions"]) >= 2):
            dim, members = "region", state.regions
            scope = Scope(state.brands, [], [], [], [])
        elif want == "brand":
            dim, members, scope = "brand", state.brands, Scope([], state.regions, [], [], [])
        elif len(slots.get("states", [])) >= 2 or (state.states and not slots.get("regions")):
            dim, members = "state", state.states
            scope = Scope(state.brands, [], [], [], [])
            scope.regions = sorted({self.state_region[s] for s in members})
        elif not state.brands:
            dim, members, scope = "brand", [], Scope([], state.regions, [], [], [])
        else:
            dim, members = "region", state.regions
            scope = Scope(state.brands, [], [], [], [])
        df = self.analytics.compare(metric, scope, period, dim, members)
        title = (f"{', '.join(state.brands) or 'All brands'} by {dim}, {period.label}"
                 f" (sorted by YoY)")
        text = nlg.table_markdown(df, dim, title)
        worst = df.iloc[0]
        text += f"\n\nWeakest: **{worst[dim]}** at {nlg.pct(worst['yoy'])} YoY."
        if assumed:
            text += "\n\n" + assumed
        sugg = [f"Why is {state.brands[0] if state.brands else worst[dim]} underperforming in "
                f"{worst[dim] if dim != 'brand' else ', '.join(state.regions)}?"]
        return text, {"type": "table", "dimension": dim, "data": _records(df)}, sugg

    def _h_drill_down(self, state: DialogueState, slots: dict[str, Any], res: NLUResult
                      ) -> tuple[str, dict[str, Any] | None, list[str]]:
        r = state.last_report
        new_scope = any(k in slots for k in ("brands", "regions", "states", "period"))
        dim = slots.get("dimension")
        if r is not None and not new_scope and dim in (None, "state", "territory"):
            if dim == "state" and not r.state:
                rows = r.states
                lines = [f"**{r.brand}, {r.region}, by state** ({r.period['label']}, gap vs "
                         "rest-of-India growth)", "", "| State | Actual | YoY | Gap vs peers |"
                         " Share of gap |", "|---|---|---|---|---|"]
                lines += [f"| {s['state']} | {nlg.inr(s['actual'])} | {nlg.pct(s['yoy'])} | "
                          f"{nlg.inr(s['gap_vs_peers'], True)} | "
                          f"{nlg.pct(s['share_of_gap'], False, 0)} |" for s in rows]
                return "\n".join(lines), {"type": "table", "dimension": "state", "data": rows}, [
                    "Which territories are worst?", "What should we do?"]
            rows = r.territories
            lines = [f"**Territories with the largest shortfall: {r.brand}, "
                     f"{r.state or r.region}** ({r.period['label']})", "",
                     "| Territory | State | Shortfall | Calls | Δ stock-out days | Retail reach |"
                     " Rx | Flags |", "|---|---|---|---|---|---|---|---|"]
            for t in rows:
                lines.append(
                    f"| {t['territory']} | {t['state']} | {nlg.inr(-t['shortfall'])} | "
                    f"{nlg.pct((t['calls_ratio'] or 1) - 1)} | {t['so_delta'] or 0:+.1f} | "
                    f"{nlg.pct((t['retail_ratio'] or 1) - 1)} | {nlg.pct((t['rx_ratio'] or 1) - 1)}"
                    f" | {t['flags'] or '-'} |")
            return "\n".join(lines), {"type": "table", "dimension": "territory", "data": rows}, [
                "Any field reports?", "What should we do?"]
        period, assumed = self._period(state)
        scope = self._scope(state)
        if dim is None:
            dim = "territory" if scope.states else "state" if scope.regions else "region"
        df = self.analytics.drill(scope, period, dim)
        key = "month" if dim == "month" else dim
        text = nlg.table_markdown(df, key, f"{', '.join(scope.brands) or 'All brands'} in "
                                           f"{scope.geo_label} by {dim}, {period.label}")
        if assumed:
            text += "\n\n" + assumed
        return text, {"type": "table", "dimension": dim, "data": _records(df)}, [
            "Why is it down?"]

    def _h_trend(self, state: DialogueState, slots: dict[str, Any], res: NLUResult
                 ) -> tuple[str, dict[str, Any] | None, list[str]]:
        metric = (slots.get("metrics") or ["net_sales"])[0]
        scope = self._scope(state)
        df = self.analytics.trend(metric, scope, state.period)
        base = self.analytics.sl.metrics.get(metric) or self.analytics.sl.metrics["net_sales"]
        last3 = df.tail(3)["value"].mean()
        prev3 = df.iloc[-6:-3]["value"].mean() if len(df) >= 6 else float("nan")
        yoy3 = df.tail(3)["value"].sum() / df.tail(3)["last_year"].sum() - 1
        direction = "up" if last3 > prev3 else "down"
        label = f"{', '.join(scope.brands) or 'All brands'} in {scope.geo_label}"
        text = (f"**{base.label}, {label}**, {df.month.iat[0]} to {df.month.iat[-1]}.\n\n"
                f"Last 3 months average {nlg.fmt_value(last3, base.unit)}, {direction} vs the "
                f"previous 3 ({nlg.pct(last3 / prev3 - 1)}), and {nlg.pct(yoy3)} vs the same months "
                f"last year. The chart shows this year against last year.")
        return text, {"type": "trend", "metric": metric, "unit": base.unit,
                      "data": _records(df)}, ["Why is it down?", "Compare regions"]

    def _h_search_notes(self, state: DialogueState, slots: dict[str, Any], res: NLUResult
                        ) -> tuple[str, dict[str, Any] | None, list[str]]:
        scope = self._scope(state)
        if slots.get("competitors"):
            scope.brands = []
        period = slots.get("period")
        cause = None
        r = state.last_report
        if r is not None and not any(k in slots for k in ("brands", "regions", "states",
                                                           "territories", "competitors")):
            # "any field reports?" right after a diagnosis: search that case's context.
            period = period or Period.from_keys(r.period["months"], r.period["label"])
            top = r.top_cause
            if top is not None:
                cause = top.cause
                if top.locations and top.cause not in ("price_cut",):
                    scope.states = [x for x in top.locations if x in self.state_region]
        if slots.get("competitors"):
            cause = "competitor_launch"
        for m in slots.get("metrics", []):
            cause = CAUSE_FROM_METRIC.get(m, cause)
        query = res.corrected + " " + " ".join(slots.get("competitors", []))
        hits = self.analytics.search_notes(query, scope, period, cause=cause, k=5)
        text = nlg.notes_markdown(hits, scope.geo_label)
        return text, {"type": "notes", "data": [{"text": h.doc.text, **h.doc.meta,
                                                 "score": h.score, "matched": h.matched}
                                                for h in hits]}, ["Why is it down?"]

    def _h_recommend(self, state: DialogueState, slots: dict[str, Any], res: NLUResult
                     ) -> tuple[str, dict[str, Any] | None, list[str]]:
        r = state.last_report
        fresh = any(k in slots for k in ("brands", "regions", "states"))
        if r is None or fresh:
            if not state.brands:
                state.pending = {"intent": "recommend", "missing": "brands"}
                return ("I need a diagnosis first. Which brand and region?", None,
                        ["Which brands are underperforming?"])
            self._h_diagnose(state, slots, res)
            r = state.last_report
        assert r is not None
        material = [c for c in r.causes if c.material]
        if not material:
            return (f"{r.brand} in {r.state or r.region} has no material root cause for "
                    f"{r.period['label']}, so I'd hold course and keep monitoring.", None,
                    ["Which brands are underperforming?"])
        lines = [f"**Action plan: {r.brand}, {r.state or r.region}** ({r.period['label']})", ""]
        for i, c in enumerate(material[:3], 1):
            where = f" in {', '.join(c.locations)}" if c.locations else ""
            lines.append(f"{i}. **{c.label}**{where} (≈ {nlg.inr(-c.impact)} at stake, "
                         f"{c.confidence_label.lower()} confidence): {c.recommendation}")
        lines += ["", "_Recommendations come from a playbook keyed by root cause; the impact "
                      "estimates rank them._"]
        return "\n".join(lines), {"type": "actions", "data": [asdict(c) for c in material[:3]]}, [
            "Break it down by territory", "Any field reports?"]

    # ------------------------------------------------------------------ conversational
    def _h_define_term(self, state: DialogueState, slots: dict[str, Any], res: NLUResult
                       ) -> tuple[str, dict[str, Any] | None, list[str]]:
        hits = self.glossary.lookup(res.corrected)
        if not hits or hits[0].score < 2.0:
            return ("I don't have that term in my glossary yet.", None, ["Help"])
        h = hits[0]
        text = f"**{h.doc.meta['term']}**: {h.doc.text}"
        if len(hits) > 1 and hits[1].score > 0.6 * h.score:
            text += f"\n\n_See also: {hits[1].doc.meta['term']}._"
        return text, {"type": "glossary", "data": {"term": h.doc.meta["term"],
                                                   "score": h.score}}, []

    def _h_help(self, state: DialogueState, slots: dict[str, Any], res: NLUResult
                ) -> tuple[str, dict[str, Any] | None, list[str]]:
        text = (
            "I'm **Sherlock Rx**. I find *why* a brand is under-performing in a region.\n\n"
            "**Things you can ask**\n"
            "- *Which brands are underperforming in the South this quarter?*\n"
            "- *Why is Cardivex down in North in Q2 FY27?* → root cause, evidence, actions\n"
            "- *Break it down by territory* · *What should we do?*\n"
            "- *Glucoryl market share in Rajasthan* · *Show the monthly trend of Respiron in East*\n"
            "- *Compare Painex across regions* · *Any field reports on Telmikind in Punjab?*\n"
            "- *What is DPCO?*\n\n"
            f"**Data:** {len(self.brands)} brands ({', '.join(self.brands)}), "
            f"{len(self.regions)} regions, Apr 2023 to {LATEST.label} (Indian fiscal year).")
        return text, None, ["Which brands are underperforming?", "Why is it down?",
                            "What is secondary sales?"]

    def _h_greet(self, state: DialogueState, slots: dict[str, Any], res: NLUResult
                 ) -> tuple[str, dict[str, Any] | None, list[str]]:
        return (nlg.pick("Hello! I'm Sherlock Rx. Ask me why a brand is under-performing in a "
                         "region, or which brands need attention.",
                         "Hi there. Which brand or region should we investigate today?"),
                None, ["Which brands are underperforming?", "Help"])

    def _h_thanks(self, state: DialogueState, slots: dict[str, Any], res: NLUResult
                  ) -> tuple[str, dict[str, Any] | None, list[str]]:
        return nlg.pick("Happy to help!", "Anytime. Anything else to investigate?",
                        "You're welcome."), None, []

    def _h_goodbye(self, state: DialogueState, slots: dict[str, Any], res: NLUResult
                   ) -> tuple[str, dict[str, Any] | None, list[str]]:
        return nlg.pick("Goodbye! Case closed for today. 🕵️", "See you. Good luck with the "
                        "review!"), None, []

    def _h_out_of_scope(self, state: DialogueState, slots: dict[str, Any], res: NLUResult
                        ) -> tuple[str, dict[str, Any] | None, list[str]]:
        return ("That's outside what I can help with. I focus on brand performance: sales, "
                "targets, market share, field force, supply and root causes.", None,
                ["Help", "Which brands are underperforming?"])
