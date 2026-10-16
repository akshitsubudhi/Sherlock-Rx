from __future__ import annotations

from pathlib import Path

from sherlock_rx.eval.dialogue_eval import evaluate as dialogue_eval
from sherlock_rx.eval.report import nlu_handwritten

ROOT = Path(__file__).resolve().parents[1]


def test_gate_nlu_handwritten(models_dir: Path) -> None:
    """Acceptance gate on the frozen, never-tuned-on v2 set: intent >= 90%, slots >= 95%."""
    r = nlu_handwritten(models_dir, files=("nlu_handwritten_v2.yaml",))["nlu_handwritten_v2.yaml"]
    assert r["intent_accuracy"] >= 0.90, r["failures"]
    assert r["slot_accuracy"] >= 0.95, r["failures"]


def test_gate_dialogue(bot) -> None:  # type: ignore[no-untyped-def]
    r = dialogue_eval(bot)
    assert r.success_rate >= 0.9, r.failures


def test_nlu_trace_is_complete(nlu) -> None:  # type: ignore[no-untyped-def]
    d = nlu.parse("why is cardivx down in UP in q1", explain=True).debug()
    assert d["intent"] == "diagnose"
    assert d["slots"]["brands"] == ["Cardivex"] and d["slots"]["states"] == ["Uttar Pradesh"]
    assert d["slots"]["period"] == "Q1 FY27"
    assert "entbrand" in d["masked"]
    assert {e["source"] for e in d["entities"]} <= {"gazetteer", "fuzzy", "code", "tagger",
                                                     "temporal"}


def test_tagger_recovers_truncated_brand(nlu) -> None:  # type: ignore[no-untyped-def]
    r = nlu.parse("how is gluco doing in the south")
    assert r.slots.get("brands") == ["Glucoryl"]
    assert any(s.source == "tagger" for s in r.spans)


def test_low_confidence_asks_instead_of_guessing(bot) -> None:  # type: ignore[no-untyped-def]
    r = bot.handle("zzzz qqq", bot.new_state())
    assert r.policy == "clarify"


def test_out_of_scope_does_not_pollute_state(bot) -> None:  # type: ignore[no-untyped-def]
    st = bot.new_state()
    bot.handle("why is Painex down in West", st)
    bot.handle("what's the weather in Delhi?", st)
    assert st.regions == ["West"] and st.states == []


def test_every_number_in_rca_reply_comes_from_the_report(bot) -> None:  # type: ignore[no-untyped-def]
    """Faithfulness: the reply's headline numbers are the engine's numbers, formatted."""
    from sherlock_rx.dialogue import nlg

    st = bot.new_state()
    r = bot.handle("why is Cardivex underperforming in North in Q2 FY27", st)
    h = st.last_report.headline
    assert nlg.inr(h.actual) in r.text and nlg.inr(h.target) in r.text
    assert nlg.pct(h.yoy) in r.text


def test_indian_number_formatting() -> None:
    from sherlock_rx.dialogue.nlg import indian_group, inr

    assert indian_group(12345678) == "1,23,45,678"
    assert inr(2.5e7) == "₹2.50 Cr" and inr(-3.4e5) == "-₹3.4 L" and inr(950) == "₹950"
