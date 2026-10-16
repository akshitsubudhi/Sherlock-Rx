"""The NLU pipeline: one call from raw text to (intent, confidence, slots, debug trace).

    raw text ─► normalise ─► tokenise ─► spell-correct OOV words
             ─► entities: gazetteer + perceptron tagger + linker + temporal parser
             ─► mask entities ("why is entbrand down in entregion")
             ─► TF-IDF ─► intent classifier ─► confidence / out-of-scope gate
"""

from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from sherlock_rx.nlp.gazetteer import Gazetteer, Span
from sherlock_rx.nlp.intent import IntentModel, LinearIntent
from sherlock_rx.nlp.ner import EntityExtractor
from sherlock_rx.nlp.spell import SpellCorrector
from sherlock_rx.nlp.tagger import PerceptronTagger
from sherlock_rx.nlp.text import normalize, tokenize

MODEL_DIR = Path(__file__).resolve().parents[3] / "models"


@dataclass
class NLUResult:
    text: str
    corrected: str
    corrections: list[tuple[str, str]]
    tokens: list[str]
    masked: str
    intent: str
    confidence: float
    top: list[tuple[str, float]]
    slots: dict[str, Any]
    spans: list[Span]
    uncertain: bool
    explanation: list[tuple[str, float]] = field(default_factory=list)

    def debug(self) -> dict[str, Any]:
        slots = {k: (v.label if hasattr(v, "label") else v) for k, v in self.slots.items()}
        return {
            "text": self.text, "corrected": self.corrected, "corrections": self.corrections,
            "tokens": self.tokens, "masked": self.masked, "intent": self.intent,
            "confidence": round(self.confidence, 3),
            "top": [(i, round(p, 3)) for i, p in self.top], "uncertain": self.uncertain,
            "slots": slots,
            "entities": [{"text": s.text, "type": s.type, "value": s.value,
                          "score": round(s.score, 1), "source": s.source,
                          "start": s.char_start, "end": s.char_end} for s in self.spans],
            "explanation": self.explanation,
        }


class NLU:
    def __init__(self, spell: SpellCorrector, extractor: EntityExtractor, model: IntentModel,
                 threshold: float = 0.45, meta: dict[str, Any] | None = None):
        self.spell = spell
        self.extractor = extractor
        self.model = model
        self.threshold = threshold
        self.meta = meta or {}

    def correct(self, text: str) -> tuple[str, list[tuple[str, str]]]:
        toks = [t.text for t in tokenize(normalize(text))]
        _, changes = self.spell.correct(toks)
        fixed = text
        for wrong, right in changes:
            fixed = re.sub(rf"\b{re.escape(wrong)}\b", right, fixed, flags=re.IGNORECASE)
        return fixed, changes

    def parse(self, text: str, explain: bool = False) -> NLUResult:
        corrected, changes = self.correct(text)
        ex = self.extractor.extract(corrected)
        masked = ex.masked()
        probs = self.model.predict_proba([masked])[0]
        order = np.argsort(-probs)
        classes = list(self.model.classes_)
        top = [(classes[i], float(probs[i])) for i in order[:3]]
        intent, conf = top[0]
        res = NLUResult(text, corrected, changes, ex.tokens, masked, intent, conf, top,
                        ex.slots, ex.spans, uncertain=conf < self.threshold)
        if explain and isinstance(self.model, LinearIntent):
            res.explanation = self.model.explain(masked, intent)
        return res

    # ------------------------------------------------------------------ io
    def save(self, out_dir: str | Path) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        if self.extractor.tagger is not None:
            self.extractor.tagger.save(out / "tagger.pkl")
        with open(out / "intent.pkl", "wb") as fh:
            pickle.dump(self.model, fh)
        with open(out / "lexical.pkl", "wb") as fh:
            pickle.dump({"spell": self.spell, "gazetteer": self.extractor.gaz,
                         "threshold": self.threshold}, fh)
        (out / "model_card.json").write_text(json.dumps(self.meta, indent=2, default=str))

    @classmethod
    def load(cls, model_dir: str | Path = MODEL_DIR,
             territory_state: dict[str, str] | None = None,
             gazetteer: Gazetteer | None = None) -> NLU:
        d = Path(model_dir)
        if not (d / "intent.pkl").exists():
            raise FileNotFoundError(f"No trained NLU models in {d}. Run `sherlock train` first.")
        with open(d / "lexical.pkl", "rb") as fh:
            lex = pickle.load(fh)
        with open(d / "intent.pkl", "rb") as fh:
            model = pickle.load(fh)
        tagger = PerceptronTagger.load(d / "tagger.pkl")
        meta = json.loads((d / "model_card.json").read_text())
        extractor = EntityExtractor(gazetteer or lex["gazetteer"], tagger,
                                    territory_state=territory_state)
        return cls(lex["spell"], extractor, model, lex["threshold"], meta)
