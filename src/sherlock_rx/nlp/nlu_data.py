"""Synthetic, BIO-annotated NLU corpus from templates, with noise augmentation.

Augmentations (each applied with a small probability, seeded):
  * keyboard typos   adjacent-key substitution, deletion, transposition, doubling
  * slot typos       brand/state/town names misspelt the way field teams type them
  * truncation       "gluco", "cardi" (teaches the tagger to use context + char features)
  * fillers          "hey sherlock ,", "quick question :", trailing "please" / "?"
  * casing           Title, UPPER, lower
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from sherlock_rx.data.catalog import BRANDS, HQ_TOWNS, REGIONS
from sherlock_rx.nlp.gazetteer import Gazetteer
from sherlock_rx.nlp.lexicon import (
    BASELINE_ALIASES,
    DIMENSION_ALIASES,
    METRIC_ALIASES,
    REGION_ALIASES,
    STATE_ALIASES,
    THERAPY_ALIASES,
)
from sherlock_rx.nlp.templates import (
    FILLER_PREFIX,
    FILLER_SUFFIX,
    SMALLTALK,
    SYNONYMS,
    TEMPLATES,
    TERMS,
    TIME_FORMS,
)
from sherlock_rx.nlp.text import normalize, tokenize

KEYBOARD = {
    "a": "qwsz", "b": "vghn", "c": "xdfv", "d": "serfcx", "e": "wsdr", "f": "drtgvc",
    "g": "ftyhbv", "h": "gyujnb", "i": "ujko", "j": "huikmn", "k": "jiolm", "l": "kop",
    "m": "njk", "n": "bhjm", "o": "iklp", "p": "ol", "q": "wa", "r": "edft", "s": "awedxz",
    "t": "rfgy", "u": "yhji", "v": "cfgb", "w": "qase", "x": "zsdc", "y": "tghu", "z": "asx",
}
SLOT_RE = re.compile(r"\{([A-Z]+)\}")
OPT_RE = re.compile(r"\[([^\]]+)\]")
TIME_PREP = re.compile(r"^(in the|in|for the|for|during|over the|over)\s+")
CLOSED_TYPES = {"METRIC", "DIM", "BASELINE", "THERAPY"}


@dataclass
class Example:
    text: str
    tokens: list[str]
    tags: list[str]
    intent: str
    template_id: str
    entities: dict[str, list[str]] = field(default_factory=dict)


def surface_forms() -> dict[str, list[str]]:
    forms: dict[str, list[str]] = {
        "BRAND": [b.name for b in BRANDS] + [a for b in BRANDS for a in b.aliases],
        "SKU": [s.name for b in BRANDS for s in b.skus],
        "COMPETITOR": [c for b in BRANDS for c in b.competitors],
        "REGION": [a for aliases in REGION_ALIASES.values() for a in aliases],
        "STATE": [s for ss in REGIONS.values() for s in ss]
        + [a for aliases in STATE_ALIASES.values() for a in aliases],
        "TERRITORY": [t for towns in HQ_TOWNS.values() for t in towns],
        "THERAPY": [a for aliases in THERAPY_ALIASES.values() for a in aliases],
        "METRIC": [a for aliases in METRIC_ALIASES.values() for a in aliases],
        "DIM": [a for aliases in DIMENSION_ALIASES.values() for a in aliases],
        "BASELINE": [a for aliases in BASELINE_ALIASES.values() for a in aliases],
        "TIME": list(TIME_FORMS),
        "TERM": list(TERMS),
    }
    return forms


def canonical_value(slot: str, surface: str) -> str:
    s = surface.lower()
    if slot == "BRAND":
        for b in BRANDS:
            if s == b.name.lower() or s in b.aliases:
                return b.name
    for slot_name, table in (("REGION", REGION_ALIASES), ("METRIC", METRIC_ALIASES),
                             ("DIM", DIMENSION_ALIASES), ("BASELINE", BASELINE_ALIASES),
                             ("THERAPY", THERAPY_ALIASES), ("STATE", STATE_ALIASES)):
        if slot == slot_name:
            for value, aliases in table.items():
                if s in aliases:
                    return value
    return surface


def typo(word: str, rng: np.random.Generator) -> str:
    if len(word) < 4 or not word.isalpha():
        return word
    i = int(rng.integers(1, len(word) - 1))
    op = rng.integers(4)
    if op == 0:
        c = word[i].lower()
        return word[:i] + str(rng.choice(list(KEYBOARD.get(c, c)))) + word[i + 1:]
    if op == 1:
        return word[:i] + word[i + 1:]
    if op == 2:
        return word[:i - 1] + word[i] + word[i - 1] + word[i + 1:]
    return word[:i] + word[i] + word[i:]


class CorpusBuilder:
    def __init__(self, seed: int = 13):
        self.rng = np.random.default_rng(seed)
        self.forms = surface_forms()
        self.gaz = Gazetteer.build()

    def _fill(self, slot: str) -> tuple[str, str]:
        rng = self.rng
        surface = str(rng.choice(self.forms[slot]))
        value = canonical_value(slot, surface)
        if slot in ("BRAND", "STATE", "TERRITORY", "COMPETITOR") and rng.random() < 0.15:
            surface = " ".join(typo(w, rng) for w in surface.split())
        if slot == "BRAND" and rng.random() < 0.08 and len(surface) > 6:
            surface = surface[: int(rng.integers(4, 6))]
        r = rng.random()
        if r < 0.25:
            surface = surface.title()
        elif r < 0.30:
            surface = surface.upper() if slot != "TIME" else surface
        return surface, value

    def _synonyms(self, text: str) -> str:
        """EDA synonym replacement: swap one cue phrase for a same-meaning alternative."""
        for group in self.rng.permutation(len(SYNONYMS)):
            words = SYNONYMS[int(group)]
            hits = [w for w in words if re.search(rf"\b{re.escape(w)}\b", text)]
            if hits:
                src = hits[0]
                dst = str(self.rng.choice([w for w in words if w != src]))
                return re.sub(rf"\b{re.escape(src)}\b", dst, text, count=1)
        return text

    def render(self, template: str, intent: str, template_id: str) -> Example | None:
        rng = self.rng
        tpl = OPT_RE.sub(lambda m: m.group(1) if rng.random() < 0.6 else "", template)
        if rng.random() < 0.4:
            tpl = self._synonyms(tpl)
        pieces: list[tuple[str, str | None, str | None]] = []  # (text, slot, value)
        pos = 0
        for m in SLOT_RE.finditer(tpl):
            if m.start() > pos:
                pieces.append((tpl[pos:m.start()], None, None))
            slot = m.group(1)
            surface, value = self._fill(slot)
            if slot == "TIME":
                pm = TIME_PREP.match(surface.lower())
                if pm:
                    pieces.append((surface[:pm.end()], None, None))
                    surface = surface[pm.end():]
            pieces.append((surface, None if slot == "TERM" else slot, value))
            pos = m.end()
        if pos < len(tpl):
            pieces.append((tpl[pos:], None, None))

        # Noise on template words.
        noisy: list[tuple[str, str | None, str | None]] = []
        for text, pslot, pvalue in pieces:
            if pslot is None and rng.random() < 0.12:
                ws = text.split()
                if ws:
                    k = int(rng.integers(len(ws)))
                    ws[k] = typo(ws[k], rng)
                    text = " ".join(ws)
            noisy.append((text, pslot, pvalue))
        if intent not in SMALLTALK:  # "ok hey please" is not a greeting; keep small talk clean
            if rng.random() < 0.2:
                noisy.insert(0, (str(rng.choice(FILLER_PREFIX)), None, None))
            if rng.random() < 0.25:
                noisy.append((str(rng.choice(FILLER_SUFFIX)), None, None))

        tokens: list[str] = []
        tags: list[str] = []
        entities: dict[str, list[str]] = {}
        for text, pslot, pvalue in noisy:
            toks = [t.text for t in tokenize(normalize(text))]
            if not toks:
                continue
            tokens += toks
            if pslot:
                tags += [f"B-{pslot}"] + [f"I-{pslot}"] * (len(toks) - 1)
                entities.setdefault(pslot, []).append(str(pvalue))
                continue
            # Template words that are closed-class entity mentions ("sales", "by state",
            # "vs target") get the same label they would get as a slot fill.
            piece_tags = ["O"] * len(toks)
            for sp in self.gaz.find(text, types=CLOSED_TYPES):
                piece_tags[sp.start] = f"B-{sp.type}"
                for k in range(sp.start + 1, sp.end):
                    piece_tags[k] = f"I-{sp.type}"
                entities.setdefault(sp.type, []).append(sp.value)
            tags += piece_tags
        raw = " ".join(t for t, _, _ in noisy)
        raw = re.sub(r"\s+([?,:])", r"\1", re.sub(r"\s+", " ", raw)).strip()
        full = [t.text for t in tokenize(normalize(raw))]
        if full != tokens or not tokens:
            return None  # normalisation merged/split across pieces; skip rather than mislabel
        return Example(raw, tokens, tags, intent, template_id, entities)

    def build(self, per_intent: int = 450, test_frac: float = 0.25, val_frac: float = 0.15
              ) -> dict[str, list[Example]]:
        splits: dict[str, list[Example]] = {"train": [], "val": [], "test": []}
        for intent, templates in TEMPLATES.items():
            ids = [f"{intent}:{i}" for i in range(len(templates))]
            order = self.rng.permutation(len(templates))
            n_test = max(2, round(test_frac * len(templates)))
            n_val = max(1, round(val_frac * len(templates)))
            assign = {}
            for rank, idx in enumerate(order):
                assign[idx] = "test" if rank < n_test else "val" if rank < n_test + n_val else "train"
            for split in ("train", "val", "test"):
                idxs = [i for i in range(len(templates)) if assign[i] == split]
                target = per_intent if split == "train" else max(per_intent // 4, 60)
                per_t = max(2, target // max(len(idxs), 1))
                for i in idxs:
                    seen: set[str] = set()
                    for _ in range(per_t * 3):
                        if len(seen) >= per_t:
                            break
                        ex = self.render(templates[i], intent, ids[i])
                        if ex is None or " ".join(ex.tokens) in seen:
                            continue
                        seen.add(" ".join(ex.tokens))
                        splits[split].append(ex)
        for examples in splits.values():
            self.rng.shuffle(examples)
        return splits
