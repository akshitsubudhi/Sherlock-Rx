"""Gazetteer NER + entity linking.

Dictionary-based recognition over token n-grams (1-4) with three match modes:

* exact        alias table lookup on the normalised token sequence
* fuzzy        Levenshtein ratio (rapidfuzz) for proper nouns only (brands, SKUs, states,
               towns, competitors), length-aware cut-offs, and never on phrases made only
               of ordinary English words ("thanks" must not become the town "Thane")
* case-exact   state codes ("UP", "TN") only when written in capitals in the raw text

Overlaps are resolved leftmost-longest, exact before fuzzy. The same alias tables power
`link()`, which normalises a span proposed by the statistical tagger to its canonical id.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from rapidfuzz import fuzz, process

from sherlock_rx.data.catalog import BRANDS, HQ_TOWNS, REGIONS, STATE_CODES
from sherlock_rx.nlp.lexicon import (
    BASELINE_ALIASES,
    DIMENSION_ALIASES,
    METRIC_ALIASES,
    REGION_ALIASES,
    STATE_ALIASES,
    THERAPY_ALIASES,
)
from sherlock_rx.nlp.text import normalize, tokenize

ENTITY_TYPES = ("BRAND", "SKU", "COMPETITOR", "REGION", "STATE", "TERRITORY", "THERAPY",
                "METRIC", "DIM", "BASELINE")
FUZZY_TYPES = {"BRAND", "SKU", "COMPETITOR", "STATE", "TERRITORY"}
MAX_NGRAM = 4


@dataclass(frozen=True)
class Span:
    start: int  # token index (inclusive)
    end: int  # token index (exclusive)
    char_start: int
    char_end: int
    type: str
    value: str
    score: float
    text: str
    source: str = "gazetteer"


def _key(alias: str) -> str:
    return " ".join(t.text for t in tokenize(normalize(alias)))


@dataclass
class Entry:
    type: str
    value: str


class Gazetteer:
    def __init__(self, entries: Iterable[tuple[str, str, str]], codes: dict[str, str],
                 common_words: Iterable[str] = ()):
        self.table: dict[str, list[Entry]] = defaultdict(list)
        self.by_type: dict[str, dict[str, str]] = defaultdict(dict)  # type -> alias key -> value
        for alias, etype, value in entries:
            k = _key(alias)
            if not k:
                continue
            if all(e.type != etype or e.value != value for e in self.table[k]):
                self.table[k].append(Entry(etype, value))
            self.by_type[etype][k] = value
        self.fuzzy_keys = sorted({k for t in FUZZY_TYPES for k in self.by_type[t] if len(k) >= 5})
        self.codes = codes  # e.g. {"UP": "Uttar Pradesh"}
        self.code_re = re.compile(r"\b(" + "|".join(map(re.escape, codes)) + r")\b") if codes else None
        self.common = {w.lower() for w in common_words}

    # ------------------------------------------------------------------ build
    @classmethod
    def build(cls, territories: Iterable[tuple[str, str]] | None = None,
              competitors: Iterable[str] = (), common_words: Iterable[str] = ()) -> Gazetteer:
        """territories: (territory name, state) pairs; defaults to the catalog HQ towns."""
        e: list[tuple[str, str, str]] = []
        for b in BRANDS:
            e.append((b.name, "BRAND", b.name))
            e += [(a, "BRAND", b.name) for a in b.aliases]
            for s in b.skus:
                e.append((s.name, "SKU", s.name))
            e += [(c, "COMPETITOR", c) for c in b.competitors]
        e += [(c, "COMPETITOR", c) for c in competitors]
        for region, aliases in REGION_ALIASES.items():
            e += [(a, "REGION", region) for a in aliases]
        for states in REGIONS.values():
            e += [(s, "STATE", s) for s in states]
        for state, aliases in STATE_ALIASES.items():
            e += [(a, "STATE", state) for a in aliases]
        if territories is None:
            territories = [(t, s) for s, towns in HQ_TOWNS.items() for t in towns]
        e += [(t, "TERRITORY", t) for t, _ in territories]
        for table, etype in ((THERAPY_ALIASES, "THERAPY"), (METRIC_ALIASES, "METRIC"),
                             (DIMENSION_ALIASES, "DIM"), (BASELINE_ALIASES, "BASELINE")):
            for value, aliases in table.items():
                e += [(a, etype, value) for a in aliases]
        codes = {code: state for state, code in STATE_CODES.items()}
        return cls(e, codes, common_words)

    def values(self, etype: str) -> list[str]:
        return sorted(set(self.by_type[etype].values()))

    # ------------------------------------------------------------------ recognise
    def _fuzzy_cutoff(self, n_chars: int) -> float:
        return 88.0 if n_chars < 7 else 85.0

    def find(self, raw: str, types: set[str] | None = None) -> list[Span]:
        norm = normalize(raw)
        toks = tokenize(norm)
        cands: list[tuple[int, int, Span]] = []  # (exact flag, n tokens, span)
        for i in range(len(toks)):
            for n in range(1, MAX_NGRAM + 1):
                j = i + n
                if j > len(toks):
                    break
                phrase = " ".join(t.text for t in toks[i:j])
                cs, ce = toks[i].start, toks[j - 1].end
                for ent in self.table.get(phrase, []):
                    if types and ent.type not in types:
                        continue
                    cands.append((1, n, Span(i, j, cs, ce, ent.type, ent.value, 100.0, phrase)))
                if phrase in self.table or len(phrase) < 5 or not phrase.replace(" ", "").isalpha():
                    continue
                if all(t.text in self.common for t in toks[i:j]):
                    continue
                hit = process.extractOne(phrase, self.fuzzy_keys, scorer=fuzz.ratio,
                                         score_cutoff=self._fuzzy_cutoff(len(phrase)))
                if hit:
                    for ent in self.table[hit[0]]:
                        if ent.type in FUZZY_TYPES and (not types or ent.type in types):
                            cands.append((0, n, Span(i, j, cs, ce, ent.type, ent.value,
                                                     float(hit[1]), phrase, "fuzzy")))
        if self.code_re is not None:
            for m in self.code_re.finditer(raw):
                # map raw offsets onto normalised tokens (normalisation preserves word order)
                low = m.group().lower()
                for ti, t in enumerate(toks):
                    if t.text == low:
                        cands.append((1, 1, Span(ti, ti + 1, t.start, t.end, "STATE",
                                                 self.codes[m.group()], 100.0, t.text, "code")))
                        break
        cands.sort(key=lambda c: (-c[0], -c[1], -c[2].score, c[2].start))
        chosen: list[Span] = []
        for _, _, sp in cands:
            if any(sp.start < o.end and o.start < sp.end for o in chosen):
                continue
            chosen.append(sp)
        return sorted(chosen, key=lambda s: s.start)

    # ------------------------------------------------------------------ link
    def link(self, etype: str, surface: str, cutoff: float = 72.0) -> tuple[str, float] | None:
        """Map a free-form surface string to the canonical value of `etype`."""
        table = self.by_type.get(etype)
        if not table:
            return None
        k = _key(surface)
        if k in table:
            return table[k], 100.0
        # prefix rule: "gluco" -> "glucoryl", "cardi" -> "cardivex"
        if len(k) >= 4:
            pref = [a for a in table if a.startswith(k)]
            if len({table[a] for a in pref}) == 1:
                return table[pref[0]], 90.0
        hit = process.extractOne(k, list(table), scorer=fuzz.WRatio, score_cutoff=cutoff)
        return (table[hit[0]], float(hit[1])) if hit else None
