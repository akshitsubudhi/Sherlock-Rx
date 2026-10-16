"""Okapi BM25, from scratch, over field notes and the glossary.

    score(q, d) = Σ_t∈q  idf(t) · tf(t,d)·(k1+1) / (tf(t,d) + k1·(1 − b + b·|d|/avgdl))
    idf(t)      = ln(1 + (N − df(t) + 0.5) / (df(t) + 0.5))

Inverted index (term -> postings) so a query only touches documents containing its terms.
Optional metadata filters (brand, state, months) are applied before ranking, and query
expansion adds cause-specific vocabulary when the chatbot looks for corroborating notes
("stock-out" -> shortage, supply, depot, C&F ...).
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from sherlock_rx.nlp.text import content_terms

GLOSSARY_PATH = Path(__file__).with_name("glossary.yaml")

EXPANSIONS: dict[str, tuple[str, ...]] = {
    "field_force_vacancy": ("vacant", "resigned", "attrition", "hiring", "coverage", "mr", "left"),
    "stockout": ("stock", "shortage", "supply", "depot", "available", "out", "c&f", "plant"),
    "competitor_launch": ("competitor", "launched", "scheme", "price", "entrant", "lower", "ptr"),
    "price_cut": ("nppa", "ceiling", "price", "dpco", "mrp", "revised"),
    "market_slowdown": ("market", "slow", "footfall", "season", "demand", "weak", "patients"),
    "distribution_loss": ("stockist", "distributor", "retailers", "boycott", "dispute", "reach"),
}


@dataclass
class Doc:
    doc_id: str
    text: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Hit:
    doc: Doc
    score: float
    matched: list[str]


class BM25:
    def __init__(self, docs: Sequence[Doc], k1: float = 1.5, b: float = 0.75,
                 index_text: Callable[[Doc], str] | None = None):
        self.docs = list(docs)
        self.k1, self.b = k1, b
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.lengths: list[int] = []
        for i, d in enumerate(self.docs):
            terms = content_terms(index_text(d) if index_text else d.text)
            self.lengths.append(len(terms))
            for t, c in Counter(terms).items():
                self.postings[t].append((i, c))
        n = len(self.docs)
        self.avgdl = sum(self.lengths) / max(n, 1)
        self.idf = {t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5))
                    for t, p in self.postings.items()}

    def search(self, query: str, k: int = 5, where: Callable[[Doc], bool] | None = None,
               expand: Sequence[str] = ()) -> list[Hit]:
        q = content_terms(query) + [t for w in expand for t in content_terms(w)]
        scores: dict[int, float] = defaultdict(float)
        matched: dict[int, set[str]] = defaultdict(set)
        for t in set(q):
            weight = q.count(t)
            for i, tf in self.postings.get(t, []):
                if where is not None and not where(self.docs[i]):
                    continue
                dl = self.lengths[i]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[i] += weight * self.idf[t] * tf * (self.k1 + 1) / denom
                matched[i].add(t)
        top = sorted(scores.items(), key=lambda kv: -kv[1])[:k]
        return [Hit(self.docs[i], round(s, 3), sorted(matched[i])) for i, s in top]


def load_glossary(path: Path = GLOSSARY_PATH) -> list[Doc]:
    items = yaml.safe_load(path.read_text())
    return [Doc(f"G{i:02d}", it["definition"], {"term": it["term"], "aliases": it["aliases"]})
            for i, it in enumerate(items)]


class Glossary:
    """Term lookup: BM25 over term name and aliases (boosted x3) plus the definition."""

    def __init__(self, path: Path = GLOSSARY_PATH):
        self.docs = load_glossary(path)

        def text(d: Doc) -> str:
            names = " ".join([d.meta["term"], *d.meta["aliases"]])
            return f"{names} {names} {names} {d.text}"

        self.index = BM25(self.docs, index_text=text)

    def lookup(self, query: str, k: int = 3) -> list[Hit]:
        return self.index.search(query, k=k)
