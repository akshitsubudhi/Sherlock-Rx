"""Bag-of-words and TF-IDF vectorisers implemented from first principles.

    tf(t, d)   = raw count of term t in document d (optionally 1 + log tf, "sublinear")
    idf(t)     = ln((1 + N) / (1 + df(t))) + 1       (smoothed, as in scikit-learn)
    tfidf(t,d) = tf(t, d) * idf(t), then each row is L2-normalised

Features are word n-grams and/or character n-grams inside word boundaries ("char_wb"),
which make the intent model robust to typos ("undreperforming" still shares most of
its 3-5-grams with "underperforming"). Output is a scipy CSR matrix so any linear model
(our Naive Bayes, scikit-learn's LogisticRegression / LinearSVC) can consume it.
`tests/test_vectorize.py` checks this matches sklearn's TfidfVectorizer to 1e-9.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Sequence

import numpy as np
import scipy.sparse as sp

from sherlock_rx.nlp.text import words


def word_ngrams(tokens: Sequence[str], n_min: int, n_max: int) -> list[str]:
    out: list[str] = []
    for n in range(n_min, n_max + 1):
        out.extend(" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1))
    return out


def char_wb_ngrams(tokens: Sequence[str], n_min: int, n_max: int) -> list[str]:
    out: list[str] = []
    for tok in tokens:
        w = f" {tok} "
        for n in range(n_min, n_max + 1):
            if len(w) < n:
                continue
            out.extend(w[i:i + n] for i in range(len(w) - n + 1))
    return out


class CountVectorizer:
    def __init__(self, word_range: tuple[int, int] | None = (1, 2),
                 char_range: tuple[int, int] | None = None, min_df: int = 1):
        self.word_range = word_range
        self.char_range = char_range
        self.min_df = min_df
        self.vocabulary_: dict[str, int] = {}

    def analyze(self, doc: str) -> list[str]:
        toks = words(doc)
        feats: list[str] = []
        if self.word_range:
            feats += ["w:" + g for g in word_ngrams(toks, *self.word_range)]
        if self.char_range:
            feats += ["c:" + g for g in char_wb_ngrams(toks, *self.char_range)]
        return feats

    def fit(self, docs: Iterable[str]) -> CountVectorizer:
        df: Counter[str] = Counter()
        for d in docs:
            df.update(set(self.analyze(d)))
        kept = sorted(t for t, c in df.items() if c >= self.min_df)
        self.vocabulary_ = {t: i for i, t in enumerate(kept)}
        return self

    def transform(self, docs: Iterable[str]) -> sp.csr_matrix:
        indptr, indices, data = [0], [], []
        for d in docs:
            counts = Counter(f for f in self.analyze(d) if f in self.vocabulary_)
            for f, c in sorted(counts.items(), key=lambda kv: self.vocabulary_[kv[0]]):
                indices.append(self.vocabulary_[f])
                data.append(float(c))
            indptr.append(len(indices))
        return sp.csr_matrix((data, indices, indptr), shape=(len(indptr) - 1, len(self.vocabulary_)))

    def fit_transform(self, docs: Sequence[str]) -> sp.csr_matrix:
        return self.fit(docs).transform(docs)


class TfidfVectorizer(CountVectorizer):
    def __init__(self, word_range: tuple[int, int] | None = (1, 2),
                 char_range: tuple[int, int] | None = (3, 5), min_df: int = 1,
                 sublinear_tf: bool = True):
        super().__init__(word_range, char_range, min_df)
        self.sublinear_tf = sublinear_tf
        self.idf_: np.ndarray = np.zeros(0)

    def fit(self, docs: Iterable[str]) -> TfidfVectorizer:
        docs = list(docs)
        super().fit(docs)
        counts = super().transform(docs)
        df = np.bincount(counts.indices, minlength=len(self.vocabulary_))
        n = counts.shape[0]
        self.idf_ = np.log((1 + n) / (1 + df)) + 1.0
        return self

    def transform(self, docs: Iterable[str]) -> sp.csr_matrix:
        x = super().transform(docs).astype(np.float64)
        if self.sublinear_tf:
            x.data = 1.0 + np.log(x.data)
        x = x @ sp.diags(self.idf_)
        norms = np.sqrt(np.asarray(x.multiply(x).sum(axis=1)).ravel())
        norms[norms == 0] = 1.0
        return sp.csr_matrix(sp.diags(1.0 / norms) @ x)

    def top_terms(self, row: sp.csr_matrix, k: int = 8) -> list[tuple[str, float]]:
        inv = {i: t for t, i in self.vocabulary_.items()}
        r = row.tocoo()
        pairs = sorted(zip(r.col, r.data, strict=True), key=lambda p: -p[1])[:k]
        return [(inv[int(c)], round(float(v), 3)) for c, v in pairs]


def cosine(a: sp.csr_matrix, b: sp.csr_matrix) -> np.ndarray:
    """Cosine similarity between L2-normalised rows (a dot product)."""
    return np.asarray((a @ b.T).todense())


def entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / math.log(max(len(p), 2)))
