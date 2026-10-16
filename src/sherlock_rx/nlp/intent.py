"""Intent classification models.

* KeywordBaseline       hand-written cue-word rules: the floor every model must beat
* NaiveBayesIntent      from-scratch Multinomial NB over word + char n-gram counts
* LinearIntent          from-scratch TF-IDF + scikit-learn LogisticRegression / LinearSVC

All models consume the *entity-masked* utterance ("why is entbrand down in entregion"),
so they learn the shape of a request rather than memorising brand or state names.
"""

from __future__ import annotations

import re
from typing import Protocol

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sherlock_rx.nlp.naive_bayes import MultinomialNB
from sherlock_rx.nlp.vectorize import CountVectorizer, TfidfVectorizer


class IntentModel(Protocol):
    name: str
    classes_: np.ndarray

    def fit(self, texts: list[str], labels: list[str]) -> IntentModel: ...

    def predict_proba(self, texts: list[str]) -> np.ndarray: ...


KEYWORD_RULES: list[tuple[str, str]] = [
    (r"\b(hi|hello|hey|namaste|good (morning|evening|afternoon))\b$", "greet"),
    (r"\b(thanks|thank you|appreciated|dhanyavaad)\b", "thanks"),
    (r"\b(bye|goodbye|exit|quit|see you)\b", "goodbye"),
    (r"\b(help|what can you do|capabilit|examples)\b", "help"),
    (r"\b(what is|what does|define|meaning|mean by)\b(?!.*ent(brand|region|state)\b)", "define_term"),
    (r"\b(why|root cause|reason|what went wrong|diagnose|explain the)\b", "diagnose"),
    (r"\b(which|underperform|worst|weakest|below target|missed)\b", "find_underperformers"),
    (r"\b(compare|versus|vs|benchmark|side by side)\b", "compare"),
    (r"\b(break|drill|split|zoom|deeper|which (states|territories|hqs))\b", "drill_down"),
    (r"\b(trend|over time|month on month|trajectory|plot|chart|graph)\b", "trend"),
    (r"\b(field|notes|reports|feedback|intel|saying)\b", "search_notes"),
    (r"\b(should we|recommend|action|next steps|fix|improve)\b", "recommend"),
    (r"\bent(brand|metric)\b", "metric_lookup"),
]


class KeywordBaseline:
    name = "keyword_rules"

    def fit(self, texts: list[str], labels: list[str]) -> KeywordBaseline:
        self.classes_ = np.array(sorted(set(labels)))
        return self

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), len(self.classes_)))
        idx = {c: i for i, c in enumerate(self.classes_)}
        for r, t in enumerate(texts):
            label = "out_of_scope"
            for pattern, intent in KEYWORD_RULES:
                if re.search(pattern, t):
                    label = intent
                    break
            out[r, idx.get(label, 0)] = 1.0
        return out


class NaiveBayesIntent:
    name = "naive_bayes"

    def __init__(self, alpha: float = 0.3):
        self.vec = CountVectorizer(word_range=(1, 2), char_range=(3, 5))
        self.nb = MultinomialNB(alpha=alpha)

    def fit(self, texts: list[str], labels: list[str]) -> NaiveBayesIntent:
        self.nb.fit(self.vec.fit_transform(texts), labels)
        self.classes_ = self.nb.classes_
        return self

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return self.nb.predict_proba(self.vec.transform(texts))


class LinearIntent:
    def __init__(self, kind: str = "logreg", c: float = 8.0):
        self.kind = kind
        self.name = {"logreg": "tfidf_logreg", "svm": "tfidf_linear_svm"}[kind]
        self.vec = TfidfVectorizer(word_range=(1, 2), char_range=(3, 5), min_df=2)
        if kind == "logreg":
            self.clf = LogisticRegression(C=c, max_iter=2000)
        else:
            self.clf = CalibratedClassifierCV(LinearSVC(C=c / 8), cv=3)

    def fit(self, texts: list[str], labels: list[str]) -> LinearIntent:
        self.clf.fit(self.vec.fit_transform(texts), labels)
        self.classes_ = self.clf.classes_
        return self

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.clf.predict_proba(self.vec.transform(texts)))

    def explain(self, text: str, label: str, k: int = 6) -> list[tuple[str, float]]:
        """Top features pushing `text` towards `label` (coefficient x tf-idf weight)."""
        if self.kind != "logreg":
            return []
        x = self.vec.transform([text]).tocoo()
        ci = list(self.classes_).index(label)
        coef = self.clf.coef_[ci]
        inv = {i: t for t, i in self.vec.vocabulary_.items()}
        contrib = sorted(((inv[int(j)], float(v * coef[j])) for j, v in zip(x.col, x.data,
                                                                         strict=True)),
                         key=lambda p: -p[1])
        return [(t, round(c, 3)) for t, c in contrib[:k]]
