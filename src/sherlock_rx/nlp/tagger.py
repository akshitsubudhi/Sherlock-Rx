"""Structured (averaged) perceptron sequence tagger with Viterbi decoding — Collins (2002).

    score(x, y) = Σ_i  w · φ(x, i, y_i)  +  Σ_i  T[y_{i-1}, y_i]
    predict     = argmax_y score(x, y)               (Viterbi, O(n · |Y|²))
    update      w += φ(x, y_gold) − φ(x, y_pred)     (only when the prediction is wrong)

Weight averaging (lazy, timestamped) makes the final weights the mean over all updates,
which is what gives the perceptron its generalisation. Illegal BIO transitions (O -> I-X,
B-X -> I-Y) are masked to -inf, so every decoded sequence is a valid span set.

Features: word identity, prefixes/suffixes, shape, ±2 context words, context bigrams and
*gazetteer features* (does a dictionary match cover this token?), which is the classic
way to combine a lexicon with a statistical model.
"""

from __future__ import annotations

import pickle
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from sherlock_rx.nlp.gazetteer import Span

NEG = -1e9


def shape(w: str) -> str:
    out: list[str] = []
    for c in w:
        k = "d" if c.isdigit() else "a" if c.isalpha() else c
        if not out or out[-1] != k:
            out.append(k)
    return "".join(out)


def gaz_tags(n: int, spans: Sequence[Span]) -> list[str]:
    tags = ["O"] * n
    for sp in spans:
        kind = "F" if sp.source == "fuzzy" else "E"
        tags[sp.start] = f"B-{sp.type}-{kind}"
        for k in range(sp.start + 1, sp.end):
            tags[k] = f"I-{sp.type}-{kind}"
    return tags


def token_features(tokens: Sequence[str], gtags: Sequence[str] | None) -> list[list[str]]:
    feats = []
    n = len(tokens)
    pad = ["<s2>", "<s1>", *tokens, "</s1>", "</s2>"]
    for i, w in enumerate(tokens):
        j = i + 2
        f = [
            "b", f"w={w}", f"p3={w[:3]}", f"p4={w[:4]}", f"s3={w[-3:]}", f"s2={w[-2:]}",
            f"sh={shape(w)}", f"len={min(len(w), 8)}",
            f"w-1={pad[j - 1]}", f"w+1={pad[j + 1]}", f"w-2={pad[j - 2]}", f"w+2={pad[j + 2]}",
            f"w-1w={pad[j - 1]}|{w}", f"ww+1={w}|{pad[j + 1]}",
            f"pos={'first' if i == 0 else 'last' if i == n - 1 else 'mid'}",
        ]
        if gtags is not None:
            g = gtags[i]
            f.append(f"g={g}")
            if g != "O":
                f.append(f"gt={g[2:].rsplit('-', 1)[0]}")
            f.append(f"g-1={gtags[i - 1] if i else '<s>'}")
            f.append(f"g+1={gtags[i + 1] if i + 1 < n else '</s>'}")
        feats.append(f)
    return feats


class PerceptronTagger:
    def __init__(self, tags: Sequence[str], use_gazetteer: bool = True):
        self.tags = list(tags)
        self.tix = {t: i for i, t in enumerate(self.tags)}
        self.use_gazetteer = use_gazetteer
        k = len(self.tags)
        self.w: dict[str, np.ndarray] = {}
        self.trans = np.zeros((k + 1, k))  # row k = start state
        self._mask = self._bio_mask()
        # averaging bookkeeping
        self._tot: dict[str, np.ndarray] = {}
        self._ts: dict[str, np.ndarray] = {}
        self._trans_tot = np.zeros_like(self.trans)
        self._trans_ts = np.zeros_like(self.trans)
        self._c = 0

    def _bio_mask(self) -> np.ndarray:
        k = len(self.tags)
        mask = np.zeros((k + 1, k))
        for j, t in enumerate(self.tags):
            if not t.startswith("I-"):
                continue
            ent = t[2:]
            for i in range(k + 1):
                prev = self.tags[i] if i < k else "<s>"
                if prev not in (f"B-{ent}", f"I-{ent}"):
                    mask[i, j] = NEG
        return mask

    # ------------------------------------------------------------------ decoding
    def _emissions(self, feats: list[list[str]]) -> np.ndarray:
        e = np.zeros((len(feats), len(self.tags)))
        for i, fs in enumerate(feats):
            for f in fs:
                v = self.w.get(f)
                if v is not None:
                    e[i] += v
        return e

    def viterbi(self, feats: list[list[str]]) -> list[int]:
        n, k = len(feats), len(self.tags)
        if n == 0:
            return []
        em = self._emissions(feats)
        trans = self.trans + self._mask
        score = trans[k] + em[0]
        back = np.zeros((n, k), dtype=np.int64)
        for i in range(1, n):
            cand = score[:, None] + trans[:k]  # [prev, cur]
            back[i] = cand.argmax(axis=0)
            score = cand.max(axis=0) + em[i]
        path = [int(score.argmax())]
        for i in range(n - 1, 0, -1):
            path.append(int(back[i, path[-1]]))
        return path[::-1]

    def features(self, tokens: Sequence[str], spans: Sequence[Span] | None) -> list[list[str]]:
        g = gaz_tags(len(tokens), spans or []) if self.use_gazetteer else None
        return token_features(tokens, g)

    def predict(self, tokens: Sequence[str], spans: Sequence[Span] | None = None) -> list[str]:
        return [self.tags[i] for i in self.viterbi(self.features(tokens, spans))]

    # ------------------------------------------------------------------ training
    def _upd_w(self, f: str, j: int, delta: float) -> None:
        if f not in self.w:
            k = len(self.tags)
            self.w[f] = np.zeros(k)
            self._tot[f] = np.zeros(k)
            self._ts[f] = np.zeros(k)
        self._tot[f][j] += (self._c - self._ts[f][j]) * self.w[f][j]
        self._ts[f][j] = self._c
        self.w[f][j] += delta

    def _upd_t(self, i: int, j: int, delta: float) -> None:
        self._trans_tot[i, j] += (self._c - self._trans_ts[i, j]) * self.trans[i, j]
        self._trans_ts[i, j] = self._c
        self.trans[i, j] += delta

    def fit(self, data: Sequence[tuple[list[str], list[str], list[Span]]], epochs: int = 8,
            seed: int = 0, log: bool = False) -> PerceptronTagger:
        rng = np.random.default_rng(seed)
        cached = [(self.features(toks, spans), [self.tix[t] for t in tags])
                  for toks, tags, spans in data]
        k = len(self.tags)
        for ep in range(epochs):
            errors = 0
            for idx in rng.permutation(len(cached)):
                feats, gold = cached[idx]
                self._c += 1
                pred = self.viterbi(feats)
                if pred == gold:
                    continue
                errors += 1
                for i, (g, p) in enumerate(zip(gold, pred, strict=True)):
                    if g != p:
                        for f in feats[i]:
                            self._upd_w(f, g, 1.0)
                            self._upd_w(f, p, -1.0)
                    gp, pp = (gold[i - 1] if i else k), (pred[i - 1] if i else k)
                    if g != p or gp != pp:
                        self._upd_t(gp, g, 1.0)
                        self._upd_t(pp, p, -1.0)
            if log:
                print(f"  epoch {ep + 1}: sentence errors {errors}/{len(cached)}")
        self._average()
        return self

    def _average(self) -> None:
        c = max(self._c, 1)
        for f, v in self.w.items():
            tot = self._tot[f] + (self._c - self._ts[f]) * v
            self.w[f] = tot / c
        self.trans = (self._trans_tot + (self._c - self._trans_ts) * self.trans) / c
        self._tot, self._ts = {}, {}
        # prune near-zero weights: smaller model, same predictions
        self.w = {f: v for f, v in self.w.items() if np.abs(v).max() > 1e-6}

    # ------------------------------------------------------------------ io
    def save(self, path: str | Path) -> None:
        with open(path, "wb") as fh:
            pickle.dump({"tags": self.tags, "w": self.w, "trans": self.trans,
                         "use_gazetteer": self.use_gazetteer}, fh)

    @classmethod
    def load(cls, path: str | Path) -> PerceptronTagger:
        with open(path, "rb") as fh:
            d = pickle.load(fh)
        t = cls(d["tags"], d["use_gazetteer"])
        t.w, t.trans = d["w"], d["trans"]
        return t


def bio_spans(tags: Sequence[str]) -> list[tuple[int, int, str]]:
    spans, start, typ = [], None, None
    for i, t in enumerate([*tags, "O"]):
        if t == "O" or t.startswith("B-") or (t.startswith("I-") and t[2:] != typ):
            if start is not None:
                spans.append((start, i, str(typ)))
            start, typ = (i, t[2:]) if t != "O" else (None, None)
    return spans


def span_prf(gold: Sequence[Sequence[str]], pred: Sequence[Sequence[str]]
             ) -> dict[str, dict[str, float]]:
    """Entity-level precision/recall/F1 (exact span + type), per type and micro-averaged."""
    tp: dict[str, int] = defaultdict(int)
    fp: dict[str, int] = defaultdict(int)
    fn: dict[str, int] = defaultdict(int)
    for g, p in zip(gold, pred, strict=True):
        gs, ps = set(bio_spans(g)), set(bio_spans(p))
        for s in gs & ps:
            tp[s[2]] += 1
        for s in ps - gs:
            fp[s[2]] += 1
        for s in gs - ps:
            fn[s[2]] += 1
    def prf(t_p: int, f_p: int, f_n: int) -> dict[str, float]:
        prec = t_p / (t_p + f_p) if t_p + f_p else 0.0
        rec = t_p / (t_p + f_n) if t_p + f_n else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        return {"precision": prec, "recall": rec, "f1": f1, "support": float(t_p + f_n)}

    out = {t: prf(tp[t], fp[t], fn[t]) for t in sorted(set(tp) | set(fp) | set(fn))}
    out["micro"] = prf(sum(tp.values()), sum(fp.values()), sum(fn.values()))
    return out
