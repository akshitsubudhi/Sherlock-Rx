"""Noisy-channel spelling correction (Norvig-style) over a domain vocabulary.

P(correction | typo) ∝ P(typo | correction) · P(correction)
  * candidate generation: edit distance 1, then 2 (deletes, transposes, replaces, inserts)
  * channel model:        prefer distance 1 over 2 (a simple but effective prior)
  * language model:       unigram counts from the NLU training corpus + gazetteer terms

Only out-of-vocabulary alphabetic tokens of length >= 4 are touched, so codes ("q2",
"fy27") and short words are never "corrected" into something else. Entity names get
their own fuzzy matcher in the NER layer; this module fixes the words *around* them
("undreperforming" -> "underperforming", "stokout" -> "stockout").
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

LETTERS = "abcdefghijklmnopqrstuvwxyz"

# Valid words the training corpus may never contain; never "correct" these.
COMMON_ENGLISH = frozenset("""
january february march april may june july august september october november december number
numbers figure figures total amount count percent percentage today tomorrow yesterday week
weeks year years quarter quarters month months day days morning evening doing going looking
working selling buying getting making taking coming running falling good bad better worse
best worst high higher low lower big bigger small smaller more less most least much many few
some every each all none other another same different data report reports table chart graph
list summary detail details overview update team manager doctor doctors patient patients
chemist chemists hospital clinic clinics problem problems issue issues reason reasons
question questions answer answers idea please thanks thank sorry okay right wrong sure maybe
really actually exactly
""".split())


class SpellCorrector:
    def __init__(self, corpus_tokens: Iterable[str], protected: Iterable[str] = ()):
        self.counts: Counter[str] = Counter(t for t in corpus_tokens if t.isalpha())
        self.total = sum(self.counts.values()) or 1
        self.protected = {p.lower() for p in protected} | COMMON_ENGLISH

    def known(self, words: Iterable[str]) -> set[str]:
        return {w for w in words if w in self.counts}

    @staticmethod
    def edits1(word: str) -> set[str]:
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [a + b[1:] for a, b in splits if b]
        transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1]
        replaces = [a + c + b[1:] for a, b in splits if b for c in LETTERS]
        inserts = [a + c + b for a, b in splits for c in LETTERS]
        return set(deletes + transposes + replaces + inserts)

    def candidates(self, word: str) -> list[str]:
        e1 = self.known(self.edits1(word))
        if e1:
            return sorted(e1, key=lambda w: -self.counts[w])
        e2 = self.known(e for e1w in self.edits1(word) for e in self.edits1(e1w))
        return sorted(e2, key=lambda w: -self.counts[w])

    def correct_word(self, word: str) -> str:
        if (len(word) < 4 or not word.isalpha() or word in self.counts
                or word in self.protected):
            return word
        cands = self.candidates(word)
        if not cands:
            return word
        best = cands[0]
        # An inflection of a real word ("number" -> "numbers") is not a typo.
        for suf in ("s", "es", "ed", "ing", "d"):
            if best == word + suf or word == best + suf:
                return word
        return best

    def correct(self, tokens: list[str]) -> tuple[list[str], list[tuple[str, str]]]:
        """Return corrected tokens and the list of (original, corrected) changes."""
        out, changes = [], []
        for t in tokens:
            c = self.correct_word(t)
            out.append(c)
            if c != t:
                changes.append((t, c))
        return out, changes
