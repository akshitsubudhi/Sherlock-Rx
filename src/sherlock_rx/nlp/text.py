"""Text preprocessing: normalisation, tokenisation, stop-words and a light stemmer.

Written from scratch (no NLTK/spaCy) so each step is visible and testable:

* normalise   NFKC unicode, lower-case, unify quotes/dashes, expand contractions and
              domain shorthand ("mr" -> medical rep is *not* done here; that is entity work).
* tokenise    a regex tokenizer that keeps domain tokens intact: "Q2", "FY27", "40mg",
              "M1", "3.5%", "₹12", "jul-sep" -> ["jul", "-", "sep"].
* stem        Harman's S-stemmer plus a few -ing/-ed/-ly rules: conservative on purpose,
              because over-stemming merges brand names ("Painex" must stay "painex").
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

CONTRACTIONS = {
    "what's": "what is", "whats": "what is", "where's": "where is", "how's": "how is",
    "why's": "why is", "it's": "it is", "that's": "that is", "isn't": "is not",
    "aren't": "are not", "didn't": "did not", "doesn't": "does not", "don't": "do not",
    "wasn't": "was not", "weren't": "were not", "can't": "can not", "won't": "will not",
    "i'm": "i am", "we're": "we are", "they're": "they are", "you're": "you are",
    "let's": "let us", "there's": "there is", "who's": "who is", "i'd": "i would",
    "i've": "i have", "we've": "we have",
}
SHORTHAND = {"pls": "please", "plz": "please", "thx": "thanks", "ty": "thanks", "u": "you",
             "ur": "your", "r": "are", "wrt": "with respect to", "vs.": "vs", "v/s": "vs",
             "y'day": "yesterday", "qtr": "quarter", "mth": "month", "yr": "year"}

STOPWORDS = frozenset("""
a an the and or but if then of to in on at by for with from into about as is are was were be
been being am do does did doing have has had having i me my we our you your he she it its
they them their this that these those there here what which who whom please can could would
should will shall may might must just so very too also any some such than only own same s t
hey hi hello ok okay kindly tell show give me us let
""".split())

TOKEN_RE = re.compile(
    r"""
    ₹\s?\d[\d,]*(?:\.\d+)?            # currency amounts
  | \d+(?:\.\d+)?%                    # percentages
  | [a-z]+\d+[a-z\d]*                 # alnum codes: q2, fy27, m1, h1, 40mg (letter first)
  | \d+[a-z]+[a-z\d]*                 # 40mg, 3rd, 2026q1
  | \d+(?:[.,]\d+)*                   # numbers
  | [a-z]+(?:'[a-z]+)?                # words (with internal apostrophe)
  | [^\sa-z\d]                        # any other single symbol
    """,
    re.VERBOSE,
)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("’", "'").replace("‘", "'").replace("`", "'")
    text = re.sub(r"[–—]", "-", text)
    text = text.lower().strip()
    words = []
    for w in text.split():
        core = w.rstrip("?!.,;:")
        tail = w[len(core):]
        if core in CONTRACTIONS:
            w = CONTRACTIONS[core] + tail
        elif core in SHORTHAND:
            w = SHORTHAND[core] + tail
        words.append(w)
    text = re.sub(r"(\w)'s\b", r"\1", " ".join(words))  # possessive: "cardivex's" -> "cardivex"
    return re.sub(r"\s+", " ", text)


@dataclass(frozen=True)
class Token:
    text: str
    start: int
    end: int


def tokenize(text: str) -> list[Token]:
    """Tokenise already-normalised text, keeping character offsets for span highlighting."""
    return [Token(m.group(), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]


def words(text: str) -> list[str]:
    return [t.text for t in tokenize(normalize(text))]


def stem(word: str) -> str:
    if len(word) <= 3 or not word.isalpha():
        return word
    # Harman S-stemmer
    if word.endswith("ies") and not word.endswith(("eies", "aies")):
        word = word[:-3] + "y"
    elif (word.endswith("es") and not word.endswith(("aes", "ees", "oes"))) or (
            word.endswith("s") and not word.endswith(("us", "ss"))):
        word = word[:-1]
    # A few derivational suffixes, only when a reasonable stem remains.
    for suf, rep in (("ing", ""), ("edly", ""), ("ed", ""), ("ly", ""), ("ment", ""),
                     ("ation", "ate")):
        if word.endswith(suf) and len(word) - len(suf) >= 4:
            word = word[: -len(suf)] + rep
            break
    return word


def content_terms(text: str) -> list[str]:
    """Normalise -> tokenise -> drop stop-words/punctuation -> stem (used by BM25)."""
    return [stem(w) for w in words(text) if w not in STOPWORDS and any(c.isalnum() for c in w)]
