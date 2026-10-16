"""Unit tests for the from-scratch NLP components, checked against reference implementations."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer as SkTfidf
from sklearn.naive_bayes import MultinomialNB as SkNB

from sherlock_rx.nlp.gazetteer import Gazetteer
from sherlock_rx.nlp.naive_bayes import MultinomialNB
from sherlock_rx.nlp.retrieval import BM25, Doc, Glossary
from sherlock_rx.nlp.spell import SpellCorrector
from sherlock_rx.nlp.tagger import PerceptronTagger, bio_spans, span_prf
from sherlock_rx.nlp.text import normalize, stem, tokenize, words
from sherlock_rx.nlp.vectorize import CountVectorizer, TfidfVectorizer

DOCS = ["why is cardivex down in north", "show cardivex sales by state",
        "what is secondary sales", "why why is glucoryl lagging in q2 fy27", "hello there"]


# ---------------------------------------------------------------- text
def test_tokenizer_keeps_domain_tokens() -> None:
    assert words("Why's Cardivex-40 down 3.5% in Q2 FY27 (₹12,000)?") == [
        "why", "is", "cardivex", "-", "40", "down", "3.5%", "in", "q2", "fy27", "(",
        "₹12,000", ")", "?"]


def test_normalize_contractions_shorthand_possessive() -> None:
    assert normalize("What's Cardivex's sales pls") == "what is cardivex sales please"


def test_token_offsets_point_into_text() -> None:
    t = normalize("Glucoryl in Rajasthan")
    for tok in tokenize(t):
        assert t[tok.start:tok.end] == tok.text


@pytest.mark.parametrize("w,s", [("territories", "territory"), ("stockists", "stockist"),
                                 ("declining", "declin"), ("painex", "painex"), ("is", "is")])
def test_stemmer_is_conservative(w: str, s: str) -> None:
    assert stem(w) == s


# ---------------------------------------------------------------- vectorisers & NB
def test_tfidf_matches_sklearn() -> None:
    mine = TfidfVectorizer(word_range=(1, 2), char_range=(3, 4), sublinear_tf=True)
    x_mine = mine.fit_transform(DOCS).toarray()
    sk = SkTfidf(analyzer=mine.analyze, sublinear_tf=True, smooth_idf=True, norm="l2")
    x_sk = sk.fit_transform(DOCS).toarray()
    order = [sk.vocabulary_[t] for t in sorted(mine.vocabulary_, key=mine.vocabulary_.get)]
    np.testing.assert_allclose(x_mine, x_sk[:, order], atol=1e-9)


def test_naive_bayes_matches_sklearn() -> None:
    labels = ["diagnose", "metric", "define", "diagnose", "greet"]
    vec = CountVectorizer(word_range=(1, 1))
    x = vec.fit_transform(DOCS)
    mine = MultinomialNB(alpha=0.5).fit(x, labels)
    ref = SkNB(alpha=0.5).fit(x, labels)
    np.testing.assert_allclose(mine.predict_proba(x), ref.predict_proba(x), atol=1e-10)


# ---------------------------------------------------------------- spelling
def test_spell_corrects_typos_but_not_valid_words() -> None:
    sc = SpellCorrector(["underperforming", "stockout", "numbers", "jul", "territory"] * 5)
    assert sc.correct_word("undreperforming") == "underperforming"
    assert sc.correct_word("stokout") == "stockout"
    assert sc.correct_word("july") == "july"  # a month name, not a typo of "jul"
    assert sc.correct_word("number") == "number"  # inflection, not a typo
    assert sc.correct_word("q2") == "q2"


# ---------------------------------------------------------------- gazetteer
@pytest.fixture(scope="module")
def gaz() -> Gazetteer:
    return Gazetteer.build(common_words={"thanks", "sales", "what", "about", "up"})


def test_gazetteer_exact_fuzzy_and_longest_match(gaz: Gazetteer) -> None:
    spans = {(s.type, s.value) for s in gaz.find("Cardivex 40 sales in rajastan, north zone")}
    assert ("SKU", "Cardivex 40") in spans
    assert ("STATE", "Rajasthan") in spans
    assert ("REGION", "North") in spans
    assert ("BRAND", "Cardivex") not in spans  # the longer SKU span wins


def test_state_codes_only_when_uppercase(gaz: Gazetteer) -> None:
    assert any(s.value == "Uttar Pradesh" for s in gaz.find("sales in UP"))
    assert not any(s.type == "STATE" for s in gaz.find("what is up"))


def test_common_words_never_fuzzy_match_towns(gaz: Gazetteer) -> None:
    assert gaz.find("thanks!") == []  # "thanks" ~ "Thane" at ratio 91


def test_entity_linking(gaz: Gazetteer) -> None:
    assert gaz.link("BRAND", "gluco") == ("Glucoryl", 90.0)
    linked = gaz.link("STATE", "karnatka")
    assert linked is not None and linked[0] == "Karnataka"
    assert gaz.link("BRAND", "zzzz") is None


# ---------------------------------------------------------------- tagger
def test_bio_spans_and_prf() -> None:
    assert bio_spans(["B-BRAND", "I-BRAND", "O", "B-REGION"]) == [(0, 2, "BRAND"), (3, 4, "REGION")]
    r = span_prf([["B-BRAND", "O"]], [["B-BRAND", "O"]])
    assert r["micro"]["f1"] == 1.0


def test_perceptron_learns_context_and_decodes_valid_bio() -> None:
    data = [
        (["why", "is", "cardivex", "down"], ["O", "O", "B-BRAND", "O"], []),
        (["why", "is", "glucoryl", "down"], ["O", "O", "B-BRAND", "O"], []),
        (["sales", "in", "north", "zone"], ["O", "O", "B-REGION", "I-REGION"], []),
        (["sales", "in", "south", "zone"], ["O", "O", "B-REGION", "I-REGION"], []),
    ] * 5
    tags = ["O", "B-BRAND", "I-BRAND", "B-REGION", "I-REGION"]
    t = PerceptronTagger(tags, use_gazetteer=False).fit(data, epochs=5)
    assert t.predict(["why", "is", "painex", "down"]) == ["O", "O", "B-BRAND", "O"]
    out = t.predict(["zone", "zone", "in", "east", "zone"])
    for i, tag in enumerate(out):  # never I-X without a preceding B-X / I-X
        if tag.startswith("I-"):
            assert i > 0 and out[i - 1][2:] == tag[2:]


# ---------------------------------------------------------------- retrieval
def test_bm25_ranks_relevant_doc_first_and_filters() -> None:
    docs = [Doc("a", "stockist stopped lifting over margin dispute", {"s": "x"}),
            Doc("b", "CME conducted with doctors", {"s": "x"}),
            Doc("c", "supply shortage at the depot, stock out", {"s": "y"})]
    idx = BM25(docs)
    assert idx.search("stock shortage")[0].doc.doc_id == "c"
    assert idx.search("stock shortage", where=lambda d: d.meta["s"] == "x") == []


@pytest.mark.parametrize("q,term", [("what does DPCO mean", "DPCO"),
                                    ("define PTR", "PTR (Price to Retailer)"),
                                    ("whats a C&F agent", "C&F agent"),
                                    ("difference between primary and secondary sales",
                                     "Primary vs secondary sales")])
def test_glossary_lookup(q: str, term: str) -> None:
    assert Glossary().lookup(q)[0].doc.meta["term"] == term
