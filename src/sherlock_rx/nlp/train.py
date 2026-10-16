"""Train and evaluate the NLU stack; writes models/ and reports/nlu_report.{json,md}.

Protocol
  * corpus split by template: train / val / test (test phrasings never seen in training)
  * model selection on val macro-F1; the confidence threshold is tuned on val
  * final numbers reported on (a) held-out templates and (b) the hand-written test set
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from sherlock_rx.nlp.gazetteer import Gazetteer
from sherlock_rx.nlp.intent import IntentModel, KeywordBaseline, LinearIntent, NaiveBayesIntent
from sherlock_rx.nlp.ner import EntityExtractor
from sherlock_rx.nlp.nlu_data import CorpusBuilder, Example
from sherlock_rx.nlp.pipeline import NLU
from sherlock_rx.nlp.spell import SpellCorrector
from sherlock_rx.nlp.tagger import PerceptronTagger, gaz_tags, span_prf

ROOT = Path(__file__).resolve().parents[3]
HANDWRITTEN = ROOT / "tests" / "data" / "nlu_handwritten.yaml"


def _corpus_hash(examples: Sequence[Example]) -> str:
    h = hashlib.sha1()
    for e in examples:
        h.update(f"{e.intent}|{' '.join(e.tokens)}|{' '.join(e.tags)}\n".encode())
    return h.hexdigest()[:12]


def load_handwritten(path: Path = HANDWRITTEN) -> list[dict[str, Any]]:
    return list(yaml.safe_load(path.read_text()))


def slot_match(expected: dict[str, Any], got: dict[str, Any]) -> tuple[int, int]:
    """(correct, total) over the slots listed in `expected`."""
    ok = 0
    for k, v in expected.items():
        g = got.get(k)
        if k == "period":
            ok += g is not None and getattr(g, "label", None) == v
        elif isinstance(v, list):
            ok += isinstance(g, list) and sorted(map(str, g)) == sorted(map(str, v))
        else:
            ok += g == v
    return ok, len(expected)


def expected_calibration_error(probs: np.ndarray, y_true: Sequence[str],
                               classes: np.ndarray, bins: int = 10) -> float:
    """ECE: |accuracy - confidence| averaged over confidence bins, weighted by bin size."""
    conf = probs.max(axis=1)
    correct = classes[probs.argmax(axis=1)] == np.asarray(y_true)
    ece = 0.0
    for lo in np.linspace(0, 1, bins, endpoint=False):
        m = (conf > lo) & (conf <= lo + 1 / bins)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def _fresh(model: IntentModel) -> IntentModel:
    if isinstance(model, LinearIntent):
        return LinearIntent(model.kind)
    if isinstance(model, NaiveBayesIntent):
        return NaiveBayesIntent()
    return KeywordBaseline()


def train(out_dir: str | Path = ROOT / "models", report_dir: str | Path = ROOT / "reports",
          seed: int = 13, per_intent: int = 450, log: bool = True) -> dict[str, Any]:
    t_start = time.perf_counter()
    say = print if log else (lambda *a, **k: None)
    splits = CorpusBuilder(seed).build(per_intent=per_intent)
    train_ex, val_ex, test_ex = splits["train"], splits["val"], splits["test"]
    say(f"corpus: train={len(train_ex)} val={len(val_ex)} test={len(test_ex)} "
        f"(split by template)")

    # Lexical resources
    gaz0 = Gazetteer.build()
    alias_tokens = {w for k in gaz0.table for w in k.split()}
    common = {t for e in train_ex for t, g in zip(e.tokens, e.tags, strict=True) if g == "O"}
    common -= alias_tokens
    gaz = Gazetteer.build(common_words=common)
    spell = SpellCorrector([t for e in train_ex for t in e.tokens] + list(alias_tokens) * 3,
                           protected=alias_tokens)

    def corrected(e: Example) -> tuple[list[str], str]:
        toks, _ = spell.correct(e.tokens)
        return toks, " ".join(toks)

    # ---------------- sequence tagger
    t0 = time.perf_counter()
    tag_set = sorted({t for e in train_ex for t in e.tags})
    tr = []
    for e in train_ex:
        toks, txt = corrected(e)
        tr.append((toks, e.tags, gaz.find(txt)))
    tagger = PerceptronTagger(tag_set, use_gazetteer=True).fit(tr, epochs=8, seed=seed)
    tagger_nogaz = PerceptronTagger(tag_set, use_gazetteer=False).fit(tr, epochs=8, seed=seed)
    say(f"tagger trained in {time.perf_counter() - t0:.1f}s ({len(tagger.w)} features)")

    ner_report: dict[str, Any] = {}
    te = []
    for e in test_ex:
        toks, txt = corrected(e)
        te.append((toks, e.tags, gaz.find(txt)))
    gold = [g for _, g, _ in te]
    for name, pred in (
        ("gazetteer_only", [[t if t == "O" else t.rsplit("-", 1)[0] for t in gaz_tags(len(x), sp)]
                            for x, _, sp in te]),
        ("perceptron_no_gazetteer", [tagger_nogaz.predict(x, sp) for x, _, sp in te]),
        ("perceptron_with_gazetteer", [tagger.predict(x, sp) for x, _, sp in te]),
    ):
        ner_report[name] = span_prf(gold, pred)
        say(f"  NER {name:28s} micro-F1 {ner_report[name]['micro']['f1']:.3f}")

    # ---------------- intent models on entity-masked text
    extractor = EntityExtractor(gaz, tagger)
    nlu_tmp = NLU(spell, extractor, KeywordBaseline())

    def masked(examples: Sequence[Example]) -> list[str]:
        return [extractor.extract(nlu_tmp.correct(e.text)[0]).masked() for e in examples]

    def raw(examples: Sequence[Example]) -> list[str]:
        return [nlu_tmp.correct(e.text)[0].lower() for e in examples]

    X = {"train": masked(train_ex), "val": masked(val_ex), "test": masked(test_ex)}
    y = {"train": [e.intent for e in train_ex], "val": [e.intent for e in val_ex],
         "test": [e.intent for e in test_ex]}
    candidates: list[IntentModel] = [KeywordBaseline(), NaiveBayesIntent(),
                                     LinearIntent("logreg"), LinearIntent("svm")]
    intent_report: dict[str, Any] = {}
    best: IntentModel | None = None
    best_f1 = -1.0
    for m in candidates:
        t0 = time.perf_counter()
        m.fit(X["train"], y["train"])
        fit_s = time.perf_counter() - t0
        row: dict[str, Any] = {}
        for split in ("val", "test"):
            proba = m.predict_proba(X[split])
            labels_pred = m.classes_[proba.argmax(axis=1)]
            row[split] = {"accuracy": accuracy_score(y[split], labels_pred),
                          "macro_f1": f1_score(y[split], labels_pred, average="macro"),
                          "ece": expected_calibration_error(proba, y[split], m.classes_)}
        row["fit_seconds"] = round(fit_s, 2)
        intent_report[m.name] = row
        say(f"  intent {m.name:18s} val F1 {row['val']['macro_f1']:.3f} "
            f"test F1 {row['test']['macro_f1']:.3f} val ECE {row['val']['ece']:.3f}")
        # The confidence gate needs calibrated probabilities: models with val ECE > 0.15
        # are only eligible if nothing calibrated exists.
        score = row["val"]["macro_f1"] - (1.0 if row["val"]["ece"] > 0.15 else 0.0)
        if m.name != "keyword_rules" and score > best_f1:
            best, best_f1 = m, score
    assert best is not None

    # Ablation: same best model class without entity masking.
    ablate = LinearIntent("logreg").fit(raw(train_ex), y["train"])
    abl_pred = ablate.classes_[ablate.predict_proba(raw(test_ex)).argmax(axis=1)]
    intent_report["tfidf_logreg_no_masking"] = {
        "test": {"accuracy": accuracy_score(y["test"], abl_pred),
                 "macro_f1": f1_score(y["test"], abl_pred, average="macro")}}

    # Confidence threshold: lowest value keeping >= 98% precision on accepted val predictions.
    pv = best.predict_proba(X["val"])
    conf, pred_v = pv.max(axis=1), best.classes_[pv.argmax(axis=1)]
    correct = pred_v == np.asarray(y["val"])
    threshold = 0.3
    for th in np.arange(0.2, 0.8, 0.025):
        acc = correct[conf >= th].mean() if (conf >= th).any() else 1.0
        if acc >= 0.98:
            threshold = float(round(th, 3))
            break

    # Production model: same configuration refit on every template (train + val + test).
    # The split above estimated generalisation; holding templates back from the shipped
    # model would only remove vocabulary. The hand-written set is independent of templates.
    final = _fresh(best).fit(X["train"] + X["val"] + X["test"],
                             y["train"] + y["val"] + y["test"])
    nlu = NLU(spell, extractor, final, threshold, {})

    # ---------------- hand-written evaluation (end-to-end through the pipeline)
    hw = load_handwritten()
    hw_pred: list[str] = []
    ok_slots = tot_slots = 0
    per_slot: dict[str, list[int]] = {}
    failures = []
    for item in hw:
        res = nlu.parse(item["text"])
        hw_pred.append(res.intent)
        o, t = slot_match(item.get("slots") or {}, res.slots)
        ok_slots += o
        tot_slots += t
        for k, v in (item.get("slots") or {}).items():
            c = per_slot.setdefault(k, [0, 0])
            c[0] += slot_match({k: v}, res.slots)[0]
            c[1] += 1
        if res.intent != item["intent"] or o < t:
            failures.append({"text": item["text"], "expected": item["intent"],
                             "got": res.intent, "confidence": round(res.confidence, 2),
                             "slots_ok": f"{o}/{t}", "slots": res.debug()["slots"]})
    hw_gold = [i["intent"] for i in hw]
    labels = list(best.classes_)
    hw_report = {
        "n": len(hw),
        "intent_accuracy": accuracy_score(hw_gold, hw_pred),
        "intent_macro_f1": f1_score(hw_gold, hw_pred, average="macro"),
        "slot_accuracy": ok_slots / max(tot_slots, 1),
        "per_slot_accuracy": {k: round(v[0] / v[1], 3) for k, v in sorted(per_slot.items())},
        "confusion": {"labels": labels,
                      "matrix": confusion_matrix(hw_gold, hw_pred, labels=labels).tolist()},
        "failures": failures,
    }
    # Keyword baseline on hand-written set, for the comparison table.
    kw = KeywordBaseline().fit(X["train"], y["train"])
    kw_pred = [kw.classes_[kw.predict_proba([extractor.extract(nlu.correct(i["text"])[0]).masked()]
                                           ).argmax(axis=1)[0]] for i in hw]
    hw_report["keyword_baseline_accuracy"] = accuracy_score(hw_gold, kw_pred)
    say(f"hand-written: intent acc {hw_report['intent_accuracy']:.3f} "
        f"(keyword baseline {hw_report['keyword_baseline_accuracy']:.3f}), "
        f"slot acc {hw_report['slot_accuracy']:.3f}, threshold {threshold}")

    meta = {
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": seed,
        "corpus_hash": _corpus_hash(train_ex),
        "corpus_sizes": {k: len(v) for k, v in splits.items()},
        "intents": labels,
        "entity_tags": tag_set,
        "selected_intent_model": best.name,
        "confidence_threshold": threshold,
        "tagger_features": len(tagger.w),
        "train_seconds": round(time.perf_counter() - t_start, 1),
    }
    nlu.meta = meta
    nlu.save(out_dir)
    report = {"meta": meta, "intent": intent_report, "ner": ner_report, "handwritten": hw_report}
    rd = Path(report_dir)
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "nlu_report.json").write_text(json.dumps(report, indent=2, default=float))
    (rd / "nlu_report.md").write_text(render_markdown(report))
    say(f"saved models to {out_dir} and report to {rd / 'nlu_report.md'}")
    return report


def render_markdown(r: dict[str, Any]) -> str:
    m, hw = r["meta"], r["handwritten"]
    lines = [
        "# NLU evaluation report", "",
        f"Trained {m['trained_at']} · seed {m['seed']} · corpus `{m['corpus_hash']}` · "
        f"{m['train_seconds']}s on CPU", "",
        f"Corpus (split **by template**): {m['corpus_sizes']}", "",
        "## Intent classification (macro-F1)", "",
        "| Model | Val F1 | Test F1 (unseen templates) | Val ECE |", "|---|---|---|---|",
    ]
    for name, row in r["intent"].items():
        val = f"{row['val']['macro_f1']:.3f}" if "val" in row else "-"
        ece = f"{row['val']['ece']:.3f}" if "val" in row else "-"
        lines.append(f"| {name} | {val} | {row['test']['macro_f1']:.3f} | {ece} |")
    lines += ["", f"Selected: **{m['selected_intent_model']}**, confidence threshold "
                  f"{m['confidence_threshold']} (≥98% precision on accepted val predictions).", "",
              "## Entity recognition (entity-level micro-F1, unseen templates)", "",
              "| Approach | Precision | Recall | F1 |", "|---|---|---|---|"]
    for name, rep in r["ner"].items():
        mi = rep["micro"]
        lines.append(f"| {name} | {mi['precision']:.3f} | {mi['recall']:.3f} | {mi['f1']:.3f} |")
    lines += ["", "## Hand-written test set (end-to-end pipeline)", "",
              f"* {hw['n']} utterances written by hand (typos, shorthand, Hinglish)",
              f"* Intent accuracy **{hw['intent_accuracy']:.3f}** (macro-F1 "
              f"{hw['intent_macro_f1']:.3f}); keyword-rule baseline "
              f"{hw['keyword_baseline_accuracy']:.3f}",
              f"* Slot accuracy **{hw['slot_accuracy']:.3f}**: "
              + ", ".join(f"{k} {v:.2f}" for k, v in hw["per_slot_accuracy"].items()), "",
              "### Failures", "", "| Utterance | Expected | Got | Conf | Slots |",
              "|---|---|---|---|---|"]
    for f in hw["failures"]:
        lines.append(f"| {f['text']} | {f['expected']} | {f['got']} | {f['confidence']} | "
                     f"{f['slots_ok']} |")
    return "\n".join(lines) + "\n"
