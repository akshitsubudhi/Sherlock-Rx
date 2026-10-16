"""One entry point for every evaluation; writes reports/EVAL.md and reports/eval.json."""

from __future__ import annotations

import json
import statistics as st
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from sklearn.metrics import accuracy_score, f1_score

from sherlock_rx.eval.rca_eval import evaluate as rca_evaluate

ROOT = Path(__file__).resolve().parents[3]
SEEDS = (7, 11, 23, 42, 101)


def rca_multi_seed(seeds: tuple[int, ...] = SEEDS, severity_scale: float = 1.0,
                   noise_scale: float = 1.0) -> dict[str, Any]:
    from sherlock_rx.data.generator import generate

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for seed in seeds:
            d = Path(tmp) / f"wh_{seed}"
            generate(d, scale=1.0, seed=seed, severity_scale=severity_scale,
                     noise_scale=noise_scale)
            rows.append(rca_evaluate(d))
    per_cause: dict[str, list[float]] = {}
    for r in rows:
        for c, v in r.per_cause.items():
            per_cause.setdefault(c, []).extend([v["acc"]] * int(v["n"]))
    return {
        "seeds": list(seeds), "severity_scale": severity_scale, "noise_scale": noise_scale,
        "scenarios": sum(r.n_scenarios for r in rows),
        "cause_at_1": st.mean(r.cause_at_1 for r in rows),
        "cause_at_2": st.mean(r.cause_at_2 for r in rows),
        "location_acc": st.mean(r.location_acc for r in rows),
        "flagged_rate": st.mean(r.flagged_rate for r in rows),
        "false_alarm_rate": st.mean(r.false_alarm_rate for r in rows),
        "median_latency_ms": st.median(r.median_latency_ms for r in rows),
        "per_cause_acc": {c: round(st.mean(v), 3) for c, v in sorted(per_cause.items())},
        "misses": [m for r in rows for m in r.misses],
    }


def nlu_handwritten(models: Path, files: tuple[str, ...] = ("nlu_handwritten.yaml",
                                                            "nlu_handwritten_v2.yaml")
                    ) -> dict[str, Any]:
    from sherlock_rx.nlp.intent import KeywordBaseline
    from sherlock_rx.nlp.pipeline import NLU
    from sherlock_rx.nlp.train import load_handwritten, slot_match

    nlu = NLU.load(models)
    kw = KeywordBaseline().fit([], list(nlu.model.classes_))
    out: dict[str, Any] = {}
    for fname in files:
        items = load_handwritten(ROOT / "tests" / "data" / fname)
        gold, pred, kw_pred, ok, tot, fails, lat = [], [], [], 0, 0, [], []
        for it in items:
            t0 = time.perf_counter()
            res = nlu.parse(it["text"])
            lat.append((time.perf_counter() - t0) * 1000)
            gold.append(it["intent"])
            pred.append(res.intent)
            kw_pred.append(str(kw.classes_[kw.predict_proba([res.masked]).argmax(axis=1)[0]]))
            o, t = slot_match(it.get("slots") or {}, res.slots)
            ok += o
            tot += t
            if res.intent != it["intent"] or o < t:
                fails.append({"text": it["text"], "expected": it["intent"], "got": res.intent,
                              "slots_ok": f"{o}/{t}", "got_slots": res.debug()["slots"]})
        lat.sort()
        out[fname] = {"n": len(items), "intent_accuracy": accuracy_score(gold, pred),
                      "intent_macro_f1": f1_score(gold, pred, average="macro"),
                      "slot_accuracy": ok / max(tot, 1), "p50_ms": round(lat[len(lat) // 2], 2),
                      "keyword_baseline_accuracy": accuracy_score(gold, kw_pred),
                      "failures": fails}
    return out


def run_all(what: str, data: Path, models: Path, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    if what in ("rca", "all"):
        print("RCA: 5 seeds, default difficulty ...")
        results["rca"] = rca_multi_seed()
        sweep = []
        for sev, noise in ((0.7, 1.0), (0.5, 1.0), (0.35, 1.0), (1.0, 2.0), (0.5, 2.0)):
            print(f"RCA sweep: severity x{sev}, noise x{noise} ...")
            r = rca_multi_seed((7, 11, 23), sev, noise)
            sweep.append({k: r[k] for k in ("severity_scale", "noise_scale", "scenarios",
                                            "cause_at_1", "location_acc", "flagged_rate",
                                            "false_alarm_rate")})
        results["rca_sweep"] = sweep
    if what in ("nlu", "all"):
        print("NLU: hand-written sets ...")
        results["nlu"] = nlu_handwritten(models)
        rep = ROOT / "reports" / "nlu_report.json"
        if rep.exists():
            results["nlu_training"] = json.loads(rep.read_text())
    if what in ("dialogue", "all"):
        print("Dialogue: scripted conversations ...")
        from sherlock_rx.cli import build_bot
        from sherlock_rx.eval.dialogue_eval import evaluate as dlg

        results["dialogue"] = asdict(dlg(build_bot(data, models)))
    existing = out_dir / "eval.json"
    if existing.exists() and what != "all":
        prev = json.loads(existing.read_text())
        prev.update(results)
        results = prev
    existing.write_text(json.dumps(results, indent=2, default=float))
    (out_dir / "EVAL.md").write_text(render(results))
    print(f"wrote {out_dir / 'EVAL.md'}")
    return results


def _p(x: float) -> str:
    return f"{x * 100:.1f}%"


def render(r: dict[str, Any]) -> str:
    L = ["# Sherlock Rx: evaluation results", "",
         f"_Generated {r['generated_at']} by `sherlock eval all`. Every number below is produced "
         "by code in `src/sherlock_rx/eval/`; nothing is hand-entered._", ""]
    if "rca" in r:
        a = r["rca"]
        L += ["## 1. Root-cause engine vs planted ground truth", "",
              f"{a['scenarios']} planted scenarios over {len(a['seeds'])} random seeds "
              f"(6 cause types), plus {len(a['seeds']) * 20} control cells with nothing planted.",
              "", "| Metric | Value |", "|---|---|",
              f"| Top-1 cause accuracy | **{_p(a['cause_at_1'])}** |",
              f"| Top-2 cause accuracy | {_p(a['cause_at_2'])} |",
              f"| Location accuracy (state-level causes) | {_p(a['location_acc'])} |",
              f"| Under-performance flagged | {_p(a['flagged_rate'])} |",
              f"| False alarms on control cells | {_p(a['false_alarm_rate'])} |",
              f"| Median RCA latency | {a['median_latency_ms']:.0f} ms |", "",
              "Per cause: " + ", ".join(f"{c} {_p(v)}" for c, v in a["per_cause_acc"].items()), ""]
        if a["misses"]:
            L += ["Misses:", ""] + [f"- {m['scenario']} {m['brand']}/{m['region']}/{m['quarter']}: "
                                    f"planted **{m['truth']}** (severity {m['severity']}), "
                                    f"predicted {m['pred']}" for m in a["misses"]] + [""]
    if "rca_sweep" in r:
        L += ["### Robustness: weaker causes and more noise (3 seeds each)", "",
              "| Cause severity | Noise | Top-1 | Location | Flagged | False alarms |",
              "|---|---|---|---|---|---|"]
        L += [f"| x{s['severity_scale']} | x{s['noise_scale']} | {_p(s['cause_at_1'])} | "
              f"{_p(s['location_acc'])} | {_p(s['flagged_rate'])} | {_p(s['false_alarm_rate'])} |"
              for s in r["rca_sweep"]]
        L += ["", "Accuracy falls as planted causes get weaker relative to noise, which is the "
                  "expected shape: at x0.35 severity many planted effects are smaller than "
                  "normal month-to-month variation.", ""]
    if "nlu" in r:
        L += ["## 2. NLU on hand-written utterances", "",
              "| Set | n | Intent accuracy | Keyword baseline | Intent macro-F1 | Slot accuracy |"
              " p50 latency |", "|---|---|---|---|---|---|---|"]
        for name, v in r["nlu"].items():
            L.append(f"| {name} | {v['n']} | {_p(v['intent_accuracy'])} | "
                     f"{_p(v['keyword_baseline_accuracy'])} | {v['intent_macro_f1']:.3f} | "
                     f"{_p(v['slot_accuracy'])} | {v['p50_ms']} ms |")
        L += ["", "`nlu_handwritten.yaml` (v1) was used during development and is optimistic. "
                  "`nlu_handwritten_v2.yaml` was written after the models were frozen and scored "
                  "once: it is the honest estimate.", ""]
        for name, v in r["nlu"].items():
            if v["failures"]:
                L += [f"Failures in {name}:", "", "| Utterance | Expected | Got | Slots |",
                      "|---|---|---|---|"]
                L += [f"| {f['text']} | {f['expected']} | {f['got']} | {f['slots_ok']} |"
                      for f in v["failures"]] + [""]
    if "nlu_training" in r:
        t = r["nlu_training"]
        L += ["### Model comparison (template-held-out test split)", "",
              "| Intent model | Val macro-F1 | Test macro-F1 | Val ECE |", "|---|---|---|---|"]
        for name, row in t["intent"].items():
            val = f"{row['val']['macro_f1']:.3f}" if "val" in row else "-"
            ece = f"{row['val']['ece']:.3f}" if "val" in row else "-"
            L.append(f"| {name} | {val} | {row['test']['macro_f1']:.3f} | {ece} |")
        L += ["", "| Entity recognition | Precision | Recall | F1 |", "|---|---|---|---|"]
        for name, rep in t["ner"].items():
            m = rep["micro"]
            L.append(f"| {name} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} |")
        L.append("")
    if "dialogue" in r:
        d = r["dialogue"]
        L += ["## 3. Multi-turn dialogue", "",
              f"{d['conversations']} scripted conversations, {d['turns']} turns "
              "(follow-ups, ellipsis, clarification, carry-over, out-of-scope isolation).", "",
              f"- Conversation success: **{_p(d['success_rate'])}**",
              f"- Turn accuracy: {_p(d['turn_accuracy'])}",
              f"- Turn latency p50 {d['p50_turn_ms']:.0f} ms, p95 {d['p95_turn_ms']:.0f} ms", ""]
        for f in d["failures"]:
            L.append(f"- ✗ {f['conversation']} turn {f['turn']} “{f['say']}”: "
                     + "; ".join(f["errors"]))
    return "\n".join(L) + "\n"
