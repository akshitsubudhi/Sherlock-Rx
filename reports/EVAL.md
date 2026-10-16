# Sherlock Rx: evaluation results

_Generated 2026-10-17 04:17:37 by `sherlock eval all`. Every number below is produced by code in `src/sherlock_rx/eval/`; nothing is hand-entered._

## 1. Root-cause engine vs planted ground truth

160 planted scenarios over 5 random seeds (6 cause types), plus 100 control cells with nothing planted.

| Metric | Value |
|---|---|
| Top-1 cause accuracy | **98.8%** |
| Top-2 cause accuracy | 98.8% |
| Location accuracy (state-level causes) | 100.0% |
| Under-performance flagged | 100.0% |
| False alarms on control cells | 0.0% |
| Median RCA latency | 68 ms |

Per cause: competitor_launch 96.7%, distribution_loss 96.7%, field_force_vacancy 100.0%, market_slowdown 100.0%, price_cut 100.0%, stockout 100.0%

Misses:

- S14 Gastrazole/Central/Q3 FY26: planted **competitor_launch** (severity 0.29), predicted market_slowdown
- S18 Glucoryl/East/Q3 FY26: planted **distribution_loss** (severity 0.41), predicted none

### Robustness: weaker causes and more noise (3 seeds each)

| Cause severity | Noise | Top-1 | Location | Flagged | False alarms |
|---|---|---|---|---|---|
| x0.7 | x1.0 | 100.0% | 100.0% | 100.0% | 0.0% |
| x0.5 | x1.0 | 95.8% | 100.0% | 100.0% | 0.0% |
| x0.35 | x1.0 | 88.5% | 100.0% | 96.9% | 0.0% |
| x1.0 | x2.0 | 97.9% | 100.0% | 99.0% | 1.7% |
| x0.5 | x2.0 | 90.6% | 100.0% | 97.9% | 1.7% |

Accuracy falls as planted causes get weaker relative to noise, which is the expected shape: at x0.35 severity many planted effects are smaller than normal month-to-month variation.

## 2. NLU on hand-written utterances

| Set | n | Intent accuracy | Keyword baseline | Intent macro-F1 | Slot accuracy | p50 latency |
|---|---|---|---|---|---|---|
| nlu_handwritten.yaml | 132 | 97.7% | 59.8% | 0.982 | 100.0% | 0.63 ms |
| nlu_handwritten_v2.yaml | 76 | 93.4% | 60.5% | 0.940 | 100.0% | 0.59 ms |

`nlu_handwritten.yaml` (v1) was used during development and is optimistic. `nlu_handwritten_v2.yaml` was written after the models were frozen and scored once: it is the honest estimate.

Failures in nlu_handwritten.yaml:

| Utterance | Expected | Got | Slots |
|---|---|---|---|
| cardivex north q2 - whats going on | diagnose | metric_lookup | 3/3 |
| month on month sales of neurovita for the last 6 months | trend | metric_lookup | 3/3 |
| what's 15% of 2400 | out_of_scope | define_term | 0/0 |

Failures in nlu_handwritten_v2.yaml:

| Utterance | Expected | Got | Slots |
|---|---|---|---|
| Cardivex numbers in Punjab look terrible this quarter, what's the story? | diagnose | metric_lookup | 3/3 |
| any problem brands in central? | find_underperformers | drill_down | 1/1 |
| which region is hurting glucoryl the most | find_underperformers | drill_down | 1/1 |
| what's the share price of a pharma company | out_of_scope | define_term | 0/0 |
| how many holidays do I have left | out_of_scope | help | 0/0 |

### Model comparison (template-held-out test split)

| Intent model | Val macro-F1 | Test macro-F1 | Val ECE |
|---|---|---|---|
| keyword_rules | 0.322 | 0.452 | 0.608 |
| naive_bayes | 0.658 | 0.657 | 0.269 |
| tfidf_logreg | 0.636 | 0.610 | 0.083 |
| tfidf_linear_svm | 0.628 | 0.634 | 0.146 |
| tfidf_logreg_no_masking | - | 0.588 | - |

| Entity recognition | Precision | Recall | F1 |
|---|---|---|---|
| gazetteer_only | 0.925 | 0.869 | 0.896 |
| perceptron_no_gazetteer | 0.894 | 0.956 | 0.924 |
| perceptron_with_gazetteer | 0.937 | 0.998 | 0.967 |

## 3. Multi-turn dialogue

13 scripted conversations, 31 turns (follow-ups, ellipsis, clarification, carry-over, out-of-scope isolation).

- Conversation success: **100.0%**
- Turn accuracy: 100.0%
- Turn latency p50 11 ms, p95 81 ms

