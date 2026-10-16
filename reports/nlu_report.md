# NLU evaluation report

Trained 2026-10-17 04:14:37 · seed 13 · corpus `b52bb1ff543f` · 73.9s on CPU

Corpus (split **by template**): {'train': 4515, 'val': 1157, 'test': 1184}

## Intent classification (macro-F1)

| Model | Val F1 | Test F1 (unseen templates) | Val ECE |
|---|---|---|---|
| keyword_rules | 0.322 | 0.452 | 0.608 |
| naive_bayes | 0.658 | 0.657 | 0.269 |
| tfidf_logreg | 0.636 | 0.610 | 0.083 |
| tfidf_linear_svm | 0.628 | 0.634 | 0.146 |
| tfidf_logreg_no_masking | - | 0.588 | - |

Selected: **tfidf_logreg**, confidence threshold 0.3 (≥98% precision on accepted val predictions).

## Entity recognition (entity-level micro-F1, unseen templates)

| Approach | Precision | Recall | F1 |
|---|---|---|---|
| gazetteer_only | 0.925 | 0.869 | 0.896 |
| perceptron_no_gazetteer | 0.894 | 0.956 | 0.924 |
| perceptron_with_gazetteer | 0.937 | 0.998 | 0.967 |

## Hand-written test set (end-to-end pipeline)

* 132 utterances written by hand (typos, shorthand, Hinglish)
* Intent accuracy **0.977** (macro-F1 0.982); keyword-rule baseline 0.598
* Slot accuracy **1.000**: baseline 1.00, brands 1.00, competitors 1.00, dimension 1.00, metrics 1.00, period 1.00, regions 1.00, skus 1.00, states 1.00, territories 1.00, therapies 1.00

### Failures

| Utterance | Expected | Got | Conf | Slots |
|---|---|---|---|---|
| cardivex north q2 - whats going on | diagnose | metric_lookup | 0.75 | 3/3 |
| month on month sales of neurovita for the last 6 months | trend | metric_lookup | 0.51 | 3/3 |
| what's 15% of 2400 | out_of_scope | define_term | 0.65 | 0/0 |
