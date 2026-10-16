# Evaluation methodology

Run everything with `make eval` → [`reports/EVAL.md`](../reports/EVAL.md). Acceptance gates
in `tests/` fail CI if RCA top-1 < 90%, NLU intent on the frozen set < 90% or slots < 95%,
or dialogue success < 90%.

## RCA: planted ground truth

* 32 scenarios per seed across 6 cause types; 5 seeds → 160 scenarios; 100 control cells.
* **Placement rules** keep the test fair. No two scenarios share a brand × region × quarter
  cell or each other's prior-year cell. A competitor entrant (which persists) blocks later
  quarters. National price cuts own every region of their brand. State-level causes must
  cover ≥ 25% of the region, so a planted cause is actually material at region level.
* Metrics: top-1 / top-2 cause, location (top state ∈ planted states), flagged rate, false
  alarms on controls (a material cause ≥ 3% of sales on an "underperforming" control cell).
* **Robustness sweep**: `severity_scale` weakens planted causes, `noise_scale` amplifies
  every noise source. Accuracy degrades smoothly: 98.8% → 88.5% at 0.35× severity.

**Caveat.** The data is synthetic, and the engine was developed against it. The engine's
logic is general (DiD, share decomposition, signatures), but real data has messier
baselines, overlapping causes and missing feeds. The sweep is the best available proxy.

## NLU: three test sets of increasing honesty

1. **Template-held-out split.** The corpus is split *by template* (60/15/25), so test
   phrasings are never seen in training. Scores are low (macro-F1 0.61–0.66) because whole
   cue phrases are held out. That's the point: it measures generalisation, not memorisation.
2. **Hand-written v1** (132 utterances with typos, shorthand and Hinglish). Used during
   development, so it is optimistic: **97.7%**.
3. **Hand-written v2** (76 utterances). Written after models and dialogue were frozen, and
   scored once: **93.4% intent, 100% slots** (keyword-rule baseline 60.5%).

### Model selection

* Models: keyword rules, Naive Bayes (scratch), TF-IDF (scratch) + logistic regression,
  TF-IDF + linear SVM (calibrated).
* Selection uses validation macro-F1 **subject to calibration** (ECE ≤ 0.15). The
  out-of-scope / clarify gate needs probabilities that mean something. Naive Bayes had the
  best F1 (0.658) but ECE 0.269, with confidence 1.0 on its errors, so logistic regression
  (ECE 0.083) was chosen.
* The confidence threshold is the lowest value keeping ≥ 98% precision on accepted
  validation predictions.
* **Production refit.** After selection, the chosen configuration is refit on all
  templates. The split estimates generalisation; holding templates back from the shipped
  model would only remove vocabulary. Hand-written sets are independent of templates.

### What changed between rounds (and why it isn't leakage)

| Round | Change | Template test F1 | v1 |
|---|---|---|---|
| 1 | baseline corpus | 0.54 | 81.8% |
| 2 | +templates, EDA synonym replacement, no filler noise on small talk | 0.65 | 86.4% |
| 3 | ECE-aware selection + production refit | 0.61* | 97.0% |
| 4 | spell corrector stops "fixing" valid words; one-token entity placeholders | 0.61 | 97.7% |

\* LogReg replaced NB as the selected model; its split F1 is lower, but it is calibrated.

The v1 failures were *read* between rounds, which is why v1 is reported as optimistic and
v2 exists. New templates were written from domain knowledge, not copied from v1 utterances.

## Dialogue

13 scripted conversations (`tests/data/dialogues.yaml`) check intent, policy, dialogue-state
slots and reply content per turn: ranking → why → drill-down → actions; ellipsis region
switch; brand switch keeping geography; clarification resume; automatic brand or region
choice; typos; Hinglish; out-of-scope isolation; glossary inside a task; field reports
using the diagnosis context. Result: 13/13 conversations, 31/31 turns.
