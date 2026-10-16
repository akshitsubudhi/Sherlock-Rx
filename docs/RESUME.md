# Resume entry and interview notes

## Resume bullets (pick 3–4)

**Sherlock Rx: conversational root-cause analysis for pharma brands** · Python, NLP, DuckDB, FastAPI · [github.com/akshitsubudhi/Sherlock-Rx](https://github.com/akshitsubudhi/Sherlock-Rx)

- Built an NLP chatbot that diagnoses why pharma brands under-perform by region. It scores
  **98.8% top-1 root-cause accuracy** on 160 planted scenarios (6 cause types, 5 seeds) with
  **0% false alarms** on 100 control cells.
- Implemented the NLU stack from first principles: noisy-channel spell correction, TF-IDF and
  Naive Bayes (both verified against scikit-learn to 1e-9), a **structured-perceptron NER
  tagger with Viterbi decoding** (F1 0.967 vs 0.896 for a gazetteer), entity linking, an Indian
  fiscal-calendar temporal parser and BM25 retrieval. It reaches **93.4% intent / 100% slot
  accuracy** on a frozen hand-written test set (keyword baseline 60.5%).
- Designed a causal synthetic data generator (calls → prescriptions → sell-out → sales ×
  price) and an RCA engine combining a price-volume-mix bridge with **territory-level
  difference-in-differences**, so every answer carries an INR impact, a confidence score and
  evidence.
- Scaled to **30M rows / 7.6M sales rows**: 36 ms median chat turn, 253 ms cold RCA.
  Benchmarking showed threads underperforming processes and an in-memory roll-up that
  measured worse, so I removed it; multi-worker throughput rose 1.6×.
- Shipped a FastAPI service (sessions, metrics, typed contracts), a Streamlit UI with an NLP
  trace panel, 71 tests with acceptance gates, `mypy --strict`, and CI that retrains from scratch.

## 60-second pitch

"Sales leaders spend review meetings asking *why* a brand missed in a region. The answer is
spread across secondary sales, field-force data, prescriptions, stock and the market audit.
I built a chatbot that answers in one question. The language side is classical NLP I wrote
myself, so it's fast (under a millisecond), cheap and explainable. The analytics side is
deterministic, so it never makes up a number. Because I can't use real company data, I
built a generator that simulates the causal chain and plants known root causes. That
let me *measure* the system: 98.8% of planted causes found, no false alarms, and an
accuracy curve as the causes get weaker."

## Questions you should expect, and short answers

**Why not just use an LLM?** Three reasons. Numbers must be exact and auditable, and
an LLM can't guarantee that. Latency and cost: 0.6 ms vs seconds and tokens per turn. And
explainability: I can show the features behind every intent decision. The LLM is still used
where it adds value, as a fallback for low-confidence utterances, with its output forced
through the same entity linker.

**How do you know the RCA is right?** Planted ground truth with placement rules, 5 seeds,
control cells for false alarms, and a robustness sweep. On real data I'd back-test against
cases the commercial team already diagnosed.

**Why difference-in-differences?** A vacancy's cost is "how much worse did these
territories do than comparable ones that had no vacancy", after adjusting for their own
market's growth. It needs no elasticity assumption.

**Your NLU test F1 is 0.61 but the hand-written accuracy is 93%. Which is true?** Both
measure different things. The template split holds out whole phrasings and the training
vocabulary for them; it's a stress test. The production model is refit on all templates.
The hand-written v2 set was written after freezing and scored once, so it's the honest
estimate, with a ±6-point interval because it's small.

**Why logistic regression over Naive Bayes when NB had higher F1?** NB was confidently
wrong (ECE 0.27), and the clarify/out-of-scope gate depends on calibrated confidence.

**What would you do with real data?** Swap the warehouse (the SQL already transpiles to
Snowflake), map the real schema into the semantic layer, back-test the RCA against
historical reviews, collect real user utterances to retrain the NLU, and add a
plan-vs-actual decomposition for chronic gaps.

**What was the hardest bug?** The engine blaming a market slowdown on a region where
nothing was planted. Another region's rebound from *last year's* planted slowdown had
inflated the pooled peer benchmark. Switching to a weighted median fixed it. It's a real
analytics lesson: baselines get contaminated.
