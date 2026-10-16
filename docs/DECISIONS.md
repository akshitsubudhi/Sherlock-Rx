# Decision log

Each row: what happened, what I measured, what I decided.

| # | Context | Measurement | Decision |
|---|---|---|---|
| 1 | First RCA evaluation: one miss was a price cut "lost" to a competitor | The planner had let a persisting competitor entrant overlap the price-cut cell, so the ground truth itself was ambiguous | Planner rejects competitor launches whose later quarters are already occupied. Fix the test before the model. |
| 2 | Stock-out diagnosed as competitor launch | Competitors' *national* share drift was counted as local excess gain | Expected competitor share now uses each competitor's momentum outside the region. Top-1: 91% → 97% |
| 3 | Market slowdown blamed in East although nothing was planted there | South's prior year held a planted slowdown; its +37% rebound inflated the pooled rest-of-India growth | Peer benchmark = weighted median of states. 5-seed top-1 → 100% at that stage |
| 4 | A planted competitor launch in Goa alone was "missed" | Goa is about 2% of West sales; the region-level impact was 0.5%, below materiality | Planted state-level causes must cover ≥ 25% of the region. The engine was right; the scenario wasn't a regional problem |
| 5 | Nearly every brand-region "underperforming" | Typical achievement 94%: target 1.04× expected, while stock availability is about 98% | Plan = expected demand at normal availability. Normal-quarter achievement now 97–99% |
| 6 | 4 of 8 brands had NPPA cuts in the same quarter | Unrealistic concentration that distorted the portfolio scan | 2 price cuts per seed (10 cases over 5 seeds) |
| 7 | Intent split-F1 0.54, v1 81.8% | Small-talk intents are single phrases, so a held-out template is an unseen word; filler noise created "ok hey please" as *greet* | More templates, EDA synonym replacement, no fillers on small talk. Split-F1 0.65 |
| 8 | Naive Bayes selected but confidently wrong | NB ECE 0.269 (confidence 1.0 on errors) vs LogReg 0.083 | Select on F1 subject to ECE ≤ 0.15, then refit on all templates. v1 86.4% → 97.0% |
| 9 | "july" corrected to "jul", "number" to "numbers" | The spell corrector treated unseen valid words as typos | Protected common words and month names; reject inflection-only edits |
| 10 | "Glucoryl market share in Rajasthan" became a diagnosis | The entity-only ellipsis rule fired on a METRIC mention | Ellipsis only for brand/geo/time mentions; METRIC/DIM/BASELINE carry intent |
| 11 | 8 threads slower than 1 | RCA is pandas-bound (GIL) | Scale with processes; documented in SCALING.md |
| 12 | 4 processes collapse at ×100 | DuckDB oversubscription + a per-process in-memory copy | Thread cap per worker; removed the roll-up (no single-process gain, 5.0 → 8.1 RCA/s) |
| 13 | First diagnosis 1.3 s at ×50 | Lazy BM25 index build over 28k notes | Build at start-up |
| 14 | Field reports "proved" causes too easily? | Notes are written for planted scenarios | Notes are retrieved as corroboration only, never used for scoring |
