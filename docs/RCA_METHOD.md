# Root-cause analysis method

Code: [`src/sherlock_rx/analytics/rca.py`](../src/sherlock_rx/analytics/rca.py). Every
comparison is the asked period versus the **same months last year** (YoY), which removes
seasonality: respiratory brands peak in winter, anti-infectives in the monsoon.

## 1. Headline

| Quantity | Definition |
|---|---|
| Achievement | actual secondary sales ÷ target, same territories and months |
| YoY | actual ÷ same months last year − 1 |
| Peer YoY | **weighted median** of the brand's YoY across states outside the region |
| Status | *Underperforming* if achievement < 95% or YoY trails peers by > 4 pts; *Slightly below plan* at 98% / 2 pts |

The peer benchmark is a weighted median, not a pooled average. That's a deliberate fix: a
region bouncing back from last year's problem (+37% YoY) dragged the pooled average up and
made unrelated regions look slow (see [DECISIONS.md](DECISIONS.md) #3).

## 2. Bridge (it adds up exactly)

```
last year ─► + market effect ─► + share effect ─► + price effect ─► this period
```

* **Price effect** = Σ_sku (p₁ − p₀) · q₁, computed **per SKU**. A price cut on one strength is
  not confused with a mix shift between strengths.
* **Market effect** = last year × market growth (molecule units, state-level market audit).
* **Share effect** = the remainder at constant (last-year) prices.

The test `test_bridge_adds_up_exactly` asserts the steps sum to actual to 1e-9.

## 3. Candidate causes and how each is estimated

| Cause | Estimator | Signature that raises confidence |
|---|---|---|
| Market slowdown | last year × (peer market growth − regional market growth) | our volume share is stable |
| Price (NPPA/DPCO) | SKU price effect | units steady; same cut seen in peer regions → *national* |
| Field-force vacancy | territory diff-in-diff, treated = calls −20%+ | prescriptions fall with sales |
| Stock-out | territory diff-in-diff, treated = +3 stock-out days/SKU-month | prescriptions **hold** while sales fall |
| Distribution loss | territory diff-in-diff, treated = active retailers −10%+ | prescriptions **hold** while sales fall |
| Competitor launch | share lost to a competitor gaining *more than proportional* share | new entrant (≈0 share last year) |

**Difference-in-differences at territory level.** For each territory:
`relative growth = (sales at last year's prices ÷ last year) ÷ (1 + its state's market growth)`.
The control is the sales-weighted mean relative growth of *untreated* territories nationally
(no driver flag, not in a competitor-hit state, 5–95% trimmed). A treated territory's
shortfall is `last year × (1 + market growth) × control − actual`. That shortfall is split
across the drivers flagged there, in proportion to how strongly each moved.

**Competitor excess share.** When we lose share, competitors gain it. The question is
whether one gained *more than its fair share*. The expected share is computed as our lost
share redistributed in proportion to each competitor's last-year share, **adjusted by that
competitor's national momentum** (its trend outside the region). Only the excess counts.
Without the momentum term, ordinary national drift was blamed on local problems
([DECISIONS.md](DECISIONS.md) #2).

**No double counting.** Share-type drivers are scaled down if together they claim more
than 115% of the share loss.

## 4. Ranking, materiality, confidence

* Causes are ranked by INR impact. A cause is **material** if it explains ≥ 1.5% of last
  year's sales; below that the answer says nothing stands out.
* Confidence = base + signature consistency + breadth (how many territories show it),
  clipped to [0, 1]; labels High ≥ 0.75, Medium ≥ 0.5.
* When the brand is below target but growing in line with peers and no cause is material,
  the answer says it is **a gap against plan, not a deterioration** (the target is stretched).

## 5. Drill-down and evidence

States are ranked by their gap against peer growth; territories by DiD shortfall, with the
flags that fired. Field notes are retrieved with BM25, filtered to the brand, states and
months of the case, and expanded with the top cause's vocabulary (stock-out → shortage,
supply, depot, C&F). They corroborate the answer; they are **never** used to score causes,
so the evaluation can't be gamed by the notes.

## 6. Known blind spots

* YoY attribution explains *change*. A chronic gap from a cause older than 12 months shows as
  "a gap against plan".
* Two simultaneous causes in the same territories are split by a heuristic weight, not
  identified separately.
* The playbook recommendations are fixed text per cause.
