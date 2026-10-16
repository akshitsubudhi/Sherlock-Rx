"""Free-text field reports (MR / ABM / RBM notes) — the unstructured evidence source.

Notes are written for planted scenarios, for background noise events, and as routine
chatter. The RCA engine never scores from notes; the chatbot retrieves them (BM25) as
corroborating qualitative evidence, which is how field intelligence is used in practice.
"""

from __future__ import annotations

import numpy as np

CAUSE_TEMPLATES: dict[str, tuple[str, ...]] = {
    "field_force_vacancy": (
        "{hq}: MR position for {brand} lying vacant, key doctors not being covered this month.",
        "Two MRs resigned in {state}; replacement hiring pending, call average has dropped.",
        "Territory {hq} vacant since last month. Competition is meeting our core prescribers.",
        "Attrition in {state} team. {brand} detailing stopped in {hq} and nearby towns.",
        "ABM covering {hq} on a part-time basis after the MR left; coverage of top doctors is low.",
    ),
    "stockout": (
        "{sku} not available at C&F {state} for {days} days; stockists reporting shortage.",
        "Supply issue: {sku} out of stock across chemists in {hq}, patients being switched to {comp}.",
        "Retailers in {hq} complaining that {sku} orders are not being serviced by the depot.",
        "Short supply of {sku} from the plant. Doctors still prescribing but chemists substituting.",
    ),
    "competitor_launch": (
        "{comp} launched in {state} at a lower PTR with heavy retailer schemes.",
        "Doctors in {hq} trialing {comp}; their team is offering bonus deals to chemists.",
        "New entrant {comp} is aggressively detailing our {brand} prescribers in {hq}.",
        "{comp} price is about 20 percent below {brand}; some loyal doctors have shifted.",
    ),
    "price_cut": (
        "NPPA ceiling price notification for {molecule} implemented; PTR of {sku} revised down.",
        "Revised MRP for {sku} effective this month after the ceiling price order. Volumes steady.",
        "Price reduction on {sku} under DPCO, value per strip lower but chemists ordering normally.",
    ),
    "market_slowdown": (
        "Overall {therapy} market slow in {state}; OPD footfall down because of {reason}.",
        "Doctors report fewer {therapy} patients this season due to {reason}. Whole market is soft.",
        "Category demand weak across {state}, all companies seeing lower offtake ({reason}).",
    ),
    "distribution_loss": (
        "Stockist in {hq} stopped lifting {brand} over a margin dispute.",
        "Chemist association boycott in {state} affecting our retail coverage.",
        "Lost two major distributors in {hq}; many retailers no longer stock {brand}.",
        "Retail reach down in {hq} after the super-stockist changed; outlets not being serviced.",
    ),
}

SLOWDOWN_REASON: dict[str, tuple[str, ...]] = {
    "Respiratory": ("a mild winter", "low pollution levels this year"),
    "Anti-infectives": ("a weak monsoon", "a lower infection season"),
    "Dermatology": ("a cooler summer", "lower humidity"),
}
DEFAULT_REASON = ("lower OPD footfall", "a slow season in clinics")

ROUTINE_TEMPLATES: tuple[str, ...] = (
    "Conducted CME on {therapy} with {n} doctors in {hq}. Good response to {brand}.",
    "Doctors in {hq} positive on {brand} efficacy; requested more samples.",
    "Routine review meeting held in {state}; monthly plan discussed with the team.",
    "Camp organised at a {hq} hospital, {n} patients screened.",
    "Key account visit in {hq} went well, {brand} added to the hospital formulary.",
    "Chemist meet in {hq}; retailers happy with {brand} availability and margins.",
    "{brand} brand reminder cards distributed to {n} clinics in {hq}.",
)


def scenario_note(cause: str, rng: np.random.Generator, **ctx: str) -> str:
    tpl = str(rng.choice(CAUSE_TEMPLATES[cause]))
    if cause == "market_slowdown":
        ctx.setdefault("reason", str(rng.choice(SLOWDOWN_REASON.get(ctx["therapy"], DEFAULT_REASON))))
    ctx.setdefault("days", str(int(rng.integers(8, 22))))
    return tpl.format(**ctx)


def routine_note(rng: np.random.Generator, **ctx: str) -> str:
    ctx.setdefault("n", str(int(rng.integers(12, 60))))
    return str(rng.choice(ROUTINE_TEMPLATES)).format(**ctx)
