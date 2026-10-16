"""Natural-language generation: data-to-text templates with Indian number formatting.

Every number rendered here was computed by the analytics layer; NLG only formats and
orders facts (no number is ever produced by a language model). Phrasing varies through
seeded template choice so repeated answers do not read like a form letter.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from sherlock_rx.analytics.rca import RCAReport
from sherlock_rx.nlp.retrieval import Hit

_rng = np.random.default_rng(3)


def pick(*options: str) -> str:
    return str(_rng.choice(options))


def indian_group(n: float) -> str:
    """12345678 -> '1,23,45,678' (lakh/crore digit grouping)."""
    neg = n < 0
    s = f"{abs(round(n)):d}"
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        parts: list[str] = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        s = ",".join([*parts, tail])
    return ("-" if neg else "") + s


def inr(v: float, signed: bool = False) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    sign = ("+" if v > 0 else "-" if v < 0 else "") if signed else ("-" if v < 0 else "")
    a = abs(v)
    if a >= 1e7:
        body = f"₹{a / 1e7:.2f} Cr"
    elif a >= 1e5:
        body = f"₹{a / 1e5:.1f} L"
    else:
        body = f"₹{indian_group(a)}"
    return sign + body


def pct(v: float | None, signed: bool = True, digits: int = 1) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return f"{v * 100:+.{digits}f}%" if signed else f"{v * 100:.{digits}f}%"


def fmt_value(v: float, unit: str) -> str:
    if unit == "inr":
        return inr(v)
    if unit == "pct":
        return pct(v, signed=False)
    if unit == "days":
        return f"{v:.1f} days"
    return indian_group(v)


# ---------------------------------------------------------------------- RCA narrative
def rca_markdown(r: RCAReport, notes: list[Hit]) -> str:
    h = r.headline
    where = r.state or r.region
    icon = {"Underperforming": "🔴", "Slightly below plan": "🟠", "On track": "🟢"}[h.status]
    lines = [
        f"**{r.brand} · {where} · {r.period['label']}**: {icon} {h.status}",
        "",
        f"Sales {inr(h.actual)} vs target {inr(h.target)} (**{pct(h.achievement, False)}** "
        f"achievement), {pct(h.yoy)} YoY against {pct(h.peer_yoy)} for the rest of India.",
        "",
    ]
    material = [c for c in r.causes if c.material]
    if not material and h.status != "On track" and h.yoy >= h.peer_yoy - 0.02:
        lines.append(
            f"This is a gap **against plan, not a deterioration**: sales grew {pct(h.yoy)} YoY, "
            f"in line with or ahead of the rest of India ({pct(h.peer_yoy)}), and no driver "
            "(coverage, supply, distribution, competition, price, market) shows a material "
            "problem. The target looks stretched for this geography; worth reviewing with "
            "the planning team.")
    elif not material:
        lines.append(pick(
            "I don't see a material root cause: no single driver explains more than 1.5% of "
            "last year's sales. The gap looks like normal variation.",
            "Nothing stands out as a root cause; every driver I checked is within normal range.",
        ))
        if r.causes:
            c = r.causes[0]
            lines.append(f"The largest (minor) factor is *{c.label.lower()}* at {inr(c.impact)}.")
    else:
        top = material[0]
        lines += [
            f"**Most likely root cause: {top.label}** ({top.confidence_label} confidence). "
            f"Impact ≈ {inr(top.impact)} ({pct(top.pct_of_py)} of last year's sales)"
            + (f", concentrated in {', '.join(top.locations)}." if top.locations else "."),
        ]
        lines += [f"- {e.text}" for e in top.evidence]
        others = material[1:3]
        if others:
            lines += ["", "Also contributing: " + "; ".join(
                f"{c.label.lower()} {inr(c.impact)} ({c.confidence_label.lower()} confidence)"
                for c in others) + "."]
    if r.offsets:
        lines += ["", "Offsets: " + ", ".join(f"{o.label.lower()} {inr(o.impact, True)}"
                                             for o in r.offsets) + "."]
    gaps = [s for s in r.states if s["gap_vs_peers"] < 0]
    if gaps and not r.state:
        s0 = gaps[0]
        lines += ["", f"**Where:** {s0['state']} accounts for {pct(s0['share_of_gap'], False, 0)} "
                      f"of the shortfall vs peer growth ({pct(s0['yoy'])} YoY)."]
    terr = [t for t in r.territories if t["shortfall"] > 0][:3]
    if terr:
        lines.append("Worst territories: " + ", ".join(
            f"{t['territory']} ({inr(-t['shortfall'])}{'; ' + t['flags'] if t['flags'] else ''})"
            for t in terr) + ".")
    if notes:
        lines += ["", "**From the field:**"]
        for n in notes[:2]:
            m = n.doc.meta
            lines.append(f"> “{n.doc.text}”  ({m['role']}, {m['territory']}, {m['month']})")
    if material:
        lines += ["", f"**Suggested action:** {material[0].recommendation}"]
    lines += ["", f"_Method: YoY bridge + territory difference-in-differences · "
                  f"{len(r.query_ids)} queries · {r.elapsed_ms:.0f} ms_"]
    return "\n".join(lines)


def scan_markdown(df: pd.DataFrame, period_label: str, scope_label: str, n: int = 6) -> str:
    bad = df[df.status != "On track"].head(n)
    if bad.empty:
        return (f"Everything in {scope_label} is on track for {period_label}: every brand-region "
                "is at or above 98% of target and growing in line with the rest of India. 🟢")
    lines = [f"**Under-performers in {scope_label}, {period_label}** (ranked by severity)", "",
             "| # | Brand | Region | Achievement | YoY | vs rest of India | Status |",
             "|---|---|---|---|---|---|---|"]
    for i, r in enumerate(bad.itertuples(), 1):
        lines.append(f"| {i} | {r.brand} | {r.region} | {pct(r.achievement, False)} | "
                     f"{pct(r.yoy)} | {pct(r.growth_gap)} pts | {r.status} |")
    top = bad.iloc[0]
    flagged = df[df.status == "Underperforming"].groupby("brand").region.nunique()
    national = [str(b) for b, n in flagged.items() if n >= 4]
    if national:
        lines += ["", f"Note: {', '.join(national)} {'is' if len(national) == 1 else 'are'} "
                      "below plan in 4+ regions, which points to a *national* driver (price, "
                      "supply, or the plan itself) rather than a regional one."]
    lines += ["", f"The most urgent is **{top.brand} in {top.region}**. Ask *“why?”* and "
                  "I'll run a root-cause analysis on it."]
    return "\n".join(lines)


def metric_markdown(s: dict[str, Any], scope_label: str, brand_label: str, period_label: str
                    ) -> str:
    v = fmt_value(s["value"], s["unit"])
    parts = [f"**{s['label']}** for {brand_label} in {scope_label}, {period_label}: **{v}**"]
    if s.get("yoy") is not None and s["unit"] != "pct":
        parts.append(f"({pct(s['yoy'])} YoY; last year {fmt_value(s['last_year'], s['unit'])})")
    elif s["unit"] == "pct" and s.get("last_year") == s.get("last_year"):
        parts.append(f"(last year {fmt_value(s['last_year'], 'pct')}, "
                     f"{(s['value'] - s['last_year']) * 100:+.1f} pts)")
    text = " ".join(parts) + "."
    if s.get("achievement") is not None:
        text += (f" Target {inr(s['target'])}, achievement **{pct(s['achievement'], False)}**.")
    return text


def table_markdown(df: pd.DataFrame, key: str, title: str) -> str:
    cols = [c for c in ("value", "last_year", "yoy", "target", "achievement") if c in df]
    head = {"value": "Actual", "last_year": "Last year", "yoy": "YoY", "target": "Target",
            "achievement": "Achievement"}
    lines = [f"**{title}**", "", "| " + key.title() + " | " + " | ".join(head[c] for c in cols)
             + " |", "|---" * (len(cols) + 1) + "|"]
    for r in df.itertuples():
        cells = []
        for c in cols:
            v = getattr(r, c)
            cells.append(pct(v, c == "yoy") if c in ("yoy", "achievement") else inr(v))
        name = getattr(r, key)
        if getattr(r, "highlight", False):
            name = f"**{name}**"
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def notes_markdown(hits: list[Hit], scope_label: str) -> str:
    if not hits:
        return f"I couldn't find field reports matching that for {scope_label}."
    lines = [f"**Field reports, {scope_label}** (BM25 ranked)", ""]
    for h in hits:
        m = h.doc.meta
        lines.append(f"- “{h.doc.text}”  ({m['role']}, {m['territory']}, {m['month']}"
                     f"{', ' + m['brand'] if m['brand'] else ''}) · score {h.score}")
    return "\n".join(lines)
