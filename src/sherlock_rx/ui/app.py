"""Streamlit front-end: chat on the left, "under the hood" NLP trace and evidence charts.

Run:  streamlit run src/sherlock_rx/ui/app.py
Set SHERLOCK_API_URL to talk to a running API instead of loading the bot in-process.
"""

from __future__ import annotations

import html
import json
import os
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Reference palette (validated): diverging blue <-> red with neutral gray; categorical 1-2.
GAIN, LOSS, NEUTRAL = "#2a78d6", "#e34948", "#8a8984"
SERIES_1, SERIES_2 = "#2a78d6", "#eb6834"
ENTITY_COLORS = {"BRAND": "#2a78d6", "SKU": "#2a78d6", "REGION": "#1baf7a", "STATE": "#1baf7a",
                 "TERRITORY": "#1baf7a", "TIME": "#eda100", "METRIC": "#e87ba4",
                 "DIM": "#4a3aa7", "BASELINE": "#4a3aa7", "COMPETITOR": "#eb6834",
                 "THERAPY": "#008300"}

st.set_page_config(page_title="Sherlock Rx", page_icon="🕵️", layout="wide")


# ---------------------------------------------------------------------- backend
@st.cache_resource(show_spinner="Loading warehouse and NLU models…")
def local_bot():  # type: ignore[no-untyped-def]
    from sherlock_rx.cli import build_bot

    return build_bot()


def ask(message: str) -> dict[str, Any]:
    api = os.environ.get("SHERLOCK_API_URL")
    if api:
        import httpx

        r = httpx.post(f"{api}/v1/chat", json={"message": message,
                                               "session_id": st.session_state.get("sid")},
                       timeout=60)
        r.raise_for_status()
        out: dict[str, Any] = r.json()
        st.session_state.sid = out["session_id"]
        return out
    from sherlock_rx.api.main import to_jsonable

    bot = local_bot()
    if "dstate" not in st.session_state:
        st.session_state.dstate = bot.new_state()
    resp = bot.handle(message, st.session_state.dstate)
    return to_jsonable(resp.to_dict())  # type: ignore[no-any-return]


# ---------------------------------------------------------------------- charts
def _layout(fig: go.Figure, title: str, height: int = 320) -> go.Figure:
    fig.update_layout(title=dict(text=title, x=0, font=dict(size=14)), height=height,
                      margin=dict(l=10, r=10, t=40, b=10), showlegend=False,
                      hoverlabel=dict(font_size=12))
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.15)", zeroline=False)
    fig.update_xaxes(showgrid=False)
    return fig


def bridge_chart(bridge: list[dict[str, Any]]) -> go.Figure:
    """What moved sales vs last year. Only the moves are plotted (zero baseline), so a
    ₹1 Cr swing is visible next to ₹16 Cr totals without truncating an axis."""
    first, last = bridge[0]["value"] / 1e7, bridge[-1]["value"] / 1e7
    moves = bridge[1:-1]
    vals = [m["value"] / 1e7 for m in moves]
    fig = go.Figure(go.Bar(
        x=[m["step"] for m in moves], y=vals,
        marker=dict(color=[GAIN if v >= 0 else LOSS for v in vals], cornerradius=4),
        text=[f"{v:+.2f}" for v in vals], textposition="outside", cliponaxis=False,
        hovertemplate="%{x}: %{y:+.2f} Cr<extra></extra>"))
    fig.add_hline(y=0, line_color=NEUTRAL, line_width=1)
    lo, hi = min(0.0, *vals), max(0.0, *vals)
    pad = 0.25 * (hi - lo or 1.0)  # headroom so outside labels are never clipped
    fig.update_yaxes(title_text="₹ crore vs last year", range=[lo - pad, hi + pad])
    return _layout(fig, f"What moved sales: ₹{first:.2f} Cr last year → ₹{last:.2f} Cr now")


def causes_chart(causes: list[dict[str, Any]]) -> go.Figure:
    rows = [c for c in causes if c["impact"] < 0][:6][::-1]
    fig = go.Figure(go.Bar(
        y=[c["label"] for c in rows], x=[-c["impact"] / 1e5 for c in rows], orientation="h",
        marker=dict(color=[LOSS if c["material"] else NEUTRAL for c in rows], cornerradius=4),
        text=[f"{c['confidence_label']} confidence{'' if c['material'] else ' · immaterial'}"
              for c in rows], textposition="auto",
        hovertemplate="%{y}<br>₹%{x:.1f} L lost<extra></extra>"))
    fig.update_xaxes(title_text="Estimated sales lost (₹ lakh)")
    return _layout(fig, "Candidate root causes by impact", 260)


def trend_chart(data: list[dict[str, Any]], unit: str) -> go.Figure:
    df = pd.DataFrame(data)
    scale, suffix = (1e7, " Cr") if unit == "inr" else (1, "")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.month, y=df.last_year / scale, name="Last year",
                             mode="lines+markers", line=dict(color=SERIES_2, width=2),
                             marker=dict(size=8),
                             hovertemplate=f"Last year: %{{y:.2f}}{suffix}<extra></extra>"))
    fig.add_trace(go.Scatter(x=df.month, y=df.value / scale, name="This year",
                             mode="lines+markers", line=dict(color=SERIES_1, width=2),
                             marker=dict(size=8),
                             hovertemplate=f"This year: %{{y:.2f}}{suffix}<extra></extra>"))
    _layout(fig, "This year vs last year", 300)
    fig.update_layout(showlegend=True, hovermode="x unified",
                      legend=dict(orientation="h", y=1.12, x=1, xanchor="right"))
    return fig


def highlight(text: str, entities: list[dict[str, Any]]) -> str:
    """Colour entity spans in the (normalised) utterance."""
    out, pos = [], 0
    low = text.lower()
    for e in sorted(entities, key=lambda e: e["start"]):
        if e["start"] < pos:
            continue
        out.append(html.escape(low[pos:e["start"]]))
        c = ENTITY_COLORS.get(e["type"], NEUTRAL)
        out.append(f"<span style='border-bottom:3px solid {c};padding:0 2px' "
                   f"title='{e['type']} → {html.escape(str(e['value']))} ({e['source']})'>"
                   f"{html.escape(low[e['start']:e['end']])}<sub style='color:{c};"
                   f"font-size:0.65em'> {e['type']}</sub></span>")
        pos = e["end"]
    out.append(html.escape(low[pos:]))
    return "".join(out)


def render_payload(p: dict[str, Any] | None) -> None:
    if not p:
        return
    kind = p.get("type")
    if kind == "rca":
        r = p["data"]
        st.plotly_chart(bridge_chart(r["bridge"]), use_container_width=True)
        if r["causes"]:
            st.plotly_chart(causes_chart(r["causes"]), use_container_width=True)
        with st.expander("States and territories"):
            st.dataframe(pd.DataFrame(r["states"]), hide_index=True, use_container_width=True)
            st.dataframe(pd.DataFrame(r["territories"]), hide_index=True, use_container_width=True)
    elif kind == "trend":
        st.plotly_chart(trend_chart(p["data"], p.get("unit", "inr")), use_container_width=True)
    elif kind in ("scan", "table", "notes"):
        with st.expander("Data", expanded=False):
            st.dataframe(pd.DataFrame(p["data"]), hide_index=True, use_container_width=True)


def render_trace(resp: dict[str, Any]) -> None:
    n = resp["nlu"]
    st.markdown("**Entities** (gazetteer · tagger · temporal)")
    st.markdown(f"<div style='font-size:1.05em;line-height:2.2'>"
                f"{highlight(n['corrected'], n['entities'])}</div>", unsafe_allow_html=True)
    if n["corrections"]:
        st.caption("Spell-corrected: " + ", ".join(f"{a} → {b}" for a, b in n["corrections"]))
    st.markdown("**Intent**")
    top = pd.DataFrame(n["top"], columns=["intent", "p"])
    st.dataframe(top, hide_index=True, use_container_width=True,
                 column_config={"p": st.column_config.ProgressColumn(
                     "probability", min_value=0.0, max_value=1.0, format="%.2f")})
    st.caption(f"Policy: **{resp['policy']}** · masked input: `{n['masked']}`")
    if n.get("explanation"):
        st.caption("Top features: " + ", ".join(f"`{f}` {w:+.2f}" for f, w in n["explanation"]))
    st.markdown("**Dialogue state**")
    st.code(json.dumps(resp["state"], indent=1), language="json")
    st.caption(" · ".join(f"{k} {v} ms" for k, v in resp["timings_ms"].items()))


# ---------------------------------------------------------------------- page
st.title("🕵️ Sherlock Rx")
st.caption("Conversational root-cause analysis for under-performing pharma brands, by region. "
           "Synthetic data with planted causes · classical NLP pipeline · deterministic analytics.")

if "history" not in st.session_state:
    st.session_state.history = []

chat_col, trace_col = st.columns([3, 2], gap="large")

with chat_col:
    for turn in st.session_state.history:
        with st.chat_message("user"):
            st.markdown(turn["user"])
        with st.chat_message("assistant", avatar="🕵️"):
            st.markdown(turn["resp"]["text"])
            render_payload(turn["resp"].get("payload"))

    last = st.session_state.history[-1]["resp"] if st.session_state.history else None
    suggestions = (last or {}).get("suggestions") or [
        "Which brands are underperforming?", "Why is Cardivex down in North?", "Help"]
    picked = None
    cols = st.columns(len(suggestions[:4]))
    for i, s in enumerate(suggestions[:4]):
        if cols[i].button(s, key=f"sugg-{len(st.session_state.history)}-{i}",
                          use_container_width=True):
            picked = s
    typed = st.chat_input("Ask Sherlock… e.g. why is Glucoryl underperforming in the South?")
    message = typed or picked
    if message:
        resp = ask(message)
        st.session_state.history.append({"user": message, "resp": resp})
        st.rerun()

with trace_col:
    st.subheader("Under the hood")
    if last:
        render_trace(last)
    else:
        st.info("Each answer shows how it was understood: spell correction, entities, intent "
                "probabilities, the policy decision and the dialogue state.")
    if st.button("Reset conversation"):
        for k in ("history", "dstate", "sid"):
            st.session_state.pop(k, None)
        st.rerun()
