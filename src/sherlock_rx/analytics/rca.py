"""Root-cause analysis engine. Deterministic: every number in an answer comes from here.

Method (all comparisons are the selected period vs the same months last year):

1. Headline: actual vs target, vs last year, and vs the brand's growth in the rest of India.
2. Bridge:  PY sales -> market growth -> share change -> price -> actual.
            Price/volume split is SKU-level, so a price cut on one SKU does not masquerade
            as a volume problem. Share is volume share from the state market audit.
3. Candidate causes, each with an INR impact estimate:
   * market slowdown   regional market growth vs national market growth
   * price             SKU-level price effect (and whether it is national)
   * vacancy / stock-out / distribution   difference-in-differences at territory level:
       a territory "treated" by a driver (calls -20%+, +3 stock-out days, retail reach
       -10%+) is compared with untreated territories nationally, after normalising for
       its own state's market growth. Its shortfall is the driver's impact.
   * competitor        share loss explained by a competitor gaining more share than
                       proportional redistribution would give it (e.g. a new entrant).
4. Signature checks raise or lower confidence (stock-outs should hurt sales but not
   prescriptions; vacancies should hurt both).
5. Drill-down of the shortfall by state and territory.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from sherlock_rx.analytics.period import Period
from sherlock_rx.warehouse.engine import Warehouse

CAUSE_LABELS: dict[str, str] = {
    "field_force_vacancy": "Field-force vacancy / low call coverage",
    "stockout": "Stock-outs (supply)",
    "competitor_launch": "Competitor launch / aggressive competitor",
    "price_cut": "Price reduction (NPPA / DPCO)",
    "market_slowdown": "Market (category) slowdown",
    "distribution_loss": "Distribution / retail reach loss",
}

RECOMMENDATIONS: dict[str, str] = {
    "field_force_vacancy": "Fast-track hiring for the vacant territories; meanwhile assign the "
                           "ABM or a neighbouring MR to the top-decile prescribers and track "
                           "call coverage weekly.",
    "stockout": "Escalate with supply chain for the affected SKU, rebalance stock from "
                "surplus depots, and run a re-stocking drive with chemists once supply is back.",
    "competitor_launch": "Defend core prescribers: increase call frequency on doctors trialing "
                         "the competitor, review the trade scheme gap, and arm the team with a "
                         "differentiation message.",
    "price_cut": "Value loss is regulatory; protect margin through mix (push non-controlled "
                 "SKUs) and reset the value target, since units are unaffected.",
    "market_slowdown": "Category-wide, not brand-specific: hold share, re-phase targets with the "
                       "seasonal outlook, and shift effort to therapy areas that are growing.",
    "distribution_loss": "Resolve the stockist dispute or appoint an alternate distributor, then "
                         "run a retail re-activation campaign in the affected towns.",
}

MATERIALITY = 0.015  # a cause must explain >= 1.5% of last year's sales to be called a root cause
VAC_CALLS, SO_DAYS, DIST_RETAIL = 0.80, 3.0, 0.90


@dataclass
class Evidence:
    text: str
    source: str


@dataclass
class Cause:
    cause: str
    label: str
    impact: float  # INR, negative = sales lost
    pct_of_py: float
    share_of_loss: float = 0.0
    confidence: float = 0.5
    confidence_label: str = "Medium"
    scope: str = ""
    locations: list[str] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    recommendation: str = ""
    material: bool = False


@dataclass
class Headline:
    actual: float
    last_year: float
    target: float
    yoy: float
    achievement: float
    peer_yoy: float
    gap_vs_target: float
    gap_vs_peers: float
    status: str


@dataclass
class RCAReport:
    brand: str
    region: str
    state: str | None
    period: dict[str, Any]
    headline: Headline
    bridge: list[dict[str, Any]]
    causes: list[Cause]
    offsets: list[Cause]
    states: list[dict[str, Any]]
    territories: list[dict[str, Any]]
    query_ids: list[str]
    elapsed_ms: float

    @property
    def top_cause(self) -> Cause | None:
        material = [c for c in self.causes if c.material]
        return material[0] if material else None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RCAError(ValueError):
    pass


def _ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0, np.nan)


def _weighted_median(values: pd.Series, weights: pd.Series) -> float:
    """Robust peer baseline: one anomalous region (e.g. bouncing back from last year's
    slowdown) must not move the benchmark the way a pooled average would."""
    order = np.argsort(values.to_numpy())
    v, w = values.to_numpy()[order], weights.to_numpy()[order]
    cw = np.cumsum(w)
    return float(v[np.searchsorted(cw, cw[-1] / 2)])


class RCAEngine:
    def __init__(self, wh: Warehouse):
        self.wh = wh
        self.geo = wh.query("SELECT * FROM dim_geo")
        self.brands = wh.query("SELECT * FROM dim_brand")
        self.skus = wh.query("SELECT * FROM dim_sku")

    # ------------------------------------------------------------------ loading
    def _frames(self, brand_id: str, period: Period) -> dict[str, pd.DataFrame]:
        cur, py = period.keys, period.py_keys
        months = cur + py
        ph = ", ".join("?" * len(months))
        cur_ph = ", ".join("?" * len(cur))
        case = f"CASE WHEN month IN ({cur_ph}) THEN 'cur' ELSE 'py' END AS period"
        base = [*cur, *months, brand_id]

        def q(sql: str) -> pd.DataFrame:
            return self.wh.query(sql, base)

        return {
            "sales": q(f"""SELECT territory_id, sku_id, {case}, sum(units) AS units,
                           sum(net_value) AS net FROM agg_sales_tbm
                           WHERE month IN ({ph}) AND brand_id = ? GROUP BY ALL"""),
            "target": q(f"""SELECT territory_id, {case}, sum(target_value) AS target
                            FROM fact_targets WHERE month IN ({ph}) AND brand_id = ? GROUP BY ALL"""),
            "field": q(f"""SELECT territory_id, {case}, sum(calls) AS calls,
                           avg(reps_active) AS reps_active, avg(reps_sanctioned) AS reps_sanctioned
                           FROM fact_field WHERE month IN ({ph}) AND brand_id = ? GROUP BY ALL"""),
            "rx": q(f"""SELECT territory_id, {case}, sum(prescriptions) AS rx
                        FROM fact_rx WHERE month IN ({ph}) AND brand_id = ? GROUP BY ALL"""),
            "dist": q(f"""SELECT territory_id, {case}, avg(active_retailers) AS retailers
                          FROM fact_distribution WHERE month IN ({ph}) AND brand_id = ? GROUP BY ALL"""),
            "inv": q(f"""SELECT i.territory_id, i.sku_id, {case.replace('month', 'i.month')},
                         avg(i.stockout_days) AS so FROM fact_inventory i
                         JOIN dim_sku s ON s.sku_id = i.sku_id
                         WHERE i.month IN ({ph}) AND s.brand_id = ? GROUP BY ALL"""),
            "market": q(f"""SELECT state, player, is_ours, {case}, sum(units) AS units
                            FROM fact_market WHERE month IN ({ph}) AND brand_id = ? GROUP BY ALL"""),
        }

    @staticmethod
    def _wide(df: pd.DataFrame, index: list[str], values: list[str]) -> pd.DataFrame:
        w = df.pivot_table(index=index, columns="period", values=values, aggfunc="sum")
        w.columns = [f"{v}_{p}" for v, p in w.columns]
        return w.reset_index()

    # ------------------------------------------------------------------ main
    def diagnose(self, brand: str, region: str, period: Period, state: str | None = None) -> RCAReport:
        t0 = time.perf_counter()
        n_log = len(self.wh.log.records)
        if not period.has_prior_year:
            raise RCAError(f"No prior-year data for {period.label}; pick a period from FY25 onwards.")
        row = self.brands[self.brands.brand.str.lower() == brand.lower()]
        if row.empty:
            raise RCAError(f"Unknown brand '{brand}'.")
        brand_id, brand = str(row.brand_id.iat[0]), str(row.brand.iat[0])
        if state is not None:
            region = str(self.geo.loc[self.geo.state == state, "region"].iat[0])
        f = self._frames(brand_id, period)
        geo = self.geo[["territory_id", "territory", "state", "region"]]
        scope_states = [state] if state else sorted(geo.loc[geo.region == region, "state"].unique())

        # ---------------- SKU-level price/volume, national PY price per SKU
        s = self._wide(f["sales"], ["territory_id", "sku_id"], ["units", "net"]).fillna(0.0)
        s = s.merge(geo, on="territory_id")
        s["in_scope"] = s.state.isin(scope_states)
        sku_nat = s.groupby("sku_id")[["units_py", "net_py"]].sum()
        p0_nat = (sku_nat.net_py / sku_nat.units_py).rename("p0")
        s = s.merge(p0_nat, left_on="sku_id", right_index=True)
        s["v1p"] = s.units_cur * s.p0

        sc = s[s.in_scope]
        peers = s[~s.state.isin(geo.loc[geo.region == region, "state"].unique())]
        V0, V1 = float(sc.net_py.sum()), float(sc.net_cur.sum())
        if V0 <= 0:
            raise RCAError(f"No sales for {brand} in {state or region} last year.")
        sku_sc = sc.groupby("sku_id")[["units_py", "units_cur", "net_py", "net_cur"]].sum()
        p0_sc = sku_sc.net_py / sku_sc.units_py
        V1p = float((sku_sc.units_cur * p0_sc).sum())
        price_effect = V1 - V1p
        peer_V0 = float(peers.net_py.sum())
        peer_st = peers.groupby("state")[["net_py", "net_cur"]].sum()
        peer_yoy = _weighted_median(peer_st.net_cur / peer_st.net_py - 1, peer_st.net_py)
        sku_peer = peers.groupby("sku_id")[["units_cur", "net_py", "units_py", "net_cur"]].sum()
        peer_price_pct = float((sku_peer.net_cur - sku_peer.units_cur * sku_peer.net_py
                                / sku_peer.units_py).sum() / peer_V0)

        # ---------------- market audit
        mk = self._wide(f["market"], ["state", "player", "is_ours"], ["units"]).fillna(0.0)
        mk_state = mk.groupby("state")[["units_py", "units_cur"]].sum()
        mk_state["growth"] = mk_state.units_cur / mk_state.units_py - 1
        in_sc_mk = mk_state.index.isin(scope_states)
        region_states = geo.loc[geo.region == region, "state"].unique()
        mg = float(mk_state[in_sc_mk].units_cur.sum() / mk_state[in_sc_mk].units_py.sum() - 1)
        peer_mk = mk_state[~mk_state.index.isin(region_states)]
        mg_peer = _weighted_median(peer_mk.growth, peer_mk.units_py)
        market_effect = V0 * mg
        share_effect = V1p - V0 - market_effect

        ours = mk[mk.is_ours].set_index("state")
        sh = pd.DataFrame({
            "sh0": ours.units_py / mk_state.units_py,
            "sh1": ours.units_cur / mk_state.units_cur,
        })

        # ---------------- territory panel (national, so we have controls)
        t = s.groupby(["territory_id", "territory", "state", "region"], as_index=False)[
            ["net_py", "net_cur", "v1p"]].sum()
        for name, cols in (("field", ["calls", "reps_active", "reps_sanctioned"]),
                           ("rx", ["rx"]), ("dist", ["retailers"]), ("target", ["target"])):
            t = t.merge(self._wide(f[name], ["territory_id"], cols), on="territory_id", how="left")
        inv = self._wide(f["inv"], ["territory_id", "sku_id"], ["so"]).fillna(0.0)
        inv = inv.merge(s[["territory_id", "sku_id", "net_py"]], on=["territory_id", "sku_id"])
        inv["w"] = inv.net_py / inv.groupby("territory_id").net_py.transform("sum").replace(0, np.nan)
        inv["d"] = (inv.so_cur - inv.so_py) * inv.w
        inv["so_cur_w"] = inv.so_cur * inv.w
        inv["so_py_w"] = inv.so_py * inv.w
        so = inv.groupby("territory_id")[["d", "so_cur_w", "so_py_w"]].sum().rename(
            columns={"d": "so_delta", "so_cur_w": "so_cur", "so_py_w": "so_py"})
        t = t.merge(so, on="territory_id", how="left")
        t["calls_ratio"] = _ratio(t.calls_cur, t.calls_py)
        t["rx_ratio"] = _ratio(t.rx_cur, t.rx_py)
        t["retail_ratio"] = _ratio(t.retailers_cur, t.retailers_py)
        t["mg_state"] = t.state.map(mk_state.growth)
        t["rel"] = _ratio(t.v1p, t.net_py) / (1 + t.mg_state)

        # Competitor excess share gain per state (computed nationally to exclude from controls).
        # Expected = our lost share redistributed in proportion to each competitor's PY share,
        # adjusted by that competitor's national momentum (its trend outside this region), so
        # a competitor that is simply growing everywhere is not blamed for a local problem.
        comp_all = mk[~mk.is_ours]
        peer_c = comp_all[~comp_all.state.isin(region_states)].groupby("player")[
            ["units_py", "units_cur"]].sum()
        w_py = peer_c.units_py / peer_c.units_py.sum()
        w_cur = peer_c.units_cur / peer_c.units_cur.sum()
        trend = (w_cur / w_py.where(w_py > 0.005)).fillna(1.0)
        comp_rows = []
        for st, g in mk.groupby("state"):
            if st not in sh.index:
                continue
            o = g[g.is_ours]
            c = g[~g.is_ours].copy()
            tot0, tot1 = g.units_py.sum(), g.units_cur.sum()
            sh0_o, sh1_o = float(o.units_py.sum() / tot0), float(o.units_cur.sum() / tot1)
            c["sh0"], c["sh1"] = c.units_py / tot0, c.units_cur / tot1
            mom = c.sh0 * c.player.map(trend).fillna(1.0)
            c["expected"] = (1 - sh1_o) * mom / max(float(mom.sum()), 1e-9)
            c["excess"] = c.sh1 - c.expected
            loss = max(sh0_o - sh1_o, 0.0)
            gain = float(c.excess.clip(lower=0).sum())
            lead = c.sort_values("excess", ascending=False).iloc[0]
            comp_rows.append({
                "state": st, "sh0": sh0_o, "sh1": sh1_o, "comp_share_loss": min(loss, gain),
                "leader": str(lead.player), "leader_sh0": float(lead.sh0),
                "leader_sh1": float(lead.sh1), "new_entrant": bool(lead.sh0 < 0.005 and lead.sh1 > 0.01),
            })
        comp = pd.DataFrame(comp_rows).set_index("state")
        comp_flag_states = set(comp.index[comp.comp_share_loss > 0.01])

        t["f_vac"] = t.calls_ratio < VAC_CALLS
        t["f_so"] = t.so_delta > SO_DAYS
        t["f_dist"] = t.retail_ratio < DIST_RETAIL
        flagged = t.f_vac | t.f_so | t.f_dist
        ctrl = t[~flagged & ~t.state.isin(comp_flag_states) & t.rel.notna()]
        lo, hi = ctrl.rel.quantile([0.05, 0.95])
        ctrl = ctrl[(ctrl.rel >= lo) & (ctrl.rel <= hi)]
        ctrl_rel = float(np.average(ctrl.rel, weights=ctrl.net_py)) if len(ctrl) else 1.0
        t["expected"] = t.net_py * (1 + t.mg_state) * ctrl_rel
        t["shortfall"] = (t.expected - t.v1p).clip(lower=0)

        ts = t[t.state.isin(scope_states)].copy()
        st_vac = (1 - ts.calls_ratio).clip(0, 1) * ts.f_vac
        st_so = (0.85 * ts.so_delta / 30).clip(0, 1) * ts.f_so
        st_dist = (1 - ts.retail_ratio).clip(0, 1) * ts.f_dist
        tot = (st_vac + st_so + st_dist).replace(0, np.nan)
        ts["loss_vac"] = (ts.shortfall * st_vac / tot).fillna(0)
        ts["loss_so"] = (ts.shortfall * st_so / tot).fillna(0)
        ts["loss_dist"] = (ts.shortfall * st_dist / tot).fillna(0)

        # ---------------- candidate causes
        causes: list[Cause] = []
        ctrl_rx = float(ctrl.rx_cur.sum() / ctrl.rx_py.sum()) if len(ctrl) else 1.0

        def mk_cause(cause: str, loss: float, **kw: Any) -> Cause:
            return Cause(cause, CAUSE_LABELS[cause], impact=-loss, pct_of_py=-loss / V0,
                         recommendation=RECOMMENDATIONS[cause], **kw)

        def fmt_pct(x: float) -> str:
            return f"{x * 100:+.1f}%"

        # Market slowdown
        mk_short = V0 * (mg_peer - mg)
        if mk_short > 0:
            c = mk_cause("market_slowdown", mk_short, scope=f"{state or region} market")
            c.evidence = [
                Evidence(f"{brand}'s molecule market in {state or region} grew {fmt_pct(mg)} YoY "
                         f"vs {fmt_pct(mg_peer)} in the rest of India.", "fact_market"),
            ]
            sh0_sc = float(ours.units_py[ours.index.isin(scope_states)].sum()
                           / mk_state[in_sc_mk].units_py.sum())
            sh1_sc = float(ours.units_cur[ours.index.isin(scope_states)].sum()
                           / mk_state[in_sc_mk].units_cur.sum())
            share_stable = abs(sh1_sc - sh0_sc) < 0.01
            c.evidence.append(Evidence(
                f"Volume share moved {sh0_sc * 100:.1f}% -> {sh1_sc * 100:.1f}% "
                f"({'stable: the brand is holding share' if share_stable else 'also changed'}).",
                "fact_market"))
            c.confidence = 0.55 + 0.25 * share_stable + 0.2 * min(1.0, (mg_peer - mg) / 0.10)
            causes.append(c)

        # Price
        if price_effect < 0:
            pct = price_effect / V0
            national = peer_price_pct < 0 and abs(peer_price_pct - pct) < 0.05
            c = mk_cause("price_cut", -price_effect,
                         scope="national (all regions)" if national else "regional")
            sku_names = self.skus.set_index("sku_id").sku
            sku_p = (sku_sc.net_cur / sku_sc.units_cur) / p0_sc - 1
            worst = sku_p.idxmin()
            c.locations = [str(sku_names[worst])]
            vol = float(sku_sc.units_cur.sum() / sku_sc.units_py.sum() - 1)
            c.evidence = [
                Evidence(f"Realised price of {sku_names[worst]} changed {fmt_pct(float(sku_p[worst]))} "
                         f"YoY while units moved {fmt_pct(vol)}.", "fact_sales/fact_pricing"),
                Evidence(f"Price effect elsewhere in India: {fmt_pct(peer_price_pct)} of sales "
                         f"({'same cut nationally' if national else 'not seen elsewhere'}).",
                         "fact_sales"),
            ]
            c.confidence = 0.7 + 0.2 * national + 0.1 * (vol > -0.05)
            causes.append(c)

        # Territory-level drivers
        def driver(cause: str, loss_col: str, flag: str, describe: Any) -> None:
            loss = float(ts[loss_col].sum())
            if loss <= 0:
                return
            hit = ts[ts[flag] & (ts[loss_col] > 0)].sort_values(loss_col, ascending=False)
            c = mk_cause(cause, loss)
            by_state = hit.groupby("state")[loss_col].sum().sort_values(ascending=False)
            c.locations = [str(x) for x in by_state.index[:3]]
            c.scope = ", ".join(c.locations)
            c.evidence = describe(hit)
            rx_r = float(hit.rx_cur.sum() / hit.rx_py.sum()) if len(hit) else 1.0
            sales_r = float(hit.v1p.sum() / hit.net_py.sum()) if len(hit) else 1.0
            c.evidence.append(Evidence(
                f"In those territories prescriptions moved {fmt_pct(rx_r - 1)} vs "
                f"{fmt_pct(ctrl_rx - 1)} for unaffected territories; sales moved "
                f"{fmt_pct(sales_r - 1)}.", "fact_rx"))
            rx_drop = rx_r < ctrl_rx - 0.05
            consistent = rx_drop if cause == "field_force_vacancy" else not rx_drop
            c.confidence = 0.45 + 0.3 * consistent + 0.25 * min(1.0, len(hit) / 4)
            causes.append(c)

        def desc_vac(h: pd.DataFrame) -> list[Evidence]:
            return [Evidence(
                f"{len(h)} territories lost call coverage: calls {h.calls_py.sum():,.0f} -> "
                f"{h.calls_cur.sum():,.0f} ({fmt_pct(h.calls_cur.sum() / h.calls_py.sum() - 1)}); "
                f"avg active MRs {h.reps_active_py.mean():.1f} -> {h.reps_active_cur.mean():.1f}. "
                f"Worst: {', '.join(h.territory.head(3))}.", "fact_field")]

        def desc_so(h: pd.DataFrame) -> list[Evidence]:
            return [Evidence(
                f"{len(h)} territories had stock-outs: avg {h.so_py.mean():.1f} -> "
                f"{h.so_cur.mean():.1f} stock-out days per SKU-month. "
                f"Worst: {', '.join(h.territory.head(3))}.", "fact_inventory")]

        def desc_dist(h: pd.DataFrame) -> list[Evidence]:
            return [Evidence(
                f"{len(h)} territories lost retail reach: active retailers "
                f"{h.retailers_py.sum():,.0f} -> {h.retailers_cur.sum():,.0f} "
                f"({fmt_pct(h.retailers_cur.sum() / h.retailers_py.sum() - 1)}). "
                f"Worst: {', '.join(h.territory.head(3))}.", "fact_distribution")]

        driver("field_force_vacancy", "loss_vac", "f_vac", desc_vac)
        driver("stockout", "loss_so", "f_so", desc_so)
        driver("distribution_loss", "loss_dist", "f_dist", desc_dist)

        # Competitor
        comp_sc = comp[comp.index.isin(scope_states) & (comp.comp_share_loss > 0.005)]
        if len(comp_sc):
            v_state = sc.groupby("state").net_py.sum()
            loss = float(sum(
                r.comp_share_loss / r.sh0 * v_state.get(st, 0.0) * (1 + mk_state.growth[st])
                for st, r in comp_sc.iterrows()))
            top = comp_sc.sort_values("comp_share_loss", ascending=False)
            c = mk_cause("competitor_launch", loss)
            c.locations = [str(x) for x in top.index[:3]]
            c.scope = ", ".join(c.locations)
            for st, r in top.head(2).iterrows():
                c.evidence.append(Evidence(
                    f"{st}: {r.leader} went {r.leader_sh0 * 100:.1f}% -> {r.leader_sh1 * 100:.1f}% "
                    f"share{' (new entrant)' if r.new_entrant else ''} while {brand} went "
                    f"{r.sh0 * 100:.1f}% -> {r.sh1 * 100:.1f}%.", "fact_market"))
            c.confidence = 0.5 + 0.3 * bool(top.new_entrant.any()) + 0.2 * min(1.0, len(top) / 2)
            causes.append(c)

        # Scale territory/competitor drivers so they never claim more than the share loss.
        share_drivers = [c for c in causes if c.cause not in ("market_slowdown", "price_cut")]
        claimed = -sum(c.impact for c in share_drivers)
        pool = max(-share_effect, 0.0) * 1.15
        if claimed > pool > 0:
            k = pool / claimed
            for c in share_drivers:
                c.impact *= k
                c.pct_of_py *= k

        total_loss = -sum(c.impact for c in causes) or 1.0
        for c in causes:
            c.share_of_loss = -c.impact / total_loss
            c.material = c.pct_of_py <= -MATERIALITY
            c.confidence = float(np.clip(c.confidence, 0, 1))
            c.confidence_label = ("High" if c.confidence >= 0.75
                                  else "Medium" if c.confidence >= 0.5 else "Low")
        causes.sort(key=lambda c: c.impact)

        offsets: list[Cause] = []
        if market_effect > 0:
            offsets.append(Cause("market_growth", "Market growth", market_effect, market_effect / V0))
        if price_effect > 0:
            offsets.append(Cause("price_increase", "Price increase", price_effect, price_effect / V0))

        # ---------------- headline + drill-down
        target = float(t[t.state.isin(scope_states)].target_cur.sum())
        yoy = V1 / V0 - 1
        ach = V1 / target if target else float("nan")
        gap_peer = V1 - V0 * (1 + peer_yoy)
        if ach < 0.95 or yoy - peer_yoy < -0.04:
            status = "Underperforming"
        elif ach < 0.98 or yoy - peer_yoy < -0.02:
            status = "Slightly below plan"
        else:
            status = "On track"
        headline = Headline(V1, V0, target, yoy, ach, peer_yoy, V1 - target, gap_peer, status)

        st_tab = sc.groupby("state")[["net_py", "net_cur"]].sum()
        st_tab["yoy"] = st_tab.net_cur / st_tab.net_py - 1
        st_tab["gap_vs_peers"] = st_tab.net_cur - st_tab.net_py * (1 + peer_yoy)
        neg = st_tab.gap_vs_peers.clip(upper=0).sum()
        st_tab["share_of_gap"] = st_tab.gap_vs_peers.clip(upper=0) / neg if neg < 0 else 0.0
        states = (st_tab.sort_values("gap_vs_peers").reset_index()
                  .rename(columns={"net_py": "last_year", "net_cur": "actual"}).to_dict("records"))
        ts["flags"] = ts.apply(lambda r: ", ".join(
            n for n, f_ in (("vacancy", r.f_vac), ("stock-out", r.f_so), ("distribution", r.f_dist))
            if f_), axis=1)
        terr = (ts.sort_values("shortfall", ascending=False).head(8)
                [["territory", "state", "net_py", "v1p", "shortfall", "calls_ratio", "so_delta",
                  "retail_ratio", "rx_ratio", "flags"]]
                .rename(columns={"net_py": "last_year", "v1p": "actual_at_py_price"})
                .round(3).to_dict("records"))

        bridge = [
            {"step": "Last year", "value": V0},
            {"step": "Market growth", "value": market_effect},
            {"step": "Share change", "value": share_effect},
            {"step": "Price", "value": price_effect},
            {"step": "This period", "value": V1},
        ]
        qids = sorted({r.query_id for r in self.wh.log.records[n_log:]})
        return RCAReport(brand, region, state, period.to_dict(), headline, bridge, causes, offsets,
                         states, terr, qids, round((time.perf_counter() - t0) * 1000, 1))
