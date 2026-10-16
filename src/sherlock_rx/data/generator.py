"""Synthetic warehouse generator.

Simulates a causal chain rather than independent random tables:

    market demand ──► Rx share (field-force calls, competitor entrants)
                  ──► prescriptions ──► sell-out (stock availability, retail reach)
                  ──► secondary sales (channel loading) × price ──► INR value

Planted scenarios perturb one lever each, so their downstream signature (e.g. stock-out
lowers sales but not prescriptions) is produced by the mechanism, not hard-coded.
All arrays are [territory, brand, month] (or [territory, sku, month]) and vectorised, so
scale factor 100 (~8M sales rows) generates in well under a minute.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from sherlock_rx.data import notes as notes_mod
from sherlock_rx.data.calendar import ALL_MONTHS, N_MONTHS, month_index
from sherlock_rx.data.catalog import (
    BRANDS,
    FLAT_SEASON,
    HQ_TOWNS,
    REGIONS,
    SEASONALITY,
    STATE_CODES,
    STATE_WEIGHT,
)
from sherlock_rx.data.scenarios import Control, Scenario, plan_scenarios

CALLS_PER_REP = 150.0
UNITS_PER_RX = 1.8
ENTRANT_NAMES = ("Curexa", "Novaril", "Zydora", "Medlyf", "Saneva", "Remidex", "Vitalix", "Pharmora")


@dataclass
class GenerationSummary:
    out_dir: str
    scale: float
    seed: int
    territories: int
    rows: dict[str, int]
    seconds: float
    scenarios: int
    controls: int


def _lognormal(rng: np.random.Generator, sigma: float, shape: tuple[int, ...]) -> np.ndarray:
    return np.exp(rng.normal(-0.5 * sigma**2, sigma, shape))


def _write(df: pd.DataFrame, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path, compression="zstd")
    return len(df)


def generate(out_dir: str | Path, scale: float = 1.0, seed: int = 7,
             severity_scale: float = 1.0, noise_scale: float = 1.0) -> GenerationSummary:
    """severity_scale < 1 weakens planted causes; noise_scale > 1 amplifies background noise.
    Both exist for robustness evaluation (see eval/rca_eval.py `sweep`)."""
    t0 = time.perf_counter()
    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    rng = np.random.default_rng(seed)

    # ---------------------------------------------------------------- dimensions
    terr_rows = []
    for region, states in REGIONS.items():
        for state in states:
            n = max(2, round(6 * STATE_WEIGHT[state] * scale))
            towns = HQ_TOWNS[state]
            for i in range(n):
                town = towns[i % len(towns)]
                hq = town if i < len(towns) else f"{town}-{i // len(towns) + 1}"
                terr_rows.append({
                    "territory_id": f"{STATE_CODES[state]}{i + 1:03d}",
                    "territory": hq,
                    "state": state,
                    "state_code": STATE_CODES[state],
                    "region": region,
                })
    geo = pd.DataFrame(terr_rows)
    T, B, M = len(geo), len(BRANDS), N_MONTHS
    state_arr = geo["state"].to_numpy()
    region_arr = geo["region"].to_numpy()
    region_names = list(REGIONS)

    skus = [(bi, s) for bi, b in enumerate(BRANDS) for s in b.skus]
    S = len(skus)
    sku_brand = np.array([bi for bi, _ in skus])
    sku_mix = np.array([s.mix for _, s in skus])

    dim_brand = pd.DataFrame([{
        "brand_id": b.brand_id, "brand": b.name, "molecule": b.molecule, "therapy": b.therapy,
    } for b in BRANDS])
    dim_sku = pd.DataFrame([{
        "sku_id": s.sku_id, "sku": s.name, "brand_id": BRANDS[bi].brand_id, "base_ptr": s.base_ptr,
    } for bi, s in skus])
    dim_month = pd.DataFrame([{
        "month": m.key, "month_label": m.label, "year": m.year, "month_num": m.month,
        "fy": m.fy, "fq": m.fq, "quarter": m.quarter_label, "month_idx": i,
    } for i, m in enumerate(ALL_MONTHS)])

    # ---------------------------------------------------------------- latent structure
    size = _lognormal(rng, 0.30, (T,)) * np.array([STATE_WEIGHT[s] for s in state_arr])
    affinity = _lognormal(rng, 0.20, (len(region_names), B))
    region_idx = np.array([region_names.index(r) for r in region_arr])
    cal_month = np.array([m.month for m in ALL_MONTHS])
    fy_idx = np.array([m.fy - ALL_MONTHS[0].fy for m in ALL_MONTHS])

    national_monthly = np.array([b.national_value_cr * 1e7 / 12 for b in BRANDS])
    terr_weight = size[:, None] * affinity[region_idx]  # [T,B]
    terr_weight /= terr_weight.sum(axis=0, keepdims=True)
    base_value = national_monthly[None, :] * terr_weight  # INR / month at launch prices

    base_ptr_brand = np.array([sum(s.mix * s.base_ptr for s in b.skus) for b in BRANDS])
    share0 = np.clip(np.array([b.share for b in BRANDS])[None, :] + rng.normal(0, 0.03, (T, B)),
                     0.05, 0.5)
    market_base = base_value / base_ptr_brand[None, :] / share0  # units / month

    growth = np.array([b.growth for b in BRANDS])[None, :] + rng.normal(0, 0.01, (T, B))
    trend = (1 + growth[:, :, None]) ** (np.arange(M)[None, None, :] / 12.0)
    season = np.array([SEASONALITY.get(b.therapy, FLAT_SEASON) for b in BRANDS])  # [B,12]
    season_tm = season[:, cal_month - 1][None, :, :]  # [1,B,M]
    market_expected = market_base[:, :, None] * trend * season_tm
    nz = noise_scale
    market_units = market_expected * _lognormal(rng, 0.03 * nz, (T, B, M))

    # Field force
    sanctioned = rng.integers(2, 4, (T, B)).astype(float)
    reps_active = np.repeat(sanctioned[:, :, None], M, axis=2)
    vacancy_events: list[tuple[int, int, int]] = []
    starts = rng.random((T, B, M)) < 0.012 * nz
    for t, b, m in zip(*np.nonzero(starts), strict=True):
        length = int(rng.integers(1, 4))
        reps_active[t, b, m:m + length] = np.maximum(sanctioned[t, b] - 1, 1)
        vacancy_events.append((int(t), int(b), int(m)))

    retail_base = np.round(size[:, None] * 350 * rng.uniform(0.5, 0.8, (1, B)))
    retail_mult = 1 + rng.normal(0, 0.015 * nz, (T, B, M))
    stockout_days = rng.poisson(0.5, (T, S, M)).astype(float)
    so_noise = rng.random((T, S, M)) < 0.03 * nz
    stockout_days += so_noise * rng.integers(3, 9, (T, S, M))
    take = np.zeros((T, B, M))
    entrant_name = {b.name: f"{ENTRANT_NAMES[i]}-{b.molecule.split()[0][:4]}"
                    for i, b in enumerate(BRANDS)}

    ptr_planned = np.array([s.base_ptr for _, s in skus])[:, None] * (1.05 ** fy_idx)[None, :]
    ptr = ptr_planned.copy()
    discount = np.clip(0.08 + rng.normal(0, 0.005, (B, M)), 0.05, 0.12)

    # ---------------------------------------------------------------- plant scenarios
    scenarios, controls = plan_scenarios(rng, severity_scale=severity_scale)
    brand_idx = {b.name: i for i, b in enumerate(BRANDS)}
    note_rows: list[dict[str, object]] = []

    def add_note(m: int, t: int, b: int | None, role: str, text: str) -> None:
        note_rows.append({
            "month": ALL_MONTHS[m].key, "territory_id": geo.territory_id[t],
            "brand_id": BRANDS[b].brand_id if b is not None else None,
            "author_role": role, "text": text,
        })

    for sc in scenarios:
        b = brand_idx[sc.brand]
        brand = BRANDS[b]
        ms = [month_index(m) for m in sc.months]
        in_region = region_arr == sc.region
        in_states = np.isin(state_arr, sc.states) if sc.states else in_region
        main_sku = int(np.nonzero(sku_brand == b)[0][0])
        sc.sku = skus[main_sku][1].name
        if sc.cause == "field_force_vacancy":
            pick = in_states & (rng.random(T) < rng.uniform(0.6, 0.9))
            for m in ms:
                reps_active[pick, b, m] = np.floor(sanctioned[pick, b] * (1 - sc.severity))
        elif sc.cause == "stockout":
            pick = in_states & (rng.random(T) < 0.85)
            sku_ids = np.nonzero(sku_brand == b)[0]
            hit = sku_ids if rng.random() < 0.3 else sku_ids[:1]
            for m in ms:
                days = 30 * sc.severity / 0.85 * rng.uniform(0.8, 1.2, (int(pick.sum()), len(hit)))
                stockout_days[np.ix_(np.nonzero(pick)[0], hit, [m])] = np.minimum(days, 27)[:, :, None]
        elif sc.cause == "competitor_launch":
            sc.competitor = entrant_name[sc.brand]
            ramp = (0.5, 0.8, 1.0)
            for k, m in enumerate(ms):
                take[in_states, b, m] = sc.severity * ramp[k]
            take[in_states, b, ms[-1] + 1:] = sc.severity
        elif sc.cause == "price_cut":
            ptr[main_sku, ms[0]:] *= 1 - sc.severity
        elif sc.cause == "market_slowdown":
            for m in ms:
                market_units[in_region, b, m] *= 1 - sc.severity
        elif sc.cause == "distribution_loss":
            pick = in_states & (rng.random(T) < rng.uniform(0.7, 1.0))
            for m in ms:
                retail_mult[pick, b, m] *= 1 - sc.severity

        # Field reports about the scenario.
        cand = np.nonzero(in_states)[0]
        for _ in range(int(rng.integers(3, 6))):
            t = int(rng.choice(cand))
            m = int(rng.choice(ms))
            comp = sc.competitor or str(rng.choice(brand.competitors))
            text = notes_mod.scenario_note(
                sc.cause, rng, hq=geo.territory[t], state=state_arr[t], brand=brand.name,
                sku=sc.sku or brand.skus[0].name, comp=comp, molecule=brand.molecule,
                therapy=brand.therapy)
            add_note(m, t, b, str(rng.choice(("MR", "ABM", "RBM"))), text)

    # Background vacancies also produce (truthful) notes some of the time.
    for t, b, m in vacancy_events:
        if rng.random() < 0.35:
            text = notes_mod.scenario_note("field_force_vacancy", rng, hq=geo.territory[t],
                                           state=state_arr[t], brand=BRANDS[b].name, sku="",
                                           comp="", molecule="", therapy="")
            add_note(m, t, b, "ABM", text)
    n_routine = int(T * M * 0.12)
    for _ in range(n_routine):
        t, b, m = int(rng.integers(T)), int(rng.integers(B)), int(rng.integers(M))
        text = notes_mod.routine_note(rng, hq=geo.territory[t], state=state_arr[t],
                                      brand=BRANDS[b].name, therapy=BRANDS[b].therapy)
        add_note(m, t, b, str(rng.choice(("MR", "ABM"))), text)

    # ---------------------------------------------------------------- outcomes
    calls = reps_active * CALLS_PER_REP * _lognormal(rng, 0.05 * nz, (T, B, M))
    ratio = calls / (sanctioned[:, :, None] * CALLS_PER_REP)
    prev = np.concatenate([ratio[:, :, :1], ratio[:, :, :-1]], axis=2)
    eff = np.clip(0.6 * ratio + 0.4 * prev, 0, 1.2)
    response = (0.35 + 0.65 * eff) ** 0.6
    drift = np.exp(np.cumsum(rng.normal(0, 0.006 * nz, (T, B, M)), axis=2))
    rx_share = share0[:, :, None] * response * (1 - take) * drift
    rx_units = market_units * rx_share

    active_retail = np.round(retail_base[:, :, None] * retail_mult)
    dist_factor = (active_retail / retail_base[:, :, None]) ** 0.7
    availability = 1 - 0.85 * np.clip(stockout_days, 0, 30) / 30.0  # [T,S,M]

    sellout_sku = (rx_units[:, sku_brand, :] * sku_mix[None, :, None]
                   * dist_factor[:, sku_brand, :] * availability)  # [T,S,M]
    loading = np.where(cal_month % 3 == 0, 1.05, np.where(cal_month % 3 == 1, 0.97, 0.98))
    sec_units = sellout_sku * loading[None, None, :] * _lognormal(rng, 0.035 * nz, (T, S, M))
    gross = sec_units * ptr[None, :, :]
    net = gross * (1 - discount[sku_brand][None, :, :])

    expected_net = (market_expected[:, sku_brand, :] * share0[:, sku_brand, None]
                    * sku_mix[None, :, None] * ptr_planned[None, :, :]
                    * (1 - 0.08))
    target_brand = np.zeros((T, B, M))
    # Plan = expected demand at planned prices, net of normal stock availability (~98%),
    # so a quarter with nothing wrong lands at roughly 97-100% achievement.
    np.add.at(target_brand, (slice(None), sku_brand, slice(None)), expected_net * 0.99)

    # ---------------------------------------------------------------- tables
    month_keys = np.array([m.key for m in ALL_MONTHS])
    tid = geo.territory_id.to_numpy()
    bid = np.array([b.brand_id for b in BRANDS])
    sid = np.array([s.sku_id for _, s in skus])

    def cat(values: np.ndarray, codes: np.ndarray) -> pd.Categorical:
        # Categoricals keep 8M-row frames small (codes, not Python strings) and map to
        # Parquet dictionary encoding.
        return pd.Categorical.from_codes(codes, categories=values)

    def grid(n_mid: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        tt, kk, mm = np.meshgrid(np.arange(T), np.arange(n_mid), np.arange(M), indexing="ij")
        return tt.ravel(), kk.ravel(), mm.ravel()

    rows: dict[str, int] = {}
    tt, ss, mm = grid(S)
    sales = pd.DataFrame({
        "month": cat(month_keys, mm), "territory_id": cat(tid, tt), "sku_id": cat(sid, ss),
        "brand_id": cat(bid, sku_brand[ss]),
        "units": np.round(sec_units.ravel(), 1),
        "gross_value": np.round(gross.ravel(), 2), "net_value": np.round(net.ravel(), 2),
    })
    fy_of_m = np.array([f"FY{m.fy:02d}" for m in ALL_MONTHS])[mm]
    rows["fact_sales"] = 0
    for fy in sorted(set(fy_of_m)):
        part = sales[fy_of_m == fy]
        part = part.assign(**{c: part[c].astype(str) for c in ("month", "territory_id",
                                                                "sku_id", "brand_id")})
        rows["fact_sales"] += _write(part, out / "fact_sales" / f"fy={fy}" / "part-0.parquet")
    del sales

    inv = pd.DataFrame({
        "month": cat(month_keys, mm).astype(str), "territory_id": cat(tid, tt).astype(str),
        "sku_id": cat(sid, ss).astype(str),
        "stockout_days": np.round(np.clip(stockout_days, 0, 30).ravel()).astype(int),
    })
    rows["fact_inventory"] = _write(inv, out / "fact_inventory.parquet")

    tt, bb, mm = grid(B)
    rows["fact_targets"] = _write(pd.DataFrame({
        "month": month_keys[mm], "territory_id": tid[tt], "brand_id": bid[bb],
        "target_value": np.round(target_brand.ravel(), 2),
    }), out / "fact_targets.parquet")
    rows["fact_field"] = _write(pd.DataFrame({
        "month": month_keys[mm], "territory_id": tid[tt], "brand_id": bid[bb],
        "reps_sanctioned": sanctioned[tt, bb].astype(int),
        "reps_active": reps_active.ravel().astype(int),
        "calls": np.round(calls.ravel()).astype(int),
        "doctors_met": np.round(calls.ravel() / 2.2).astype(int),
    }), out / "fact_field.parquet")
    rx = rx_units.ravel() / UNITS_PER_RX * _lognormal(rng, 0.04, (T * B * M,))
    rows["fact_rx"] = _write(pd.DataFrame({
        "month": month_keys[mm], "territory_id": tid[tt], "brand_id": bid[bb],
        "prescriptions": np.round(rx).astype(int),
        "prescribers": np.round(rx / rng.uniform(6, 10, rx.shape)).astype(int),
    }), out / "fact_rx.parquet")
    rows["fact_distribution"] = _write(pd.DataFrame({
        "month": month_keys[mm], "territory_id": tid[tt], "brand_id": bid[bb],
        "active_retailers": active_retail.ravel().astype(int),
        "retail_universe": retail_base[tt, bb].astype(int),
    }), out / "fact_distribution.parquet")

    # Market audit (state level, our brand + competitors + any entrant), in units.
    ours_tbm = sellout_sku.copy()
    our_units = np.zeros((T, B, M))
    np.add.at(our_units, (slice(None), sku_brand, slice(None)), ours_tbm)
    entrant_units = market_units * share0[:, :, None] * take * drift * 1.25
    rest = np.maximum(market_units - our_units - entrant_units, 0)
    state_names = list(STATE_CODES)
    st_idx = np.array([state_names.index(s) for s in state_arr])
    mk_rows = []
    for b, brand in enumerate(BRANDS):
        w = rng.dirichlet(np.full(len(brand.competitors), 4.0))
        comp_drift = np.exp(np.cumsum(rng.normal(0, 0.01 * nz, (len(brand.competitors), M)),
                                      axis=1))
        for si, state in enumerate(state_names):
            mask = st_idx == si
            if not mask.any():
                continue
            ours_s = our_units[mask, b].sum(axis=0)
            rest_s = rest[mask, b].sum(axis=0)
            ent_s = entrant_units[mask, b].sum(axis=0)
            cw = w[:, None] * comp_drift
            cw = cw / cw.sum(axis=0, keepdims=True)
            players = [(brand.name, ours_s, True)]
            players += [(c, rest_s * cw[k], False) for k, c in enumerate(brand.competitors)]
            players.append((entrant_name[brand.name], ent_s, False))
            for name, units, is_ours in players:
                for m in range(M):
                    if units[m] <= 0.5:
                        continue
                    mk_rows.append((month_keys[m], state, brand.brand_id, brand.molecule,
                                    name, is_ours, round(float(units[m]), 1)))
    market = pd.DataFrame(mk_rows, columns=["month", "state", "brand_id", "molecule",
                                            "player", "is_ours", "units"])
    rows["fact_market"] = _write(market, out / "fact_market.parquet")

    sm_s, sm_m = np.meshgrid(np.arange(S), np.arange(M), indexing="ij")
    rows["fact_pricing"] = _write(pd.DataFrame({
        "month": month_keys[sm_m.ravel()], "sku_id": sid[sm_s.ravel()],
        "ptr": np.round(ptr[sm_s.ravel(), sm_m.ravel()], 2),
        "trade_discount": np.round(discount[sku_brand[sm_s.ravel()], sm_m.ravel()], 4),
    }), out / "fact_pricing.parquet")

    notes_df = pd.DataFrame(note_rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    notes_df.insert(0, "note_id", [f"N{i:06d}" for i in range(len(notes_df))])
    rows["field_notes"] = _write(notes_df, out / "field_notes.parquet")

    rows["dim_geo"] = _write(geo, out / "dim_geo.parquet")
    rows["dim_brand"] = _write(dim_brand, out / "dim_brand.parquet")
    rows["dim_sku"] = _write(dim_sku, out / "dim_sku.parquet")
    rows["dim_month"] = _write(dim_month, out / "dim_month.parquet")
    comp = pd.DataFrame([{"brand_id": b.brand_id, "competitor": c}
                         for b in BRANDS for c in b.competitors])
    rows["dim_competitor"] = _write(comp, out / "dim_competitor.parquet")

    _write_ground_truth(out, scenarios, controls, scale, seed)
    summary = GenerationSummary(str(out), scale, seed, T, rows, round(time.perf_counter() - t0, 2),
                                len(scenarios), len(controls))
    (out / "_generation.json").write_text(json.dumps(summary.__dict__, indent=2))
    return summary


def _write_ground_truth(out: Path, scenarios: list[Scenario], controls: list[Control],
                        scale: float, seed: int) -> None:
    payload = {
        "seed": seed,
        "scale": scale,
        "scenarios": [s.to_dict() for s in scenarios],
        "controls": [c.to_dict() for c in controls],
    }
    (out / "_ground_truth.json").write_text(json.dumps(payload, indent=2))
