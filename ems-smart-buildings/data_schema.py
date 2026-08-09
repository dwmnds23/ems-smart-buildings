# data_schema.py
"""
Data schema + synthetic generator for the Smart Building single-CP study.

This mirrors the case study in Foroozandeh et al. (2022) — 15 apartments, 15‑min
resolution, bi-hourly tariff, PV per apartment (3.6 kW), one EV per apartment.
Case I (NSM), Case II (SMSCP), Case III (SMSCPB).

References: Foroozandeh et al., Int. J. Electrical Power & Energy Systems 135 (2022) 107534.
Key equations/tables mirrored here are described in your PDF (objective (1), EV constraints (2)-(9),
power balance (10), CP bounds (11)-(12), and BESS extensions (13)-(20)).
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import pathlib

# ---------------------------
# Schema
# ---------------------------

@dataclass
class BuildingConfig:
    # Apartments J, step duration tau hours, horizon in steps I
    J: int = 15
    tau_hours: float = 0.25  # 15 minutes
    # EV parameters (Foroozandeh 2022, eq. (21))
    EV_Smax_kWh: float = 27.2
    EV_Pch_kW: float = 3.7
    EV_Pdis_kW: float = 3.33
    EV_eta_ch: float = 0.95
    EV_eta_dis: float = 0.95
    # BESS parameters (Case III), eq. (22)
    BESS_Smax_kWh: float = 50.0
    BESS_Pch_kW: float = 6.3
    BESS_Pdis_kW: float = 5.67
    BESS_eta_ch: float = 0.95
    BESS_eta_dis: float = 0.95

CP_CANDIDATES_KW = [13.8, 17.25, 20.7, 27.6, 34.5, 41.4]

def empty_timeseries(index: pd.DatetimeIndex, J: int):
    cols = [f"apt{j+1}" for j in range(J)]
    return pd.DataFrame(0.0, index=index, columns=cols)

@dataclass
class StudyData:
    # Time index at 15-min resolution for one year (or custom subset) in local time
    index: pd.DatetimeIndex
    # Per-apartment load (kW) and PV (kW)
    load_kw: pd.DataFrame
    pv_kw: pd.DataFrame
    # Common services load (kW) as a Series
    common_kw: pd.Series
    # EV arrival/departure per apt: dict apt -> list of (arrive_ts, depart_ts, initial_soc_kWh, min_soc_on_departure_kWh)
    ev_stints: Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp, float, float]]]
    # Tariffs (€/kWh): buy and sell, indexed by time
    price_buy: pd.Series
    price_sell: pd.Series

# ---------------------------
# Synthetic generator (lightweight, 7 days, hourly or 15-min)
# ---------------------------

def make_synthetic_week(config: BuildingConfig, start="2022-01-03", days=7, freq="15min", J=3, seed=42) -> StudyData:
    rng = np.random.default_rng(seed)
    # figure out number of steps per day
    if freq.endswith('min'):
        step_minutes = int(freq[:-3])
        steps_per_day = int(24*60/step_minutes)
    else:
        # assume like '1H', '30T'
        idx_tmp = pd.date_range(start=start, periods=2, freq=freq)
        delta = (idx_tmp[1]-idx_tmp[0]).total_seconds()/60.0
        steps_per_day = int(round(24*60/delta))
    periods = steps_per_day*days
    idx = pd.date_range(start=start, periods=periods, freq=freq)
    # Loads: base + diurnal pattern per apt
    base_load = 0.3 + 0.7 * (1 + np.sin((idx.hour.values + idx.minute.values/60) / 24 * 2*np.pi - 1))
    load_kw = empty_timeseries(idx, J)
    for j in range(J):
        scale = 1.0 + 0.2*rng.standard_normal()
        noise = 0.1*rng.standard_normal(len(idx))
        load_kw.iloc[:, j] = np.maximum(0.05, scale*base_load + noise)
    # PV: bell-shaped around noon, peak 3.6 kW
    pv_kw = empty_timeseries(idx, J)
    solar = np.maximum(0, np.sin((idx.hour.values + idx.minute.values/60 - 6)/12*np.pi))
    for j in range(J):
        pv_kw.iloc[:, j] = 3.6 * solar * (0.9 + 0.2*rng.random(len(idx)))
    # Common services small base
    common_kw = pd.Series(0.12 + 0.18*np.where((idx.hour>=8)&(idx.hour<=22), 1, 0), index=idx)
    # Tariff (bi-hourly): cheaper at night/weekend
    buy = pd.Series(0.30, index=idx)
    off = (idx.weekday>=5) | (idx.hour<7) | (idx.hour>=23)
    buy[off] = 0.10
    sell = pd.Series(0.05, index=idx)
    # EV stints: each day arrive 18:00, depart 07:30 next day
    ev_stints = {}
    for j in range(J):
        name = f"apt{j+1}"
        stints = []
        for d in range(days):
            day = pd.Timestamp(start) + pd.Timedelta(days=d)
            arr = day.replace(hour=18, minute=0)
            dep = (day + pd.Timedelta(days=1)).replace(hour=7, minute=30)
            init = float(rng.uniform(0.3, 0.7)*config.EV_Smax_kWh)
            min_dep = float(rng.uniform(0.3, 0.8)*config.EV_Smax_kWh)
            stints.append((arr, dep, init, min_dep))
        ev_stints[name] = stints
    return StudyData(index=idx, load_kw=load_kw, pv_kw=pv_kw, common_kw=common_kw,
                     ev_stints=ev_stints, price_buy=buy, price_sell=sell)

def export_csv(data: StudyData, folder: str):
    p = pathlib.Path(folder)
    p.mkdir(parents=True, exist_ok=True)
    data.load_kw.to_csv(p/"load_kw.csv")
    data.pv_kw.to_csv(p/"pv_kw.csv")
    data.common_kw.to_frame("common_kw").to_csv(p/"common_kw.csv")
    data.price_buy.to_frame("price_buy").to_csv(p/"price_buy.csv")
    data.price_sell.to_frame("price_sell").to_csv(p/"price_sell.csv")
    # Save EV stints JSON
    import json
    ev_json = {k: [(str(a), str(d), float(i), float(m)) for (a,d,i,m) in v] for k,v in data.ev_stints.items()}
    (p/"ev_stints.json").write_text(json.dumps(ev_json, indent=2, ensure_ascii=False))
    return str(p)

if __name__ == "__main__":
    cfg = BuildingConfig()
    demo = make_synthetic_week(cfg, J=3, days=7, freq="1H")  # hourly quick demo
    out = export_csv(demo, "./demo_data")
    print("Wrote demo data to:", out)
