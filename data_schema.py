"""
Data schema, I/O and synthetic data generator for the single-contract-power study.

Mirrors the setting of Foroozandeh et al. (2022): a residential building with
J = 15 apartments, 15-min resolution, bi-hourly (time-of-use) tariff, a 3.6 kW
PV array and one bidirectional EV per apartment, and (Case III) a shared BESS.

Units used everywhere in this repository:
    power  -> kW        (loads, PV, all decision flows)
    energy -> kWh       (state of charge)
    price  -> EUR/kWh   (tariffs)
    time   -> hours     (tau = step length, inferred from the data index)

Reference: Z. Foroozandeh, S. Ramos, J. Soares, Z. Vale, M. Dias,
"Single contract power optimization: A novel business model for smart buildings
using intelligent energy management", Int. J. Electr. Power Energy Syst. 135
(2022) 107534.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# (arrival, departure, SoC at arrival [kWh], minimum SoC at departure [kWh])
Stint = Tuple[pd.Timestamp, pd.Timestamp, float, float]

# Candidate single-contract-power values tested in the paper (Tables 7-8), kW.
CP_CANDIDATES_KW: List[float] = [13.8, 17.25, 20.7, 27.6, 34.5, 41.4]


@dataclass
class BuildingConfig:
    """Physical and tariff parameters.

    Values marked (paper) come from Foroozandeh et al. (2022), eqs. (21)-(22)
    and Section 5. Values marked (assumption) are not given in the paper and
    are configurable.
    """

    J: int = 15
    # EV, eq. (21) (paper)
    EV_Smax_kWh: float = 27.2
    EV_Pch_kW: float = 3.7
    EV_Pdis_kW: float = 3.33
    EV_eta_ch: float = 0.95   # (assumption) paper does not report efficiencies
    EV_eta_dis: float = 0.95  # (assumption)
    # BESS, eq. (22) (paper)
    BESS_Smax_kWh: float = 50.0
    BESS_Smin_kWh: float = 0.0
    BESS_Pch_kW: float = 6.3
    BESS_Pdis_kW: float = 5.67
    BESS_eta_ch: float = 0.95  # (assumption)
    BESS_eta_dis: float = 0.95  # (assumption)
    BESS_init_kWh: float = 0.0  # (paper) S_BE^initial = 0
    # Export to the grid limited to a fraction of CP, eq. (12) (paper)
    export_fraction_of_cp: float = 0.5
    # Fixed contract-power charge, SCP(CP) in eq. (1). (assumption) The paper
    # does not publish its tariff table; a linear EUR per kW per day charge
    # makes the cost of over-sizing the contract explicit.
    cp_charge_eur_per_kw_day: float = 0.05
    # Individual contracts of the "traditional" building, Case I (paper, Sec. 5)
    apartment_cp_kW: List[float] = field(
        default_factory=lambda: [6.9] * 6 + [10.35] * 6 + [13.8] * 3
    )
    common_cp_kW: float = 13.8


@dataclass
class StudyData:
    index: pd.DatetimeIndex
    load_kw: pd.DataFrame        # per-apartment load, columns apt1..aptJ
    pv_kw: pd.DataFrame          # per-apartment PV generation
    common_kw: pd.Series         # common-services load
    price_buy: pd.Series         # EUR/kWh
    price_sell: pd.Series        # EUR/kWh
    ev_stints: Dict[str, List[Stint]]

    @property
    def tau(self) -> float:
        """Step length in hours, inferred from the time index."""
        if len(self.index) < 2:
            raise ValueError("Need at least two time steps to infer tau.")
        return (self.index[1] - self.index[0]).total_seconds() / 3600.0

    @property
    def days(self) -> float:
        return len(self.index) * self.tau / 24.0

    @property
    def apartments(self) -> List[str]:
        return list(self.load_kw.columns)

    def validate(self) -> None:
        idx = self.index
        steps = np.diff(idx.asi8)
        if len(steps) and not np.all(steps == steps[0]):
            raise ValueError("Time index must be evenly spaced.")
        for name, df in [("load_kw", self.load_kw), ("pv_kw", self.pv_kw)]:
            if not df.index.equals(idx):
                raise ValueError(f"{name} index does not match load_kw index.")
            if df.isna().any().any():
                raise ValueError(f"{name} contains missing values.")
        for name, s in [("common_kw", self.common_kw), ("price_buy", self.price_buy),
                        ("price_sell", self.price_sell)]:
            if not s.index.equals(idx):
                raise ValueError(f"{name} index does not match load_kw index.")
        if list(self.pv_kw.columns) != self.apartments:
            raise ValueError("pv_kw columns must match load_kw columns.")
        missing = set(self.apartments) - set(self.ev_stints)
        if missing:
            raise ValueError(f"No EV stints for: {sorted(missing)}")


# ---------------------------------------------------------------------------
# Synthetic generator
# ---------------------------------------------------------------------------

def make_synthetic(config: BuildingConfig, start: str = "2022-07-04", days: int = 7,
                   freq: str = "15min", seed: int = 42) -> StudyData:
    """Plausible synthetic week for a 15-apartment building.

    Real data from the paper is not public, so this generator produces profiles
    with the same structure: morning/evening household peaks, clear-sky PV with
    day-to-day cloudiness, evening EV arrivals and morning departures.
    """
    rng = np.random.default_rng(seed)
    J = config.J
    idx = pd.date_range(start=start, periods=int(days * 24 * 60 / pd.Timedelta(freq).total_seconds() * 60),
                        freq=freq)
    hours = idx.hour.values + idx.minute.values / 60.0
    weekend = idx.weekday.values >= 5
    cols = [f"apt{j + 1}" for j in range(J)]

    # Household load: base + morning bump + larger evening bump (kW)
    morning = np.exp(-0.5 * ((hours - 7.5) / 1.2) ** 2)
    evening = np.exp(-0.5 * ((hours - 20.0) / 2.0) ** 2)
    midday_home = np.where(weekend, np.exp(-0.5 * ((hours - 13.0) / 3.0) ** 2), 0.0)
    load = np.zeros((len(idx), J))
    for j in range(J):
        size = rng.uniform(0.45, 1.15)
        prof = 0.30 + 0.6 * morning + 1.2 * evening + 0.6 * midday_home
        noise = rng.gamma(2.0, 0.12, len(idx))
        load[:, j] = np.maximum(0.08, size * prof + noise - 0.24)
    load_kw = pd.DataFrame(load, index=idx, columns=cols)

    # PV: 3.6 kWp per apartment, clear-sky bell 06:00-20:00 (summer), daily
    # cloud factor. The derate (orientation/shading/inverter) is calibrated so
    # that yield is close to the paper's ~2,700 kWh/year per apartment (Table 6).
    derate = 0.45
    clear = np.clip(np.sin((hours - 6.0) / 14.0 * np.pi), 0.0, None) ** 1.3
    cloud_day = rng.uniform(0.55, 1.0, days)
    cloud = np.repeat(cloud_day, len(idx) // days)
    pv = np.zeros((len(idx), J))
    for j in range(J):
        pv[:, j] = 3.6 * derate * clear * cloud * rng.uniform(0.92, 1.0)
    pv_kw = pd.DataFrame(pv, index=idx, columns=cols)

    common_kw = pd.Series(0.25 + 0.35 * ((hours >= 7) & (hours < 23)), index=idx, name="common_kw")

    # Bi-hourly tariff: cheap at night (22:00-08:00) and all weekend
    off_peak = weekend | (hours < 8) | (hours >= 22)
    price_buy = pd.Series(np.where(off_peak, 0.10, 0.20), index=idx, name="price_buy")
    price_sell = pd.Series(0.05, index=idx, name="price_sell")

    # EV stints: arrive 17:00-20:00, leave next day 07:00-09:00 (15-min aligned)
    step_h = pd.Timedelta(freq).total_seconds() / 3600.0
    def snap(h: float) -> pd.Timedelta:
        return pd.Timedelta(hours=round(h / step_h) * step_h)

    ev_stints: Dict[str, List[Stint]] = {}
    t0 = pd.Timestamp(start)
    for j, name in enumerate(cols):
        stints: List[Stint] = []
        for d in range(days):
            day = t0 + pd.Timedelta(days=d)
            arr = day + snap(rng.uniform(17.0, 20.0))
            dep = day + pd.Timedelta(days=1) + snap(rng.uniform(7.0, 9.0))
            e_init = float(rng.uniform(0.25, 0.55) * config.EV_Smax_kWh)
            e_min = float(rng.uniform(0.60, 0.85) * config.EV_Smax_kWh)
            stints.append((arr, dep, e_init, e_min))
        ev_stints[name] = stints

    data = StudyData(idx, load_kw, pv_kw, common_kw, price_buy, price_sell, ev_stints)
    data.validate()
    return data


# ---------------------------------------------------------------------------
# CSV / JSON I/O
# ---------------------------------------------------------------------------

def export_csv(data: StudyData, folder: str) -> str:
    p = pathlib.Path(folder)
    p.mkdir(parents=True, exist_ok=True)
    data.load_kw.round(4).to_csv(p / "load_kw.csv")
    data.pv_kw.round(4).to_csv(p / "pv_kw.csv")
    data.common_kw.round(4).to_frame("common_kw").to_csv(p / "common_kw.csv")
    data.price_buy.to_frame("price_buy").to_csv(p / "price_buy.csv")
    data.price_sell.to_frame("price_sell").to_csv(p / "price_sell.csv")
    ev_json = {k: [[str(a), str(d), round(i, 3), round(m, 3)] for (a, d, i, m) in v]
               for k, v in data.ev_stints.items()}
    (p / "ev_stints.json").write_text(json.dumps(ev_json, indent=2))
    return str(p)


def load_csv(folder: str) -> StudyData:
    p = pathlib.Path(folder)
    rd = lambda f: pd.read_csv(p / f, index_col=0, parse_dates=True)
    load_kw = rd("load_kw.csv")
    data = StudyData(
        index=pd.DatetimeIndex(load_kw.index),
        load_kw=load_kw,
        pv_kw=rd("pv_kw.csv"),
        common_kw=rd("common_kw.csv")["common_kw"],
        price_buy=rd("price_buy.csv")["price_buy"],
        price_sell=rd("price_sell.csv")["price_sell"],
        ev_stints={k: [(pd.Timestamp(a), pd.Timestamp(d), float(i), float(m)) for a, d, i, m in v]
                   for k, v in json.loads((p / "ev_stints.json").read_text()).items()},
    )
    data.validate()
    return data


if __name__ == "__main__":
    out = export_csv(make_synthetic(BuildingConfig()), "./demo_data")
    print("Wrote demo data to:", out)
