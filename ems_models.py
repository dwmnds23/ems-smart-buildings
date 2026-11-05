"""
Optimisation models for the three cases of Foroozandeh et al. (2022).

    Case I   (NSM)    : no management - individual contracts, EVs charge on arrival
    Case II  (SMSCP)  : one building contract, EV charge/discharge scheduled (MILP)
    Case III (SMSCPB) : Case II + a shared battery energy storage system (MILP)

Notation follows the paper (eqs. (1)-(20)). All flows are powers in kW held
constant over a step of tau hours; state of charge is energy in kWh, so every
power that enters the SoC equation or the cost is multiplied by tau.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pyomo.environ import (Binary, ConcreteModel, Constraint, NonNegativeReals,
                           Objective, RangeSet, Set, SolverFactory, TerminationCondition,
                           Var, minimize, value)

from data_schema import BuildingConfig, StudyData


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class StintIdx:
    """An EV parking stint expressed in time-step indices.

    The EV is plugged in during steps a, a+1, ..., d-1 (d exclusive).
    e_init is the SoC at arrival; e_min the minimum SoC at step d-1 (the
    departure target, or the arrival SoC if the stint runs past the horizon).
    """
    a: int
    d: int
    e_init: float
    e_min: float


def stint_indices(data: StudyData) -> Dict[str, List[StintIdx]]:
    T = len(data.index)
    out: Dict[str, List[StintIdx]] = {}
    for apt, stints in data.ev_stints.items():
        lst = []
        for arr, dep, e_init, e_min in stints:
            a = int(data.index.searchsorted(pd.Timestamp(arr), side="left"))
            d = int(data.index.searchsorted(pd.Timestamp(dep), side="left"))
            if a >= T or d <= 0 or d <= a:
                continue
            # If the EV is still plugged in at the end of the horizon there is no
            # departure target; we then require it to end with at least the
            # energy it arrived with (end-of-horizon condition).
            lst.append(StintIdx(max(a, 0), min(d, T), e_init, e_min if d <= T - 1 else e_init))
        out[apt] = lst
    return out


# ---------------------------------------------------------------------------
# Case I - no management (reference case, no optimisation)
# ---------------------------------------------------------------------------

def uncontrolled_ev_charging(data: StudyData, cfg: BuildingConfig) -> pd.DataFrame:
    """EVs charge at full power as soon as they arrive (paper, Sec. 4.1), until
    they reach the same departure target used in Cases II/III, so all cases
    deliver the same energy to the cars."""
    tau = data.tau
    ch = pd.DataFrame(0.0, index=data.index, columns=data.apartments)
    for apt, stints in stint_indices(data).items():
        col = ch.columns.get_loc(apt)
        for s in stints:
            e = s.e_init
            for t in range(s.a, s.d):
                p = min(cfg.EV_Pch_kW, (s.e_min - e) / (cfg.EV_eta_ch * tau))
                if p <= 1e-9:
                    break
                ch.iat[t, col] = p
                e += cfg.EV_eta_ch * p * tau
    return ch


def solve_case_I(data: StudyData, cfg: BuildingConfig) -> dict:
    """Each apartment is a separate consumer with its own contract and PV."""
    tau = data.tau
    ev = uncontrolled_ev_charging(data, cfg)
    net = data.load_kw + ev - data.pv_kw                       # kW, per apartment
    buy = data.price_buy.values[:, None]
    sell = data.price_sell.values[:, None]
    energy_cost = (np.where(net > 0, net * buy, net * sell) * tau).sum(axis=0)

    cps = list(cfg.apartment_cp_kW)[: len(data.apartments)]
    if len(cps) < len(data.apartments):
        cps += [cps[-1] if cps else 6.9] * (len(data.apartments) - len(cps))
    cp_cost = np.array(cps) * cfg.cp_charge_eur_per_kw_day * data.days

    common_cost = float((data.common_kw * data.price_buy).sum() * tau
                        + cfg.common_cp_kW * cfg.cp_charge_eur_per_kw_day * data.days)
    per_apt = pd.DataFrame({
        "contract_kW": cps,
        "energy_cost_eur": energy_cost,
        "contract_cost_eur": cp_cost,
        "total_eur": energy_cost + cp_cost,
        "peak_kW": net.max().values,
        "ev_charge_kWh": (ev * tau).sum().values,
    }, index=data.apartments)
    building_import = (net.clip(lower=0).sum(axis=1) + data.common_kw)
    return {
        "total_eur": float(per_apt["total_eur"].sum() + common_cost),
        "common_eur": common_cost,
        "per_apartment": per_apt,
        "sum_of_contracts_kW": float(sum(cps) + cfg.common_cp_kW),
        "building_peak_kW": float(building_import.max()),
        "ev_charge": ev,
    }


# ---------------------------------------------------------------------------
# Cases II and III - MILP
# ---------------------------------------------------------------------------

def build_model(data: StudyData, cfg: BuildingConfig, CP: float, use_bess: bool) -> ConcreteModel:
    """Single-contract-power MILP (Case II if use_bess=False, Case III otherwise)."""
    tau = data.tau
    T = len(data.index)
    apts = data.apartments
    LA = data.load_kw.values
    PV = data.pv_kw.values.sum(axis=1)
    PC = data.common_kw.values
    buy = data.price_buy.values
    sell = data.price_sell.values
    stints = stint_indices(data)

    # (t, j) pairs where EV j is plugged in; variables exist only there, which
    # is the paper's eq. (8) (SoC = 0 / no flow while the EV is away).
    present = sorted({(t, j) for j, a in enumerate(apts) for s in stints[a] for t in range(s.a, s.d)})
    first_step = {}  # (t, j) -> e_init when t is an arrival step
    for j, a in enumerate(apts):
        for s in stints[a]:
            first_step[(s.a, j)] = s.e_init

    m = ConcreteModel(name=f"SMSCP{'B' if use_bess else ''}_CP{CP}")
    m.T = RangeSet(0, T - 1)
    m.EV = Set(initialize=present, dimen=2)

    # Grid exchange (kW)
    m.p_imp = Var(m.T, within=NonNegativeReals)              # P_G->M
    m.p_exp = Var(m.T, within=NonNegativeReals)              # P_M->G
    m.p_curt = Var(m.T, bounds=lambda mm, t: (0.0, float(PV[t])))  # PV curtailment

    # EVs (eqs. (2)-(9))
    m.ev_ch = Var(m.EV, within=NonNegativeReals)             # P_M->EV (kW)
    m.ev_dis = Var(m.EV, within=NonNegativeReals)            # P_EV->M (kW)
    m.ev_e = Var(m.EV, bounds=(0.0, cfg.EV_Smax_kWh))        # S_EV (kWh), eq. (2)
    m.ev_a = Var(m.EV, within=Binary)                        # alpha_EV
    m.ev_b = Var(m.EV, within=Binary)                        # beta_EV

    m.ev_ch_lim = Constraint(m.EV, rule=lambda mm, t, j: mm.ev_ch[t, j] <= cfg.EV_Pch_kW * mm.ev_a[t, j])
    m.ev_dis_lim = Constraint(m.EV, rule=lambda mm, t, j: mm.ev_dis[t, j] <= cfg.EV_Pdis_kW * mm.ev_b[t, j])
    m.ev_excl = Constraint(m.EV, rule=lambda mm, t, j: mm.ev_a[t, j] + mm.ev_b[t, j] <= 1)

    def ev_soc(mm, t, j):  # eqs. (3) and (6): SoC at the END of step t
        prev = first_step[(t, j)] if (t, j) in first_step else mm.ev_e[t - 1, j]
        return mm.ev_e[t, j] == prev + tau * (cfg.EV_eta_ch * mm.ev_ch[t, j]
                                              - mm.ev_dis[t, j] / cfg.EV_eta_dis)
    m.ev_soc = Constraint(m.EV, rule=ev_soc)

    dep = [(s.d - 1, j, s.e_min) for j, a in enumerate(apts) for s in stints[a]]
    m.DEP = Set(initialize=[(t, j) for t, j, _ in dep], dimen=2)
    dep_min = {(t, j): e for t, j, e in dep}
    m.ev_departure = Constraint(m.DEP, rule=lambda mm, t, j: mm.ev_e[t, j] >= dep_min[(t, j)])  # eq. (7)

    ev_ch_at = {t: [] for t in range(T)}
    for (t, j) in present:
        ev_ch_at[t].append(j)

    # Shared BESS (Case III, eqs. (14)-(19))
    if use_bess:
        m.b_ch = Var(m.T, within=NonNegativeReals)
        m.b_dis = Var(m.T, within=NonNegativeReals)
        m.b_e = Var(m.T, bounds=(cfg.BESS_Smin_kWh, cfg.BESS_Smax_kWh))
        m.b_a = Var(m.T, within=Binary)
        m.b_b = Var(m.T, within=Binary)
        m.b_ch_lim = Constraint(m.T, rule=lambda mm, t: mm.b_ch[t] <= cfg.BESS_Pch_kW * mm.b_a[t])
        m.b_dis_lim = Constraint(m.T, rule=lambda mm, t: mm.b_dis[t] <= cfg.BESS_Pdis_kW * mm.b_b[t])
        m.b_excl = Constraint(m.T, rule=lambda mm, t: mm.b_a[t] + mm.b_b[t] <= 1)

        def b_soc(mm, t):
            prev = cfg.BESS_init_kWh if t == 0 else mm.b_e[t - 1]
            return mm.b_e[t] == prev + tau * (cfg.BESS_eta_ch * mm.b_ch[t] - mm.b_dis[t] / cfg.BESS_eta_dis)
        m.b_soc = Constraint(m.T, rule=b_soc)
        # End-of-horizon condition: do not end with less energy than we started
        # with, otherwise the optimiser "sells" the initial charge for free.
        m.b_terminal = Constraint(expr=m.b_e[T - 1] >= cfg.BESS_init_kWh)

    # Power balance, eq. (10) / (13)
    def balance(mm, t):
        supply = mm.p_imp[t] + (PV[t] - mm.p_curt[t]) + sum(mm.ev_dis[t, j] for j in ev_ch_at[t])
        demand = mm.p_exp[t] + float(LA[t].sum() + PC[t]) + sum(mm.ev_ch[t, j] for j in ev_ch_at[t])
        if use_bess:
            supply += mm.b_dis[t]
            demand += mm.b_ch[t]
        return supply == demand
    m.balance = Constraint(m.T, rule=balance)

    # Contract-power limits, eqs. (11)-(12)
    m.cp_imp = Constraint(m.T, rule=lambda mm, t: mm.p_imp[t] <= CP)
    m.cp_exp = Constraint(m.T, rule=lambda mm, t: mm.p_exp[t] <= cfg.export_fraction_of_cp * CP)

    # Objective, eq. (1): contract charge + energy bought - energy sold (EUR)
    cp_cost = cfg.cp_charge_eur_per_kw_day * CP * data.days
    m.cp_cost = cp_cost
    m.OBJ = Objective(expr=cp_cost + tau * sum(buy[t] * m.p_imp[t] - sell[t] * m.p_exp[t] for t in range(T)),
                      sense=minimize)
    m._meta = dict(CP=CP, use_bess=use_bess, apts=apts, T=T)
    return m


def _pick_solver(name: str):
    names = ["cbc", "glpk", "appsi_highs"] if name == "auto" else [name]
    for n in names:
        opt = SolverFactory(n)
        try:
            if opt.available(exception_flag=False):
                return n, opt
        except Exception:
            continue
    raise RuntimeError(f"No MILP solver available among {names}. Install CBC (conda install -c "
                       "conda-forge coincbc) or HiGHS (pip install highspy).")


def solve(m: ConcreteModel, solver: str = "auto", mip_gap: float = 0.005,
          time_limit: int = 300) -> Tuple[str, str]:
    """Solve the MILP. Returns (status, solver_name); status is 'optimal',
    'feasible' (stopped at gap/time limit with a solution) or 'infeasible'."""
    name, opt = _pick_solver(solver)
    if name == "appsi_highs":
        opt.config.load_solution = False
        opt.config.time_limit = time_limit
        opt.config.mip_gap = mip_gap
        res = opt.solve(m)
        if res.best_feasible_objective is None:
            return "infeasible", name
        res.solution_loader.load_vars()
        tc = str(res.termination_condition).lower()
        return ("optimal" if "optimal" in tc else "feasible"), name

    if name == "cbc":
        opts = {"ratioGap": mip_gap, "seconds": time_limit}
    elif name == "glpk":
        opts = {"mipgap": mip_gap, "tmlim": time_limit}
    else:
        opts = {}
    res = opt.solve(m, options=opts, load_solutions=False)
    tc = res.solver.termination_condition
    if tc in (TerminationCondition.infeasible, TerminationCondition.infeasibleOrUnbounded) \
            or len(res.solution) == 0:
        return "infeasible", name
    m.solutions.load_from(res)
    return ("optimal" if tc == TerminationCondition.optimal else "feasible"), name


def extract_dispatch(m: ConcreteModel, data: StudyData) -> pd.DataFrame:
    """Time series of the solved model (kW), aggregated to building level."""
    T = len(data.index)
    ev_ch = np.zeros(T)
    ev_dis = np.zeros(T)
    for (t, j) in m.EV:
        ev_ch[t] += value(m.ev_ch[t, j])
        ev_dis[t] += value(m.ev_dis[t, j])
    df = pd.DataFrame({
        "apartments_kW": data.load_kw.sum(axis=1).values,
        "common_kW": data.common_kw.values,
        "pv_kW": data.pv_kw.sum(axis=1).values - np.array([value(m.p_curt[t]) for t in range(T)]),
        "pv_curtailed_kW": [value(m.p_curt[t]) for t in range(T)],
        "ev_charge_kW": ev_ch,
        "ev_discharge_kW": ev_dis,
        "grid_import_kW": [value(m.p_imp[t]) for t in range(T)],
        "grid_export_kW": [value(m.p_exp[t]) for t in range(T)],
    }, index=data.index)
    if m._meta["use_bess"]:
        df["bess_charge_kW"] = [value(m.b_ch[t]) for t in range(T)]
        df["bess_discharge_kW"] = [value(m.b_dis[t]) for t in range(T)]
        df["bess_soc_kWh"] = [value(m.b_e[t]) for t in range(T)]
    return df.clip(lower=0).round(6)


def allocate_costs(m: ConcreteModel, data: StudyData) -> pd.DataFrame:
    """Step (ii) of the paper: share the building bill among apartments.

    d(t, j) = P_A + P_M->EV - P_PV - P_EV->M is apartment j's net demand.
    Positive d is billed at the buy price, negative d credited at the sell price
    (paper, Sec. 4.2.1(b)). Whatever remains of the building bill (contract
    charge, common services, BESS losses/gains, curtailment) is split in
    proportion to each apartment's gross consumption.
    """
    tau = data.tau
    apts = data.apartments
    T = len(data.index)
    ch = np.zeros((T, len(apts)))
    dis = np.zeros((T, len(apts)))
    for (t, j) in m.EV:
        ch[t, j] = value(m.ev_ch[t, j])
        dis[t, j] = value(m.ev_dis[t, j])
    d = data.load_kw.values + ch - data.pv_kw.values - dis
    buy = data.price_buy.values[:, None]
    sell = data.price_sell.values[:, None]
    own = (np.where(d > 0, d * buy, d * sell) * tau).sum(axis=0)
    total = value(m.OBJ)
    residual = total - own.sum()
    gross = (data.load_kw.values + ch).sum(axis=0)
    share = residual * gross / gross.sum()
    return pd.DataFrame({"energy_share_eur": own, "shared_costs_eur": share,
                         "total_eur": own + share}, index=apts)
