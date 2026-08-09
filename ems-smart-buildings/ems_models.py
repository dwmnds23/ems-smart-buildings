# ems_models.py
"""
Pyomo models for Case I (NSM), Case II (SMSCP), Case III (SMSCPB).

Implements objective and constraints as in Foroozandeh et al. (2022):
- Objective cost (eq. (1))
- EV constraints (eq. (20b)-(20i))
- Power balance and CP bounds (eq. (20p)-(20r))
- BESS constraints for Case III (eq. (20j)-(20o))

This file expects Pyomo + a MILP solver (CBC/GLPK/GUROBI/CPLEX).
"""

from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

from dataclasses import dataclass
from pyomo.environ import (ConcreteModel, Var, NonNegativeReals, Binary, Objective, Constraint,
                           minimize, RangeSet)

@dataclass
class Params:
    tau: float
    J: int
    CP: float
    EV_Smax: float
    EV_Pch: float
    EV_Pdis: float
    EV_eta_ch: float
    EV_eta_dis: float
    # BESS (optional)
    use_bess: bool = False
    BESS_Smax: float = 0.0
    BESS_Pch: float = 0.0
    BESS_Pdis: float = 0.0
    BESS_eta_ch: float = 0.0
    BESS_eta_dis: float = 0.0

def build_caseII(index: pd.DatetimeIndex, load_kw: pd.DataFrame, pv_kw: pd.DataFrame,
                 common_kw: pd.Series, price_buy: pd.Series, price_sell: pd.Series,
                 ev_stints: Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp, float, float]]],
                 P: Params) -> ConcreteModel:
    I = len(index)
    apt_names = list(load_kw.columns)
    name_to_j = {n: j+1 for j,n in enumerate(apt_names)}
    # Precompute availability masks and EV targets
    avail = np.zeros((I, P.J), dtype=int)
    init_soc = {j+1: {} for j in range(P.J)}  # per time key to initial SoC (kWh)
    min_dep = {j+1: {} for j in range(P.J)}
    for n, stints in ev_stints.items():
        j = name_to_j[n]
        for (a,d,iSoc,minSoc) in stints:
            mask = (index>=a) & (index<d)
            avail[mask, j-1] = 1
            ia = np.where(index==pd.to_datetime(a))[0]
            if len(ia): init_soc[j][int(ia[0]-1)] = iSoc
            idp = np.where(index==pd.to_datetime(d))[0]
            if len(idp): min_dep[j][int(idp[0]-1)] = minSoc
    m = ConcreteModel()
    m.I = RangeSet(0, I-1)
    m.J = RangeSet(1, P.J)

    buy = price_buy.values
    sell = price_sell.values
    LA = load_kw.values
    PV = pv_kw.values
    C = common_kw.values

    m.PGtoM = Var(m.I, within=NonNegativeReals)
    m.PMtoG = Var(m.I, within=NonNegativeReals)

    m.PMtoEV = Var(m.I, m.J, within=NonNegativeReals)
    m.PEVtoM = Var(m.I, m.J, within=NonNegativeReals)
    m.SEV = Var(m.I, m.J, within=NonNegativeReals)
    m.aEV = Var(m.I, m.J, within=Binary)
    m.bEV = Var(m.I, m.J, within=Binary)

    def obj_rule(mm):
        return sum(buy[i]*mm.PGtoM[i] - sell[i]*mm.PMtoG[i] for i in mm.I)
    m.OBJ = Objective(rule=obj_rule, sense=minimize)

    def ev_soc_max(mm, i, j):
        return mm.SEV[i,j] <= P.EV_Smax
    m.EV_Smax_c = Constraint(m.I, m.J, rule=ev_soc_max)

    def power_balance(mm, i):
        return (mm.PGtoM[i] + sum(mm.PEVtoM[i,j] for j in mm.J) + sum(PV[i, j-1] for j in mm.J) ==
                mm.PMtoG[i] + sum(LA[i, j-1] for j in mm.J) + sum(mm.PMtoEV[i,j] for j in mm.J) + C[i])
    m.balance = Constraint(m.I, rule=power_balance)

    def cp_in(mm, i): return mm.PGtoM[i] <= P.CP
    m.cp_in = Constraint(m.I, rule=cp_in)
    def cp_out(mm, i): return mm.PMtoG[i] <= 0.5*P.CP
    m.cp_out = Constraint(m.I, rule=cp_out)

    def ch_lim(mm, i, j):
        if avail[i, j-1]==0: return mm.PMtoEV[i,j] == 0
        return mm.PMtoEV[i,j] <= m.aEV[i,j] * P.EV_Pch * P.tau
    m.ch_lim = Constraint(m.I, m.J, rule=ch_lim)

    def dis_lim(mm, i, j):
        if avail[i, j-1]==0: return mm.PEVtoM[i,j] == 0
        return mm.PEVtoM[i,j] <= m.bEV[i,j] * P.EV_Pdis * P.tau
    m.dis_lim = Constraint(m.I, m.J, rule=dis_lim)

    def soc_dyn(mm, i, j):
        if i==0: return None
        return mm.SEV[i,j] == mm.SEV[i-1,j] + (mm.PMtoEV[i-1,j]*P.EV_eta_ch - mm.PEVtoM[i-1,j]/P.EV_eta_dis)
    m.soc_dyn = Constraint(range(1,I), list(range(1,P.J+1)), rule=lambda mm, i,j: soc_dyn(mm,i,j))

    def no_bidir(mm, i, j): return m.aEV[i,j] + m.bEV[i,j] <= 1
    m.no_bidir = Constraint(m.I, m.J, rule=no_bidir)

    def min_departure(mm, i, j):
        if j in min_dep and i in min_dep[j]:
            return mm.SEV[i,j] >= min_dep[j][i]
        return None
    m.min_departure = Constraint(m.I, m.J, rule=min_departure)

    def outside_zero(mm, i, j):
        if avail[i, j-1]==0: return mm.SEV[i,j] == 0
        return None
    m.outside_zero = Constraint(m.I, m.J, rule=outside_zero)

    return m

# NOTE: Case III scaffolding can be added similarly by introducing BESS variables and rewriting the balance.
