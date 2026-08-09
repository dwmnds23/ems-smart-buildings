# run_cases.py
"""
Runner script: loads CSV/JSON data (see data_schema.export_csv), builds and solves
Cases I/II/III with Pyomo.

USAGE (Anaconda prompt):
    conda install -c conda-forge pyomo coincbc pandas numpy matplotlib
    python run_cases.py --data ./demo_data --case all

This file does not execute here because solvers are not installed in this environment.
"""

import json, argparse, sys, pathlib
import numpy as np, pandas as pd
from data_schema import BuildingConfig, StudyData, CP_CANDIDATES_KW
from ems_models import Params, build_caseII

from pyomo.environ import SolverFactory, value

def load_data(folder: str):
    p = pathlib.Path(folder)
    load_kw = pd.read_csv(p/"load_kw.csv", index_col=0, parse_dates=True)
    pv_kw = pd.read_csv(p/"pv_kw.csv", index_col=0, parse_dates=True)
    common_kw = pd.read_csv(p/"common_kw.csv", index_col=0, parse_dates=True)["common_kw"]
    price_buy = pd.read_csv(p/"price_buy.csv", index_col=0, parse_dates=True)["price_buy"]
    price_sell = pd.read_csv(p/"price_sell.csv", index_col=0, parse_dates=True)["price_sell"]
    ev_stints = json.loads((p/"ev_stints.json").read_text())
    for k, lst in ev_stints.items():
        ev_stints[k] = [(pd.Timestamp(a), pd.Timestamp(d), float(i), float(m)) for (a,d,i,m) in lst]
    idx = load_kw.index
    return idx, load_kw, pv_kw, common_kw, price_buy, price_sell, ev_stints

def solve_case_I(idx, load_kw, pv_kw, common_kw, price_buy, price_sell):
    net_demand = (load_kw.sum(axis=1) + common_kw) - pv_kw.sum(axis=1)
    cost = (np.where(net_demand>=0, net_demand*price_buy, net_demand*price_sell)).sum()
    return float(cost), float(net_demand.max())

def enumerate_case(idx, load_kw, pv_kw, common_kw, price_buy, price_sell, ev_stints, Pbase, cps):
    best = None
    rows = []
    for CP in cps:
        P = Pbase
        P.CP = CP
        m = build_caseII(idx, load_kw, pv_kw, common_kw, price_buy, price_sell, ev_stints, P)
        opt = SolverFactory("cbc") or SolverFactory("glpk")
        res = opt.solve(m, tee=False)
        val = m.OBJ()
        peak = max(float(m.PGtoM[i]()) for i in m.I)
        rows.append((CP, float(val), peak))
        if (best is None) or (val < best[1]):
            best = (CP, float(val), peak)
    return best, rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./demo_data")     # <-- default
    ap.add_argument("--case", choices=["I","II","III","all"], default="all")
    args = ap.parse_args()
    # ...


    idx, load_kw, pv_kw, common_kw, price_buy, price_sell, ev_stints = load_data(args.data)
    cfg = BuildingConfig(J=load_kw.shape[1])

    if args.case in ("I","all"):
        costI, peakI = solve_case_I(idx, load_kw, pv_kw, common_kw, price_buy, price_sell)
        print(f"Case I — cost: €{costI:,.2f} | peak: {peakI:.2f} kW")

    if args.case in ("II","all"):
        P = Params(tau=cfg.tau_hours, J=cfg.J, CP=27.6,
                   EV_Smax=cfg.EV_Smax_kWh, EV_Pch=cfg.EV_Pch_kW, EV_Pdis=cfg.EV_Pdis_kW,
                   EV_eta_ch=cfg.EV_eta_ch, EV_eta_dis=cfg.EV_eta_dis)
        from data_schema import CP_CANDIDATES_KW
        best, table = enumerate_case(idx, load_kw, pv_kw, common_kw, price_buy, price_sell, ev_stints, P, CP_CANDIDATES_KW)
        print("Case II — CP sweep (CP kW, total € , peak kW):")
        for CP, cost, peak in table: print(f"  {CP:5.2f}  €{cost:,.2f}  {peak:5.2f}")
        print(f"Best CP (paper expects 27.6 kW): {best[0]} | cost €{best[1]:,.2f} | peak {best[2]:.2f} kW")

    if args.case in ("III","all"):
        print("Case III scaffold present in ems_models.py — extend to include BESS flows and rerun sweep. Paper expects 20.7 kW.")

if __name__ == "__main__":
    main()
