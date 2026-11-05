"""
Run Cases I, II and III and sweep the single contract power (CP).

    python run_cases.py                      # all cases on ./demo_data
    python run_cases.py --case II --solver glpk
    python run_cases.py --regenerate --days 7

Outputs (in --out, default ./results):
    summary.csv            one row per case at its cost-optimal CP
    cp_sweep.csv           cost / peak / status for every CP candidate
    apartment_costs.csv    per-apartment bill for each case (step (ii))
    dispatch_case_*.csv    building-level time series of the optimal schedule
    *.png                  figures (see plots.py)
"""

from __future__ import annotations

import argparse
import pathlib
import time

import pandas as pd
from pyomo.environ import value

from data_schema import CP_CANDIDATES_KW, BuildingConfig, export_csv, load_csv, make_synthetic
from ems_models import allocate_costs, build_model, extract_dispatch, solve, solve_case_I


def sweep(data, cfg, use_bess, cps, solver, mip_gap, time_limit, label):
    rows, best = [], None
    for CP in cps:
        t0 = time.time()
        m = build_model(data, cfg, CP, use_bess)
        status, solver_name = solve(m, solver, mip_gap, time_limit)
        row = {"case": label, "CP_kW": CP, "status": status, "solver": solver_name,
               "seconds": round(time.time() - t0, 1)}
        if status != "infeasible":
            disp = extract_dispatch(m, data)
            row.update(total_eur=value(m.OBJ), contract_eur=m.cp_cost,
                       energy_eur=value(m.OBJ) - m.cp_cost,
                       peak_import_kW=disp["grid_import_kW"].max())
            if best is None or row["total_eur"] < best["row"]["total_eur"]:
                best = {"row": row, "model": m, "dispatch": disp}
        rows.append(row)
        cost = f"EUR {row['total_eur']:9.2f}  peak {row['peak_import_kW']:5.2f} kW" \
            if status != "infeasible" else "infeasible"
        print(f"  {label:<3} CP {CP:5.2f} kW  {cost}  [{status}, {row['seconds']} s]")
    return rows, best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="./demo_data")
    ap.add_argument("--out", default="./results")
    ap.add_argument("--case", choices=["I", "II", "III", "all"], default="all")
    ap.add_argument("--solver", default="auto", help="auto | cbc | glpk | appsi_highs")
    ap.add_argument("--mip-gap", type=float, default=0.005)
    ap.add_argument("--time-limit", type=int, default=300, help="seconds per MILP")
    ap.add_argument("--regenerate", action="store_true", help="rebuild synthetic demo data first")
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    cfg = BuildingConfig()
    if args.regenerate:
        export_csv(make_synthetic(cfg, days=args.days), args.data)
    data = load_csv(args.data)
    cfg.J = len(data.apartments)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(data.apartments)} apartments, {len(data.index)} steps of {data.tau * 60:.0f} min "
          f"({data.days:.1f} days)\n")

    summary, sweep_rows, apt_costs, dispatches = [], [], {}, {}

    if args.case in ("I", "all"):
        r1 = solve_case_I(data, cfg)
        summary.append({"case": "I", "CP_kW": r1["sum_of_contracts_kW"], "total_eur": r1["total_eur"],
                        "peak_import_kW": r1["building_peak_kW"], "status": "no optimisation"})
        apt_costs["I"] = r1["per_apartment"]["total_eur"]
        print(f"Case I   (NSM)    EUR {r1['total_eur']:9.2f}  | sum of 16 individual contracts "
              f"{r1['sum_of_contracts_kW']:.1f} kW | building peak {r1['building_peak_kW']:.2f} kW\n")

    for label, use_bess in (("II", False), ("III", True)):
        if args.case not in (label, "all"):
            continue
        print(f"Case {label:<3} ({'SMSCPB' if use_bess else 'SMSCP'}) - CP sweep")
        rows, best = sweep(data, cfg, use_bess, CP_CANDIDATES_KW, args.solver,
                           args.mip_gap, args.time_limit, label)
        sweep_rows += rows
        if best is None:
            print(f"  Case {label}: no feasible CP among the candidates.\n")
            continue
        b = best["row"]
        summary.append({k: b[k] for k in ("case", "CP_kW", "total_eur", "peak_import_kW", "status")})
        apt_costs[label] = allocate_costs(best["model"], data)["total_eur"]
        dispatches[label] = best["dispatch"]
        best["dispatch"].to_csv(out / f"dispatch_case_{label}.csv")
        print(f"  -> optimal CP {b['CP_kW']} kW, EUR {b['total_eur']:.2f}\n")

    summ = pd.DataFrame(summary)
    if "I" in set(summ.get("case", [])):
        base = summ.loc[summ.case == "I", "total_eur"].iloc[0]
        summ["saving_vs_case_I_pct"] = (100 * (1 - summ.total_eur / base)).round(1)
    summ.round(2).to_csv(out / "summary.csv", index=False)
    pd.DataFrame(sweep_rows).round(2).to_csv(out / "cp_sweep.csv", index=False)
    if apt_costs:
        pd.DataFrame(apt_costs).round(2).rename_axis("apartment").to_csv(out / "apartment_costs.csv")

    print("Summary")
    print(summ.round(2).to_string(index=False))

    if not args.no_plots:
        import plots
        plots.make_all(out, summ, pd.DataFrame(sweep_rows), dispatches)
        print(f"\nFigures written to {out}/")


if __name__ == "__main__":
    main()
