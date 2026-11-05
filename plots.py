"""Figures for the README: CP sweep and the optimal dispatch of Cases II/III."""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a",
     "yellow": "#eda100", "violet": "#6250d6"}


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_cp_sweep(sweep: pd.DataFrame, summary: pd.DataFrame, path: pathlib.Path):
    fig, ax = plt.subplots(figsize=(8, 4.6), facecolor=SURFACE)
    _style(ax)
    cps = sorted(sweep.CP_kW.unique())
    x = np.arange(len(cps))
    w = 0.38
    series = [("II", "Case II  (EVs only)", C["blue"], -w / 2 - 0.01),
              ("III", "Case III (EVs + BESS)", C["orange"], w / 2 + 0.01)]
    ymax = sweep["total_eur"].max() if "total_eur" in sweep else 1
    for case, label, color, off in series:
        sub = sweep[sweep.case == case].set_index("CP_kW")
        if sub.empty:
            continue
        for k, cp in enumerate(cps):
            r = sub.loc[cp]
            if r.status == "infeasible":
                ax.text(x[k] + off, ymax * 0.03, "infeasible", rotation=90, ha="center",
                        va="bottom", fontsize=8, color=INK_2)
            else:
                ax.bar(x[k] + off, r.total_eur, w, color=color, edgecolor=SURFACE, linewidth=2,
                       label=label if k == next(i for i, c in enumerate(cps)
                                               if sub.loc[c].status != "infeasible") else None)
        best = summary[summary.case == case]
        if not best.empty:
            k = cps.index(best.CP_kW.iloc[0])
            v = best.total_eur.iloc[0]
            ax.annotate(f"€{v:,.0f}", (x[k] + off, v), xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=8.5, color=INK, fontweight="bold")
    base = summary[summary.case == "I"]
    if not base.empty:
        v = base.total_eur.iloc[0]
        ax.axhline(v, color=INK_2, linestyle=(0, (4, 3)), linewidth=1.2)
        ax.text(len(cps) - 0.5, v, f"Case I, no management: €{v:,.0f}", ha="right", va="bottom",
                fontsize=8.5, color=INK_2)
    ax.set_xticks(x, [f"{c:g}" for c in cps])
    ax.set_xlabel("Single contract power of the building (kW)", color=INK_2, fontsize=9)
    ax.set_ylabel("Building electricity cost (€)", color=INK_2, fontsize=9)
    ax.set_title("Cost vs. contract power — bold label marks the optimum", loc="left",
                 fontsize=11, color=INK)
    ax.set_xlim(-0.8, len(cps) - 0.4)
    ax.set_ylim(0, ymax * 1.18 if base.empty else max(ymax, base.total_eur.iloc[0]) * 1.12)
    ax.legend(frameon=False, fontsize=9, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.16),
              labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_dispatch(disp: pd.DataFrame, title: str, path: pathlib.Path, days: int = 3, cp: float = None):
    d = disp.iloc[: int(days * 24 / ((disp.index[1] - disp.index[0]).total_seconds() / 3600))]
    t = d.index
    fig, ax = plt.subplots(figsize=(10, 4.6), facecolor=SURFACE)
    _style(ax)
    load = (d.apartments_kW + d.common_kW).values
    pos = [("Apartments + common load", load, C["blue"], None),
           ("EV charging", d.ev_charge_kW.values, C["orange"], None)]
    neg = [("PV", d.pv_kW.values, C["aqua"], None),
           ("EV discharging (V2H)", d.ev_discharge_kW.values, C["violet"], None)]
    if "bess_charge_kW" in d:
        pos.append(("BESS charging", d.bess_charge_kW.values, C["yellow"], None))
        neg.append(("BESS discharging", d.bess_discharge_kW.values, C["yellow"], "////"))
    for group, sign in ((pos, 1), (neg, -1)):
        base = np.zeros(len(d))
        for label, y, color, hatch in group:
            ax.fill_between(t, sign * base, sign * (base + y), step="post", color=color,
                            alpha=0.9 if hatch is None else 0.35, hatch=hatch, edgecolor=color if hatch else None,
                            linewidth=0, label=label)
            base = base + y
    ax.step(t, d.grid_import_kW.values, where="post", color=INK, linewidth=1.6, label="Import from grid")
    if cp is not None:
        ax.axhline(cp, color=INK_2, linestyle=(0, (4, 3)), linewidth=1)
        ax.text(t[2], cp, f"CP = {cp:g} kW", va="bottom", fontsize=8.5, color=INK_2)
    ax.axhline(0, color=INK_2, linewidth=0.8)
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %H:%M"))
    ax.set_ylabel("kW  (consumption ↑ / sources ↓)", color=INK_2, fontsize=9)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=8.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12),
              labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def make_all(out, summary, sweep, dispatches):
    out = pathlib.Path(out)
    if not sweep.empty:
        plot_cp_sweep(sweep, summary, out / "cp_sweep.png")
    for case, disp in dispatches.items():
        cp = summary.loc[summary.case == case, "CP_kW"].iloc[0]
        name = "Case II — smart management, single contract" if case == "II" else \
            "Case III — smart management, single contract + shared BESS"
        plot_dispatch(disp, f"{name} (first 3 days)", out / f"dispatch_case_{case}.png", cp=cp)
