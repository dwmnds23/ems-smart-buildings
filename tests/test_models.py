"""Physical-consistency checks on a small instance (3 apartments, 2 days)."""

import sys
import pathlib

import numpy as np
import pytest
from pyomo.environ import value

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from data_schema import BuildingConfig, make_synthetic  # noqa: E402
from ems_models import (build_model, extract_dispatch, solve, solve_case_I,  # noqa: E402
                        stint_indices)

CP = 12.0


@pytest.fixture(scope="module")
def setup():
    cfg = BuildingConfig(J=3, apartment_cp_kW=[6.9, 10.35, 13.8])
    data = make_synthetic(cfg, days=2, seed=1)
    out = {}
    for bess in (False, True):
        m = build_model(data, cfg, CP, bess)
        status, _ = solve(m)
        assert status != "infeasible"
        out[bess] = m
    return cfg, data, out


def test_tau_inferred(setup):
    _, data, _ = setup
    assert data.tau == pytest.approx(0.25)


def test_power_balance(setup):
    _, data, models = setup
    for bess, m in models.items():
        d = extract_dispatch(m, data)
        supply = d.grid_import_kW + d.pv_kW + d.ev_discharge_kW
        demand = d.grid_export_kW + d.apartments_kW + d.common_kW + d.ev_charge_kW
        if bess:
            supply = supply + d.bess_discharge_kW
            demand = demand + d.bess_charge_kW
        assert np.allclose(supply, demand, atol=1e-5)
        assert d.grid_import_kW.max() <= CP + 1e-6
        assert d.grid_export_kW.max() <= 0.5 * CP + 1e-6


def test_ev_soc_and_departure(setup):
    cfg, data, models = setup
    m = models[False]
    for j, apt in enumerate(data.apartments):
        for s in stint_indices(data)[apt]:
            e = s.e_init
            for t in range(s.a, s.d):
                ch, dis = value(m.ev_ch[t, j]), value(m.ev_dis[t, j])
                assert ch * dis == pytest.approx(0, abs=1e-6)          # no simultaneous ch/dis
                e += data.tau * (cfg.EV_eta_ch * ch - dis / cfg.EV_eta_dis)
                assert value(m.ev_e[t, j]) == pytest.approx(e, abs=1e-4)
                assert -1e-6 <= e <= cfg.EV_Smax_kWh + 1e-6
            assert e >= s.e_min - 1e-4                                   # departure target


def test_bess_soc(setup):
    cfg, data, models = setup
    m = models[True]
    e = cfg.BESS_init_kWh
    for t in range(len(data.index)):
        ch, dis = value(m.b_ch[t]), value(m.b_dis[t])
        assert ch * dis == pytest.approx(0, abs=1e-6)
        e += data.tau * (cfg.BESS_eta_ch * ch - dis / cfg.BESS_eta_dis)
        assert value(m.b_e[t]) == pytest.approx(e, abs=1e-4)
    assert e >= cfg.BESS_init_kWh - 1e-4


def test_cost_ordering(setup):
    cfg, data, models = setup
    case1 = solve_case_I(data, cfg)["total_eur"]
    c2, c3 = value(models[False].OBJ), value(models[True].OBJ)
    assert c3 <= c2 * 1.005 + 1e-6      # a BESS can always stay idle (0.5 % MIP gap)
    assert c2 < case1
