# EMS for Smart Buildings — Single Connection Point Optimization

Optimization of energy management in a multi-apartment smart building with PV,
electric vehicles (EVs), and a battery energy storage system (BESS), sharing a
single grid connection point (CP).

The models reproduce the case study of **Foroozandeh et al. (2022)**,
*"Single contract power optimization: A novel business model for smart buildings
using intelligent energy management"*, Int. J. Electrical Power & Energy Systems
**135**, 107534.

## Cases

| Case | Name   | Description                                                        |
|------|--------|--------------------------------------------------------------------|
| I    | NSM    | No smart management — baseline net-demand cost, no control          |
| II   | SMSCP  | Smart management with single contracted power (EV scheduling + CP sweep) |
| III  | SMSCPB | SMSCP extended with a shared BESS                                   |

The setup mirrors the reference: 15 apartments, 15-min resolution, bi-hourly
tariff, 3.6 kW PV per apartment, one EV per apartment.

## Repository layout

```
data_schema.py     # Data schema (BuildingConfig, StudyData) + synthetic data generator
ems_models.py      # Pyomo MILP models (objective, EV/BESS/CP constraints)
run_cases.py       # Runner: loads data, builds & solves Cases I/II/III
demo_data/         # Sample 15-min time series (load, PV, prices, EV stints)
```

## Requirements

- Python 3.11+
- A MILP solver: [CBC](https://github.com/coin-or/Cbc) (recommended) or GLPK

Python packages:

```bash
pip install -r requirements.txt
```

Install a solver (via conda-forge is easiest):

```bash
conda install -c conda-forge pyomo coincbc pandas numpy matplotlib
```

## Usage

```bash
# Run all cases on the bundled demo data
python run_cases.py --data ./demo_data --case all

# Or a single case
python run_cases.py --data ./demo_data --case II
```

Case II performs a sweep over the candidate contracted-power values and reports
the cost-optimal CP (the reference paper expects 27.6 kW; with BESS, Case III
expects 20.7 kW).

## Notes

- `data_schema.py` can regenerate synthetic time series if you want a different
  horizon or building configuration.
- Case III is scaffolded in `ems_models.py`; extend the BESS flow constraints to
  reproduce the full SMSCPB result.

## License

MIT — see [LICENSE](LICENSE).

## Reference

S. Foroozandeh et al., *Single contract power optimization: A novel business
model for smart buildings using intelligent energy management*, International
Journal of Electrical Power & Energy Systems 135 (2022) 107534.
