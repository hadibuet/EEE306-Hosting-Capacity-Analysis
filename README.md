# Probabilistic Hosting Capacity Analysis of Rooftop Solar PV

A Monte Carlo approach to net-metering interconnection screening on a realistic
11 kV / 0.4 kV distribution feeder, with Battery Energy Storage (BESS)
active-power mitigation.

This README explains what every file does, how they depend on each other, and
the exact commands to run the study end to end.

---

## 1. Files were renamed

All files were renamed from their original `phaseN_...` / `stageN_...` names to
names that describe what each one actually does, and every `import` statement
across the codebase was updated to match. Nothing else was changed — same
logic, same functions, same variable names.

| Current name | Original name |
|---|---|
| `model_grid.py` | `phase1c_realistic_feeder.py` |
| `model_validation.py` | `phase1b_model_validation.py` |
| `sampling_monte_carlo.py` | `phase2_stochastic_sampling.py` |
| `analysis_hosting_capacity.py` | `phase4_result_analysis.py` |
| `pipeline1_generate_scenarios.py` | `stage1_generate_scenarios.py` |
| `pipeline2_powerflow_bess.py` | `stage2_run_powerflow.py` |
| `pipeline3_plot_results.py` | `stage3_plot_and_analyze.py` |
| `legacy_powerflow_voltvar.py` | `phase3_power_flow_loop.py` |
| `legacy_mitigation_study_voltvar.py` | `phase5_mitigation_analysis.py` |

**Why two different prefix families (`pipeline*` vs `legacy_*`)?** The codebase
actually contains two separate power-flow engines built at different times:

- **`pipeline1` → `pipeline2` → `pipeline3`** is the **active, current
  pipeline** — the decoupled, CSV-checkpointed, BESS-mitigated study that
  produces the final results (this is the "three-stage pipeline" and the BESS
  mitigation described in the project presentation).
- **`legacy_powerflow_voltvar.py`** and **`legacy_mitigation_study_voltvar.py`**
  are an **earlier, superseded engine** — a single-pass multiprocessing loop
  using Smart-Inverter Volt-VAR (reactive power) mitigation instead of BESS.
  They're kept in the repo for reference/comparison but are **not** part of
  the current results.

---

## 2. File-by-file reference

### Shared library modules (imported by everything, never run "as the study" themselves)

**`model_grid.py`** — *the feeder itself.*
Builds the realistic 11 kV / 0.4 kV feeder in `pandapower`: 15 MV buses, 14
real ACSR-conductor branches (Weasel/Rabbit/Dog, BS/IS catalog values), 14
pole-mounted Dyn11 distribution transformers in three tiers (100/150/250 kVA),
and one aggregate residential load per LV bus sized from a household count ×
ADMD. Exposes `build_realistic_feeder()`, the topology/constants
(`CANDIDATE_PV_BUSES`, `TRANSFORMER_CAPACITIES_KVA`, voltage/thermal limits,
etc.), and helper functions like `pv_max_kw_for_node()`. Every other file in
this project imports from here.

**`sampling_monte_carlo.py`** — *the Monte Carlo distributions.*
Defines `sample_scenario()`: for a target aggregate PV penetration level, it
draws which buses adopt PV, how big each installation is, what hour of day the
scenario represents, and the matching solar-irradiance and per-bus load
multipliers for that hour. This is the randomness engine both the current and
legacy pipelines are built on.

**`analysis_hosting_capacity.py`** — *turns raw power-flow output into
findings.*
Takes a raw results table (bus voltages, line loadings, transformer loadings
per scenario) and adds violation flags (MV voltage, LV voltage, line thermal,
transformer thermal), computes the hosting-capacity curve (violation
probability vs. PV level) and the hosting capacity itself at a given risk
threshold, builds vulnerability heat maps (by MV bus, LV bus, line, and
transformer), and extracts **per-bus** hosting-capacity curves via binning
(`extract_buswise_curves`). Used by both the current pipeline (stages 2 and 3)
and the legacy engine.

### Standalone validation script

**`model_validation.py`** — *proves the grid model in `model_grid.py` is
correct.*
Runs two independent checks against the base-case (0% PV) power flow: (1) a
hand-rolled Backward-Forward Sweep solver — a completely different numerical
method from pandapower's Newton-Raphson — cross-checked bus-by-bus, and (2) an
energy-conservation check (load + losses = slack supply). Run this on its own,
before trusting any downstream result: `python model_validation.py`.

### Active pipeline (run these three, in order)

**`pipeline1_generate_scenarios.py`** — **Stage 1.**
Pure, fast, single-threaded scenario generation — no power flow here. Sweeps
33 PV levels from 0 to 4 MW, draws 300 Monte Carlo scenarios per level (9,900
scenarios total) using `sampling_monte_carlo.py`, and writes everything to
`pipeline_data/sampled_scenarios.csv`.

**`pipeline2_powerflow_bess.py`** — **Stage 2.**
Loads the CSV from Stage 1, solves every scenario's power flow across all CPU
cores (`multiprocessing.Pool`), once with no mitigation and once with BESS
active-power mitigation. BESS is modeled as a controllable load at every LV
bus, sized proportionally to that bus's transformer tier (40/60/100 kW for
100/150/250 kVA transformers), and charges in 10 kW steps whenever transformer
loading exceeds 98% or LV voltage exceeds 1.05 p.u. Writes
`pipeline_data/raw_results_base.csv` and `pipeline_data/raw_results_mit.csv`.

**`pipeline3_plot_results.py`** — **Stage 3.**
Loads both solved CSVs from Stage 2 and renders every chart used in the
report/presentation into `outputs_final/`: the whole-feeder hosting-capacity
comparison, the 14 bus-wise curves (split into two figures), the MV/LV/line/
transformer violation heat maps, and the ranked bus-wise bar chart. Also
writes `outputs_final/comparative_buswise_summary.csv`. Because Stage 2's
results are checkpointed to CSV, you can re-run this stage as many times as
you like — to restyle a chart, say — without ever re-solving a single power
flow.

### Legacy / superseded pipeline (kept for reference only)

**`legacy_powerflow_voltvar.py`** — the original single-pass multiprocessing
power-flow engine. Supports an optional Smart-Inverter Volt-VAR droop control
(reactive power only) instead of BESS. Superseded by
`pipeline1_generate_scenarios.py` + `pipeline2_powerflow_bess.py` once the
project moved to a decoupled, CSV-checkpointed, BESS-based approach.

**`legacy_mitigation_study_voltvar.py`** — runs the legacy engine twice (base
case vs. Volt-VAR mitigated) and compares hosting capacity at the 5% and 10%
risk thresholds. Superseded by `pipeline3_plot_results.py`.

> ⚠️ **Known pre-existing issue (not introduced by this renaming):**
> `legacy_mitigation_study_voltvar.py` imports `EARLY_TERM_BATCH_SIZE`,
> `EARLY_TERM_LOW_THRESHOLD`, and `EARLY_TERM_HIGH_THRESHOLD` from
> `legacy_powerflow_voltvar.py`, but that file doesn't actually define them (it
> also doesn't accept the `batch_size=`/`early_termination=` keyword arguments
> that `legacy_mitigation_study_voltvar.py` passes in). This mismatch already
> existed in the original `phase3_power_flow_loop.py` /
> `phase5_mitigation_analysis.py` pair before the rename — it means the legacy
> engine's early-termination path is incomplete/out of sync. Since this file
> is superseded and not part of the active pipeline, it was left exactly as
> found rather than patched. If you need to run it, add those three constants
> and the matching parameters to `legacy_powerflow_voltvar.py` first.

---

## 3. Dependency graph

```
model_grid.py  (feeder definition — no internal dependencies)
   ├── model_validation.py            (validates model_grid.py)
   ├── sampling_monte_carlo.py        (Monte Carlo distributions)
   │      └── pipeline1_generate_scenarios.py   [STAGE 1]
   ├── analysis_hosting_capacity.py   (violation logic, HC curves, heat maps)
   │      ├── pipeline2_powerflow_bess.py       [STAGE 2]
   │      └── pipeline3_plot_results.py         [STAGE 3]
   │
   └── (legacy engine, superseded)
          legacy_powerflow_voltvar.py
                 └── legacy_mitigation_study_voltvar.py
```

---

## 4. How to run the current study, step by step

```bash
# 0. (optional but recommended) sanity-check the grid model itself
python model_validation.py

# 1. generate the Monte Carlo scenario set (fast, single-threaded, no power flow)
python pipeline1_generate_scenarios.py
#    -> writes pipeline_data/sampled_scenarios.csv

# 2. solve every scenario's power flow, base case and BESS-mitigated
python pipeline2_powerflow_bess.py
#    -> writes pipeline_data/raw_results_base.csv
#    -> writes pipeline_data/raw_results_mit.csv

# 3. build every chart and summary table used in the report
python pipeline3_plot_results.py
#    -> writes outputs_final/*.png and outputs_final/comparative_buswise_summary.csv
```

Steps 1 and 2 only need to be re-run if you change the sampling assumptions,
the feeder model, or the BESS control logic. Step 3 can be re-run on its own
any time you just want different-looking charts from the same solved data.

---

## 5. Requirements

```bash
pip install pandapower numpy pandas matplotlib
```

`pipeline2_powerflow_bess.py` and `legacy_powerflow_voltvar.py` use Python's
built-in `multiprocessing` module and will use every CPU core available on the
machine they run on.

---

## 6. Output directories

| Directory | Created by | Contents |
|---|---|---|
| `pipeline_data/` | Stage 1 & 2 | `sampled_scenarios.csv`, `raw_results_base.csv`, `raw_results_mit.csv` |
| `outputs_final/` | Stage 3 | All final `.png` charts + `comparative_buswise_summary.csv` |

Both directories are created automatically (`os.makedirs(..., exist_ok=True)`)
the first time their respective stage runs.
