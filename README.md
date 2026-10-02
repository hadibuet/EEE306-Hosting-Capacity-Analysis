# Probabilistic Hosting Capacity Analysis of Rooftop Solar PV

A Monte Carlo approach to net-metering interconnection screening on a realistic 11 kV / 0.4 kV distribution feeder, featuring Battery Energy Storage System (BESS) active-power mitigation. 

**Course:** EEE 306 · Power System I Laboratory · BUET

### Team Members
* Abdullah Al Hadi (2206049)
* Md. Mahir Al Islam (2206050)
* AKM Abdullah Al Mison (2206051)
* Pial Mahmud Khan (2206052)
* Fabiha Mahabub (2206053)

---

## 1. File Reference & Architecture

### Shared Library Modules (Core Dependencies)
* **`model_grid.py`**: Builds the realistic 11 kV / 0.4 kV feeder in `pandapower`. Features 15 MV buses, real ACSR conductors (Weasel/Rabbit/Dog), and 14 distribution transformers scaled into three tiers (100, 150, 250 kVA).
* **`sampling_monte_carlo.py`**: The stochastic engine. Draws spatial adoption, localized PV installation sizes, and synchronized temporal demand/irradiance profiles for each scenario.
* **`analysis_hosting_capacity.py`**: Contains the statistical logic to parse raw power-flow outputs, flag violations (voltage limits and thermal limits), and compute the final hosting capacity curves.

### Validation Script
* **`model_validation.py`**: Proves the grid model is mathematically sound by cross-checking the base-case power flow against an independently implemented Backward-Forward Sweep solver and verifying energy conservation. **Run this before trusting downstream results.**

### The Simulation Pipeline (Run in Order)
* **`pipeline1_generate_scenarios.py` (Stage 1)**: Generates 9,900 Monte Carlo scenarios across 33 PV penetration levels and writes them to a CSV. 
* **`pipeline2_powerflow_bess.py` (Stage 2)**: Loads the scenarios and solves the AC power flow across all CPU cores. It evaluates both the unmitigated base case and the actively mitigated case (where localized BESS units charge incrementally to relieve thermal/voltage constraints).
* **`pipeline3_plot_results.py` (Stage 3)**: Reads the solved datasets to generate the final analytical charts, vulnerability heatmaps, and bus-wise summary tables used in the project report.

---

## 2. Dependency Graph

```text
model_grid.py  
   ├── model_validation.py            
   ├── sampling_monte_carlo.py        
   │      └── pipeline1_generate_scenarios.py   [STAGE 1]
   ├── analysis_hosting_capacity.py   
   │      ├── pipeline2_powerflow_bess.py       [STAGE 2]
   │      └── pipeline3_plot_results.py         [STAGE 3]
```

---

## 3. How to Run the Study

**Step 0: Validate the Grid Model**
```bash
python model_validation.py
```

**Step 1: Generate the Monte Carlo Scenarios** (Fast, no power flow)
```bash
python pipeline1_generate_scenarios.py
```
*(Writes `pipeline_data/sampled_scenarios.csv`)*

**Step 2: Solve the Power Flows** (Computationally intensive)
```bash
python pipeline2_powerflow_bess.py
```
*(Writes `raw_results_base.csv` and `raw_results_mit.csv` to the data folder)*

**Step 3: Generate Final Plots and Analytics**
```bash
python pipeline3_plot_results.py
```
*(Outputs all final `.png` charts and summary CSVs into the `outputs_final/` directory)*

---

## 4. Requirements

The simulation requires Python 3 and the following packages:
```bash
pip install pandapower numpy pandas matplotlib
```
*Note: `pipeline2_powerflow_bess.py` utilizes Python's built-in `multiprocessing` module and will automatically scale to use all available CPU cores.*
