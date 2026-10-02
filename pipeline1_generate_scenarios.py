"""
pipeline1_generate_scenarios.py
Generates Monte Carlo scenarios and saves them to a CSV.
"""
import os
import pandas as pd
import numpy as np
from model_grid import CANDIDATE_PV_BUSES
from sampling_monte_carlo import sample_scenario

OUTPUT_DIR = "pipeline_data"
PV_LEVELS_MW = [round(0.125 * i, 3) for i in range(33)] # 0 to 4 MW
N_SCENARIOS = 300
SEED = 42

def generate_dataset():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    records = []
    
    scenario_id = 0
    for level_mw in PV_LEVELS_MW:
        for _ in range(N_SCENARIOS):
            pv_inst, pv_act, load_scale, hour = sample_scenario(rng, level_mw)
            
            row = {
                "scenario_id": scenario_id,
                "pv_target_mw": level_mw,
                "hour": hour
            }
            # Log per-bus loads and PV allocations
            for b in CANDIDATE_PV_BUSES:
                row[f"load_scale_bus{b}"] = load_scale.get(b, 1.0)
                row[f"pv_inst_mw_bus{b}"] = pv_inst.get(b, 0.0)
                row[f"pv_act_mw_bus{b}"] = pv_act.get(b, 0.0)
            
            records.append(row)
            scenario_id += 1
            
    df = pd.DataFrame(records)
    out_file = os.path.join(OUTPUT_DIR, "sampled_scenarios.csv")
    df.to_csv(out_file, index=False)
    print(f"Generated {len(df)} scenarios -> Saved to {out_file}")

if __name__ == "__main__":
    generate_dataset()