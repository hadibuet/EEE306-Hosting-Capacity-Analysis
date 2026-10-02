"""
pipeline2_powerflow_bess.py
Loads generated scenarios from CSV, solves power flows via multiprocessing,
and writes raw simulation results to CSV. Includes Proportional BESS Active-Power Mitigation.
"""
import os
import time
import multiprocessing as mp
import numpy as np
import pandas as pd
import pandapower as pp

from model_grid import build_realistic_feeder, N_MV_BUSES, MV_BRANCHES, CANDIDATE_PV_BUSES
from analysis_hosting_capacity import analyze_results

INPUT_FILE = os.path.join("pipeline_data", "sampled_scenarios.csv")
OUTPUT_DIR = "pipeline_data"

# Proportional BESS Sizing (40% of Local Transformer Rating)
BESS_KW_SIZES = {
    # 40 kW BESS for 100 kVA transformers (Tail-end spurs)
    5: 40.0, 9: 40.0, 10: 40.0, 12: 40.0, 14: 40.0, 15: 40.0,
    
    # 60 kW BESS for 150 kVA transformers (Standard laterals)
    2: 60.0, 4: 60.0, 7: 60.0, 8: 60.0, 13: 60.0,
    
    # 100 kW BESS for 250 kVA transformers (Main junctions)
    3: 100.0, 6: 100.0, 11: 100.0
}

def solve_scenario_row(row_dict, mitigation=False, max_iter=12):
    per_bus_load = {b: row_dict[f"load_scale_bus{b}"] for b in CANDIDATE_PV_BUSES}
    pv_act = {b: row_dict[f"pv_act_mw_bus{b}"] for b in CANDIDATE_PV_BUSES if row_dict[f"pv_act_mw_bus{b}"] > 0}
    pv_inst = {b: row_dict[f"pv_inst_mw_bus{b}"] for b in CANDIDATE_PV_BUSES if row_dict[f"pv_inst_mw_bus{b}"] > 0}
    
    net, mvb, lvb = build_realistic_feeder(load_scale=per_bus_load)
    
    sgen_idx = {}
    for bus, mw in pv_act.items():
        sgen_idx[bus] = pp.create_sgen(net, bus=lvb[bus], p_mw=mw, q_mvar=0.0, name=f"PV_{bus}")
    
    bess_p_mw = {bus: 0.0 for bus in CANDIDATE_PV_BUSES}
    bess_idx = {}
    
    if mitigation:
        for bus in CANDIDATE_PV_BUSES:
            bess_idx[bus] = pp.create_load(net, bus=lvb[bus], p_mw=0.0, q_mvar=0.0, name=f"BESS_{bus}")

    converged = True
    try:
        pp.runpp(net)
    except Exception:
        converged = False

    # UPDATED BESS LOOP: Triggers aggressively on BOTH Thermal and Voltage
    if converged and mitigation and pv_act:
        for _ in range(max_iter):
            changed = False
            
            for i, bus in enumerate(CANDIDATE_PV_BUSES):
                loading = net.res_trafo.loading_percent.iloc[i]
                v_lv = net.res_bus.vm_pu.at[lvb[bus]]
                
                # Trigger BESS if transformer > 98% OR voltage > 1.05 p.u.
                if loading > 98.0 or v_lv > 1.05:
                    current_p = net.load.at[bess_idx[bus], "p_mw"]
                    
                    # Fetch proportional max capacity for this specific bus
                    max_mw = BESS_KW_SIZES[bus] / 1000.0
                    
                    # Charge in smaller 10 kW steps to prevent overshooting small transformers
                    new_p = min(max_mw, current_p + 0.01) 
                    
                    if abs(new_p - current_p) > 1e-4:
                        net.load.at[bess_idx[bus], "p_mw"] = new_p
                        bess_p_mw[bus] = new_p
                        changed = True
                        
            if not changed:
                break
                
            try:
                pp.runpp(net)
            except Exception:
                converged = False
                break

    if converged:
        v_mv = {n: float(net.res_bus.vm_pu.at[mvb[n]]) for n in range(1, N_MV_BUSES + 1)}
        v_lv = {n: float(net.res_bus.vm_pu.at[lvb[n]]) for n in CANDIDATE_PV_BUSES}
        line_load = {f"{fb}_{tb}": float(net.res_line.loading_percent.iloc[i]) for i, (fb, tb, _, _) in enumerate(MV_BRANCHES)}
        trafo_load = {n: float(net.res_trafo.loading_percent.iloc[i]) for i, n in enumerate(CANDIDATE_PV_BUSES)}
    else:
        v_mv = {n: np.nan for n in range(1, N_MV_BUSES + 1)}
        v_lv = {n: np.nan for n in CANDIDATE_PV_BUSES}
        line_load = {f"{fb}_{tb}": np.nan for fb, tb, _, _ in MV_BRANCHES}
        trafo_load = {n: np.nan for n in CANDIDATE_PV_BUSES}

    res = {
        "pv_total_target_mw": row_dict["pv_target_mw"],
        "pv_total_installed_mw": sum(pv_inst.values()),
        "hour": row_dict["hour"],
        "converged": converged,
    }
    for n in CANDIDATE_PV_BUSES:
        res[f"PV_bus{n}_kw"] = row_dict[f"pv_inst_mw_bus{n}"] * 1000.0
        res[f"BESS_bus{n}_kw"] = bess_p_mw.get(n, 0.0) * 1000.0
        
    res.update({f"V_MV_bus{n}": v_mv[n] for n in range(1, N_MV_BUSES + 1)})
    res.update({f"V_LV_bus{n}": v_lv[n] for n in CANDIDATE_PV_BUSES})
    res.update({f"I_line{k}": v for k, v in line_load.items()})
    res.update({f"T_trafo{n}": trafo_load[n] for n in CANDIDATE_PV_BUSES})
    return res

def _pool_worker(args):
    row_dict, mitigation = args
    return solve_scenario_row(row_dict, mitigation)

def run_simulation(mitigation=False):
    df_scenarios = pd.read_csv(INPUT_FILE)
    tag = "BESS-Mitigated" if mitigation else "Base"
    print(f"\n=== {tag} Simulation ===")
    
    n_cores = mp.cpu_count()
    t0 = time.time()
    all_results = []
    
    grouped = df_scenarios.groupby("pv_target_mw")
    
    with mp.Pool(processes=n_cores) as pool:
        for level_mw, group_df in grouped:
            records = group_df.to_dict(orient="records")
            tasks = [(r, mitigation) for r in records]
            
            level_results = pool.map(_pool_worker, tasks)
            all_results.extend(level_results)
            
            df_level = analyze_results(pd.DataFrame(level_results))
            p = df_level["violated"].mean()
            
            # Adjusted string to display standard 5% risk threshold
            if p <= 0.10:
                status = f"Safe -- Within 5% Risk Limit (cumulative p={p:.3f} <= 0.10)"
            else:
                status = f"Violation Limit Exceeded (cumulative p={p:.3f} > 0.10)"
                
            elapsed = time.time() - t0
            print(f"[{tag[:4]}]  PV target {level_mw:.3f} MW -- {len(tasks)}/{len(tasks)} scenarios "
                  f"({elapsed:.1f}s elapsed) -- {status}")
                  
    df_out = analyze_results(pd.DataFrame(all_results))
    out_name = f"raw_results_{'mit' if mitigation else 'base'}.csv"
    save_path = os.path.join(OUTPUT_DIR, out_name)
    df_out.to_csv(save_path, index=False)
    print(f"Completed {tag} in {time.time() - t0:.1f}s -> Saved to {save_path}")

if __name__ == "__main__":
    run_simulation(mitigation=False)
    run_simulation(mitigation=True)