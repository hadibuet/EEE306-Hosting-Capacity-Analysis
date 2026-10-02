"""
RESULT ANALYSIS & DATA LOGGING

Runs on the realistic MV/LV feeder's logged output (phase1c + phase3).
Takes the raw scenario log produced by Phase 3 (full MV bus voltages, LV
bus voltages, line loadings, and transformer loadings for every
scenario, no judgement applied yet) and:
  - flags MV voltage violations (outside the MV band), LV voltage
    violations (outside the separately-justified LV band), thermal
    overloads on any line, and thermal overloads on any distribution
    transformer,
  - computes violation probability at each penetration level -- the
    "hosting capacity curve",
  - identifies the hosting capacity at the risk threshold (10%)
  - builds vulnerability heat maps: by bus (MV and LV), by feeder
    segment (line), and by transformer.

"""

import numpy as np
import pandas as pd
import ast

from model_grid import (
    MV_V_MIN_PU, MV_V_MAX_PU, LV_V_MIN_PU, LV_V_MAX_PU,
    THERMAL_LIMIT_PCT, N_MV_BUSES, CANDIDATE_PV_BUSES,
)

RISK_THRESHOLD = 0.10       # primary risk threshold, per the proposal
ALT_RISK_THRESHOLD = 0.10   # looser tolerance, reported as a sensitivity comparison


def analyze_results(df):
    """
    Add violation columns to a Phase 3 raw results DataFrame.

    Adds:
      - 'mv_voltage_violation'  : bool, any MV bus outside [MV_V_MIN_PU, MV_V_MAX_PU]
      - 'lv_voltage_violation'  : bool, any LV bus outside [LV_V_MIN_PU, LV_V_MAX_PU]
      - 'thermal_violation'     : bool, any line loading exceeds THERMAL_LIMIT_PCT
      - 'trafo_violation'       : bool, any transformer loading exceeds 100%
      - 'violated'              : bool, any of the above (or non-convergence)
      - 'violated_mv_buses'     : list of MV node numbers that breached the MV band
      - 'violated_lv_buses'     : list of LV node numbers that breached the LV band
      - 'violated_trafos'       : list of node numbers whose transformer overloaded

    Returns the same DataFrame with these columns appended.
    """
    df = df.copy()
    v_mv_cols = [c for c in df.columns if c.startswith("V_MV_bus")]
    v_lv_cols = [c for c in df.columns if c.startswith("V_LV_bus")]
    i_cols = [c for c in df.columns if c.startswith("I_line")]
    t_cols = [c for c in df.columns if c.startswith("T_trafo")]

    mv_matrix = df[v_mv_cols].values
    lv_matrix = df[v_lv_cols].values
    i_matrix = df[i_cols].values
    t_matrix = df[t_cols].values

    mv_violation = np.any((mv_matrix > MV_V_MAX_PU) | (mv_matrix < MV_V_MIN_PU), axis=1)
    lv_violation = np.any((lv_matrix > LV_V_MAX_PU) | (lv_matrix < LV_V_MIN_PU), axis=1)
    thermal_violation = np.any(i_matrix > THERMAL_LIMIT_PCT, axis=1)
    trafo_violation = np.any(t_matrix > 100.0, axis=1)
    non_convergence = ~df["converged"].values

    df["mv_voltage_violation"] = mv_violation
    df["lv_voltage_violation"] = lv_violation
    df["thermal_violation"] = thermal_violation
    df["trafo_violation"] = trafo_violation
    df["violated"] = (mv_violation | lv_violation | thermal_violation
                       | trafo_violation | non_convergence)

    def _violated_mv_buses(row):
        return [int(c.replace("V_MV_bus", "")) for c in v_mv_cols
                if pd.notna(row[c]) and (row[c] > MV_V_MAX_PU or row[c] < MV_V_MIN_PU)]

    def _violated_lv_buses(row):
        return [int(c.replace("V_LV_bus", "")) for c in v_lv_cols
                if pd.notna(row[c]) and (row[c] > LV_V_MAX_PU or row[c] < LV_V_MIN_PU)]

    def _violated_trafos(row):
        return [int(c.replace("T_trafo", "")) for c in t_cols
                if pd.notna(row[c]) and row[c] > 100.0]

    df["violated_mv_buses"] = df.apply(_violated_mv_buses, axis=1)
    df["violated_lv_buses"] = df.apply(_violated_lv_buses, axis=1)
    df["violated_trafos"] = df.apply(_violated_trafos, axis=1)
    return df


def hosting_capacity_curve(df, risk_threshold=RISK_THRESHOLD):
    """
    Compute violation probability vs. PV penetration level (the hosting
    capacity curve) and identify the hosting capacity itself.

    Returns
    -------
    curve : DataFrame with columns [pv_total_target_mw, violation_probability]
    hosting_capacity_mw : float -- highest level with violation_probability
                           <= risk_threshold (i.e. the last safe level before
                           the curve crosses the risk line)
    """
    curve = (df.groupby("pv_total_target_mw")["violated"]
               .mean()
               .reset_index()
               .rename(columns={"violated": "violation_probability"})
               .sort_values("pv_total_target_mw"))

    safe_levels = curve[curve.violation_probability <= risk_threshold]
    hosting_capacity_mw = (safe_levels.pv_total_target_mw.iloc[-1]
                            if len(safe_levels) else 0.0)
    return curve, hosting_capacity_mw


def hosting_capacities_multi_threshold(curve, thresholds=(RISK_THRESHOLD, ALT_RISK_THRESHOLD)):
    """
    Given an already-computed hosting capacity curve (from
    hosting_capacity_curve above -- the curve itself doesn't depend on
    the threshold, only which level counts as "the" hosting capacity
    does), return the hosting capacity at EACH of several risk
    thresholds without recomputing anything. Used to report "hosting
    capacity at 5% vs. 10% risk tolerance" side by side as a sensitivity
    comparison.

    Returns
    -------
    dict {threshold: hosting_capacity_mw}
    """
    result = {}
    for th in thresholds:
        safe = curve[curve.violation_probability <= th]
        result[th] = safe.pv_total_target_mw.iloc[-1] if len(safe) else 0.0
    return result


def _safe_parse_list(val):
    if isinstance(val, list):
        return val
    if pd.isna(val) or val == "" or str(val).lower() == "nan":
        return []
    try:
        res = ast.literal_eval(str(val))
        return res if isinstance(res, list) else []
    except Exception:
        return []

def vulnerability_heatmap(df):
    """
    Count, per LV bus, how many scenarios that bus was in voltage violation.
    Safely handles both live lists and stringified lists loaded from CSV.
    """
    counts = {b: 0 for b in CANDIDATE_PV_BUSES}
    for buses_entry in df["violated_lv_buses"]:
        buses = _safe_parse_list(buses_entry)
        for b in buses:
            if int(b) in counts:
                counts[int(b)] += 1
    return pd.Series(counts, name="violation_count").sort_values(ascending=False)


def mv_vulnerability_heatmap(df):
    """Same as vulnerability_heatmap, but for MV backbone buses."""
    counts = {b: 0 for b in range(1, N_MV_BUSES + 1)}
    for buses_entry in df["violated_mv_buses"]:
        buses = _safe_parse_list(buses_entry)
        for b in buses:
            if int(b) in counts:
                counts[int(b)] += 1
    return pd.Series(counts, name="violation_count").sort_values(ascending=False)


def line_vulnerability_heatmap(df):
    """
    Count, per LINE (MV feeder segment), how many scenarios that specific
    line was thermally overloaded (> THERMAL_LIMIT_PCT). Part of the
    proposal's "Vulnerability Mapping" outcome -- "specific feeder
    segments AND buses that require physical upgrades."

    Uses the already-logged I_line* columns from Phase 3, so this is
    pure re-analysis of existing data -- no additional power flow runs.
    """
    i_cols = [c for c in df.columns if c.startswith("I_line")]
    counts = {}
    for c in i_cols:
        line_name = c.replace("I_line", "").replace("_", "\u2192")  # "1_2" -> "1->2"
        counts[line_name] = int((df[c] > THERMAL_LIMIT_PCT).sum())
    return pd.Series(counts, name="thermal_violation_count").sort_values(ascending=False)


def transformer_vulnerability_heatmap(df):
    """
    Count, per distribution transformer, how many scenarios it was
    thermally overloaded (> 100% loading). This is a NEW heat map that
    didn't exist before the realistic-feeder rewiring -- the abstract
    test system had no transformers to overload. In testing, this turned
    out to be a significant finding: transformers can bind before MV
    voltage limits do, since they're sized for evening peak load, not
    simultaneous midday PV export.
    """
    t_cols = [c for c in df.columns if c.startswith("T_trafo")]
    counts = {}
    for c in t_cols:
        node = int(c.replace("T_trafo", ""))
        counts[node] = int((df[c] > 100.0).sum())
    return pd.Series(counts, name="trafo_overload_count").sort_values(ascending=False)


if __name__ == "__main__":
    from legacy_powerflow_voltvar import run_monte_carlo_loop

    print("=== Phase 4 validation: analyze a small Phase 3 run ===\n")
    df_raw = run_monte_carlo_loop([0.0, 0.5, 1.0, 2.0, 2.5], n_scenarios_per_level=40,
                                   mitigation=False, seed=7, verbose=False, batch_size=20)
    df = analyze_results(df_raw)
    curve, hc_mw = hosting_capacity_curve(df)
    hc_multi = hosting_capacities_multi_threshold(curve)
    heat = vulnerability_heatmap(df)
    mv_heat = mv_vulnerability_heatmap(df)
    line_heat = line_vulnerability_heatmap(df)
    trafo_heat = transformer_vulnerability_heatmap(df)

    print("Hosting capacity curve:")
    print(curve.to_string(index=False))
    print(f"\nHosting capacity at each threshold: {hc_multi}")
    print("\nLV bus vulnerability heat map (top 5):")
    print(heat.head(5).to_string())
    print("\nMV bus vulnerability heat map (top 5):")
    print(mv_heat.head(5).to_string())
    print("\nLine/branch vulnerability heat map (top 5):")
    print(line_heat.head(5).to_string())
    print("\nTransformer vulnerability heat map (top 5):")
    print(trafo_heat.head(5).to_string())

def extract_buswise_curves(df, n_bins=15):
    """
    Extracts individual bus hosting capacity curves based ONLY on LOCAL violations 
    (that specific bus's LV voltage or transformer overload, plus non-convergence).
    """
    curves = {}
    # Use the local limits from phase1c
    LV_V_MIN_PU = 0.94
    LV_V_MAX_PU = 1.06
    
    for bus in CANDIDATE_PV_BUSES:
        pv_col = f"PV_bus{bus}_kw"
        lv_col = f"V_LV_bus{bus}"
        trafo_col = f"T_trafo{bus}"
        
        # Filter scenarios where this bus actually received PV
        df_bus = df[df[pv_col] > 0.0].copy()
        
        if df_bus.empty:
            continue
            
        # Define LOCAL violation strictly for this bus
        df_bus['local_violated'] = (
            (df_bus[lv_col] > LV_V_MAX_PU) | 
            (df_bus[lv_col] < LV_V_MIN_PU) | 
            (df_bus[trafo_col] > 100.0) |
            (~df_bus['converged'])
        )
        
        # Create discrete bins (e.g., 0-20kW, 20-40kW) to group the random draws
        max_pv = df_bus[pv_col].max()
        bins = np.linspace(0, max_pv, n_bins)
        df_bus['pv_bin'] = pd.cut(df_bus[pv_col], bins=bins)
        
        # Calculate violation probability per bin using the LOCAL violations
        curve = df_bus.groupby('pv_bin')['local_violated'].mean().reset_index()
        curve['pv_kw'] = curve['pv_bin'].apply(lambda x: x.mid).astype(float)
        curve['violation_probability'] = curve['local_violated'].fillna(0.0)
        
        # Clean up curve for plotting
        curve = curve.dropna(subset=['pv_kw'])
        curves[bus] = curve[['pv_kw', 'violation_probability']]
        
    return curves

def buswise_hosting_capacities(curves, risk_threshold=RISK_THRESHOLD):
    """Finds the MW/kW limit for each bus from the binned curves."""
    hc_results = {}
    for bus, curve in curves.items():
        safe_levels = curve[curve['violation_probability'] <= risk_threshold]
        hc_results[bus] = safe_levels['pv_kw'].iloc[-1] if len(safe_levels) > 0 else 0.0
    return hc_results