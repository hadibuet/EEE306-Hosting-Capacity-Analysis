"""
pipeline3_plot_results.py
Loads solved power flow CSVs. Generates all comparative plots, percentage heatmaps,
and ranked bar charts for presentation.
"""
import os
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np  # Required for bar chart formatting

from model_grid import CANDIDATE_PV_BUSES
from analysis_hosting_capacity import (
    hosting_capacity_curve, extract_buswise_curves, buswise_hosting_capacities,
    vulnerability_heatmap, mv_vulnerability_heatmap, line_vulnerability_heatmap,
    transformer_vulnerability_heatmap, RISK_THRESHOLD
)

DATA_DIR = "pipeline_data"
PLOT_DIR = "outputs_final"

def plot_both_feeder_hc(base_c, base_hc, mit_c, mit_hc, outpath):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(base_c.pv_total_target_mw, base_c.violation_probability * 100, "-o", color="#c0392b", label=f"Base Case (HC: {base_hc:.3f} MW)")
    ax.plot(mit_c.pv_total_target_mw, mit_c.violation_probability * 100, "-s", color="#2980b9", label=f"BESS Mitigation (HC: {mit_hc:.3f} MW)")
    ax.axhline(RISK_THRESHOLD * 100, color="orange", linestyle="--", label=f"Risk Threshold ({RISK_THRESHOLD*100:.0f}%)")
    ax.set_xlabel("Aggregate Feeder PV (MW)")
    ax.set_ylabel("Violation Probability (%)")
    ax.set_title("Whole-Feeder Hosting Capacity")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

def plot_both_buswise_split(base_curves, base_hcs, mit_curves, mit_hcs, outpath_prefix):
    """Splits the bus-wise plots into two 7-panel figures with shared y-axis labels."""
    buses = list(CANDIDATE_PV_BUSES)
    chunks = [buses[:7], buses[7:]]
    
    for part, chunk in enumerate(chunks, 1):
        n_cols = 4 if len(chunk) > 4 else len(chunk)
        n_rows = math.ceil(len(chunk) / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3.5 * n_rows), sharey=True)
        axes = axes.flatten()

        for i, b in enumerate(chunk):
            ax = axes[i]
            if b in base_curves:
                bc = base_curves[b]
                ax.plot(bc["pv_kw"], bc["violation_probability"] * 100, "-o", color="#c0392b", markersize=4, label=f"Base: {base_hcs[b]:.0f} kW")
            if b in mit_curves:
                mc = mit_curves[b]
                ax.plot(mc["pv_kw"], mc["violation_probability"] * 100, "-s", color="#2980b9", markersize=4, label=f"BESS: {mit_hcs.get(b, 0):.0f} kW")
                
            ax.axhline(RISK_THRESHOLD * 100, color="orange", linestyle="--")
            ax.set_title(f"Bus {b}", fontsize=12)
            ax.legend(fontsize=8)
            ax.grid(True, linestyle=":", alpha=0.6)
            
            # Add y-axis label to leftmost subplots in each row
            if i % n_cols == 0:
                ax.set_ylabel("Violation Probability (%)", fontsize=10)
            
        # Hide empty subplots in the grid
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(f"Bus-Wise Hosting Capacity (Part {part})", fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(f"{outpath_prefix}_part{part}.png", dpi=150)
        plt.close(fig)

def plot_heatmap(series, outpath, title, ylabel, prefix="Node", total_scenarios=1):
    """Converts raw violation counts into probabilities and prevents negative axes."""
    fig, ax = plt.subplots(figsize=(9, 5))
    
    # Calculate percentage
    s_pct = (series.sort_index() / total_scenarios) * 100
    
    colors = plt.cm.YlOrRd(s_pct / max(s_pct.max(), 1e-9))
    bars = ax.bar([f"{prefix} {k}" for k in s_pct.index], s_pct.values, color=colors, edgecolor="black", linewidth=0.8)
    
    # 1. Update the label to explicitly state Probability
    if ylabel.endswith("s"): # Strip trailing 's' if passed (e.g. Violations -> Violation)
        ylabel = ylabel[:-1]
    ax.set_ylabel(f"{ylabel} Probability (%)")
    
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    
    # 2. Force the y-axis to always start at exactly 0
    ax.set_ylim(bottom=0)
    
    # Add numerical labels on top of the bars for clarity
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), 
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
                        
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

def plot_ranked_bar(summary_df, outpath):
    """Generates a ranked horizontal bar chart of hosting capacities with proper right margin padding."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort buses by Base Hosting Capacity
    df_sorted = summary_df.sort_values("Base_HC_kW", ascending=True).copy()
    buses = [f"Bus {int(b)}" for b in df_sorted["Bus"]]
    y_pos = np.arange(len(buses))
    
    # Apply a gradient colormap 
    norm_hc = df_sorted["Base_HC_kW"] / max(df_sorted["Base_HC_kW"].max(), 1)
    colors = plt.cm.RdYlGn(norm_hc)
    
    bars = ax.barh(y_pos, df_sorted["Base_HC_kW"], color=colors, edgecolor="black", alpha=0.9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(buses)
    ax.set_xlabel("Hosting Capacity (kW)")
    ax.set_title("Ranked Bus-Wise Hosting Capacity (Base Case)")
    
    # Dynamically expand the x-axis limit by 15% to give room for text labels
    max_val = df_sorted["Base_HC_kW"].max()
    ax.set_xlim(0, max_val * 1.15)
    
    # Print the exact capacity next to the bar safely within boundaries
    for i, v in enumerate(df_sorted["Base_HC_kW"]):
        ax.text(v + (max_val * 0.02), i, f"{v:.0f} kW", va='center', ha='left', fontsize=9)
        
    "ax.grid(axis='x', linestyle='--', alpha=0.5)"
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    print("Loading solved datasets...")
    df_base = pd.read_csv(os.path.join(DATA_DIR, "raw_results_base.csv"))
    df_mit = pd.read_csv(os.path.join(DATA_DIR, "raw_results_mit.csv"))

    # Compute HC curves
    base_feeder_c, base_f_hc = hosting_capacity_curve(df_base)
    mit_feeder_c, mit_f_hc = hosting_capacity_curve(df_mit)

    base_b_curves = extract_buswise_curves(df_base, n_bins=15)
    base_b_hcs = buswise_hosting_capacities(base_b_curves)
    mit_b_curves = extract_buswise_curves(df_mit, n_bins=15)
    mit_b_hcs = buswise_hosting_capacities(mit_b_curves)

    print("Rendering plots...")
    
    # 1. Whole Feeder Output
    plot_both_feeder_hc(base_feeder_c, base_f_hc, mit_feeder_c, mit_f_hc, os.path.join(PLOT_DIR, "whole_feeder_hc_comparison.png"))
    
    # 2. Split Bus-wise Plots (2 slides instead of 1)
    plot_both_buswise_split(base_b_curves, base_b_hcs, mit_b_curves, mit_b_hcs, os.path.join(PLOT_DIR, "buswise_hc_comparison"))

    # 3. Heatmaps (Converted to Percentages)
    total_scen = len(df_base)
    plot_heatmap(vulnerability_heatmap(df_base), os.path.join(PLOT_DIR, "heatmap_lv_voltage.png"), "LV Bus Overvoltage Frequency", "Violations", "LV", total_scen)
    plot_heatmap(mv_vulnerability_heatmap(df_base), os.path.join(PLOT_DIR, "heatmap_mv_voltage.png"), "MV Bus Overvoltage Frequency", "Violations", "MV", total_scen)
    plot_heatmap(line_vulnerability_heatmap(df_base), os.path.join(PLOT_DIR, "heatmap_line_loading.png"), "Line Thermal Overload Frequency", "Overloads", "Line", total_scen)
    plot_heatmap(transformer_vulnerability_heatmap(df_base), os.path.join(PLOT_DIR, "heatmap_trafo_loading.png"), "Transformer Overload Frequency", "Overloads", "DT", total_scen)

    # Compile Summary Dataframe
    summary_df = pd.DataFrame({
        "Bus": list(base_b_hcs.keys()),
        "Base_HC_kW": list(base_b_hcs.values()),
        "BESS_HC_kW": [mit_b_hcs.get(b, 0.0) for b in base_b_hcs.keys()]
    })
    summary_df["Gain_kW"] = summary_df["BESS_HC_kW"] - summary_df["Base_HC_kW"]
    summary_df.to_csv(os.path.join(PLOT_DIR, "comparative_buswise_summary.csv"), index=False)

    # 4. Ranked Bar Chart Generation
    plot_ranked_bar(summary_df, os.path.join(PLOT_DIR, "ranked_buswise_hc.png"))

    print(f"\nAll plots and metrics generated in ./{PLOT_DIR}/")
    print(f"Feeder Base HC:     {base_f_hc:.3f} MW")
    print(f"Feeder BESS HC:     {mit_f_hc:.3f} MW")

if __name__ == "__main__":
    main()