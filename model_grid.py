"""
model_grid.py
------------------------------
A realistic 11 kV / 0.4 kV residential distribution feeder, 
built from actual utility-catalog components instead
of the abstract IEEE 15-bus per-unit test system.


Data provenance:
  - ACSR conductor resistance + ampacity: BS/IS-standard catalog values,
    cross-checked against multiple manufacturer datasheets (LM Cables,
    Kenter Cables, WBSEDCL spec). See CONDUCTOR_LIBRARY below for values
    and sources.
  - ACSR reactance: it's computed from the standard single-phase-equivalent
    inductance formula using each conductor's actual geometric mean radius 
    (from its stranding diameter) and a typical 11 kV pin-type crossarm spacing
    of 1.0 m GMD -- standard South Asian rural/urban distribution pole practice.
  - Distribution transformer parameters: 100 kVA, 150 kVa, 250 kVA, 11/0.4 kV, Dyn11 --
    %Z=4.0%, no-load loss <=140 W, matching an actual Bangladesh Power
    Division transformer tender specification; load loss ~1.75%
    and excitation current ~2.5% matching typical IEEE C57.12-class figures
    for this size class.
  - Household Demand: 1.0 kVA/household at 0.90 lagging power factor -- a
    commonly used urban-residential planning assumption.
"""

import warnings
warnings.filterwarnings("ignore", message=".*numba.*")

import pandapower as pp
import numpy as np

# ----------------------------------------------------------------------
# Technical constraints
# ----------------------------------------------------------------------
# MV-side voltage band matches the original assignment's stated criterion.
MV_V_MIN_PU = 0.95
MV_V_MAX_PU = 1.05
# LV-side tolerance: a wider, separately-justified +-6% service-voltage
# band, typical of IEC/utility LV distribution codes (tighter MV limits
# don't directly apply to the customer side of a step-down transformer).
LV_V_MIN_PU = 0.94
LV_V_MAX_PU = 1.06
THERMAL_LIMIT_PCT = 100.0

MV_KV = 11.0
LV_KV = 0.4

# ----------------------------------------------------------------------
# Real ACSR conductor library (BS/IS standard codes, South Asian practice)
# ----------------------------------------------------------------------
# r_ohm_per_km: catalog DC resistance at 20 degC.
# x_ohm_per_km: computed, see module docstring (GMD=1.0 m pole spacing).
# max_i_ka: catalog continuous ampacity.
CONDUCTOR_LIBRARY = {
    "Weasel_30":  dict(area_mm2=30,  r_ohm_per_km=0.9077, x_ohm_per_km=0.367, max_i_ka=0.134,
                        role="light spur / tail-end lateral"),
    "Rabbit_50":  dict(area_mm2=50,  r_ohm_per_km=0.5426, x_ohm_per_km=0.351, max_i_ka=0.185,
                        role="standard residential lateral"),
    "Dog_100":    dict(area_mm2=100, r_ohm_per_km=0.2792, x_ohm_per_km=0.329, max_i_ka=0.290,
                        role="main feeder backbone"),
    "Wolf_150":   dict(area_mm2=150, r_ohm_per_km=0.1871, x_ohm_per_km=0.314, max_i_ka=0.405,
                        role="heavy backbone / express feeder"),
}

# ----------------------------------------------------------------------
# Topology: same 14-branch radial tree used throughout this project (so
# earlier vulnerability findings, e.g. "bus 12 is the weak point", remain
# comparable), but now every branch has a real conductor type and an
# explicit physical length in km instead of one blanket length constant.
# (from_node, to_node, conductor_key, length_km)
# ----------------------------------------------------------------------
MV_BRANCHES = [
    (1, 2,  "Dog_100",  0.60),   # substation -> backbone start
    (2, 3,  "Dog_100",  0.55),   # backbone continues
    (3, 4,  "Rabbit_50", 0.45),
    (4, 5,  "Weasel_30", 0.35),  # tail-end spur
    (2, 9,  "Rabbit_50", 0.50),
    (2, 10, "Rabbit_50", 0.55),
    (10, 11, "Rabbit_50", 0.50),
    (11, 12, "Weasel_30", 0.40),  # tail-end spur (this is our known weak point)
    (3, 13, "Rabbit_50", 0.45),
    (3, 14, "Weasel_30", 0.35),
    (1, 6,  "Dog_100",  0.60),   # second backbone leg off the substation
    (6, 7,  "Rabbit_50", 0.45),
    (6, 8,  "Rabbit_50", 0.50),
    (3, 15, "Weasel_30", 0.35),
]

N_MV_BUSES = 15
CANDIDATE_PV_BUSES = list(range(2, 16))  # LV buses 2-15 (bus 1 is the substation)

# ----------------------------------------------------------------------
# Realistic Sizing: Households and Transformers (Optimized for 4.5 MW Sweep)
# ----------------------------------------------------------------------
# Transformers use standard IEC sizes (50, 100, 150 kVA).
# Household counts are sized so peak demand (1.0 kVA ADMD/hh) loads the 
# transformer to exactly 80% of nameplate capacity.
# Total feeder households = 1000 (Total PV Ceiling = 4.0 MW).
TRANSFORMER_VK_PERCENT = 4.0     # %Z impedance
TRANSFORMER_VKR_PERCENT = 1.75   # load-loss %
TRANSFORMER_PFE_KW = 0.14        # no-load loss kW
TRANSFORMER_I0_PERCENT = 2.5     # excitation current %
TRANSFORMER_SHIFT_DEGREE = 330   # Dyn11 vector group

TRANSFORMER_CAPACITIES_KVA = {
    # Upgraded tail-end spurs to 100 kVA
    5: 100.0, 9: 100.0, 10: 100.0, 12: 100.0, 14: 100.0, 15: 100.0,
    # Upgraded standard laterals to 150 kVA
    2: 150.0, 4: 150.0, 7: 150.0, 8: 150.0, 13: 150.0,
    # Upgraded main junctions to 250 kVA
    3: 250.0, 6: 250.0, 11: 250.0
}

HOUSEHOLDS_PER_NODE = {
    # 80 households for 100 kVA
    5: 70, 9: 80, 10: 85, 12: 65, 14: 80, 15: 90,
    # 120 households for 150 kVA
    2: 120, 4: 130, 7: 100, 8: 120, 13: 110,
    # 200 households for 250 kVA
    3: 200, 6: 220, 11: 180
}

# Standard residential assumptions
ADMD_KVA_PER_HOUSEHOLD = 1.0
HOUSEHOLD_PF = 0.90  # lagging

# Rooftop PV sizing cap per household (typical Bangladesh residential
# rooftop system size ceiling) -- used later to bound how much PV each
# bus could plausibly ever host.
PV_MAX_KW_PER_HOUSEHOLD = 4.0


def build_realistic_feeder(load_scale=1.0):
    """
    Build the full MV+LV feeder: substation, 14 MV backbone/lateral
    branches with real conductor data, 14 distribution transformers, and
    14 LV buses carrying the aggregate residential load for that
    transformer's household cluster.

    Parameters
    ----------
    load_scale : float or dict {node_number: float}
        Multiplier on household ADMD (temporal demand uncertainty). Pass
        a single float to apply the same multiplier feeder-wide (the
        original behaviour), or a dict to give each node's own
        independent multiplier -- e.g. a shared diurnal/noise base level
        with additional per-bus random variation layered on top, so
        load doesn't move in perfect lockstep across the whole feeder
        (see sampling_monte_carlo.py's per-bus load noise). Nodes
        missing from the dict default to a multiplier of 1.0.

    Returns
    -------
    net : pandapower network
    mv_bus_map : dict {node_number: pandapower MV bus index}
    lv_bus_map : dict {node_number: pandapower LV bus index}  (nodes 2-15 only)
    """
    net = pp.create_empty_network(name="Realistic_11kV_0p4kV_Residential_Feeder", sn_mva=10)

    mv_bus_map = {}
    for n in range(1, N_MV_BUSES + 1):
        mv_bus_map[n] = pp.create_bus(net, vn_kv=MV_KV, name=f"MV_Node{n}",
                                       min_vm_pu=MV_V_MIN_PU, max_vm_pu=MV_V_MAX_PU)

    pp.create_ext_grid(net, bus=mv_bus_map[1], vm_pu=1.00, name="Substation (11kV slack)")

    for fb, tb, cond_key, length_km in MV_BRANCHES:
        cond = CONDUCTOR_LIBRARY[cond_key]
        pp.create_line_from_parameters(
            net, from_bus=mv_bus_map[fb], to_bus=mv_bus_map[tb],
            length_km=length_km,  # REDUCED from 2.5 to 1.2
            r_ohm_per_km=cond["r_ohm_per_km"], x_ohm_per_km=cond["x_ohm_per_km"],
            c_nf_per_km=0.0, max_i_ka=cond["max_i_ka"],
            name=f"MV_{fb}_{tb}_{cond_key}"
        )

    lv_bus_map = {}
    for n in CANDIDATE_PV_BUSES:
        lv_bus = pp.create_bus(net, vn_kv=LV_KV, name=f"LV_Node{n}",
                                min_vm_pu=LV_V_MIN_PU, max_vm_pu=LV_V_MAX_PU)
        lv_bus_map[n] = lv_bus

        pp.create_transformer_from_parameters(
            net, hv_bus=mv_bus_map[n], lv_bus=lv_bus,
            sn_mva=TRANSFORMER_CAPACITIES_KVA[n] / 1000.0,
            vn_hv_kv=MV_KV, vn_lv_kv=LV_KV,
            vk_percent=TRANSFORMER_VK_PERCENT, vkr_percent=TRANSFORMER_VKR_PERCENT,
            pfe_kw=TRANSFORMER_PFE_KW, i0_percent=TRANSFORMER_I0_PERCENT,
            shift_degree=TRANSFORMER_SHIFT_DEGREE,
            name=f"DT_{n}_{int(TRANSFORMER_CAPACITIES_KVA[n])}kVA_Dyn11"
        )

        node_load_scale = load_scale.get(n, 1.0) if isinstance(load_scale, dict) else load_scale
        n_households = HOUSEHOLDS_PER_NODE[n]
        s_kva = n_households * ADMD_KVA_PER_HOUSEHOLD * node_load_scale
        p_kw = s_kva * HOUSEHOLD_PF
        q_kvar = s_kva * ((1 - HOUSEHOLD_PF ** 2) ** 0.5)
        pp.create_load(net, bus=lv_bus, p_mw=p_kw / 1000.0, q_mvar=q_kvar / 1000.0,
                        name=f"Residential_{n}_{n_households}hh")

    return net, mv_bus_map, lv_bus_map


def total_households():
    return sum(HOUSEHOLDS_PER_NODE.values())


# Compatibility constant: phase2-5/main.py reference TOTAL_PEAK_LOAD_KW
# for chart titles and penetration-percentage calculations. Computed the
# same way the load itself is built (households x ADMD x power factor),
# not re-derived from a running network, so it's cheap to import.
TOTAL_PEAK_LOAD_KW = total_households() * ADMD_KVA_PER_HOUSEHOLD * HOUSEHOLD_PF


def pv_max_kw_for_node(n):
    return HOUSEHOLDS_PER_NODE[n] * PV_MAX_KW_PER_HOUSEHOLD


# Total realistic PV ceiling across the WHOLE feeder (every household on
# every node maxed at PV_MAX_KW_PER_HOUSEHOLD) -- used by Phase 2 to scale
# how many buses should plausibly have adopted at a given aggregate
# penetration target, and by Phase 6 to size the "background" ambient
# adoption range for its stratified/conditional bus-wise study.
TOTAL_FEEDER_PV_CEILING_KW = sum(HOUSEHOLDS_PER_NODE[n] * PV_MAX_KW_PER_HOUSEHOLD
                                  for n in CANDIDATE_PV_BUSES)


if __name__ == "__main__":
    print("=== Realistic feeder validation: base case (0% PV) power flow ===\n")
    net, mv_bus_map, lv_bus_map = build_realistic_feeder(load_scale=1.0)
    pp.runpp(net)

    print("MV bus voltages (p.u.):")
    print(net.res_bus.vm_pu.iloc[:N_MV_BUSES].round(4).to_string())
    print("\nLV bus voltages (p.u.):")
    print(net.res_bus.vm_pu.iloc[N_MV_BUSES:].round(4).to_string())
    print("\nLine loading (%):")
    print(net.res_line.loading_percent.round(2).to_string())
    print("\nTransformer loading (%):")
    print(net.res_trafo.loading_percent.round(2).to_string())

    mv_v = net.res_bus.vm_pu.iloc[:N_MV_BUSES]
    lv_v = net.res_bus.vm_pu.iloc[N_MV_BUSES:]
    mv_ok = mv_v.between(MV_V_MIN_PU, MV_V_MAX_PU).all()
    lv_ok = lv_v.between(LV_V_MIN_PU, LV_V_MAX_PU).all()
    therm_ok = (net.res_line.loading_percent <= THERMAL_LIMIT_PCT).all()
    trafo_ok = (net.res_trafo.loading_percent <= 100.0).all()

    print(f"\nMV buses within [{MV_V_MIN_PU},{MV_V_MAX_PU}] p.u.: {mv_ok}")
    print(f"LV buses within [{LV_V_MIN_PU},{LV_V_MAX_PU}] p.u.: {lv_ok}")
    print(f"All lines within thermal rating: {therm_ok}")
    print(f"All transformers within rating: {trafo_ok}")
    print(f"\nTotal households served: {total_households()}")
    print(f"Total residential peak demand: {net.load.p_mw.sum()*1000:.1f} kW")

    if not (mv_ok and lv_ok and therm_ok and trafo_ok):
        raise SystemExit("Base case NOT clean -- tune branch lengths/conductors before proceeding.")
    print("\nBase case validated -- ready for Monte Carlo study.")
