"""
model_validation.py
------------------------------

Two independent checks on the base-case (0% PV) power flow from
model_grid.py's realistic MV+LV feeder:

  1. Cross-validation against a second, independently-implemented solver.
     backward_forward_sweep() below is the classical BFS/"ladder"/DistFlow
     method for radial networks -- a completely different numerical
     algorithm from pandapower's Newton-Raphson, sharing no code with it.
     The BFS tree now spans BOTH voltage levels: each of the 14
     distribution transformers is modeled as an ordinary series branch
     (its per-unit impedance derived from vk_percent/vkr_percent, scaled
     from the transformer's own dynamic nameplate base onto the study's 1 MVA
     system base) hanging an LV "child" node off its MV parent node --
     the same BFS math handles MV lines and MV->LV transformer branches
     without any special-casing.

     This simplified BFS does NOT model: the transformer's
     no-load magnetizing branch (the small shunt current implied by
     i0_percent / pfe_kw). That's a real, if small, second-order effect
     that pandapower's transformer model includes and this hand-rolled
     BFS does not, so expect a tiny residual disagreement (still normally
     well under 0.1%) rather than agreement to full solver precision as
     in the original all-MV model -- noted explicitly in the pass/fail
     tolerance below.

  2. Energy conservation: total load + total line losses + total
     transformer losses must equal what the slack bus supplies (basic
     physics, solver-agnostic, and unaffected by the magnetizing-branch
     caveat above since it comes straight from pandapower's own energy
     balance).

Run this file directly.
"""

import warnings
warnings.filterwarnings("ignore", message=".*numba.*")

import cmath
import pandapower as pp

from model_grid import (
    build_realistic_feeder, MV_BRANCHES, CANDIDATE_PV_BUSES, HOUSEHOLDS_PER_NODE,
    ADMD_KVA_PER_HOUSEHOLD, HOUSEHOLD_PF, MV_KV, CONDUCTOR_LIBRARY,
    TRANSFORMER_CAPACITIES_KVA, TRANSFORMER_VK_PERCENT, TRANSFORMER_VKR_PERCENT,
)

S_BASE_MVA = 1.0
Z_BASE_OHM_MV = (MV_KV ** 2) / S_BASE_MVA


def _transformer_pu_impedance(node):
    """
    Per-unit transformer impedance (R, X), converted from the
    transformer's own dynamic kVA nameplate base onto the study's
    S_BASE_MVA system base -- see module docstring.
    """
    z_pct = TRANSFORMER_VK_PERCENT / 100.0
    r_pct = TRANSFORMER_VKR_PERCENT / 100.0
    x_pct = (max(z_pct ** 2 - r_pct ** 2, 0.0)) ** 0.5
    base_ratio = S_BASE_MVA / (TRANSFORMER_CAPACITIES_KVA[node] / 1000.0)
    return complex(r_pct, x_pct) * base_ratio


def backward_forward_sweep(tol=1e-9, max_iter=300):
    """
    Independent BFS power flow solver spanning both MV lines and MV->LV
    transformer branches -- see module docstring. LV nodes are keyed as
    the string f"LV{n}" to keep them distinct from MV node numbers.
    """
    children = {n: [] for n in range(1, 16)}
    for n in CANDIDATE_PV_BUSES:
        children[f"LV{n}"] = []

    branch_z = {}
    parent_of = {}
    for fb, tb, cond_key, length_km in MV_BRANCHES:
        cond = CONDUCTOR_LIBRARY[cond_key]
        children[fb].append(tb)
        branch_z[tb] = complex(cond["r_ohm_per_km"], cond["x_ohm_per_km"]) * length_km / Z_BASE_OHM_MV
        parent_of[tb] = fb

    s_load = {}
    for n in CANDIDATE_PV_BUSES:
        lv_key = f"LV{n}"
        children[n].append(lv_key)
        branch_z[lv_key] = _transformer_pu_impedance(n)
        parent_of[lv_key] = n

        n_households = HOUSEHOLDS_PER_NODE[n]
        s_kva = n_households * ADMD_KVA_PER_HOUSEHOLD
        p_kw = s_kva * HOUSEHOLD_PF
        q_kvar = s_kva * ((1 - HOUSEHOLD_PF ** 2) ** 0.5)
        s_load[lv_key] = complex(p_kw, q_kvar) / (S_BASE_MVA * 1000.0)

    all_nodes = list(children.keys())
    order = []
    stack = [1]
    while stack:
        node = stack.pop()
        order.append(node)
        stack.extend(children[node])

    V = {node: complex(1.0, 0.0) for node in all_nodes}

    for it in range(max_iter):
        branch_current = {node: complex(0, 0) for node in all_nodes if node != 1}
        for node in reversed(order):
            if node == 1:
                continue
            i_own = (s_load.get(node, complex(0, 0)) / V[node]).conjugate()
            branch_current[node] = i_own + sum(branch_current[c] for c in children[node])

        V_new = dict(V)
        max_delta = 0.0
        for node in order:
            if node == 1:
                continue
            p = parent_of[node]
            V_new[node] = V_new[p] - branch_current[node] * branch_z[node]
            max_delta = max(max_delta, abs(V_new[node] - V[node]))
        V = V_new

        if max_delta < tol:
            return V, it + 1

    raise RuntimeError("BFS did not converge")


def run_validation():
    print("=== Check 1: Cross-validation vs. independent BFS solver (MV + transformers) ===\n")
    net, mv_bus_map, lv_bus_map = build_realistic_feeder(load_scale=1.0)
    pp.runpp(net)
    V_bfs, iters = backward_forward_sweep()
    inv_mv_map = {v: k for k, v in mv_bus_map.items()}
    inv_lv_map = {v: k for k, v in lv_bus_map.items()}

    print(f"BFS solver converged in {iters} iterations\n")
    print(f"{'Node':>8} {'pandapower (NR)':>17} {'BFS (indep.)':>15} {'diff':>12}")
    max_diff = 0.0

    for pp_idx in net.res_bus.index:
        if pp_idx in inv_mv_map:
            node = inv_mv_map[pp_idx]
            v_bfs = abs(V_bfs[node])
            label = f"MV{node}"
        else:
            node = inv_lv_map[pp_idx]
            v_bfs = abs(V_bfs[f"LV{node}"])
            label = f"LV{node}"
        v_nr = net.res_bus.vm_pu.at[pp_idx]
        diff = abs(v_nr - v_bfs)
        max_diff = max(max_diff, diff)
        print(f"{label:>8} {v_nr:>17.6f} {v_bfs:>15.6f} {diff:>12.2e}")

    print(f"\nMax voltage difference between the two independent solvers: "
          f"{max_diff:.2e} p.u.")
    # Looser tolerance than the original all-MV model: the BFS neglects
    # each transformer's magnetizing (no-load) branch, a real but small
    # second-order effect pandapower's transformer model does include.
    check1_pass = max_diff < 5e-3
    print("PASS" if check1_pass else "FAIL",
          "-- the two independent algorithms agree within the tolerance "
          "expected once the (deliberately neglected) transformer "
          "magnetizing branch is accounted for"
          if check1_pass else
          "-- disagreement is larger than the magnetizing-branch effect "
          "can explain, investigate before trusting results")

    print("\n=== Check 2: Energy conservation (load + losses = slack supply) ===\n")
    total_load_kw = net.load.p_mw.sum() * 1000
    total_line_loss_kw = net.res_line.pl_mw.sum() * 1000
    total_trafo_loss_kw = net.res_trafo.pl_mw.sum() * 1000
    slack_kw = net.res_ext_grid.p_mw.iloc[0] * 1000
    balance_error = abs(slack_kw - (total_load_kw + total_line_loss_kw + total_trafo_loss_kw))

    print(f"Total load:               {total_load_kw:.2f} kW")
    print(f"Total line losses:        {total_line_loss_kw:.2f} kW")
    print(f"Total transformer losses: {total_trafo_loss_kw:.2f} kW "
          f"(incl. no-load/core losses -- this is exactly the magnetizing "
          f"effect Check 1's BFS neglects)")
    print(f"Slack bus supplies:       {slack_kw:.2f} kW")
    print(f"Balance error:            {balance_error:.6f} kW")
    check2_pass = balance_error < 1e-4
    print("PASS" if check2_pass else "FAIL",
          "-- power balance holds" if check2_pass else "-- energy is not conserved, bug present")

    print(f"\n{'='*60}")
    if check1_pass and check2_pass:
        print("Model validation: PASSED both checks.")
    else:
        print("Model validation: FAILED -- do not trust downstream results "
              "until this is fixed.")
    return check1_pass and check2_pass


if __name__ == "__main__":
    run_validation()