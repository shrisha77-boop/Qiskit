"""
frontend/services/dispatch_service.py
Bridge between the dashboard and the backend solver modules.
Normalizes all solver outputs into a consistent data contract.
"""

import sys
import os
import time
import traceback
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

# Ensure project root is on the path (run from project root)
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data.instance import generate_instance, SEVERITY_WEIGHT_MAP, DispatchInstance
from model.qubo import build_qubo, evaluate_qubo_energy
from model.ising import qubo_to_ising, decode_qiskit_bitstring
from solvers.exact import solve_hungarian, solve_brute_force_qubo, is_bitstring_feasible
from solvers.greedy import solve_nearest_vehicle, solve_severity_first_greedy
from solvers.sa import solve_simulated_annealing
from solvers.qaoa import run_qaoa


# ─────────────────────────────────────────────────────────────
# Scenario builder
# ─────────────────────────────────────────────────────────────

def build_scenario(n_vehicles: int, n_incidents: int, seed: int) -> Dict[str, Any]:
    """
    Generates a dispatch scenario and returns a rich dictionary
    suitable for display in the dashboard.
    """
    inst: DispatchInstance = generate_instance(n_vehicles, n_incidents, seed=seed)

    ambulances = {}
    for v in range(n_vehicles):
        pos = inst.vehicle_positions[v]
        ambulances[f"A{v+1}"] = (float(pos[0]), float(pos[1]))

    incidents = []
    for i in range(n_incidents):
        pos = inst.incident_positions[i]
        sev = int(inst.severities[i])
        incidents.append({
            "id": f"I{i+1}",
            "idx": i,
            "position": (float(pos[0]), float(pos[1])),
            "severity": sev,
            "weight": int(SEVERITY_WEIGHT_MAP[sev]),
        })

    travel_times = inst.travel_time_matrix.tolist()
    cost_matrix = inst.cost_matrix.tolist()

    return {
        "instance": inst,
        "n_vehicles": n_vehicles,
        "n_incidents": n_incidents,
        "seed": seed,
        "ambulances": ambulances,          # {"A1": (x,y), ...}
        "incidents": incidents,             # list of dicts
        "travel_times": travel_times,       # list[list[float]]
        "cost_matrix": cost_matrix,         # list[list[float]]
        "severity_weight_map": SEVERITY_WEIGHT_MAP,
    }


# ─────────────────────────────────────────────────────────────
# Result normalizer
# ─────────────────────────────────────────────────────────────

def _normalize_result(
    solver_name: str,
    raw_assignments: List[Tuple[int, int]],   # (vehicle_idx, incident_idx)
    scenario: Dict[str, Any],
    feasible: bool,
    runtime: float,
    metadata: Optional[Dict] = None,
    optimal_cost: Optional[float] = None,
) -> Dict[str, Any]:
    """Builds the standard solver_result dict."""
    inst: DispatchInstance = scenario["instance"]
    n_v = scenario["n_vehicles"]
    n_i = scenario["n_incidents"]
    ambulance_ids = list(scenario["ambulances"].keys())
    incident_ids = [inc["id"] for inc in scenario["incidents"]]

    # Convert raw (v_idx, i_idx) to named assignments
    named_assignments = []
    total_weighted_cost = 0.0
    for v_idx, i_idx in raw_assignments:
        if v_idx < n_v and i_idx < n_i:
            a_id = ambulance_ids[v_idx]
            inc = scenario["incidents"][i_idx]
            t = float(inst.travel_time_matrix[v_idx, i_idx])
            c = float(inst.cost_matrix[v_idx, i_idx])
            total_weighted_cost += c
            named_assignments.append({
                "ambulance_id": a_id,
                "incident_id": inc["id"],
                "severity": inc["severity"],
                "weight": inc["weight"],
                "travel_time": round(t, 2),
                "weighted_cost": round(c, 2),
            })

    approx_ratio = None
    if optimal_cost and optimal_cost > 0 and feasible:
        approx_ratio = round(total_weighted_cost / optimal_cost, 4)

    return {
        "solver_name": solver_name,
        "assignments": named_assignments,
        "weighted_cost": round(total_weighted_cost, 4),
        "feasible": feasible,
        "runtime_seconds": round(runtime, 4),
        "approximation_ratio": approx_ratio,
        "metadata": metadata or {},
    }


# ─────────────────────────────────────────────────────────────
# Individual solvers
# ─────────────────────────────────────────────────────────────

def run_hungarian(scenario: Dict[str, Any]) -> Dict[str, Any]:
    inst: DispatchInstance = scenario["instance"]
    t0 = time.perf_counter()
    res = solve_hungarian(inst.cost_matrix)
    elapsed = time.perf_counter() - t0
    return _normalize_result(
        "Hungarian (Exact)", res["assignments"], scenario,
        feasible=True, runtime=elapsed,
        metadata={"algorithm": "scipy.optimize.linear_sum_assignment", "complexity": "O(N³)"},
    )


def run_nearest_vehicle(scenario: Dict[str, Any]) -> Dict[str, Any]:
    inst: DispatchInstance = scenario["instance"]
    t0 = time.perf_counter()
    res = solve_nearest_vehicle(inst.travel_time_matrix, inst.weights)
    elapsed = time.perf_counter() - t0
    return _normalize_result(
        "Nearest-Vehicle Greedy", res["assignments"], scenario,
        feasible=True, runtime=elapsed,
        metadata={"algorithm": "Greedy (nearest free ambulance per call)"},
    )


def run_severity_first(scenario: Dict[str, Any]) -> Dict[str, Any]:
    inst: DispatchInstance = scenario["instance"]
    t0 = time.perf_counter()
    res = solve_severity_first_greedy(inst.travel_time_matrix, inst.severities, inst.weights)
    elapsed = time.perf_counter() - t0
    return _normalize_result(
        "Severity-First Greedy", res["assignments"], scenario,
        feasible=True, runtime=elapsed,
        metadata={"algorithm": "Greedy (highest severity first, then nearest)"},
    )


def run_sa(scenario: Dict[str, Any], seed: int = 42) -> Dict[str, Any]:
    inst: DispatchInstance = scenario["instance"]
    Q, offset, P = build_qubo(inst.cost_matrix)
    N = inst.n_vehicles  # square after build_qubo
    res = solve_simulated_annealing(Q, offset, N, seed=seed)
    cost = res["best_feasible_energy"] if res["best_feasible_energy"] is not None else res["best_energy"]
    # Decode assignments from best bitstring
    from model.qubo import decode_bitstring_to_assignments
    assignments, feas = decode_bitstring_to_assignments(res["best_bitstring"], N,
                                                         scenario["n_vehicles"],
                                                         scenario["n_incidents"])
    return _normalize_result(
        "Simulated Annealing", assignments, scenario,
        feasible=res["is_feasible"], runtime=res["runtime_sec"],
        metadata={"algorithm": "SA on QUBO (geometric cooling)", "n_qubits": N * N},
    )


def run_qaoa_solver(
    scenario: Dict[str, Any],
    p: int = 1,
    cvar_alpha: Optional[float] = None,
    n_restarts: int = 3,
    maxiter: int = 80,
    seed: int = 42,
    optimal_cost: Optional[float] = None,
) -> Dict[str, Any]:
    """Runs QAOA on the current scenario and returns normalized result + raw QAOA data."""
    inst: DispatchInstance = scenario["instance"]
    Q, offset, P = build_qubo(inst.cost_matrix)
    N = inst.n_vehicles

    res = run_qaoa(
        Q, offset, N=N, p=p,
        cvar_alpha=cvar_alpha,
        shots=4096, seed=seed,
        n_restarts=n_restarts,
        maxiter=maxiter,
        optimal_cost=optimal_cost,
    )

    from model.qubo import decode_bitstring_to_assignments
    assignments = res["assignments"]  # already decoded by run_qaoa

    solver_label = f"CVaR-QAOA (p={p}, α={cvar_alpha})" if cvar_alpha else f"QAOA (p={p})"

    normalized = _normalize_result(
        solver_label, assignments, scenario,
        feasible=res["feasibility_rate"] > 0 and not res["fallback_used"],
        runtime=res["runtime_sec"],
        optimal_cost=optimal_cost,
        metadata={
            "p": p,
            "cvar_alpha": cvar_alpha,
            "num_qubits": N * N,
            "backend": "Qiskit Statevector Simulator",
            "optimizer": "COBYLA",
            "n_restarts": n_restarts,
            "maxiter": maxiter,
            "optimal_hit_prob": res["optimal_hit_prob"],
            "feasibility_rate": res["feasibility_rate"],
            "best_energy": res["best_energy"],
            "approx_ratio": res["approx_ratio"],
        },
    )
    # Attach raw QAOA data for Quantum Lab tab
    normalized["_qaoa_raw"] = res
    normalized["_Q"] = Q
    normalized["_offset"] = offset
    normalized["_P"] = P
    normalized["_N"] = N
    return normalized


def run_all_classical(scenario: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Runs all classical solvers and returns a name→result dict."""
    results = {}
    try:
        results["Hungarian (Exact)"] = run_hungarian(scenario)
    except Exception as e:
        results["Hungarian (Exact)"] = {"error": str(e)}

    try:
        results["Nearest-Vehicle Greedy"] = run_nearest_vehicle(scenario)
    except Exception as e:
        results["Nearest-Vehicle Greedy"] = {"error": str(e)}

    try:
        results["Severity-First Greedy"] = run_severity_first(scenario)
    except Exception as e:
        results["Severity-First Greedy"] = {"error": str(e)}

    try:
        results["Simulated Annealing"] = run_sa(scenario)
    except Exception as e:
        results["Simulated Annealing"] = {"error": str(e)}

    return results


def get_qubo_info(scenario: Dict[str, Any]) -> Dict[str, Any]:
    """Returns QUBO/Ising metadata for the Quantum Lab tab."""
    inst: DispatchInstance = scenario["instance"]
    Q, offset, P = build_qubo(inst.cost_matrix)
    N = inst.n_vehicles
    H_ising, ising_offset = qubo_to_ising(Q, offset)
    return {
        "N": N,
        "num_vars": N * N,
        "num_qubits": N * N,
        "P": round(P, 4),
        "offset": round(offset, 4),
        "H_ising": H_ising,
        "Q": Q,
        "n_pauli_terms": len(H_ising.paulis),
    }
