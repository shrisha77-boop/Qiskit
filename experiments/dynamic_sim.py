"""
experiments/dynamic_sim.py
Rolling-horizon dynamic emergency dispatch simulation.
Models online incident arrivals over discrete time windows, tracking vehicle availability,
accumulated waiting times, and total severity-weighted delay across dispatch algorithms.
"""

from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
import os
import sys
# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data.instance import (
    SEVERITY_WEIGHT_MAP,
    compute_manhattan_travel_times
)
from model.qubo import pad_to_square, build_qubo
from solvers.exact import solve_hungarian
from solvers.greedy import solve_nearest_vehicle, solve_severity_first_greedy
from solvers.qaoa import run_qaoa


@dataclass
class DynamicIncident:
    id: int
    arrival_time: float
    position: np.ndarray
    severity: int
    weight: float
    served: bool = False
    service_start_time: Optional[float] = None


@dataclass
class DynamicVehicle:
    id: int
    position: np.ndarray
    busy_until: float = 0.0


def generate_dynamic_scenario(
    n_windows: int = 10,
    window_duration: float = 5.0,
    grid_size: float = 10.0,
    arrival_rate: float = 1.4,
    fleet_size: int = 4,
    seed: int = 42
) -> Tuple[List[DynamicVehicle], List[List[DynamicIncident]]]:
    """
    Generates a reproducible dynamic arrival scenario over n_windows.
    Returns:
      (initial_vehicles, incident_arrivals_per_window)
    """
    rng = np.random.default_rng(seed)
    
    # Initialize fleet distributed across grid
    vehicles = [
        DynamicVehicle(
            id=v,
            position=rng.uniform(1.0, grid_size - 1.0, size=2),
            busy_until=0.0
        )
        for v in range(fleet_size)
    ]
    
    # Generate arrivals for each time window
    window_arrivals: List[List[DynamicIncident]] = []
    global_inc_id = 0
    
    for w in range(n_windows):
        t_now = w * window_duration
        n_new = int(rng.poisson(arrival_rate))
        # Ensure at least 1 and at most 4 per window for tractability
        n_new = max(1, min(n_new, 4))
        
        incidents = []
        for _ in range(n_new):
            pos = rng.uniform(0.5, grid_size - 0.5, size=2)
            sev = int(rng.integers(1, 6))
            weight = float(SEVERITY_WEIGHT_MAP[sev])
            inc = DynamicIncident(
                id=global_inc_id,
                arrival_time=t_now,
                position=pos,
                severity=sev,
                weight=weight
            )
            incidents.append(inc)
            global_inc_id += 1
            
        window_arrivals.append(incidents)
        
    return vehicles, window_arrivals


def simulate_solver_policy(
    solver_name: str,
    initial_vehicles: List[DynamicVehicle],
    window_arrivals: List[List[DynamicIncident]],
    window_duration: float = 5.0,
    service_time: float = 6.0
) -> Dict[str, Any]:
    """
    Simulates a given dispatch solver policy through the rolling-horizon event stream.
    """
    import copy
    vehicles = copy.deepcopy(initial_vehicles)
    pending_incidents: List[DynamicIncident] = []
    
    cumulative_delay = 0.0
    history: List[Dict[str, Any]] = []
    
    n_windows = len(window_arrivals)
    
    for w in range(n_windows):
        current_time = w * window_duration
        
        # 1. New arrivals join pending queue
        new_incidents = copy.deepcopy(window_arrivals[w])
        pending_incidents.extend(new_incidents)
        
        # 2. Check available vehicles
        free_vehicles = [v for v in vehicles if v.busy_until <= current_time]
        
        # 3. Formulate dispatch if free vehicles and pending incidents exist
        if free_vehicles and pending_incidents:
            # Take up to 4 pending and up to 4 free vehicles
            v_subset = free_vehicles[:4]
            i_subset = pending_incidents[:4]
            
            v_positions = np.array([v.position for v in v_subset])
            i_positions = np.array([inc.position for inc in i_subset])
            severities = np.array([inc.severity for inc in i_subset])
            weights = np.array([inc.weight for inc in i_subset])
            
            T = compute_manhattan_travel_times(v_positions, i_positions)
            C = T * weights[np.newaxis, :]
            
            # Solve assignment
            assignments: List[Tuple[int, int]] = []
            
            if solver_name == "Hungarian":
                h_res = solve_hungarian(C)
                assignments = h_res["assignments"]
            elif solver_name == "Severity-First":
                sf_res = solve_severity_first_greedy(T, severities, weights)
                assignments = sf_res["assignments"]
            elif solver_name == "Nearest-Vehicle":
                nv_res = solve_nearest_vehicle(T, weights)
                assignments = nv_res["assignments"]
            elif solver_name == "QAOA":
                C_pad, orig_nv, orig_ni = pad_to_square(C, weights)
                N = C_pad.shape[0]
                Q, offset, _ = build_qubo(C_pad)
                q_res = run_qaoa(Q, offset, N, p=2, cvar_alpha=0.20, n_restarts=1, maxiter=40, seed=w)
                assignments = [(v, i) for v, i in q_res["assignments"] if v < orig_nv and i < orig_ni]
            else:
                raise ValueError(f"Unknown solver: {solver_name}")
                
            # Execute assignments
            served_incident_ids = set()
            for v_idx, i_idx in assignments:
                v = v_subset[v_idx]
                inc = i_subset[i_idx]
                
                travel_t = T[v_idx, i_idx]
                # Response delay experienced by this patient
                response_delay = (current_time - inc.arrival_time) + travel_t
                weighted_delay = inc.weight * response_delay
                cumulative_delay += weighted_delay
                
                # Update vehicle status
                v.busy_until = current_time + travel_t + service_time
                v.position = inc.position.copy()  # Vehicle moves to scene
                
                served_incident_ids.add(inc.id)
                
            # Remove served incidents from pending queue
            pending_incidents = [inc for inc in pending_incidents if inc.id not in served_incident_ids]
            
        # Accumulate waiting penalty for unserved incidents remaining in queue
        for inc in pending_incidents:
            cumulative_delay += inc.weight * window_duration
            
        history.append({
            "window": w,
            "time_minutes": current_time,
            "cumulative_delay": round(cumulative_delay, 2),
            "pending_count": len(pending_incidents),
            "solver": solver_name
        })
        
    return {
        "solver": solver_name,
        "final_cumulative_delay": round(cumulative_delay, 2),
        "history": history
    }


def run_dynamic_experiment(n_windows: int = 8, seed: int = 42) -> pd.DataFrame:
    """
    Runs dynamic dispatch simulation across Hungarian, Severity-First, Nearest-Vehicle, and QAOA.
    """
    print(f"\n--- Running Rolling-Horizon Dynamic Dispatch Simulation ({n_windows} windows) ---")
    vehicles, window_arrivals = generate_dynamic_scenario(n_windows=n_windows, seed=seed)
    
    solvers = ["Hungarian", "Severity-First", "QAOA", "Nearest-Vehicle"]
    all_records: List[Dict[str, Any]] = []
    
    for s_name in solvers:
        print(f"  Simulating policy: {s_name}...")
        res = simulate_solver_policy(s_name, vehicles, window_arrivals)
        all_records.extend(res["history"])
        print(f"    {s_name} Final Cumulative Delay: {res['final_cumulative_delay']:.1f}")
        
    df = pd.DataFrame(all_records)
    return df


def plot_dynamic_simulation(df: pd.DataFrame, save_path: str = "figures/dynamic_dispatch_simulation.png") -> None:
    """
    Renders cumulative delay over time curves across policies, labelled 'simulator'.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)
    
    color_map = {
        "Hungarian": "#2ecc71",
        "Severity-First": "#9b59b6",
        "QAOA": "#e67e22",
        "Nearest-Vehicle": "#e74c3c"
    }
    
    # Panel 1: Cumulative Delay over Time
    for solver, group in df.groupby("solver"):
        c = color_map.get(solver, "#34495e")
        ls = "-" if solver in ["Hungarian", "Severity-First"] else ("--" if solver == "QAOA" else "-.")
        ax1.plot(group["time_minutes"], group["cumulative_delay"], marker="o", linestyle=ls, label=solver, color=c, linewidth=2.0)
        
    ax1.set_xlabel("Elapsed Time (Minutes)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Cumulative Severity-Weighted Delay", fontsize=10, fontweight="bold")
    ax1.set_title("Cumulative Response Delay Over Time\nSimulator: Rolling Horizon Dispatch", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left", framealpha=0.9, fontsize=8.5)
    
    # Panel 2: Queue Length (Pending Incidents) over Time
    for solver, group in df.groupby("solver"):
        c = color_map.get(solver, "#34495e")
        ax2.plot(group["time_minutes"], group["pending_count"], marker="s", label=solver, color=c, linewidth=1.8)
        
    ax2.set_xlabel("Elapsed Time (Minutes)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Pending Calls in Queue", fontsize=10, fontweight="bold")
    ax2.set_title("Queue Congestion Over Time\nSimulator: Unserved Calls", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left", framealpha=0.9, fontsize=8.5)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Dynamic dispatch simulation figure saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Dynamic Dispatch Simulation")
    parser.add_argument("--windows", type=int, default=8, help="Number of rolling horizon windows")
    args = parser.parse_args()
    
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    
    df_dynamic = run_dynamic_experiment(n_windows=args.windows)
    csv_path = "results/dynamic_simulation.csv"
    df_dynamic.to_csv(csv_path, index=False)
    print(f"Dynamic simulation results written to {csv_path}")
    
    plot_dynamic_simulation(df_dynamic, save_path="figures/dynamic_dispatch_simulation.png")
