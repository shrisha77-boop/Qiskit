"""
solvers/greedy.py
Classical heuristic greedy dispatch baselines:
1. Nearest-Vehicle Greedy (First-Come, First-Served naive dispatch)
2. Severity-First Greedy (Priority triage dispatch)
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np


def solve_nearest_vehicle(
    travel_time_matrix: np.ndarray,
    weights: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Naive First-Come, First-Served nearest vehicle heuristic:
    Incidents are considered in arrival order (columns 0 to m-1).
    Each incident is assigned to the nearest currently unassigned vehicle.
    """
    n_v, n_i = travel_time_matrix.shape
    if weights is None:
        weights = np.ones(n_i, dtype=float)
        
    available_vehicles = set(range(n_v))
    assignments: List[Tuple[int, int]] = []
    total_cost = 0.0
    total_travel_time = 0.0
    
    for i in range(n_i):
        if not available_vehicles:
            break
        # Find closest available vehicle to incident i
        best_v = min(available_vehicles, key=lambda v: travel_time_matrix[v, i])
        available_vehicles.remove(best_v)
        
        t_vi = travel_time_matrix[best_v, i]
        w_i = weights[i]
        cost_vi = w_i * t_vi
        
        assignments.append((best_v, i))
        total_cost += cost_vi
        total_travel_time += t_vi
        
    return {
        "assignments": assignments,
        "total_cost": round(total_cost, 4),
        "total_travel_time": round(total_travel_time, 2),
        "unassigned_vehicles": list(available_vehicles),
        "unassigned_incidents": list(range(len(assignments), n_i))
    }


def solve_severity_first_greedy(
    travel_time_matrix: np.ndarray,
    severities: np.ndarray,
    weights: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Severity-First triage heuristic:
    Incidents are sorted in descending order of severity (highest first).
    Ties broken by index. Each critical incident is assigned to its nearest
    currently unassigned vehicle.
    """
    n_v, n_i = travel_time_matrix.shape
    if weights is None:
        weights = np.ones(n_i, dtype=float)
        
    # Sort incident indices by severity descending
    incident_order = np.argsort(-severities)
    
    available_vehicles = set(range(n_v))
    assignments: List[Tuple[int, int]] = []
    total_cost = 0.0
    total_travel_time = 0.0
    
    for i in incident_order:
        if not available_vehicles:
            break
        best_v = min(available_vehicles, key=lambda v: travel_time_matrix[v, i])
        available_vehicles.remove(best_v)
        
        t_vi = travel_time_matrix[best_v, i]
        w_i = weights[i]
        cost_vi = w_i * t_vi
        
        assignments.append((best_v, int(i)))
        total_cost += cost_vi
        total_travel_time += t_vi
        
    return {
        "assignments": assignments,
        "total_cost": round(total_cost, 4),
        "total_travel_time": round(total_travel_time, 2),
        "unassigned_vehicles": list(available_vehicles),
        "unassigned_incidents": [int(i) for i in incident_order[len(assignments):]]
    }
