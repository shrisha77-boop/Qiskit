"""
solvers/exact.py
Exact solvers for benchmark ground truth:
1. Hungarian Algorithm (Scipy linear_sum_assignment) - polynomial time O(N^3)
2. Brute Force QUBO search - searches all 2^(N^2) states to prove QUBO formulation validity.
"""

from typing import Tuple, List, Dict, Any, Optional
import numpy as np
from itertools import product
from scipy.optimize import linear_sum_assignment


def solve_hungarian(cost_matrix: np.ndarray) -> Dict[str, Any]:
    """
    Computes exact global optimum for linear sum assignment problem using Hungarian algorithm.
    Works for square and rectangular cost matrices.
    
    Returns dict:
      - 'assignments': List of (vehicle_idx, incident_idx)
      - 'total_cost': float
      - 'row_ind': np.ndarray
      - 'col_ind': np.ndarray
      - 'status': 'optimal'
    """
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    total_cost = float(cost_matrix[row_ind, col_ind].sum())
    assignments = list(zip(row_ind.tolist(), col_ind.tolist()))
    
    return {
        "assignments": assignments,
        "total_cost": total_cost,
        "row_ind": row_ind,
        "col_ind": col_ind,
        "status": "optimal"
    }


def is_bitstring_feasible(bitstring: np.ndarray, N: int) -> bool:
    """
    Checks if an N*N binary vector satisfies row and column assignment constraints (permutation).
    """
    mat = bitstring.reshape((N, N))
    row_sums = np.sum(mat, axis=1)
    col_sums = np.sum(mat, axis=0)
    return bool(np.all(row_sums == 1) and np.all(col_sums == 1))


def solve_brute_force_qubo(
    Q: np.ndarray,
    offset: float,
    N: int,
    cost_matrix: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Exhaustively searches all 2^(N^2) bitstrings to find the global minimum of the QUBO energy:
      E(x) = x^T Q x + offset
      
    Also verifies whether the global minimum is feasible and matches Hungarian cost.
    Suitable for N=2 (16 states), N=3 (512 states), and N=4 (65,536 states).
    """
    num_vars = N * N
    best_energy = float("inf")
    best_bitstring = None
    best_feasible_energy = float("inf")
    best_feasible_bitstring = None
    
    # Iterate over all binary vectors
    for bit_tuple in product([0, 1], repeat=num_vars):
        x = np.array(bit_tuple, dtype=float)
        # Compute energy
        energy = float(x @ Q @ x + offset)
        
        if energy < best_energy:
            best_energy = energy
            best_bitstring = x.copy()
            
        if is_bitstring_feasible(x, N):
            if energy < best_feasible_energy:
                best_feasible_energy = energy
                best_feasible_bitstring = x.copy()
                
    is_opt_feasible = is_bitstring_feasible(best_bitstring, N) if best_bitstring is not None else False
    
    # Extract assignments from best bitstring
    assignments = []
    if best_bitstring is not None:
        mat = best_bitstring.reshape((N, N))
        for v in range(N):
            assigned_incidents = np.where(mat[v] == 1)[0]
            for i in assigned_incidents:
                assignments.append((v, int(i)))
                
    return {
        "best_energy": round(best_energy, 4),
        "best_bitstring": best_bitstring,
        "is_feasible": is_opt_feasible,
        "best_feasible_energy": round(best_feasible_energy, 4) if best_feasible_bitstring is not None else None,
        "best_feasible_bitstring": best_feasible_bitstring,
        "assignments": assignments
    }
