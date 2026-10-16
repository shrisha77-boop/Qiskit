"""
model/qubo.py
Constructs, pads, and validates the Quadratic Unconstrained Binary Optimization (QUBO)
formulation for severity-aware emergency vehicle dispatch.
"""

from typing import Tuple, Optional, Dict, Any, List
import numpy as np


def pad_to_square(
    cost_matrix: np.ndarray,
    weights: Optional[np.ndarray] = None,
    unserved_penalty_factor: float = 2.0
) -> Tuple[np.ndarray, int, int]:
    """
    Pads rectangular cost matrix (n_vehicles x n_incidents) to a square N x N matrix.
    
    Rules:
      - If n_vehicles < n_incidents: Add dummy vehicles. Assigning a dummy vehicle
        to incident i incurs penalty = unserved_penalty_factor * max(cost_matrix)
        (or weighted unserved cost), ensuring all real vehicles are prioritized.
      - If n_vehicles > n_incidents: Add dummy incidents with cost 0.0 (idle vehicle).
      
    Returns:
      (padded_cost_matrix, original_n_v, original_n_i)
    """
    n_v, n_i = cost_matrix.shape
    if n_v == n_i:
        return cost_matrix.copy(), n_v, n_i
        
    N = max(n_v, n_i)
    C_padded = np.zeros((N, N), dtype=float)
    
    max_c = float(np.max(cost_matrix)) if cost_matrix.size > 0 else 10.0
    unserved_cost = max(max_c * unserved_penalty_factor, 50.0)
    
    if n_v < n_i:
        # Copy existing vehicle costs
        C_padded[:n_v, :n_i] = cost_matrix
        # Fill dummy vehicles with unserved incident penalty
        for dummy_v in range(n_v, N):
            for i in range(n_i):
                if weights is not None and i < len(weights):
                    C_padded[dummy_v, i] = weights[i] * (unserved_cost / max(weights))
                else:
                    C_padded[dummy_v, i] = unserved_cost
    else:
        # n_v > n_i: idle vehicles have 0 cost for dummy incidents
        C_padded[:n_v, :n_i] = cost_matrix
        # Dummy incidents (columns n_i to N) remain 0.0
        
    return C_padded, n_v, n_i


def build_qubo(
    cost_matrix: np.ndarray,
    P: Optional[float] = None,
    penalty_multiplier: float = 1.5
) -> Tuple[np.ndarray, float, float]:
    """
    Builds the upper-triangular QUBO matrix Q and constant offset for the assignment:
      H(x) = x^T Q x + offset
      
    Where x is binary vector of length N*N, indexed by k = v * N + i.
    
    Constraints:
      1. Each vehicle assigned to exactly one incident: sum_i x_vi = 1
      2. Each incident assigned to exactly one vehicle: sum_v x_vi = 1
      
    Objective:
      sum_{v, i} c_{vi} x_{vi}
      + P * sum_v (sum_i x_vi - 1)^2
      + P * sum_i (sum_v x_vi - 1)^2
      
    Using x_k^2 = x_k:
      (sum_k x_k - 1)^2 = -sum_k x_k + 2 sum_{a < b} x_a x_b + 1
      
    Returns:
      (Q, offset, P)
    """
    N = cost_matrix.shape[0]
    num_vars = N * N
    
    max_c = float(np.max(cost_matrix)) if cost_matrix.size > 0 else 1.0
    if P is None:
        P = max(max_c * penalty_multiplier, 1.0)
        
    Q = np.zeros((num_vars, num_vars), dtype=float)
    idx = lambda v, i: v * N + i
    
    # 1. Linear dispatch costs on the diagonal
    for v in range(N):
        for i in range(N):
            k = idx(v, i)
            Q[k, k] += cost_matrix[v, i]
            
    # 2. Row constraint: each vehicle v serves exactly one incident
    # (sum_i x_vi - 1)^2 = -sum_i x_vi + 2 sum_{i < j} x_vi x_vj + 1
    for v in range(N):
        for i in range(N):
            k = idx(v, i)
            Q[k, k] -= P
            for j in range(i + 1, N):
                k2 = idx(v, j)
                Q[k, k2] += 2.0 * P
                
    # 3. Column constraint: each incident i is served by exactly one vehicle
    # (sum_v x_vi - 1)^2 = -sum_v x_vi + 2 sum_{u < v} x_ui x_vi + 1
    for i in range(N):
        for v in range(N):
            k = idx(v, i)
            Q[k, k] -= P
            for u in range(v + 1, N):
                k2 = idx(u, i)
                # Ensure upper triangular: since v < u, idx(v, i) < idx(u, i)
                Q[k, k2] += 2.0 * P
                
    # Offset: 2 * N constraints, each contributing +P constant
    offset = float(2 * N * P)
    
    return Q, offset, P


def compute_direct_hamiltonian(
    x: np.ndarray,
    cost_matrix: np.ndarray,
    P: float
) -> float:
    """
    Direct non-expanded evaluation of H(x) to test QUBO algebraic equivalence:
      H(x) = sum c_vi x_vi + P sum_v (sum_i x_vi - 1)^2 + P sum_i (sum_v x_vi - 1)^2
    """
    N = cost_matrix.shape[0]
    X = x.reshape((N, N))
    cost = float(np.sum(cost_matrix * X))
    row_violations = float(np.sum((np.sum(X, axis=1) - 1.0) ** 2))
    col_violations = float(np.sum((np.sum(X, axis=0) - 1.0) ** 2))
    return cost + P * row_violations + P * col_violations


def evaluate_qubo_energy(Q: np.ndarray, offset: float, x: np.ndarray) -> float:
    """Computes E(x) = x^T Q x + offset."""
    return float(x @ Q @ x + offset)


def decode_bitstring_to_assignments(
    bitstring: np.ndarray,
    N: int,
    original_n_v: Optional[int] = None,
    original_n_i: Optional[int] = None
) -> Tuple[List[Tuple[int, int]], bool]:
    """
    Decodes an N*N bitstring into a list of (vehicle, incident) pairs.
    Checks feasibility (each vehicle has at most 1, each incident has at most 1).
    Filters out dummy vehicles/incidents if original dimensions are provided.
    """
    mat = bitstring.reshape((N, N))
    row_sums = np.sum(mat, axis=1)
    col_sums = np.sum(mat, axis=0)
    is_feasible = bool(np.all(row_sums == 1) and np.all(col_sums == 1))
    
    assignments: List[Tuple[int, int]] = []
    max_v = original_n_v if original_n_v is not None else N
    max_i = original_n_i if original_n_i is not None else N
    
    for v in range(N):
        matched = np.where(mat[v] == 1)[0]
        for i in matched:
            if v < max_v and i < max_i:
                assignments.append((v, int(i)))
                
    return assignments, is_feasible
