"""
solvers/sa.py
Simulated Annealing on the QUBO objective function with geometric cooling
and multiple restarts for a robust classical metaheuristic baseline.
"""

from typing import Dict, Any, Optional, Tuple, List
import time
import numpy as np
from model.qubo import evaluate_qubo_energy, decode_bitstring_to_assignments
from solvers.exact import is_bitstring_feasible


def solve_simulated_annealing(
    Q: np.ndarray,
    offset: float,
    N: int,
    n_steps: int = 3000,
    t_start: float = 20.0,
    t_end: float = 0.05,
    n_restarts: int = 5,
    seed: Optional[int] = 42
) -> Dict[str, Any]:
    """
    Simulated Annealing on the QUBO energy landscape:
      E(x) = x^T Q x + offset
      
    Uses geometric cooling schedule with single-bit flip transitions and
    fast O(D) delta-energy updates.
    
    Returns:
      Dict with 'best_energy', 'best_bitstring', 'is_feasible',
      'best_feasible_energy', 'assignments', 'runtime_sec'.
    """
    start_time = time.perf_counter()
    rng = np.random.default_rng(seed)
    
    D = N * N
    # Symmetrize cross terms for fast delta evaluation: W_jk = Q_jk + Q_kj
    W = Q + Q.T
    for i in range(D):
        W[i, i] = 0.0
        
    cooling_factor = (t_end / t_start) ** (1.0 / max(n_steps - 1, 1))
    
    global_best_energy = float("inf")
    global_best_bitstring = None
    best_feasible_energy = float("inf")
    best_feasible_bitstring = None
    
    for restart in range(n_restarts):
        # Initialize either from random binary or random permutation
        if rng.random() < 0.5:
            # Random permutation mapped to N*N
            perm = rng.permutation(N)
            current_x = np.zeros(D, dtype=int)
            for v, i in enumerate(perm):
                current_x[v * N + i] = 1
        else:
            current_x = rng.integers(0, 2, size=D, dtype=int)
            
        current_energy = evaluate_qubo_energy(Q, offset, current_x)
        t = t_start
        
        for step in range(n_steps):
            # Pick a random variable to flip
            flip_idx = rng.integers(0, D)
            current_val = current_x[flip_idx]
            new_val = 1 - current_val
            delta_val = new_val - current_val  # +1 or -1
            
            # Fast delta energy computation:
            # E_new - E_old = delta_val * [ Q_jj + sum_{k != j} W_jk * x_k ]
            delta_energy = delta_val * (Q[flip_idx, flip_idx] + np.dot(W[flip_idx], current_x))
            
            # Acceptance decision (Metropolis criterion)
            if delta_energy < 0 or rng.random() < np.exp(-delta_energy / max(t, 1e-8)):
                current_x[flip_idx] = new_val
                current_energy += delta_energy
                
                # Check global best
                if current_energy < global_best_energy:
                    global_best_energy = current_energy
                    global_best_bitstring = current_x.copy()
                    
                if is_bitstring_feasible(current_x, N):
                    if current_energy < best_feasible_energy:
                        best_feasible_energy = current_energy
                        best_feasible_bitstring = current_x.copy()
                        
            t *= cooling_factor
            
    elapsed = time.perf_counter() - start_time
    
    # Choose final bitstring: prefer best feasible if found, otherwise global best
    chosen_bitstring = best_feasible_bitstring if best_feasible_bitstring is not None else global_best_bitstring
    chosen_energy = best_feasible_energy if best_feasible_bitstring is not None else global_best_energy
    is_feasible = (best_feasible_bitstring is not None)
    
    assignments, _ = decode_bitstring_to_assignments(chosen_bitstring, N) if chosen_bitstring is not None else ([], False)
    
    return {
        "best_energy": round(float(chosen_energy), 4),
        "best_bitstring": chosen_bitstring,
        "is_feasible": is_feasible,
        "best_feasible_energy": round(float(best_feasible_energy), 4) if best_feasible_bitstring is not None else None,
        "assignments": assignments,
        "runtime_sec": round(elapsed, 4)
    }
