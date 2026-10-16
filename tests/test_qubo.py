"""
tests/test_qubo.py
Validation test suite confirming:
1. Direct Hamiltonian formula matches QUBO energy: H(x) == x^T Q x + offset
2. Exhaustive QUBO minimum equals Hungarian cost across random instances
3. Global QUBO minimum is feasible (permutation matrix)
4. SparsePauliOp Ising diagonal eigenvalues match QUBO energy for all states
5. Hungarian cost <= greedy costs on all tested instances
6. Rectangular padding creates valid square assignments
"""

import pytest
import numpy as np
from itertools import product
from qiskit.quantum_info import Statevector

from data.instance import generate_instance, generate_greedy_failure_instance
from model.qubo import (
    build_qubo,
    compute_direct_hamiltonian,
    evaluate_qubo_energy,
    pad_to_square,
    decode_bitstring_to_assignments
)
from model.ising import (
    qubo_to_ising,
    decode_qiskit_bitstring,
    encode_vector_to_qiskit_bitstring
)
from solvers.exact import solve_hungarian, solve_brute_force_qubo
from solvers.greedy import solve_nearest_vehicle, solve_severity_first_greedy


def test_qubo_algebraic_equivalence():
    """Confirms x^T Q x + offset == direct formula H(x) on random bitstrings."""
    inst = generate_instance(3, 3, seed=101)
    Q, offset, P = build_qubo(inst.cost_matrix)
    
    rng = np.random.default_rng(42)
    test_vectors = rng.integers(0, 2, size=(30, 9))
    
    for x in test_vectors:
        q_energy = evaluate_qubo_energy(Q, offset, x)
        direct_h = compute_direct_hamiltonian(x, inst.cost_matrix, P)
        assert abs(q_energy - direct_h) < 1e-10, f"Mismatch: {q_energy} != {direct_h}"


def test_qubo_minimum_matches_hungarian():
    """
    Validates that the brute-force global minimum of the QUBO
    strictly equals Hungarian cost on 20 random 3x3 instances.
    """
    for seed in range(20):
        inst = generate_instance(3, 3, seed=seed)
        Q, offset, _ = build_qubo(inst.cost_matrix)
        
        bf_res = solve_brute_force_qubo(Q, offset, N=3)
        h_res = solve_hungarian(inst.cost_matrix)
        
        assert bf_res["is_feasible"] is True, f"Seed {seed}: QUBO minimum is infeasible"
        assert abs(bf_res["best_energy"] - h_res["total_cost"]) < 1e-4, (
            f"Seed {seed}: QUBO min {bf_res['best_energy']} != Hungarian {h_res['total_cost']}"
        )


def test_ising_matches_qubo_all_states_3x3():
    """
    Confirms that SparsePauliOp Ising Hamiltonian expectation value equals
    QUBO energy for all 512 states of a 3x3 problem.
    """
    inst = generate_instance(3, 3, seed=42)
    Q, offset, _ = build_qubo(inst.cost_matrix)
    H, const_I = qubo_to_ising(Q, offset)
    
    D = 9
    for bit_tuple in product([0, 1], repeat=D):
        x = np.array(bit_tuple)
        q_energy = evaluate_qubo_energy(Q, offset, x)
        
        # Build Qiskit label (little endian: qubit 0 is rightmost)
        label = encode_vector_to_qiskit_bitstring(x)
        sv = Statevector.from_label(label)
        ising_energy = float(sv.expectation_value(H).real)
        
        assert abs(q_energy - ising_energy) < 1e-9, (
            f"State {label}: QUBO {q_energy} != Ising {ising_energy}"
        )


def test_baselines_hungarian_optimal():
    """Confirms Hungarian cost <= greedy heuristic costs on 30 random instances."""
    for seed in range(30):
        inst = generate_instance(3, 3, seed=seed)
        h_res = solve_hungarian(inst.cost_matrix)
        nv_res = solve_nearest_vehicle(inst.travel_time_matrix, inst.weights)
        sf_res = solve_severity_first_greedy(inst.travel_time_matrix, inst.severities, inst.weights)
        
        assert h_res["total_cost"] <= nv_res["total_cost"] + 1e-6
        assert h_res["total_cost"] <= sf_res["total_cost"] + 1e-6


def test_greedy_failure_scenario():
    """Verifies that nearest-vehicle greedy fails on the handcrafted scenario."""
    inst = generate_greedy_failure_instance()
    h_res = solve_hungarian(inst.cost_matrix)
    nv_res = solve_nearest_vehicle(inst.travel_time_matrix, inst.weights)
    
    # Hungarian achieves 65.0, nearest vehicle achieves 169.0
    assert h_res["total_cost"] < nv_res["total_cost"]
    assert nv_res["total_cost"] >= 1.5 * h_res["total_cost"], "Greedy failure not pronounced enough"


def test_rectangular_padding():
    """Confirms rectangular matrix padding to square works correctly."""
    # 2 vehicles, 3 incidents
    inst = generate_instance(2, 3, seed=77)
    C_padded, n_v, n_i = pad_to_square(inst.cost_matrix, inst.weights)
    assert C_padded.shape == (3, 3)
    assert n_v == 2
    assert n_i == 3
    
    # 3 vehicles, 2 incidents
    inst2 = generate_instance(3, 2, seed=88)
    C_padded2, n_v2, n_i2 = pad_to_square(inst2.cost_matrix, inst2.weights)
    assert C_padded2.shape == (3, 3)
    assert n_v2 == 3
    assert n_i2 == 2
    # Third column should be 0.0 (dummy incident)
    assert np.all(C_padded2[:, 2] == 0.0)
