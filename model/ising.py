"""
model/ising.py
Converts QUBO formulations into Qiskit SparsePauliOp Ising Hamiltonians
and handles bitstring endianness decoding between Qiskit and problem variables.
"""

from typing import Tuple, List, Dict, Optional, Any
import numpy as np
from qiskit.quantum_info import SparsePauliOp


def qubo_to_ising(Q: np.ndarray, offset: float) -> Tuple[SparsePauliOp, float]:
    """
    Directly converts a QUBO matrix Q and scalar offset into a Qiskit SparsePauliOp:
      H_C = sum_k h_k Z_k + sum_{j < k} J_jk Z_j Z_k + constant * I
      
    Where x_k in {0, 1} is mapped to Pauli Z eigenvalues via:
      x_k = (I - Z_k) / 2
      
    Note on Qiskit endianness:
      In Qiskit SparsePauliOp string representation, qubit k corresponds to
      character position (num_qubits - 1 - k) from the left.
      
    Returns:
      (H_ising, ising_offset)
    """
    D = Q.shape[0]
    
    # Symmetrize off-diagonal terms: W_jk = Q_jk + Q_kj
    W = Q + Q.T
    for i in range(D):
        W[i, i] = 0.0  # Zero out diagonal for off-diagonal cross matrix
        
    # Constant term (Identity)
    const_I = offset + 0.5 * float(np.sum(np.diag(Q))) + 0.25 * float(np.sum(np.triu(W, 1)))
    
    pauli_list = []
    
    # Identity term
    if abs(const_I) > 1e-12:
        pauli_list.append(("I" * D, float(const_I)))
        
    # 1. Single Z terms: h_k = -0.5 * Q_kk - 0.25 * sum_{j != k} W_jk
    for k in range(D):
        hk = -0.5 * float(Q[k, k]) - 0.25 * float(np.sum(W[k, :]))
        if abs(hk) > 1e-12:
            s = ["I"] * D
            s[D - 1 - k] = "Z"
            pauli_list.append(("".join(s), float(hk)))
            
    # 2. Two-qubit ZZ terms: J_jk = 0.25 * W_jk
    for j in range(D):
        for k in range(j + 1, D):
            Jjk = 0.25 * float(W[j, k])
            if abs(Jjk) > 1e-12:
                s = ["I"] * D
                s[D - 1 - j] = "Z"
                s[D - 1 - k] = "Z"
                pauli_list.append(("".join(s), float(Jjk)))
                
    H = SparsePauliOp.from_list(pauli_list).simplify()
    return H, const_I


def decode_qiskit_bitstring(bitstring_str: str, N: int) -> np.ndarray:
    """
    Converts a Qiskit measurement bitstring (e.g. '010001001') where qubit 0 is
    the rightmost character into a problem-ordered binary vector x of shape (N*N,)
    where x[k] corresponds to variable x_{v, i} (k = v * N + i).
    
    Example:
      If bitstring_str has length D, then qubit k = 0 has value bitstring_str[D - 1].
      In general: x[k] = int(bitstring_str[D - 1 - k]).
    """
    D = len(bitstring_str)
    assert D == N * N, f"Bitstring length {D} does not match N*N = {N*N}"
    x = np.zeros(D, dtype=int)
    for k in range(D):
        x[k] = int(bitstring_str[D - 1 - k])
    return x


def encode_vector_to_qiskit_bitstring(x: np.ndarray) -> str:
    """
    Converts a problem binary vector x into Qiskit's little-endian bitstring string.
    Qubit 0 is rightmost.
    """
    D = len(x)
    chars = ["0"] * D
    for k in range(D):
        chars[D - 1 - k] = str(int(x[k]))
    return "".join(chars)
