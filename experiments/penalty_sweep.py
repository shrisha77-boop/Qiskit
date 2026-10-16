"""
experiments/penalty_sweep.py
Analyzes the sensitivity of QUBO formulation and QAOA optimization to penalty weight P.
Sweeps P / max(C) from 0.5 to 4.0, tracking:
- Feasibility of the theoretical QUBO minimum
- QAOA Feasibility Rate
- QAOA Optimal-Hit Probability
Saves results to results/penalty_sweep.csv and generates figures/penalty_sensitivity.png.
"""

from typing import List, Dict, Any
import os
import sys
# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data.instance import generate_instance
from model.qubo import build_qubo
from solvers.exact import solve_hungarian, solve_brute_force_qubo
from solvers.qaoa import run_qaoa


def run_penalty_sweep(
    multipliers: List[float] = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0],
    n_instances: int = 5,
    seed_offset: int = 200
) -> pd.DataFrame:
    """
    Sweeps penalty multiplier P / max(C) across multiple instances.
    """
    records: List[Dict[str, Any]] = []
    print(f"\n--- Running Penalty Weight P Sensitivity Sweep ({len(multipliers)} multipliers, {n_instances} instances) ---")
    
    for mult in multipliers:
        print(f"  Evaluating multiplier P / max(C) = {mult:.2f}...")
        for inst_idx in range(n_instances):
            seed = seed_offset + inst_idx
            inst = generate_instance(3, 3, seed=seed)
            max_c = float(np.max(inst.cost_matrix))
            P = mult * max_c
            
            Q, offset, _ = build_qubo(inst.cost_matrix, P=P)
            h_res = solve_hungarian(inst.cost_matrix)
            opt_cost = h_res["total_cost"]
            
            # 1. Theoretical QUBO ground state feasibility
            bf_res = solve_brute_force_qubo(Q, offset, 3)
            qubo_min_feasible = bf_res["is_feasible"]
            
            # 2. QAOA performance at this penalty weight (p=2, 2 restarts)
            q_res = run_qaoa(
                Q, offset, 3, p=2, cvar_alpha=0.20,
                seed=seed, n_restarts=2, maxiter=60,
                optimal_cost=opt_cost
            )
            
            records.append({
                "multiplier": mult,
                "instance_id": inst_idx,
                "P": round(P, 2),
                "max_C": round(max_c, 2),
                "qubo_min_feasible": 1.0 if qubo_min_feasible else 0.0,
                "qaoa_feasibility_rate": q_res["feasibility_rate"],
                "qaoa_optimal_hit_prob": q_res["optimal_hit_prob"],
                "approx_ratio": q_res["approx_ratio"]
            })
            
    df = pd.DataFrame(records)
    return df


def plot_penalty_sensitivity(df: pd.DataFrame, save_path: str = "figures/penalty_sensitivity.png") -> None:
    """
    Renders two-panel penalty sensitivity plot demonstrating the delicate balance of QUBO penalty tuning.
    """
    summary = df.groupby("multiplier")[["qubo_min_feasible", "qaoa_feasibility_rate", "qaoa_optimal_hit_prob"]].mean().reset_index()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)
    
    # Panel 1: Feasibility vs Penalty
    ax1.plot(
        summary["multiplier"], summary["qubo_min_feasible"] * 100,
        marker="s", color="#e74c3c", linewidth=2.0, label="QUBO Theoretical Ground State Feasibility (%)"
    )
    ax1.plot(
        summary["multiplier"], summary["qaoa_feasibility_rate"] * 100,
        marker="o", color="#2980b9", linewidth=2.0, label="QAOA Sampled Feasibility Rate (%)"
    )
    ax1.axvline(1.0, color="gray", linestyle=":", label="P = max(c) threshold")
    ax1.axvspan(1.25, 2.0, color="#2ecc71", alpha=0.15, label="Sweet Spot (P in [1.25, 2.0])")
    
    ax1.set_xlabel("Penalty Multiplier (P / max(C))", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Feasibility Rate (%)", fontsize=10, fontweight="bold")
    ax1.set_title("Feasibility vs Penalty Weight\nSimulator: Qiskit Statevector", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right", fontsize=7.5, framealpha=0.9)
    ax1.set_ylim(-5, 105)
    
    # Panel 2: Optimal-Hit Probability vs Penalty
    ax2.plot(
        summary["multiplier"], summary["qaoa_optimal_hit_prob"] * 100,
        marker="^", color="#27ae60", linewidth=2.2, label="QAOA Optimal-Hit Probability (%)"
    )
    ax2.axvline(1.0, color="gray", linestyle=":", label="P = max(c) threshold")
    ax2.axvspan(1.25, 2.0, color="#2ecc71", alpha=0.15, label="Optimal Tuning Window")
    
    ax2.set_xlabel("Penalty Multiplier (P / max(C))", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Optimal Hit Probability (%)", fontsize=10, fontweight="bold")
    ax2.set_title("Optimal Hit Probability vs Penalty Weight\nSimulator: Qiskit Statevector", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=7.5, framealpha=0.9)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Penalty sensitivity plot saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run QUBO Penalty Weight Sensitivity Sweep")
    parser.add_argument("--instances", type=int, default=5, help="Number of instances per multiplier")
    args = parser.parse_args()
    
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    
    df_sweep = run_penalty_sweep(n_instances=args.instances)
    csv_path = "results/penalty_sweep.csv"
    df_sweep.to_csv(csv_path, index=False)
    print(f"Penalty sweep results saved to {csv_path}")
    
    plot_penalty_sensitivity(df_sweep, save_path="figures/penalty_sensitivity.png")
