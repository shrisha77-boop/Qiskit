"""
experiments/benchmark.py
Comparative benchmark across Hungarian, Greedy heuristics, Simulated Annealing,
and QAOA / CVaR-QAOA over multiple random problem instances.
Saves results to results/benchmark.csv and generates comparative figures.
"""

from typing import Dict, Any, List, Optional
import os
import sys
# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data.instance import generate_instance
from model.qubo import build_qubo
from solvers.exact import solve_hungarian
from solvers.greedy import solve_nearest_vehicle, solve_severity_first_greedy
from solvers.sa import solve_simulated_annealing
from solvers.qaoa import run_qaoa


def run_benchmarks(
    n_instances_3x3: int = 30,
    n_instances_4x4: int = 10,
    run_qaoa_solvers: bool = True,
    seed_offset: int = 100
) -> pd.DataFrame:
    """
    Executes benchmark matrix across all solvers on 3x3 and 4x4 instances.
    """
    records: List[Dict[str, Any]] = []
    
    # 1. 3x3 instances
    print(f"\n--- Running 3x3 Benchmarks ({n_instances_3x3} instances) ---")
    for i in range(n_instances_3x3):
        seed = seed_offset + i
        inst = generate_instance(3, 3, seed=seed)
        Q, offset, P = build_qubo(inst.cost_matrix)
        
        # Ground truth: Hungarian
        t0 = time.perf_counter()
        h_res = solve_hungarian(inst.cost_matrix)
        h_time = time.perf_counter() - t0
        opt_cost = h_res["total_cost"]
        
        records.append({
            "instance_id": i,
            "size": "3x3 (9 qubits)",
            "N": 3,
            "solver": "Hungarian (Exact)",
            "cost": opt_cost,
            "optimal_cost": opt_cost,
            "approx_ratio": 1.0,
            "is_feasible": True,
            "optimal_hit_prob": 1.0,
            "feasibility_rate": 1.0,
            "runtime_sec": round(h_time, 5)
        })
        
        # Greedy 1: Nearest Vehicle
        t0 = time.perf_counter()
        nv_res = solve_nearest_vehicle(inst.travel_time_matrix, inst.weights)
        nv_time = time.perf_counter() - t0
        records.append({
            "instance_id": i,
            "size": "3x3 (9 qubits)",
            "N": 3,
            "solver": "Nearest-Vehicle Greedy",
            "cost": nv_res["total_cost"],
            "optimal_cost": opt_cost,
            "approx_ratio": round(nv_res["total_cost"] / opt_cost, 4) if opt_cost > 0 else 1.0,
            "is_feasible": True,
            "optimal_hit_prob": 1.0 if abs(nv_res["total_cost"] - opt_cost) < 1e-4 else 0.0,
            "feasibility_rate": 1.0,
            "runtime_sec": round(nv_time, 5)
        })
        
        # Greedy 2: Severity-First
        t0 = time.perf_counter()
        sf_res = solve_severity_first_greedy(inst.travel_time_matrix, inst.severities, inst.weights)
        sf_time = time.perf_counter() - t0
        records.append({
            "instance_id": i,
            "size": "3x3 (9 qubits)",
            "N": 3,
            "solver": "Severity-First Greedy",
            "cost": sf_res["total_cost"],
            "optimal_cost": opt_cost,
            "approx_ratio": round(sf_res["total_cost"] / opt_cost, 4) if opt_cost > 0 else 1.0,
            "is_feasible": True,
            "optimal_hit_prob": 1.0 if abs(sf_res["total_cost"] - opt_cost) < 1e-4 else 0.0,
            "feasibility_rate": 1.0,
            "runtime_sec": round(sf_time, 5)
        })
        
        # Classical Heuristic: Simulated Annealing
        sa_res = solve_simulated_annealing(Q, offset, 3, seed=seed)
        sa_cost = sa_res["best_feasible_energy"] if sa_res["best_feasible_energy"] is not None else sa_res["best_energy"]
        records.append({
            "instance_id": i,
            "size": "3x3 (9 qubits)",
            "N": 3,
            "solver": "Simulated Annealing (QUBO)",
            "cost": sa_cost,
            "optimal_cost": opt_cost,
            "approx_ratio": round(sa_cost / opt_cost, 4) if opt_cost > 0 else 1.0,
            "is_feasible": sa_res["is_feasible"],
            "optimal_hit_prob": 1.0 if abs(sa_cost - opt_cost) < 1e-4 else 0.0,
            "feasibility_rate": 1.0 if sa_res["is_feasible"] else 0.0,
            "runtime_sec": sa_res["runtime_sec"]
        })
        
        # QAOA Solvers on 3x3
        if run_qaoa_solvers:
            # QAOA p=1
            q1 = run_qaoa(Q, offset, 3, p=1, seed=seed, n_restarts=2, maxiter=60, optimal_cost=opt_cost)
            q1_cost = q1["best_feasible_energy"] if q1["best_feasible_energy"] is not None else q1["best_energy"]
            records.append({
                "instance_id": i,
                "size": "3x3 (9 qubits)",
                "N": 3,
                "solver": "QAOA (p=1)",
                "cost": q1_cost,
                "optimal_cost": opt_cost,
                "approx_ratio": q1["approx_ratio"],
                "is_feasible": not q1["fallback_used"] and q1["feasibility_rate"] > 0,
                "optimal_hit_prob": q1["optimal_hit_prob"],
                "feasibility_rate": q1["feasibility_rate"],
                "runtime_sec": q1["runtime_sec"]
            })
            
            # QAOA p=2
            q2 = run_qaoa(Q, offset, 3, p=2, seed=seed, n_restarts=2, maxiter=60, optimal_cost=opt_cost)
            q2_cost = q2["best_feasible_energy"] if q2["best_feasible_energy"] is not None else q2["best_energy"]
            records.append({
                "instance_id": i,
                "size": "3x3 (9 qubits)",
                "N": 3,
                "solver": "QAOA (p=2)",
                "cost": q2_cost,
                "optimal_cost": opt_cost,
                "approx_ratio": q2["approx_ratio"],
                "is_feasible": not q2["fallback_used"] and q2["feasibility_rate"] > 0,
                "optimal_hit_prob": q2["optimal_hit_prob"],
                "feasibility_rate": q2["feasibility_rate"],
                "runtime_sec": q2["runtime_sec"]
            })
            
            # CVaR-QAOA p=2 (alpha=0.20)
            qcvar = run_qaoa(Q, offset, 3, p=2, cvar_alpha=0.20, seed=seed, n_restarts=2, maxiter=60, optimal_cost=opt_cost)
            qcvar_cost = qcvar["best_feasible_energy"] if qcvar["best_feasible_energy"] is not None else qcvar["best_energy"]
            records.append({
                "instance_id": i,
                "size": "3x3 (9 qubits)",
                "N": 3,
                "solver": "CVaR-QAOA (p=2, a=0.2)",
                "cost": qcvar_cost,
                "optimal_cost": opt_cost,
                "approx_ratio": qcvar["approx_ratio"],
                "is_feasible": not qcvar["fallback_used"] and qcvar["feasibility_rate"] > 0,
                "optimal_hit_prob": qcvar["optimal_hit_prob"],
                "feasibility_rate": qcvar["feasibility_rate"],
                "runtime_sec": qcvar["runtime_sec"]
            })
            
        if (i + 1) % 5 == 0 or (i + 1) == n_instances_3x3:
            print(f"  Completed {i+1}/{n_instances_3x3} 3x3 instances...")
            
    # 2. 4x4 instances (16 qubits)
    if n_instances_4x4 > 0:
        print(f"\n--- Running 4x4 Benchmarks ({n_instances_4x4} instances) ---")
        for i in range(n_instances_4x4):
            seed = seed_offset + 500 + i
            inst = generate_instance(4, 4, seed=seed)
            Q, offset, P = build_qubo(inst.cost_matrix)
            
            # Hungarian
            t0 = time.perf_counter()
            h_res = solve_hungarian(inst.cost_matrix)
            h_time = time.perf_counter() - t0
            opt_cost = h_res["total_cost"]
            
            records.append({
                "instance_id": i,
                "size": "4x4 (16 qubits)",
                "N": 4,
                "solver": "Hungarian (Exact)",
                "cost": opt_cost,
                "optimal_cost": opt_cost,
                "approx_ratio": 1.0,
                "is_feasible": True,
                "optimal_hit_prob": 1.0,
                "feasibility_rate": 1.0,
                "runtime_sec": round(h_time, 5)
            })
            
            # Nearest Vehicle
            t0 = time.perf_counter()
            nv_res = solve_nearest_vehicle(inst.travel_time_matrix, inst.weights)
            nv_time = time.perf_counter() - t0
            records.append({
                "instance_id": i,
                "size": "4x4 (16 qubits)",
                "N": 4,
                "solver": "Nearest-Vehicle Greedy",
                "cost": nv_res["total_cost"],
                "optimal_cost": opt_cost,
                "approx_ratio": round(nv_res["total_cost"] / opt_cost, 4) if opt_cost > 0 else 1.0,
                "is_feasible": True,
                "optimal_hit_prob": 1.0 if abs(nv_res["total_cost"] - opt_cost) < 1e-4 else 0.0,
                "feasibility_rate": 1.0,
                "runtime_sec": round(nv_time, 5)
            })
            
            # Severity-First
            t0 = time.perf_counter()
            sf_res = solve_severity_first_greedy(inst.travel_time_matrix, inst.severities, inst.weights)
            sf_time = time.perf_counter() - t0
            records.append({
                "instance_id": i,
                "size": "4x4 (16 qubits)",
                "N": 4,
                "solver": "Severity-First Greedy",
                "cost": sf_res["total_cost"],
                "optimal_cost": opt_cost,
                "approx_ratio": round(sf_res["total_cost"] / opt_cost, 4) if opt_cost > 0 else 1.0,
                "is_feasible": True,
                "optimal_hit_prob": 1.0 if abs(sf_res["total_cost"] - opt_cost) < 1e-4 else 0.0,
                "feasibility_rate": 1.0,
                "runtime_sec": round(sf_time, 5)
            })
            
            # Simulated Annealing
            sa_res = solve_simulated_annealing(Q, offset, 4, seed=seed)
            sa_cost = sa_res["best_feasible_energy"] if sa_res["best_feasible_energy"] is not None else sa_res["best_energy"]
            records.append({
                "instance_id": i,
                "size": "4x4 (16 qubits)",
                "N": 4,
                "solver": "Simulated Annealing (QUBO)",
                "cost": sa_cost,
                "optimal_cost": opt_cost,
                "approx_ratio": round(sa_cost / opt_cost, 4) if opt_cost > 0 else 1.0,
                "is_feasible": sa_res["is_feasible"],
                "optimal_hit_prob": 1.0 if abs(sa_cost - opt_cost) < 1e-4 else 0.0,
                "feasibility_rate": 1.0 if sa_res["is_feasible"] else 0.0,
                "runtime_sec": sa_res["runtime_sec"]
            })
            
            # Run QAOA on a subset of 4x4 (e.g. 3 instances) to prevent excessive runtime
            if run_qaoa_solvers and i < 3:
                q1 = run_qaoa(Q, offset, 4, p=1, seed=seed, n_restarts=1, maxiter=40, optimal_cost=opt_cost)
                records.append({
                    "instance_id": i,
                    "size": "4x4 (16 qubits)",
                    "N": 4,
                    "solver": "QAOA (p=1)",
                    "cost": q1["best_feasible_energy"] if q1["best_feasible_energy"] is not None else q1["best_energy"],
                    "optimal_cost": opt_cost,
                    "approx_ratio": q1["approx_ratio"],
                    "is_feasible": not q1["fallback_used"] and q1["feasibility_rate"] > 0,
                    "optimal_hit_prob": q1["optimal_hit_prob"],
                    "feasibility_rate": q1["feasibility_rate"],
                    "runtime_sec": q1["runtime_sec"]
                })
                
            if (i + 1) % 5 == 0 or (i + 1) == n_instances_4x4:
                print(f"  Completed {i+1}/{n_instances_4x4} 4x4 instances...")
                
    df = pd.DataFrame(records)
    return df


def plot_benchmark_figures(df: pd.DataFrame, output_dir: str = "figures") -> None:
    """
    Generates all comparative benchmark plots labelled 'simulator'.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Approximation Ratio Comparison (Bar chart with standard deviation)
    df_3x3 = df[df["size"] == "3x3 (9 qubits)"]
    summary_3x3 = df_3x3.groupby("solver")["approx_ratio"].agg(["mean", "std", "count"]).reset_index()
    
    # Sort solvers logically
    solver_order = [
        "Hungarian (Exact)",
        "Simulated Annealing (QUBO)",
        "Severity-First Greedy",
        "CVaR-QAOA (p=2, a=0.2)",
        "QAOA (p=2)",
        "QAOA (p=1)",
        "Nearest-Vehicle Greedy"
    ]
    summary_3x3["sort_key"] = summary_3x3["solver"].apply(lambda s: solver_order.index(s) if s in solver_order else 99)
    summary_3x3 = summary_3x3.sort_values("sort_key")
    
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    bars = ax.bar(
        summary_3x3["solver"],
        summary_3x3["mean"],
        yerr=summary_3x3["std"].fillna(0),
        capsize=4,
        color=["#2ecc71", "#3498db", "#9b59b6", "#e67e22", "#f39c12", "#f1c40f", "#e74c3c"],
        edgecolor="black",
        linewidth=1.2
    )
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.5, label="Optimal Ground Truth (Ratio = 1.0)")
    ax.set_ylabel("Approximation Ratio (Found Cost / Optimal)", fontsize=10, fontweight="bold")
    ax.set_title(
        "Solver Approximation Ratio Comparison (3x3 Instances)\n"
        "Simulator: Qiskit Statevector (Lower is better; 1.0 = Globally Optimal)",
        fontsize=11, fontweight="bold", pad=10
    )
    ax.set_xticks(range(len(summary_3x3)))
    ax.set_xticklabels(summary_3x3["solver"], rotation=30, ha="right", fontsize=8.5)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_ylim(0.9, max(summary_3x3["mean"].max() + 0.3, 1.6))
    
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 0.02, f"{h:.3f}", ha="center", va="bottom", fontsize=8)
        
    ax.legend(loc="upper left", framealpha=0.9, fontsize=8.5)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "approx_ratio_comparison.png"), bbox_inches="tight")
    plt.close(fig)
    
    # 2. QAOA Metrics vs Layer p and CVaR
    qaoa_solvers = ["QAOA (p=1)", "QAOA (p=2)", "CVaR-QAOA (p=2, a=0.2)"]
    df_q = df_3x3[df_3x3["solver"].isin(qaoa_solvers)]
    if not df_q.empty:
        q_summary = df_q.groupby("solver")[["optimal_hit_prob", "feasibility_rate"]].mean().reindex(qaoa_solvers).reset_index()
        
        fig, ax1 = plt.subplots(figsize=(8, 4.5), dpi=150)
        x = np.arange(len(qaoa_solvers))
        width = 0.35
        
        rects1 = ax1.bar(x - width/2, q_summary["optimal_hit_prob"] * 100, width, label="Optimal Hit Probability (%)", color="#27ae60", edgecolor="black")
        rects2 = ax1.bar(x + width/2, q_summary["feasibility_rate"] * 100, width, label="Feasibility Rate (%)", color="#2980b9", edgecolor="black")
        
        ax1.set_ylabel("Percentage (%)", fontsize=10, fontweight="bold")
        ax1.set_title(
            "QAOA Optimal Hit Probability & Feasibility Rate (3x3 Instances)\n"
            "Simulator: Qiskit Statevector / CVaR Comparison",
            fontsize=11, fontweight="bold", pad=10
        )
        ax1.set_xticks(x)
        ax1.set_xticklabels(qaoa_solvers, fontsize=9)
        ax1.legend(loc="upper left", framealpha=0.9, fontsize=8.5)
        ax1.grid(axis="y", linestyle="--", alpha=0.5)
        
        for r in rects1:
            h = r.get_height()
            ax1.text(r.get_x() + r.get_width()/2.0, h + 0.1, f"{h:.2f}%", ha="center", va="bottom", fontsize=7.5)
        for r in rects2:
            h = r.get_height()
            ax1.text(r.get_x() + r.get_width()/2.0, h + 0.1, f"{h:.2f}%", ha="center", va="bottom", fontsize=7.5)
            
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, "hit_prob_and_feasibility_vs_p.png"), bbox_inches="tight")
        plt.close(fig)
        
    # 3. Severity-Weighted Response Time Comparison (Violin / Box / Mean chart)
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=150)
    solvers_to_plot = [
        "Hungarian (Exact)",
        "Severity-First Greedy",
        "CVaR-QAOA (p=2, a=0.2)",
        "Nearest-Vehicle Greedy"
    ]
    df_resp = df_3x3[df_3x3["solver"].isin(solvers_to_plot)]
    resp_means = df_resp.groupby("solver")["cost"].mean().reindex(solvers_to_plot)
    resp_stds = df_resp.groupby("solver")["cost"].std().reindex(solvers_to_plot)
    
    ax.bar(
        solvers_to_plot, resp_means, yerr=resp_stds, capsize=4,
        color=["#2ecc71", "#9b59b6", "#e67e22", "#e74c3c"],
        edgecolor="black", linewidth=1.2
    )
    ax.set_ylabel("Severity-Weighted Delay (Minutes * Weight)", fontsize=10, fontweight="bold")
    ax.set_title(
        "Severity-Weighted Response Time Comparison\n"
        "Simulator: Mean +/- Std across test instances (Lower is better)",
        fontsize=11, fontweight="bold", pad=10
    )
    ax.set_xticks(range(len(solvers_to_plot)))
    ax.set_xticklabels(solvers_to_plot, rotation=20, ha="right", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    
    for i, (mean_v, std_v) in enumerate(zip(resp_means, resp_stds)):
        ax.text(i, mean_v / 2.0, f"{mean_v:.1f}", ha="center", va="center", color="white", fontweight="bold", fontsize=9)
        
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "weighted_response_time_comparison.png"), bbox_inches="tight")
    plt.close(fig)
    print(f"Benchmark figures saved to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Dispatch QAOA Benchmarks")
    parser.add_argument("--n3", type=int, default=30, help="Number of 3x3 instances")
    parser.add_argument("--n4", type=int, default=5, help="Number of 4x4 instances")
    parser.add_argument("--no-qaoa", action="store_true", help="Skip QAOA solvers for rapid testing")
    args = parser.parse_args()
    
    os.makedirs("results", exist_ok=True)
    df_results = run_benchmarks(
        n_instances_3x3=args.n3,
        n_instances_4x4=args.n4,
        run_qaoa_solvers=not args.no_qaoa
    )
    csv_path = "results/benchmark.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"Benchmark data written to {csv_path}")
    plot_benchmark_figures(df_results, output_dir="figures")
