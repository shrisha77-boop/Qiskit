"""
run_all.py
One-command reproducibility script for the Severity-Aware Emergency Dispatch QAOA project.

Usage:
    python run_all.py               # Full run (all solvers including QAOA)
    python run_all.py --quick       # Quick run (skip QAOA, classical solvers only)
    python run_all.py --no-qaoa-4x4 # Skip QAOA on 4x4 (faster benchmark)

Generates all figures in figures/ and tables in results/.
Expected runtime: 8-15 minutes (full), < 2 minutes (--quick).
"""

import os
import sys
import time
import argparse
import traceback

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for reproducibility
import matplotlib.pyplot as plt

from data.instance import (
    generate_instance,
    generate_greedy_failure_instance,
    plot_dispatch_map
)
from model.qubo import build_qubo, evaluate_qubo_energy
from solvers.exact import solve_hungarian, solve_brute_force_qubo
from solvers.greedy import solve_nearest_vehicle, solve_severity_first_greedy
from solvers.sa import solve_simulated_annealing
from solvers.qaoa import run_qaoa, draw_and_save_qaoa_circuit, plot_qaoa_probabilities
from experiments.benchmark import run_benchmarks, plot_benchmark_figures
from experiments.penalty_sweep import run_penalty_sweep, plot_penalty_sensitivity
from experiments.dynamic_sim import run_dynamic_experiment, plot_dynamic_simulation


# ─────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────

def section(title: str) -> None:
    """Print a clearly delimited section header."""
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  {title}")
    print(f"{bar}")


def ensure_dirs() -> None:
    for d in ["figures", "results", "data", "model", "solvers", "experiments", "tests", "notebooks"]:
        os.makedirs(d, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# Step 1: Greedy-Failure Figure
# ─────────────────────────────────────────────────────────────

def step_greedy_failure_figure() -> None:
    """Generates the greedy-failure motivation figure."""
    section("Step 1: Greedy-Failure Demonstration Figure")

    inst = generate_greedy_failure_instance()
    h_res = solve_hungarian(inst.cost_matrix)
    nv_res = solve_nearest_vehicle(inst.travel_time_matrix, inst.weights)
    sf_res = solve_severity_first_greedy(inst.travel_time_matrix, inst.severities, inst.weights)

    print(f"  Hungarian (optimal):    cost = {h_res['total_cost']:.1f}")
    print(f"  Nearest-Vehicle greedy: cost = {nv_res['total_cost']:.1f}  "
          f"(ratio: {nv_res['total_cost']/h_res['total_cost']:.2f}x)")
    print(f"  Severity-First greedy:  cost = {sf_res['total_cost']:.1f}  "
          f"(ratio: {sf_res['total_cost']/h_res['total_cost']:.2f}x)")

    # Side-by-side plot: left = greedy (bad), right = optimal
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=130)

    # Left: Nearest-vehicle (bad)
    plt.sca(axes[0])
    fig_l = plot_dispatch_map(
        inst,
        assignments=nv_res["assignments"],
        title=f"Nearest-Vehicle Greedy (POOR)\nWeighted Cost = {nv_res['total_cost']:.1f}",
        subtitle="Sends nearest ambulance to minor call — critical case waits!",
        figsize=(7, 6)
    )
    plt.close(fig_l)  # redrawn into axes[0] directly below
    axes[0] = plot_dispatch_map(
        inst,
        assignments=nv_res["assignments"],
        title=f"Nearest-Vehicle Greedy (POOR)  Cost={nv_res['total_cost']:.1f}",
        subtitle="Near ambulance dispatched to minor call — critical waits!",
    ).axes[0] if False else axes[0]  # keep axes[0] as is; draw into figure inline

    # Use plot_dispatch_map but capture into the subplot axes
    for ax, res, title, color in [
        (axes[0], nv_res, f"Nearest-Vehicle Greedy  (Cost={nv_res['total_cost']:.1f})", "crimson"),
        (axes[1], h_res,  f"Hungarian Optimal       (Cost={h_res['total_cost']:.1f})",  "darkgreen")
    ]:
        ax.set_facecolor("#f8f9fa")
        ax.grid(True, linestyle="--", alpha=0.4)
        vp = inst.vehicle_positions
        ip = inst.incident_positions
        ax.scatter(vp[:, 0], vp[:, 1], s=200, c="#2980b9", marker="s",
                   edgecolors="black", linewidths=1.5, label="Ambulance", zorder=5)
        for v, (x, y) in enumerate(vp):
            ax.annotate(f"V{v}", (x, y), xytext=(0, -16), textcoords="offset points",
                        ha="center", fontsize=9, fontweight="bold")
        sev_colors = {1: "#2ecc71", 2: "#f1c40f", 3: "#e67e22", 4: "#e74c3c", 5: "#8e44ad"}
        for i, (x, y) in enumerate(ip):
            sev = inst.severities[i]
            ax.scatter(x, y, s=180+sev*25, c=sev_colors[sev], marker="o",
                       edgecolors="black", linewidths=1.5, zorder=6)
            ax.annotate(f"I{i}\n(s={sev})", (x, y), xytext=(0, 12),
                        textcoords="offset points", ha="center", fontsize=8, fontweight="bold")
        for v_idx, i_idx in res["assignments"]:
            vxy = vp[v_idx]; ixy = ip[i_idx]
            w = inst.weights[i_idx]
            arrow_color = "#c0392b" if w >= 8 else "#27ae60"
            ax.annotate("", xy=(ixy[0], ixy[1]), xytext=(vxy[0], vxy[1]),
                        arrowprops=dict(arrowstyle="->", color=arrow_color,
                                        lw=2.2, shrinkA=8, shrinkB=8), zorder=4)
            mx, my = (vxy[0]+ixy[0])/2, (vxy[1]+ixy[1])/2
            ax.text(mx, my, f"{inst.travel_time_matrix[v_idx,i_idx]:.0f}m",
                    fontsize=7.5, ha="center",
                    bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8))
        ax.set_title(title, fontsize=10, color=color, fontweight="bold")
        ax.set_xlim(-0.5, inst.grid_size + 0.5)
        ax.set_ylim(-0.5, inst.grid_size + 0.5)
        ax.set_aspect("equal")

    fig.suptitle(
        "Greedy Failure: Why Nearest-Vehicle Dispatch Fails\n"
        "Severity weights — Low(1)=1x, Medium(3)=4x, Critical(5)=16x",
        fontsize=12, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    fig.savefig("figures/greedy_failure.png", bbox_inches="tight")
    plt.close(fig)
    print("  [OK] figures/greedy_failure.png saved.")


# ─────────────────────────────────────────────────────────────
# Step 2: QUBO Validation
# ─────────────────────────────────────────────────────────────

def step_qubo_validation() -> None:
    """Runs QUBO validation tests and prints results."""
    section("Step 2: QUBO Validation (QUBO min == Hungarian on 10 random instances)")
    passed = 0
    failed = 0
    for seed in range(10):
        inst = generate_instance(3, 3, seed=seed)
        Q, offset, P = build_qubo(inst.cost_matrix)
        h_res = solve_hungarian(inst.cost_matrix)
        bf_res = solve_brute_force_qubo(Q, offset, 3)

        h_cost = h_res["total_cost"]
        bf_cost = bf_res["best_feasible_energy"] if bf_res["best_feasible_energy"] is not None else bf_res["best_energy"]
        match = abs(h_cost - bf_cost) < 1e-6

        if match:
            passed += 1
        else:
            failed += 1
            print(f"  [FAIL] seed={seed}: Hungarian={h_cost:.4f}, BruteForce={bf_cost:.4f}")

    print(f"  QUBO Validation: {passed}/10 passed, {failed}/10 failed.")
    if failed == 0:
        print("  [OK] All QUBO validations passed.")


# ─────────────────────────────────────────────────────────────
# Step 3: QAOA Circuit Diagram
# ─────────────────────────────────────────────────────────────

def step_qaoa_circuit_diagram() -> None:
    """Generates and saves the QAOA circuit diagram."""
    section("Step 3: QAOA Circuit Diagram")
    try:
        draw_and_save_qaoa_circuit(p=2, save_path="figures/qaoa_circuit.png")
        print("  [OK] figures/qaoa_circuit.png saved.")
    except Exception as e:
        print(f"  [WARN] Could not save circuit diagram (matplotlib backend issue): {e}")


# ─────────────────────────────────────────────────────────────
# Step 4: Single QAOA Showcase Run
# ─────────────────────────────────────────────────────────────

def step_qaoa_showcase(run_qaoa_flag: bool = True) -> None:
    """Runs QAOA p=1,2 on a fixed 3x3 instance and saves probability plots."""
    section("Step 4: QAOA Showcase — 3x3 Instance (seed=42)")

    inst = generate_instance(3, 3, seed=42)
    Q, offset, P = build_qubo(inst.cost_matrix)
    h_res = solve_hungarian(inst.cost_matrix)
    opt_cost = h_res["total_cost"]

    print(f"  Instance: 3 vehicles, 3 incidents")
    print(f"  Hungarian optimal cost: {opt_cost:.2f}")
    print(f"  QUBO penalty P: {P:.2f}")

    if not run_qaoa_flag:
        print("  [SKIP] QAOA skipped (--quick mode).")
        return

    results_summary = []
    for p_val in [1, 2]:
        print(f"  Running QAOA p={p_val} (3 restarts, 80 iters)...", end=" ", flush=True)
        t0 = time.perf_counter()
        res = run_qaoa(
            Q, offset, N=3, p=p_val,
            cvar_alpha=None, shots=4096, seed=42,
            n_restarts=3, maxiter=80,
            optimal_cost=opt_cost
        )
        elapsed = time.perf_counter() - t0
        print(f"done ({elapsed:.1f}s)")

        print(f"    Approx ratio:        {res['approx_ratio']:.4f}")
        print(f"    Optimal-hit prob:    {res['optimal_hit_prob']*100:.2f}%")
        print(f"    Feasibility rate:    {res['feasibility_rate']*100:.2f}%")
        print(f"    Best feasible cost:  {res['best_feasible_energy']}")

        # Save probability plot
        fig = plot_qaoa_probabilities(res, top_k=12,
                                       save_path=f"figures/qaoa_p{p_val}_probs.png")
        plt.close(fig)
        print(f"  [OK] figures/qaoa_p{p_val}_probs.png saved.")

        results_summary.append({
            "solver": f"QAOA (p={p_val})",
            "approx_ratio": res["approx_ratio"],
            "optimal_hit_prob": res["optimal_hit_prob"],
            "feasibility_rate": res["feasibility_rate"],
            "runtime_sec": res["runtime_sec"]
        })

    # CVaR run
    print("  Running CVaR-QAOA p=2 alpha=0.20...", end=" ", flush=True)
    t0 = time.perf_counter()
    res_cvar = run_qaoa(
        Q, offset, N=3, p=2,
        cvar_alpha=0.20, shots=4096, seed=42,
        n_restarts=3, maxiter=80,
        optimal_cost=opt_cost
    )
    elapsed = time.perf_counter() - t0
    print(f"done ({elapsed:.1f}s)")
    fig = plot_qaoa_probabilities(res_cvar, top_k=12,
                                   save_path="figures/qaoa_cvar_probs.png")
    plt.close(fig)
    print("  [OK] figures/qaoa_cvar_probs.png saved.")

    results_summary.append({
        "solver": "CVaR-QAOA (p=2, a=0.2)",
        "approx_ratio": res_cvar["approx_ratio"],
        "optimal_hit_prob": res_cvar["optimal_hit_prob"],
        "feasibility_rate": res_cvar["feasibility_rate"],
        "runtime_sec": res_cvar["runtime_sec"]
    })

    # Save showcase results table
    df_show = pd.DataFrame(results_summary)
    df_show.to_csv("results/qaoa_showcase.csv", index=False)
    print("  [OK] results/qaoa_showcase.csv saved.")
    print()
    print(df_show.to_string(index=False))


# ─────────────────────────────────────────────────────────────
# Step 5: Full Benchmark
# ─────────────────────────────────────────────────────────────

def step_benchmark(run_qaoa_flag: bool = True, n3: int = 30, n4: int = 5) -> None:
    """Runs the full benchmark and saves results/benchmark.csv."""
    section(f"Step 5: Full Benchmark ({n3} x 3x3, {n4} x 4x4 instances)")
    df = run_benchmarks(
        n_instances_3x3=n3,
        n_instances_4x4=n4,
        run_qaoa_solvers=run_qaoa_flag,
        seed_offset=100
    )
    df.to_csv("results/benchmark.csv", index=False)
    print("  [OK] results/benchmark.csv saved.")

    # Print summary table
    summary = (
        df.groupby(["size", "solver"])[["approx_ratio", "cost", "runtime_sec"]]
        .agg({"approx_ratio": ["mean", "std"], "cost": "mean", "runtime_sec": "mean"})
        .round(4)
    )
    print("\n  Benchmark Summary:")
    print(summary.to_string())

    plot_benchmark_figures(df, output_dir="figures")
    print("  [OK] Benchmark figures saved to figures/.")


# ─────────────────────────────────────────────────────────────
# Step 6: Penalty Sensitivity Sweep
# ─────────────────────────────────────────────────────────────

def step_penalty_sweep(run_qaoa_flag: bool = True) -> None:
    """Runs penalty sensitivity sweep and saves figure."""
    section("Step 6: Penalty Sensitivity Sweep")
    try:
        # run_penalty_sweep always runs QAOA; skip this step in quick mode
        if not run_qaoa_flag:
            print("  [SKIP] Penalty sweep skipped (--quick mode, requires QAOA).")
            return
        df_pen = run_penalty_sweep(
            n_instances=5,
            seed_offset=200
        )
        df_pen.to_csv("results/penalty_sweep.csv", index=False)
        plot_penalty_sensitivity(df_pen, save_path="figures/penalty_sensitivity.png")
        print("  [OK] Penalty sweep done. results/penalty_sweep.csv saved.")
    except Exception as e:
        print(f"  [WARN] Penalty sweep failed: {e}")
        traceback.print_exc()


# ─────────────────────────────────────────────────────────────
# Step 7: Dynamic Dispatch Simulation
# ─────────────────────────────────────────────────────────────

def step_dynamic_sim(run_qaoa_flag: bool = True) -> None:
    """Runs rolling-horizon dynamic dispatch simulation."""
    section("Step 7: Dynamic Dispatch Simulation (Rolling Horizon)")
    try:
        df_dyn = run_dynamic_experiment(
            n_windows=8 if run_qaoa_flag else 5,
            seed=77
        )
        df_dyn.to_csv("results/dynamic_simulation.csv", index=False)
        plot_dynamic_simulation(df_dyn, save_path="figures/dynamic_dispatch_cumulative.png")
        print("  [OK] Dynamic simulation done. results/dynamic_simulation.csv saved.")
    except Exception as e:
        print(f"  [WARN] Dynamic simulation failed: {e}")
        traceback.print_exc()


# ─────────────────────────────────────────────────────────────
# Step 8: Final Summary
# ─────────────────────────────────────────────────────────────

def step_final_summary() -> None:
    """Prints a summary of all generated outputs."""
    section("Final Output Summary")

    print("\n  Generated figures:")
    for f in sorted(os.listdir("figures")):
        if f.endswith(".png"):
            size_kb = os.path.getsize(os.path.join("figures", f)) // 1024
            print(f"    figures/{f}  ({size_kb} KB)")

    print("\n  Generated results:")
    for f in sorted(os.listdir("results")):
        if f.endswith(".csv"):
            try:
                df_tmp = pd.read_csv(os.path.join("results", f))
                print(f"    results/{f}  ({len(df_tmp)} rows x {len(df_tmp.columns)} cols)")
            except Exception:
                print(f"    results/{f}")

    print("\n  All outputs are labelled 'simulator' (Qiskit Statevector / no real hardware).")
    print("  To reproduce: python run_all.py")
    print()


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce all figures and tables for Severity-Aware Emergency Dispatch QAOA"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Skip QAOA solvers (classical baseline only). Runs in < 2 minutes."
    )
    parser.add_argument(
        "--n3", type=int, default=30,
        help="Number of 3x3 benchmark instances (default: 30)"
    )
    parser.add_argument(
        "--n4", type=int, default=5,
        help="Number of 4x4 benchmark instances (default: 5)"
    )
    parser.add_argument(
        "--skip-dynamic", action="store_true",
        help="Skip dynamic dispatch simulation"
    )
    parser.add_argument(
        "--skip-penalty", action="store_true",
        help="Skip penalty sensitivity sweep"
    )
    args = parser.parse_args()

    run_qaoa_flag = not args.quick

    total_start = time.perf_counter()
    print("=" * 60)
    print("  Severity-Aware Emergency Dispatch -- QAOA")
    print("  IBM Qiskit Fall Fest 2026")
    print("  Reproducing all figures and tables")
    if args.quick:
        print("  Mode: QUICK (QAOA skipped)")
    print("=" * 60)

    ensure_dirs()

    step_greedy_failure_figure()
    step_qubo_validation()
    step_qaoa_circuit_diagram()
    step_qaoa_showcase(run_qaoa_flag=run_qaoa_flag)
    step_benchmark(run_qaoa_flag=run_qaoa_flag, n3=args.n3, n4=args.n4)

    if not args.skip_penalty:
        step_penalty_sweep(run_qaoa_flag=run_qaoa_flag)

    if not args.skip_dynamic:
        step_dynamic_sim(run_qaoa_flag=run_qaoa_flag)

    step_final_summary()

    total_elapsed = time.perf_counter() - total_start
    print(f"  Total runtime: {total_elapsed/60:.1f} minutes")
    print("=" * 60)


if __name__ == "__main__":
    main()
