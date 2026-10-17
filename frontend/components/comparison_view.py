"""
frontend/components/comparison_view.py
Renders Tab 3 – Solver Comparison: bar charts, metric cards, runtime plot.
"""

import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List

SOLVER_COLORS = {
    "Hungarian (Exact)":     "#2ecc71",
    "Nearest-Vehicle Greedy":"#e74c3c",
    "Severity-First Greedy": "#9b59b6",
    "Simulated Annealing":   "#3498db",
}
QAOA_COLOR = "#f39c12"


def _color_for(name: str) -> str:
    for k, v in SOLVER_COLORS.items():
        if k.lower() in name.lower():
            return v
    return QAOA_COLOR


def render_comparison_tab(
    scenario: Dict[str, Any],
    classical_results: Dict[str, Dict],
    qaoa_result: Optional[Dict] = None,
) -> None:
    """Full Tab 3 renderer."""
    all_results = {**classical_results}
    if qaoa_result and "error" not in qaoa_result:
        all_results[qaoa_result["solver_name"]] = qaoa_result

    if not all_results:
        st.info("Run classical solvers and/or QAOA to see the comparison.")
        return

    # Build summary rows
    rows = []
    for name, res in all_results.items():
        if "error" in res:
            rows.append({
                "Solver": name, "Weighted Cost": None,
                "Feasible": "Error", "Runtime (ms)": None,
                "Approx Ratio": None,
            })
        else:
            rows.append({
                "Solver": name,
                "Weighted Cost": res["weighted_cost"],
                "Feasible": "✅ Yes" if res["feasible"] else "❌ No",
                "Runtime (ms)": round(res["runtime_seconds"] * 1000, 2),
                "Approx Ratio": res.get("approximation_ratio"),
            })

    df = pd.DataFrame(rows)
    names = [r["Solver"] for r in rows]
    costs = [r.get("Weighted Cost") for r in rows]
    runtimes = [r.get("Runtime (ms)") for r in rows]
    ratios = [r.get("Approx Ratio") for r in rows]
    colors = [_color_for(n) for n in names]

    # ── Metric cards ─────────────────────────────────────────────
    st.markdown("#### 📊 Solver Metrics")
    cols = st.columns(min(len(all_results), 4))
    for idx, (name, res) in enumerate(all_results.items()):
        c = cols[idx % 4]
        if "error" in res:
            c.error(f"**{name}**\nFailed")
            continue
        short = name.split("(")[0].strip()[:18]
        c.metric(
            label=short,
            value=f"{res['weighted_cost']:.1f}",
            delta=f"ratio {res['approximation_ratio']:.3f}" if res.get("approximation_ratio") else "ratio N/A",
            delta_color="inverse",
            help=f"Runtime: {res['runtime_seconds']*1000:.1f} ms | {'Feasible' if res['feasible'] else 'INFEASIBLE'}",
        )

    st.markdown("---")

    # ── Weighted cost bar chart ───────────────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### 💰 Weighted Cost by Solver (lower is better)")
        valid_idx = [i for i, c in enumerate(costs) if c is not None]
        if valid_idx:
            fig_cost = go.Figure(go.Bar(
                x=[names[i] for i in valid_idx],
                y=[costs[i] for i in valid_idx],
                marker_color=[colors[i] for i in valid_idx],
                marker_line_color="white",
                marker_line_width=1.5,
                text=[f"{costs[i]:.1f}" for i in valid_idx],
                textposition="outside",
                textfont=dict(color="#ecf0f1"),
                hovertemplate="<b>%{x}</b><br>Cost: %{y:.2f}<extra></extra>",
            ))
            fig_cost.update_layout(
                paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
                font=dict(color="#ecf0f1"),
                xaxis=dict(gridcolor="#334155", tickangle=-30),
                yaxis=dict(gridcolor="#334155", title="Weighted Cost"),
                margin=dict(l=10, r=10, t=20, b=80),
                height=350,
                showlegend=False,
            )
            st.plotly_chart(fig_cost, use_container_width=True)
        # Flag infeasible
        infeasible = [n for n, r in all_results.items() if "error" not in r and not r["feasible"]]
        if infeasible:
            st.warning(f"⚠️ Infeasible results (constraint violated): {', '.join(infeasible)}")

    # ── Runtime bar chart ─────────────────────────────────────────
    with col_b:
        st.markdown("#### ⏱️ Runtime by Solver")
        valid_rt = [i for i, r in enumerate(runtimes) if r is not None]
        if valid_rt:
            rt_vals = [runtimes[i] for i in valid_rt]
            use_log = max(rt_vals) / (min(rt_vals) + 1e-9) > 100
            fig_rt = go.Figure(go.Bar(
                x=[names[i] for i in valid_rt],
                y=rt_vals,
                marker_color=[colors[i] for i in valid_rt],
                marker_line_color="white",
                marker_line_width=1.5,
                text=[f"{r:.1f}" for r in rt_vals],
                textposition="outside",
                textfont=dict(color="#ecf0f1"),
                hovertemplate="<b>%{x}</b><br>Runtime: %{y:.2f} ms<extra></extra>",
            ))
            fig_rt.update_layout(
                paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
                font=dict(color="#ecf0f1"),
                xaxis=dict(gridcolor="#334155", tickangle=-30),
                yaxis=dict(
                    gridcolor="#334155",
                    title="Runtime (ms)" + (" [log scale]" if use_log else ""),
                    type="log" if use_log else "linear",
                ),
                margin=dict(l=10, r=10, t=20, b=80),
                height=350,
                showlegend=False,
            )
            st.plotly_chart(fig_rt, use_container_width=True)
            if use_log:
                st.caption("Log scale used because runtimes span more than 2 orders of magnitude.")

    # ── Approximation ratio chart ────────────────────────────────
    ratio_valid = [(names[i], ratios[i]) for i in range(len(names)) if ratios[i] is not None]
    if ratio_valid:
        st.markdown("#### 📐 Approximation Ratio (vs Hungarian Exact Optimum, lower = better)")
        r_names, r_vals = zip(*ratio_valid)
        fig_ar = go.Figure()
        fig_ar.add_trace(go.Bar(
            x=list(r_names), y=list(r_vals),
            marker_color=[_color_for(n) for n in r_names],
            marker_line_color="white", marker_line_width=1.5,
            text=[f"{v:.4f}" for v in r_vals],
            textposition="outside",
            textfont=dict(color="#ecf0f1"),
            hovertemplate="<b>%{x}</b><br>Ratio: %{y:.4f}<extra></extra>",
        ))
        fig_ar.add_hline(y=1.0, line_dash="dash", line_color="#95a5a6",
                          annotation_text="Optimal (1.00)",
                          annotation_font_color="#95a5a6")
        fig_ar.update_layout(
            paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
            font=dict(color="#ecf0f1"),
            xaxis=dict(gridcolor="#334155", tickangle=-30),
            yaxis=dict(gridcolor="#334155", title="Approx Ratio (found / optimal)"),
            margin=dict(l=10, r=10, t=20, b=80),
            height=320,
            showlegend=False,
        )
        st.plotly_chart(fig_ar, use_container_width=True)
        st.caption(
            "Approximation ratio = found cost ÷ Hungarian optimal. "
            "Shown only for feasible results with a valid reference. "
            "Hungarian is the exact ground truth for this base assignment problem."
        )

    # ── Summary table ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📋 Full Results Table")
    st.dataframe(
        df.style.format({"Weighted Cost": "{:.2f}", "Runtime (ms)": "{:.2f}",
                          "Approx Ratio": lambda x: f"{x:.4f}" if x else "—"},
                          na_rep="—"),
        use_container_width=True,
        hide_index=True,
    )
