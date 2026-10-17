"""
frontend/components/assignment_view.py
Renders Tab 2 – Assignments: side-by-side solver grid and assignment table.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import pandas as pd
from typing import Dict, Any, List, Optional

from frontend.components.scenario_view import SEV_COLORS, SEV_LABELS, render_city_grid


def _feasibility_badge(feasible: bool) -> str:
    if feasible:
        return "✅ Feasible"
    return "❌ Infeasible (constraint violated)"


def render_assignment_table(result: Dict[str, Any]) -> None:
    """Shows a styled assignment table for one solver result."""
    if "error" in result:
        st.error(f"Solver error: {result['error']}")
        return

    asgns = result["assignments"]
    if not asgns:
        st.warning("No assignments returned by this solver.")
        return

    rows = []
    for a in asgns:
        rows.append({
            "Ambulance": a["ambulance_id"],
            "Incident": a["incident_id"],
            "Severity": a["severity"],
            "Label": SEV_LABELS.get(a["severity"], ""),
            "Travel Time (min)": a["travel_time"],
            "Weight": a["weight"],
            "Weighted Cost": a["weighted_cost"],
        })
    df = pd.DataFrame(rows)

    st.dataframe(
        df.style.background_gradient(
            subset=["Weighted Cost"], cmap="YlOrRd"
        ).format({"Travel Time (min)": "{:.2f}", "Weighted Cost": "{:.2f}"}),
        use_container_width=True,
        hide_index=True,
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Weighted Cost", f"{result['weighted_cost']:.2f}")
    col2.metric("Feasibility", _feasibility_badge(result["feasible"]))
    col3.metric("Runtime", f"{result['runtime_seconds']*1000:.1f} ms")

    if result.get("approximation_ratio") is not None:
        ar = result["approximation_ratio"]
        delta = f"+{(ar-1)*100:.1f}% above optimal" if ar > 1 else "= optimal"
        st.metric("Approximation Ratio", f"{ar:.4f}", delta=delta,
                  delta_color="inverse")

    if not result["feasible"]:
        st.warning(
            "⚠️ This result is **infeasible** — at least one constraint is violated. "
            "Costs shown are raw QUBO energies, not valid dispatch costs."
        )


def render_assignments_tab(
    scenario: Dict[str, Any],
    classical_results: Dict[str, Dict],
    qaoa_result: Optional[Dict] = None,
) -> None:
    """Full Tab 2 renderer."""
    all_results = {**classical_results}
    if qaoa_result and "error" not in qaoa_result:
        all_results[qaoa_result["solver_name"]] = qaoa_result

    if not all_results:
        st.info("Run solvers from the sidebar to see assignments.")
        return

    # ── Solver selector ──────────────────────────────────────────
    solver_names = list(all_results.keys())
    selected = st.selectbox("Select solver to inspect:", solver_names)
    result = all_results[selected]

    if "error" in result:
        st.error(f"Solver failed: {result['error']}")
        return

    # ── Grid with assignments ────────────────────────────────────
    st.markdown(f"#### 🗺️ Assignment Map — {selected}")
    fig = render_city_grid(
        scenario,
        assignments=result["assignments"],
        title=f"{selected}  |  Cost: {result['weighted_cost']:.1f}  |  {_feasibility_badge(result['feasible'])}",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Assignment table ─────────────────────────────────────────
    st.markdown(f"#### 📋 Assignment Table — {selected}")
    render_assignment_table(result)

    # ── Before / After comparison ────────────────────────────────
    st.markdown("---")
    st.markdown("#### ⚖️ Side-by-Side Comparison")
    baseline_options = [n for n in solver_names if n != selected]
    if baseline_options:
        baseline = st.selectbox("Compare with:", baseline_options, key="baseline_sel")
        base_res = all_results[baseline]

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"**{baseline}**")
            if "error" not in base_res:
                b_fig = render_city_grid(
                    scenario,
                    assignments=base_res["assignments"],
                    title=f"{baseline}  Cost={base_res['weighted_cost']:.1f}",
                )
                st.plotly_chart(b_fig, use_container_width=True)
                render_assignment_table(base_res)
            else:
                st.error(base_res["error"])

        with col_b:
            st.markdown(f"**{selected}**")
            s_fig = render_city_grid(
                scenario,
                assignments=result["assignments"],
                title=f"{selected}  Cost={result['weighted_cost']:.1f}",
            )
            st.plotly_chart(s_fig, use_container_width=True)
            render_assignment_table(result)
