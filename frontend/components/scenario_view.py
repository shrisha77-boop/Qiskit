"""
frontend/components/scenario_view.py
Renders Tab 1 – Scenario: city grid, incident table, travel-time matrix.
"""

import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List

SEV_COLORS = {
    1: "#2ecc71",   # green
    2: "#f1c40f",   # yellow
    3: "#e67e22",   # orange
    4: "#e74c3c",   # red
    5: "#8e44ad",   # purple
}
SEV_LABELS = {1: "Low", 2: "Moderate", 3: "Serious", 4: "Critical", 5: "Catastrophic"}


def render_city_grid(
    scenario: Dict[str, Any],
    assignments: Optional[List[Dict]] = None,
    title: str = "Synthetic City Grid",
) -> go.Figure:
    """
    Renders ambulances, incidents (colored by severity), and optional
    assignment arrows on a Plotly scatter figure.
    """
    fig = go.Figure()

    # ── Ambulances ──────────────────────────────────────────────
    amb_x = [pos[0] for pos in scenario["ambulances"].values()]
    amb_y = [pos[1] for pos in scenario["ambulances"].values()]
    amb_labels = list(scenario["ambulances"].keys())

    fig.add_trace(go.Scatter(
        x=amb_x, y=amb_y,
        mode="markers+text",
        marker=dict(symbol="square", size=18, color="#2980b9",
                    line=dict(width=2, color="white")),
        text=amb_labels,
        textposition="bottom center",
        textfont=dict(size=11, color="white"),
        name="Ambulance",
        hovertemplate="<b>%{text}</b><br>(%{x:.1f}, %{y:.1f})<extra></extra>",
    ))

    # ── Incidents (by severity group) ───────────────────────────
    for sev in range(1, 6):
        inc_group = [i for i in scenario["incidents"] if i["severity"] == sev]
        if not inc_group:
            continue
        ix = [i["position"][0] for i in inc_group]
        iy = [i["position"][1] for i in inc_group]
        labels = [f"{i['id']} (s={i['severity']})" for i in inc_group]

        fig.add_trace(go.Scatter(
            x=ix, y=iy,
            mode="markers+text",
            marker=dict(
                symbol="circle", size=20 + sev * 3,
                color=SEV_COLORS[sev],
                line=dict(width=2, color="white"),
            ),
            text=[i["id"] for i in inc_group],
            textposition="top center",
            textfont=dict(size=11, color="white"),
            name=f"Sev {sev} – {SEV_LABELS[sev]} (w={2**(sev-1) if sev > 1 else 1})",
            customdata=[[i["id"], i["severity"], i["weight"]] for i in inc_group],
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Severity: %{customdata[1]} (%{name})<br>"
                "Weight: %{customdata[2]}<br>"
                "(%{x:.1f}, %{y:.1f})<extra></extra>"
            ),
        ))

    # ── Assignment arrows ────────────────────────────────────────
    if assignments:
        amb_pos = scenario["ambulances"]
        inc_pos = {i["id"]: i["position"] for i in scenario["incidents"]}
        inc_sev = {i["id"]: i["severity"] for i in scenario["incidents"]}

        for asgn in assignments:
            a_id = asgn["ambulance_id"]
            i_id = asgn["incident_id"]
            if a_id not in amb_pos or i_id not in inc_pos:
                continue
            ax, ay = amb_pos[a_id]
            ix_, iy_ = inc_pos[i_id]
            sev = inc_sev.get(i_id, 1)
            color = "#e74c3c" if sev >= 4 else ("#e67e22" if sev == 3 else "#2ecc71")

            fig.add_annotation(
                x=ix_, y=iy_, ax=ax, ay=ay,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True,
                arrowhead=3, arrowsize=1.5, arrowwidth=2.5,
                arrowcolor=color,
            )
            fig.add_trace(go.Scatter(
                x=[(ax + ix_) / 2], y=[(ay + iy_) / 2],
                mode="text",
                text=[f"{asgn['travel_time']:.0f}m"],
                textfont=dict(size=9, color="#ecf0f1"),
                showlegend=False,
                hoverinfo="skip",
            ))

    fig.update_layout(
        title=dict(text=f"<b>{title}</b><br><sub>Synthetic scenario · Qiskit simulator · No live emergency data</sub>",
                   font=dict(size=14, color="#ecf0f1")),
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        font=dict(color="#ecf0f1"),
        xaxis=dict(title="X Coordinate (synthetic grid)", gridcolor="#334155", zeroline=False),
        yaxis=dict(title="Y Coordinate (synthetic grid)", gridcolor="#334155", zeroline=False),
        legend=dict(bgcolor="#1e293b", bordercolor="#334155", borderwidth=1),
        margin=dict(l=10, r=10, t=80, b=10),
        height=450,
    )
    return fig


def render_incident_table(scenario: Dict[str, Any]) -> None:
    """Shows incident details as a styled dataframe."""
    rows = []
    for inc in scenario["incidents"]:
        rows.append({
            "Incident": inc["id"],
            "X": round(inc["position"][0], 2),
            "Y": round(inc["position"][1], 2),
            "Severity": inc["severity"],
            "Severity Label": SEV_LABELS[inc["severity"]],
            "Priority Weight": inc["weight"],
        })
    df = pd.DataFrame(rows)
    styler = df.style
    style_fn = lambda v: f"color: {SEV_COLORS.get(v, '#ecf0f1')}" if isinstance(v, int) and 1 <= v <= 5 else ""
    if hasattr(styler, "map"):
        styled = styler.map(style_fn, subset=["Severity"])
    elif hasattr(styler, "applymap"):
        styled = styler.applymap(style_fn, subset=["Severity"])
    else:
        styled = styler

    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
    )


def render_travel_time_matrix(scenario: Dict[str, Any]) -> None:
    """Displays the travel-time matrix as a heatmap and dataframe."""
    T = np.array(scenario["travel_times"])
    amb_ids = list(scenario["ambulances"].keys())
    inc_ids = [i["id"] for i in scenario["incidents"]]

    df = pd.DataFrame(T, index=amb_ids, columns=inc_ids)
    df.index.name = "Ambulance \\ Incident"

    fig = go.Figure(go.Heatmap(
        z=T,
        x=inc_ids,
        y=amb_ids,
        colorscale="Blues",
        text=[[f"{v:.1f}" for v in row] for row in T],
        texttemplate="%{text}",
        hovertemplate="<b>%{y} → %{x}</b><br>Travel time: %{z:.1f} min<extra></extra>",
        colorbar=dict(title=dict(text="Minutes", font=dict(color="#ecf0f1")), tickfont=dict(color="#ecf0f1")),
    ))
    fig.update_layout(
        title="<b>Travel-Time Matrix (Manhattan distance, synthetic grid)</b>",
        paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
        font=dict(color="#ecf0f1"),
        xaxis=dict(title="Incident", gridcolor="#334155"),
        yaxis=dict(title="Ambulance", gridcolor="#334155"),
        height=280,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("Raw travel-time table"):
        st.dataframe(df.round(2), use_container_width=True)


def render_scenario_tab(scenario: Dict[str, Any], assignments: Optional[List[Dict]] = None) -> None:
    """Full Tab 1 renderer."""
    col1, col2 = st.columns([3, 2])

    with col1:
        subtitle = "Assignments shown" if assignments else "No assignments yet – run a solver first"
        fig = render_city_grid(scenario, assignments=assignments, title=f"City Grid – {subtitle}")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### 🚑 Incident List")
        render_incident_table(scenario)

        st.markdown(
            """
            #### ℹ️ Objective
            > The model multiplies each incident's **travel time** by its **severity weight**.
            > Delays to high-severity incidents carry a larger penalty.
            > The optimizer searches for the lowest-cost valid assignment.

            **Severity weight map:**
            | Sev | Label | Weight |
            |-----|-------|--------|
            | 1 | Low | 1 |
            | 2 | Moderate | 2 |
            | 3 | Serious | 4 |
            | 4 | Critical | 8 |
            | 5 | Catastrophic | 16 |
            """
        )

    st.markdown("---")
    st.markdown("#### 🗺️ Travel-Time Matrix")
    render_travel_time_matrix(scenario)
