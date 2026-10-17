"""
frontend/components/quantum_view.py
Renders Tab 4 – Quantum Lab: pipeline diagram, QUBO/Ising summary,
circuit drawing, measurement probability chart, and parameter values.
"""

import plotly.graph_objects as go
import streamlit as st
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional


# ─────────────────────────────────────────────────────────────
# Pipeline flow diagram
# ─────────────────────────────────────────────────────────────

_PIPELINE_STEPS = [
    ("Dispatch Scenario", "Travel times, severities, constraints"),
    ("QUBO", "Assignment costs + penalty terms → Q matrix"),
    ("Ising Hamiltonian", "QUBO rewritten with Pauli-Z operators"),
    ("QAOA Circuit", "Parameterized cost + mixer unitaries"),
    ("Simulator Measurements", "Statevector probabilities → sampled bitstrings"),
    ("Decode & Validate", "Bitstring → assignments; check feasibility"),
]

def _pipeline_figure() -> go.Figure:
    n = len(_PIPELINE_STEPS)
    x = list(range(n))
    labels = [s[0] for s in _PIPELINE_STEPS]
    descs  = [s[1] for s in _PIPELINE_STEPS]

    fig = go.Figure()
    for i, (lab, desc) in enumerate(_PIPELINE_STEPS):
        fig.add_trace(go.Scatter(
            x=[i], y=[0],
            mode="markers+text",
            marker=dict(size=54, color="#1d4ed8", line=dict(width=2, color="#60a5fa")),
            text=[f"<b>{i+1}</b>"],
            textfont=dict(size=13, color="white"),
            textposition="middle center",
            hovertemplate=f"<b>{lab}</b><br>{desc}<extra></extra>",
            showlegend=False,
        ))
        fig.add_annotation(
            x=i, y=-0.22,
            text=f"<b>{lab}</b><br><span style='font-size:10px;color:#94a3b8'>{desc}</span>",
            showarrow=False,
            font=dict(size=10, color="#e2e8f0"),
            align="center",
        )
        if i < n - 1:
            fig.add_annotation(
                x=i + 0.5, y=0,
                text="→", showarrow=False,
                font=dict(size=22, color="#60a5fa"),
            )

    fig.update_layout(
        paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
        xaxis=dict(visible=False, range=[-0.5, n - 0.5]),
        yaxis=dict(visible=False, range=[-0.6, 0.4]),
        height=220,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig


# ─────────────────────────────────────────────────────────────
# QUBO/Ising summary
# ─────────────────────────────────────────────────────────────

def _render_qubo_summary(qubo_info: Dict[str, Any]) -> None:
    cols = st.columns(5)
    cols[0].metric("Ambulances (N)", qubo_info["N"])
    cols[1].metric("Incidents (N)", qubo_info["N"])
    cols[2].metric("Binary Variables", qubo_info["num_vars"])
    cols[3].metric("Qubits", qubo_info["num_qubits"])
    cols[4].metric("Pauli Terms", qubo_info["n_pauli_terms"])

    st.markdown(
        f"**Penalty weight P** = `{qubo_info['P']:.2f}`  "
        f"(= 1.5 × max cost, ensures infeasible states always cost more than feasible ones)\n\n"
        f"**QUBO offset** = `{qubo_info['offset']:.2f}`  "
        f"(constant term from constraint expansion)"
    )


# ─────────────────────────────────────────────────────────────
# Measurement probability bar chart
# ─────────────────────────────────────────────────────────────

def _render_probability_chart(qaoa_raw: Dict[str, Any], top_k: int = 14) -> None:
    probs     = qaoa_raw["exact_probs"]
    energies  = qaoa_raw["diag_energies"]
    opt_cost  = qaoa_raw["optimal_cost"]
    D = int(np.log2(len(probs)))
    N = int(np.sqrt(D))

    from solvers.exact import is_bitstring_feasible
    top_idx = np.argsort(-probs)[:top_k]

    labels, prob_vals, bar_colors, hover_texts = [], [], [], []
    for idx in top_idx:
        b_str = format(idx, f"0{D}b")
        p     = probs[idx]
        e     = energies[idx]
        x_vec = np.array([int(b_str[D - 1 - k]) for k in range(D)], dtype=int)
        feas  = is_bitstring_feasible(x_vec, N)
        is_opt = abs(e - opt_cost) < 1e-3

        labels.append(b_str)
        prob_vals.append(float(p))
        bar_colors.append(
            "#27ae60" if is_opt else ("#2980b9" if feas else "#64748b")
        )
        hover_texts.append(
            f"<b>{b_str}</b><br>"
            f"Prob: {p*100:.2f}%<br>"
            f"Energy: {e:.2f}<br>"
            f"{'✅ Optimal' if is_opt else ('✅ Feasible' if feas else '❌ Infeasible')}"
        )

    fig = go.Figure(go.Bar(
        x=list(range(len(labels))),
        y=prob_vals,
        marker_color=bar_colors,
        marker_line_color="white", marker_line_width=1,
        text=[f"{p*100:.1f}%" for p in prob_vals],
        textposition="outside",
        textfont=dict(size=8, color="#ecf0f1"),
        hovertext=hover_texts,
        hoverinfo="text",
    ))
    fig.update_layout(
        paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
        font=dict(color="#ecf0f1"),
        xaxis=dict(
            tickvals=list(range(len(labels))),
            ticktext=labels,
            tickangle=-45, tickfont=dict(size=7),
            gridcolor="#334155",
        ),
        yaxis=dict(title="Measurement Probability", gridcolor="#334155"),
        height=380,
        margin=dict(l=10, r=10, t=30, b=80),
        title=dict(
            text=(
                f"<b>QAOA Measurement Probabilities (p={qaoa_raw['p']})</b><br>"
                f"<sub>Simulator: Qiskit Statevector | "
                f"🟢 Optimal &nbsp; 🔵 Feasible &nbsp; ⚫ Infeasible</sub>"
            ),
            font=dict(size=13, color="#ecf0f1"),
        ),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Metric summary
    c1, c2, c3 = st.columns(3)
    c1.metric("Optimal-Hit Probability",
              f"{qaoa_raw['optimal_hit_prob']*100:.2f}%",
              help="Fraction of total probability mass on optimal bitstrings")
    c2.metric("Feasibility Rate",
              f"{qaoa_raw['feasibility_rate']*100:.2f}%",
              help="Fraction of probability mass on feasible assignments")
    c3.metric("Approx Ratio",
              f"{qaoa_raw['approx_ratio']:.4f}" if qaoa_raw["approx_ratio"] else "N/A")


# ─────────────────────────────────────────────────────────────
# Top measured bitstrings table
# ─────────────────────────────────────────────────────────────

def _render_bitstring_table(qaoa_raw: Dict[str, Any], scenario: Dict[str, Any]) -> None:
    from solvers.exact import is_bitstring_feasible

    probs    = qaoa_raw["exact_probs"]
    energies = qaoa_raw["diag_energies"]
    D = int(np.log2(len(probs)))
    N = int(np.sqrt(D))
    top_idx  = np.argsort(-probs)[:8]

    amb_ids = list(scenario["ambulances"].keys())
    inc_ids = [inc["id"] for inc in scenario["incidents"]]

    rows = []
    for idx in top_idx:
        b_str = format(idx, f"0{D}b")
        p     = float(probs[idx])
        e     = float(energies[idx])
        x_vec = np.array([int(b_str[D - 1 - k]) for k in range(D)], dtype=int)
        feas  = is_bitstring_feasible(x_vec, N)

        # Decode assignment
        mat = x_vec.reshape((N, N))
        decoded = []
        for v in range(min(N, len(amb_ids))):
            for i_col in range(min(N, len(inc_ids))):
                if mat[v, i_col] == 1:
                    decoded.append(f"{amb_ids[v]}→{inc_ids[i_col]}")

        rows.append({
            "Bitstring": b_str,
            "Probability": f"{p*100:.2f}%",
            "Energy": round(e, 2),
            "Feasible": "✅" if feas else "❌",
            "Decoded": ", ".join(decoded) if decoded else "—",
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.caption(
        "Bit ordering follows Qiskit convention: qubit 0 is the **rightmost** character. "
        "Energy = QUBO energy (includes penalty terms). "
        "A high-probability bitstring is not necessarily the lowest-cost feasible assignment."
    )


# ─────────────────────────────────────────────────────────────
# Main tab renderer
# ─────────────────────────────────────────────────────────────

def render_quantum_tab(
    scenario: Dict[str, Any],
    qaoa_result: Optional[Dict] = None,
    qubo_info: Optional[Dict] = None,
) -> None:
    """Full Tab 4 renderer."""

    # A. Pipeline diagram
    st.markdown("#### ⚙️ Optimization Pipeline")
    st.plotly_chart(_pipeline_figure(), use_container_width=True)

    # B. Model summary
    st.markdown("---")
    st.markdown("#### 🧮 QUBO / Ising Model Summary")
    if qubo_info:
        _render_qubo_summary(qubo_info)
    else:
        st.info("Generate a scenario to see the QUBO model summary.")

    # C & D. QAOA-specific content
    st.markdown("---")
    if qaoa_result is None or "error" in (qaoa_result or {}):
        st.info("Run QAOA from the sidebar to see circuit details and measurement results.")
        if qaoa_result and "error" in qaoa_result:
            st.error(f"QAOA error: {qaoa_result['error']}")
    else:
        meta = qaoa_result.get("metadata", {})
        qaoa_raw = qaoa_result.get("_qaoa_raw", {})

        # Model summary update with QAOA specifics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("QAOA Depth (p)", meta.get("p", "—"))
        c2.metric("Qubits", meta.get("num_qubits", "—"))
        c3.metric("Optimizer", meta.get("optimizer", "COBYLA"))
        c4.metric("Backend", "Statevector Simulator")

        # C. Circuit diagram
        st.markdown("#### 🔬 QAOA Circuit Diagram")
        circ = qaoa_raw.get("circuit")
        if circ is not None:
            try:
                import matplotlib.pyplot as plt
                fig_circ, ax = plt.subplots(figsize=(min(circ.num_qubits * 1.5, 18), 4))
                circ.draw("mpl", ax=ax, style="iqp", fold=80)
                ax.set_title(
                    f"QAOA Ansatz Circuit (p={meta.get('p',1)}, {meta.get('num_qubits','?')} qubits)\n"
                    "Simulator: Qiskit Statevector – not a quantum computer photograph",
                    fontsize=9,
                )
                fig_circ.tight_layout()
                st.pyplot(fig_circ, use_container_width=True)
                plt.close(fig_circ)
            except Exception as e:
                st.warning(f"Circuit diagram not available in this environment: {e}")
                st.markdown(
                    "**Circuit structure:** Hadamard layer → "
                    f"p={meta.get('p',1)} × (Cost unitary exp(-iγH_C) + X-mixer exp(-iβΣX_k)) → Measure"
                )
        else:
            st.warning("Circuit object not available.")

        # D. Measurement results
        if qaoa_raw:
            st.markdown("#### 📊 Measurement Probabilities")
            _render_probability_chart(qaoa_raw, top_k=14)

            st.markdown("#### 🔢 Top Measured Bitstrings")
            _render_bitstring_table(qaoa_raw, scenario)

            # E. Optimized parameters
            best_params = qaoa_raw.get("best_params")
            if best_params is not None:
                p_val = meta.get("p", 1)
                st.markdown("#### 🎛️ Optimized QAOA Parameters")
                half = len(best_params) // 2
                betas  = best_params[:half]
                gammas = best_params[half:]
                param_df = pd.DataFrame({
                    "Layer": list(range(1, p_val + 1)),
                    "β (mixer angle)": [round(b, 5) for b in betas],
                    "γ (cost angle)":  [round(g, 5) for g in gammas],
                })
                st.dataframe(param_df, use_container_width=True, hide_index=True)
                st.caption(
                    "β and γ are tuned by COBYLA to minimize the QAOA energy expectation "
                    "(or CVaR objective). These are the parameters bound into the final circuit."
                )

    # F. Honest role of quantum
    st.markdown("---")
    st.info(
        "**Quantum's role in this project:**\n\n"
        "Severity weights and assignment constraints define the dispatch objective; "
        "quantum computing does not decide which patient is more important. "
        "QAOA uses a parameterized quantum-circuit model to explore candidate bitstrings, "
        "while a classical optimizer (COBYLA) tunes the circuit parameters. "
        "**This demo uses a Qiskit Statevector Simulator and does not establish quantum "
        "advantage over classical algorithms.** "
        "The base one-to-one assignment problem is solvable in polynomial time by the Hungarian algorithm."
    )
