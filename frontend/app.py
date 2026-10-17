"""
frontend/app.py
QuantumDispatch – Severity-Aware Emergency Dispatch Using QAOA
Streamlit dashboard connecting to backend solver modules.

Run from project root:
    python -m streamlit run frontend/app.py
"""

import sys
import os
import traceback

# ── Path setup: run from project root so backend imports work ──
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────
st.set_page_config(
    page_title="QuantumDispatch",
    page_icon="🚑",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS: dark premium theme ────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0f172a;
    color: #e2e8f0;
}
.stApp { background-color: #0f172a; }

/* Header */
.quantum-header {
    background: linear-gradient(135deg, #1e3a5f 0%, #0f172a 50%, #1a1a2e 100%);
    border: 1px solid #1d4ed8;
    border-radius: 12px;
    padding: 20px 28px 14px;
    margin-bottom: 18px;
}
.quantum-header h1 {
    font-size: 2rem; font-weight: 700; margin: 0;
    background: linear-gradient(90deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.quantum-header p { color: #94a3b8; margin: 4px 0 0; font-size: 0.92rem; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    border-right: 1px solid #1e40af;
}
section[data-testid="stSidebar"] label { color: #94a3b8 !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #4f46e5);
    color: white; border: none; border-radius: 8px;
    font-weight: 600; width: 100%; padding: 10px;
    transition: all 0.2s;
}
.stButton > button:hover { opacity: 0.85; transform: translateY(-1px); }
.stButton > button:active { transform: translateY(0); }

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background-color: #1e293b;
    border-radius: 10px; padding: 4px; gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background-color: transparent;
    color: #94a3b8; border-radius: 8px;
    font-weight: 600; padding: 8px 18px;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1d4ed8, #4f46e5) !important;
    color: white !important;
}

/* Metrics */
[data-testid="metric-container"] {
    background: #1e293b; border: 1px solid #334155;
    border-radius: 10px; padding: 12px;
}
[data-testid="stMetricValue"] { color: #60a5fa; font-weight: 700; }
[data-testid="stMetricLabel"] { color: #94a3b8; }

/* Info / warning boxes */
.stInfo { background: #1e3a5f; border-color: #2563eb; }
.stWarning { background: #2d1f06; border-color: #d97706; }
.stError { background: #2d0a0a; border-color: #dc2626; }
.stSuccess { background: #0d2d1e; border-color: #16a34a; }

/* DataFrames */
.stDataFrame { border-radius: 8px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Imports (after path is set) ───────────────────────────────
from frontend.services.dispatch_service import (
    build_scenario,
    run_all_classical,
    run_qaoa_solver,
    get_qubo_info,
    run_hungarian,
)
from frontend.components.scenario_view import render_scenario_tab
from frontend.components.assignment_view import render_assignments_tab
from frontend.components.comparison_view import render_comparison_tab
from frontend.components.quantum_view import render_quantum_tab


# ─────────────────────────────────────────────────────────────
# Session state defaults
# ─────────────────────────────────────────────────────────────
for key, default in {
    "scenario": None,
    "classical_results": {},
    "qaoa_result": None,
    "qubo_info": None,
    "last_error": None,
    "scenario_seed": 42,
    "n_vehicles": 3,
    "n_incidents": 3,
    "last_scenario_hash": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# Pre-populate default 3x3 scenario on first run so user immediately sees visualizations
if st.session_state.scenario is None:
    try:
        init_sc = build_scenario(3, 3, seed=42)
        st.session_state.scenario = init_sc
        st.session_state.qubo_info = get_qubo_info(init_sc)
        st.session_state.last_scenario_hash = "3_3_42"
    except Exception as e:
        st.session_state.last_error = str(e)


# ─────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="quantum-header">
  <h1>🚑 QuantumDispatch</h1>
  <p>Severity-Aware Emergency Dispatch Using QAOA &nbsp;·&nbsp;
     IBM Qiskit Fall Fest 2026 – RV University &nbsp;·&nbsp;
     <em>Synthetic scenario · Qiskit Statevector simulator · No live emergency data</em>
  </p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Sidebar – Scenario controls
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Scenario Setup")

    preset = st.selectbox(
        "Preset",
        ["Small demo (3×3)", "Medium demo (4×4)", "Custom"],
    )
    if preset == "Small demo (3×3)":
        n_v, n_i = 3, 3
    elif preset == "Medium demo (4×4)":
        n_v, n_i = 4, 4
    else:
        n_v = st.slider("Ambulances", 2, 4, st.session_state.n_vehicles)
        n_i = st.slider("Incidents",  2, 4, st.session_state.n_incidents)

    seed = st.number_input("Random seed (for reproducibility)", min_value=0,
                            max_value=9999, value=st.session_state.scenario_seed, step=1)

    st.markdown("---")

    if st.button("🗺️ Generate Scenario", type="primary"):
        with st.spinner("Generating scenario…"):
            try:
                sc = build_scenario(n_v, n_i, int(seed))
                # Clear stale results when scenario changes
                sc_hash = f"{n_v}_{n_i}_{seed}"
                if sc_hash != st.session_state.last_scenario_hash:
                    st.session_state.classical_results = {}
                    st.session_state.qaoa_result = None
                    st.session_state.last_scenario_hash = sc_hash
                st.session_state.scenario = sc
                st.session_state.n_vehicles = n_v
                st.session_state.n_incidents = n_i
                st.session_state.scenario_seed = int(seed)
                st.session_state.qubo_info = get_qubo_info(sc)
                st.session_state.last_error = None
                st.success(f"Scenario ready: {n_v} ambulances, {n_i} incidents, seed={seed}")
            except Exception as e:
                st.session_state.last_error = str(e)
                st.error(f"Error: {e}")

    st.markdown("---")

    if st.button("🏃 Run Classical Solvers"):
        if st.session_state.scenario is None:
            st.warning("Generate a scenario first.")
        else:
            with st.spinner("Running Hungarian, Greedy, SA…"):
                try:
                    results = run_all_classical(st.session_state.scenario)
                    st.session_state.classical_results = results
                    st.success("Classical solvers done.")
                except Exception as e:
                    st.session_state.last_error = str(e)
                    st.error(f"Classical solver error: {e}")

    st.markdown("---")
    st.markdown("#### ⚛️ QAOA Options")
    p_val = st.select_slider("QAOA depth (p)", options=[1, 2, 3], value=1)
    use_cvar = st.checkbox("Use CVaR objective (α=0.20)", value=False)
    n_restarts = st.slider("Optimizer restarts", 1, 5, 2)
    maxiter = st.slider("Max optimizer iterations", 40, 200, 80, step=20)

    if st.button("⚛️ Run QAOA"):
        if st.session_state.scenario is None:
            st.warning("Generate a scenario first.")
        else:
            # Compute reference optimal cost
            opt_cost = None
            if st.session_state.classical_results.get("Hungarian (Exact)"):
                h_res = st.session_state.classical_results["Hungarian (Exact)"]
                if "error" not in h_res:
                    opt_cost = h_res["weighted_cost"]

            n_qubits = n_v * n_i
            warn_msg = ""
            if n_qubits > 12:
                warn_msg = (
                    f"⚠️ {n_qubits} qubits — statevector sim may take several minutes. "
                    "Consider using 3×3 for a live demo."
                )

            label = f"CVaR-QAOA (p={p_val}, α=0.20)" if use_cvar else f"QAOA (p={p_val})"
            with st.spinner(f"Running {label}… (this may take 30–120 s)"):
                if warn_msg:
                    st.warning(warn_msg)
                try:
                    res = run_qaoa_solver(
                        st.session_state.scenario,
                        p=p_val,
                        cvar_alpha=0.20 if use_cvar else None,
                        n_restarts=n_restarts,
                        maxiter=maxiter,
                        seed=int(seed),
                        optimal_cost=opt_cost,
                    )
                    st.session_state.qaoa_result = res
                    st.success(
                        f"QAOA done. Approx ratio: "
                        f"{res.get('approximation_ratio', 'N/A')}"
                    )
                except Exception as e:
                    st.session_state.qaoa_result = {"error": str(e)}
                    st.error(f"QAOA error: {e}")
                    with st.expander("Traceback"):
                        st.code(traceback.format_exc())

    st.markdown("---")
    if st.button("🔄 Reset Everything"):
        for key in ["scenario", "classical_results", "qaoa_result",
                    "qubo_info", "last_error", "last_scenario_hash"]:
            st.session_state[key] = {} if key in ["classical_results"] else None
        st.success("Reset done.")


# ─────────────────────────────────────────────────────────────
# Error banner
# ─────────────────────────────────────────────────────────────
if st.session_state.last_error:
    st.error(f"Last error: {st.session_state.last_error}")


# ─────────────────────────────────────────────────────────────
# No scenario yet – show welcome screen
# ─────────────────────────────────────────────────────────────
if st.session_state.scenario is None:
    st.markdown(
        """
        <div style="text-align:center; padding: 60px 0;">
            <div style="font-size:5rem">🚑</div>
            <h2 style="color:#60a5fa">Welcome to QuantumDispatch</h2>
            <p style="color:#94a3b8; font-size:1.1rem; max-width:600px; margin:auto">
                Use the sidebar to <strong>generate a scenario</strong>, then run classical solvers
                and QAOA to compare severity-aware ambulance assignments.
            </p>
            <br/>
            <div style="display:flex;justify-content:center;gap:40px;flex-wrap:wrap;margin-top:20px">
              <div style="background:#1e293b;border:1px solid #334155;border-radius:12px;padding:20px 30px;min-width:160px">
                <div style="font-size:2rem">🗺️</div>
                <p style="color:#60a5fa;font-weight:700;margin:8px 0 4px">Scenario</p>
                <p style="color:#94a3b8;font-size:0.85rem">City grid, severities, travel times</p>
              </div>
              <div style="background:#1e293b;border:1px solid #334155;border-radius:12px;padding:20px 30px;min-width:160px">
                <div style="font-size:2rem">📋</div>
                <p style="color:#60a5fa;font-weight:700;margin:8px 0 4px">Assignments</p>
                <p style="color:#94a3b8;font-size:0.85rem">Side-by-side solver results</p>
              </div>
              <div style="background:#1e293b;border:1px solid #334155;border-radius:12px;padding:20px 30px;min-width:160px">
                <div style="font-size:2rem">📊</div>
                <p style="color:#60a5fa;font-weight:700;margin:8px 0 4px">Comparison</p>
                <p style="color:#94a3b8;font-size:0.85rem">Cost, runtime, approx ratio</p>
              </div>
              <div style="background:#1e293b;border:1px solid #334155;border-radius:12px;padding:20px 30px;min-width:160px">
                <div style="font-size:2rem">⚛️</div>
                <p style="color:#60a5fa;font-weight:700;margin:8px 0 4px">Quantum Lab</p>
                <p style="color:#94a3b8;font-size:0.85rem">QUBO, circuit, probabilities</p>
              </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()


# ─────────────────────────────────────────────────────────────
# Main tabs
# ─────────────────────────────────────────────────────────────
scenario = st.session_state.scenario
classical_results = st.session_state.classical_results
qaoa_result = st.session_state.qaoa_result

# Figure out the best current assignments to show on the grid
def _best_assignments():
    # Prefer QAOA if feasible, else Hungarian
    if qaoa_result and "error" not in qaoa_result and qaoa_result.get("feasible"):
        return qaoa_result["assignments"]
    if classical_results.get("Hungarian (Exact)") and "error" not in classical_results["Hungarian (Exact)"]:
        return classical_results["Hungarian (Exact)"]["assignments"]
    return None

tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️  Scenario",
    "📋  Assignments",
    "📊  Solver Comparison",
    "⚛️  Quantum Lab",
])

with tab1:
    render_scenario_tab(scenario, assignments=_best_assignments())

with tab2:
    render_assignments_tab(scenario, classical_results, qaoa_result)

with tab3:
    render_comparison_tab(scenario, classical_results, qaoa_result)

with tab4:
    render_quantum_tab(scenario, qaoa_result, st.session_state.qubo_info)
