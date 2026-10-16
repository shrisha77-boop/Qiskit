"""
data/instance.py
Emergency Dispatch problem instance generator and visualization utilities.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List
import numpy as np
import matplotlib.pyplot as plt

SEVERITY_WEIGHT_MAP: Dict[int, int] = {
    1: 1,   # Low: minor injuries / routine
    2: 2,   # Moderate: urgent but non-life-threatening
    3: 4,   # Serious: severe trauma / high fever
    4: 8,   # Critical: cardiac arrest, acute stroke
    5: 16   # Catastrophic: multi-casualty / life-threatening imminent
}

SEVERITY_COLORS: Dict[int, str] = {
    1: "#2ecc71",  # Green
    2: "#f1c40f",  # Yellow
    3: "#e67e22",  # Orange
    4: "#e74c3c",  # Red
    5: "#8e44ad"   # Purple / Dark Magenta
}


@dataclass
class DispatchInstance:
    """Represents an emergency dispatch scenario."""
    n_vehicles: int
    n_incidents: int
    vehicle_positions: np.ndarray       # Shape (n_vehicles, 2)
    incident_positions: np.ndarray      # Shape (n_incidents, 2)
    severities: np.ndarray              # Shape (n_incidents,), values 1..5
    weights: np.ndarray                 # Shape (n_incidents,), mapped convex weights
    travel_time_matrix: np.ndarray      # Shape (n_vehicles, n_incidents), Manhattan minutes
    cost_matrix: np.ndarray             # Shape (n_vehicles, n_incidents), C[v, i] = w_i * T[v, i]
    grid_size: float = 10.0


def compute_manhattan_travel_times(
    vehicles: np.ndarray,
    incidents: np.ndarray,
    speed: float = 1.0,
    traffic_noise: float = 0.0,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Computes Manhattan travel times between vehicles and incidents.
    T[v, i] = (|x_v - x_i| + |y_v - y_i|) / speed
    """
    n_v = vehicles.shape[0]
    n_i = incidents.shape[0]
    T = np.zeros((n_v, n_i), dtype=float)
    
    for v in range(n_v):
        for i in range(n_i):
            dist = np.sum(np.abs(vehicles[v] - incidents[i]))
            t = dist / speed
            if traffic_noise > 0.0 and rng is not None:
                # Add log-normal or positive noise
                factor = 1.0 + rng.uniform(0.0, traffic_noise)
                t *= factor
            T[v, i] = round(float(t), 2)
            
    return T


def generate_instance(
    n_vehicles: int,
    n_incidents: int,
    seed: Optional[int] = None,
    grid_size: float = 10.0,
    speed: float = 1.0,
    traffic_noise: float = 0.0
) -> DispatchInstance:
    """
    Generates a reproducible dispatch problem instance.
    """
    rng = np.random.default_rng(seed)
    
    # Generate integer or uniform grid positions
    vehicle_pos = rng.uniform(0.5, grid_size - 0.5, size=(n_vehicles, 2))
    incident_pos = rng.uniform(0.5, grid_size - 0.5, size=(n_incidents, 2))
    
    # Sample severities uniformly in {1, 2, 3, 4, 5}
    severities = rng.integers(1, 6, size=n_incidents)
    weights = np.array([SEVERITY_WEIGHT_MAP[s] for s in severities], dtype=float)
    
    # Compute travel time matrix
    T = compute_manhattan_travel_times(vehicle_pos, incident_pos, speed=speed, traffic_noise=traffic_noise, rng=rng)
    
    # Cost matrix C[v, i] = w_i * T[v, i]
    # w is shape (m,), broadcast across columns
    C = T * weights[np.newaxis, :]
    
    return DispatchInstance(
        n_vehicles=n_vehicles,
        n_incidents=n_incidents,
        vehicle_positions=vehicle_pos,
        incident_positions=incident_pos,
        severities=severities,
        weights=weights,
        travel_time_matrix=T,
        cost_matrix=C,
        grid_size=grid_size
    )


def generate_greedy_failure_instance() -> DispatchInstance:
    """
    Hand-crafted 3x3 scenario highlighting where naive nearest-vehicle greedy fails
    severely compared to severity-aware assignment.
    
    Scenario:
      - Vehicle 0 is located at (2, 2)
      - Vehicle 1 is located at (9, 9)
      - Vehicle 2 is located at (8, 9)
      - Incident 0 is at (2, 3), severity 1 (weight 1), dist from V0 = 1 min
      - Incident 1 is at (2, 5), severity 5 (weight 16), dist from V0 = 3 min
      - Incident 2 is at (8, 8), severity 3 (weight 4), near V1 and V2
    
    Greedy takes V0 -> I0 (dist 1, cost 1). Leaves critical I1 to distant V1 (dist 11, cost 176). Total = 181.
    Optimal takes V0 -> I1 (dist 3, cost 48). V1 -> I0 (dist 13, cost 13). Total = 65.
    """
    v_pos = np.array([
        [2.0, 2.0],  # V0: close to both I0 and I1
        [9.0, 9.0],  # V1: far northeast
        [8.0, 9.0]   # V2: far northeast
    ])
    i_pos = np.array([
        [2.0, 3.0],  # I0: minor call (severity 1), 1 step from V0
        [2.0, 5.0],  # I1: critical call (severity 5), 3 steps from V0
        [8.0, 8.0]   # I2: urgent call (severity 3), near V1/V2
    ])
    severities = np.array([1, 5, 3])
    weights = np.array([SEVERITY_WEIGHT_MAP[s] for s in severities], dtype=float)
    
    T = compute_manhattan_travel_times(v_pos, i_pos)
    C = T * weights[np.newaxis, :]
    
    return DispatchInstance(
        n_vehicles=3,
        n_incidents=3,
        vehicle_positions=v_pos,
        incident_positions=i_pos,
        severities=severities,
        weights=weights,
        travel_time_matrix=T,
        cost_matrix=C,
        grid_size=10.0
    )


def plot_dispatch_map(
    instance: DispatchInstance,
    assignments: Optional[List[Tuple[int, int]]] = None,
    title: str = "Emergency Dispatch Grid Scenario",
    subtitle: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 7)
) -> plt.Figure:
    """
    Renders a clear grid map showing vehicles, incidents colored by severity,
    and optional assignment vectors.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    
    # Grid styling
    ax.set_facecolor("#f8f9fa")
    ax.grid(True, linestyle="--", alpha=0.5, color="#ced4da")
    ax.set_xlim(-0.5, instance.grid_size + 0.5)
    ax.set_ylim(-0.5, instance.grid_size + 0.5)
    ax.set_aspect("equal")
    
    # Plot vehicles
    v_pos = instance.vehicle_positions
    ax.scatter(
        v_pos[:, 0], v_pos[:, 1],
        s=200, c="#2980b9", marker="s", edgecolors="black", linewidths=1.5,
        label="Ambulance (V)", zorder=5
    )
    for v, (x, y) in enumerate(v_pos):
        ax.annotate(
            f"V{v}", (x, y), xytext=(0, -18), textcoords="offset points",
            ha="center", fontsize=9, fontweight="bold", color="#1f4e78"
        )
        
    # Plot incidents colored by severity
    i_pos = instance.incident_positions
    for s_level in range(1, 6):
        mask = (instance.severities == s_level)
        if np.any(mask):
            color = SEVERITY_COLORS[s_level]
            pts = i_pos[mask]
            ax.scatter(
                pts[:, 0], pts[:, 1],
                s=160 + s_level * 30, c=color, marker="o", edgecolors="black",
                linewidths=1.5, zorder=6,
                label=f"Sev {s_level} (w={SEVERITY_WEIGHT_MAP[s_level]})"
            )
            
    for i, (x, y) in enumerate(i_pos):
        sev = instance.severities[i]
        ax.annotate(
            f"I{i}\n(s={sev})", (x, y), xytext=(0, 10), textcoords="offset points",
            ha="center", fontsize=8, fontweight="bold", color="#2c3e50"
        )
        
    # Draw assignment arrows if provided
    if assignments is not None:
        for v, i in assignments:
            if v < instance.n_vehicles and i < instance.n_incidents:
                v_xy = v_pos[v]
                i_xy = i_pos[i]
                t_val = instance.travel_time_matrix[v, i]
                w_val = instance.weights[i]
                cost_val = instance.cost_matrix[v, i]
                
                # Draw arrow
                ax.annotate(
                    "", xy=(i_xy[0], i_xy[1]), xytext=(v_xy[0], v_xy[1]),
                    arrowprops=dict(
                        arrowstyle="->", color="#c0392b" if w_val >= 8 else "#27ae60",
                        lw=2.2, linestyle="-", shrinkA=8, shrinkB=8
                    ),
                    zorder=4
                )
                
                # Annotate travel time at midpoint
                mid_x = (v_xy[0] + i_xy[0]) / 2.0
                mid_y = (v_xy[1] + i_xy[1]) / 2.0
                ax.text(
                    mid_x, mid_y, f"{t_val}m ({cost_val:.0f})",
                    fontsize=7.5, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.85),
                    zorder=7
                )
                
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    if subtitle:
        ax.set_xlabel(f"{subtitle}\nSimulator Model (Manhattan grid)", fontsize=9, color="#495057")
    else:
        ax.set_xlabel("X Coordinate (km) [Simulator Model]", fontsize=9, color="#495057")
    ax.set_ylabel("Y Coordinate (km)", fontsize=9, color="#495057")
    
    # Legend
    ax.legend(loc="upper right", framealpha=0.9, fontsize=8)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        
    return fig
