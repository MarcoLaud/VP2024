import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

def latin_hypercube_sampling_safe(N, L, l, margin=2.0, seed=None):
    """
    Generate N samples in 2D using Latin hypercube sampling over the safe box:
    [margin, L - margin] x [margin, l - margin].

    margin=2 enforces that a 4 m panel (±2 m) never intersects the walls,
    regardless of orientation.
    """
    if L <= 2*margin or l <= 2*margin:
        raise ValueError(
            f"No feasible centers: need L>{2*margin} and l>{2*margin}, got L={L}, l={l}"
        )

    rng = np.random.default_rng(seed)

    # LHS on [0,1] for each axis
    u_x = (np.arange(N) + rng.random(N)) / N
    u_y = (np.arange(N) + rng.random(N)) / N
    rng.shuffle(u_x)
    rng.shuffle(u_y)

    # Scale to the safe box
    x = margin + u_x * (L - 2*margin)
    y = margin + u_y * (l  - 2*margin)

    return np.column_stack((x, y))

if __name__ == "__main__":
    # --- USER PARAMETERS ---
    N = 100       # number of panel-center positions
    L = 23.5      # room length in meters  (x ∈ [0, L])
    l = 8.6       # room width  in meters  (y ∈ [0, l])
    SEED = 42
    SAFE_MARGIN = 2.0  # meters (half of 4 m): keep at least this distance from each wall

    # Generate the samples
    points = latin_hypercube_sampling_safe(N, L, l, margin=SAFE_MARGIN, seed=SEED)

    # Print the (x, y) coordinates line by line
    for i, (x, y) in enumerate(points, start=1):
        print(f"Sample {i:3d}: x = {x:.4f} m,  y = {y:.4f} m")

    # Quick visualization: room, safe region, and samples
    fig, ax = plt.subplots()
    ax.add_patch(Rectangle((0, 0), L, l, fill=False, lw=1.5, label="Room"))
    ax.add_patch(Rectangle((SAFE_MARGIN, SAFE_MARGIN),
                           L - 2*SAFE_MARGIN, l - 2*SAFE_MARGIN,
                           fill=False, lw=1.0, ls="--", label="Safe centers"))
    ax.plot(points[:, 0], points[:, 1], linestyle="", marker="o", ms=4, label="LHS centers")
    ax.set_xlim(0, L); ax.set_ylim(0, l); ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title("LHS panel centers with ≥2 m clearance to all walls")
    ax.legend()
    plt.show()
