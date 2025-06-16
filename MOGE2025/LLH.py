import matplotlib.pyplot as plt
import numpy as np

def latin_hypercube_sampling(N, L, l, seed=None):
    """
    Generate N samples in 2D using Latin hypercube sampling over [0, L] x [0, l].

    Parameters
    ----------
    N : int
        Number of samples to generate.
    L : float
        Maximum x-coordinate (room length).
    l : float
        Maximum y-coordinate (room width).
    seed : int or None
        If provided, use this value to seed the RNG for reproducibility.

    Returns
    -------
    samples : np.ndarray, shape (N, 2)
        Array of (x, y) sample coordinates.
    """
    rng = np.random.default_rng(seed)

    # Step 1: For each dimension (x and y), create N strata and sample one point per stratum.
    # We first generate, for each dimension, the centers within each stratum [i/N, (i+1)/N):
    u_x = (np.arange(N) + rng.random(N)) / N
    u_y = (np.arange(N) + rng.random(N)) / N

    # Step 2: Shuffle each dimension independently so that points are spread across strata.
    rng.shuffle(u_x)
    rng.shuffle(u_y)

    # Step 3: Scale the unit‐interval samples to the actual room dimensions.
    x_samples = u_x * L
    y_samples = u_y * l

    # Combine into an (N, 2) array
    samples = np.column_stack((x_samples, y_samples))
    return samples

if __name__ == "__main__":
    # --- USER PARAMETERS ---
    N = 20        # number of panel‐center positions (change as needed)
    L = 23.5      # room length in meters (x ∈ [0, L])
    l = 8.6       # room width in meters  (y ∈ [0, l])
    SEED = 42     # optional: set to None for non‐reproducible randomness

    # Generate the samples
    points = latin_hypercube_sampling(N, L, l, seed=SEED)

    # Print the (x, y) coordinates line by line
    for i, (x, y) in enumerate(points, start=1):
        print(f"Sample {i:2d}: x = {x:.4f}  m,  y = {y:.4f}  m")

    plt.plot(points[:,0], points[:,1], linestyle="", marker="o")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
