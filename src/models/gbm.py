import numpy as np


def simulate_gbm_paths(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
) -> np.ndarray:
    """
    Simulate Geometric Brownian Motion paths under the risk-neutral measure:

        dS_t = r S_t dt + sigma S_t dW_t

    Discretisation uses the exact lognormal transition:
        S_{t+dt} = S_t * exp((r - 0.5 sigma^2) dt + sigma sqrt(dt) Z)

    Returns
    -------
    paths : np.ndarray
        Shape (n_paths, n_steps + 1). Column 0 is S0.
    """
    if n_steps <= 0:
        raise ValueError("n_steps must be >= 1")
    if n_paths <= 0:
        raise ValueError("n_paths must be >= 1")
    if T <= 0:
        # Degenerate case: return constant paths
        return np.full((n_paths, 1), float(S0))

    rng = np.random.default_rng(seed)
    dt = T / n_steps

    # Z ~ N(0,1) for each path & step
    Z = rng.standard_normal(size=(n_paths, n_steps))

    increments = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    log_paths = np.cumsum(increments, axis=1)

    # prepend zeros so S_0 is included
    log_paths = np.hstack([np.zeros((n_paths, 1)), log_paths])

    paths = S0 * np.exp(log_paths)
    return paths


def simulate_gbm_terminal(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    n_paths: int,
    seed: int | None = None,
) -> np.ndarray:
    """
    Simulate terminal prices S_T directly (no path), using:
        S_T = S0 * exp((r - 0.5 sigma^2)T + sigma sqrt(T) Z)
    """
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(size=n_paths)
    return S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
